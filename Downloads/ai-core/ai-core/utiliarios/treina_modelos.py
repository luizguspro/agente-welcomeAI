#!/usr/bin/env python
"""Treino completo dos três modelos BERT.

* 20 % de descrições perturbadas sorteadas a cada época.
* 1.3 % dos NCMs zerados + 1.3 % embaralhados por época (exceto classe
  "EMBALAGEM, ESTAMPAGEM").
* Ajuste dinâmico de pesos de classe via callback após cada época do
  modelo híbrido e solos.
* Fine-tuning progressivo: desbloqueia todas as camadas do backbone após a época 3.
* Arquitetura de saída com BatchNorm, Dropout, regularização L1/L2 e softmax temperada.

Exemplo:
    python treina_modelos.py --data artefatos/ --epochs 10 --batch 32 \
                             --noise-frac 0.20 --lr 1e-5
"""
import os,random


import argparse, json, logging, pickle
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from transformers import TFAutoModel, TFPreTrainedModel

from ai_toolkit.models.utils import train_bert_model

# ---------------------------------------------------------------------------
# Configurações iniciais -----------------------------------------------------
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data",        type=Path, required=True)
parser.add_argument("--epochs",      type=int,  default=8)
parser.add_argument("--batch",       type=int,  default=32)
parser.add_argument("--lr",          type=float, default=1e-5)
parser.add_argument("--seed",        type=int,  default=42)
parser.add_argument("--epoch-noise-frac", type=float, default=0.20)
parser.add_argument("--ncm-zero-frac",    type=float, default=0.013)
parser.add_argument("--ncm-shuffle-frac", type=float, default=0.013)
parser.add_argument("--features", default="desc",
                    help="Espaço‐separado: desc, fon, ncm")
parser.add_argument("--unfreeze-mode", choices=["all", "gradual"],
                    default="all", help="Estratégia de fine-tuning progressivo")
parser.add_argument("--unfreeze-start", type=int, default=3,
                    help="Época (0-based) para começar a liberar camadas")
parser.add_argument("--unfreeze-block", type=int, default=4,
                    help="Camadas BERT liberadas por época no modo gradual")
parser.add_argument("--lr-multiplier", type=float, default=0.1,
                    help="Fator de redução do LR após a 1ª liberação")
parser.add_argument("--clipnorm", type=float, default=1.0,
                    help="Global norm clipping; 0 para desativar")
parser.add_argument("--temperature", type=float, default=3.0,
                    help="Temperatura do softmax temperado")

args = parser.parse_args()

DATA: Path       = args.data
EPOCHS: int      = args.epochs
BATCH: int       = args.batch
LR: float        = args.lr
SEED: int        = args.seed
P_NOISE          = args.epoch_noise_frac
ZERAR_PERC       = args.ncm_zero_frac
EMBAR_PERC       = args.ncm_shuffle_frac
LR: float        = args.lr



opt_kwargs = {"learning_rate": LR}
# só adiciona clipnorm se for > 0
if args.clipnorm and args.clipnorm > 0:
    opt_kwargs["clipnorm"] = args.clipnorm

# use AdamW para ter decaimento de peso embutido
optimizer = tf.keras.optimizers.AdamW(**opt_kwargs)


# Temperatura para softmax temperada e regularizações L1/L2
TEMPERATURE: float = args.temperature
REG_L1 = 1e-6
REG_L2 = 1e-6

#os.environ["TF_USE_LEGACY_KERAS"] = "1"
random.seed(SEED)                   
os.environ["PYTHONHASHSEED"] = str(SEED)

np.random.seed(SEED)
tf.random.set_seed(SEED)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Funções auxiliares --------------------------------------------------------
# ---------------------------------------------------------------------------

def load(fname: str):
    return np.load(DATA / fname)

# Softmax temperada
def tempered_softmax(x):
    return tf.nn.softmax(x / TEMPERATURE)

# ---------------------------------------------------------------------------
# Carregamento dos tensores -------------------------------------------------
# ---------------------------------------------------------------------------
logging.info("Carregando tensores …")
Xc_desc_ids  = load("X_train_descricao_input_ids.npy")
Xc_desc_mask = load("X_train_descricao_attention_mask.npy")
Xc_fon_ids   = load("X_train_fonetica_input_ids.npy")
Xc_fon_mask  = load("X_train_fonetica_attention_mask.npy")
Xc_ncm_ids   = load("X_train_ncm_input_ids.npy")
Xc_ncm_mask  = load("X_train_ncm_attention_mask.npy")
yc           = load("y_train.npy")

Xa_desc_ids  = load("X_train_augmented_descricao_input_ids.npy")
Xa_desc_mask = load("X_train_augmented_descricao_attention_mask.npy")
Xa_fon_ids   = load("X_train_augmented_fonetica_input_ids.npy")
Xa_fon_mask  = load("X_train_augmented_fonetica_attention_mask.npy")
Xa_ncm_ids   = load("X_train_augmented_ncm_input_ids.npy")
Xa_ncm_mask  = load("X_train_augmented_ncm_attention_mask.npy")
ya           = load("y_train_augmented.npy")

Xv_desc_ids  = load("X_val_descricao_input_ids.npy")
Xv_desc_mask = load("X_val_descricao_attention_mask.npy")
Xv_fon_ids   = load("X_val_fonetica_input_ids.npy")
Xv_fon_mask  = load("X_val_fonetica_attention_mask.npy")
Xv_ncm_ids   = load("X_val_ncm_input_ids.npy")
Xv_ncm_mask  = load("X_val_ncm_attention_mask.npy")
yv           = load("y_val.npy")

num_labels: int = int(np.max(yc)) + 1

# ---------------------------------------------------------------------------
# Catálogo de features e seleção via CLI ------------------------------------
# ---------------------------------------------------------------------------
BERT_ID = "neuralmind/bert-base-portuguese-cased"

bert_desc = TFAutoModel.from_pretrained(BERT_ID, name="bert_desc")
bert_fon  = TFAutoModel.from_pretrained(BERT_ID, name="bert_fon")
bert_ncm  = TFAutoModel.from_pretrained(BERT_ID, name="bert_ncm")
for b in (bert_desc, bert_fon, bert_ncm):
    b.trainable = False     

FEATURES = {
    "desc": dict(
        train_ids=Xc_desc_ids,  train_mask=Xc_desc_mask,
        aug_ids  =Xa_desc_ids,  aug_mask =Xa_desc_mask,
        val_ids  =Xv_desc_ids,  val_mask =Xv_desc_mask,
        backbone =bert_desc
    ),
    "fon": dict(
        train_ids=Xc_fon_ids,   train_mask=Xc_fon_mask,
        aug_ids  =Xa_fon_ids,   aug_mask =Xa_fon_mask,
        val_ids  =Xv_fon_ids,   val_mask =Xv_fon_mask,
        backbone =bert_fon
    ),
    "ncm": dict(
        train_ids=Xc_ncm_ids,   train_mask=Xc_ncm_mask,
        aug_ids  =Xa_ncm_ids,   aug_mask =Xa_ncm_mask,   # sem augmentation? deixe assim p/ manter interface
        val_ids  =Xv_ncm_ids,   val_mask =Xv_ncm_mask,
        backbone =bert_ncm,
        noise    = True         # flag p/ injetar ruído
    ),
}


selected = args.features.split()
assert all(f in FEATURES for f in selected), f"features inválidas: {selected}"
active_feats = {k: FEATURES[k] for k in selected}

# ---------------------------------------------------------------------------
# Preparação para ruído em NCM ------------------------------------------------
# ---------------------------------------------------------------------------
zeroed_ncm_id_tensor = tf.zeros([Xc_ncm_ids.shape[1]], dtype=tf.int64)
unique_labels, _, counts = tf.unique_with_counts(yc)
label_encoder = joblib.load(DATA / "label_encoder.pkl")
label_embalagem = next((i for i, c in enumerate(label_encoder.classes_) if c == "EMBALAGEM, ESTAMPAGEM"), None)

if label_embalagem is not None:
    logging.info(f"[AUDIT] Classe tributária especial identificada: índice {label_embalagem}")
    logging.info(f"[AUDIT] Esta classe terá tratamento diferenciado no NCM")
    # Salvar em arquivo de auditoria
    from datetime import datetime
    with open("artefatos/audit_special_classes.log", "a") as f:
        f.write(f"{datetime.now()}: EMBALAGEM_ESTAMPAGEM index={label_embalagem}\n")

# Funções de ruído
def zerar_ncm(X: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    out = tf.identity(X)
    for lbl, cnt in zip(unique_labels.numpy(), counts.numpy()):
        if lbl == label_embalagem:
            continue
        k = int(np.ceil(ZERAR_PERC * cnt))
        idx = tf.reshape(tf.where(y == lbl), [-1])
        sel = tf.random.shuffle(idx)[:k]
        out = tf.tensor_scatter_nd_update(
            out, tf.expand_dims(sel, 1),
            tf.repeat(zeroed_ncm_id_tensor[None, :], k, 0)
        )
    return out

def embaralhar_ncm(X: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    idx_all, vals_all = [], []
    for lbl, cnt in zip(unique_labels.numpy(), counts.numpy()):
        if lbl == label_embalagem:
            continue
        idx_lbl = tf.squeeze(tf.where((y == lbl) &
                                      (tf.reduce_any(tf.not_equal(X, zeroed_ncm_id_tensor), axis=1))))
        k = int(np.ceil(EMBAR_PERC * tf.size(idx_lbl).numpy()))
        sel = tf.random.shuffle(idx_lbl)[:k]
        idx_all.append(sel)
        vals_all.append(tf.gather(X, sel))
    if not idx_all:
        return X
    idx_all = tf.concat(idx_all, 0)
    shuffled = tf.random.shuffle(tf.concat(vals_all, 0))
    return tf.tensor_scatter_nd_update(X, tf.expand_dims(idx_all, 1), shuffled)

def extend_if_needed(ids, mask, base_idx):
    if base_idx.shape[0] == 0:
        return ids, mask
    extra_ids  = tf.gather(ids,  base_idx)
    extra_mask = tf.gather(mask, base_idx)
    return tf.concat([ids,  extra_ids],  0), tf.concat([mask, extra_mask], 0)

# ---------------------------------------------------------------------------
# Dataset builder -----------------------------------------------------------
# ---------------------------------------------------------------------------

def build_train_ds() -> tf.data.Dataset:
    # k linhas de augmentation sorteadas uma única vez
    k   = int(P_NOISE * Xa_desc_ids.shape[0])
    idx = tf.random.shuffle(tf.range(Xa_desc_ids.shape[0]))[:k]

    # vetor-rótulo para TODO o batch (originais + augmentados)
    y_full = tf.concat([yc, tf.gather(ya, idx)], 0)

    tensors = []
    for f, cfg in active_feats.items():
        ids, mask = cfg["train_ids"], cfg["train_mask"]

        # 1) augmentation se existir para a view
        if f in ("desc", "fon") and cfg["aug_ids"] is not None:
            ids  = tf.concat([ids,  tf.gather(cfg["aug_ids"],  idx)], 0)
            mask = tf.concat([mask, tf.gather(cfg["aug_mask"], idx)], 0)
        else:
            ids, mask = extend_if_needed(ids, mask, idx)  # garante alinhamento

        # 2) ruído exclusivo do NCM usando y_full já alinhado
        if f == "ncm":
            ids = zerar_ncm(ids, y_full)
            ids = embaralhar_ncm(ids, y_full)

        tensors.extend([ids, mask])

    ds = (tf.data.Dataset
          .from_tensor_slices((tuple(tensors), y_full))
          .shuffle(10_000).batch(BATCH).prefetch(tf.data.AUTOTUNE))
    return ds

val_tensors = []
for f, cfg in active_feats.items():
    val_tensors.extend([cfg["val_ids"], cfg["val_mask"]])
val_ds = tf.data.Dataset.from_tensor_slices((tuple(val_tensors), yv)).batch(BATCH)


# ---------------------------------------------------------------------------
# Callbacks ------------------------------------------------------------------
# ---------------------------------------------------------------------------

# Ajuste dinâmico de pesos
def load_class_weights():
    with open(DATA / "class_weights_dict.pkl", "rb") as f:
        return pickle.load(f)

class ClassWeightAdjuster(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, class_weights, max_weights, label_enc):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.cw = class_weights
        self.max_w = max_weights
        self.label_enc = label_enc

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        report = classification_report(self.y_val, y_pred, output_dict=True, zero_division=1)
        for k, w in self.cw.items():
            f1 = report.get(str(k), {}).get("f1-score", 1.0)
            factor = 1.7 if f1 < 0.7 else 1.4 if f1 < 0.85 else 1.2 if f1 < 0.95 else None
            if factor:
                new_w = min(w * factor, self.max_w[k])
                if new_w != w:
                    name = self.label_enc.inverse_transform([k])[0]
                    logging.info("Ajuste peso %s: %.2f -> %.2f (F1=%.2f)", name, w, new_w, f1)
                    self.cw[k] = new_w

# Fine-tuning progressivo: libera todas as camadas após época 3
class ProgressiveUnfreeze(tf.keras.callbacks.Callback):
    def __init__(self, mode, start, block, lr_mult):
        super().__init__()
        self.mode = mode
        self.start = start
        self.block = block
        self.lr_mult = lr_mult
        self._released = 0        

    def _bert_layers(self, model):
        # supõe encoder de 12 camadas
        return [l for l in model.layers if l.name.startswith("bert")]

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.start:
            return

        if self.mode == "all" and self._released == 0:
            # libera tudo de uma vez
            for layer in self.model.layers:
                layer.trainable = True
            self._released = float("inf")

        elif self.mode == "gradual":
            # libera blocos de N camadas por época
            for bert in self._bert_layers(self.model):
                encoder_layers = bert.encoder.layer
                lim = min(self._released + self.block, len(encoder_layers))
                for i in range(self._released, lim):
                    encoder_layers[i].trainable = True
            self._released += self.block

        else:
            return  

        new_lr = tf.keras.backend.get_value(self.model.optimizer.lr) * self.lr_mult
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        self.model.compile(
            optimizer=self.model.optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        logging.info(
            "Unfreeze (%s): liberadas %s camadas; LR -> %.3e",
            self.mode, "todas" if self._released == float("inf") else self._released,
            new_lr,
        )

# ---------------------------------------------------------------------------
# Model builder -------------------------------------------------------------
# ---------------------------------------------------------------------------
def bert_rep(model, ids, mask):    
    return model(ids, attention_mask=mask).pooler_output

def build_model(features_cfg):
    inputs, reps = [], []
    for f, cfg in features_cfg.items():
        ids_in  = tf.keras.Input(shape=cfg["train_ids"].shape[1:], dtype=tf.int32)
        mask_in = tf.keras.Input(shape=cfg["train_mask"].shape[1:], dtype=tf.int32)
        inputs += [ids_in, mask_in]
        reps.append(cfg["backbone"](ids_in, attention_mask=mask_in).pooler_output)

    merged = reps[0] if len(reps)==1 else tf.keras.layers.Concatenate()(reps)
    x = tf.keras.layers.BatchNormalization()(merged)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(REG_L2),
        activity_regularizer=tf.keras.regularizers.l1(REG_L1)
    )(x)
    logits = tf.keras.layers.Dense(
        num_labels,
        kernel_regularizer=tf.keras.regularizers.l2(REG_L2)
    )(x)
    out = tf.keras.layers.Activation(tempered_softmax)(logits)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Instancia modelos
model_desc   = build_model({"desc": FEATURES["desc"]})
model_fon    = build_model({"fon":  FEATURES["fon"]})
model_hibrid = build_model(active_feats)

# ---------------------------------------------------------------------------
# Treino do modelo híbrido com ajuste dinâmico e progressive unfreeze --------
# ---------------------------------------------------------------------------
logging.info("Iniciando treino do modelo híbrido …")

# Carrega pesos e calcula máximos
class_w_init = load_class_weights()

progressive_cb = ProgressiveUnfreeze(
    args.unfreeze_mode,
    args.unfreeze_start,
    args.unfreeze_block,
    args.lr_multiplier
)


hist_h, cw_h = train_bert_model(
    model          = model_hibrid,
    build_train_ds = build_train_ds,
    val_inputs     = tuple(val_tensors),
    y_val          = yv,
    class_weights_init = load_class_weights(),
    epochs         = EPOCHS,
    save_path      = DATA / "bert_hibrido.keras",
    label_encoder  = label_encoder,
    progressive_unfreeze   = progressive_cb,
    batch_size = BATCH
)

# ---------------------------------------------------------------------------
# Treino dos modelos solo com ajuste dinâmico --------------------------------
# ---------------------------------------------------------------------------
logging.info("Treinando modelos solo …")
hist_solo: Dict[str, Dict[str, List[float]]] = {}

def build_train_desc():
    return tf.data.Dataset.from_tensor_slices(
        ((FEATURES["desc"]["train_ids"],
          FEATURES["desc"]["train_mask"]), yc)
    ).shuffle(10_000).batch(BATCH)

hist_desc, cw_desc = train_bert_model(
    model          = model_desc,
    build_train_ds = build_train_desc,
    val_inputs     = (FEATURES["desc"]["val_ids"],
                      FEATURES["desc"]["val_mask"]),
    y_val          = yv,
    class_weights_init = load_class_weights(),
    epochs         = EPOCHS,
    save_path      = DATA / "bert_descricao.keras",
    label_encoder  = label_encoder,
    progressive_unfreeze   = progressive_cb,
    batch_size = BATCH
)

def build_train_fon():
    return tf.data.Dataset.from_tensor_slices(
        ((FEATURES["fon"]["train_ids"],
          FEATURES["fon"]["train_mask"]), yc)
    ).shuffle(10_000).batch(BATCH)

hist_fon, cw_fon = train_bert_model(
    model          = model_fon,
    build_train_ds = build_train_fon,
    val_inputs     = (FEATURES["fon"]["val_ids"],
                      FEATURES["fon"]["val_mask"]),
    y_val          = yv,
    class_weights_init = load_class_weights(),
    epochs         = EPOCHS,
    save_path      = DATA / "bert_fonetica.keras",
    label_encoder  = label_encoder,
    progressive_unfreeze   = progressive_cb,
    batch_size = BATCH
)

# ---------------------------------------------------------------------------
# Salva históricos e pesos finais -------------------------------------------
# ---------------------------------------------------------------------------
logging.info("Salvando históricos …")

history = {"hibrido": hist_h, "descricao": hist_desc, "fonetica": hist_fon}
(Path(DATA / "history.json")).write_text(json.dumps(history, indent=2))

(Path(DATA / "class_weights_final.json")
 ).write_text(json.dumps(cw_h, indent=2))

for n, cw in [("descricao", cw_desc), ("fonetica", cw_fon)]:
    (Path(DATA / f"class_weights_final_{n}.json")
    ).write_text(json.dumps(cw, indent=2))

logging.info("Treino concluído – modelos e históricos armazenados em %s", DATA)