#!/usr/bin/env python
"""Preparação de dados – versão com augmentation de NCM.

Executar:
    python prepara_dados.py --raw data/dataset.csv --out artefatos/ --seed 42
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# STD lib & third‑party ------------------------------------------------------
# ---------------------------------------------------------------------------
import argparse, json, logging
from pathlib import Path
from typing import Tuple

import joblib, numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer

# ---------------------------------------------------------------------------
# Internal helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------
from ai_toolkit.text.limpeza import limpa_texto #OK
from ai_toolkit.text.fonetica import aplica_fonetica #OK
from ai_toolkit.data.renomeia import padroniza_colunas #OK
from ai_toolkit.data.limpeza import filtra_deduplica     #OK      
from ai_toolkit.data.ncm import normaliza_ncm, ncm_para_tokens #OK
from ai_toolkit.data.oversample_faixas import oversample_faixas #OK
from ai_toolkit.data.amostragem import perturba_descricoes #OK
from ai_toolkit.data.garante_split import split_stratified_with_rare #OK
from ai_toolkit.features.encoding import gera_label_encoder #OK

# ---------------------------------------------------------------------------
# Logging -------------------------------------------------------------------
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Preparação de dados para modelos BERT")
parser.add_argument("--raw", type=Path, required=True, help="CSV bruto de entrada")
parser.add_argument("--out", type=Path, required=True, help="Diretório dos artefatos")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-size", type=float, default=0.015)
parser.add_argument("--aug-prob", type=float, default=0.1,  # 0 = sem ruído
                    help="Fraç. de linhas de treino a serem perturbadas")
parser.add_argument("--val-noisy", action="store_true",
                    help="Gerar versão ruidosa do conjunto de validação")
args = parser.parse_args()
RAW_CSV_PATH, OUT_DIR, RNG_SEED, TEST_SIZE = args.raw, args.out, args.seed, args.test_size

# ---------------------------------------------------------------------------
# Constantes ----------------------------------------------------------------
# ---------------------------------------------------------------------------
MAX_LEN_DESCRICAO = 64 # 20 ENCODINGS
MAX_LEN_FONETICA = 20
MAX_LEN_NCM = 11  # 5 blocos + 6 pads
TOKENIZER_NAME = "neuralmind/bert-base-portuguese-cased"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Leitura & coluna padrão -------------------------------------------------
# ---------------------------------------------------------------------------
logging.info("Carregando CSV bruto…")
df = pd.read_csv(RAW_CSV_PATH, sep=";", encoding="latin1", dtype="string")
df = padroniza_colunas(df)
df = filtra_deduplica(df)
logging.info("Linhas originais: %d", len(df))

# ---------------------------------------------------------------------------
# 2. Limpeza, fonética, NCM --------------------------------------------------
# ---------------------------------------------------------------------------
logging.info("Limpando texto, extraindo fonética, normalizando NCM…")
df["descricao_limpa"] = df["ncm_desc"].astype(str).map(limpa_texto)
df["fonetica"] = df["descricao_limpa"].map(aplica_fonetica)
df["ncm_norm"] = normaliza_ncm(df["valor_ncm"])
df["ncm_tokenized"] = ncm_para_tokens(df["ncm_norm"])

# ---------------------------------------------------------------------------
# 3. Label encoding ----------------------------------------------------------
# ---------------------------------------------------------------------------
TARGET_COL = "genero"
le, labels_all = gera_label_encoder(df, col=TARGET_COL)
df["label_enc"] = labels_all

# ---------------------------------------------------------------------------
# 4. Split estratificado -----------------------------------------------------
# ---------------------------------------------------------------------------
train_df, val_df = split_stratified_with_rare(
    df,
    label_col="label_enc",
    test_size=TEST_SIZE,
    min_val=20,             # piso de 20 no val
    extra_train=50,         # 50 exemplos a mais no train, se faltar
    rare_threshold=100,     # classes <100 linhas saem do val
    seed=RNG_SEED
)

# ---------------------------------------------------------------------------
# 5. Oversampling ------------------------------------------------------------
# ---------------------------------------------------------------------------

train_df = oversample_faixas(train_df, col="label_enc",metodo="aleatorio", seed=RNG_SEED)
logging.info("Após oversampling, treino: %d", len(train_df))


# from ai_toolkit.data.oversample_ros import oversample_ros
# train_df = oversample_ros(train_df, col="label_enc", strategy=0.6, seed=RNG_SEED)

if args.aug_prob > 0:
    logging.info("Gerando ruído em %.1f%% do treino…", args.aug_prob * 100)
    aug_df = perturba_descricoes(
        train_df,
        frac=args.aug_prob,
        seed=RNG_SEED,
        coluna="descricao_limpa"  
    )
    # Garantir alinhamento de rótulos
    aug_df["label_enc"] = train_df.loc[aug_df.index, "label_enc"].values
    train_df = pd.concat([train_df, aug_df]).reset_index(drop=True)
    logging.info('Linhas após augmentation: %d', len(train_df))
    logging.info("Linhas no conjunto de ruído: %d linhas", len(train_df))
else:
   aug_df = None           
# ---------------------------------------------------------------------------
# 6. Class weights (pós‑oversampling) ---------------------------------------
# ---------------------------------------------------------------------------

#Deliberadamente movido para pós-oversampling -> calculo único
classes = np.unique(train_df["label_enc"])
weights = compute_class_weight("balanced", classes=classes, y=train_df["label_enc"])
class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
logging.info("Class weights: %s", class_weights)

# ---------------------------------------------------------------------------
# 7. Tokenização -------------------------------------------------------------
# ---------------------------------------------------------------------------
logging.info("Instanciando tokenizer…")


if args.val_noisy and args.aug_prob > 0:
    val_noisy_df = perturba_descricoes(
        val_df,
        frac=args.aug_prob,
        seed=RNG_SEED + 1, 
        coluna="descricao_limpa"
    )
else:
    val_noisy_df = None

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME, do_lower_case=True)
(tokenizer_path := OUT_DIR / "tokenizer").mkdir(exist_ok=True)
_ = tokenizer.save_pretrained(tokenizer_path)

def encode(texts: pd.Series, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    enc = tokenizer(texts.tolist(), padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="np")
    return enc["input_ids"], enc["attention_mask"]

# Descrição & fonética
tr_desc_ids, tr_desc_mask = encode(train_df["descricao_limpa"], MAX_LEN_DESCRICAO)
vl_desc_ids, vl_desc_mask = encode(val_df["descricao_limpa"], MAX_LEN_DESCRICAO)
tr_fon_ids, tr_fon_mask = encode(train_df["fonetica"], MAX_LEN_FONETICA)
vl_fon_ids, vl_fon_mask = encode(val_df["fonetica"], MAX_LEN_FONETICA)
# NCM
tr_ncm_ids, tr_ncm_mask = encode(train_df["ncm_tokenized"], MAX_LEN_NCM)
vl_ncm_ids, vl_ncm_mask = encode(val_df["ncm_tokenized"], MAX_LEN_NCM)

if val_noisy_df is not None:
    # Descrição
    vl_noisy_desc_ids, vl_noisy_desc_mask = encode(
        val_noisy_df["descricao_limpa"].fillna(""), MAX_LEN_DESCRICAO
    )
    np.save(OUT_DIR / "X_val_noisy_descricao_input_ids.npy",      vl_noisy_desc_ids)
    np.save(OUT_DIR / "X_val_noisy_descricao_attention_mask.npy", vl_noisy_desc_mask)

    # Fonética
    vl_noisy_fon_ids, vl_noisy_fon_mask = encode(
        val_noisy_df["fonetica"].fillna(""), MAX_LEN_FONETICA
    )
    np.save(OUT_DIR / "X_val_noisy_fonetica_input_ids.npy",       vl_noisy_fon_ids)
    np.save(OUT_DIR / "X_val_noisy_fonetica_attention_mask.npy",  vl_noisy_fon_mask)

    # NCM
    vl_noisy_ncm_ids, vl_noisy_ncm_mask = encode(
        val_noisy_df["ncm_tokenized"].fillna(""), MAX_LEN_NCM
    )
    np.save(OUT_DIR / "X_val_noisy_ncm_input_ids.npy",            vl_noisy_ncm_ids)
    np.save(OUT_DIR / "X_val_noisy_ncm_attention_mask.npy",       vl_noisy_ncm_mask)

    # Rótulos
    np.save(OUT_DIR / "y_val_noisy.npy", val_noisy_df["label_enc"].values)

if aug_df is not None and args.aug_prob > 0:
    # Descrição
    aug_desc_ids, aug_desc_mask = encode(
        aug_df["descricao_limpa"].fillna(""), MAX_LEN_DESCRICAO
    )
    np.save(OUT_DIR / "X_train_augmented_descricao_input_ids.npy",      aug_desc_ids)
    np.save(OUT_DIR / "X_train_augmented_descricao_attention_mask.npy", aug_desc_mask)

    # Fonética
    aug_fon_ids, aug_fon_mask = encode(
        aug_df["fonetica"].fillna(""), MAX_LEN_FONETICA
    )
    np.save(OUT_DIR / "X_train_augmented_fonetica_input_ids.npy",       aug_fon_ids)
    np.save(OUT_DIR / "X_train_augmented_fonetica_attention_mask.npy",  aug_fon_mask)

    # NCM
    aug_ncm_ids, aug_ncm_mask = encode(
        aug_df["ncm_tokenized"].fillna(""), MAX_LEN_NCM
    )
    np.save(OUT_DIR / "X_train_augmented_ncm_input_ids.npy",            aug_ncm_ids)
    np.save(OUT_DIR / "X_train_augmented_ncm_attention_mask.npy",       aug_ncm_mask)

    # Rótulos
    np.save(OUT_DIR / "y_train_augmented.npy", aug_df["label_enc"].values)


y_train = train_df["label_enc"].values

# ---------------------------------------------------------------------------
# 8. Export -----------------------------------------------------------------
# ---------------------------------------------------------------------------
assert tr_desc_ids.shape[0] == len(y_train)

logging.info("Salvando artefatos…")
np.save(OUT_DIR / "X_train_descricao_input_ids.npy", tr_desc_ids)
np.save(OUT_DIR / "X_train_descricao_attention_mask.npy", tr_desc_mask)
np.save(OUT_DIR / "X_val_descricao_input_ids.npy", vl_desc_ids)
np.save(OUT_DIR / "X_val_descricao_attention_mask.npy", vl_desc_mask)

np.save(OUT_DIR / "X_train_fonetica_input_ids.npy", tr_fon_ids)
np.save(OUT_DIR / "X_train_fonetica_attention_mask.npy", tr_fon_mask)
np.save(OUT_DIR / "X_val_fonetica_input_ids.npy", vl_fon_ids)
np.save(OUT_DIR / "X_val_fonetica_attention_mask.npy", vl_fon_mask)

np.save(OUT_DIR / "X_train_ncm_input_ids.npy", tr_ncm_ids)
np.save(OUT_DIR / "X_train_ncm_attention_mask.npy", tr_ncm_mask)
np.save(OUT_DIR / "X_val_ncm_input_ids.npy", vl_ncm_ids)
np.save(OUT_DIR / "X_val_ncm_attention_mask.npy", vl_ncm_mask)

np.save(OUT_DIR / "y_train.npy", y_train)
np.save(OUT_DIR / "y_val.npy", val_df["label_enc"].values)
joblib.dump(le, OUT_DIR / "label_encoder.pkl")
joblib.dump(class_weights, OUT_DIR / "class_weights_dict.pkl")

if val_noisy_df is not None:
    # Fonética
    vl_noisy_fon_ids, vl_noisy_fon_mask = encode(
        val_noisy_df["fonetica"].fillna(""), MAX_LEN_FONETICA
    )
    np.save(OUT_DIR / "X_val_noisy_fonetica_input_ids.npy",  vl_noisy_fon_ids)
    np.save(OUT_DIR / "X_val_noisy_fonetica_attention_mask.npy", vl_noisy_fon_mask)

    # NCM
    vl_noisy_ncm_ids, vl_noisy_ncm_mask = encode(
        val_noisy_df["ncm_tokenized"].fillna(""), MAX_LEN_NCM
    )
    np.save(OUT_DIR / "X_val_noisy_ncm_input_ids.npy",  vl_noisy_ncm_ids)
    np.save(OUT_DIR / "X_val_noisy_ncm_attention_mask.npy", vl_noisy_ncm_mask)

    # Rótulos
    np.save(OUT_DIR / "y_val_noisy.npy", val_noisy_df["label_enc"].values)

meta = {
    "raw_dataset": str(RAW_CSV_PATH),
    "rows_total": len(df),
    "train_rows_clean": len(train_df),
    "train_rows_augmented": len(aug_df) if aug_df is not None else 0,
    "val_noisy": bool(val_noisy_df is not None),
    "val_rows": len(val_df),
    "inputs": {
        "descricao":  {"max_len": MAX_LEN_DESCRICAO, "files": ["X_*_descricao_*"]},
        "fonetica":   {"max_len": MAX_LEN_FONETICA,   "files": ["X_*_fonetica_*"]},
        "ncm":        {"max_len": MAX_LEN_NCM,        "files": ["X_*_ncm_*"]},
    },
    "max_len": {"descricao": MAX_LEN_DESCRICAO, "fonetica": MAX_LEN_FONETICA, "ncm": MAX_LEN_NCM},
    "seed": RNG_SEED,
    "tokenizer": TOKENIZER_NAME,
    "class_weights": class_weights,
    "augmentation": bool(args.aug_prob > 0),
    "aug_prob": args.aug_prob,
}
(OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
logging.info("Pré-processamento concluído – artefatos em %s", OUT_DIR)