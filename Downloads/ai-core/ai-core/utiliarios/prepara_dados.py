"""
Executar:
    python prepara_dados.py --raw data/dataset.csv --out artefatos/ --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer

# ---------------------------------------------------------------------------
# Import utilitários do pacote interno --------------------------------------
# ---------------------------------------------------------------------------

from ai_toolkit.text.limpeza import limpa_texto
from ai_toolkit.text.fonetica import aplica_fonetica
from ai_toolkit.data.renomeia import padroniza_colunas
from ai_toolkit.data.limpeza import filtra_deduplica           # deduplicação
from ai_toolkit.data.ncm import normaliza_ncm, ncm_para_tokens
from ai_toolkit.data.amostragem import perturba_descricoes     # data‑aug
from ai_toolkit.features.encoding import gera_label_encoder,calcula_pesos_classe
from ai_toolkit.models.callbacks import zerar_ncm, embaralhar_ncm

# ---------------------------------------------------------------------------
# Logging -------------------------------------------------------------------
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------------
# Argumentos CLI -------------------------------------------------------------
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Preparação de dados para modelos BERT")
parser.add_argument("--raw", type=Path, required=True, help="CSV bruto de entrada")
parser.add_argument("--out", type=Path, required=True, help="Diretório alvo dos artefatos")
parser.add_argument("--seed", type=int, default=42, help="Seed para split reprodutível")
parser.add_argument("--test-size", type=float, default=0.2, help="Proporção de validação")
parser.add_argument("--aug-prob", type=float, default=0.0,  # 0 = sem ruído
                    help="Fraç. de linhas de treino a serem perturbadas")
parser.add_argument("--val-noisy", action="store_true",
                    help="Gerar versão ruidosa do conjunto de validação")
    
args = parser.parse_args()

RAW_CSV_PATH: Path = args.raw
OUT_DIR: Path = args.out
RNG_SEED: int = args.seed
TEST_SIZE: float = args.test_size

# ---------------------------------------------------------------------------
# Hiper‑parâmetros fixos (mude aqui ou via metadata.json) --------------------
# ---------------------------------------------------------------------------

MAX_LEN_DESCRICAO = 64
MAX_LEN_FONETICA = 64
MAX_LEN_NCM = 11  # 5 blocos + 6 pads
TOKENIZER_NAME = "neuralmind/bert-base-portuguese-cased"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Carregar dados brutos ----------------------------------------------------
# ---------------------------------------------------------------------------

logging.info("Carregando CSV bruto…")
df = pd.read_csv(RAW_CSV_PATH, sep=';', encoding='latin1', dtype='string')
df = padroniza_colunas(df)
logging.info("Linhas originais: %d", len(df))

# --- (Opcional) Deduplicação -----------------------------------------------
# Descomente se quiser remover duplicatas antes de seguir
# df = filtra_deduplica(df)
# logging.info("Após deduplicação: %d", len(df))

# ---------------------------------------------------------------------------
# 2. Limpeza, fonética & NCM -------------------------------------------------
# ---------------------------------------------------------------------------

logging.info("Limpando texto, extraindo fonética e normalizando NCM…")
df["descricao_limpa"] = df["ncm_desc"].astype(str).apply(limpa_texto)
df["fonetica"] = df["descricao_limpa"].apply(aplica_fonetica)

# NCM: somente dígitos, completo em 8 posições, depois tokenização 2-2-2-1-1

df["ncm_norm"]      = normaliza_ncm(df["valor_ncm"])
df["ncm_tokenized"] = ncm_para_tokens(df["valor_ncm"])

# ---------------------------------------------------------------------------
# 3. Label encoding & class weights -----------------------------------------
# ---------------------------------------------------------------------------

TARGET_COL = "genero"
le = LabelEncoder()
le, labels_all = gera_label_encoder(df, col=TARGET_COL)
df["label_enc"] = labels_all
#class_weights = calcula_pesos_classe(labels_all)

# ---------------------------------------------------------------------------
# 4. Split treino/validação --------------------------------------------------
# ---------------------------------------------------------------------------

train_df, val_df = split_stratified_with_rare(
    df,
    label_col="label_enc",
    test_size=0.015, 
    min_val=20,
    extra_train=50,
    rare_threshold=100,
    seed=42,
)

# --- Oversampling -----------------------------------------------------------
# O notebook de produção recalcula class_weights APÓS duplicar exemplos raros.


from ai_toolkit.data.oversample_faixas import oversample_faixas
train_df = oversample_faixas(train_df, col="label_enc",limites = [200, 140, 70])
logging.info("Após oversampling, treino: %d", len(train_df))

# from ai_toolkit.data.oversample_ros import oversample_ros
# train_df = oversample_ros(train_df, col="label_enc", strategy=0.6, seed=RNG_SEED)

from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(train_df["label_enc"])
weights = compute_class_weight("balanced", classes=classes, y=train_df["label_enc"])
class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
logging.info("Class weights (pós-oversampling): %s", class_weights)

# --- (Opcional) Perturbação / Augmentação ---------------------------------- / Augmentação ----------------------------------
# Exemplo: gerar pequenas variações na descrição apenas no conjunto de treino
# aug_df = perturba_descricoes(train_df, prob=0.2)
  train_df = pd.concat([train_df, aug_df]).reset_index(drop=True)

logging.info("Tamanho final – treino: %d | validação: %d", len(train_df), len(val_df))

# ---------------------------------------------------------------------------
# 5. Tokenizer ---------------------------------------------------------------
# ---------------------------------------------------------------------------

logging.info("Instanciando tokenizer (%s)…", TOKENIZER_NAME)
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME, do_lower_case=True)
(tokenizer_path := OUT_DIR / "tokenizer").mkdir(exist_ok=True)
_ = tokenizer.save_pretrained(tokenizer_path)


def encode(texts: pd.Series, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    enc = tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np",
    )
    return enc["input_ids"], enc["attention_mask"]

# ------------ Descrição
tr_desc_ids, tr_desc_mask = encode(train_df["descricao_limpa"], MAX_LEN_DESCRICAO)
vl_desc_ids, vl_desc_mask = encode(val_df["descricao_limpa"], MAX_LEN_DESCRICAO)

# ------------ Fonética
tr_fon_ids, tr_fon_mask = encode(train_df["fonetica"], MAX_LEN_FONETICA)
vl_fon_ids, vl_fon_mask = encode(val_df["fonetica"], MAX_LEN_FONETICA)

# ------------ NCM
tr_ncm_ids, tr_ncm_mask = encode(train_df["ncm_tokenized"], MAX_LEN_NCM)
vl_ncm_ids, vl_ncm_mask = encode(val_df["ncm_tokenized"], MAX_LEN_NCM)

# ------------ Labels
y_train = train_df["label_enc"].values
y_val = val_df["label_enc"].values

# ---------------------------------------------------------------------------
# 6. Assert + Export ---------------------------------------------------------
# ---------------------------------------------------------------------------

assert tr_desc_ids.shape[0] == y_train.shape[0], "Descrição e labels fora de alinhamento"
assert tr_desc_ids.shape[1] == MAX_LEN_DESCRICAO, "Max_len DESCRICAO mudou?"

logging.info("Salvando artefatos…")

# Descrição
np.save(OUT_DIR / "X_train_descricao_input_ids.npy", tr_desc_ids)
np.save(OUT_DIR / "X_train_descricao_attention_mask.npy", tr_desc_mask)
np.save(OUT_DIR / "X_val_descricao_input_ids.npy", vl_desc_ids)
np.save(OUT_DIR / "X_val_descricao_attention_mask.npy", vl_desc_mask)

# Fonética
np.save(OUT_DIR / "X_train_fonetica_input_ids.npy", tr_fon_ids)
np.save(OUT_DIR / "X_train_fonetica_attention_mask.npy", tr_fon_mask)
np.save(OUT_DIR / "X_val_fonetica_input_ids.npy", vl_fon_ids)
np.save(OUT_DIR / "X_val_fonetica_attention_mask.npy", vl_fon_mask)

# NCM
np.save(OUT_DIR / "X_train_ncm_input_ids.npy", tr_ncm_ids)
np.save(OUT_DIR / "X_train_ncm_attention_mask.npy", tr_ncm_mask)
np.save(OUT_DIR / "X_val_ncm_input_ids.npy", vl_ncm_ids)
np.save(OUT_DIR / "X_val_ncm_attention_mask.npy", vl_ncm_mask)

# Labels e auxiliares
np.save(OUT_DIR / "y_train.npy", y_train)
np.save(OUT_DIR / "y_val.npy", y_val)
joblib.dump(le, OUT_DIR / "label_encoder.pkl")
joblib.dump(class_weights, OUT_DIR / "class_weights_dict.pkl")

# Metadata para rastreabilidade
meta = {
    "raw_dataset": str(RAW_CSV_PATH),
    "rows_total": len(df),
    "train_rows": len(train_df),
    "val_rows": len(val_df),
    "max_len": {
        "descricao": MAX_LEN_DESCRICAO,
        "fonetica": MAX_LEN_FONETICA,
        "ncm": MAX_LEN_NCM,
    },
    "seed": RNG_SEED,
    "tokenizer": TOKENIZER_NAME,
    "class_weights": class_weights,
}
(OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))

logging.info("Pré-processamento concluído — artefatos prontos em %s", OUT_DIR)