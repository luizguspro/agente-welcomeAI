from __future__ import annotations
import re
import pandas as pd

__all__ = ["padroniza_colunas"]

_ALIAS = {
    "descricao do produto": "ncm_desc",
    "descrição do produto": "ncm_desc",   # acento eventual
    "ncm": "valor_ncm",
    "cnae": "cnae",
    "genero": "genero",
    "gênero": "genero",
}

_space_re = re.compile(r"\s+")

def _canon(col: str) -> str:
    """minuscula, sem espaços múltiplos, sem acentos triviais."""
    col = col.lower().strip()
    col = _space_re.sub(" ", col)
    return col

def padroniza_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """
    • Renomeia colunas ignorando caixa/letras maiúsculas.
    • Converte `valor_ncm` para string zero-padded (8 dígitos).
    """
    rename_map = {}
    for col in df.columns:
        key = _canon(col)
        if key in _ALIAS:
            rename_map[col] = _ALIAS[key]

    df = df.rename(columns=rename_map)

    if "valor_ncm" in df.columns:
        v = pd.to_numeric(df["valor_ncm"], errors="coerce").fillna(0).astype(int)
        df["valor_ncm"] = v.astype(str).str.zfill(8)

    return df
