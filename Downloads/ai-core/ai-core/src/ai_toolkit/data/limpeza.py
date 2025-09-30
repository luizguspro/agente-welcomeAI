from __future__ import annotations
import logging
import pandas as pd

__all__ = ["filtra_deduplica"]

log = logging.getLogger(__name__)


def filtra_deduplica(
    df: pd.DataFrame,
    col_desc: str = "ncm_desc",
    col_val: str = "valor_ncm",
) -> pd.DataFrame:
    """
    • Ordena pelo campo de descrição e remove duplicados
    • Descarta linhas com descrição vazia ou NCM nulo/whitespace.
    """
    anterior = len(df)
    df = (
        df.sort_values(by=col_desc)
        .drop_duplicates(subset=[col_desc, col_val], keep="first")
    )
    df = df[df[col_desc].str.strip() != ""]
    log.info("filtra_deduplica: %d → %d linhas", anterior, len(df))
    return df.reset_index(drop=True)