from __future__ import annotations
import logging, pandas as pd
from tqdm.auto import tqdm
from ai_toolkit.text.ruido import perturba_candidato  

__all__ = ["perturba_descricoes"]

log = logging.getLogger(__name__)


def _amostra_por_genero(df: pd.DataFrame, frac: float, seed: int):
    idx = (
        df.groupby("genero", group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=seed))
        .index
    )
    return df.loc[idx].copy(), idx


def perturba_descricoes(
    df: pd.DataFrame,
    frac: float = 0.10,           
    seed: int = 42,
    coluna: str = "ncm_desc",
) -> pd.DataFrame:
    """Aplica ruído fonético  em *frac* das linhas por gênero."""
    df_sub, idx_sub = _amostra_por_genero(df, frac, seed)
    for idx in tqdm(idx_sub, desc="Perturbando descrições", unit="linha"):
        df_sub.at[idx, coluna] = perturba_candidato(df_sub.at[idx, coluna])
    log.info("Perturbadas %d descrições (%.1f%% do total).", len(idx_sub), frac * 100)
    return df_sub