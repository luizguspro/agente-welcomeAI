"""Rotinas de *split* com reforço de classes raras na validação.

Função principal
────────────────
``split_stratified_with_rare`` executa um ``train_test_split`` estratificado e
em seguida garante:

1. **Validação com cobertura mínima** – cada classe aparece ao menos
   ``min_val`` vezes no conjunto de validação; se faltar, amostras são copiadas
   do treino.
2. **Treino não vazio** – se alguma classe ficar ausente no treino, copia-se
   ``extra_train`` exemplos da validação de volta para o treino.
3. **Classes ultra‑raras (< ``rare_threshold`` no dataset)** – retiradas da
   validação (para não distorcer métrica) e reinseridas no treino.

Essa lógica replica o bloco usado pelo terceiro notebook entregue pelo
analista.
"""
from __future__ import annotations

import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = ["split_stratified_with_rare"]

log = logging.getLogger(__name__)


def _ensure_min_examples(
    src_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    label_col: str,
    classes: np.ndarray,
    min_examples: int,
    seed: int,
) -> pd.DataFrame:
    """Garante que *tgt_df* contenha `min_examples` de cada classe."""
    for lbl in classes:
        n_tgt = (tgt_df[label_col] == lbl).sum()
        if n_tgt < min_examples:
            need = min_examples - n_tgt
            extra = src_df[src_df[label_col] == lbl].sample(
                n=need, random_state=seed, replace=True
            )
            tgt_df = pd.concat([tgt_df, extra])
    return tgt_df


def split_stratified_with_rare(
    df: pd.DataFrame,
    label_col: str = "label_enc",
    test_size: float = 0.015,
    min_val: int = 20,
    extra_train: int = 50,
    rare_threshold: int = 100,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split estratificado + reforço de classes raras na validação.

    Retorna *(train_df, val_df)* já reajustados.
    """
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=seed
    )

    classes_all = df[label_col].unique()

    # 1) Cobertura mínima na validação
    val_df = _ensure_min_examples(train_df, val_df, label_col, classes_all, min_val, seed)

    # 2) Garante que o treino não fique sem alguma classe
    train_df = _ensure_min_examples(val_df, train_df, label_col, classes_all, extra_train, seed)

    # 3) Move labels ultra‑raros (< rare_threshold) totalmente para o treino
    rare_labels = (
        df[label_col]
        .value_counts()
        .loc[lambda s: s < rare_threshold]
        .index
    )
    rare_examples = val_df[val_df[label_col].isin(rare_labels)]
    if not rare_examples.empty:
        train_df = pd.concat([train_df, rare_examples])
        val_df = val_df.drop(index=rare_examples.index)

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    log.info(
        "split_stratified_with_rare: treino=%d | val=%d (min_val=%d, rare<%d)",
        len(train_df), len(val_df), min_val, rare_threshold,
    )
    return train_df, val_df