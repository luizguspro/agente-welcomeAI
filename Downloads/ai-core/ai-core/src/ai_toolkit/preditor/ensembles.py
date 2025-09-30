"""Estratégias de ensemble para combinar logits de múltiplos modelos."""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from scipy.stats import mode

log = logging.getLogger(__name__)


def unanime(preds: Dict[str, np.ndarray]) -> np.ndarray:
    """Retorna predição apenas quando todos os modelos concordam."""
    if not preds:
        raise ValueError("Dicionário de predições vazio")
    
    votes = np.stack([np.argmax(p, axis=1) for p in preds.values()], axis=0)
    unanimous = np.all(votes == votes[0, :], axis=0)
    chosen = votes[0].copy()
    chosen[~unanimous] = -1
    return chosen


def maioria(preds: Dict[str, np.ndarray]) -> np.ndarray:
    """Retorna predição por voto majoritário."""
    if not preds:
        raise ValueError("Dicionário de predições vazio")
    
    votes = np.stack([np.argmax(p, axis=1) for p in preds.values()], axis=0)
    mode_vals, counts = mode(votes, axis=0, keepdims=False)
    counts = counts.astype(int)
    min_votes_needed = (len(preds) + 1) // 2
    ties = counts < min_votes_needed
    mode_vals = mode_vals.astype(int)
    mode_vals[ties] = -1
    return mode_vals


def weighted(preds: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """Retorna predição ponderada pelos pesos fornecidos."""
    if not preds:
        raise ValueError("Dicionário de predições vazio")
    
    if not weights:
        raise ValueError("Dicionário de pesos vazio")
    
    common_shape = next(iter(preds.values())).shape
    acc = np.zeros(common_shape, dtype=np.float32)
    total_weight = 0.0
    
    for name, p in preds.items():
        weight = weights.get(name, 0.0)
        if weight > 0:
            acc += p * weight
            total_weight += weight
    
    if total_weight == 0:
        raise ValueError("Soma dos pesos é zero")
    
    acc /= total_weight
    return np.argmax(acc, axis=1)


STRATEGY_MAP = {
    "unanime": unanime,
    "maioria": maioria,
    "pesos": weighted,
    "weighted": weighted
}