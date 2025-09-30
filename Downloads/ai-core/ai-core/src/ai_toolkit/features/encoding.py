from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

__all__ = ["gera_label_encoder", "calcula_pesos_classe"]

log = logging.getLogger(__name__)


def gera_label_encoder(
    df,
    col: str = "genero",
) -> Tuple[LabelEncoder, np.ndarray]:
    """Cria e ajusta LabelEncoder; devolve (encoder, labels numéricos)."""
    le = LabelEncoder()
    y = le.fit_transform(df[col])
    log.info("gera_label_encoder: %d classes", len(le.classes_))
    return le, y


def calcula_pesos_classe(labels: np.ndarray, strategy: str = "balanced") -> dict[int, float]:
    """Retorna dicionário {classe: peso} no formato Keras."""
    classes = np.unique(labels)
    weights = compute_class_weight(strategy, classes=classes, y=labels)
    pesos = {int(c): float(w) for c, w in zip(classes, weights)}
    log.info("Pesos de classe calculados: %s", pesos)
    return pesos