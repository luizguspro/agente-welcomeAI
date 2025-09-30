"""
ai_core.features
================
Criação de features, encodings e utilidades para montar
datasets de treinamento.
"""

# encoding / classes
from .encoding import gera_label_encoder, calcula_pesos_classe
from .utils import normaliza_ncm
from .tensor import build_tf_dataset
from .augment import aumenta_dataset
from .split import split_train_val

__all__ = [
    "gera_label_encoder",
    "calcula_pesos_classe",
    "normaliza_ncm",
    "build_tf_dataset",
    "aumenta_dataset",
    "split_train_val",
]