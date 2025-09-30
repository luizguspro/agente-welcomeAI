"""Loaders para diferentes tipos de modelos."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import tensorflow as tf

log = logging.getLogger(__name__)


def get_custom_objects(key: Optional[str] = None) -> Dict[str, Any]:
    """Retorna objetos customizados para carregamento de modelos."""
    custom_objects = {}
    
    if key == "alfa":
        custom_objects.update({
            "tempered_softmax": lambda x: tf.nn.softmax(x / 3.0)
        })
    
    return custom_objects


def bert_generic(
    model_path: str,
    custom_key: Optional[str] = None
) -> tf.keras.Model:
    """Carrega modelo BERT genérico."""
    path = Path(model_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    try:
        custom_objects = get_custom_objects(custom_key)
        
        if custom_objects:
            model = tf.keras.models.load_model(
                str(path),
                custom_objects=custom_objects
            )
        else:
            model = tf.keras.models.load_model(str(path))
        
        log.info(f"Modelo carregado: {model_path}")
        return model
    
    except Exception as e:
        log.error(f"Erro ao carregar modelo {model_path}: {e}")
        raise


def bert_with_compile(
    model_path: str,
    custom_key: Optional[str] = None
) -> tf.keras.Model:
    """Carrega e recompila modelo BERT."""
    model = bert_generic(model_path, custom_key)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model