
"""Carregador de modelos."""
import yaml
import tensorflow as tf
from pathlib import Path
from typing import List, Dict, Optional

class ModelRegistry:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "model_registry.yaml"
        self.config_path = Path(config_path)
        self.registry = self._load_registry()
        self._model_cache = {}
    
    def _load_registry(self):
        if not self.config_path.exists():
            # Retorna config mínima se não existir
            return {
                "models": [
                    {"id": "logits_v1", "enabled": True, "path": "models/checkpoint/modelo_logits.keras"},
                    {"id": "augmented_v1", "enabled": True, "path": "models/checkpoint/modelo_augmented.keras"}
                ],
                "ensemble": {"default_strategy": "maioria"}
            }
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def get_model_info(self, model_id):
        for model in self.registry["models"]:
            if model["id"] == model_id:
                return model
        raise ValueError(f"Modelo não encontrado: {model_id}")
    
    def list_models(self, enabled_only=True):
        models = []
        for model in self.registry["models"]:
            if not enabled_only or model.get("enabled", True):
                models.append(model["id"])
        return models
    
    def load(self, model_id, validate=False):
        if model_id in self._model_cache:
            return self._model_cache[model_id]
        
        info = self.get_model_info(model_id)
        model_path = Path(info["path"])
        
        if not model_path.exists():
            alt_path = Path.cwd() / model_path
            if alt_path.exists():
                model_path = alt_path
            else:
                # Retorna modelo mock se não existir
                print(f"Aviso: Modelo {model_path} não encontrado, usando mock")
                return self._create_mock_model()
        
        model = tf.keras.models.load_model(str(model_path), compile=False)
        self._model_cache[model_id] = model
        return model
    
    def _create_mock_model(self):
        """Cria modelo mock para teste."""
        inputs = [
            tf.keras.Input(shape=(11,), dtype=tf.int32),
            tf.keras.Input(shape=(11,), dtype=tf.int32),
            tf.keras.Input(shape=(20,), dtype=tf.int32),
            tf.keras.Input(shape=(20,), dtype=tf.int32),
        ]
        concat = tf.keras.layers.Concatenate()(inputs)
        dense = tf.keras.layers.Dense(32, activation='relu')(concat)
        output = tf.keras.layers.Dense(5, activation='softmax')(dense)
        return tf.keras.Model(inputs=inputs, outputs=output)
    
    def get_ensemble_config(self):
        return self.registry.get("ensemble", {})
