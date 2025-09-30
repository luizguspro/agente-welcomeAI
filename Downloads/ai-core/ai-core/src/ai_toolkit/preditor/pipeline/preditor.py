"""Carrega modelos habilitados, roda predições e combina via ensemble."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder

from ..ensembles import STRATEGY_MAP

log = logging.getLogger(__name__)


class ModelSpec:
    """Especificação de um modelo individual."""
    
    def __init__(self, name: str, config: dict):
        self.name = name
        self.enabled = bool(config.get("enabled", True))
        self.path = self._resolve_path(config["path"])
        self.loader_key = config.get("loader", "bert_generic")
        self.custom_key = config.get("custom_objects_key", None)
    
    def _resolve_path(self, path: str) -> str:
        """Resolve variáveis de ambiente no path."""
        if "${MODEL_BASE_PATH}" in path:
            base = os.environ.get("MODEL_BASE_PATH", "./models/checkpoint")
            path = path.replace("${MODEL_BASE_PATH}", base)
        return path
    
    def load(self):
        """Carrega o modelo usando o loader apropriado."""
        from ..io import loaders
        
        if not Path(self.path).exists():
            raise FileNotFoundError(f"Modelo não encontrado: {self.path}")
        
        loader_fn = getattr(loaders, self.loader_key, None)
        if loader_fn is None:
            raise ValueError(f"Loader '{self.loader_key}' não encontrado")
        
        return loader_fn(self.path, custom_key=self.custom_key)


class ModelRegistry:
    """Registro central de modelos."""
    
    def __init__(self, cfg_path: Path | str):
        self.cfg_path = Path(cfg_path)
        if not self.cfg_path.exists():
            raise FileNotFoundError(f"Config não encontrada: {cfg_path}")
        
        self.cfg = yaml.safe_load(self.cfg_path.read_text())
        self.model_specs = self._load_specs()
    
    def _load_specs(self) -> Dict[str, ModelSpec]:
        """Carrega especificações dos modelos habilitados."""
        specs = {}
        for name, config in self.cfg.get("models", {}).items():
            if config.get("enabled", False):
                try:
                    specs[name] = ModelSpec(name, config)
                except Exception as e:
                    log.error(f"Erro ao carregar spec do modelo {name}: {e}")
        return specs
    
    def load_all(self) -> Dict[str, object]:
        """Carrega todos os modelos habilitados."""
        models = {}
        for name, spec in self.model_specs.items():
            try:
                log.info(f"Carregando modelo {name} de {spec.path}")
                models[name] = spec.load()
            except Exception as e:
                log.error(f"Erro ao carregar modelo {name}: {e}")
                raise
        return models


class Preditor:
    """Preditor com ensemble de modelos."""
    
    def __init__(
        self,
        tensors: Dict[str, np.ndarray],
        label_encoder: LabelEncoder,
        cfg_path: Optional[Path | str] = None,
        strategy: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        batch_size: int = 300,
        verbose: int = 0,
    ):
        self.tensors = self._validate_tensors(tensors)
        self.encoder = label_encoder
        self.batch_size = batch_size
        self.verbose = verbose
        
        if cfg_path is None:
            cfg_path = self._default_config_path()
        
        self.registry = ModelRegistry(cfg_path)
        self.models = self.registry.load_all()
        
        if not self.models:
            raise ValueError("Nenhum modelo foi carregado")
        
        ensemble_cfg = self.registry.cfg.get("ensembles", {})
        self.strategy = strategy or ensemble_cfg.get("strategy", "unanime")
        self.weights = weights or ensemble_cfg.get("weighted_params", {})
        
        if self.strategy not in STRATEGY_MAP:
            raise ValueError(f"Estratégia '{self.strategy}' não suportada")
    
    def _default_config_path(self) -> Path:
        """Retorna caminho padrão do arquivo de configuração."""
        import ai_toolkit.config as config_pkg
        from importlib.resources import files
        
        return files(config_pkg) / "model_registry.yaml"
    
    def _validate_tensors(self, tensors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Valida tensores de entrada."""
        required = [
            "X_ncm_ids", "X_ncm_mask",
            "X_desc_ids", "X_desc_mask"
        ]
        
        for key in required:
            if key not in tensors:
                raise ValueError(f"Tensor obrigatório ausente: {key}")
            
            if not isinstance(tensors[key], np.ndarray):
                raise TypeError(f"Tensor {key} deve ser np.ndarray")
        
        return tensors
    
    def _build_feed(self, model_name: str) -> list:
        """Constrói feed de entrada para cada modelo."""
        try:
            if model_name == "fonetica":
                if "X_fon_ids" not in self.tensors:
                    raise ValueError("Tensores fonéticos ausentes para modelo fonética")
                
                return [
                    self.tensors["X_ncm_ids"],
                    self.tensors["X_ncm_mask"],
                    self.tensors["X_fon_ids"],
                    self.tensors["X_fon_mask"],
                    self.tensors["X_desc_ids"],
                    self.tensors["X_desc_mask"],
                ]
            else:
                return [
                    self.tensors["X_ncm_ids"],
                    self.tensors["X_ncm_mask"],
                    self.tensors["X_desc_ids"],
                    self.tensors["X_desc_mask"],
                ]
        except KeyError as e:
            log.error(f"Tensor ausente para modelo {model_name}: {e}")
            raise
    
    def _raw_predictions(self) -> Dict[str, np.ndarray]:
        """Executa predições em todos os modelos."""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                log.info(f"Executando predição com modelo {name}")
                feed = self._build_feed(name)
                predictions[name] = model.predict(
                    feed,
                    batch_size=self.batch_size,
                    verbose=self.verbose
                )
            except Exception as e:
                log.error(f"Erro na predição do modelo {name}: {e}")
                raise
        
        return predictions
    
    def _ensemble_indices(self, raw: Dict[str, np.ndarray]) -> np.ndarray:
        """Aplica estratégia de ensemble."""
        strategy_fn = STRATEGY_MAP[self.strategy]
        
        if self.strategy in ["pesos", "weighted"]:
            return strategy_fn(raw, self.weights)
        else:
            return strategy_fn(raw)
    
    def predict(self) -> Dict[str, np.ndarray]:
        """Retorna predições brutas de todos os modelos."""
        return self._raw_predictions()
    
    def predict_labels(self) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Retorna labels finais e predições brutas."""
        raw = self._raw_predictions()
        idx_final = self._ensemble_indices(raw)
        
        valid_mask = idx_final != -1
        labels_final = np.full(len(idx_final), "INCONCLUSIVO", dtype=object)
        
        if np.any(valid_mask):
            valid_indices = idx_final[valid_mask]
            labels_final[valid_mask] = self.encoder.inverse_transform(valid_indices)
        
        return labels_final, raw
    
    def attach_predictions(self, df, legacy_columns: bool = True):
        """Anexa predições ao DataFrame."""
        labels_final, raw = self.predict_labels()
        
        if len(df) != len(labels_final):
            raise ValueError(
                f"Tamanho do DataFrame ({len(df)}) diferente das predições ({len(labels_final)})"
            )
        
        df = df.copy()
        df["RESPOSTA_FINAL"] = labels_final
        
        for name, preds in raw.items():
            idx = np.argmax(preds, axis=1)
            cls = self.encoder.inverse_transform(idx)
            prob = (preds[np.arange(len(preds)), idx] * 100).round().astype(int)
            df[f"{name}_class"] = cls
            df[f"{name}_prob"] = prob
        
        if legacy_columns:
            self._add_legacy_columns(df, raw)
        
        return df
    
    def _add_legacy_columns(self, df, raw):
        """Adiciona colunas no formato legado."""
        legacy_map = {
            "FISCO GÊNERO PREVISTO AUGUMENTED LAST": "augmented_last_class",
            "FISCO: % GÊNERO AUGUMENTED LAST": "augmented_last_prob",
            "FISCO GÊNERO PREVISTO LOGITS": "logits_class",
            "FISCO: % GÊNERO LOGITS": "logits_prob",
            "FISCO GÊNERO PREVISTO FONETICA": "fonetica_class",
            "FISCO: % GÊNERO FONETICA": "fonetica_prob",
        }
        
        for legacy_name, base_col in legacy_map.items():
            if base_col in df.columns:
                df[legacy_name] = df[base_col]
        
        if all(col in df.columns for col in ["logits_class", "augmented_last_class", "fonetica_class"]):
            df["RESPOSTA UNANIME"] = np.where(
                (df["logits_class"] == df["augmented_last_class"]) &
                (df["logits_class"] == df["fonetica_class"]),
                df["logits_class"],
                "INCONCLUSIVO"
            )