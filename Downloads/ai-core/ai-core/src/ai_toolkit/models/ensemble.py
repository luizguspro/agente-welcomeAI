"""Estratégias de ensemble com calibração."""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import mode
from scipy.special import softmax

log = logging.getLogger(__name__)


class EnsemblePredictor:
    """Preditor com ensemble de modelos."""
    
    def __init__(
        self,
        strategy: str = "unanime",
        config: Optional[Dict] = None
    ):
        """
        Inicializa ensemble.
        
        Args:
            strategy: unanime, maioria ou pesos
            config: Configuração adicional
        """
        self.strategy = strategy
        self.config = config or {}
        self.strategies = {
            "unanime": self._unanime,
            "maioria": self._maioria,
            "pesos": self._weighted
        }
        
        if strategy not in self.strategies:
            raise ValueError(f"Estratégia inválida: {strategy}")
    
    def _apply_temperature(
        self,
        logits: np.ndarray,
        temperature: float = 1.0
    ) -> np.ndarray:
        """Aplica temperature scaling para calibração."""
        if temperature == 1.0:
            return softmax(logits, axis=-1)
        
        return softmax(logits / temperature, axis=-1)
    
    def _unanime(
        self,
        predictions: Dict[str, np.ndarray],
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estratégia unânime - todos devem concordar.
        
        Returns:
            (classes, confidences)
        """
        if not predictions:
            raise ValueError("Nenhuma predição fornecida")
        
        # Obtém classe predita por cada modelo
        votes = []
        probs = []
        
        for model_id, preds in predictions.items():
            classes = np.argmax(preds, axis=-1)
            confidences = np.max(preds, axis=-1)
            votes.append(classes)
            probs.append(confidences)
        
        votes = np.stack(votes, axis=0)
        probs = np.stack(probs, axis=0)
        
        # Verifica unanimidade
        first_vote = votes[0]
        unanimous = np.all(votes == first_vote[None, :], axis=0)
        
        # Resultado
        result_classes = np.where(unanimous, first_vote, -1)
        result_confidence = np.where(
            unanimous,
            np.mean(probs, axis=0),
            0.0
        )
        
        return result_classes, result_confidence
    
    def _maioria(
        self,
        predictions: Dict[str, np.ndarray],
        min_agreement: float = 0.5,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estratégia por maioria.
        
        Args:
            min_agreement: Fração mínima de concordância
            
        Returns:
            (classes, confidences)
        """
        votes = []
        probs = []
        
        for model_id, preds in predictions.items():
            classes = np.argmax(preds, axis=-1)
            confidences = np.max(preds, axis=-1)
            votes.append(classes)
            probs.append(confidences)
        
        votes = np.stack(votes, axis=0)
        probs = np.stack(probs, axis=0)
        
        # Calcula moda
        mode_result, counts = mode(votes, axis=0, keepdims=False)
        
        # Verifica se tem maioria suficiente
        n_models = len(predictions)
        min_votes = int(np.ceil(n_models * min_agreement))
        
        has_majority = counts >= min_votes
        
        result_classes = np.where(has_majority, mode_result, -1)
        
        # Confiança é média das probabilidades dos que votaram na classe vencedora
        result_confidence = np.zeros(len(result_classes))
        
        for i in range(len(result_classes)):
            if result_classes[i] != -1:
                mask = votes[:, i] == result_classes[i]
                if np.any(mask):
                    result_confidence[i] = np.mean(probs[mask, i])
        
        return result_classes.astype(int), result_confidence
    
    def _weighted(
        self,
        predictions: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estratégia com pesos.
        
        Args:
            weights: Pesos por modelo
            
        Returns:
            (classes, confidences)
        """
        if weights is None:
            weights = self.config.get("weights", {})
        
        if not weights:
            # Pesos iguais se não especificado
            weights = {m: 1.0 / len(predictions) for m in predictions}
        
        # Normaliza pesos
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Média ponderada
        first_model = next(iter(predictions.values()))
        weighted_sum = np.zeros_like(first_model)
        
        for model_id, preds in predictions.items():
            weight = weights.get(model_id, 0.0)
            weighted_sum += preds * weight
        
        # Resultado
        result_classes = np.argmax(weighted_sum, axis=-1)
        result_confidence = np.max(weighted_sum, axis=-1)
        
        return result_classes, result_confidence
    
    def predict(
        self,
        predictions: Dict[str, np.ndarray],
        temperatures: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict:
        """
        Executa ensemble com calibração.
        
        Args:
            predictions: Predições por modelo
            temperatures: Temperaturas para calibração
            
        Returns:
            Dict com classes, confidências e metadados
        """
        # Aplica calibração se fornecida
        if temperatures:
            calibrated = {}
            for model_id, preds in predictions.items():
                temp = temperatures.get(model_id, 1.0)
                calibrated[model_id] = self._apply_temperature(preds, temp)
            predictions = calibrated
        
        # Aplica estratégia
        strategy_fn = self.strategies[self.strategy]
        classes, confidences = strategy_fn(predictions, **kwargs)
        
        # Computa métricas de discordância
        all_classes = []
        for preds in predictions.values():
            all_classes.append(np.argmax(preds, axis=-1))
        all_classes = np.stack(all_classes, axis=0)
        
        # Desvio padrão das probabilidades
        all_probs = np.stack(list(predictions.values()), axis=0)
        prob_std = np.std(all_probs, axis=0).mean(axis=-1)
        
        return {
            "classes": classes,
            "confidences": confidences,
            "prob_std": prob_std,
            "strategy": self.strategy,
            "n_models": len(predictions)
        }