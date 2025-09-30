"""Integração com Spark para processamento distribuído."""
import os
import pandas as pd
import numpy as np
from typing import Iterator

# Silenciar TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Cache global para não recarregar modelos
_MODELS_CACHE = None
_TOKENIZER_CACHE = None
_ENCODER_CACHE = None


def init_models():
    """Inicializa modelos uma vez por executor."""
    global _MODELS_CACHE, _TOKENIZER_CACHE, _ENCODER_CACHE
    
    if _MODELS_CACHE is None:
        import joblib
        from pathlib import Path
        from .preprocessing import get_tokenizer
        from .registry import ModelRegistry
        
        # Tokenizer
        _TOKENIZER_CACHE = get_tokenizer()
        
        # Label encoder
        encoder_path = Path("models/label_encoder.pkl")
        if encoder_path.exists():
            _ENCODER_CACHE = joblib.load(encoder_path)
        else:
            from sklearn.preprocessing import LabelEncoder
            _ENCODER_CACHE = LabelEncoder()
            _ENCODER_CACHE.fit(["MECANICA", "ELETRICA", "EMBALAGEM", "HIDRAULICA", "CONSTRUCAO"])
        
        # Registry
        registry = ModelRegistry()
        _MODELS_CACHE = registry
    
    return _MODELS_CACHE, _TOKENIZER_CACHE, _ENCODER_CACHE


def predict_dataframe(
    df: pd.DataFrame,
    strategy: str = "maioria",
    batch_size: int = 512
) -> pd.DataFrame:
    """
    Predição em DataFrame pandas (para Spark).
    
    Args:
        df: DataFrame com colunas 'Descricao do produto' e 'NCM'
        strategy: Estratégia de ensemble
        batch_size: Tamanho do batch
        
    Returns:
        DataFrame com predições
    """
    from .models.predict import predict_batch
    
    # Inicializa modelos se necessário
    init_models()
    
    # Executa predição
    try:
        result = predict_batch(
            df,
            strategy=strategy,
            batch_size=batch_size
        )
        return result
    except Exception as e:
        print(f"Erro na predição: {e}")
        # Retorna DataFrame vazio com schema esperado
        return pd.DataFrame({
            "descricao": [],
            "NCM": [],
            "RESPOSTA_FINAL": [],
            "confidence": [],
            "strategy": []
        })


def predict_partition(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """
    Processa partição do Spark.
    
    Args:
        iterator: Iterator de DataFrames pandas
        
    Yields:
        DataFrames com predições
    """
    # Inicializa modelos uma vez por partição
    init_models()
    
    for pdf in iterator:
        if len(pdf) > 0:
            result = predict_dataframe(pdf, strategy="maioria", batch_size=512)
            yield result
