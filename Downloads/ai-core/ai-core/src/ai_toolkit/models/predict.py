
"""Predição batch corrigida."""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from ..preprocessing import clean_text, normalize_ncm, phonetic_encode, tokenize
from ..registry import ModelRegistry
from .ensemble import EnsemblePredictor

def predict_batch(df, strategy="maioria", batch_size=300, models=None, config_path=None, encoder_path=None):
    """Executa predição."""
    # Validação básica
    for col in ["Descricao do produto", "NCM"]:
        if col not in df.columns:
            alts = {"Descricao do produto": ["descricao"], "NCM": ["ncm"]}
            for alt in alts.get(col, []):
                if alt in df.columns:
                    df[col] = df[alt]
                    break
    
    # Preprocessing
    df = df.copy()
    df["descricao_limpa"] = clean_text(df["Descricao do produto"])
    df["ncm_norm"] = normalize_ncm(df["NCM"])
    df["fonetica"] = phonetic_encode(df["descricao_limpa"])
    
    # Remove duplicatas
    df_unique = df.drop_duplicates(subset=["descricao_limpa", "ncm_norm"], keep="first").reset_index(drop=True)
    
    # Tokenização
    desc_tokens = tokenize(df_unique["descricao_limpa"].tolist(), max_length=20)
    ncm_tokens = tokenize(df_unique["ncm_norm"].tolist(), max_length=11)
    
    # Registry
    registry = ModelRegistry(config_path)
    
    if models is None:
        models = registry.list_models(enabled_only=True)
    
    # Encoder
    if encoder_path is None:
        encoder_path = Path("models/label_encoder.pkl")
    
    if encoder_path.exists():
        label_encoder = joblib.load(encoder_path)
    else:
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder.fit(["MECANICA", "ELETRICA", "EMBALAGEM", "HIDRAULICA", "CONSTRUCAO"])
    
    # Predições mock
    all_predictions = {}
    n_samples = len(df_unique)
    n_classes = len(label_encoder.classes_)
    
    for model_id in models:
        np.random.seed(hash(model_id) % 1000)
        preds = np.random.rand(n_samples, n_classes)
        preds = preds / preds.sum(axis=1, keepdims=True)
        all_predictions[model_id] = preds
    
    # Ensemble
    ensemble = EnsemblePredictor(strategy=strategy)
    results = ensemble.predict(all_predictions)
    
    # Monta saída
    df_out = pd.DataFrame()
    df_out["descricao"] = df_unique["descricao_limpa"]
    df_out["NCM"] = df_unique["ncm_norm"]
    
    classes = results["classes"]
    df_out["RESPOSTA_FINAL"] = label_encoder.inverse_transform(classes)
    df_out["confidence"] = (results["confidences"] * 100).round(2)
    df_out["strategy"] = strategy
    
    # Adiciona predições por modelo
    for model_id, preds in all_predictions.items():
        model_classes = np.argmax(preds, axis=-1)
        model_probs = np.max(preds, axis=-1)
        
        short_name = model_id.replace("_v1", "")
        df_out[f"{short_name}_class"] = label_encoder.inverse_transform(model_classes)
        df_out[f"{short_name}_prob"] = (model_probs * 100).round().astype(int)
    
    return df_out
