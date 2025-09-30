"""Tokenização com versionamento e cache."""
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from transformers import BertTokenizer

log = logging.getLogger(__name__)

# Cache global de tokenizers
_TOKENIZER_CACHE = {}


def get_tokenizer(
    model_name: str = "neuralmind/bert-base-portuguese-cased",
    version: str = "bert_pt_v1",
    cache_dir: Optional[Path] = None
) -> BertTokenizer:
    """
    Obtém tokenizer com cache e versionamento.
    
    Args:
        model_name: Nome do modelo HuggingFace
        version: Versão para rastreabilidade
        cache_dir: Diretório de cache local
        
    Returns:
        Tokenizer configurado
    """
    cache_key = f"{model_name}:{version}"
    
    if cache_key not in _TOKENIZER_CACHE:
        log.info(f"Carregando tokenizer {model_name} (versão: {version})")
        
        tokenizer = BertTokenizer.from_pretrained(
            model_name,
            do_lower_case=True,
            cache_dir=cache_dir
        )
        
        # Adiciona metadados
        tokenizer.version = version
        tokenizer.model_name = model_name
        
        _TOKENIZER_CACHE[cache_key] = tokenizer
    
    return _TOKENIZER_CACHE[cache_key]


def tokenize(
    texts: List[str],
    tokenizer: Optional[BertTokenizer] = None,
    max_length: int = 128,
    return_tensors: str = "np"
) -> Dict[str, np.ndarray]:
    """
    Tokeniza textos de forma determinística.
    
    Args:
        texts: Lista de textos
        tokenizer: Tokenizer (usa padrão se None)
        max_length: Comprimento máximo
        return_tensors: Formato de retorno (np, pt, tf)
        
    Returns:
        Dict com input_ids e attention_mask
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    if not texts:
        raise ValueError("Lista de textos vazia")
    
    # Tokenização batch
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors=return_tensors
    )
    
    log.debug(f"Tokenizados {len(texts)} textos, shape: {encoded['input_ids'].shape}")
    
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }


def compute_hash(data: Dict[str, np.ndarray]) -> str:
    """Computa hash determinístico dos dados tokenizados."""
    hasher = hashlib.sha256()
    
    for key in sorted(data.keys()):
        hasher.update(key.encode())
        hasher.update(data[key].tobytes())
    
    return hasher.hexdigest()