"""Normalização e tokenização de códigos NCM."""
import re
import logging
from typing import Union, List

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# Regex para manter apenas dígitos
REGEX_DIGITS = re.compile(r"[^0-9]")


def normalize_ncm(
    code: Union[str, pd.Series, List], 
    num_digits: int = 8
) -> Union[str, pd.Series, List[str]]:
    """
    Normaliza código NCM para formato padrão.
    
    Args:
        code: Código(s) NCM
        num_digits: Número de dígitos (padrão 8)
        
    Returns:
        Código normalizado com zero-padding
    """
    def _normalize_single(c) -> str:
        if pd.isna(c) or c is None:
            return "0" * num_digits
        
        # Remove não-dígitos e aplica padding
        clean = REGEX_DIGITS.sub("", str(c))
        return clean[:num_digits].zfill(num_digits)
    
    if isinstance(code, pd.Series):
        log.debug(f"Normalizando {len(code)} códigos NCM")
        return code.map(_normalize_single)
    elif isinstance(code, list):
        return [_normalize_single(c) for c in code]
    
    return _normalize_single(code)


def ncm_to_tokens(
    code: Union[str, pd.Series, List],
    format: str = "spaced"
) -> Union[str, pd.Series, List[str]]:
    """
    Converte NCM normalizado para tokens.
    
    Args:
        code: Código(s) NCM normalizado(s)
        format: 'spaced' para "01 23 45 6 7" ou 'raw' para "01234567"
        
    Returns:
        NCM tokenizado
    """
    def _tokenize_single(c: str) -> str:
        c = normalize_ncm(c)
        
        if format == "spaced":
            # Formato: "01 23 45 6 7"
            return f"{c[:2]} {c[2:4]} {c[4:6]} {c[6]} {c[7]}"
        
        return c
    
    if isinstance(code, pd.Series):
        return code.map(_tokenize_single)
    elif isinstance(code, list):
        return [_tokenize_single(c) for c in code]
    
    return _tokenize_single(code)