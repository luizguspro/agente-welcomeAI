"""Limpeza e normalização de texto."""
import re
import logging
from typing import Union

import pandas as pd
import unidecode

log = logging.getLogger(__name__)

# Padrões compilados para performance
REGEX_SPECIAL = re.compile(r"\s*[-./\|!@&+(){}$%=,#?ºª°;<>:\"*]+\s*")
REGEX_BRACKETS = re.compile(r"[\[\]\\']")
REGEX_SPACES = re.compile(r"\s{2,}")


def clean_text(text: Union[str, pd.Series], config: dict = None) -> Union[str, pd.Series]:
    """
    Limpa e normaliza texto de forma determinística.
    
    Args:
        text: String ou Series para limpar
        config: Configurações opcionais de limpeza
        
    Returns:
        Texto limpo no mesmo formato da entrada
    """
    config = config or {}
    
    def _clean_single(t: str) -> str:
        if pd.isna(t) or t is None:
            return ""
        
        # Converte para string e aplica lower
        t = str(t).lower()
        
        # Remove acentos se configurado (padrão: True)
        if config.get("remove_accents", True):
            t = unidecode.unidecode(t)
        
        # Remove caracteres especiais
        t = REGEX_SPECIAL.sub(" ", t)
        t = REGEX_BRACKETS.sub("", t)
        t = REGEX_SPACES.sub(" ", t)
        
        return t.strip()
    
    if isinstance(text, pd.Series):
        log.debug(f"Limpando {len(text)} textos")
        return text.map(_clean_single)
    
    return _clean_single(text)