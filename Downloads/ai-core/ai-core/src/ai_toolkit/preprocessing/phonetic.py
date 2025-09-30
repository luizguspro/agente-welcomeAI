"""Codificação fonética determinística."""
import logging
from typing import Union, List

import pandas as pd
from pyphonetics import Soundex

log = logging.getLogger(__name__)

# Instância única do Soundex
_soundex = Soundex()


def phonetic_encode(
    text: Union[str, pd.Series, List],
    algorithm: str = "soundex",
    separator: str = "-"
) -> Union[str, pd.Series, List[str]]:
    """
    Codifica texto foneticamente.
    
    Args:
        text: Texto(s) para codificar
        algorithm: Algoritmo fonético (soundex, metaphone)
        separator: Separador entre códigos
        
    Returns:
        Código fonético
    """
    def _encode_word(word: str) -> str:
        if not word or not word.isalpha():
            return "F000"
        
        try:
            if algorithm == "soundex":
                return _soundex.phonetics(word)
            else:
                # Futuro: outros algoritmos
                return _soundex.phonetics(word)
        except Exception as e:
            log.debug(f"Erro fonético para '{word}': {e}")
            return "F000"
    
    def _encode_single(t: str) -> str:
        if pd.isna(t) or not t:
            return "F000"
        
        # Codifica cada palavra
        words = str(t).split()
        codes = [_encode_word(w) for w in words]
        
        return separator.join(codes)
    
    if isinstance(text, pd.Series):
        log.debug(f"Codificando foneticamente {len(text)} textos")
        return text.map(_encode_single)
    elif isinstance(text, list):
        return [_encode_single(t) for t in text]
    
    return _encode_single(text)


def add_sep_tokens(text: Union[str, pd.Series], sep_token: str = "[SEP]") -> Union[str, pd.Series]:
    """Adiciona tokens de separação para compatibilidade."""
    def _add_sep(t: str) -> str:
        return f" {sep_token} {t} {sep_token} "
    
    if isinstance(text, pd.Series):
        return text.map(_add_sep)
    
    return _add_sep(text)