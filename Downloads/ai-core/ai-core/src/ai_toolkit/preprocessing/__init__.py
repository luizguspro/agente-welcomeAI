"""Módulo de pré-processamento - única fonte da verdade."""
from .text import clean_text
from .ncm import normalize_ncm
from .phonetic import phonetic_encode
from .tokenizer import tokenize, get_tokenizer

__all__ = [
    "clean_text",
    "normalize_ncm",
    "phonetic_encode",
    "tokenize",
    "get_tokenizer",
]