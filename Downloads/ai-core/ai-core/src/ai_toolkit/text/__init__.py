"""
ai_core.text
==============
Subpacote de transformações textuais: limpeza, fonética, ruído e tokenização.
"""

from .limpeza import limpa_texto
from .fonetica import aplica_fonetica
from .ruido import (
    introduzir_erro,
    selecionar_mais_foneticamente_similar,
)
from .tokenizacao import codifica_descricoes

__all__ = [
    "limpa_texto",
    "aplica_fonetica",
    "introduzir_erro",
    "selecionar_mais_foneticamente_similar",
    "codifica_descricoes",
]