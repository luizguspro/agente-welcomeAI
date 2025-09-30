from __future__ import annotations

import logging
from pyphonetics import Soundex

__all__ = ["aplica_fonetica"]

log = logging.getLogger(__name__)
_soundex = Soundex()


def _soundex_seguro(word: str) -> str:
    """Aplica Soundex a *word*; retorna "F" se não alfabética ou erro."""
    if not word.isalpha():
        return "F"
    try:
        return _soundex.phonetics(word)
    except Exception as exc:  # pragma: no cover
        log.debug("Soundex erro para '%s': %s", word, exc)
        return "F"


def aplica_fonetica(texto: str, *, add_sep: bool = False) -> str:
    """Converte cada palavra em código Soundex separado por hífen.

    Parâmetros
    ----------
    texto : str
        Entrada de texto.
    add_sep : bool, padrão *False*
        Se *True*, envolve a string resultante em " [SEP] " em ambos lados –
        útil para compatibilizar dumps antigos onde fonética era concatenada
        à descrição numa única sequência BERT.
    """
    codigos = [_soundex_seguro(p) for p in texto.split()]
    out = "-".join(codigos)
    return f" [SEP] {out} [SEP] " if add_sep else out