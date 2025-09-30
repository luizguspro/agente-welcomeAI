import logging
import re
import unidecode
import pandas as pd

__all__ = ["limpa_texto"]

log = logging.getLogger(__name__)
_regex_bruta   = re.compile(r"\s*[-./\|!@&+(){}$%=,#?ºª°;<>:\"*]+\s*")
_regex_espacos = re.compile(r"\s{2,}")

def limpa_texto(texto: object) -> str:
    """
    Limpa e normaliza texto.
    • Se `texto` é nulo (NaN / None / <NA>) → devolve "".
    • Converte qualquer outro tipo para string antes de processar.
    """
    if pd.isna(texto):
        return ""

    # Converte para string e aplica lower
    texto = unidecode.unidecode(str(texto).lower())
    texto = _regex_bruta.sub(" ", texto)
    texto = re.sub(r"[\[\]\\']", "", texto)
    texto = _regex_espacos.sub(" ", texto).strip()
    return texto