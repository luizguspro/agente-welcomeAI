import logging, re, pandas as pd
log = logging.getLogger(__name__)

__all__ = ["normaliza_ncm", "ncm_para_tokens"]

_ren = re.compile(r"[^0-9]")

def _norm_one(x: str) -> str:
    return _ren.sub("", str(x)).zfill(8)

def normaliza_ncm(code: str | pd.Series) -> str | pd.Series:
    """Remove caracteres não‑numéricos e aplica zfill(8)."""
    if isinstance(code, pd.Series):
        antes  = code.nunique()
        out = code.map(_norm_one)
        depois = out.nunique()
        log.info("normaliza_ncm: %d → %d códigos únicos", antes, depois)
        return out
    return _norm_one(code)

def ncm_para_tokens(code: str | pd.Series) -> str | pd.Series:
    """Formata 8 dígitos em blocos '01 23 45 6 7'."""
    def _fmt(s: str) -> str:
        s = _norm_one(s)
        return f"{s[:2]} {s[2:4]} {s[4:6]} {s[6]} {s[7]}"
    if isinstance(code, pd.Series):
        return code.map(_fmt)
    return _fmt(code)