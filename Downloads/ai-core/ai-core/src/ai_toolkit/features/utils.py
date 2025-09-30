"""
ai_core.features.utils
----------------------
Pequenos utilitários diversos.
"""

def normaliza_ncm(ncms, num_digits: int = 8):
    """
    Aceita lista/Series e devolve strings zero-padded de N dígitos.
    """
    return [str(x).zfill(num_digits) for x in ncms]