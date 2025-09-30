"""
ai_core.data
=============
Subpacote para operações em DataFrame:
amostragem, balanceamento, filtros, deduplicação.
"""

from .amostragem import perturba_descricoes 

from .limpeza import filtra_deduplica

from .renomeia import padroniza_colunas

from .oversample_faixas import oversample_faixas

from .oversample_ros    import oversample_ros

from .oversample_smote  import oversample_smote

from .oversample_smote  import oversample_smote

from ._valida           import precisa_balancear

from .ncm import (
    normaliza_ncm,
    ncm_para_tokens,
)
__all__ = [
    "perturba_descricoes",
    "filtra_deduplica",
    "padroniza_colunas",
    "oversample_faixas",
    "oversample_ros",
    "oversample_smote",
    "precisa_balancear",
    "normaliza_ncm",
    "ncm_para_tokens",
] 