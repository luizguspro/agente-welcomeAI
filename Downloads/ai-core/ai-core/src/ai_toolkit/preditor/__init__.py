"""ai_core – Framework de predição de gênero fiscal com ensemble configurável."""

from .pipeline.prepara_dados import PreparaDados
from .pipeline.preditor import Preditor  

__all__ = ["PreparaDados", "Preditor"]