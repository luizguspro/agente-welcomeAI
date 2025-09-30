"""
Geração de ruído foneticamente consistente para descrições.

• introduzir_erro          – insere/substitui chars em 1 das 3 primeiras palavras
• selecionar_mais_foneticamente_similar – gera N candidatos e devolve o +-próximo
• perturba_candidato       – “alias” usado pelos scripts de preparação
"""

from __future__ import annotations
import random, jellyfish
import logging

__all__ = [
    "introduzir_erro",
    "selecionar_mais_foneticamente_similar",
    "perturba_candidato",
]

log = logging.getLogger(__name__)


def introduzir_erro(
    sent: str,
    taxa_sub: float = 0.1,
    taxa_ins: float = 0.1,
) -> str:
    """Altera caracteres em 1 das 3 primeiras palavras – preserva espaços."""
    palavras = sent.split(" ")
    if not palavras:
        return sent
    idx = random.choice(range(min(3, len(palavras))))
    trg = palavras[idx]
    novo = []
    for ch in trg:
        novo.append(chr(random.randint(97, 122)) if random.random() < taxa_sub else ch)
        if random.random() < taxa_ins:
            novo.append(chr(random.randint(97, 122)))
    palavras[idx] = "".join(novo)
    return " ".join(palavras)


def selecionar_mais_foneticamente_similar(
    original: str,
    num_cands: int = 10,
    taxa_sub: float = 0.1,
    taxa_ins: float = 0.1,
) -> str:
    """Retorna, dentre *num_cands* perturbações, a de maior similaridade Soundex."""
    orig_sdx = jellyfish.soundex(original)
    melhor, melhor_sim = original, -1
    tentativas = 0
    while tentativas < num_cands:
        cand = introduzir_erro(original, taxa_sub, taxa_ins)
        if cand == original:
            continue
        cand_sdx = jellyfish.soundex(cand)
        sim = jellyfish.jaro_winkler_similarity(orig_sdx, cand_sdx)
        if sim > melhor_sim:
            melhor, melhor_sim = cand, sim
        tentativas += 1
    return melhor


# Wrapper exportado para o script de preparação
def perturba_candidato(texto: str) -> str:
    return selecionar_mais_foneticamente_similar(texto)