"""
ai_core.texto.tokenizacao
-------------------------
Tokenização com tokenizer HuggingFace.
"""

from __future__ import annotations

from typing import Dict, List

__all__ = ["codifica_descricoes"]


def codifica_descricoes(
    textos: List[str],
    tokenizer,
    max_len: int = 128,
) -> Dict[str, list]:
    """
    Envelopa tokenizer.encode_plus para lista de textos.
    Retorna dict com input_ids e attention_mask.
    """
    enc = tokenizer(
        textos,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        add_special_tokens=True,
        return_attention_mask=True,
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}