"""Pré-processamento: normaliza colunas, gera textos fonéticos e tokenização."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from transformers import BertTokenizer

from ...data import normaliza_ncm, ncm_para_tokens
from ...text.fonetica import aplica_fonetica
from ...text.limpeza import limpa_texto

log = logging.getLogger(__name__)


class PreparaDados:
    """Preparador de dados para predição."""
    
    def __init__(
        self,
        df_raw: pd.DataFrame,
        max_lens: Tuple[int, int, int] = (20, 20, 11),
        tokenizer: Optional[BertTokenizer] = None,
        tokenizer_path: Optional[str] = None
    ):
        self.df = df_raw.copy()
        self.max_len_desc, self.max_len_fon, self.max_len_ncm = max_lens
        
        if tokenizer is None:
            if tokenizer_path and Path(tokenizer_path).exists():
                self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            else:
                self.tokenizer = BertTokenizer.from_pretrained(
                    "neuralmind/bert-base-portuguese-cased",
                    do_lower_case=True
                )
        else:
            self.tokenizer = tokenizer
    
    def _validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida e renomeia colunas necessárias."""
        column_map = {
            "Descricao da NFe": ["descricao", "Descricao do produto", "ncm_desc"],
            "NCM": ["ncm", "valor_ncm", "codigo_ncm"]
        }
        
        df_renamed = df.copy()
        
        for target, alternatives in column_map.items():
            if target not in df_renamed.columns:
                for alt in alternatives:
                    if alt in df_renamed.columns:
                        df_renamed[target] = df_renamed[alt]
                        break
                
                if target not in df_renamed.columns:
                    raise ValueError(f"Coluna obrigatória não encontrada: {target}")
        
        return df_renamed
    
    def _encode(self, texts: list, max_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Tokeniza textos usando BERT tokenizer."""
        if not texts:
            raise ValueError("Lista de textos vazia")
        
        try:
            enc = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="np"  # Retorna numpy arrays
            )
            return enc["input_ids"], enc["attention_mask"]
        except Exception as e:
            log.error(f"Erro na tokenização: {e}")
            raise
    
    def run(self) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        """Executa pipeline completo de preparação. RETORNA NUMPY ARRAYS."""
        df = self._validate_columns(self.df)
        
        df = df.dropna(subset=["Descricao da NFe"])
        
        if df.empty:
            raise ValueError("DataFrame vazio após remover NaN")
        
        log.info(f"Processando {len(df)} linhas")
        
        df["descricao"] = df["Descricao da NFe"].astype(str).map(limpa_texto)
        df["valor_ncm"] = normaliza_ncm(df["NCM"].astype(str))
        df["ncm_normalized"] = ncm_para_tokens(df["valor_ncm"])
        df["valor_fonetica"] = df["descricao"].map(aplica_fonetica)
        df["fonetica"] = " [SEP] " + df["valor_fonetica"] + " [SEP] "
        
        df_unique = df.drop_duplicates(
            subset=["descricao", "ncm_normalized"],
            keep="first"
        ).reset_index(drop=True)
        
        log.info(f"Linhas únicas: {len(df_unique)}")
        
        desc_texts = df_unique["descricao"].fillna("").tolist()
        fon_texts = df_unique["fonetica"].fillna("").tolist()
        ncm_texts = df_unique["ncm_normalized"].fillna("").tolist()
        
        d_ids, d_mask = self._encode(desc_texts, self.max_len_desc)
        f_ids, f_mask = self._encode(fon_texts, self.max_len_fon)
        n_ids, n_mask = self._encode(ncm_texts, self.max_len_ncm)
        
        # IMPORTANTE: Retornar numpy arrays, não tf.Tensor
        tensors = {
            "X_desc_ids": np.array(d_ids, dtype=np.int32),
            "X_desc_mask": np.array(d_mask, dtype=np.int32),
            "X_fon_ids": np.array(f_ids, dtype=np.int32),
            "X_fon_mask": np.array(f_mask, dtype=np.int32),
            "X_ncm_ids": np.array(n_ids, dtype=np.int32),
            "X_ncm_mask": np.array(n_mask, dtype=np.int32),
        }
        
        return tensors, df_unique