"""Script principal para execução do pipeline de predição."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from .preditor import Preditor
from .prepara_dados import PreparaDados

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)


def load_dataframe(path: str) -> pd.DataFrame:
    """Carrega DataFrame de diferentes formatos."""
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    
    suffix = path_obj.suffix.lower()
    
    try:
        if suffix == ".parquet":
            return pd.read_parquet(path)
        elif suffix == ".csv":
            return pd.read_csv(path, sep=";", encoding="latin1", dtype="string")
        elif suffix == ".xlsx":
            return pd.read_excel(path, dtype="string")
        else:
            raise ValueError(f"Formato não suportado: {suffix}")
    except Exception as e:
        log.error(f"Erro ao carregar arquivo: {e}")
        raise


def save_dataframe(df: pd.DataFrame, path: str):
    """Salva DataFrame em diferentes formatos."""
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    
    try:
        if suffix == ".parquet":
            df.to_parquet(path, index=False)
        elif suffix == ".csv":
            df.to_csv(path, sep=";", encoding="latin1", index=False)
        elif suffix == ".xlsx":
            df.to_excel(path, index=False)
        else:
            df.to_csv(path, sep=";", encoding="utf-8", index=False)
    except Exception as e:
        log.error(f"Erro ao salvar arquivo: {e}")
        raise


def run_full(
    in_path: str,
    encoder_path: str,
    out_path: str,
    cfg_path: Optional[str] = None,
    strategy: Optional[str] = None,
    weights: Optional[dict] = None,
    legacy_columns: bool = True,
    batch_size: int = 300,
    verbose: int = 0
) -> pd.DataFrame:
    """Executa pipeline completo de predição."""
    
    log.info("Iniciando pipeline de predição")
    
    log.info(f"Carregando dados de {in_path}")
    df_raw = load_dataframe(in_path)
    log.info(f"Dados carregados: {len(df_raw)} linhas")
    
    log.info("Preparando dados")
    prep = PreparaDados(df_raw)
    tensors, df_unique = prep.run()
    
    log.info(f"Carregando encoder de {encoder_path}")
    label_encoder = joblib.load(encoder_path)
    
    log.info("Inicializando preditor")
    preditor = Preditor(
        tensors=tensors,
        label_encoder=label_encoder,
        cfg_path=cfg_path,
        strategy=strategy,
        weights=weights,
        batch_size=batch_size,
        verbose=verbose
    )
    
    log.info("Executando predições")
    df_preds = preditor.attach_predictions(df_unique, legacy_columns=legacy_columns)
    
    log.info(f"Salvando resultados em {out_path}")
    save_dataframe(df_preds, out_path)
    
    log.info("Pipeline concluído com sucesso")
    return df_preds


def main():
    """Função principal para CLI."""
    parser = argparse.ArgumentParser(
        description="Pipeline de predição de gênero fiscal"
    )
    
    parser.add_argument(
        "--in-path",
        required=True,
        help="Arquivo de entrada (CSV, Parquet ou Excel)"
    )
    
    parser.add_argument(
        "--encoder-path",
        required=True,
        help="Caminho do LabelEncoder (.pkl)"
    )
    
    parser.add_argument(
        "--out-path",
        required=True,
        help="Arquivo de saída"
    )
    
    parser.add_argument(
        "--cfg-path",
        default=None,
        help="Caminho do arquivo de configuração YAML"
    )
    
    parser.add_argument(
        "--strategy",
        choices=["unanime", "maioria", "pesos"],
        default=None,
        help="Estratégia de ensemble"
    )
    
    parser.add_argument(
        "--weights",
        default=None,
        help='Pesos JSON para estratégia "pesos" (ex: \'{"logits": 0.5, "fonetica": 0.5}\')'
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=300,
        help="Tamanho do batch para predição"
    )
    
    parser.add_argument(
        "--no-legacy",
        action="store_true",
        help="Desabilita colunas no formato legado"
    )
    
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Nível de verbosidade (0=silencioso, 1=progresso)"
    )
    
    args = parser.parse_args()
    
    weights = None
    if args.weights:
        try:
            weights = json.loads(args.weights)
        except json.JSONDecodeError as e:
            log.error(f"Erro ao decodificar pesos JSON: {e}")
            sys.exit(1)
    
    try:
        run_full(
            in_path=args.in_path,
            encoder_path=args.encoder_path,
            out_path=args.out_path,
            cfg_path=args.cfg_path,
            strategy=args.strategy,
            weights=weights,
            legacy_columns=not args.no_legacy,
            batch_size=args.batch_size,
            verbose=args.verbose
        )
    except Exception as e:
        log.error(f"Erro no pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()