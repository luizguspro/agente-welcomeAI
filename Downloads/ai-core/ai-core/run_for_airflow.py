#!/usr/bin/env python3
"""
Script simplificado para Airflow DAG
Sem API, sem mensagens exageradas, configuração realista
"""
import sys
import pandas as pd
from pathlib import Path

# Adicionar ai_toolkit ao path
sys.path.insert(0, '/home/lgsilva/SAT_IA/ai_core/src')

def predict_batch(input_file, output_file):
    """Função principal para Airflow chamar"""
    
    # Importar com configs realistas
    from ai_toolkit.preditor import Preditor, PreparaDados
    
    # Carregar dados
    df = pd.read_csv(input_file, sep=';')
    
    # Preparar
    prep = PreparaDados(df)
    tensors, df_clean = prep.run()
    
    # Predizer com batch pequeno (8GB RAM)
    import joblib
    label_encoder = joblib.load('models/label_encoder.pkl')
    
    preditor = Preditor(
        tensors=tensors,
        label_encoder=label_encoder,
        batch_size=128,  # Ajustado para 8GB
        strategy='maioria'
    )
    
    # Gerar predições
    df_result = preditor.attach_predictions(df_clean)
    
    # Salvar
    df_result.to_csv(output_file, sep=';', index=False)
    
    return len(df_result)

if __name__ == "__main__":
    # Para teste local
    if len(sys.argv) == 3:
        n = predict_batch(sys.argv[1], sys.argv[2])
        print(f"Processados: {n} registros")
    else:
        print("Uso: python run_for_airflow.py input.csv output.csv")
