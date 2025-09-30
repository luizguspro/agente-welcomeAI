


#!/bin/bash
# Pipeline de produção AI Toolkit

# Silenciar warnings do TensorFlow
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

echo "==================================="
echo "    PIPELINE DE PRODUÇÃO          "
echo "==================================="

echo ""
echo "[1] Executando predições..."
python3 -m ai_toolkit.cli predict \
    --in-path producao_batch.csv \
    --out-path resultado_batch.csv \
    --strategy maioria \
    --batch-size 100

if [ -f resultado_batch.csv ]; then
    TOTAL=$(tail -n +2 resultado_batch.csv | wc -l)
    echo "✓ Processados $TOTAL registros"
    
    echo ""
    echo "[2] Estatísticas das predições:"
    echo "--------------------------------"
    tail -n +2 resultado_batch.csv | cut -d';' -f3 | sort | uniq -c | sort -rn
    
    echo ""
    echo "[3] Amostra dos resultados:"
    echo "----------------------------"
    head -4 resultado_batch.csv | column -t -s ';'
else
    echo "✗ Erro na predição"
fi

echo ""
echo "==================================="
echo "✓ Pipeline concluído"
echo "==================================="
