#!/bin/bash

PROJECT_DIR="/home/lgsilva/SAT_IA/ai_core"
echo "Analisando projeto em: $PROJECT_DIR"
echo "========================================"

# Função helper para buscar arquivos
buscar_arquivos() {
    local pattern=$1
    local desc=$2
    echo -e "\n[${desc}]"
    find "$PROJECT_DIR" -type f -name "$pattern" 2>/dev/null | while read -r file; do
        echo "  ✓ $(basename "$file") -> $(dirname "$file" | sed "s|$PROJECT_DIR||")"
        # Mostra tamanho do arquivo
        ls -lh "$file" | awk '{print "    Tamanho: "$5}'
    done
}

# Buscar componentes críticos
buscar_arquivos "*.keras" "MODELOS KERAS"
buscar_arquivos "*.h5" "MODELOS H5"
buscar_arquivos "*.pkl" "ARQUIVOS PICKLE"
buscar_arquivos "*label_encoder*" "LABEL ENCODERS"
buscar_arquivos "*tokenizer*" "TOKENIZERS"
buscar_arquivos "*registry*.yaml" "REGISTRY CONFIG"
buscar_arquivos "*weights*" "CLASS WEIGHTS"

# Verificar diretórios importantes
echo -e "\n[DIRETÓRIOS IMPORTANTES]"
for dir in models artefatos dados originais checkpoints; do
    if [ -d "$PROJECT_DIR/$dir" ]; then
        count=$(find "$PROJECT_DIR/$dir" -type f 2>/dev/null | wc -l)
        echo "  ✓ /$dir -> $count arquivos"
    else
        echo "  ✗ /$dir -> não existe"
    fi
done

# Verificar ambiente Python
echo -e "\n[AMBIENTE PYTHON]"
python3 -c "
import sys
print(f'  Python: {sys.version.split()[0]}')
try:
    import tensorflow as tf
    print(f'  TensorFlow: {tf.__version__}')
    gpus = tf.config.list_physical_devices('GPU')
    print(f'  GPUs TF: {len(gpus)} disponíveis')
except: 
    print('  TensorFlow: não instalado')
try:
    import torch
    print(f'  PyTorch: {torch.__version__}')
    print(f'  CUDA PyTorch: {torch.cuda.is_available()}')
except:
    print('  PyTorch: não instalado')
try:
    import transformers
    print(f'  Transformers: {transformers.__version__}')
except:
    print('  Transformers: não instalado')
"

echo -e "\n========================================"
echo "Análise concluída!"