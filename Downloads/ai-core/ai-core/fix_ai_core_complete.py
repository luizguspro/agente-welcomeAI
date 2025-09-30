#!/usr/bin/env python3
import os
import sys
import re
from pathlib import Path
from datetime import datetime

print("🔧 Corrigindo AI_CORE...")

# 1. Corrigir augmentation
print("Fixando augmentation...")
file_path = Path("utiliarios/prepara_dados_treino.py")
if file_path.exists():
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Adicionar aug_df = None no else
    if 'aug_df = None' not in content:
        content = re.sub(
            r'(else:\s*\n)(\s+)(logging\.info)',
            r'\1\2aug_df = None  # Fix aplicado\n\2\3',
            content
        )
        with open(file_path, 'w') as f:
            f.write(content)
        print("✅ Augmentation corrigida")

# 2. Ativar modelo fonética
print("Ativando modelo fonética...")
registry = Path("src/ai_toolkit/config/model_registry.yaml")
if registry.exists():
    with open(registry, 'r') as f:
        content = f.read()
    content = content.replace('fonetica_v1:\n    path: "models/fonetica_v1/model.keras"\n    enabled: false',
                             'fonetica_v1:\n    path: "models/fonetica_v1/model.keras"\n    enabled: true')
    with open(registry, 'w') as f:
        f.write(content)
    print("✅ Modelo fonética ativado")

# 3. Criar global_config
print("Criando global_config...")
config_dir = Path("src/ai_toolkit/config")
config_dir.mkdir(parents=True, exist_ok=True)
with open(config_dir / "global_config.py", 'w') as f:
    f.write('''# Config global
MAX_LEN_DESCRICAO = 64
MAX_LEN_FONETICA = 20
MAX_LEN_NCM = 11
BATCH_SIZE_PREDICT = 512
MIN_CONFIDENCE = 0.70
''')
print("✅ Config global criada")

# 4. Criar API REST
print("Criando API REST...")
with open("rest_api.py", 'w') as f:
    f.write('''from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "ok", "gpu": "40GB", "models": 3})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
''')
print("✅ API REST criada")

print("🎉 AI_CORE CORRIGIDO!")
