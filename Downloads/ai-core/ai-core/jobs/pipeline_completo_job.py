import argparse, sys, runpy, os, socket, datetime, traceback
import mlflow
import torch

def log(msg: str) -> None:
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    print(f"[{ts}] {msg}", flush=True)

def detect_gpu() -> dict:
    """Detecta e retorna informações sobre GPUs disponíveis"""
    gpu_info = {
        'available': False,
        'count': 0,
        'cuda_version': None,
        'device_names': []
    }
    
    try:
        if torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['count'] = torch.cuda.device_count()
            gpu_info['cuda_version'] = torch.version.cuda
            gpu_info['device_names'] = [torch.cuda.get_device_name(i) 
                                       for i in range(gpu_info['count'])]
    except:
        pass
    
    return gpu_info

def setup_mlflow():
    """Configura MLflow tracking"""
    # Define URI do MLflow (ajustar conforme seu ambiente)
    mlflow.set_tracking_uri("http://bdaworkernode09:5000")
    mlflow.set_experiment("ai_toolkit_training")

def main() -> int:
    # IO consistente em execução distribuída
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    # Se TensorFlow estiver presente, silencia verbosidade (não afeta PyTorch)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=['train', 'predict', 'rollback'])
    p.add_argument('--in-path', required=True)
    p.add_argument('--out-path', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--git-hash')
    p.add_argument('--use-gpu', action='store_true', help='Força uso de GPU se disponível')
    p.add_argument('--gpu-id', type=int, default=0, help='ID da GPU a usar (default: 0)')
    p.add_argument('--track-mlflow', action='store_true', help='Ativa tracking com MLflow')
    p.add_argument('--dry-run', action='store_true', help='Apenas imprime argv efetivo e sai 0')
    args, extra = p.parse_known_args()

    # Detecta e loga informações de GPU
    gpu_info = detect_gpu()
    log(f"host={socket.gethostname()} mode={args.mode} python={sys.version.split()[0]}")
    log(f"GPU disponível: {gpu_info['available']} | Count: {gpu_info['count']} | CUDA: {gpu_info['cuda_version']}")
    
    if gpu_info['available'] and gpu_info['device_names']:
        for i, name in enumerate(gpu_info['device_names']):
            log(f"  GPU {i}: {name}")
    
    # Configura GPU se solicitado e disponível
    if args.use_gpu:
        if gpu_info['available']:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
            log(f"Usando GPU {args.gpu_id}")
            # Para PyTorch
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            # Para TensorFlow
            os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        else:
            log("AVISO: GPU solicitada mas não disponível, continuando com CPU")

    argv = [
        "ai_toolkit.cli", args.mode,
        "--in-path", args.in_path,
        "--out-path", args.out_path,
        "--model", args.model,
    ]
    if args.git_hash:
        argv += ["--git-hash", args.git_hash]
    
    # Passa flag de GPU para o ai_toolkit se disponível
    if args.use_gpu and gpu_info['available']:
        argv += ["--use-gpu"]
        
    argv += extra

    log("argv efetivo: " + " ".join(map(str, argv)))

    if args.dry_run:
        log("dry-run: não executando ai_toolkit.cli")
        return 0

    # Configura MLflow se solicitado
    mlflow_run = None
    if args.track_mlflow and args.mode == 'train':
        try:
            setup_mlflow()
            mlflow_run = mlflow.start_run()
            
            # Loga parâmetros básicos
            mlflow.log_param("mode", args.mode)
            mlflow.log_param("model", args.model)
            mlflow.log_param("in_path", args.in_path)
            mlflow.log_param("out_path", args.out_path)
            mlflow.log_param("hostname", socket.gethostname())
            mlflow.log_param("use_gpu", args.use_gpu)
            mlflow.log_param("gpu_available", gpu_info['available'])
            mlflow.log_param("gpu_count", gpu_info['count'])
            
            if args.git_hash:
                mlflow.log_param("git_hash", args.git_hash)
                
            log("MLflow tracking ativado")
        except Exception as e:
            log(f"AVISO: Não foi possível iniciar MLflow: {e}")
            mlflow_run = None

    # Executa o módulo exatamente como `python -m ai_toolkit.cli`
    sys.argv = argv
    try:
        # Marca tempo de início para métricas
        start_time = datetime.datetime.now()
        
        runpy.run_module("ai_toolkit.cli", run_name="__main__")
        
        # Calcula tempo de execução
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        log(f"Tempo de execução: {elapsed_time:.2f} segundos")
        
        # Loga métricas no MLflow se ativo
        if mlflow_run:
            try:
                mlflow.log_metric("execution_time_seconds", elapsed_time)
                mlflow.log_metric("success", 1)
                
                # Se houver arquivo de métricas do ai_toolkit, loga também
                metrics_file = os.path.join(args.out_path, f"{args.model}_metrics.json")
                if os.path.exists(metrics_file):
                    import json
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                mlflow.log_metric(key, value)
                                
            except Exception as e:
                log(f"AVISO: Erro ao logar métricas no MLflow: {e}")
        
        return 0
        
    except SystemExit as e:
        code = int(e.code) if isinstance(e.code, int) else 0
        log(f"ai_toolkit.cli terminou com exit code {code}")
        
        if mlflow_run:
            mlflow.log_metric("success", 0 if code != 0 else 1)
            mlflow.log_metric("exit_code", code)
            
        return code
        
    except Exception:
        error_msg = traceback.format_exc()
        log("Erro não tratado:\n" + error_msg)
        
        if mlflow_run:
            mlflow.log_metric("success", 0)
            mlflow.log_text(error_msg, "error.txt")
            
        return 1
        
    finally:
        # Finaliza run do MLflow se estiver ativo
        if mlflow_run:
            mlflow.end_run()
            log("MLflow run finalizado")

if __name__ == "__main__":
    sys.exit(main())