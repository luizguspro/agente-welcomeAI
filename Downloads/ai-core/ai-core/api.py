#!/usr/bin/env python3
"""
AI Core Classification API
Secretaria da Fazenda - SC
Version: 1.0.0
"""

import os
import sys
import json
import logging
import datetime
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

# Flask imports
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configuration
CONFIG = {
    'UPLOAD_FOLDER': '/tmp/ai_core_uploads',
    'MAX_FILE_SIZE': 1024 * 1024 * 1024,  # 1GB
    'ALLOWED_EXTENSIONS': {'csv', 'parquet'},
    'YARN_URL': 'http://localhost:8088',
    'HDFS_PATH': 'hdfs:///user/lgsilva/aicore',
    'MODEL_PATH': '/home/lgsilva/SAT_IA/ai_core/models',
    'BATCH_SIZE': 512
}

# Create upload folder if not exists
Path(CONFIG['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# Initialize Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = CONFIG['MAX_FILE_SIZE']

# Enable CORS for frontend
CORS(app, origins=['http://localhost:5173', 'http://localhost:*'])

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

# In-memory storage for demo (replace with database in production)
jobs_store = {}
models_store = {
    'logits_v1': {
        'id': 'logits_v1',
        'name': 'Logits',
        'version': 'v1.2.0',
        'enabled': True,
        'metrics': {
            'accuracy': 0.92,
            'f1': 0.89,
            'precision': 0.91,
            'recall': 0.88
        },
        'lastUpdated': '2024-01-15',
        'trainingTime': 4.5,
        'datasetSize': 50000
    },
    'augmented_v1': {
        'id': 'augmented_v1',
        'name': 'Augmented',
        'version': 'v1.1.0',
        'enabled': True,
        'metrics': {
            'accuracy': 0.94,
            'f1': 0.91,
            'precision': 0.93,
            'recall': 0.90
        },
        'lastUpdated': '2024-01-10',
        'trainingTime': 6.2,
        'datasetSize': 75000
    },
    'fonetica_v1': {
        'id': 'fonetica_v1',
        'name': 'Fonetica',
        'version': 'v1.0.0',
        'enabled': False,
        'metrics': {
            'accuracy': 0.88,
            'f1': 0.85,
            'precision': 0.87,
            'recall': 0.84
        },
        'lastUpdated': '2024-01-05',
        'trainingTime': 3.8,
        'datasetSize': 45000
    }
}

@dataclass
class Job:
    id: str
    type: str
    status: str
    filename: str
    recordCount: int
    user: str
    startTime: str
    duration: int
    progress: int
    queue: str = 'cpu'
    priority: str = 'normal'
    estimatedTime: int = 0


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in CONFIG['ALLOWED_EXTENSIONS']


def get_yarn_metrics():
    """Get YARN cluster metrics"""
    try:
        # Simulate YARN metrics (replace with actual YARN API calls)
        return {
            'activeNodes': 10,
            'totalMemory': 256000,
            'usedMemory': 120000,
            'totalCores': 48,
            'usedCores': 31
        }
    except Exception as e:
        logger.error(f"Error getting YARN metrics: {e}")
        return {}


def get_gpu_status():
    """Get GPU status from nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines:
                used, total, temp = lines[0].split(', ')
                return {
                    'used': int(int(used) / 1024),  # Convert to GB
                    'total': int(int(total) / 1024),
                    'temperature': int(temp),
                    'percentage': int((int(used) / int(total)) * 100)
                }
    except Exception as e:
        logger.error(f"Error getting GPU status: {e}")
    
    # Return mock data if nvidia-smi fails
    return {
        'used': 28,
        'total': 40,
        'temperature': 72,
        'percentage': 70
    }


def process_file_async(job_id, filepath, model, strategy):
    """Process file asynchronously (simulate)"""
    try:
        jobs_store[job_id]['status'] = 'running'
        jobs_store[job_id]['startTime'] = datetime.datetime.now().isoformat()
        
        # Simulate processing
        import time
        for i in range(0, 101, 10):
            time.sleep(1)
            jobs_store[job_id]['progress'] = i
        
        jobs_store[job_id]['status'] = 'completed'
        jobs_store[job_id]['progress'] = 100
        jobs_store[job_id]['duration'] = 10
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        jobs_store[job_id]['status'] = 'failed'


# Routes

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'service': 'ai_core_classification',
        'port': 5000,
        'version': '1.0.0'
    })


@app.route('/metrics')
def metrics():
    """Get system metrics"""
    yarn = get_yarn_metrics()
    gpu = get_gpu_status()
    
    running_jobs = len([j for j in jobs_store.values() if j['status'] == 'running'])
    queued_jobs = len([j for j in jobs_store.values() if j['status'] == 'queued'])
    completed_jobs = len([j for j in jobs_store.values() if j['status'] == 'completed'])
    total_jobs = len(jobs_store)
    
    return jsonify({
        'runningJobs': running_jobs,
        'queuedJobs': queued_jobs,
        'successRate': completed_jobs / total_jobs if total_jobs > 0 else 0,
        'avgProcessingTime': 245,
        'runningTrend': 12,
        'queuedTrend': -5,
        'successTrend': 3,
        'timeTrend': -8,
        'gpuUsage': [
            {'time': f'{i}:00', 'usage': 20 + (i * 2) % 40}
            for i in range(24)
        ],
        'jobsByStatus': [
            {'status': 'Completed', 'count': completed_jobs},
            {'status': 'Running', 'count': running_jobs},
            {'status': 'Queued', 'count': queued_jobs},
            {'status': 'Failed', 'count': len([j for j in jobs_store.values() if j['status'] == 'failed'])},
        ],
        'yarn': yarn,
        'gpu': gpu
    })


@app.route('/models')
def get_models():
    """Get available models"""
    return jsonify(list(models_store.values()))


@app.route('/models/<model_id>', methods=['PATCH'])
def update_model(model_id):
    """Enable/disable model"""
    data = request.get_json()
    if model_id in models_store:
        models_store[model_id]['enabled'] = data.get('enabled', models_store[model_id]['enabled'])
        return jsonify(models_store[model_id])
    return jsonify({'error': 'Model not found'}), 404


@app.route('/queue/status')
def queue_status():
    """Get queue status"""
    jobs_list = list(jobs_store.values())
    return jsonify({
        'total': len(jobs_list),
        'running': len([j for j in jobs_list if j['status'] == 'running']),
        'queued': len([j for j in jobs_list if j['status'] == 'queued']),
        'jobs': jobs_list[:20]  # Return last 20 jobs
    })


@app.route('/jobs/recent')
def recent_jobs():
    """Get recent jobs"""
    jobs_list = sorted(
        jobs_store.values(),
        key=lambda x: x.get('startTime', ''),
        reverse=True
    )
    return jsonify(jobs_list[:10])


@app.route('/jobs/history')
def job_history():
    """Get job history with filtering"""
    status = request.args.get('status', '')
    date_from = request.args.get('dateFrom', '')
    date_to = request.args.get('dateTo', '')
    user = request.args.get('user', '')
    
    jobs_list = list(jobs_store.values())
    
    # Apply filters
    if status:
        jobs_list = [j for j in jobs_list if j['status'] == status]
    if user:
        jobs_list = [j for j in jobs_list if user.lower() in j.get('user', '').lower()]
    
    return jsonify({
        'jobs': jobs_list,
        'total': len(jobs_list),
        'showing': min(20, len(jobs_list))
    })


@app.route('/jobs/<job_id>')
def get_job(job_id):
    """Get specific job details"""
    if job_id in jobs_store:
        return jsonify(jobs_store[job_id])
    return jsonify({'error': 'Job not found'}), 404


@app.route('/jobs/<job_id>/prioritize', methods=['POST'])
def prioritize_job(job_id):
    """Prioritize a job in the queue"""
    if job_id in jobs_store:
        jobs_store[job_id]['priority'] = 'high'
        return jsonify({'status': 'success', 'jobId': job_id})
    return jsonify({'error': 'Job not found'}), 404


@app.route('/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    """Cancel a job"""
    if job_id in jobs_store:
        jobs_store[job_id]['status'] = 'cancelled'
        return jsonify({'status': 'success', 'jobId': job_id})
    return jsonify({'error': 'Job not found'}), 404


@app.route('/predict', methods=['POST'])
def predict():
    """Process file for prediction"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(CONFIG['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get parameters
        model = request.form.get('model', 'ensemble')
        strategy = request.form.get('strategy', 'unanime')
        options = json.loads(request.form.get('options', '{}'))
        
        # Create job
        job_id = f"job_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(jobs_store)}"
        job = {
            'id': job_id,
            'type': 'Prediction',
            'status': 'queued',
            'filename': filename,
            'recordCount': 1000,  # Would count actual records
            'user': 'lgsilva',
            'startTime': datetime.datetime.now().isoformat(),
            'duration': 0,
            'progress': 0,
            'queue': 'gpu' if options.get('useGPU') else 'cpu',
            'priority': options.get('priority', 'normal'),
            'estimatedTime': 10,
            'model': model,
            'strategy': strategy
        }
        jobs_store[job_id] = job
        
        # Process async
        executor.submit(process_file_async, job_id, filepath, model, strategy)
        
        return jsonify({
            'status': 'success',
            'jobId': job_id,
            'message': f'Processing started for {filename}'
        })
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def train():
    """Start model training"""
    try:
        data = request.get_json()
        
        # Create training job
        job_id = f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job = {
            'id': job_id,
            'type': 'Training',
            'status': 'queued',
            'filename': data.get('dataset', 'dataset.csv'),
            'recordCount': 50000,
            'user': 'lgsilva',
            'startTime': datetime.datetime.now().isoformat(),
            'duration': 0,
            'progress': 0,
            'queue': 'gpu',
            'priority': 'normal',
            'estimatedTime': 240
        }
        jobs_store[job_id] = job
        
        return jsonify({
            'status': 'success',
            'jobId': job_id,
            'message': 'Training job queued'
        })
        
    except Exception as e:
        logger.error(f"Error in train endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/resources')
def resources():
    """Get system resources"""
    gpu = get_gpu_status()
    
    return jsonify({
        'gpu': {
            'used': gpu['used'],
            'total': gpu['total'],
            'percentage': gpu['percentage'],
            'temperature': gpu['temperature'],
            'processes': [
                {'name': 'bert_training', 'memory': 12288},
                {'name': 'inference_batch', 'memory': 8192},
            ]
        },
        'cpu': {
            'usage': 65,
            'cores': 48,
            'usedCores': 31,
            'topProcesses': [
                {'name': 'spark-executor', 'usage': 25},
                {'name': 'yarn-nodemanager', 'usage': 15},
            ]
        },
        'memory': {
            'used': 120,
            'total': 256,
            'percentage': 47,
            'available': 136,
            'cache': 32,
            'swapUsed': 2,
            'swapTotal': 16
        },
        'nodes': [
            {
                'id': 'node01',
                'name': 'bdaworkernode01',
                'status': 'online',
                'cpuUsage': 45,
                'memoryUsage': 60,
                'hasGpu': False,
                'gpuUsage': 0,
                'runningJobs': 2
            },
            {
                'id': 'node09',
                'name': 'bdaworkernode09',
                'status': 'online',
                'cpuUsage': 78,
                'memoryUsage': 85,
                'hasGpu': True,
                'gpuUsage': 70,
                'runningJobs': 3
            }
        ]
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


def initialize_demo_data():
    """Create some demo jobs for testing"""
    for i in range(5):
        job_id = f"demo_job_{i}"
        jobs_store[job_id] = {
            'id': job_id,
            'type': ['Prediction', 'Training', 'Validation'][i % 3],
            'status': ['completed', 'running', 'queued', 'failed'][i % 4],
            'filename': f'dataset_{i}.csv',
            'recordCount': 1000 * (i + 1),
            'user': 'lgsilva',
            'startTime': (datetime.datetime.now() - datetime.timedelta(hours=i)).isoformat(),
            'duration': 10 * (i + 1),
            'progress': [100, 60, 0, 0][i % 4],
            'queue': ['cpu', 'gpu'][i % 2],
            'priority': 'normal',
            'estimatedTime': 5 * (i + 1)
        }


if __name__ == '__main__':
    initialize_demo_data()
    logger.info("AI Core Classification API starting on port 5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
    