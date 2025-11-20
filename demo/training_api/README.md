# Training API

FastAPI-based REST API for LLM fine-tuning pipeline.

## Features

- **Data Preparation**: Convert SAM2 exports, validate datasets, split data
- **Training Management**: Start, monitor, and cancel training jobs
- **Experiment Tracking**: Track and compare multiple training experiments
- **Model Export**: Export trained models in various formats

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
cd demo/training_api
python main.py
```

The API will be available at `http://localhost:8000`

### 3. View API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Data Preparation (`/api/data`)

- `POST /api/data/convert` - Convert SAM2 export to training format
- `POST /api/data/validate` - Validate dataset quality
- `POST /api/data/split` - Split dataset into train/val/test

### Training (`/api/train`)

- `POST /api/train/start` - Start training job
- `GET /api/train/{job_id}/status` - Get job status
- `POST /api/train/{job_id}/cancel` - Cancel training
- `GET /api/train/jobs` - List all training jobs

### Experiments (`/api/experiments`)

- `GET /api/experiments` - List experiments
- `GET /api/experiments/{id}` - Get experiment details
- `POST /api/experiments/compare` - Compare experiments
- `DELETE /api/experiments/{id}` - Delete experiment

### Export (`/api/export`)

- `POST /api/export/{job_id}` - Export trained model
- `GET /api/export/{job_id}/download` - Get download info
- `GET /api/export/download_file/{export_id}` - Download model file

## Example Usage

### Start Training

```python
import requests

config = {
    "config": {
        "model_name": "liuhaotian/llava-v1.5-7b",
        "use_lora": True,
        "use_qlora": True,  # For 8GB GPU
        "num_epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "train_data_path": "./output/splits/train.jsonl",
        "val_data_path": "./output/splits/val.jsonl",
        "output_dir": "./output/checkpoints",
    },
    "experiment_name": "llava-7b-qlora-8gb",
    "tags": ["qlora", "8gb-gpu"]
}

response = requests.post("http://localhost:8000/api/train/start", json=config)
job_id = response.json()["job_id"]
print(f"Training started: {job_id}")
```

### Monitor Training

```python
import requests
import time

job_id = "your-job-id"

while True:
    response = requests.get(f"http://localhost:8000/api/train/{job_id}/status")
    status = response.json()

    print(f"Status: {status['status']}")
    print(f"Progress: {status['progress_percentage']:.1f}%")
    print(f"Train Loss: {status['train_loss']}")

    if status['status'] in ['completed', 'failed', 'cancelled']:
        break

    time.sleep(5)
```

## Configuration

### For 8GB GPU

Use QLoRA with minimal batch size:

```python
{
    "use_qlora": True,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_length": 1024,  # Reduce sequence length
}
```

### For 24GB GPU

Use LoRA with larger batches:

```python
{
    "use_lora": True,
    "use_qlora": False,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_length": 2048,
}
```

## Development

### Run in Development Mode

```bash
# Auto-reload on code changes
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Production Deployment

### With Docker

```bash
docker build -t training-api .
docker run -p 8000:8000 --gpus all training-api
```

### With Gunicorn

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Architecture

```
training_api/
├── main.py              # FastAPI application
├── models.py            # Pydantic models
├── requirements.txt     # Dependencies
└── routes/
    ├── data_prep.py     # Data preparation endpoints
    ├── training.py      # Training management
    ├── experiments.py   # Experiment tracking
    └── export.py        # Model export
```

## Notes

- **Storage**: Currently uses in-memory storage. For production, use Redis/PostgreSQL.
- **Authentication**: Not implemented. Add JWT auth for production.
- **File Storage**: Local filesystem. Use S3/cloud storage for production.
- **Job Queue**: Uses threading. For production, use Celery + Redis.
