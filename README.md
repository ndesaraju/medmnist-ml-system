# ML Project — MedMNIST Classification System

This repository implements an end-to-end, production-oriented machine learning system for medical image classification using the MedMNIST dataset.

The project emphasizes:
- modular architecture  
- reproducibility  
- clean pipeline orchestration  
- deployable inference  

Rather than focusing purely on model performance, the goal is to demonstrate how to design a system that is easy to run, understand, and extend.

---

## What the Project Does

The system implements a full ML pipeline:

data → preprocessing → training → testing → evaluation → serving

Key features:
- Config-driven training (`config.yaml`)
- Modular pipeline stages
- Reproducible outputs (versioned models + metrics)
- FastAPI-based inference API
- Containerized deployment with Docker

---

## Repository Structure

## Repo layout (relevant files)

```
├── README.md          <- The top-level README
├── Makefile           <- shortcuts: setup, data, preprocess, train, test, serve
├── config.yaml        <- YAML config with paths & training options
├── requirements.txt   <- Python dependencies             
├── data/
│   ├── raw/           <- MedMNIST cached dataset
├── models/            <- saved model artifacts
├── results/           <- saved metrics and other outputs
└── src/               <- source code
    ├── data/          <- Scripts to download or generate data
    │   └── make_dataset.py
    ├── preprocess/    <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    ├── model/         <- Scripts to train models and apply models  
    │   ├── model.py
    │   ├── train.py  
    │   └── predict.py
    ├── evaluate/      <- Scripts to evaluate model and output metrics
    │   ├── evaluate.py
    │   └── formatting.py  
    ├── serving/       <- Scripts for serving layer with sample image
    │   ├── app.py
    │   └── adipose.jpg 
    └── common/        <- Helpers (config loader, IO helpers)
```

---
## Running with Docker

To ensure reproducibility and avoid dependency issues, the project can be run inside a Docker container. If you prefer to run this project locally, please see instructions in the ["Running Locally"](#running-locally) section below

### 1. Build the Docker image

```bash
docker build -t medmnist-ml .
```

### 2. Run the full pipeline + serve the model
``` bash
docker run -p 8000:8000 medmnist-ml \
    bash -c "make all && make serve"
```
This will:

- download the dataset
- preprocess the data
- train the model
- evaluate the model
- start the FastAPI server

### 3. Send a test request
In a separate terminal:
```bash
curl -X POST -F "file=@src/serving/adipose.jpg" http://localhost:8000/predict
```
This will use the sample image provided in the repo. To use your own image, replace the path shown above
```bash
curl -X POST -F "file=@full/path/to/test_image.jpg" http://localhost:8000/predict
```
Example response:

```JSON
{
    "prediction_index":6,
    "prediction_label":"Norm Colon Mucosa",
    "confidence":0.5956103801727295
}
```
## Running Locally

### 1. Create environment
Note: You will need Python version 3.10. You might need to download it from https://www.python.org/downloads/release/python-3100/

```bash
conda create -n myenv python=3.10 # to enforce python version
conda activate myenv
python3 -m pip install --upgrade pip && pip install -r requirements.txt
```
Note: make setup is available, but manual activation is recommended since Make does not persist environments across commands.

### 2. Running the Pipeline

Each stage is independently runnable:
```bash
make data        # download dataset
make preprocess  # validate preprocessing pipeline
make train       # train model + save artifacts
make test        # generate predictions
make serve       # launch API
```
To run the full pipeline:
```bash
make all
make serve
```
### 3. Send a test request to the serving layer
In a separate terminal:
```bash
curl -X POST -F "file=@src/serving/adipose.jpg" http://localhost:8000/predict
```
This will use the sample image provided in the repo. To use your own image, replace the path shown above
```bash
curl -X POST -F "file=@full/path/to/test_image.jpg" http://localhost:8000/predict
```
Example response:

```JSON
{
    "prediction_index":6,
    "prediction_label":"Norm Colon Mucosa",
    "confidence":0.5956103801727295
}
```
## Pipeline Breakdown
### 1. Data (make data)

Downloads and caches data to data/raw/

### 2. Preprocessing (make preprocess)
Validates the data pipeline and ensures transforms are correctly applied.

Design choice:
Preprocessing is applied dynamically using PyTorch transforms rather than saving processed images.

This enables:
- stochastic data augmentation
- reduced storage requirements
- simpler pipeline design

### 3. Training (make train)
- Loads data via DataLoaders
- Trains model defined in model.py
- Saves:
    - versioned model (models/model_<timestamp>.pt)
    - models/latest.pt (used for inference)
    - metrics (results/metrics_<timestamp>.json)

### 4. Testing (make test)
- Loads latest.pt
- Runs inference on test dataset
- Outputs predictions

### 5. Evaluation

Metrics computed:
- Accuracy
- F1 Score
- AUC
- Confusion Matrix

Evaluation is decoupled from the model for modularity.

### 6. Online inference (make serve)
Provides real-time predictions via a REST API.

- Built with FastAPI
- Loads models/latest.pt at startup
- Reuses model architecture and preprocessing from training

Key Design Decisions:

- Single source of truth model (latest.pt)
- Shared preprocessing pipeline (prevents train/serve skew)
- Stateless API for easy scaling and containerization

## Configuration (config.yaml)

Example:
``` YAML
data:
  data_flag: pathmnist
  raw_dir: ./data/raw

training:
  device: cpu 
  seed: 42
  batch_size: 64
  epochs: 5
  lr: 0.001

model:
  num_classes: 3
  type: custom_cnn # recommended
  class_names: ["Adipose", "Background", "Debris"]

output:
  model_dir: ./models
  metrics_dir: ./results
```

## Model Design

The model is wrapped in a class (`model.py`) that provides a consistent interface:

```python
train()
predict()
predict_proba()
save() / load()
```

This abstraction:
- decouples model logic from the training pipeline
- makes it easy to swap architectures

## Model Recommendation

Two model options are supported:
- custom_cnn (recommended)
- resnet18

Why use custom_cnn:
- Faster training (optimized for 28×28 images)
- Lower computational cost
- Strong performance on MedMNIST

resnet18 is included as a configurable option to demonstrate extensibility, but custom_cnn is preferred for this task due to better efficiency-performance tradeoff.

## Reproducibility
- Config-driven pipeline
- Fixed random seed
- Versioned outputs with timestamps
- latest.pt pointer for consistent inference

## Makefile Commands
```bash
make data
make preprocess
make train
make test
make serve
make all
make clean
```

## Cleaning
```bash
make clean
```

Removes:
- cached Python files
- model artifacts
- metrics

## Troubleshooting

- File upload error (curl)
    Ensure correct path:
    ```bash
    -F "file=@src/serving/adipose.jpg"
    ```

- Model not found
    ```bash
    make train
    ```

- Import errors
    Run commands from repo root using:
    ```bash
    python -m <module>
    ```

## Design Decisions

**On-the-fly preprocessing** 

Transforms are applied dynamically to:

- enable augmentation
- reduce storage overhead
- simplify pipeline

**Modular architecture**

Each pipeline stage is independently runnable and replaceable.

**Evaluation separation**

Metrics are computed outside the model for flexibility and reuse.

## Future Improvements

- Experiment tracking (MLflow / Weights & Biases)
- Model registry and versioning
- CI/CD and unit tests
- GPU auto-detection and thread utilization
- Batch inference endpoint
- Structured logging to replace prints
- Config validation
- Sanitize requirement.txt so it only contains the essentials
- Strictly version and share preprocessing steps between training and inference to ensure consistency

