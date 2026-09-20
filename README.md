# Concrete Surface Crack Detection — End-to-End MLOps Pipeline

[![CI/CD](https://github.com/bartekpurc/concrete-crack-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/bartekpurc/concrete-crack-detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![AWS](https://img.shields.io/badge/AWS-Lambda%20%7C%20ECR%20%7C%20S3%20%7C%20APIGW-232F3E?logo=amazonwebservices)
![Terraform](https://img.shields.io/badge/IaC-Terraform-7B42BC?logo=terraform)
![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

Binary image classifier that detects cracks in concrete surfaces, wrapped in a production-ready MLOps lifecycle: experiment tracking, orchestrated training, containerized cloud deployment, and automated drift monitoring with conditional retraining triggers.

---

## Problem Overview
Manual visual inspection of concrete infrastructure (buildings, bridges, roads) for structural defects is slow, expensive, and subject to human inspector variance. 

This project automates the first pass of visual inspection: given a photo of a concrete surface, the model classifies it as containing a crack or being intact. The service is exposed as a Serverless HTTP API ready to sit behind mobile inspection tools or edge devices.

* **Dataset:** [Surface Crack Detection](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) (40,000 images, 227×227px, balanced binary labels).

---

## Architecture

```text
┌─────────────────┐      ┌──────────────┐      ┌───────────────────┐
│   Kaggle data   │ ───▶ │ prepare_data │ ───▶ │  data/processed/  │
└─────────────────┘      │  (dedup,     │      │  train/val/test   │
                         │   balanced   │      │  + reference set  │
                         │   split)     │      └─────────┬─────────┘
                         └──────────────┘                │
                                                         ▼
┌──────────────────────────────────────────────────────────────────┐
│  training_flow.py (Prefect)                                      │
│  download → dedup → split → train.py (MobileNetV3, MLflow,       │
│  early stopping) → register best model in MLflow Model Registry  │
└───────────────────────────────┬──────────────────────────────────┘
                                │ model weights baked into image
                                ▼
┌──────────────┐   push   ┌───────────┐   pulls image   ┌────────────┐
│ Docker build │ ───────▶ │    ECR    │ ───────────────▶ │   Lambda   │
└──────────────┘          └───────────┘                  │ (container)│
                                                         └─────┬──────┘
                                                  API Gateway  │ logs prediction
                                                  (HTTP API)   │ + image stats
                                                       ▲       ▼
                                                       │   ┌────────┐
                                             client ───┘   │   S3   │
                                                            └───┬────┘
                                                                │
                                                                ▼
                                       ┌────────────────────────────────────┐
                                       │ monitoring_flow.py (Prefect)       │
                                       │ calculate_drift.py (Evidently):    │
                                       │ reference vs. production image     │
                                       │ stats → if drift_share ≥ threshold │
                                       │ → re-trigger training_flow()       │
                                       └────────────────────────────────────┘

```

All AWS infrastructure (S3, ECR, IAM, Lambda, API Gateway) is fully automated and provisioned via Terraform.

Tech Stack

Category                                        Tools / Technologies
```text
Model & Framework                    PyTorch, torchvision (MobileNetV3-Small, transfer learning)
Tracking & Registry                                    MLflow
Workflow Orchestration                              Prefect 3.x
Containerization                                       Docker
Cloud Infrastructure               AWS Lambda (container image), Amazon ECR, Amazon S3, API Gateway
Infrastructure as Code                                Terraform
Monitoring & Drift                   Evidently AI (DataDriftPreset on image stats)
Testing,pytest                          moto (mocked AWS integration tests)
Code Quality                          Ruff (linter & formatter), pre-commit hooks
CI/CD Pipeline                      GitHub Actions (automated testing, Docker build & push to ECR)

```
Repository Structure
```text
.
├── prepare_data.py           # Download, deduplicate, balance, and split dataset
├── train.py                  # MobileNetV3 training loop + MLflow logging/registry
├── training_flow.py          # Prefect flow: data preparation → training → registry
├── calculate_drift.py        # Reference vs. production drift via Evidently AI
├── monitoring_flow.py        # Prefect flow: scheduled drift check → auto-retraining
├── lambda_model/
│   ├── lambda_function.py    # Serverless inference handler + S3 telemetry logger
│   ├── Dockerfile            # Multi-stage Lambda container image
│   ├── model_weights.pt      # Optimized model weights baked into image
│   └── class_mapping.json    # Label encoder mappings
├── terraform/                # IaC: S3 bucket, ECR, IAM roles, Lambda, API Gateway
├── tests/                    # Unit tests + Moto-based AWS integration tests
├── .github/workflows/ci.yml  # CI/CD: Ruff, Pytest, Docker ECR push
├── Makefile                  # Automation shortcuts (lint, test, build, push)
└── .pre-commit-config.yaml   # Local Git hooks

```
### Local Development & Setup
1. Environment Setup
```text
python -m venv .venv
.venv\Scripts\activate        # Windows (or: source .venv/bin/activate on Linux/macOS)
pip install -r lambda_model/requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```
2. Model Training
```text
# Start local tracking services (in separate terminals if running locally)
prefect server start
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts

# Run training pipeline
python training_flow.py
```
Pipeline downloads data via kagglehub, deduplicates near-identical patches, balances classes, trains MobileNetV3-Small with early stopping on validation F1 score, and registers the checkpoint under the champion alias in MLflow.

3. Testing & Code Quality
```text
make lint              # Ruff checking & formatting
make test              # Run all tests
make test-unit         # Fast unit tests (no cloud dependencies)
make test-integration  # AWS integration test with mocked S3 via moto

```
### Cloud Deployment & Inference
1. Deploy Infrastructure (Terraform)
```text
cd terraform
terraform init
terraform plan
terraform apply
```
Terraform outputs the API Gateway invoke URL (e.g. https://<api_id>.execute-api.<region>.amazonaws.com/).

2. Running Inference (HTTP POST)

Via PowerShell:
```text
$imgBytes = [System.IO.File]::ReadAllBytes("test_image.jpg"); `
$imgBase64 = [System.Convert]::ToBase64String($imgBytes); `
$body = @{ image =$imgBase64 } | ConvertTo-Json; `
Invoke-RestMethod -Uri "https://<api_endpoint>/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```
Via cURL:
```text
IMG_B64=$(base64 -w 0 test_image.jpg) # on macOS: base64 -i test_image.jpg
curl -X POST "https://<api_endpoint>/predict" \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMG_B64\"}"
```
Example Response:
{
  "prediction": "Positive",
  "confidence": 0.9998
}

Telemetry & Drift Monitoring
1. Prediction Logging: Every inference request executed by AWS Lambda asynchronously logs metadata and image statistics (dimensions, aspect ratio, mean brightness, per-channel RGB standard deviations, predicted class, and confidence) as JSON files into an Amazon S3 telemetry bucket.

2. Drift Detection: calculate_drift.py loads these production statistics alongside the baseline reference set and evaluates statistical drift using Evidently AI (DataDriftPreset).

3. Automated Retraining: The monitoring_flow.py (orchestrated by Prefect) regularly executes drift checks. If drift_share exceeds the safety threshold, it triggers training_flow() automatically, achieving closed-loop continuous learning.
