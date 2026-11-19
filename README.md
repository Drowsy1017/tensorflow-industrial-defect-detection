
# Industrial Defect Detection with TensorFlow

An industrial surface defect classification project built with TensorFlow, MobileNetV2, FastAPI, and Docker GPU inference.

## Project Overview

This project demonstrates an end-to-end computer vision workflow:

1. Dataset preparation
2. Binary defect classification
3. Transfer learning
4. Fine-tuning
5. Model evaluation
6. Classification threshold tuning
7. REST API deployment
8. Docker GPU inference

## Tech Stack

- Python 3.12
- TensorFlow 2.21
- Keras
- MobileNetV2
- tf.data
- FastAPI
- Uvicorn
- Docker
- NVIDIA CUDA / cuDNN

## Dataset

Dataset: KolektorSDD2

Images are classified as:

- `normal` (0)
- `defect` (1)

Labels are derived from ground-truth segmentation masks.

### Dataset Split

| Split | Normal | Defect | Total |
|---|---:|---:|---:|
| Train | 1772 | 209 | 1981 |
| Validation | 313 | 37 | 350 |
| Test | 894 | 110 | 1004 |

The official test split is kept separate from training.

## Model Architecture

Input: 192 × 512 × 3

MobileNetV2:
- ImageNet pretrained weights
- Classification head with Global Average Pooling
- Dropout 0.3
- Dense layer with sigmoid activation

Training strategy:

- Stage 1: Freeze the backbone and train the classification head.
- Stage 2: Fine-tune the last 30 backbone layers.
- BatchNormalization layers remain frozen during fine-tuning.

## Validation Results

Threshold: 0.5

| Metric | Stage 1 | Fine-tuned |
|---|---:|---:|
| Accuracy | 90.57% | 94.86% |
| Precision | 56.67% | 88.00% |
| Recall | 45.95% | 59.46% |
| F1 | 50.75% | 70.97% |
| PR-AUC | 0.5768 | 0.7527 |

Fine-tuning improved validation classification performance.

## Threshold Tuning

Using the validation set, a threshold was selected under the constraint:

Recall >= 80%

Selected fine-tuned threshold: 0.109373

Validation performance:

- Recall: 81.08%
- Precision: 40.00%
- F1: 53.57%
- False positives: 45
- False negatives: 7

The selected threshold is experimental and has not been approved for production deployment.

## Run Inference API

Start locally:

```bash
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Swagger UI:

http://127.0.0.1:8000/docs

## Docker GPU Deployment

Build image:

```bash
docker build -t tensorflow-defect-api:1.3 .
```

Run with GPU:

```bash
docker run --rm --gpus all \
  -p 8003:8000 \
  tensorflow-defect-api:1.3
```

Swagger UI:

http://127.0.0.1:8003/docs

## API Endpoints

### GET /health

Returns API health status.

### POST /predict

Upload a PNG or JPEG image.

Example response:

```json
{
  "filename": "20000.png",
  "predicted_class": "normal",
  "defect_score": 0.07520077377557755,
  "threshold": 0.10937318205833435,
  "is_defect": false
}
```

## Known Limitations

- The defect class is significantly underrepresented.
- Some defects remain undetected.
- Validation metrics are based on a relatively small defect sample.
- Sigmoid scores are not calibrated probabilities.
- Additional testing is necessary before production use.
- GPU availability does not guarantee every operation executes on GPU.

## Dataset License

KolektorSDD2 is provided under a non-commercial license.

Check the official dataset terms before redistribution or commercial use.
