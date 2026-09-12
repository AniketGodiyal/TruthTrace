# TruthTrace — AI-Powered Deepfake Detection

TruthTrace is a hackathon project for detecting potentially manipulated images and
videos using an AI-based deepfake detection pipeline.

## Core workflow

Input image/video → face extraction & preprocessing → deepfake model inference →
prediction/result → explainability/forensic output

## Main technologies

- Python
- FastAPI
- OpenCV
- PyTorch / TensorFlow
- Xception-based deepfake detection
- Grad-CAM
- React / frontend tooling

## Project structure

```text
backend/     Backend API and inference pipeline
frontend/    Frontend application
```

## Setup

### Backend

Create a Python environment and install the dependencies listed in
`requirements.txt` (if present).

Then start the FastAPI application using the entry point provided by the project.

### Frontend

Install the frontend dependencies with:

```bash
npm install
```

Then start the frontend using the project's package scripts.

## Model

The trained model weights are not included in this repository because of GitHub
file-size limitations. See `backend/models/README.md` for the expected model file.

## Team

Developed as a hackathon project.

## Note

This repository contains the application/source code and configuration needed to
understand and run the project. Large datasets, generated outputs, local databases,
dependency folders, and model weights are intentionally excluded.
