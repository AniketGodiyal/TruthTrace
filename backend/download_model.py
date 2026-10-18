
import os
import shutil
from pathlib import Path

import kagglehub

BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"
DOWNLOAD_DIR = BACKEND_DIR / ".kaggle_download"

MODEL_NAME = "xception_deepfake_final.keras"

def main():
    resource_type = os.environ.get(
        "KAGGLE_RESOURCE_TYPE", "model"
    ).lower()

    handle = os.environ.get("KAGGLE_RESOURCE_HANDLE")

    if not handle:
        raise RuntimeError(
            "Set KAGGLE_RESOURCE_HANDLE in Render Environment."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if resource_type == "model":
        downloaded = kagglehub.model_download(
            handle,
            output_dir=str(DOWNLOAD_DIR),
        )
    elif resource_type == "dataset":
        downloaded = kagglehub.dataset_download(handle)
    else:
        raise ValueError(
            "KAGGLE_RESOURCE_TYPE must be 'model' or 'dataset'."
        )

    search_roots = [
        Path(downloaded),
        DOWNLOAD_DIR,
    ]

    candidates = []
    for root in search_roots:
        if root.exists():
            candidates.extend(root.rglob("*.keras"))

    # Remove duplicate paths
    candidates = list(dict.fromkeys(p.resolve() for p in candidates))

    if not candidates:
        raise FileNotFoundError(
            "No .keras file found in the downloaded Kaggle resource."
        )

    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple .keras files found. Select the correct file "
            "explicitly in download_model.py."
        )

    destination = MODELS_DIR / MODEL_NAME
    source = candidates[0]

    if source != destination.resolve():
        shutil.copy2(source, destination)

    if destination.stat().st_size == 0:
        raise RuntimeError("Downloaded model file is empty.")

    print(
        f"Model ready: {destination} "
        f"({destination.stat().st_size} bytes)"
    )

if __name__ == "__main__":
    main()
