"""
pipeline.py
-----------
Refactored version of analyze_video_fixed.py, turned into importable
functions instead of a CLI script. The model and YuNet detector are
loaded ONCE at server startup (not on every request) for speed.

Exposes:
    load_models()                  -> call once at FastAPI startup
    analyze_image(image_path)      -> dict result for a single image
    analyze_video(video_path)      -> dict result for a video (frame sampling)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from pipeline_gradcam_addition import make_gradcam

# ============================================================
# CONFIG (same values as your original script)
# ============================================================

MODEL_PATH = Path(os.environ.get("XCEPTION_MODEL_PATH", "models/xception_deepfake_final.keras"))
YUNET_PATH = Path(os.environ.get("YUNET_MODEL_PATH", "models/face_detection_yunet_2023mar.onnx"))

FRAME_STRIDE = 30
FACE_CONFIDENCE = 0.60
NMS_THRESHOLD = 0.30
TOP_K = 5000
FACE_MARGIN = 0.20
IMAGE_SIZE = (299, 299)

THRESHOLD = 0.3328  # calibrated threshold
FAKE_IF_SCORE_GTE_THRESHOLD = True

GRADCAM_DIR = Path(os.environ.get("GRADCAM_OUTPUT_DIR", "input/gradcam"))
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

# Populated by load_models()
_model = None
_detector = None


def load_models():
    """Call this once when the FastAPI app starts up."""
    global _model, _detector

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Xception model not found: {MODEL_PATH}")
    if not YUNET_PATH.exists():
        raise FileNotFoundError(f"YuNet model not found: {YUNET_PATH}")

    print(f"Loading Xception model from {MODEL_PATH} ...")
    _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Xception loaded. Input:", _model.input_shape, "Output:", _model.output_shape)

    print(f"Loading YuNet from {YUNET_PATH} ...")
    _detector = cv2.FaceDetectorYN.create(
        str(YUNET_PATH), "", (320, 320), FACE_CONFIDENCE, NMS_THRESHOLD, TOP_K
    )
    print("YuNet loaded.")


def _preprocess_face(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    face = face_rgb.astype(np.float32)
    face = face / 127.5 - 1.0
    return face


def _crop_face(frame, face):
    x, y, w, h = face[:4]
    x, y, w, h = int(x), int(y), int(w), int(h)
    margin_x = int(w * FACE_MARGIN)
    margin_y = int(h * FACE_MARGIN)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(frame.shape[1], x + w + margin_x)
    y2 = min(frame.shape[0], y + h + margin_y)
    return frame[y1:y2, x1:x2]


def _score_to_label(score: float) -> str:
    if FAKE_IF_SCORE_GTE_THRESHOLD:
        return "FAKE" if score >= THRESHOLD else "REAL"
    return "REAL" if score >= THRESHOLD else "FAKE"


def _predict_face(face_crop_bgr):
    processed = _preprocess_face(face_crop_bgr)
    input_tensor = np.expand_dims(processed, axis=0)
    prediction = _model(input_tensor, training=False).numpy()
    score = float(prediction[0][0])
    return score, _score_to_label(score)



def _detect_largest_face(frame):
    frame_h, frame_w = frame.shape[:2]
    _detector.setInputSize((frame_w, frame_h))
    _, faces = _detector.detect(frame)
    if faces is None or len(faces) == 0:
        return None, None
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    face_crop = _crop_face(frame, largest_face)
    return largest_face, face_crop if face_crop.size else None


def generate_gradcam(image_path: str, output_path: str):
    if _model is None or _detector is None:
        raise RuntimeError("Models not loaded. Call load_models() at startup.")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    _, face_crop = _detect_largest_face(frame)
    if face_crop is None:
        raise RuntimeError("No face detected for Grad-CAM.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return make_gradcam(_model, face_crop, output_path)


def analyze_image(image_path: str) -> dict:
    """
    Runs face detection + Xception scoring on a single image.
    Returns a result dict with score/label, or an error/no-face status.
    """
    if _model is None or _detector is None:
        raise RuntimeError("Models not loaded. Call load_models() at startup.")

    frame = cv2.imread(str(image_path))
    if frame is None:
        return {"status": "error", "message": f"Could not read image: {image_path}"}

    frame_h, frame_w = frame.shape[:2]
    _detector.setInputSize((frame_w, frame_h))

    _, faces = _detector.detect(frame)
    if faces is None or len(faces) == 0:
        return {"status": "no_face_detected"}

    largest_face = max(faces, key=lambda f: f[2] * f[3])
    face_crop = _crop_face(frame, largest_face)
    if face_crop.size == 0:
        return {"status": "error", "message": "Empty face crop"}

    score, label = _predict_face(face_crop)

    result = {
        "status": "completed",
        "score": score,
        "threshold": THRESHOLD,
        "label": label,
        "face_box": {
            "x": int(largest_face[0]),
            "y": int(largest_face[1]),
            "w": int(largest_face[2]),
            "h": int(largest_face[3]),
        },
        "gradcam_path": None,
    }

    try:
        gradcam_path = GRADCAM_DIR / f"{Path(image_path).stem}_gradcam.jpg"
        make_gradcam(_model, face_crop, str(gradcam_path))
        result["gradcam_path"] = str(gradcam_path)
    except Exception as exc:
        print(f"[WARN] Image Grad-CAM unavailable: {exc}")
        result["gradcam_error"] = str(exc)

    return result


def analyze_video(video_path: str) -> dict:
    """
    Samples frames every FRAME_STRIDE, runs face detection + scoring on each,
    and aggregates a video-level verdict. (Grad-CAM omitted here for speed —
    add back from analyze_video_fixed.py if you need heatmaps in the demo.)
    """
    if _model is None or _detector is None:
        raise RuntimeError("Models not loaded. Call load_models() at startup.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "error", "message": f"Could not open video: {video_path}"}

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = []
    frame_number = 0
    start_time = time.time()
    gradcam_candidate = None
    gradcam_distance = None
    midpoint = total_frames // 2 if total_frames > 0 else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % FRAME_STRIDE != 0:
            frame_number += 1
            continue

        frame_h, frame_w = frame.shape[:2]
        _detector.setInputSize((frame_w, frame_h))

        try:
            _, faces = _detector.detect(frame)
            if faces is None or len(faces) == 0:
                frame_number += 1
                continue

            largest_face = max(faces, key=lambda f: f[2] * f[3])
            face_crop = _crop_face(frame, largest_face)
            if face_crop.size == 0:
                frame_number += 1
                continue

            score, label = _predict_face(face_crop)
            results.append({"frame_number": frame_number, "score": score, "label": label})

            if midpoint is not None:
                distance = abs(frame_number - midpoint)
                if gradcam_distance is None or distance < gradcam_distance:
                    gradcam_distance = distance
                    gradcam_candidate = (frame.copy(), frame_number)

        except Exception as exc:
            print(f"[ERROR] Frame {frame_number}: {exc}")

        frame_number += 1

    cap.release()
    processing_time = time.time() - start_time

    if not results:
        return {"status": "no_face_detected", "total_frames": total_frames}

    scores = [r["score"] for r in results]
    mean_score = float(np.mean(scores))
    max_score = float(np.max(scores))
    fake_frame_count = sum(1 for r in results if r["label"] == "FAKE")

    video_label = "LIKELY_MANIPULATION" if mean_score >= THRESHOLD else "LIKELY_REAL"

    result = {
        "status": "completed",
        "threshold": THRESHOLD,
        "video_result": {
            "mean_score": mean_score,
            "max_score": max_score,
            "fake_frame_count": fake_frame_count,
            "frames_analyzed": len(results),
            "label": video_label,
        },
        "processing_time_sec": processing_time,
        "frame_results": results,
        "gradcam_path": None,
        "gradcam_frame": None,
        "gradcam_timestamp_sec": None,
    }

    if gradcam_candidate is not None:
        try:
            selected_frame, selected_frame_number = gradcam_candidate
            _, selected_face = _detect_largest_face(selected_frame)
            if selected_face is None:
                raise RuntimeError("No face detected in selected Grad-CAM frame.")

            gradcam_path = GRADCAM_DIR / f"{Path(video_path).stem}_gradcam.jpg"
            make_gradcam(_model, selected_face, str(gradcam_path))

            result["gradcam_path"] = str(gradcam_path)
            result["gradcam_frame"] = int(selected_frame_number)
            result["gradcam_timestamp_sec"] = (
                float(selected_frame_number / fps) if fps > 0 else None
            )
        except Exception as exc:
            print(f"[WARN] Video Grad-CAM unavailable: {exc}")
            result["gradcam_error"] = str(exc)

    return result
