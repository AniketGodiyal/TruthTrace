
"""
main.py — Unified VeriTrace backend
------------------------------------
Combines two previously-separate FastAPI apps into ONE server:
    1. pipeline.py       -> Xception deepfake detection (verify)
    2. trace_module.py   -> pHash/SHA256 duplicate + origin tracing (trace)
 
Run:
    uvicorn main:app --reload --port 8000
 
Folder layout expected:
    backend/
        main.py
        pipeline.py
        trace_module.py
        models/
            xception_deepfake_final.keras
            face_detection_yunet_2023mar.onnx
        input/              (trace module's upload storage, auto-created)
        outputs/            (verify module's temp uploads, auto-created)
        veritrace.db        (auto-created SQLite db for trace matching)
"""
 
import shutil
import uuid
import traceback
from datetime import datetime
from pathlib import Path
 
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
 
import pipeline
import trace_module
 
UPLOAD_DIR = Path("outputs/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
 
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
 
app = FastAPI(title="VeriTrace API")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated Grad-CAM images to React.
app.mount("/media", StaticFiles(directory="input"), name="media")
 
 
@app.on_event("startup")
def _startup():
    pipeline.load_models()               # loads Xception + YuNet once
    trace_module.init_db_and_dirs("input")  # creates veritrace.db + input/ folder
 
 
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )
 
 
@app.get("/")
def home():
    return {"message": "VeriTrace backend is running"}
 
 
@app.get("/api/health")
def health():
    return {"status": "ok"}
 
 
# ============================================================
# Standalone endpoints (kept for direct/manual testing)
# ============================================================
 
@app.post("/api/verify")
async def verify_only(file: UploadFile = File(...)):
    """Runs ONLY the deepfake detection pipeline."""
    ext = Path(file.filename).suffix.lower()
    if ext not in VIDEO_EXTS and ext not in IMAGE_EXTS:
        raise HTTPException(400, f"Unsupported file type: {ext}")
 
    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
 
    if ext in VIDEO_EXTS:
        return pipeline.analyze_video(str(temp_path))
    return pipeline.analyze_image(str(temp_path))
 
 
@app.post("/api/trace")
async def trace_only(file: UploadFile = File(...), source: str = Form("unknown")):
    """Runs ONLY the origin-tracing (pHash/SHA256 + EXIF) module."""
    original_filename = file.filename
    unique_id = uuid.uuid4().hex[:10]
    ext = Path(original_filename).suffix.lower()
    stored_filename = f"{unique_id}_{original_filename}"
    file_path = trace_module.os.path.join("input", stored_filename)
 
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
 
    if ext in VIDEO_EXTS:
        frame_path = trace_module.extract_middle_frame(file_path, "input")
        stored_filename = trace_module.os.path.basename(frame_path)
        file_path = frame_path
 
    return trace_module.trace_media(
        file_path=file_path,
        original_filename=original_filename,
        stored_filename=stored_filename,
        source=source,
    )
 
 
@app.get("/api/evidence")
def list_evidence():
    rows = trace_module.get_all_images()
    evidence = [
        {
            "original_filename": row[0],
            "stored_filename": row[1],
            "phash": row[2],
            "sha256": row[3],
            "filepath": row[4],
            "source": row[5],
            "uploaded_at": row[6],
        }
        for row in rows
    ]
    return {"total_records": len(evidence), "evidence": evidence}
 
 
# ============================================================
# Combined endpoint — this is what the React frontend calls
# ============================================================
 
def _label_to_verdict(label: str, is_video: bool) -> str:
    if is_video:
        if label == "LIKELY_MANIPULATION":
            return "AI-altered"
        if label == "LIKELY_REAL":
            return "Authentic"
        return "Inconclusive"
    if label == "FAKE":
        return "AI-generated"
    if label == "REAL":
        return "Authentic"
    return "Inconclusive"
 
 
def _score_to_confidence(score: float, threshold: float) -> int:
    distance = abs(score - threshold)
    return int(round(50 + min(distance * 100, 49)))
 
 
def _custody_icon(event_action: str) -> str:
    mapping = {
        "EVIDENCE INGESTED": "upload",
    }
    return mapping.get(event_action, "scan")

def _build_gradcam_url(request: Request, verify_result: dict):
    gradcam_path = verify_result.get("gradcam_path")

    if not gradcam_path:
        print("[WARN] pipeline did not return gradcam_path")
        return None

    try:
        input_dir = Path("input").resolve()
        gradcam_file = Path(gradcam_path).resolve()

        relative_path = gradcam_file.relative_to(input_dir)

        url = (
            str(request.base_url).rstrip("/")
            + "/media/"
            + str(relative_path).replace("\\", "/")
        )

        print("[INFO] Grad-CAM URL:", url)
        return url

    except Exception as exc:
        print(
            f"[WARN] Could not build Grad-CAM URL "
            f"from '{gradcam_path}': {exc}"
        )
        return None
 
 
@app.post("/api/analyze")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    source: str = Form("unknown")
):
    """
    Runs BOTH the deepfake-detection pipeline and the origin-tracing module
    on the same upload, and returns one merged object shaped for the
    frontend's EvidenceCase type (client/src/mock-data.ts), plus extra
    trace-specific fields (propagation/custody) the CaseDetail page can use.
    """
    original_filename = file.filename
    ext = Path(original_filename).suffix.lower()
    if ext not in VIDEO_EXTS and ext not in IMAGE_EXTS:
        raise HTTPException(400, f"Unsupported file type: {ext}")
 
    is_video = ext in VIDEO_EXTS
 
    # Save ONE copy of the upload, used by both modules.
    case_id = f"VT-{datetime.now().strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:6]}"
    stored_filename = f"{case_id}{ext}"
    file_path = trace_module.os.path.join("input", stored_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
 
    # ---- 1. Deepfake detection ----
    try:
        if is_video:
            verify_result = pipeline.analyze_video(file_path)
        else:
            verify_result = pipeline.analyze_image(file_path)
    except Exception as exc:
        raise HTTPException(500, f"Verification failed: {exc}")
 
    if verify_result.get("status") == "no_face_detected":
        raise HTTPException(422, "No face detected in the submitted evidence.")
    if verify_result.get("status") == "error":
        raise HTTPException(500, verify_result.get("message", "Unknown verify error"))
    gradcam_url = _build_gradcam_url(request, verify_result)
 
    if is_video:
        score = verify_result["video_result"]["mean_score"]
        label = verify_result["video_result"]["label"]
    else:
        score = verify_result["score"]
        label = verify_result["label"]
 
    # ---- 2. Origin tracing (pHash/SHA256 + EXIF) ----
    trace_file_path = file_path
    if is_video:
        # trace_media works on a still image, so extract a frame for it
        # (separately from the pipeline's own frame sampling).
        trace_file_path = trace_module.extract_middle_frame(file_path, "input")
 
    try:
        trace_result = trace_module.trace_media(
            file_path=trace_file_path,
            original_filename=original_filename,
            stored_filename=stored_filename,
            source=source,
        )
    except Exception as exc:
        # Tracing failing shouldn't block the verify result from reaching the UI.
        trace_result = {"error": f"Tracing failed: {exc}"}
 
    verdict = _label_to_verdict(label, is_video)
    confidence = _score_to_confidence(score, verify_result["threshold"])
 
    # Map trace's chain_of_custody entries -> frontend's CustodyEntry shape
    custody_entries = []
    for entry in trace_result.get("chain_of_custody", []):
        custody_entries.append({
            "time": entry.get("timestamp", ""),
            "actor": entry.get("handler", "System"),
            "action": entry.get("event", ""),
            "detail": f"SHA-256: {entry.get('sha256_at_ingestion', '')[:16]}...",
            "icon": _custody_icon(entry.get("event", "")),
        })
 
    # NOTE: trace_module only searches the LOCAL database (no external
    # platform search configured), so this is NOT the same as the frontend's
    # mock PropagationNode data (which shows Telegram/X/WhatsApp handles).
    # This reflects re-uploads *within this system* only.
    local_propagation_path = trace_result.get("propagation", {}).get("path", [])
 
    evidence_case = {
        "id": case_id,
        "title": Path(original_filename).stem,
        "filename": original_filename,
        "type": "Video" if is_video else "Image",
        "verdict": verdict,
        "confidence": confidence,
        "urgency": confidence,
        "status": "new",
        "submitted": datetime.now().strftime("%d %b %Y, %H:%M"),
        "relativeTime": "just now",
        "officer": "Unassigned",
        "department": "Cyber Cell",
        "source": source,
        "location": trace_result.get("exif_metadata", {}).get("gps_location", {}).get("google_maps_link", "Unknown"),
        "thumb": "generic",
        "accent": "#6f8fb8",
        "summary": f"Model raw score: {score:.4f} (threshold {verify_result['threshold']}). Label: {label}.",
        "hash": trace_result.get("sha256", ""),
        "ingestedAt": datetime.now().isoformat(),
        "provenance": "valid" if trace_result.get("exif_metadata", {}).get("available") else "missing",
        "frequencyScore": confidence,
        "frequencyFinding": "Frequency-domain analysis not yet wired up — placeholder from confidence score.",
 
        # Extra fields beyond the base EvidenceCase type — CaseDetail can
        # opt into displaying these if you extend the frontend types.
        "local_propagation_path": local_propagation_path,
        "custody_entries": custody_entries,
        "investigation_findings": trace_result.get("investigation_findings", []),
        "risk_alert": trace_result.get("risk_alert", {}),
        "exif_metadata": trace_result.get("exif_metadata", {}),
        "gradcam_url": gradcam_url,
        "gradcam_frame": verify_result.get("gradcam_frame"),
        "gradcam_timestamp_sec": verify_result.get("gradcam_timestamp_sec"),
        "gradcam_error": verify_result.get("gradcam_error"),

        "raw_verify_result": verify_result,
        "raw_trace_result": trace_result,
    }
    print("[INFO] API gradcam_url:", evidence_case["gradcam_url"])
    return evidence_case
