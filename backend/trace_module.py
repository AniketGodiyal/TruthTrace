# ============================================================
# VERITRACE - TRACE MODULE (importable functions only)
# The FastAPI app/routes have been moved to main.py, which combines
# this module with the deepfake-detection pipeline (pipeline.py) into
# one server.
# ============================================================

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import imagehash
import hashlib
import sqlite3
import cv2
import shutil
import os
import uuid
import traceback
from datetime import datetime


# ============================================================
# SECTION 1: DATABASE (formerly database.py)
# ============================================================

def init_db():
    conn = sqlite3.connect("veritrace.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL,
            phash TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            filepath TEXT NOT NULL UNIQUE,
            source TEXT,
            uploaded_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def insert_image(original_filename, stored_filename, phash, sha256, filepath, source="unknown"):
    conn = sqlite3.connect("veritrace.db")
    cursor = conn.cursor()

    uploaded_at = datetime.now().isoformat()
    evidence_id = None

    try:
        cursor.execute(
            """INSERT INTO evidence
               (original_filename, stored_filename, phash, sha256, filepath, source, uploaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (original_filename, stored_filename, str(phash), sha256, filepath, source, uploaded_at)
        )
        conn.commit()
        evidence_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        cursor.execute("SELECT id FROM evidence WHERE filepath = ?", (filepath,))
        row = cursor.fetchone()
        if row:
            evidence_id = row[0]

    conn.close()
    return evidence_id


def get_all_images():
    conn = sqlite3.connect("veritrace.db")
    cursor = conn.cursor()

    cursor.execute(
        """SELECT original_filename, stored_filename, phash, sha256,
                  filepath, source, uploaded_at
           FROM evidence"""
    )
    rows = cursor.fetchall()

    conn.close()
    return rows


# ============================================================
# SECTION 2: EXIF METADATA (formerly exif_utils.py)
# ============================================================

def extract_exif(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
    except Exception:
        return {
            "available": False,
            "message": "Could not read image or unsupported format for EXIF."
        }

    if not exif_data:
        return {
            "available": False,
            "message": "No EXIF metadata found. This is common for images "
                        "downloaded from social media platforms (WhatsApp, "
                        "Instagram, Telegram), which strip metadata on upload."
        }

    metadata = {}
    gps_info = {}

    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, tag_id)

        if tag == "GPSInfo":
            for gps_tag_id, gps_value in value.items():
                gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                gps_info[gps_tag] = gps_value
        else:
            try:
                str_value = str(value)
                if len(str_value) < 200:
                    metadata[tag] = str_value
            except Exception:
                continue

    result = {
        "available": True,
        "metadata": metadata
    }

    gps_coords = parse_gps(gps_info)
    if gps_coords:
        result["gps_location"] = gps_coords
    elif gps_info:
        result["gps_raw"] = str(gps_info)

    return result


def parse_gps(gps_info):
    try:
        if not gps_info or "GPSLatitude" not in gps_info or "GPSLongitude" not in gps_info:
            return None

        def to_decimal(dms, ref):
            degrees = float(dms[0])
            minutes = float(dms[1])
            seconds = float(dms[2])
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            if ref in ["S", "W"]:
                decimal = -decimal
            return decimal

        lat = to_decimal(gps_info["GPSLatitude"], gps_info.get("GPSLatitudeRef", "N"))
        lon = to_decimal(gps_info["GPSLongitude"], gps_info.get("GPSLongitudeRef", "E"))

        return {
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "google_maps_link": f"https://www.google.com/maps?q={lat},{lon}"
        }
    except Exception:
        return None


# ============================================================
# SECTION 3: VIDEO UTILS (formerly video_utils.py)
# ============================================================

def extract_middle_frame(video_path, output_folder="input"):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not read video or video has no frames")

    middle_frame_number = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_number)

    success, frame = cap.read()

    cap.release()

    if not success:
        raise ValueError("Could not extract frame from video")

    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    unique_id = uuid.uuid4().hex[:8]
    frame_filename = f"{video_filename}_frame_{unique_id}.jpg"
    frame_path = os.path.join(output_folder, frame_filename)

    cv2.imwrite(frame_path, frame)

    return frame_path


# ============================================================
# SECTION 4: TRACE ENGINE (formerly trace.py)
# ============================================================

STRONG_VISUAL_THRESHOLD = 10
POSSIBLE_MATCH_THRESHOLD = 32


def get_phash(image_path):
    image = Image.open(image_path)
    return imagehash.phash(image)


def get_sha256(image_path):
    sha256_hash = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def fingerprint_media(file_path):
    return {
        "phash": get_phash(file_path),
        "sha256": get_sha256(file_path)
    }


def classify_match(sha_match, distance):
    if sha_match:
        return (
            "EXACT DUPLICATE",
            "CONFIRMED DUPLICATE",
            "SHA-256 hash is byte-for-byte identical to a stored record."
        )

    if distance <= STRONG_VISUAL_THRESHOLD:
        return (
            "VISUALLY SIMILAR",
            "STRONG VISUAL MATCH",
            f"Perceptual hash distance is {distance}, well within the "
            f"strong-match range (<= {STRONG_VISUAL_THRESHOLD}). Likely the "
            f"same media with minor edits (compression, resize, brightness)."
        )

    if distance <= POSSIBLE_MATCH_THRESHOLD:
        return (
            "VISUALLY SIMILAR",
            "POSSIBLE MATCH",
            f"Perceptual hash distance is {distance}, within the configured "
            f"threshold (<= {POSSIBLE_MATCH_THRESHOLD}). May be the same "
            f"media with moderate edits, or a coincidental similarity."
        )

    return (
        "NOT SIMILAR",
        "NO MATCH",
        f"Perceptual hash distance is {distance}, above the configured "
        f"threshold (> {POSSIBLE_MATCH_THRESHOLD})."
    )


def search_local_evidence(file_path, fingerprint, threshold=POSSIBLE_MATCH_THRESHOLD):

    new_phash = fingerprint["phash"]
    new_sha256 = fingerprint["sha256"]

    existing_images = get_all_images()

    matches = []

    for row in existing_images:
        (db_original_filename, db_stored_filename, db_phash_str,
         db_sha256, db_filepath, db_source, db_uploaded_at) = row

        if os.path.abspath(db_filepath) == os.path.abspath(file_path):
            continue

        sha_match = (db_sha256 == new_sha256)

        if sha_match:
            distance = 0
        else:
            db_hash = imagehash.hex_to_hash(db_phash_str)
            distance = int(new_phash - db_hash)

        match_type, category, reason = classify_match(sha_match, distance)

        if category == "NO MATCH":
            continue

        matches.append({
            "origin": "local_database",
            "filename": db_original_filename,
            "stored_filename": db_stored_filename,
            "filepath": db_filepath,
            "source": db_source,
            "url": None,
            "uploaded_at": db_uploaded_at,
            "sha256": db_sha256,
            "sha256_match": sha_match,
            "phash": db_phash_str,
            "distance": distance,
            "threshold": threshold,
            "match_type": match_type,
            "category": category,
            "status": "CONFIRMED MATCH" if sha_match else "POSSIBLY RELATED",
            "reason": reason
        })

    return matches


def search_external_sources(file_path, fingerprint):
    return {
        "status": "NOT_CONFIGURED",
        "message": "No external search API connected. Only local evidence database is being searched.",
        "matches": []
    }


def merge_matches(local_matches, external_matches):
    all_matches = local_matches + external_matches

    seen = set()
    deduped = []

    for m in all_matches:
        key = (m.get("filename"), m.get("uploaded_at"), m.get("origin"))
        if key not in seen:
            seen.add(key)
            deduped.append(m)

    return deduped


def build_timeline(matches):
    return sorted(matches, key=lambda x: x["uploaded_at"])


def build_propagation(timeline, current_filename, current_source):

    if not timeline:
        return {
            "earliest_observed": None,
            "path": [],
            "summary": "No prior matches found. This may be the first observed instance."
        }

    earliest = timeline[0]

    path = [
        {
            "step": 1,
            "filename": earliest["filename"],
            "source": earliest["source"],
            "uploaded_at": earliest["uploaded_at"],
            "role": "EARLIEST OBSERVED"
        }
    ]

    for i, match in enumerate(timeline[1:], start=2):
        path.append({
            "step": i,
            "filename": match["filename"],
            "source": match["source"],
            "uploaded_at": match["uploaded_at"],
            "role": "RE-UPLOAD"
        })

    path.append({
        "step": len(path) + 1,
        "filename": current_filename,
        "source": current_source,
        "uploaded_at": "now",
        "role": "CURRENT UPLOAD"
    })

    summary = (
        f"Tracing begins from the earliest observed matching record "
        f"({len(timeline)} prior observation(s) found). "
        f"Earliest observed source: {earliest['source']}. "
        f"This reflects observed evidence only and does not claim to "
        f"identify the true original creator or location."
    )

    return {
        "earliest_observed": earliest["source"],
        "path": path,
        "summary": summary
    }


def build_investigation_summary(timeline, propagation):

    exact_duplicates = [m for m in timeline if m["category"] == "CONFIRMED DUPLICATE"]
    strong_visual = [m for m in timeline if m["category"] == "STRONG VISUAL MATCH"]
    possible = [m for m in timeline if m["category"] == "POSSIBLE MATCH"]

    sources = sorted(set(m["source"] for m in timeline if m["source"]))

    strongest_match = None
    if timeline:
        strongest_match = min(timeline, key=lambda m: m["distance"])

    return {
        "total_related_records": len(timeline),
        "exact_duplicates": len(exact_duplicates),
        "strong_visual_matches": len(strong_visual),
        "possible_matches": len(possible),
        "earliest_observed_record": {
            "filename": timeline[0]["filename"],
            "source": timeline[0]["source"],
            "uploaded_at": timeline[0]["uploaded_at"]
        } if timeline else None,
        "distinct_sources_observed": len(sources),
        "sources_observed": sources,
        "propagation_steps": len(propagation["path"]),
        "strongest_match": {
            "filename": strongest_match["filename"],
            "category": strongest_match["category"],
            "distance": strongest_match["distance"]
        } if strongest_match else None,
        "integrity_status": "VERIFIED"
    }


def build_phash_chart_data(timeline, threshold):
    return {
        "chart_type": "bar",
        "x_label": "Evidence Filename",
        "y_label": "pHash Hamming Distance",
        "threshold_line": threshold,
        "data": [
            {
                "filename": m["filename"],
                "distance": m["distance"],
                "category": m["category"]
            }
            for m in timeline
        ]
    }


def build_chain_of_custody(evidence_id, fingerprint, source, original_filename, stored_filename):
    return [
        {
            "evidence_id": evidence_id,
            "event": "EVIDENCE INGESTED",
            "original_filename": original_filename,
            "stored_filename": stored_filename,
            "sha256_at_ingestion": fingerprint["sha256"],
            "phash_at_ingestion": str(fingerprint["phash"]),
            "timestamp": datetime.now().isoformat(),
            "source_reported": source,
            "handler": "Veritrace System (Automated)",
            "integrity_check": "PASSED"
        }
    ]


def build_investigation_findings(timeline, summary):

    findings = []

    if not timeline:
        findings.append("No related records were found in the evidence database.")
        findings.append("This may be the first observed instance of this media.")
        return findings

    findings.append(f"{summary['total_related_records']} related record(s) were found.")

    if summary["exact_duplicates"] > 0:
        findings.append(f"{summary['exact_duplicates']} exact duplicate(s) were found (identical SHA-256 hash).")

    if summary["strong_visual_matches"] > 0:
        findings.append(f"{summary['strong_visual_matches']} strongly similar record(s) were found (minor edits detected).")

    if summary["possible_matches"] > 0:
        findings.append(f"{summary['possible_matches']} possible match(es) were found (moderate visual similarity).")

    earliest = summary["earliest_observed_record"]
    if earliest:
        findings.append(
            f"The earliest observed matching record was recorded at {earliest['uploaded_at']} "
            f"(reported source: {earliest['source']})."
        )

    findings.append(
        f"Records were observed across {summary['distinct_sources_observed']} "
        f"distinct reported source(s)."
    )

    findings.append(
        "No external platform search was performed. Findings are limited to "
        "records present in this system's evidence database."
    )

    return findings


def build_risk_alert(timeline):

    if not timeline:
        return {
            "alert_level": "NO MATCH",
            "message": "No related evidence found. This appears to be a new, previously unseen record."
        }

    categories_present = set(m["category"] for m in timeline)

    if "CONFIRMED DUPLICATE" in categories_present:
        return {
            "alert_level": "CONFIRMED DUPLICATE",
            "message": "This media is an exact byte-for-byte duplicate of at least one existing evidence record."
        }

    if "STRONG VISUAL MATCH" in categories_present:
        return {
            "alert_level": "STRONG VISUAL MATCH",
            "message": "This media is visually near-identical to at least one existing evidence record, with only minor edits detected."
        }

    if "POSSIBLE MATCH" in categories_present:
        return {
            "alert_level": "POSSIBLE MATCH",
            "message": "This media shows moderate visual similarity to at least one existing evidence record. Manual review recommended."
        }

    return {
        "alert_level": "NO MATCH",
        "message": "No related evidence found above the configured similarity threshold."
    }


def trace_media(file_path, original_filename, stored_filename, source="unknown",
                 threshold=POSSIBLE_MATCH_THRESHOLD):

    fingerprint = fingerprint_media(file_path)

    local_matches = search_local_evidence(file_path, fingerprint, threshold=threshold)
    external_result = search_external_sources(file_path, fingerprint)

    all_matches = merge_matches(local_matches, external_result["matches"])
    timeline = build_timeline(all_matches)
    propagation = build_propagation(timeline, original_filename, source)

    evidence_id = insert_image(
        original_filename=original_filename,
        stored_filename=stored_filename,
        phash=str(fingerprint["phash"]),
        sha256=fingerprint["sha256"],
        filepath=file_path,
        source=source
    )

    investigation_summary = build_investigation_summary(timeline, propagation)
    phash_chart = build_phash_chart_data(timeline, threshold)
    chain_of_custody = build_chain_of_custody(
        evidence_id, fingerprint, source, original_filename, stored_filename
    )
    investigation_findings = build_investigation_findings(timeline, investigation_summary)
    risk_alert = build_risk_alert(timeline)
    exif_data = extract_exif(file_path)

    result = {
        "uploaded_file": original_filename,
        "stored_as": stored_filename,
        "phash": str(fingerprint["phash"]),
        "sha256": fingerprint["sha256"],
        "source": source,
        "external_search": {
            "status": external_result["status"],
            "message": external_result["message"]
        },
        "total_matches": len(timeline),
        "timeline": timeline,
        "propagation": propagation,
        "chain_of_custody": chain_of_custody,
        "evidence_id": evidence_id,
        "threshold_used": threshold,
        "investigation_summary": investigation_summary,
        "investigation_findings": investigation_findings,
        "risk_alert": risk_alert,
        "phash_chart": phash_chart,
        "exif_metadata": exif_data
    }

    return result


# ============================================================
# NOTE: FastAPI app/routes now live in main.py, which calls
# init_db_and_dirs() at startup and trace_media() / get_all_images()
# per-request. See main.py for the combined server.
# ============================================================

def init_db_and_dirs(upload_dir="input"):
    """Call once at server startup (replaces the old module-level side effects)."""
    os.makedirs(upload_dir, exist_ok=True)
    init_db()