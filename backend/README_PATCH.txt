Replace the three backend files in D:\subodh\backend:
1. pipeline_gradcam_addition.py
2. pipeline.py
3. main.py

No change to trace_module.py is required.

Run from backend:
python -m uvicorn main:app --reload --port 8000

The /api/analyze response now includes gradcam_url.
For videos it also includes gradcam_frame and gradcam_timestamp_sec.
