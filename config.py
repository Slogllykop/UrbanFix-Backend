"""
Application configuration loaded from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Supabase ──────────────────────────────────────────────────────────────────
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "https://your-project-id.supabase.co")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "your-supabase-service-role-key")

# ── Database ──────────────────────────────────────────────────────────────────
TABLE_NAME: str = os.getenv("TABLE_NAME", "reports")

# ── ONNX Model Paths ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS: dict[str, Path] = {
    "pothole": BASE_DIR / "pothole_best.onnx",
    "garbage": BASE_DIR / "garbage_best.onnx",
}

# ── YOLO Detection Thresholds ────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
