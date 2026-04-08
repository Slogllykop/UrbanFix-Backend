"""
UrbanFix Backend – FastAPI application entry point.

Exposes a single POST endpoint that validates images against YOLO models
and updates the Supabase database accordingly.
"""

import logging
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import config

from models import ValidationRequest, ValidationResponse
from services.supabase_client import fetch_image_bytes, update_validation_status
from services.yolo_detector import get_detector

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="UrbanFix Validation API",
    description=(
        "Accepts an image URL and category, runs YOLOv8 ONNX detection, "
        "and marks the report as valid or invalid in Supabase."
    ),
    version="1.0.0",
)

# Allow all origins during development – tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def health_check():
    """Simple liveness probe."""
    return {"status": "ok", "service": "UrbanFix Validation API"}


# ── Validation endpoint ──────────────────────────────────────────────────────
@app.post("/api/validate", response_model=ValidationResponse, tags=["Validation"])
async def validate_image(payload: ValidationRequest):
    """
    Validate an image against the appropriate YOLO model.

    **Flow**:
    1. Download the image from Supabase Storage via its public URL.
    2. Run the YOLO ONNX model that corresponds to ``category``.
    3. If at least one object is detected → ``status = "verified"``.
    4. Otherwise → ``status = "rejected"``.
    5. If there is no model for the category → ``status = "pending"``.
    6. Update the ``status`` and ``ai_verified`` columns in Supabase.

    Returns the validation result including the number of detections.
    """
    row_id = payload.id
    image_url = payload.image_url
    category = payload.category

    logger.info("Received validation request  id=%s  category=%s", row_id, category)

    # ── 1. Download image ─────────────────────────────────────────────────
    try:
        image_bytes = await fetch_image_bytes(image_url)
        logger.info("Downloaded image (%d bytes)", len(image_bytes))
    except Exception as exc:
        logger.error("Failed to download image: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Could not download image from the provided URL: {exc}",
        )

    # ── 2. Decode image ──────────────────────────────────────────────────
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        logger.error("Failed to decode image bytes")
        raise HTTPException(
            status_code=400,
            detail="Downloaded data could not be decoded as an image.",
        )

    # ── 3. Run YOLO detection ─────────────────────────────────────────────
    if category in config.MODEL_PATHS:
        try:
            detector = get_detector(category)
            detections = detector.detect(image)
            logger.info("Detection complete  detections=%d", detections)
        except FileNotFoundError as exc:
            logger.error("Model file missing: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))
        except Exception as exc:
            logger.error("Detection failed: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"YOLO detection error: {exc}",
            )
        
        status = "verified" if detections > 0 else "rejected"
        ai_verified = True
    else:
        detections = 0
        status = "pending"
        ai_verified = False
        logger.info("No AI model for category '%s', setting status to pending.", category)

    # ── 5. Update Supabase ────────────────────────────────────────────────
    try:
        update_validation_status(row_id, status, ai_verified, category)
        logger.info("Updated row %s → %s (ai_verified=%s)", row_id, status, ai_verified)
    except Exception as exc:
        logger.error("Supabase update failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update database: {exc}",
        )

    return ValidationResponse(
        id=row_id,
        status=status,
        message=f"Image processed as '{status}' with {detections} detection(s).",
        detections=detections,
    )
