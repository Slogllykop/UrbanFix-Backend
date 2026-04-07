"""
Pydantic models for request / response validation.
"""

from pydantic import BaseModel, Field
from typing import Literal


class ValidationRequest(BaseModel):
    """Incoming payload from the frontend."""
    id: str = Field(..., description="Row ID in the database to update")
    image_url: str = Field(..., description="Supabase Storage URL of the image")
    category: Literal["pothole", "garbage"] = Field(
        ..., description="Detection category — determines which ONNX model to run"
    )


class ValidationResponse(BaseModel):
    """Response sent back to the frontend."""
    id: str
    status: str = Field(..., description="'valid' if object detected, 'invalid' otherwise")
    message: str
    detections: int = Field(..., description="Number of objects detected in the image")
