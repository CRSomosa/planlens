"""
PlanLens — Pydantic data models.

Every Claude response is validated through these models before touching
the job store. Nothing reaches the frontend as raw LLM text.
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Page classification
# ---------------------------------------------------------------------------

class PageType(str, Enum):
    FLOOR_PLAN      = "FLOOR_PLAN"
    ELEVATION       = "ELEVATION"
    SECTION         = "SECTION"
    ELECTRICAL_PLAN = "ELECTRICAL_PLAN"
    PLUMBING_PLAN   = "PLUMBING_PLAN"
    DEMOLITION_PLAN = "DEMOLITION_PLAN"
    SITE_PLAN       = "SITE_PLAN"
    ROOF_PLAN       = "ROOF_PLAN"
    RENDER_3D       = "RENDER_3D"
    DETAIL_DRAWING  = "DETAIL_DRAWING"
    SCHEDULE_LEGEND = "SCHEDULE_LEGEND"
    OTHER           = "OTHER"


class PageClassification(BaseModel):
    page_number: int
    page_type: PageType
    confidence: float                   # 0.0 – 1.0
    reasoning: str                      # one-sentence visual justification
    selected: bool                      # True → route to measurement pipeline


# ---------------------------------------------------------------------------
# Scale / calibration
# ---------------------------------------------------------------------------

class ScaleInfo(BaseModel):
    found: bool
    notation: Optional[str] = None      # e.g. "1:100" or "1/4\" = 1'"
    ratio: Optional[float] = None       # real-world units per 1 drawing unit
    unit: Optional[str] = None          # "metric" | "imperial"
    method: str = "unknown"             # "notation" | "scale_bar" | "unknown"
    confidence: float = 0.0
    error: Optional[str] = None         # set when found=False


# ---------------------------------------------------------------------------
# Measurements & errors
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Normalised image coordinates. (0,0) = top-left, (1,1) = bottom-right."""
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    w: float = Field(..., ge=0.0, le=1.0)
    h: float = Field(..., ge=0.0, le=1.0)


class MeasurementStatus(str, Enum):
    PENDING   = "pending"
    APPROVED  = "approved"
    REJECTED  = "rejected"
    CORRECTED = "corrected"


class Measurement(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str                          # e.g. "Bedroom 1 – Width"
    value: float
    unit: str                           # exactly as read from the plan
    bbox: BoundingBox                   # where the annotation sits on the image
    confidence: float                   # 0.0 – 1.0
    reasoning: str                      # what Claude could see
    status: MeasurementStatus = MeasurementStatus.PENDING
    corrected_value: Optional[float] = None   # builder override


class MeasurementError(BaseModel):
    """
    Returned instead of guessing when Claude cannot read a value clearly.

    error_type values:
      SCALE_UNKNOWN      – no scale info found; raw annotation values only
      LOW_CONFIDENCE     – value visible but not reliably legible
      ILLEGIBLE          – text/dimension line too blurry or cut off
      UNIT_UNCLEAR       – value readable but unit missing or ambiguous
      AMBIGUOUS_VALUE    – multiple conflicting annotations
      EXTRACTION_FAILED  – Claude API / JSON parse error
    """
    error_type: str
    message: str
    bbox: Optional[BoundingBox] = None  # approximate location if known
    recoverable: bool = True


# ---------------------------------------------------------------------------
# Per-page and job-level containers
# ---------------------------------------------------------------------------

class PageResult(BaseModel):
    page_number: int
    classification: PageClassification
    scale: Optional[ScaleInfo] = None
    measurements: List[Measurement] = []
    errors: List[MeasurementError] = []


class AnalysisJob(BaseModel):
    job_id: str
    filename: str
    status: str = "processing"          # "processing" | "complete" | "error"
    total_pages: int
    pages_processed: int = 0
    results: List[PageResult] = []
    error_message: Optional[str] = None
