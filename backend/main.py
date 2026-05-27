"""
PlanLens — FastAPI backend.

Endpoints:
  POST /upload                                   Upload a PDF and start analysis
  GET  /jobs/{job_id}                            Poll job status + results
  GET  /pages/{job_id}/{page_number}/image       Rendered page PNG
  POST /jobs/{job_id}/measurements/{mid}/approve
  POST /jobs/{job_id}/measurements/{mid}/reject
  POST /jobs/{job_id}/measurements/{mid}/correct?new_value=<float>
  GET  /jobs/{job_id}/export                     Approved measurements as JSON

Architecture notes:
  - All job state is in-memory (fine for prototype; swap for Redis/DB in prod).
  - PDF pages are rendered once on upload; rasters live in memory keyed by job_id.
  - The analysis pipeline runs as an asyncio background task, using
    asyncio.to_thread() so blocking Claude API calls don't stall the event loop.
  - The frontend polls GET /jobs/{job_id} every 2 s and renders progressively
    as pages_processed increments.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Optional

# Load .env from the project root (one level above backend/)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import fitz  # PyMuPDF
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from classifier import classify_page
from measurer import extract_measurements
from models import AnalysisJob, MeasurementError, MeasurementStatus, PageResult
from scale_detector import detect_scale

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="PlanLens", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the single-page frontend from /app
_FRONTEND = Path(__file__).parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/app", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

# job_id → AnalysisJob
_jobs: dict[str, AnalysisJob] = {}

# job_id → { page_number → PNG bytes }
_images: dict[str, dict[int, bytes]] = {}

# Render quality: 150 DPI gives ~1240px wide for an A1 plan — readable but
# not enormous to hold in memory.
_RENDER_DPI = 150
_MAX_PAGES = 30       # safety cap
_MAX_FILE_MB = 80


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"service": "PlanLens", "docs": "/docs", "ui": "/app"}


@app.post("/upload")
async def upload_plan(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accept a PDF upload, render all pages to PNG, create a job, and kick off
    the background analysis pipeline.
    """
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    pdf_bytes = await file.read()

    if len(pdf_bytes) > _MAX_FILE_MB * 1_048_576:
        raise HTTPException(413, f"File exceeds {_MAX_FILE_MB} MB limit.")

    # Render pages
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(422, f"Could not open PDF: {exc}")

    total_pages = min(len(doc), _MAX_PAGES)
    job_id = str(uuid.uuid4())[:8]

    page_store: dict[int, bytes] = {}
    mat = fitz.Matrix(_RENDER_DPI / 72, _RENDER_DPI / 72)

    for i in range(total_pages):
        pix = doc[i].get_pixmap(matrix=mat)
        page_store[i] = pix.tobytes("png")

    doc.close()
    _images[job_id] = page_store

    job = AnalysisJob(
        job_id=job_id,
        filename=file.filename or "upload.pdf",
        status="processing",
        total_pages=total_pages,
    )
    _jobs[job_id] = job

    background_tasks.add_task(_run_analysis, job_id)

    return {"job_id": job_id, "total_pages": total_pages}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Poll for job progress and results."""
    return _require_job(job_id)


@app.get("/pages/{job_id}/{page_number}/image")
async def get_page_image(job_id: str, page_number: int):
    """Return the rendered PNG for a specific page."""
    if job_id not in _images:
        raise HTTPException(404, "Job not found.")
    images = _images[job_id]
    if page_number not in images:
        raise HTTPException(404, f"Page {page_number} not found.")
    return Response(content=images[page_number], media_type="image/png")


@app.post("/jobs/{job_id}/measurements/{measurement_id}/approve")
async def approve(job_id: str, measurement_id: str):
    m = _require_measurement(job_id, measurement_id)
    m.status = MeasurementStatus.APPROVED
    return {"status": m.status}


@app.post("/jobs/{job_id}/measurements/{measurement_id}/reject")
async def reject(job_id: str, measurement_id: str):
    m = _require_measurement(job_id, measurement_id)
    m.status = MeasurementStatus.REJECTED
    return {"status": m.status}


@app.post("/jobs/{job_id}/measurements/{measurement_id}/correct")
async def correct(
    job_id: str,
    measurement_id: str,
    new_value: float = Query(..., description="Builder-corrected value"),
):
    m = _require_measurement(job_id, measurement_id)
    m.corrected_value = new_value
    m.status = MeasurementStatus.CORRECTED
    return {"status": m.status, "corrected_value": new_value}


@app.get("/jobs/{job_id}/export")
async def export_approved(job_id: str):
    """
    Return all APPROVED or CORRECTED measurements as structured JSON.
    This is the payload that would feed a downstream estimating calculator.
    """
    job = _require_job(job_id)

    approved = []
    for page in job.results:
        for m in page.measurements:
            if m.status in (MeasurementStatus.APPROVED, MeasurementStatus.CORRECTED):
                approved.append(
                    {
                        "id": m.id,
                        "label": m.label,
                        "value": m.corrected_value if m.corrected_value is not None else m.value,
                        "original_ai_value": m.value,
                        "unit": m.unit,
                        "status": m.status,
                        "page": page.page_number,
                        "bbox": m.bbox.model_dump(),
                        "confidence": m.confidence,
                        "source": "AI-extracted, builder-verified",
                    }
                )

    return {
        "job_id": job_id,
        "filename": job.filename,
        "approved_measurements": approved,
        "total_approved": len(approved),
    }


# ---------------------------------------------------------------------------
# Background analysis pipeline
# ---------------------------------------------------------------------------

async def _run_analysis(job_id: str) -> None:
    """
    Process each page sequentially:
      1. Classify the page type.
      2. If it's a floor plan, detect scale then extract measurements.
      3. Append results to the job so the frontend can render progressively.
    """
    job = _jobs[job_id]
    images = _images[job_id]

    try:
        for page_num in range(job.total_pages):
            img = images[page_num]

            # Step 1 — classify (runs in thread pool; doesn't block event loop)
            classification = await asyncio.to_thread(classify_page, page_num, img)

            page_result = PageResult(
                page_number=page_num,
                classification=classification,
            )

            # Step 2 — measure only selected (high-confidence floor plan) pages
            if classification.selected:
                scale = await asyncio.to_thread(detect_scale, img)
                page_result.scale = scale

                measurements, errors = await asyncio.to_thread(
                    extract_measurements, img
                )
                page_result.measurements = measurements
                page_result.errors = errors

                # If scale is missing, prepend a SCALE_UNKNOWN error
                if not scale.found:
                    page_result.errors.insert(
                        0,
                        MeasurementError(
                            error_type="SCALE_UNKNOWN",
                            message=(
                                f"Drawing scale not detected: "
                                f"{scale.error or 'no scale notation found'}. "
                                "Measurements shown in the units annotated on the "
                                "drawing; real-world calibration is not available."
                            ),
                            recoverable=True,
                        ),
                    )

            job.results.append(page_result)
            job.pages_processed += 1

        job.status = "complete"

    except Exception as exc:
        job.status = "error"
        job.error_message = str(exc)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_job(job_id: str) -> AnalysisJob:
    if job_id not in _jobs:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    return _jobs[job_id]


def _require_measurement(job_id: str, measurement_id: str):
    job = _require_job(job_id)
    for page in job.results:
        for m in page.measurements:
            if m.id == measurement_id:
                return m
    raise HTTPException(404, f"Measurement '{measurement_id}' not found.")
