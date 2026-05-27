# PlanLens 🏗️

**AI-powered construction plan measurement extraction with human-in-the-loop verification.**

PlanLens reads architectural floor plan PDFs, extracts dimension annotations using Claude Vision, and produces a builder-verified JSON dataset ready to feed downstream estimating tools. It is designed around the principle that AI measurements must be *confirmed* by a human before they drive cost decisions.

---

## What it does

1. **Upload** a multi-page construction PDF
2. **Classify** every page automatically — floor plan, elevation, section, electrical, demolition, etc.
3. **Detect scale** from title block notation or graphical scale bar
4. **Extract measurements** — only from explicitly annotated dimension lines; never calculated or estimated
5. **Review** each measurement in a visual overlay UI — approve, reject, or correct
6. **Export** verified measurements as structured JSON for downstream estimating pipelines

---

## Architecture

```
planlens/
├── backend/
│   ├── main.py            # FastAPI app — upload, poll, approve/reject/export
│   ├── classifier.py      # Claude Vision page classifier (sonnet-4-6)
│   ├── scale_detector.py  # Scale notation / scale bar reader (sonnet-4-6)
│   ├── measurer.py        # Measurement extractor (opus-4-6) with error typing
│   ├── models.py          # Pydantic v2 data models
│   └── requirements.txt
├── frontend/
│   └── index.html         # Single-file dark-themed SPA, no build step
├── .env                   # ANTHROPIC_API_KEY (not committed)
└── run.sh                 # One-command setup & launch
```

### AI pipeline (per page)

```
PDF page → PNG (150 DPI)
    │
    ▼
classify_page()          claude-sonnet-4-6
    │  selected=True only for FLOOR_PLAN ≥ 0.65 confidence
    ▼
detect_scale()           claude-sonnet-4-6
    │  Returns ratio, unit, method, confidence
    │  Inserts SCALE_UNKNOWN error if not found
    ▼
extract_measurements()   claude-opus-4-6
    │  Reads ONLY printed annotations — never calculates
    │  Returns normalised bounding boxes (0–1) for UI overlays
    │  Moves anything < 0.70 confidence to errors[]
    ▼
PageResult { measurements[], errors[], scale }
```

All Claude calls run via `asyncio.to_thread()` so the FastAPI event loop stays non-blocking while pages process progressively.

---

## Key design decisions

### Preventing hallucinated measurements
The measurer prompt opens with an absolute prohibition: *"Do NOT calculate any measurement. Only read what is printed."* Anything the model is uncertain about goes to a typed `errors[]` list rather than being silently included in results.

### Structured errors instead of guessing
Five error types — `LOW_CONFIDENCE`, `ILLEGIBLE`, `UNIT_UNCLEAR`, `AMBIGUOUS_VALUE`, `SCALE_CONFLICT` — give the reviewer actionable information about *why* a measurement wasn't extracted, rather than a vague failure.

### Separating value accuracy from location accuracy
A measurement can have high confidence in its numeric value (the number is clearly legible) while having an approximate bounding box (the annotation is small or crowded). The `reasoning` field documents this distinction explicitly, and zero-sized bboxes are flagged as `[NOTE: bounding box location is approximate]`.

### Human-in-the-loop as a first-class feature
No measurement reaches the export without explicit builder approval. The UI supports three outcomes: **approve** (AI value accepted), **correct** (builder overrides the value), **reject** (measurement discarded). The export JSON carries `original_ai_value` alongside the final `value` so the correction is auditable.

### Truncation resilience
Dense plans can exceed token limits. The parser detects `stop_reason == "max_tokens"`, salvages every complete measurement object emitted before the cutoff, and surfaces a `TRUNCATED_RESPONSE` error with count of recovered items — rather than discarding everything with a generic failure.

---

## Setup

**Requirements:** Python 3.9–3.12, an Anthropic API key

```bash
# 1. Clone / download the project
cd planlens

# 2. Add your API key
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env

# 3. Install & launch
bash run.sh
```

Open **http://localhost:8000/app** in your browser.

> **Windows users:** run the install and launch commands manually:
> ```powershell
> cd backend
> py -3.12 -m pip install -r requirements.txt
> py -3.12 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
> ```

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload a PDF; returns `job_id` |
| `GET` | `/jobs/{job_id}` | Poll progress + results |
| `GET` | `/pages/{job_id}/{page}/image` | Rendered PNG for a page |
| `POST` | `/jobs/{job_id}/measurements/{mid}/approve` | Approve a measurement |
| `POST` | `/jobs/{job_id}/measurements/{mid}/reject` | Reject a measurement |
| `POST` | `/jobs/{job_id}/measurements/{mid}/correct?new_value=3.6` | Override value |
| `GET` | `/jobs/{job_id}/export` | Download approved measurements as JSON |

Interactive docs available at **http://localhost:8000/docs**

---

## Export format

```json
{
  "job_id": "a3f9c1b2",
  "filename": "housing_project.pdf",
  "total_approved": 14,
  "approved_measurements": [
    {
      "id": "uuid",
      "label": "Bedroom 1 – Width",
      "value": 3.5,
      "original_ai_value": 3.5,
      "unit": "m",
      "status": "APPROVED",
      "page": 2,
      "confidence": 0.94,
      "bbox": { "x": 0.42, "y": 0.31, "w": 0.08, "h": 0.02 },
      "source": "AI-extracted, builder-verified"
    }
  ]
}
```

The `original_ai_value` / `value` split means corrections are always auditable. This payload is designed to feed directly into estimating tools (Buildsoft, CostX, Cubit) or a cost-per-unit-rate calculation pipeline.

---

## Models used

| Component | Model | Reason |
|-----------|-------|--------|
| Page classifier | `claude-sonnet-4-6` | Fast binary classification; 12 page types |
| Scale detector | `claude-sonnet-4-6` | Pattern matching on title block text |
| Measurement extractor | `claude-opus-4-6` | Spatial reasoning over dense annotation fields |

---

## Limitations (prototype)

- Job state is **in-memory** — restarting the server clears all jobs. Production would use Redis or a database.
- Images are held in memory — not suitable for very large batches without a blob store.
- Bounding box placement is approximate for small or crowded annotations; the overlay is indicative, not pixel-perfect.
- Best results on black-and-white or greyscale plans at standard architectural scales (1:50, 1:100, 1/4"=1').

---

## Tech stack

- **Claude API** (Anthropic) — multimodal vision, structured JSON output
- **FastAPI** — async REST API with background task pipeline
- **PyMuPDF** — PDF-to-PNG rendering at 150 DPI
- **Pydantic v2** — data validation and serialisation
- **Vanilla JS** — no-build frontend with real-time polling and canvas overlays
