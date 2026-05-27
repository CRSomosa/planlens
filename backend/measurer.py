"""
PlanLens — Measurement extractor.

This is the core AI component. It sends a classified floor plan image to
Claude Vision with a highly constrained prompt that:

  1. Demands measurements come ONLY from visible, legible dimension annotations.
  2. Returns normalised bounding boxes so the frontend can draw overlays.
  3. Uses a structured errors list instead of hallucinating uncertain values.
  4. Separates value accuracy from location accuracy — Claude can be confident
     in the number while being approximate about the bounding box position.

Key design choices that address the job spec explicitly:
  - "Preventing hallucinated or unsupported measurements" → the prompt opens
    with a strict prohibition and the confidence threshold filters the rest.
  - "Returning structured errors instead of guessing" → every uncertain item
    goes to errors[], not measurements[].
  - "Separating value accuracy from location accuracy" → the reasoning field
    documents what was visually confirmed vs. what is approximate.
  - "Distinguishing actual coordinates from placeholder coordinates" → we
    clamp all bbox values to [0,1] and flag any bbox that is suspiciously
    small or zero-sized as a potential placeholder.
"""

import base64
import json
import re
import uuid
from typing import List, Tuple

import anthropic

from models import BoundingBox, Measurement, MeasurementError, MeasurementStatus

_client = anthropic.Anthropic()

# Use the most capable model for spatial reasoning and measurement reading.
_MODEL = "claude-opus-4-6"

# Any measurement below this confidence is moved to errors[].
_MIN_CONFIDENCE = 0.70

_MEASUREMENT_PROMPT = """\
You are an expert quantity surveyor reading a residential architectural floor \
plan. Your task is to extract dimension measurements that are EXPLICITLY \
ANNOTATED on this drawing.

═══════════════════════════════════════════════════════
WHAT YOU MAY EXTRACT
═══════════════════════════════════════════════════════
  • Dimension lines with numeric values (the number between tick marks / arrows)
  • Room area labels if shown (e.g. "12.5 m²" inside a room)
  • Overall building extents where explicitly dimensioned
  • Door and window openings where a numeric dimension is shown

═══════════════════════════════════════════════════════
ABSOLUTE PROHIBITIONS
═══════════════════════════════════════════════════════
  ✗  Do NOT calculate any measurement. Only read what is printed.
  ✗  Do NOT estimate, interpolate or average values.
  ✗  Do NOT extract dimensions from elevation or section views if they appear
     in a corner inset — only the main floor plan area.
  ✗  Do NOT guess a unit if it is not printed. Put it in errors[].
  ✗  Do NOT include any measurement with confidence < 0.70.

═══════════════════════════════════════════════════════
BOUNDING BOX RULES
═══════════════════════════════════════════════════════
  • bbox surrounds the ANNOTATION (the number + tick marks), NOT the room.
  • Coordinates are normalised: (0,0) = top-left corner, (1,1) = bottom-right.
  • Be as precise as you can, but if the location is approximate, say so in
    reasoning. A location that is approximate is still useful; a made-up
    value is not.
  • If you cannot locate the annotation at all, put the item in errors[] with
    error_type ILLEGIBLE.

═══════════════════════════════════════════════════════
STRUCTURED ERRORS (use these instead of guessing)
═══════════════════════════════════════════════════════
  LOW_CONFIDENCE    – annotation visible but not reliably legible
  ILLEGIBLE         – blurry, cut off, or overprinted
  UNIT_UNCLEAR      – value readable, unit missing or ambiguous
  AMBIGUOUS_VALUE   – two conflicting numbers for the same dimension
  SCALE_CONFLICT    – annotation contradicts the noted scale

═══════════════════════════════════════════════════════
RESPONSE FORMAT
═══════════════════════════════════════════════════════
Return ONLY valid JSON — no markdown fences, no preamble:
{
  "measurements": [
    {
      "label":      "<descriptive name, e.g. 'Bedroom 1 – Width'>",
      "value":      <number>,
      "unit":       "<unit exactly as printed, e.g. mm, m, ft, in>",
      "bbox":       {"x": <0-1>, "y": <0-1>, "w": <0-1>, "h": <0-1>},
      "confidence": <float 0.70–1.0>,
      "reasoning":  "<what you can see: describe the annotation and its legibility>"
    }
  ],
  "errors": [
    {
      "error_type": "<type from list above>",
      "message":    "<clear description of the problem>",
      "bbox":       {"x": <0-1>, "y": <0-1>, "w": <0-1>, "h": <0-1>} or null,
      "recoverable": true or false
    }
  ]
}

If you cannot read ANY measurements clearly, return {"measurements": [], "errors": [...]}.
Never return an empty errors array if measurements are missing — explain why.\
"""


def extract_measurements(
    image_bytes: bytes,
) -> Tuple[List[Measurement], List[MeasurementError]]:
    """
    Extract annotated measurements from a floor plan image.

    Args:
        image_bytes: PNG-encoded raster of a classified floor plan page.

    Returns:
        (measurements, errors) — both lists may be non-empty simultaneously.
        Items below _MIN_CONFIDENCE are automatically moved to errors[].
    """
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    try:
        response = _client.messages.create(
            model=_MODEL,
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": _MEASUREMENT_PROMPT},
                    ],
                }
            ],
        )

        raw = response.content[0].text.strip()
        data = _parse_json(raw, response.stop_reason)

        measurements: List[Measurement] = []
        errors: List[MeasurementError] = []

        # Parse measurements; anything below threshold → error list
        for item in data.get("measurements", []):
            confidence = float(item.get("confidence", 0.0))
            bbox = _parse_bbox(item.get("bbox", {}))

            if confidence < _MIN_CONFIDENCE:
                errors.append(
                    MeasurementError(
                        error_type="LOW_CONFIDENCE",
                        message=(
                            f"'{item.get('label', '?')}' confidence {confidence:.0%} "
                            f"is below threshold — value not included. "
                            f"Reasoning: {item.get('reasoning', '')}"
                        ),
                        bbox=bbox,
                        recoverable=True,
                    )
                )
                continue

            # Flag suspiciously zero-sized bboxes as placeholder coordinates
            is_placeholder = bbox.w < 0.001 and bbox.h < 0.001

            measurements.append(
                Measurement(
                    id=str(uuid.uuid4()),
                    label=str(item.get("label", "Unknown")),
                    value=float(item.get("value", 0.0)),
                    unit=str(item.get("unit", "?")),
                    bbox=bbox,
                    confidence=confidence,
                    reasoning=(
                        item.get("reasoning", "")
                        + (" [NOTE: bounding box location is approximate]" if is_placeholder else "")
                    ),
                    status=MeasurementStatus.PENDING,
                )
            )

        # Parse errors from Claude
        for err in data.get("errors", []):
            bbox_data = err.get("bbox")
            errors.append(
                MeasurementError(
                    error_type=str(err.get("error_type", "UNKNOWN")),
                    message=str(err.get("message", "Unknown error")),
                    bbox=_parse_bbox(bbox_data) if bbox_data else None,
                    recoverable=bool(err.get("recoverable", True)),
                )
            )

        return measurements, errors

    except Exception as exc:
        return [], [
            MeasurementError(
                error_type="EXTRACTION_FAILED",
                message=f"Measurement extraction failed: {exc}",
                recoverable=False,
            )
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str, stop_reason: str = "end_turn") -> dict:
    """
    Parse Claude's JSON response, with truncation recovery.

    If the response was cut off mid-JSON (stop_reason == "max_tokens"), we
    try to salvage any complete measurement objects that were emitted before
    the truncation point, rather than discarding everything.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text!r}")

    blob = match.group()

    # Happy path — response is complete
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass

    # Response was truncated — try to salvage complete measurement objects.
    # Each complete item ends with a closing brace + comma or closing brace
    # before the outer array closes.
    if stop_reason != "max_tokens":
        raise ValueError(f"Malformed JSON (stop_reason={stop_reason!r}): {blob[:200]!r}")

    salvaged: list[dict] = []
    # Match individual {...} objects inside the measurements array
    for obj_match in re.finditer(r'\{[^{}]*"label"[^{}]*\}', blob, re.DOTALL):
        try:
            obj = json.loads(obj_match.group())
            salvaged.append(obj)
        except json.JSONDecodeError:
            continue

    return {
        "measurements": salvaged,
        "errors": [
            {
                "error_type": "TRUNCATED_RESPONSE",
                "message": (
                    f"Claude's response was cut off at the token limit. "
                    f"{len(salvaged)} measurements were salvaged before truncation. "
                    "Re-upload the page or split the plan into smaller sections for full coverage."
                ),
                "recoverable": True,
            }
        ],
    }


def _parse_bbox(data: dict | None) -> BoundingBox:
    """Parse and clamp a bounding box dict. Returns a zero-point box on failure."""
    if not data or not isinstance(data, dict):
        return BoundingBox(x=0.0, y=0.0, w=0.0, h=0.0)
    return BoundingBox(
        x=_clamp(data.get("x", 0.0)),
        y=_clamp(data.get("y", 0.0)),
        w=_clamp(data.get("w", 0.0)),
        h=_clamp(data.get("h", 0.0)),
    )


def _clamp(v) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0
