"""
PlanLens - Scale detector.

Asks Claude Vision to locate the scale notation or graphical scale bar on a
floor plan page and returns a structured ScaleInfo.

When scale is not found we return found=False with a descriptive error.
The pipeline inserts a SCALE_UNKNOWN MeasurementError so the builder knows
measurements are in raw annotation units rather than calibrated distances.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Optional

import anthropic

from models import ScaleInfo

_client = anthropic.Anthropic()
_MODEL = "claude-sonnet-4-6"

_SCALE_PROMPT = (
    "You are reading a residential architectural floor plan to find its scale.\n\n"
    "Look for: title block text like 'Scale 1:100' or '1/4\"=1\'', or a graphical "
    "scale bar.\n\n"
    "Common ratios: 1:100 -> 100, 1:50 -> 50, 1/4\"=1\' -> 48, 1/8\"=1\' -> 96.\n\n"
    "Return ONLY valid JSON:\n"
    "{\n"
    '  "found": true or false,\n'
    '  "notation": "<exact text or null>",\n'
    '  "ratio": <real/drawing ratio as number or null>,\n'
    '  "unit": "metric" or "imperial" or null,\n'
    '  "method": "notation" or "scale_bar" or "unknown",\n'
    '  "confidence": <0.0 to 1.0>,\n'
    '  "error": "<reason if not found, otherwise null>"\n'
    "}"
)


def detect_scale(image_bytes: bytes) -> ScaleInfo:
    """Detect the drawing scale from a rendered floor plan image."""
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    try:
        response = _client.messages.create(
            model=_MODEL,
            max_tokens=256,
            messages=[{
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
                    {"type": "text", "text": _SCALE_PROMPT},
                ],
            }],
        )
        raw = response.content[0].text.strip()
        data = _parse_json(raw)
        return ScaleInfo(
            found=bool(data.get("found", False)),
            notation=data.get("notation"),
            ratio=_safe_float(data.get("ratio")),
            unit=data.get("unit"),
            method=str(data.get("method", "unknown")),
            confidence=float(data.get("confidence", 0.0)),
            error=data.get("error"),
        )
    except Exception as exc:
        return ScaleInfo(
            found=False,
            method="unknown",
            confidence=0.0,
            error=f"Scale detection failed: {exc}",
        )


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in response: {text!r}")
    return json.loads(match.group())


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
