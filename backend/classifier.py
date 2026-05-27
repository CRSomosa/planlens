"""
PlanLens — Page classifier.

Sends each rendered PDF page to Claude Vision and returns a PageClassification.
Only FLOOR_PLAN pages with confidence >= CONFIDENCE_THRESHOLD are routed to
the measurement pipeline. Every other page type is labelled and skipped.

Key design decisions:
  - Demolition, electrical, and elevation pages are explicitly listed so Claude
    cannot accidentally route them to measurement.
  - The prompt demands JSON-only output; we regex-extract it defensively.
  - Any API or parse failure returns PageType.OTHER with selected=False so the
    pipeline degrades gracefully rather than crashing.
"""

import base64
import json
import re

import anthropic

from models import PageClassification, PageType

# Only select floor plans with at least this confidence for measurement.
CONFIDENCE_THRESHOLD = 0.65

_client = anthropic.Anthropic()

# Model used for classification — sonnet is fast and accurate for categorisation.
_MODEL = "claude-sonnet-4-6"

_CLASSIFICATION_PROMPT = """\
You are an expert construction document reviewer analysing a page from a \
residential building plan set.

TASK
Classify this page into exactly ONE of the categories below.

CATEGORIES
  FLOOR_PLAN       Top-down view showing room layout, walls, doors, windows, \
and dimension lines. Rooms are labelled (e.g. "BEDROOM", "KITCHEN").
  ELEVATION        Side or front view of an exterior or interior wall face. \
Shows the building facade, not a top-down room plan.
  SECTION          A vertical cut through the building revealing internal \
structure — joists, footings, insulation layers.
  ELECTRICAL_PLAN  Shows electrical circuits, outlets, switches, switchboards.
  PLUMBING_PLAN    Shows pipe runs, fixtures, drainage.
  DEMOLITION_PLAN  Shows elements to be removed. Often labelled "DEMO" or uses \
heavy dashed lines for removal items.
  SITE_PLAN        Shows the building footprint on the land, boundaries, north \
arrow, street frontage.
  ROOF_PLAN        Top-down view of the roof structure only.
  RENDER_3D        A 3-D perspective or isometric rendering of the building.
  DETAIL_DRAWING   An enlarged close-up of a specific construction joint or \
element.
  SCHEDULE_LEGEND  Tables, door/window schedules, material legends, title \
pages, notes pages.
  OTHER            Anything not covered above.

CRITICAL RULES
1. A page titled "DEMOLITION PLAN" or showing items marked "REMOVE" is \
DEMOLITION_PLAN — never FLOOR_PLAN.
2. A page showing only one wall face (no rooms visible from above) is \
ELEVATION — never FLOOR_PLAN.
3. Electrical symbols (zigzag lines, outlet circles, switch symbols) with no \
room dimension lines → ELECTRICAL_PLAN.
4. If you see room labels, walls from above, AND dimension lines → FLOOR_PLAN.
5. When in doubt between two categories, choose the one with lower consequence \
of error (i.e. prefer not selecting for measurement).

RESPONSE FORMAT
Respond with ONLY valid JSON — no markdown fences, no explanatory text:
{
  "page_type": "<CATEGORY>",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<one sentence citing the key visual evidence>"
}"""


def classify_page(page_number: int, image_bytes: bytes) -> PageClassification:
    """
    Classify a single rendered PDF page image.

    Args:
        page_number: 0-based index of the page in the PDF.
        image_bytes:  PNG-encoded raster of the page.

    Returns:
        PageClassification with selected=True only for high-confidence floor plans.
    """
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    try:
        response = _client.messages.create(
            model=_MODEL,
            max_tokens=256,
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
                        {"type": "text", "text": _CLASSIFICATION_PROMPT},
                    ],
                }
            ],
        )

        raw = response.content[0].text.strip()
        data = _parse_json(raw)

        page_type = _safe_page_type(data.get("page_type", "OTHER"))
        confidence = float(data.get("confidence", 0.0))
        reasoning = str(data.get("reasoning", ""))

        selected = (
            page_type == PageType.FLOOR_PLAN
            and confidence >= CONFIDENCE_THRESHOLD
        )

        return PageClassification(
            page_number=page_number,
            page_type=page_type,
            confidence=confidence,
            reasoning=reasoning,
            selected=selected,
        )

    except Exception as exc:
        return PageClassification(
            page_number=page_number,
            page_type=PageType.OTHER,
            confidence=0.0,
            reasoning=f"Classification failed: {exc}",
            selected=False,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    """Extract the first JSON object from text, even if surrounded by prose."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in response: {text!r}")
    return json.loads(match.group())


def _safe_page_type(value: str) -> PageType:
    try:
        return PageType(value)
    except ValueError:
        return PageType.OTHER
