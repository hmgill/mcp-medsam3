"""
server.py — medsam3-mcp
========================
FastMCP server exposing MedSAM3 medical image segmentation as MCP tools,
deployed via Prefect Horizon.

Image preprocessing runs locally (decode + resize + validate); GPU inference
is dispatched to a Modal serverless endpoint.

Required environment variables:
    MODAL_ENDPOINT_URL  Full Modal endpoint base URL,
                        e.g. https://<workspace>--medsam3-inference-medsam3-api.modal.run

Tools:
    segment_medical(image_b64, image_id, prompt, threshold)  → masks + scores + boxes
    health()                                                  → liveness check
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os

import requests
from fastmcp import FastMCP
from PIL import Image

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

MODAL_ENDPOINT_URL = os.environ.get("MODAL_ENDPOINT_URL", "").rstrip("/")
MAX_IMAGE_SIZE = 1024  # resize longest edge to this before dispatch

if not MODAL_ENDPOINT_URL:
    logger.warning("MODAL_ENDPOINT_URL is not set — inference calls will fail.")


# ---------------------------------------------------------------------------
# Modal client
# ---------------------------------------------------------------------------

def _modal_dispatch(image_id: str, image_b64: str, prompt: str, threshold: float) -> dict:
    if not MODAL_ENDPOINT_URL:
        raise RuntimeError("MODAL_ENDPOINT_URL is not set.")

    url = f"{MODAL_ENDPOINT_URL}"
    logger.info(f"[{image_id}] Dispatching to Modal: {url}")

    resp = requests.post(
        url,
        json={
            "image_b64": image_b64,
            "prompt": prompt,
            "threshold": threshold,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Preprocessing — runs locally, no GPU needed
# ---------------------------------------------------------------------------

def _preprocess_image(image_b64: str, image_id: str, max_size: int = MAX_IMAGE_SIZE) -> str:
    """
    Decode, validate, resize, and re-encode as JPEG.
    Keeps the Modal payload small and ensures consistent input format.
    """
    try:
        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError(f"[{image_id}] Could not decode image: {e}")

    w, h = img.size
    if w < 64 or h < 64:
        raise ValueError(
            f"[{image_id}] Image too small ({w}x{h}). "
            "Expected a medical image of reasonable resolution."
        )

    # resize longest edge
    scale = max_size / max(w, h)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.info(f"[{image_id}] Resized: {w}x{h} → {new_w}x{new_h}")
    else:
        logger.info(f"[{image_id}] No resize needed: {w}x{h}")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# FastMCP app
# ---------------------------------------------------------------------------

mcp = FastMCP("medsam3-mcp")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def segment_medical(
    image_b64: str,
    image_id: str,
    prompt: str,
    threshold: float = 0.5,
) -> str:
    """
    Segment a medical image using MedSAM3.

    Uses text-guided segmentation to identify and delineate anatomical
    structures or lesions. Supports a wide range of medical modalities
    including CT, MRI, X-ray, ultrasound, histopathology, and OCT.

    MedSAM3 accepts natural medical concept prompts such as:
        "lung", "right lower lobe consolidation", "skin lesion",
        "optic disc", "liver", "pleural effusion"

    This model is for research and educational purposes only. Outputs
    should not be used for clinical diagnosis or treatment decisions.

    Args:
        image_b64:   Base64-encoded medical image (JPEG or PNG).
        image_id:    Identifier for this image (used for logging).
        prompt:      Medical concept to segment (e.g. "lung nodule").
        threshold:   Confidence threshold for predictions (default 0.5).
                     Lower values return more candidates; higher values
                     return only high-confidence masks.

    Returns:
        JSON with predictions (mask, score, box) and metadata.
    """
    from datetime import datetime, timezone

    try:
        clean_b64 = _preprocess_image(image_b64, image_id)
        output = _modal_dispatch(image_id, clean_b64, prompt, threshold)

        predictions = output.get("predictions", [])
        logger.info(f"[{image_id}] Got {len(predictions)} prediction(s) for prompt='{prompt}'")

        payload = json.dumps({
            "success":     True,
            "image_id":    image_id,
            "prompt":      prompt,
            "threshold":   threshold,
            "n_masks":     len(predictions),
            "predictions": [
                {"score": p["score"], "box": p["box"]}  # omit raw mask arrays from MCP response
                for p in predictions
            ],
            "created_at":  datetime.now(timezone.utc).isoformat(),
            "disclaimer":  (
                "For research and educational use only. "
                "Not for clinical diagnosis or treatment decisions."
            ),
        })

        return payload

    except ValueError as e:
        return json.dumps({"success": False, "reason": str(e), "image_id": image_id})
    except Exception as e:
        logger.error(f"segment_medical failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool()
async def health() -> str:
    """Liveness probe. Reports Modal endpoint configuration status."""
    return json.dumps({
        "status":  "ok",
        "service": "medsam3-mcp",
        "modal": {
            "endpoint_url": MODAL_ENDPOINT_URL or "(not set)",
            "configured":   bool(MODAL_ENDPOINT_URL),
        },
    })


if __name__ == "__main__":
    mcp.run(stateless_http=True, json_response=True)
