"""
IraAI Inference Server
Vision inference routes.

Model:
Qwen3-VL-8B-Instruct
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request

from config import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE
from model_manager import get_model_manager


vision_bp = Blueprint(
    "vision",
    __name__,
    url_prefix="/v1",
)

model_manager = get_model_manager()


# ============================================================
# Helpers
# ============================================================

def _decode_image(image_data: str):
    """Decode a base64 image into a PIL image."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for vision inference."
        ) from exc

    if not isinstance(image_data, str):
        raise ValueError(
            "image must be a base64 string."
        )

    image_data = image_data.strip()

    if image_data.startswith("data:"):
        try:
            image_data = image_data.split(
                ",",
                1,
            )[1]
        except IndexError as exc:
            raise ValueError(
                "Invalid data URI image."
            ) from exc

    try:
        raw = base64.b64decode(
            image_data,
            validate=True,
        )
    except Exception as exc:
        raise ValueError(
            "Invalid base64 image data."
        ) from exc

    try:
        image = Image.open(
            io.BytesIO(raw)
        )

        image.load()

        return image.convert("RGB")

    except Exception as exc:
        raise ValueError(
            "Unable to decode image."
        ) from exc


def _generate_vision(
    image,
    message: str,
    temperature: float,
    max_new_tokens: int,
) -> str:

    model_data = model_manager.get_model(
        "vision"
    )

    processor = model_data["processor"]
    model = model_data["model"]

    prompt = message.strip()

    if not prompt:
        prompt = "Describe the image."

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    try:

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    except Exception:

        text = prompt

    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
    )

    try:

        device = model.device

        inputs = {
            key: value.to(device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

    except Exception:
        pass

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }

    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    output = model.generate(
        **inputs,
        **generation_kwargs,
    )

    input_ids = inputs.get(
        "input_ids"
    )

    if input_ids is not None:

        input_length = (
            input_ids.shape[-1]
        )

        generated_tokens = output[
            0,
            input_length:
        ]

    else:

        generated_tokens = output[0]

    response = processor.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


# ============================================================
# POST /v1/vision
# ============================================================

@vision_bp.post("/vision")
def vision():

    data = request.get_json(
        silent=True
    )

    if not isinstance(data, dict):
        return jsonify(
            {
                "success": False,
                "error": "Request body must be JSON.",
            }
        ), 400

    image_data = data.get(
        "image"
    )

    if not image_data:
        return jsonify(
            {
                "success": False,
                "error": "image is required.",
            }
        ), 400

    message = data.get(
        "message",
        "Describe the image.",
    )

    if not isinstance(message, str):
        message = "Describe the image."

    temperature = data.get(
        "temperature",
        DEFAULT_TEMPERATURE,
    )

    max_new_tokens = data.get(
        "max_new_tokens",
        DEFAULT_MAX_NEW_TOKENS,
    )

    try:

        temperature = float(
            temperature
        )

        max_new_tokens = int(
            max_new_tokens
        )

    except (
        TypeError,
        ValueError,
    ):

        return jsonify(
            {
                "success": False,
                "error": (
                    "Invalid generation settings."
                ),
            }
        ), 400

    try:

        image = _decode_image(
            image_data
        )

        response = _generate_vision(
            image=image,
            message=message,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        return jsonify(
            {
                "success": True,
                "role": "vision",
                "model": "Qwen3-VL-8B-Instruct",
                "response": response,
            }
        )

    except ValueError as exc:

        return jsonify(
            {
                "success": False,
                "error": str(exc),
            }
        ), 400

    except Exception as exc:

        return jsonify(
            {
                "success": False,
                "error": (
                    f"Vision inference failed: {exc}"
                ),
            }
        ), 500


# ============================================================
# Health information
# ============================================================

@vision_bp.get("/vision/status")
def vision_status():

    return jsonify(
        {
            "success": True,
            "role": "vision",
            "model": "Qwen3-VL-8B-Instruct",
            "loaded": model_manager.is_loaded(
                "vision"
            ),
        }
    )


__all__ = [
    "vision_bp",
]