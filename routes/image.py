"""
IraAI Inference Server
Image generation and refinement routes.

Models:
- Stable Diffusion XL Base 1.0
- Stable Diffusion XL Refiner 1.0
"""

from __future__ import annotations

import base64
import io
from typing import Any, Optional

from flask import Blueprint, jsonify, request

from model_manager import get_model_manager


image_bp = Blueprint(
    "image",
    __name__,
    url_prefix="/v1",
)

model_manager = get_model_manager()


# ============================================================
# Helpers
# ============================================================

def _image_to_base64(image: Any) -> str:
    """Convert a PIL image into PNG base64."""

    if image is None:
        raise ValueError(
            "Generated image is empty."
        )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def _get_int(
    data: dict,
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:

    value = data.get(
        key,
        default,
    )

    try:
        value = int(value)
    except (
        TypeError,
        ValueError,
    ):
        raise ValueError(
            f"{key} must be an integer."
        )

    if value < minimum or value > maximum:
        raise ValueError(
            f"{key} must be between "
            f"{minimum} and {maximum}."
        )

    return value


def _get_float(
    data: dict,
    key: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:

    value = data.get(
        key,
        default,
    )

    try:
        value = float(value)
    except (
        TypeError,
        ValueError,
    ):
        raise ValueError(
            f"{key} must be a number."
        )

    if value < minimum or value > maximum:
        raise ValueError(
            f"{key} must be between "
            f"{minimum} and {maximum}."
        )

    return value


def _get_optional_seed(
    data: dict,
) -> Optional[int]:

    seed = data.get(
        "seed"
    )

    if seed is None:
        return None

    try:
        seed = int(seed)
    except (
        TypeError,
        ValueError,
    ):
        raise ValueError(
            "seed must be an integer."
        )

    return seed


# ============================================================
# Image Generation
# ============================================================

def _generate_image(
    prompt: str,
    negative_prompt: Optional[str],
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: Optional[int],
):

    model_data = model_manager.get_model(
        "image"
    )

    pipeline = model_data["model"]

    kwargs = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
    }

    if negative_prompt:
        kwargs["negative_prompt"] = (
            negative_prompt
        )

    if seed is not None:

        try:
            import torch

            generator = torch.Generator(
                device="cpu"
            ).manual_seed(
                seed
            )

            kwargs["generator"] = generator

        except ImportError:
            pass

    result = pipeline(
        **kwargs
    )

    if not hasattr(
        result,
        "images",
    ):

        raise RuntimeError(
            "Image pipeline did not return images."
        )

    if not result.images:

        raise RuntimeError(
            "Image generation returned no image."
        )

    return result.images[0]


@image_bp.post(
    "/image"
)
def image():

    data = request.get_json(
        silent=True
    )

    if not isinstance(
        data,
        dict,
    ):

        return jsonify(
            {
                "success": False,
                "error": (
                    "Request body must be JSON."
                ),
            }
        ), 400

    prompt = data.get(
        "prompt"
    )

    if not isinstance(
        prompt,
        str,
    ) or not prompt.strip():

        return jsonify(
            {
                "success": False,
                "error": "prompt is required.",
            }
        ), 400

    prompt = prompt.strip()

    negative_prompt = data.get(
        "negative_prompt"
    )

    if negative_prompt is not None:

        if not isinstance(
            negative_prompt,
            str,
        ):

            return jsonify(
                {
                    "success": False,
                    "error": (
                        "negative_prompt must be a string."
                    ),
                }
            ), 400

        negative_prompt = (
            negative_prompt.strip()
        )

        if not negative_prompt:
            negative_prompt = None

    try:

        width = _get_int(
            data,
            "width",
            1024,
            256,
            1536,
        )

        height = _get_int(
            data,
            "height",
            1024,
            256,
            1536,
        )

        steps = _get_int(
            data,
            "steps",
            30,
            1,
            100,
        )

        guidance_scale = _get_float(
            data,
            "guidance_scale",
            7.5,
            0.0,
            30.0,
        )

        seed = _get_optional_seed(
            data
        )

        generated_image = _generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        image_data = _image_to_base64(
            generated_image
        )

        return jsonify(
            {
                "success": True,
                "role": "image",
                "model": (
                    "Stable Diffusion XL Base 1.0"
                ),
                "image_format": "png",
                "width": generated_image.width,
                "height": generated_image.height,
                "image": image_data,
                "seed": seed,
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
                    f"Image generation failed: {exc}"
                ),
            }
        ), 500


# ============================================================
# Image Refinement
# ============================================================

def _decode_image(
    image_data: str,
):

    if not isinstance(
        image_data,
        str,
    ) or not image_data.strip():

        raise ValueError(
            "image must be a base64 string."
        )

    image_data = image_data.strip()

    if image_data.startswith(
        "data:"
    ):

        try:

            image_data = image_data.split(
                ",",
                1,
            )[1]

        except IndexError as exc:

            raise ValueError(
                "Invalid image data URI."
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

        from PIL import Image

        image = Image.open(
            io.BytesIO(raw)
        )

        return image.convert(
            "RGB"
        )

    except Exception as exc:

        raise ValueError(
            "Unable to decode image."
        ) from exc


def _refine_image(
    image,
    prompt: Optional[str],
    negative_prompt: Optional[str],
    strength: float,
    steps: int,
    guidance_scale: float,
):

    model_data = model_manager.get_model(
        "image_refiner"
    )

    pipeline = model_data["model"]

    kwargs = {
        "image": image,
        "strength": strength,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
    }

    if prompt:
        kwargs["prompt"] = prompt

    if negative_prompt:
        kwargs["negative_prompt"] = (
            negative_prompt
        )

    result = pipeline(
        **kwargs
    )

    if not hasattr(
        result,
        "images",
    ):

        raise RuntimeError(
            "Refiner pipeline did not return images."
        )

    if not result.images:

        raise RuntimeError(
            "Image refinement returned no image."
        )

    return result.images[0]


@image_bp.post(
    "/image-refine"
)
def image_refine():

    data = request.get_json(
        silent=True
    )

    if not isinstance(
        data,
        dict,
    ):

        return jsonify(
            {
                "success": False,
                "error": (
                    "Request body must be JSON."
                ),
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

    prompt = data.get(
        "prompt"
    )

    if prompt is not None:

        if not isinstance(
            prompt,
            str,
        ):

            return jsonify(
                {
                    "success": False,
                    "error": (
                        "prompt must be a string."
                    ),
                }
            ), 400

        prompt = prompt.strip()

        if not prompt:
            prompt = None

    negative_prompt = data.get(
        "negative_prompt"
    )

    if negative_prompt is not None:

        if not isinstance(
            negative_prompt,
            str,
        ):

            return jsonify(
                {
                    "success": False,
                    "error": (
                        "negative_prompt must be a string."
                    ),
                }
            ), 400

        negative_prompt = (
            negative_prompt.strip()
        )

        if not negative_prompt:
            negative_prompt = None

    try:

        strength = _get_float(
            data,
            "strength",
            0.35,
            0.0,
            1.0,
        )

        steps = _get_int(
            data,
            "steps",
            30,
            1,
            100,
        )

        guidance_scale = _get_float(
            data,
            "guidance_scale",
            7.5,
            0.0,
            30.0,
        )

        source_image = _decode_image(
            image_data
        )

        refined_image = _refine_image(
            image=source_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            steps=steps,
            guidance_scale=guidance_scale,
        )

        result_data = _image_to_base64(
            refined_image
        )

        return jsonify(
            {
                "success": True,
                "role": "image_refiner",
                "model": (
                    "Stable Diffusion XL Refiner 1.0"
                ),
                "image_format": "png",
                "width": refined_image.width,
                "height": refined_image.height,
                "image": result_data,
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
                    f"Image refinement failed: {exc}"
                ),
            }
        ), 500


# ============================================================
# Status
# ============================================================

@image_bp.get(
    "/image/status"
)
def image_status():

    return jsonify(
        {
            "success": True,
            "models": {
                "image": {
                    "name": (
                        "Stable Diffusion XL Base 1.0"
                    ),
                    "loaded": model_manager.is_loaded(
                        "image"
                    ),
                },
                "image_refiner": {
                    "name": (
                        "Stable Diffusion XL Refiner 1.0"
                    ),
                    "loaded": model_manager.is_loaded(
                        "image_refiner"
                    ),
                },
            },
        }
    )


__all__ = [
    "image_bp",
]