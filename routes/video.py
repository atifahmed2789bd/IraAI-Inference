"""
IraAI Inference Server
Video generation route.

Model:
Wan2.1 T2V 1.3B
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from typing import Any, Optional

from flask import Blueprint, jsonify, request

from model_manager import get_model_manager


video_bp = Blueprint(
    "video",
    __name__,
    url_prefix="/v1",
)

model_manager = get_model_manager()


# ============================================================
# Helpers
# ============================================================

def _read_video_bytes(
    video_path: str,
) -> bytes:

    with open(
        video_path,
        "rb",
    ) as video_file:

        return video_file.read()


def _encode_base64(
    data: bytes,
) -> str:

    return base64.b64encode(
        data
    ).decode(
        "utf-8"
    )


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


def _write_video(
    frames: Any,
    fps: int,
) -> str:

    try:

        import imageio.v2 as imageio

    except ImportError as exc:

        raise RuntimeError(
            "imageio is required for video encoding."
        ) from exc

    if frames is None:

        raise RuntimeError(
            "Video generation returned no frames."
        )

    if hasattr(
        frames,
        "detach",
    ):

        frames = frames.detach()

    if hasattr(
        frames,
        "cpu",
    ):

        frames = frames.cpu()

    if hasattr(
        frames,
        "numpy",
    ):

        frames = frames.numpy()

    frames = list(
        frames
    )

    if not frames:

        raise RuntimeError(
            "Video generation returned no frames."
        )

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
    )

    temp_path = temp_file.name

    temp_file.close()

    try:

        with imageio.get_writer(
            temp_path,
            fps=fps,
            codec="libx264",
        ) as writer:

            for frame in frames:

                writer.append_data(
                    frame
                )

    except Exception:

        if os.path.exists(
            temp_path
        ):

            os.remove(
                temp_path
            )

        raise

    return temp_path


# ============================================================
# Wan2.1 inference
# ============================================================

def _generate_video(
    prompt: str,
    negative_prompt: Optional[str],
    width: int,
    height: int,
    frames: int,
    steps: int,
    guidance_scale: float,
    fps: int,
):

    model_data = model_manager.get_model(
        "video"
    )

    pipeline = model_data["model"]

    kwargs = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_frames": frames,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
    }

    if negative_prompt:

        kwargs[
            "negative_prompt"
        ] = negative_prompt

    result = pipeline(
        **kwargs
    )

    # --------------------------------------------------------
    # Diffusers pipeline output
    # --------------------------------------------------------

    generated_frames = None

    if hasattr(
        result,
        "frames",
    ):

        generated_frames = result.frames

    elif isinstance(
        result,
        dict,
    ):

        generated_frames = result.get(
            "frames"
        )

    if generated_frames is None:

        raise RuntimeError(
            "Video pipeline did not return frames."
        )

    # Some diffusers pipelines return:
    # [frames]
    if (
        isinstance(
            generated_frames,
            list,
        )
        and generated_frames
        and isinstance(
            generated_frames[0],
            list,
        )
    ):

        generated_frames = (
            generated_frames[0]
        )

    video_path = _write_video(
        generated_frames,
        fps=fps,
    )

    return video_path


# ============================================================
# POST /v1/video
# ============================================================

@video_bp.post(
    "/video"
)
def video():

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
            832,
            256,
            1280,
        )

        height = _get_int(
            data,
            "height",
            480,
            256,
            1280,
        )

        frames = _get_int(
            data,
            "frames",
            81,
            1,
            161,
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
            5.0,
            0.0,
            20.0,
        )

        fps = _get_int(
            data,
            "fps",
            16,
            1,
            60,
        )

        video_path = _generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            frames=frames,
            steps=steps,
            guidance_scale=guidance_scale,
            fps=fps,
        )

        try:

            video_bytes = _read_video_bytes(
                video_path
            )

        finally:

            if os.path.exists(
                video_path
            ):

                os.remove(
                    video_path
                )

        return jsonify(
            {
                "success": True,
                "role": "video",
                "model": (
                    "Wan2.1 T2V 1.3B"
                ),
                "video_format": "mp4",
                "fps": fps,
                "frames": frames,
                "width": width,
                "height": height,
                "video": _encode_base64(
                    video_bytes
                ),
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
                    f"Video generation failed: {exc}"
                ),
            }
        ), 500


# ============================================================
# Status
# ============================================================

@video_bp.get(
    "/video/status"
)
def video_status():

    return jsonify(
        {
            "success": True,
            "role": "video",
            "model": "Wan2.1 T2V 1.3B",
            "loaded": model_manager.is_loaded(
                "video"
            ),
        }
    )


__all__ = [
    "video_bp",
]