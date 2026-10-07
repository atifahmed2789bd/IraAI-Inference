"""
IraAI Inference Server
Speech-to-text inference route.

Model:
Whisper Small
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict

from flask import Blueprint, jsonify, request

from model_manager import get_model_manager


speech_bp = Blueprint(
    "speech",
    __name__,
    url_prefix="/v1",
)

model_manager = get_model_manager()


# ============================================================
# Helpers
# ============================================================

def _decode_audio(audio_data: str) -> bytes:
    """Decode base64 audio data."""

    if not isinstance(audio_data, str):
        raise ValueError(
            "audio must be a base64 string."
        )

    audio_data = audio_data.strip()

    if audio_data.startswith("data:"):
        try:
            audio_data = audio_data.split(
                ",",
                1,
            )[1]
        except IndexError as exc:
            raise ValueError(
                "Invalid audio data URI."
            ) from exc

    try:
        return base64.b64decode(
            audio_data,
            validate=True,
        )
    except Exception as exc:
        raise ValueError(
            "Invalid base64 audio data."
        ) from exc


def _load_audio(audio_bytes: bytes):
    """
    Convert audio bytes into a waveform.

    soundfile is preferred because Whisper expects
    decoded audio samples rather than encoded files.
    """

    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "soundfile is required for speech inference."
        ) from exc

    try:

        audio_buffer = io.BytesIO(
            audio_bytes
        )

        waveform, sample_rate = sf.read(
            audio_buffer,
            dtype="float32",
        )

        return waveform, sample_rate

    except Exception as exc:

        raise ValueError(
            "Unable to decode audio."
        ) from exc


def _resample_audio(
    waveform,
    sample_rate: int,
    target_rate: int = 16000,
):
    """Resample audio to Whisper's expected sample rate."""

    if sample_rate == target_rate:
        return waveform

    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(
            "librosa is required for audio resampling."
        ) from exc

    return librosa.resample(
        waveform,
        orig_sr=sample_rate,
        target_sr=target_rate,
    )


def _prepare_audio(waveform):
    """Convert stereo audio into mono."""

    try:

        if getattr(
            waveform,
            "ndim",
            1,
        ) > 1:

            waveform = waveform.mean(
                axis=1
            )

    except Exception:
        pass

    return waveform


# ============================================================
# Whisper inference
# ============================================================

def _transcribe(
    waveform,
    language: str | None = None,
    task: str = "transcribe",
) -> str:

    model_data = model_manager.get_model(
        "speech_to_text"
    )

    processor = model_data["processor"]
    model = model_data["model"]

    waveform = _prepare_audio(
        waveform
    )

    inputs = processor(
        waveform,
        sampling_rate=16000,
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

    generate_kwargs: Dict[str, Any] = {}

    if language:
        generate_kwargs["language"] = language

    if task in {
        "transcribe",
        "translate",
    }:
        generate_kwargs["task"] = task

    generated_ids = model.generate(
        **inputs,
        **generate_kwargs,
    )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    if not text:
        return ""

    return text[0].strip()


# ============================================================
# POST /v1/speech-to-text
# ============================================================

@speech_bp.post("/speech-to-text")
def speech_to_text():

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

    audio_data = data.get(
        "audio"
    )

    if not audio_data:

        return jsonify(
            {
                "success": False,
                "error": "audio is required.",
            }
        ), 400

    language = data.get(
        "language"
    )

    if language is not None:
        language = str(
            language
        ).strip()

        if not language:
            language = None

    task = data.get(
        "task",
        "transcribe",
    )

    if task not in {
        "transcribe",
        "translate",
    }:

        return jsonify(
            {
                "success": False,
                "error": (
                    "task must be "
                    "'transcribe' or 'translate'."
                ),
            }
        ), 400

    try:

        audio_bytes = _decode_audio(
            audio_data
        )

        waveform, sample_rate = _load_audio(
            audio_bytes
        )

        waveform = _resample_audio(
            waveform,
            sample_rate,
        )

        text = _transcribe(
            waveform=waveform,
            language=language,
            task=task,
        )

        return jsonify(
            {
                "success": True,
                "role": "speech_to_text",
                "model": "Whisper Small",
                "text": text,
                "response": text,
                "language": language,
                "task": task,
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
                    f"Speech inference failed: {exc}"
                ),
            }
        ), 500


# ============================================================
# Status
# ============================================================

@speech_bp.get("/speech-to-text/status")
def speech_status():

    return jsonify(
        {
            "success": True,
            "role": "speech_to_text",
            "model": "Whisper Small",
            "loaded": model_manager.is_loaded(
                "speech_to_text"
            ),
        }
    )


__all__ = [
    "speech_bp",
]