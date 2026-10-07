"""
IraAI Inference Server
Audio generation routes.

Models:
- Kokoro-82M       -> Text-to-Speech
- MusicGen Small   -> Music Generation
"""

from __future__ import annotations

import base64
import io
import wave
from typing import Any

from flask import Blueprint, jsonify, request

from model_manager import get_model_manager


audio_bp = Blueprint(
    "audio",
    __name__,
    url_prefix="/v1",
)

model_manager = get_model_manager()


# ============================================================
# Helpers
# ============================================================

def _audio_to_wav_bytes(
    audio: Any,
    sample_rate: int,
) -> bytes:
    """
    Convert generated audio data into WAV bytes.
    """

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "numpy is required for audio encoding."
        ) from exc

    if hasattr(audio, "detach"):
        audio = audio.detach()

    if hasattr(audio, "cpu"):
        audio = audio.cpu()

    if hasattr(audio, "numpy"):
        audio = audio.numpy()

    audio = np.asarray(audio)

    audio = np.squeeze(audio)

    if audio.ndim > 1:
        audio = audio[0]

    audio = np.clip(
        audio,
        -1.0,
        1.0,
    )

    audio_int16 = (
        audio * 32767.0
    ).astype(
        np.int16
    )

    buffer = io.BytesIO()

    with wave.open(
        buffer,
        "wb",
    ) as wav_file:

        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(
            int(sample_rate)
        )

        wav_file.writeframes(
            audio_int16.tobytes()
        )

    return buffer.getvalue()


def _encode_base64(
    data: bytes,
) -> str:

    return base64.b64encode(
        data
    ).decode(
        "utf-8"
    )


# ============================================================
# Text-to-Speech
# ============================================================

def _generate_speech(
    text: str,
    voice: str | None = None,
    speed: float = 1.0,
):

    model_data = model_manager.get_model(
        "text_to_speech"
    )

    model = model_data["model"]

    processor = model_data.get(
        "processor"
    )

    tokenizer = model_data.get(
        "tokenizer"
    )

    sample_rate = int(
        model_data.get(
            "sample_rate",
            24000,
        )
    )

    # --------------------------------------------------------
    # Prefer a model-provided generation method.
    # --------------------------------------------------------

    if hasattr(
        model,
        "generate_speech",
    ):

        kwargs = {
            "text": text,
        }

        if voice:
            kwargs["voice"] = voice

        if speed != 1.0:
            kwargs["speed"] = speed

        audio = model.generate_speech(
            **kwargs
        )

        return audio, sample_rate

    # --------------------------------------------------------
    # Generic callable fallback.
    # --------------------------------------------------------

    if callable(model):

        kwargs = {
            "text": text,
        }

        if voice:
            kwargs["voice"] = voice

        if speed != 1.0:
            kwargs["speed"] = speed

        try:

            audio = model(
                **kwargs
            )

            return audio, sample_rate

        except TypeError:
            pass

    # --------------------------------------------------------
    # Processor/tokenizer based fallback.
    # --------------------------------------------------------

    if processor is not None:

        inputs = processor(
            text,
            return_tensors="pt",
        )

    elif tokenizer is not None:

        inputs = tokenizer(
            text,
            return_tensors="pt",
        )

    else:

        raise RuntimeError(
            "Kokoro model does not expose a supported "
            "text-to-speech interface."
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

    if hasattr(
        model,
        "generate",
    ):

        generated = model.generate(
            **inputs
        )

        return generated, sample_rate

    raise RuntimeError(
        "Kokoro model does not expose a supported "
        "generation method."
    )


@audio_bp.post(
    "/text-to-speech"
)
def text_to_speech():

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

    text = data.get(
        "text"
    )

    if not isinstance(
        text,
        str,
    ) or not text.strip():

        return jsonify(
            {
                "success": False,
                "error": "text is required.",
            }
        ), 400

    text = text.strip()

    voice = data.get(
        "voice"
    )

    if voice is not None:
        voice = str(
            voice
        ).strip()

        if not voice:
            voice = None

    try:

        speed = float(
            data.get(
                "speed",
                1.0,
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        return jsonify(
            {
                "success": False,
                "error": "speed must be a number.",
            }
        ), 400

    if speed <= 0:

        return jsonify(
            {
                "success": False,
                "error": (
                    "speed must be greater than zero."
                ),
            }
        ), 400

    try:

        audio, sample_rate = _generate_speech(
            text=text,
            voice=voice,
            speed=speed,
        )

        wav_bytes = _audio_to_wav_bytes(
            audio,
            sample_rate,
        )

        return jsonify(
            {
                "success": True,
                "role": "text_to_speech",
                "model": "Kokoro-82M",
                "audio_format": "wav",
                "sample_rate": sample_rate,
                "audio": _encode_base64(
                    wav_bytes
                ),
            }
        )

    except Exception as exc:

        return jsonify(
            {
                "success": False,
                "error": (
                    f"Text-to-speech inference failed: {exc}"
                ),
            }
        ), 500


# ============================================================
# MusicGen
# ============================================================

def _generate_music(
    prompt: str,
    duration: float | None = None,
):

    model_data = model_manager.get_model(
        "music"
    )

    model = model_data["model"]

    processor = model_data.get(
        "processor"
    )

    tokenizer = model_data.get(
        "tokenizer"
    )

    sample_rate = int(
        model_data.get(
            "sample_rate",
            32000,
        )
    )

    if processor is not None:

        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )

    elif tokenizer is not None:

        inputs = tokenizer(
            [prompt],
            padding=True,
            return_tensors="pt",
        )

    else:

        raise RuntimeError(
            "MusicGen processor/tokenizer is unavailable."
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

    generate_kwargs = {}

    if duration is not None:
        generate_kwargs[
            "max_new_tokens"
        ] = max(
            1,
            int(
                duration * 50
            ),
        )

    if hasattr(
        model,
        "generate",
    ):

        generated = model.generate(
            **inputs,
            **generate_kwargs,
        )

    else:

        raise RuntimeError(
            "MusicGen model does not support generation."
        )

    return generated, sample_rate


@audio_bp.post(
    "/music"
)
def music():

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

    duration = data.get(
        "duration"
    )

    if duration is not None:

        try:
            duration = float(
                duration
            )

        except (
            TypeError,
            ValueError,
        ):

            return jsonify(
                {
                    "success": False,
                    "error": (
                        "duration must be a number."
                    ),
                }
            ), 400

        if duration <= 0:

            return jsonify(
                {
                    "success": False,
                    "error": (
                        "duration must be greater than zero."
                    ),
                }
            ), 400

    try:

        audio, sample_rate = _generate_music(
            prompt=prompt,
            duration=duration,
        )

        wav_bytes = _audio_to_wav_bytes(
            audio,
            sample_rate,
        )

        return jsonify(
            {
                "success": True,
                "role": "music",
                "model": "MusicGen Small",
                "audio_format": "wav",
                "sample_rate": sample_rate,
                "audio": _encode_base64(
                    wav_bytes
                ),
            }
        )

    except Exception as exc:

        return jsonify(
            {
                "success": False,
                "error": (
                    f"Music generation failed: {exc}"
                ),
            }
        ), 500


# ============================================================
# Status
# ============================================================

@audio_bp.get(
    "/audio/status"
)
def audio_status():

    return jsonify(
        {
            "success": True,
            "models": {
                "text_to_speech": {
                    "name": "Kokoro-82M",
                    "loaded": model_manager.is_loaded(
                        "text_to_speech"
                    ),
                },
                "music": {
                    "name": "MusicGen Small",
                    "loaded": model_manager.is_loaded(
                        "music"
                    ),
                },
            },
        }
    )


__all__ = [
    "audio_bp",
]