"""
IraAI Inference Server
Central configuration for the remote model inference service.

This server loads and runs the models.
Render/backend only communicates with this server.
"""

import os


# ============================================================
# Server
# ============================================================

HOST = os.getenv("INFERENCE_HOST", "0.0.0.0")

PORT = int(
    os.getenv(
        "INFERENCE_PORT",
        "8000",
    )
)

DEBUG = os.getenv(
    "INFERENCE_DEBUG",
    "false",
).lower() == "true"


# ============================================================
# Hugging Face
# ============================================================

# Optional token for private Hugging Face repositories.
#
# IMPORTANT:
# Keep the token on the inference server only.
# Never commit it to GitHub.
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()


# ============================================================
# Model repositories
# ============================================================

HF_REPOSITORIES = {
    "general": "atifahmed2789/IraAI-Qwen3-8-27B",

    "coder": "atifahmed2789/IraAI-Qwen3-Coder-30B-A3B-Instruct",

    "vision": "atifahmed2789/IraAI-Qwen3-VL-8B-Instruct",

    "reasoning": "atifahmed2789/IraAI-DeepSeek-R1",

    "speech_to_text": "atifahmed2789/IraAI-Whisper-Small",

    "text_to_speech": "atifahmed2789/IraAI-Kokoro-82M",

    "music": "atifahmed2789/IraAI-MusicGen-Small",

    "video": "atifahmed2789/IraAI-Wan2.1-T2V-1.3B",

    "image": "atifahmed2789/IraAI-SDXL-Base-1.0",

    "image_refiner": "atifahmed2789/IraAI-SDXL-Refiner-1.0",

    "embedding": "atifahmed2789/IraAI-BGE-M3",
}


# ============================================================
# Model names
# ============================================================

MODEL_NAMES = {
    "general": "Qwen3-8-27B",

    "coder": "Qwen3-Coder-30B-A3B-Instruct",

    "vision": "Qwen3-VL-8B-Instruct",

    "reasoning": "DeepSeek-R1",

    "speech_to_text": "Whisper Small",

    "text_to_speech": "Kokoro-82M",

    "music": "MusicGen Small",

    "video": "Wan2.1 T2V 1.3B",

    "image": "Stable Diffusion XL Base 1.0",

    "image_refiner": "Stable Diffusion XL Refiner 1.0",

    "embedding": "BGE-M3",
}


# ============================================================
# Runtime settings
# ============================================================

DEVICE = os.getenv(
    "INFERENCE_DEVICE",
    "auto",
).strip().lower()


# Maximum number of models allowed to remain loaded.
#
# 0 means no artificial limit.
# The actual number depends on available RAM/VRAM.
MAX_LOADED_MODELS = int(
    os.getenv(
        "MAX_LOADED_MODELS",
        "0",
    )
)


# ============================================================
# Generation settings
# ============================================================

DEFAULT_TEMPERATURE = float(
    os.getenv(
        "DEFAULT_TEMPERATURE",
        "0.7",
    )
)

DEFAULT_MAX_NEW_TOKENS = int(
    os.getenv(
        "DEFAULT_MAX_NEW_TOKENS",
        "2048",
    )
)


# ============================================================
# Request limits
# ============================================================

REQUEST_TIMEOUT = int(
    os.getenv(
        "REQUEST_TIMEOUT",
        "600",
    )
)

MAX_INPUT_LENGTH = int(
    os.getenv(
        "MAX_INPUT_LENGTH",
        "32768",
    )
)


# ============================================================
# Health
# ============================================================

HEALTH_CHECK_ENABLED = os.getenv(
    "HEALTH_CHECK_ENABLED",
    "true",
).lower() == "true"


# ============================================================
# Security
# ============================================================

# Optional shared secret between Render backend and
# this inference server.
#
# Keep this empty during initial local development.
#
# When enabled, Render must send:
# Authorization: Bearer <INFERENCE_API_KEY>
INFERENCE_API_KEY = os.getenv(
    "INFERENCE_API_KEY",
    "",
).strip()


# ============================================================
# Helpers
# ============================================================

def get_repository(role: str) -> str:
    """Return the Hugging Face repository for a model role."""

    if role not in HF_REPOSITORIES:
        raise ValueError(
            f"Unknown model role: {role}"
        )

    return HF_REPOSITORIES[role]


def get_model_name(role: str) -> str:
    """Return the display name for a model role."""

    if role not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model role: {role}"
        )

    return MODEL_NAMES[role]


def get_all_models() -> dict:
    """Return a copy of all configured models."""

    return {
        role: {
            "name": MODEL_NAMES[role],
            "repository": HF_REPOSITORIES[role],
        }
        for role in HF_REPOSITORIES
    }


def is_private_repository() -> bool:
    """Return whether a Hugging Face token is configured."""

    return bool(HF_TOKEN)


# ============================================================
# Validation
# ============================================================

def validate_config() -> None:
    """Validate the inference server configuration."""

    if not HF_REPOSITORIES:
        raise RuntimeError(
            "No Hugging Face repositories configured."
        )

    if set(HF_REPOSITORIES) != set(MODEL_NAMES):
        raise RuntimeError(
            "HF_REPOSITORIES and MODEL_NAMES must contain "
            "the same model roles."
        )

    if PORT < 1 or PORT > 65535:
        raise RuntimeError(
            f"Invalid inference server port: {PORT}"
        )

    if DEFAULT_MAX_NEW_TOKENS < 1:
        raise RuntimeError(
            "DEFAULT_MAX_NEW_TOKENS must be greater than 0."
        )

    if REQUEST_TIMEOUT < 1:
        raise RuntimeError(
            "REQUEST_TIMEOUT must be greater than 0."
        )

    if MAX_INPUT_LENGTH < 1:
        raise RuntimeError(
            "MAX_INPUT_LENGTH must be greater than 0."
        )


validate_config()