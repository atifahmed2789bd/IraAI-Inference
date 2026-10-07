"""
IraAI Inference Server
Chat inference routes.

Handles:
- General Qwen model
- Coding Qwen model
- DeepSeek reasoning model
"""

from __future__ import annotations

from typing import Any, Dict

from flask import Blueprint, jsonify, request

from config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
)
from model_manager import get_model_manager


chat_bp = Blueprint(
    "chat",
    __name__,
    url_prefix="/v1",
)

model_manager = get_model_manager()


# ============================================================
# Helpers
# ============================================================

def _get_device(model: Any) -> str:
    """Get the device used by a loaded model."""

    try:
        return str(model.device)
    except Exception:
        return "auto"


def _generate_text(
    role: str,
    message: str,
    context: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """
    Generate text using one of the three text models.

    No system prompt is created here.
    Any application-level instructions must come from
    the backend AnswerBuilder/context.
    """

    model_data = model_manager.get_model(role)

    tokenizer = model_data["tokenizer"]
    model = model_data["model"]

    user_message = message.strip()

    if context.strip():
        user_message = (
            f"{context.strip()}\n\n"
            f"{user_message}"
        )

    messages = [
        {
            "role": "user",
            "content": user_message,
        }
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = user_message

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    try:
        device = model.device
        inputs = {
            key: value.to(device)
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

    input_length = inputs["input_ids"].shape[-1]

    generated_tokens = output[
        0,
        input_length:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


# ============================================================
# Generic chat generation
# ============================================================

def generate_chat(
    role: str,
    message: str,
    context: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Generate a response from a selected text model."""

    if not isinstance(message, str):
        raise ValueError(
            "message must be a string."
        )

    message = message.strip()

    if not message:
        raise ValueError(
            "message cannot be empty."
        )

    if role not in {
        "general",
        "coder",
        "reasoning",
    }:
        raise ValueError(
            f"Unsupported chat model role: {role}"
        )

    response = _generate_text(
        role=role,
        message=message,
        context=context,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    return {
        "success": True,
        "role": role,
        "model": model_manager.list_models()[role]["name"],
        "response": response,
    }


# ============================================================
# POST /v1/chat
# ============================================================

@chat_bp.post("/chat")
def chat():

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

    message = data.get("message")

    if not isinstance(message, str):
        return jsonify(
            {
                "success": False,
                "error": "message must be a string.",
            }
        ), 400

    role = data.get(
        "role",
        "general",
    )

    if not isinstance(role, str):
        role = "general"

    context = data.get(
        "context",
        "",
    )

    if not isinstance(context, str):
        context = ""

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
                    "temperature must be a number "
                    "and max_new_tokens must be an integer."
                ),
            }
        ), 400

    try:

        result = generate_chat(
            role=role,
            message=message,
            context=context,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        return jsonify(result)

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
                    f"Chat inference failed: {exc}"
                ),
            }
        ), 500


# ============================================================
# GET /v1/chat/models
# ============================================================

@chat_bp.get("/chat/models")
def chat_models():

    models = model_manager.list_models()

    return jsonify(
        {
            "success": True,
            "models": {
                role: models[role]
                for role in (
                    "general",
                    "coder",
                    "reasoning",
                )
            },
            "loaded": [
                role
                for role in model_manager.loaded_models()
                if role in {
                    "general",
                    "coder",
                    "reasoning",
                }
            ],
        }
    )


__all__ = [
    "chat_bp",
    "generate_chat",
]