"""
IraAI Inference Server

Central inference server for IraAI.

This server:
- Loads models locally on the inference machine.
- Exposes inference APIs.
- Does not contain user-facing prompts.
- Does not act as the Render backend.
"""

from __future__ import annotations

import atexit
import os
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import (
    HOST,
    PORT,
    DEBUG,
    INFERENCE_API_KEY,
    HEALTH_CHECK_ENABLED,
    get_all_models,
    validate_config,
)

from model_manager import get_model_manager

from routes.chat import chat_bp
from routes.vision import vision_bp
from routes.speech import speech_bp
from routes.audio import audio_bp
from routes.image import image_bp
from routes.video import video_bp
from routes.embedding import embedding_bp


# ============================================================
# Configuration
# ============================================================

validate_config()

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    },
)

model_manager = get_model_manager()


# ============================================================
# Blueprint Registration
# ============================================================

app.register_blueprint(chat_bp)
app.register_blueprint(vision_bp)
app.register_blueprint(speech_bp)
app.register_blueprint(audio_bp)
app.register_blueprint(image_bp)
app.register_blueprint(video_bp)
app.register_blueprint(embedding_bp)


# ============================================================
# Authentication
# ============================================================

def _is_authorized() -> bool:
    """Validate the optional inference-server API key."""

    if not INFERENCE_API_KEY:
        return True

    authorization = request.headers.get(
        "Authorization",
        "",
    ).strip()

    expected = f"Bearer {INFERENCE_API_KEY}"

    return authorization == expected


@app.before_request
def authenticate_request():

    public_paths = {
        "/",
        "/health",
    }

    if request.path in public_paths:
        return None

    if _is_authorized():
        return None

    return jsonify(
        {
            "success": False,
            "error": "Unauthorized.",
        }
    ), 401


# ============================================================
# Root
# ============================================================

@app.get("/")
def index():

    return jsonify(
        {
            "success": True,
            "name": "IraAI Inference Server",
            "status": "online",
            "server": "inference",
            "external_ai_api": False,
            "local_model_inference": True,
            "render_backend": False,
            "models": len(get_all_models()),
        }
    )


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    if not HEALTH_CHECK_ENABLED:
        return jsonify(
            {
                "success": True,
                "status": "online",
                "health_check": False,
            }
        )

    loaded_models = model_manager.loaded_models()

    return jsonify(
        {
            "success": True,
            "status": "healthy",
            "server": "inference",
            "model_count": len(get_all_models()),
            "loaded_models": loaded_models,
            "loaded_count": len(loaded_models),
        }
    )


# ============================================================
# Models
# ============================================================

@app.get("/v1/models")
def models():

    configured_models = get_all_models()
    loaded_models = model_manager.loaded_models()

    result = []

    for role, model_name in configured_models.items():
        result.append(
            {
                "role": role,
                "model": model_name,
                "loaded": role in loaded_models,
            }
        )

    return jsonify(
        {
            "success": True,
            "models": result,
        }
    )


@app.get("/v1/models/<role>")
def model_info(role: str):

    configured_models = get_all_models()

    if role not in configured_models:
        return jsonify(
            {
                "success": False,
                "error": f"Unknown model role: {role}",
            }
        ), 404

    return jsonify(
        {
            "success": True,
            "role": role,
            "model": configured_models[role],
            "loaded": model_manager.is_loaded(role),
        }
    )


@app.post("/v1/models/<role>/load")
def load_model(role: str):

    configured_models = get_all_models()

    if role not in configured_models:
        return jsonify(
            {
                "success": False,
                "error": f"Unknown model role: {role}",
            }
        ), 404

    try:
        model_manager.load_model(role)

        return jsonify(
            {
                "success": True,
                "role": role,
                "model": configured_models[role],
                "loaded": True,
            }
        )

    except Exception:
        app.logger.exception(
            "Model loading failed for role: %s",
            role,
        )

        return jsonify(
            {
                "success": False,
                "role": role,
                "error": "Model loading failed.",
            }
        ), 500


@app.post("/v1/models/<role>/unload")
def unload_model(role: str):

    configured_models = get_all_models()

    if role not in configured_models:
        return jsonify(
            {
                "success": False,
                "error": f"Unknown model role: {role}",
            }
        ), 404

    try:
        unloaded = model_manager.unload_model(role)

        return jsonify(
            {
                "success": True,
                "role": role,
                "unloaded": bool(unloaded),
            }
        )

    except Exception:
        app.logger.exception(
            "Model unload failed for role: %s",
            role,
        )

        return jsonify(
            {
                "success": False,
                "role": role,
                "error": "Model unload failed.",
            }
        ), 500


@app.post("/v1/models/unload-all")
def unload_all_models():

    try:
        model_manager.unload_all()

        return jsonify(
            {
                "success": True,
                "message": "All loaded models were unloaded.",
                "loaded_models": model_manager.loaded_models(),
            }
        )

    except Exception:
        app.logger.exception("Failed to unload all models")

        return jsonify(
            {
                "success": False,
                "error": "Failed to unload models.",
            }
        ), 500


# ============================================================
# Server Status
# ============================================================

@app.get("/v1/status")
def server_status():

    loaded_models = model_manager.loaded_models()

    return jsonify(
        {
            "success": True,
            "server": "IraAI Inference Server",
            "status": "online",
            "local_model_inference": True,
            "external_ai_api": False,
            "configured_models": len(get_all_models()),
            "loaded_models": loaded_models,
            "loaded_count": len(loaded_models),
            "pid": os.getpid(),
        }
    )


# ============================================================
# Error Handlers
# ============================================================

@app.errorhandler(404)
def not_found(error: Any):

    return jsonify(
        {
            "success": False,
            "error": "Endpoint not found.",
            "path": request.path,
        }
    ), 404


@app.errorhandler(405)
def method_not_allowed(error: Any):

    return jsonify(
        {
            "success": False,
            "error": "HTTP method not allowed.",
            "method": request.method,
            "path": request.path,
        }
    ), 405


@app.errorhandler(500)
def internal_server_error(error: Any):

    return jsonify(
        {
            "success": False,
            "error": "Internal inference server error.",
        }
    ), 500


# ============================================================
# Shutdown
# ============================================================

def cleanup():

    try:
        model_manager.unload_all()
    except Exception:
        app.logger.exception(
            "Error while cleaning up loaded models"
        )


atexit.register(cleanup)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    print("==============================================")
    print("        IraAI Inference Server")
    print("==============================================")
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Debug: {DEBUG}")
    print(f"Models: {len(get_all_models())}")
    print("External AI API: DISABLED")
    print("Local Model Inference: ENABLED")
    print("Render Model Loading: DISABLED")
    print("==============================================")

    app.run(
        host=HOST,
        port=PORT,
        debug=DEBUG,
    )