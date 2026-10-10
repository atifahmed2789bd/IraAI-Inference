"""
IraAI Inference Server
Model Manager

Responsible for:
- Loading models from Hugging Face
- Keeping loaded models in memory
- Managing model cache
- Unloading models when necessary
- Providing a common interface for inference

No prompts or assistant personality instructions belong here.
"""

from __future__ import annotations

import gc
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

from config import (
    HF_REPOSITORIES,
    HF_TOKEN,
    MAX_LOADED_MODELS,
    get_model_name,
    get_repository,
)


class ModelManager:
    """
    Central manager for all IraAI models.

    Models are loaded lazily and cached using an LRU policy.
    """

    def __init__(self) -> None:
        self._models: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.RLock()

    # ========================================================
    # Model information
    # ========================================================

    def list_models(self) -> Dict[str, Dict[str, str]]:
        """Return information about all configured models."""
        return {
            role: {
                "name": get_model_name(role),
                "repository": repository,
            }
            for role, repository in HF_REPOSITORIES.items()
        }

    def is_loaded(self, role: str) -> bool:
        """Check whether a model is currently loaded."""
        with self._lock:
            return role in self._models

    def loaded_models(self) -> list[str]:
        """Return currently loaded model roles."""
        with self._lock:
            return list(self._models.keys())

    def list_loaded_models(self) -> list[str]:
        """
        Compatibility alias used by server.py.
        """
        return self.loaded_models()

    # ========================================================
    # Model loading
    # ========================================================

    def load_model(self, role: str) -> Any:
        """Load a model lazily and return its model data."""

        with self._lock:

            if role not in HF_REPOSITORIES:
                raise ValueError(
                    f"Unknown model role: {role}"
                )

            if role in self._models:
                self._models.move_to_end(role)
                return self._models[role]

            model_data = self._create_model(role)

            self._models[role] = model_data
            self._models.move_to_end(role)

            self._enforce_cache_limit()

            return model_data

    # ========================================================
    # Model creation
    # ========================================================

    def _create_model(self, role: str) -> Any:
        """Create a model according to its configured role."""

        repository = get_repository(role)

        if role in {
            "general",
            "coder",
            "reasoning",
        }:
            return self._load_text_model(
                role,
                repository,
            )

        if role == "vision":
            return self._load_vision_model(
                role,
                repository,
            )

        if role == "speech_to_text":
            return self._load_speech_model(
                role,
                repository,
            )

        if role == "text_to_speech":
            return self._load_tts_model(
                role,
                repository,
            )

        if role == "music":
            return self._load_music_model(
                role,
                repository,
            )

        if role == "video":
            return self._load_video_model(
                role,
                repository,
            )

        if role == "image":
            return self._load_image_model(
                role,
                repository,
            )

        if role == "image_refiner":
            return self._load_image_refiner_model(
                role,
                repository,
            )

        if role == "embedding":
            return self._load_embedding_model(
                role,
                repository,
            )

        raise ValueError(
            f"No loader registered for model role: {role}"
        )

    # ========================================================
    # Hugging Face authentication
    # ========================================================

    @staticmethod
    def _token_kwargs() -> Dict[str, Any]:
        """
        Return Hugging Face authentication arguments.

        Public repositories work without HF_TOKEN.
        Private repositories can use HF_TOKEN.
        """

        if HF_TOKEN:
            return {
                "token": HF_TOKEN
            }

        return {}

    # ========================================================
    # Text models
    # ========================================================

    def _load_text_model(
        self,
        role: str,
        repository: str,
    ) -> Any:

        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for text models."
            ) from exc

        token_kwargs = self._token_kwargs()

        tokenizer = AutoTokenizer.from_pretrained(
            repository,
            **token_kwargs,
        )

        model = AutoModelForCausalLM.from_pretrained(
            repository,
            device_map="auto",
            torch_dtype="auto",
            **token_kwargs,
        )

        return {
            "type": "text",
            "role": role,
            "repository": repository,
            "tokenizer": tokenizer,
            "model": model,
        }

    # ========================================================
    # Vision
    # ========================================================

    def _load_vision_model(
        self,
        role: str,
        repository: str,
    ) -> Any:

        try:
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for vision models."
            ) from exc

        token_kwargs = self._token_kwargs()

        processor = AutoProcessor.from_pretrained(
            repository,
            **token_kwargs,
        )

        model = AutoModelForImageTextToText.from_pretrained(
            repository,
            device_map="auto",
            torch_dtype="auto",
            **token_kwargs,
        )

        return {
            "type": "vision",
            "role": role,
            "repository": repository,
            "processor": processor,
            "model": model,
        }

    # ========================================================
    # Speech-to-text
    # ========================================================

    def _load_speech_model(
        self,
        role: str,
        repository: str,
    ) -> Any:

        try:
            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for speech models."
            ) from exc

        token_kwargs = self._token_kwargs()

        processor = AutoProcessor.from_pretrained(
            repository,
            **token_kwargs,
        )

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            repository,
            device_map="auto",
            torch_dtype="auto",
            **token_kwargs,
        )

        return {
            "type": "speech_to_text",
            "role": role,
            "repository": repository,
            "processor": processor,
            "model": model,
        }

    # ========================================================
    # Text-to-speech
    # ========================================================

    def _load_tts_model(
        self,
        role: str,
        repository: str,
    ) -> Any:

        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "Required TTS dependencies are not installed."
            ) from exc

        model = AutoModel.from_pretrained(
            repository,
            **self._token_kwargs(),
        )

        return {
            "type": "text_to_speech",
            "role": role,
            "repository": repository,
            "model": model,
        }

    # ========================================================
    # Music
    # ========================================================

    def _load_music_model(
        self,
        role: str,
        repository: str,
    ) -> Any:

        try:
            from transformers import (
                AutoProcessor,
                MusicgenForConditionalGeneration,
            )
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for MusicGen."
            ) from exc

        token_kwargs = self._token_kwargs()

        processor = AutoProcessor.from_pretrained(
            repository,
            **token_kwargs,
        )

        model = MusicgenForConditionalGeneration.from_pretrained(
            repository,
            device_map="auto",
            torch_dtype="auto",
            **token_kwargs,
        )

        return {
            "type": "music",
            "role": role,
            "repository": repository,
            "processor": processor,
            "model": model,
        }

    # ========================================================
    # Video
    # ========================================================

    def _load_video_model(
        self,
        role: str,
        repository: str,
    ) -> Any:

        try:
            from diffusers import DiffusionPipeline
        except ImportError as exc:
            raise RuntimeError(
                "diffusers is required for video generation."
            ) from exc

        pipeline = DiffusionPipeline.from_pretrained(
            repository,
            torch_dtype="auto",
            **self._token_kwargs(),
        )

        return {
            "type": "video",
            "role": role,
            "repository": repository,

            # Keep both names for route compatibility.
            "model": pipeline,
            "pipeline": pipeline,
        }

    # ========================================================
    # Image generation
    # ========================================================

    def _load_image_model(
        self,
        role: str,
        repository: str,
    ) -> Any:

        try:
            from diffusers import StableDiffusionXLPipeline
        except ImportError as exc:
            raise RuntimeError(
                "diffusers is required for SDXL."
            ) from exc

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            repository,
            torch_dtype="auto",
            **self._token_kwargs(),
        )

        return {
            "type": "image",
            "role": role,
            "repository": repository,

            # Keep both names for route compatibility.
            "model": pipeline,
            "pipeline": pipeline,
        }

    # ========================================================
    # Image refinement
    # ========================================================

    def _load_image_refiner_model(
        self,
        role: str,
        repository: str,
    ) -> Any:

        try:
            from diffusers import (
                StableDiffusionXLImg2ImgPipeline
            )
        except ImportError as exc:
            raise RuntimeError(
                "diffusers is required for SDXL Refiner."
            ) from exc

        pipeline = (
            StableDiffusionXLImg2ImgPipeline.from_pretrained(
                repository,
                torch_dtype="auto",
                **self._token_kwargs(),
            )
        )

        return {
            "type": "image_refiner",
            "role": role,
            "repository": repository,

            # Keep both names for route compatibility.
            "model": pipeline,
            "pipeline": pipeline,
        }

    # ========================================================
    # Embedding
    # ========================================================

    def _load_embedding_model(
        self,
        role: str,
        repository: str,
    ) -> Any:

        try:
            from sentence_transformers import (
                SentenceTransformer
            )
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for embeddings."
            ) from exc

        model = SentenceTransformer(
            repository,
            **self._token_kwargs(),
        )

        return {
            "type": "embedding",
            "role": role,
            "repository": repository,
            "model": model,
        }

    # ========================================================
    # Cache management
    # ========================================================

    def _enforce_cache_limit(self) -> None:
        """Unload least recently used models."""

        if MAX_LOADED_MODELS <= 0:
            return

        while len(self._models) > MAX_LOADED_MODELS:

            role, model_data = self._models.popitem(
                last=False
            )

            self._release_model(model_data)

    def unload_model(
        self,
        role: str,
    ) -> bool:
        """Unload one model."""

        with self._lock:

            if role not in self._models:
                return False

            model_data = self._models.pop(role)

            self._release_model(model_data)

            return True

    def unload_all(self) -> None:
        """Unload all currently loaded models."""

        with self._lock:

            models = list(
                self._models.values()
            )

            self._models.clear()

            for model_data in models:
                self._release_model(model_data)

    # ========================================================
    # Memory cleanup
    # ========================================================

    def _release_model(
        self,
        model_data: Any,
    ) -> None:
        """
        Release model references and clear available memory.

        Some pipelines are stored under both "model" and
        "pipeline", so duplicate objects are handled safely.
        """

        try:

            seen: set[int] = set()

            if isinstance(model_data, dict):
                values = model_data.values()
            else:
                values = [model_data]

            for value in values:

                if id(value) in seen:
                    continue

                seen.add(id(value))

                if hasattr(value, "to"):

                    try:
                        value.to("cpu")
                    except Exception:
                        pass

            del model_data

        except Exception:
            pass

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception:
            pass

    # ========================================================
    # Access
    # ========================================================

    def get_loaded_model(
        self,
        role: str,
    ) -> Optional[Any]:
        """Return a loaded model without loading it."""

        with self._lock:

            model_data = self._models.get(role)

            if model_data is not None:
                self._models.move_to_end(role)

            return model_data

    def get_model(
        self,
        role: str,
    ) -> Any:
        """Return a model, loading it when necessary."""

        return self.load_model(role)


# ============================================================
# Global model manager
# ============================================================

model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Return the global ModelManager instance."""

    return model_manager