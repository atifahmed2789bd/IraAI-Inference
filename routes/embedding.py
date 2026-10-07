"""
IraAI Inference Server
Embedding generation route.

Model:
BGE-M3
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from model_manager import get_model_manager


embedding_bp = Blueprint(
    "embedding",
    __name__,
    url_prefix="/v1",
)

model_manager = get_model_manager()


# ============================================================
# Helpers
# ============================================================

def _normalize_embeddings(
    embeddings,
):
    """
    Normalize embedding vectors when requested.
    """

    import numpy as np

    array = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if array.ndim == 1:
        array = array.reshape(
            1,
            -1,
        )

    norms = np.linalg.norm(
        array,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-12,
    )

    return array / norms


def _convert_to_lists(
    embeddings,
):
    """
    Convert numpy/tensor embeddings into
    JSON-compatible Python lists.
    """

    if hasattr(
        embeddings,
        "detach",
    ):
        embeddings = embeddings.detach()

    if hasattr(
        embeddings,
        "cpu",
    ):
        embeddings = embeddings.cpu()

    if hasattr(
        embeddings,
        "numpy",
    ):
        embeddings = embeddings.numpy()

    return embeddings.tolist()


# ============================================================
# BGE-M3 inference
# ============================================================

def _generate_embeddings(
    texts,
    normalize: bool = True,
):

    model_data = model_manager.get_model(
        "embedding"
    )

    model = model_data["model"]

    # --------------------------------------------------------
    # SentenceTransformer interface
    # --------------------------------------------------------

    if hasattr(
        model,
        "encode",
    ):

        embeddings = model.encode(
            texts,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings

    # --------------------------------------------------------
    # Generic embedding interface
    # --------------------------------------------------------

    if hasattr(
        model,
        "embed",
    ):

        embeddings = model.embed(
            texts
        )

        if normalize:
            embeddings = _normalize_embeddings(
                embeddings
            )

        return embeddings

    raise RuntimeError(
        "BGE-M3 model does not expose a supported "
        "embedding interface."
    )


# ============================================================
# POST /v1/embedding
# ============================================================

@embedding_bp.post(
    "/embedding"
)
def embedding():

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

    texts = data.get(
        "texts"
    )

    # --------------------------------------------------------
    # Accept a single text as a convenience.
    # --------------------------------------------------------

    if texts is None:

        text = data.get(
            "text"
        )

        if isinstance(
            text,
            str,
        ) and text.strip():

            texts = [
                text
            ]

    if isinstance(
        texts,
        str,
    ):

        texts = [
            texts
        ]

    if not isinstance(
        texts,
        list,
    ) or not texts:

        return jsonify(
            {
                "success": False,
                "error": (
                    "texts must be a non-empty list "
                    "or text must be provided."
                ),
            }
        ), 400

    cleaned_texts = []

    for index, text in enumerate(
        texts
    ):

        if not isinstance(
            text,
            str,
        ):

            return jsonify(
                {
                    "success": False,
                    "error": (
                        f"texts[{index}] must be a string."
                    ),
                }
            ), 400

        text = text.strip()

        if not text:

            return jsonify(
                {
                    "success": False,
                    "error": (
                        f"texts[{index}] cannot be empty."
                    ),
                }
            ), 400

        cleaned_texts.append(
            text
        )

    normalize = data.get(
        "normalize",
        True,
    )

    if not isinstance(
        normalize,
        bool,
    ):

        return jsonify(
            {
                "success": False,
                "error": (
                    "normalize must be true or false."
                ),
            }
        ), 400

    try:

        embeddings = _generate_embeddings(
            cleaned_texts,
            normalize=normalize,
        )

        embedding_lists = _convert_to_lists(
            embeddings
        )

        dimension = 0

        if embedding_lists:

            dimension = len(
                embedding_lists[0]
            )

        return jsonify(
            {
                "success": True,
                "role": "embedding",
                "model": "BGE-M3",
                "count": len(
                    cleaned_texts
                ),
                "dimension": dimension,
                "normalize": normalize,
                "embeddings": embedding_lists,
            }
        )

    except Exception as exc:

        return jsonify(
            {
                "success": False,
                "error": (
                    f"Embedding generation failed: {exc}"
                ),
            }
        ), 500


# ============================================================
# Similarity
# ============================================================

@embedding_bp.post(
    "/embedding/similarity"
)
def embedding_similarity():

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

    query = data.get(
        "query"
    )

    documents = data.get(
        "documents"
    )

    if not isinstance(
        query,
        str,
    ) or not query.strip():

        return jsonify(
            {
                "success": False,
                "error": "query is required.",
            }
        ), 400

    if not isinstance(
        documents,
        list,
    ) or not documents:

        return jsonify(
            {
                "success": False,
                "error": (
                    "documents must be a non-empty list."
                ),
            }
        ), 400

    for index, document in enumerate(
        documents
    ):

        if not isinstance(
            document,
            str,
        ):

            return jsonify(
                {
                    "success": False,
                    "error": (
                        f"documents[{index}] must be a string."
                    ),
                }
            ), 400

    try:

        texts = [
            query.strip()
        ]

        texts.extend(
            document.strip()
            for document in documents
        )

        embeddings = _generate_embeddings(
            texts,
            normalize=True,
        )

        embeddings = _normalize_embeddings(
            embeddings
        )

        query_vector = embeddings[0]

        document_vectors = embeddings[1:]

        similarities = (
            document_vectors
            @ query_vector
        )

        results = []

        for index, score in enumerate(
            similarities
        ):

            results.append(
                {
                    "index": index,
                    "document": documents[index],
                    "score": float(score),
                }
            )

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return jsonify(
            {
                "success": True,
                "model": "BGE-M3",
                "query": query,
                "results": results,
            }
        )

    except Exception as exc:

        return jsonify(
            {
                "success": False,
                "error": (
                    f"Similarity calculation failed: {exc}"
                ),
            }
        ), 500


# ============================================================
# Status
# ============================================================

@embedding_bp.get(
    "/embedding/status"
)
def embedding_status():

    return jsonify(
        {
            "success": True,
            "role": "embedding",
            "model": "BGE-M3",
            "loaded": model_manager.is_loaded(
                "embedding"
            ),
        }
    )


__all__ = [
    "embedding_bp",
]