"""
IraAI Inference Server
Shared utility helpers.
"""

from __future__ import annotations

import base64
import binascii
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional


# ============================================================
# JSON Helpers
# ============================================================

def is_dict(
    value: Any,
) -> bool:
    """Return True when value is a dictionary."""

    return isinstance(
        value,
        dict,
    )


def is_non_empty_string(
    value: Any,
) -> bool:
    """Return True for a non-empty string."""

    return (
        isinstance(value, str)
        and bool(value.strip())
    )


def clean_string(
    value: Any,
    default: str = "",
) -> str:
    """Safely convert a value into a trimmed string."""

    if value is None:
        return default

    if not isinstance(
        value,
        str,
    ):
        return default

    return value.strip()


# ============================================================
# Number Helpers
# ============================================================

def safe_int(
    value: Any,
    default: int,
) -> int:
    """Convert a value to int or return default."""

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def safe_float(
    value: Any,
    default: float,
) -> float:
    """Convert a value to float or return default."""

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """Keep a numeric value within a range."""

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


# ============================================================
# Base64 Helpers
# ============================================================

def strip_data_uri(
    value: str,
) -> str:
    """
    Remove a data URI prefix.

    Example:
    data:image/png;base64,AAAA
    ->
    AAAA
    """

    value = value.strip()

    if value.startswith(
        "data:"
    ):

        if "," not in value:

            raise ValueError(
                "Invalid data URI."
            )

        value = value.split(
            ",",
            1,
        )[1]

    return value


def decode_base64(
    value: str,
) -> bytes:
    """
    Decode Base64 data safely.
    """

    if not isinstance(
        value,
        str,
    ):

        raise ValueError(
            "Base64 input must be a string."
        )

    value = strip_data_uri(
        value
    )

    if not value:

        raise ValueError(
            "Base64 input is empty."
        )

    try:

        return base64.b64decode(
            value,
            validate=True,
        )

    except (
        ValueError,
        binascii.Error,
    ) as exc:

        raise ValueError(
            "Invalid Base64 data."
        ) from exc


def encode_base64(
    value: bytes,
) -> str:
    """
    Encode bytes as Base64.
    """

    if not isinstance(
        value,
        bytes,
    ):

        raise TypeError(
            "Input must be bytes."
        )

    return base64.b64encode(
        value
    ).decode(
        "utf-8"
    )


# ============================================================
# Size Helpers
# ============================================================

def validate_bytes_size(
    data: bytes,
    max_bytes: int,
    name: str = "data",
) -> None:
    """
    Raise an error if data exceeds max_bytes.
    """

    if not isinstance(
        data,
        bytes,
    ):

        raise TypeError(
            f"{name} must be bytes."
        )

    if len(data) > max_bytes:

        raise ValueError(
            f"{name} exceeds the maximum "
            f"allowed size of {max_bytes} bytes."
        )


def megabytes(
    value: float,
) -> int:
    """Convert megabytes to bytes."""

    return int(
        value * 1024 * 1024
    )


# ============================================================
# Temporary Files
# ============================================================

def create_temp_file(
    suffix: str = "",
    prefix: str = "iraai_",
) -> str:
    """
    Create an empty temporary file and return its path.
    """

    file_handle = tempfile.NamedTemporaryFile(
        suffix=suffix,
        prefix=prefix,
        delete=False,
    )

    path = file_handle.name

    file_handle.close()

    return path


def remove_file(
    path: Optional[str],
) -> bool:
    """
    Safely remove a file.

    Returns True when removed.
    """

    if not path:
        return False

    try:

        if os.path.isfile(
            path
        ):

            os.remove(
                path
            )

            return True

    except OSError:
        return False

    return False


# ============================================================
# Path Helpers
# ============================================================

def safe_filename(
    filename: str,
    default: str = "file",
) -> str:
    """
    Sanitize a filename so it cannot contain
    directory traversal components.
    """

    if not isinstance(
        filename,
        str,
    ):

        return default

    filename = filename.strip()

    if not filename:

        return default

    filename = os.path.basename(
        filename
    )

    filename = re.sub(
        r"[^A-Za-z0-9._-]",
        "_",
        filename,
    )

    filename = filename.strip(
        "._"
    )

    if not filename:

        return default

    return filename


def ensure_directory(
    path: str,
) -> str:
    """
    Create a directory if it does not exist.
    """

    directory = Path(
        path
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return str(
        directory
    )


# ============================================================
# Image Helpers
# ============================================================

def validate_image_size(
    width: int,
    height: int,
    minimum: int = 64,
    maximum: int = 4096,
) -> None:
    """
    Validate image dimensions.
    """

    width = int(width)
    height = int(height)

    if width < minimum or width > maximum:

        raise ValueError(
            f"Image width must be between "
            f"{minimum} and {maximum}."
        )

    if height < minimum or height > maximum:

        raise ValueError(
            f"Image height must be between "
            f"{minimum} and {maximum}."
        )


# ============================================================
# Text Helpers
# ============================================================

def truncate_text(
    text: str,
    maximum_length: int,
) -> str:
    """
    Truncate text safely without changing the
    beginning of the content.
    """

    if not isinstance(
        text,
        str,
    ):

        return ""

    if maximum_length < 0:

        raise ValueError(
            "maximum_length cannot be negative."
        )

    if len(text) <= maximum_length:

        return text

    return text[
        :maximum_length
    ]


# ============================================================
# Response Helpers
# ============================================================

def success_response(
    **data: Any,
) -> dict:

    return {
        "success": True,
        **data,
    }


def error_response(
    message: str,
    **data: Any,
) -> dict:

    return {
        "success": False,
        "error": message,
        **data,
    }


__all__ = [
    "is_dict",
    "is_non_empty_string",
    "clean_string",
    "safe_int",
    "safe_float",
    "clamp",
    "strip_data_uri",
    "decode_base64",
    "encode_base64",
    "validate_bytes_size",
    "megabytes",
    "create_temp_file",
    "remove_file",
    "safe_filename",
    "ensure_directory",
    "validate_image_size",
    "truncate_text",
    "success_response",
    "error_response",
]