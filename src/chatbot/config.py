"""Configuration utilities for environment-dependent settings."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv


# Load environment variables from a local .env if present.
load_dotenv()


class MissingEnvironmentVariable(RuntimeError):
    """Raised when a required environment variable is not defined."""


@lru_cache(maxsize=None)
def get_env(key: str, default: Optional[str] = None, *, required: bool = False) -> str:
    """Retrieve an environment variable, optionally enforcing its presence.

    Args:
        key: Name of the environment variable to fetch.
        default: Optional fallback value if the variable is unset.
        required: When ``True``, raises ``MissingEnvironmentVariable`` if the
            variable is not defined and no default is provided.

    Returns:
        The resolved environment variable value (or the provided default).

    Raises:
        MissingEnvironmentVariable: If ``required`` is ``True`` and the
            variable is missing.
    """

    value = os.getenv(key, default)
    if required and value is None:
        raise MissingEnvironmentVariable(
            f"Environment variable '{key}' must be set for this component to function."
        )
    return value


def get_bool_env(key: str, default: bool = False) -> bool:
    """Retrieve a boolean environment variable with standard truthy coercions."""

    raw_value = os.getenv(key)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def get_float_env(key: str, default: float) -> float:
    """Retrieve a floating-point environment variable with validation."""

    raw_value = os.getenv(key)
    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable '{key}' must be a valid float, got: {raw_value!r}."
        ) from exc
