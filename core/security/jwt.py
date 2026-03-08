"""JWT token verification using RS256 with RSA public key."""

from __future__ import annotations

from pathlib import Path

import jwt

from core.logging import get_logger

logger = get_logger(__name__)

_public_key: str | None = None

JWT_ALGORITHM = "RS256"


def get_public_key_path() -> Path:
    """Return the path to the RSA public key file."""
    return Path(__file__).resolve().parent.parent.parent / "keys" / "public.pem"


def load_public_key(*, force_reload: bool = False) -> str:
    """Load the RSA public key from disk and cache it.

    Args:
        force_reload: If True, re-read from disk even if already cached.

    Returns:
        The PEM-encoded public key string.

    Raises:
        FileNotFoundError: If public.pem does not exist.
        ValueError: If the file is empty.
    """
    global _public_key

    if _public_key is not None and not force_reload:
        return _public_key

    key_path = get_public_key_path()
    if not key_path.exists():
        raise FileNotFoundError(
            f"JWT public key not found at {key_path}. Ensure keys/public.pem exists or set JWT_PUBLIC_KEY env var."
        )

    key_content = key_path.read_text().strip()
    if not key_content:
        raise ValueError(f"JWT public key file at {key_path} is empty.")

    _public_key = key_content
    logger.info("JWT public key loaded", path=str(key_path))
    return _public_key


def verify_token(token: str) -> dict:
    """Verify a JWT token using the RS256 public key.

    Args:
        token: The raw JWT string (without "Bearer " prefix).

    Returns:
        The decoded token payload dict.

    Raises:
        ExpiredSignatureError: Token has expired.
        DecodeError: Token is malformed.
        InvalidTokenError: Any other JWT validation failure.
        FileNotFoundError: Public key file missing.
    """
    public_key = load_public_key()

    return jwt.decode(
        token,
        public_key,
        algorithms=[JWT_ALGORITHM],
        options={
            "verify_signature": True,
            "verify_exp": True,
            "require": ["exp"],
        },
    )


def reset_key_cache() -> None:
    """Clear the cached public key. Useful for testing."""
    global _public_key
    _public_key = None
