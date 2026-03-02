import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


def configure_cors(app: FastAPI) -> None:
    """Configure CORS middleware with environment-specific settings."""
    # Get allowed origins based on environment
    allowed_origins = _get_allowed_origins()

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
        expose_headers=settings.CORS_EXPOSE_HEADERS,
        max_age=settings.CORS_MAX_AGE,
    )

    logger.info(
        "CORS middleware configured",
        environment=settings.ENVIRONMENT.value,
        allow_origins=allowed_origins if len(str(allowed_origins)) < 200 else f"{len(allowed_origins)} origins",
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
    )


def _get_allowed_origins() -> list[str] | list[str]:
    """Get allowed origins based on environment and configuration."""
    if settings.ENVIRONMENT.value == "development":
        # In development, be more permissive but still secure
        origins = []

        # Add configured origins
        if settings.CORS_ALLOWED_ORIGINS:
            origins.extend(settings.CORS_ALLOWED_ORIGINS)

        # Add common development origins if not already present
        dev_origins = [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
            "http://127.0.0.1:8080",
        ]

        for origin in dev_origins:
            if origin not in origins:
                origins.append(origin)

        # Allow wildcard only if explicitly configured
        if settings.CORS_ALLOW_ALL_ORIGINS:
            return ["*"]

        return origins

    elif settings.ENVIRONMENT.value == "production":
        # In production, be strict with origins
        if not settings.CORS_ALLOWED_ORIGINS:
            logger.warning("No CORS origins configured for production environment. This may block legitimate requests.")
            return []

        # Validate origins in production
        validated_origins = []
        for origin in settings.CORS_ALLOWED_ORIGINS:
            if _is_valid_origin(origin):
                validated_origins.append(origin)
            else:
                logger.warning(f"Invalid CORS origin skipped: {origin}")

        return validated_origins

    else:
        # Testing or other environments
        return settings.CORS_ALLOWED_ORIGINS or ["*"]


def _is_valid_origin(origin: str) -> bool:
    """Validate origin format."""
    if not origin:
        return False

    # Allow wildcard
    if origin == "*":
        return True

    # Split origin into components
    try:
        # Basic pattern check first
        if not re.match(r"^https?://.+", origin):
            return False

        # Remove protocol
        without_protocol = origin.split("://", 1)[1]

        # Split host and port
        if ":" in without_protocol:
            host, port_str = without_protocol.rsplit(":", 1)
            try:
                port = int(port_str)
                if port < 1 or port > 65535:
                    return False
            except ValueError:
                return False
        else:
            host = without_protocol

        # Validate host
        if not host:
            return False

        # Allow localhost
        if host in ["localhost", "127.0.0.1", "0.0.0.0"]:
            return True

        # Validate IP address
        if re.match(r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$", host):
            return True

        # Validate domain name
        # Domain must have at least one dot for TLD
        if "." not in host:
            return False

        # Check each part of the domain
        parts = host.split(".")
        for part in parts:
            if not part:  # Empty part (double dot)
                return False
            if len(part) > 63:  # Domain part too long
                return False
            if not re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?$", part):
                return False

        return True

    except Exception:
        return False
