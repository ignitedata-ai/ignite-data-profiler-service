"""JWT authentication middleware.

Validates Bearer tokens on all requests except explicitly excluded paths.
Sets request.state.user with decoded payload on success.
Returns 401 JSON response on failure.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import settings
from core.logging import get_logger
from core.security.jwt import verify_token

logger = get_logger(__name__)

PUBLIC_PATHS: set[str] = {
    "/profile/health",
    "/profile/metrics",
}

PUBLIC_PATH_PREFIXES: tuple[str, ...] = (
    "/profile/docs",
    "/profile/redoc",
    "/profile/openapi.json",
)


def _is_public_path(path: str) -> bool:
    """Return True if the request path should bypass authentication."""
    if path in PUBLIC_PATHS:
        return True
    return path.startswith(PUBLIC_PATH_PREFIXES)


def _auth_error_response(message: str, error_code: str = "AUTHENTICATION_ERROR") -> JSONResponse:
    """Build a 401 JSON response matching the project's error envelope."""
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "code": error_code,
                "message": message,
                "timestamp": None,
            }
        },
        headers={"WWW-Authenticate": "Bearer"},
    )


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Middleware that verifies JWT Bearer tokens on protected routes."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not settings.JWT_AUTH_ENABLED:
            return await call_next(request)

        if _is_public_path(request.url.path):
            return await call_next(request)

        if request.method == "OPTIONS":
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            logger.warning(
                "Missing Authorization header",
                path=request.url.path,
                method=request.method,
            )
            return _auth_error_response("Missing Authorization header")

        parts = auth_header.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            logger.warning(
                "Invalid Authorization scheme",
                path=request.url.path,
                method=request.method,
            )
            return _auth_error_response("Invalid Authorization header format. Expected: Bearer <token>")

        token = parts[1].strip()
        if not token:
            return _auth_error_response("Empty token")

        try:
            payload = verify_token(token)
        except ExpiredSignatureError:
            logger.warning("Expired JWT token", path=request.url.path)
            return _auth_error_response("Token has expired")
        except FileNotFoundError:
            logger.error("JWT public key file not found")
            return _auth_error_response(
                "Authentication service unavailable",
                error_code="CONFIGURATION_ERROR",
            )
        except InvalidTokenError as exc:
            logger.warning(
                "Invalid JWT token",
                path=request.url.path,
                error=str(exc),
            )
            return _auth_error_response("Invalid or malformed token")

        request.state.user = payload

        return await call_next(request)
