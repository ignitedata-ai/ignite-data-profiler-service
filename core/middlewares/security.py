from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        if not settings.SECURITY_HEADERS_ENABLED:
            return response

        # X-Content-Type-Options: Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # X-Frame-Options: Prevent clickjacking
        response.headers["X-Frame-Options"] = settings.X_FRAME_OPTIONS

        # X-XSS-Protection: Enable browser XSS protection (legacy but still useful)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Strict-Transport-Security: Force HTTPS (only in production)
        if settings.ENVIRONMENT.value == "production" and settings.HSTS_ENABLED:
            hsts_value = f"max-age={settings.HSTS_MAX_AGE}"
            if settings.HSTS_INCLUDE_SUBDOMAINS:
                hsts_value += "; includeSubDomains"
            if settings.HSTS_PRELOAD:
                hsts_value += "; preload"
            response.headers["Strict-Transport-Security"] = hsts_value

        # Referrer-Policy: Control referrer information
        response.headers["Referrer-Policy"] = settings.REFERRER_POLICY

        # Content-Security-Policy: Prevent XSS and data injection
        if settings.CSP_ENABLED and settings.CONTENT_SECURITY_POLICY:
            # Skip CSP in development if configured to do so
            if settings.ENVIRONMENT.value == "development" and settings.CSP_DISABLE_IN_DEVELOPMENT:
                pass  # Don't set CSP header
            # Use more permissive CSP for OpenAPI docs
            elif self._is_docs_endpoint(request.url.path):
                response.headers["Content-Security-Policy"] = self._get_docs_csp()
            else:
                response.headers["Content-Security-Policy"] = settings.CONTENT_SECURITY_POLICY

        # Permissions-Policy: Control browser features/APIs
        if settings.PERMISSIONS_POLICY:
            response.headers["Permissions-Policy"] = settings.PERMISSIONS_POLICY

        # X-Permitted-Cross-Domain-Policies: Control cross-domain content
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"

        # Server header removal (optional)
        if settings.REMOVE_SERVER_HEADER and "Server" in response.headers:
            del response.headers["Server"]

        return response

    def _is_docs_endpoint(self, path: str) -> bool:
        """Check if the request is for OpenAPI docs endpoints."""
        return path in ["/docs", "/redoc", "/openapi.json"] or path.startswith("/docs/") or path.startswith("/redoc/")

    def _get_docs_csp(self) -> str:
        """Get a more permissive CSP for OpenAPI documentation."""
        return (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://unpkg.com; "
            "img-src 'self' data: https: blob:; "
            "font-src 'self' https: data:; "
            "connect-src 'self' https:; "
            "media-src 'self'; "
            "object-src 'none'; "
            "child-src 'self'; "
            "worker-src 'self' blob:; "
            "frame-ancestors 'none'; "
            "form-action 'self'; "
            "base-uri 'self';"
        )
