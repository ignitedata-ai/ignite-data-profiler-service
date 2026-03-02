"""Rate limiting utilities using slowapi.

Provides user-based rate limiting with Redis backend storage.
Falls back to IP-based limiting for unauthenticated requests.
"""

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.config import settings


def get_user_identifier(request: Request) -> str:
    """Get unique identifier for rate limiting.

    Priority:
    1. User ID from authenticated request (request.state.user)
    2. IP address as fallback for unauthenticated requests

    Args:
        request: FastAPI request object

    Returns:
        Unique identifier string for rate limiting

    """
    # Try to get user_id from request.state.user (set by JWTAuthMiddleware)
    if hasattr(request.state, "user") and request.state.user:
        user_data = request.state.user
        if isinstance(user_data, dict) and "user_id" in user_data:
            user_id = user_data["user_id"]
            return f"user:{user_id}"

    # Fallback to IP address for unauthenticated requests
    ip_address = get_remote_address(request)
    return f"ip:{ip_address}"


# Initialize limiter with Redis storage backend
limiter = Limiter(
    key_func=get_user_identifier,
    storage_uri=settings.REDIS_URL,
    enabled=settings.RATE_LIMIT_ENABLED,
    headers_enabled=True,  # Add rate limit headers to responses
    strategy="fixed-window",  # Use fixed window strategy for rate limiting
)
