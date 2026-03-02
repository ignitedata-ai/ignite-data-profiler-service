import time
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from core.logging import get_logger, get_trace_id, set_correlation_id

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging with correlation IDs."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        start_time = time.time()

        # Set correlation ID from header or generate new one
        correlation_id = request.headers.get("X-Correlation-ID")
        if correlation_id:
            set_correlation_id(correlation_id)

        # Log incoming request
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            path=request.url.path,
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
        )

        try:
            response = await call_next(request)

            # Calculate request duration
            process_time = time.time() - start_time

            # Log successful response
            logger.info(
                "Request completed",
                method=request.method,
                url=str(request.url),
                path=request.url.path,
                status_code=response.status_code,
                process_time=round(process_time * 1000, 2),  # Convert to milliseconds
            )

            # Add correlation ID and trace ID to response headers
            if correlation_id:
                response.headers["X-Correlation-ID"] = correlation_id

            # Add trace ID to response headers for debugging
            trace_id = get_trace_id()
            if trace_id:
                response.headers["X-Trace-ID"] = trace_id

            return response

        except Exception as exc:
            # Calculate request duration for failed requests
            process_time = time.time() - start_time

            # Log failed request
            logger.error(
                "Request failed",
                method=request.method,
                url=str(request.url),
                path=request.url.path,
                process_time=round(process_time * 1000, 2),
                error=str(exc),
                exc_info=True,
            )

            raise
