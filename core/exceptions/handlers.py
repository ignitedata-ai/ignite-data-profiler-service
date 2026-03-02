import traceback
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.exc import IntegrityError, OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.exceptions.base import (
    AuthenticationError,
    AuthorizationError,
    BaseException,
    BusinessLogicError,
    CacheError,
    ConfigurationError,
    DatabaseError,
    ExternalServiceError,
    NotFoundError,
    ProfilingTimeoutError,
    TaskLimitError,
    ValidationError,
)
from core.logging import get_logger

logger = get_logger(__name__)


def create_error_response(
    error_code: str,
    message: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Create standardized error response."""
    content = {
        "error": {
            "code": error_code,
            "message": message,
            "timestamp": None,  # Will be added by middleware
        }
    }

    if details:
        content["error"]["details"] = details

    return JSONResponse(status_code=status_code, content=content)


async def exception_handler(request: Request, exc: BaseException) -> JSONResponse:
    """Handle custom exceptions."""
    status_map = {
        ValidationError: status.HTTP_400_BAD_REQUEST,
        NotFoundError: status.HTTP_404_NOT_FOUND,
        AuthenticationError: status.HTTP_401_UNAUTHORIZED,
        AuthorizationError: status.HTTP_403_FORBIDDEN,
        BusinessLogicError: status.HTTP_422_UNPROCESSABLE_ENTITY,
        DatabaseError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        CacheError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ExternalServiceError: status.HTTP_502_BAD_GATEWAY,
        ConfigurationError: status.HTTP_400_BAD_REQUEST,
        ProfilingTimeoutError: status.HTTP_408_REQUEST_TIMEOUT,
        TaskLimitError: status.HTTP_429_TOO_MANY_REQUESTS,
    }

    status_code = status_map.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)

    logger.error(
        "Exception occurred",
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        path=request.url.path,
        method=request.method,
        exc_info=True,
    )

    return create_error_response(
        error_code=exc.error_code,
        message=exc.message,
        status_code=status_code,
        details=exc.details,
    )


async def validation_exception_handler(request: Request, exc: PydanticValidationError) -> JSONResponse:
    """Handle Pydantic validation errors."""
    errors = []
    for error in exc.errors():
        field_path = ".".join(str(loc) for loc in error["loc"])
        errors.append(
            {
                "field": field_path,
                "message": error["msg"],
                "type": error["type"],
                "input": error.get("input"),
            }
        )

    logger.warning(
        "Validation error occurred",
        errors=errors,
        path=request.url.path,
        method=request.method,
    )

    return create_error_response(
        error_code="VALIDATION_ERROR",
        message="Request validation failed",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details={"validation_errors": errors},
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method,
    )

    return create_error_response(
        error_code="HTTP_ERROR",
        message=str(exc.detail),
        status_code=exc.status_code,
    )


async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle Starlette HTTP exceptions."""
    logger.warning(
        "Starlette HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method,
    )

    return create_error_response(
        error_code="HTTP_ERROR",
        message=str(exc.detail),
        status_code=exc.status_code,
    )


async def database_exception_handler(request: Request, exc: IntegrityError) -> JSONResponse:
    """Handle database integrity errors."""
    logger.error(
        "Database integrity error occurred",
        error=str(exc.orig) if hasattr(exc, "orig") else str(exc),
        path=request.url.path,
        method=request.method,
        exc_info=True,
    )

    return create_error_response(
        error_code="DATABASE_INTEGRITY_ERROR",
        message="Database constraint violation",
        status_code=status.HTTP_409_CONFLICT,
    )


async def database_operational_exception_handler(request: Request, exc: OperationalError) -> JSONResponse:
    """Handle database operational errors."""
    logger.error(
        "Database operational error occurred",
        error=str(exc.orig) if hasattr(exc, "orig") else str(exc),
        path=request.url.path,
        method=request.method,
        exc_info=True,
    )

    return create_error_response(
        error_code="DATABASE_OPERATIONAL_ERROR",
        message="Database operation failed",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    error_id = id(exc)  # Use object ID as error identifier

    logger.error(
        "Unexpected exception occurred",
        error_id=error_id,
        error_type=type(exc).__name__,
        error=str(exc),
        path=request.url.path,
        method=request.method,
        traceback=traceback.format_exc(),
        exc_info=True,
    )

    return create_error_response(
        error_code="INTERNAL_SERVER_ERROR",
        message="An unexpected error occurred",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        details={"error_id": error_id},
    )
