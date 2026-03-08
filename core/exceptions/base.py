from typing import Any


class BaseException(Exception):
    """Base exception class for Ignite Data Profiler Service application."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary representation."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class ValidationError(BaseException):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str = "Validation failed",
        field: str | None = None,
        value: Any | None = None,
        **kwargs,
    ) -> None:
        details = kwargs.get("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        super().__init__(message, details=details, **kwargs)


class NotFoundError(BaseException):
    """Raised when a requested resource is not found."""

    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: str | None = None,
        resource_id: str | None = None,
        **kwargs,
    ) -> None:
        details = kwargs.get("details", {})
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = str(resource_id)
        super().__init__(message, details=details, **kwargs)


class AuthenticationError(BaseException):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed", **kwargs) -> None:
        super().__init__(message, **kwargs)


class AuthorizationError(BaseException):
    """Raised when authorization fails."""

    def __init__(self, message: str = "Access denied", **kwargs) -> None:
        super().__init__(message, **kwargs)


class DatabaseError(BaseException):
    """Raised when database operations fail."""

    def __init__(
        self,
        message: str = "Database operation failed",
        operation: str | None = None,
        **kwargs,
    ) -> None:
        details = kwargs.get("details", {})
        if operation:
            details["operation"] = operation
        super().__init__(message, details=details, **kwargs)


class CacheError(BaseException):
    """Raised when cache operations fail."""

    def __init__(
        self,
        message: str = "Cache operation failed",
        operation: str | None = None,
        **kwargs,
    ) -> None:
        details = kwargs.get("details", {})
        if operation:
            details["operation"] = operation
        super().__init__(message, details=details, **kwargs)


class BusinessLogicError(BaseException):
    """Raised when business logic validation fails."""

    def __init__(self, message: str = "Business logic error", **kwargs) -> None:
        super().__init__(message, **kwargs)


class ExternalServiceError(BaseException):
    """Raised when external service calls fail."""

    def __init__(
        self,
        message: str = "External service error",
        service_name: str | None = None,
        status_code: int | None = None,
        **kwargs,
    ) -> None:
        details = kwargs.get("details", {})
        if service_name:
            details["service_name"] = service_name
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, details=details, **kwargs)


class ConfigurationError(BaseException):
    """Raised when configuration is invalid."""

    def __init__(
        self,
        message: str = "Configuration error",
        config_key: str | None = None,
        **kwargs,
    ) -> None:
        details = kwargs.get("details", {})
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, details=details, **kwargs)


class ProfilingTimeoutError(BaseException):
    """Raised when the overall profiling operation exceeds its configured timeout."""

    def __init__(
        self,
        message: str = "Profiling operation timed out",
        timeout_seconds: int | None = None,
        **kwargs,
    ) -> None:
        details = kwargs.get("details", {})
        if timeout_seconds is not None:
            details["timeout_seconds"] = timeout_seconds
        super().__init__(message, details=details, **kwargs)


class InternalTimeoutError(BaseException):
    """Raised when an internal operation (pool acquisition, query) times out.

    This is distinct from :class:`ProfilingTimeoutError` which represents the
    overall profiling deadline.  ``InternalTimeoutError`` wraps
    ``asyncio.TimeoutError`` that originates *inside* the profiling run — for
    example when the connection-pool acquisition timeout is exceeded because
    all connections are busy.
    """

    def __init__(
        self,
        message: str = "Internal operation timed out (connection pool or query)",
        source: str | None = None,
        **kwargs,
    ) -> None:
        details = kwargs.get("details", {})
        if source is not None:
            details["source"] = source
        super().__init__(message, details=details, **kwargs)


class TaskLimitError(BaseException):
    """Raised when the maximum number of concurrent profiling tasks is reached."""

    def __init__(
        self,
        message: str = "Too many concurrent profiling tasks",
        max_tasks: int | None = None,
        **kwargs,
    ) -> None:
        details = kwargs.get("details", {})
        if max_tasks is not None:
            details["max_tasks"] = max_tasks
        super().__init__(message, details=details, **kwargs)
