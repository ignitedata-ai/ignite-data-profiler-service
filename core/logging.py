import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource

from core.config import settings
from core.observability import get_logging_handler

# Context variable to store correlation ID across async requests
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str:
    """Get or create a correlation ID for the current request context."""
    correlation_id = correlation_id_var.get()
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
        correlation_id_var.set(correlation_id)
    return correlation_id


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current request context."""
    correlation_id_var.set(correlation_id)


def add_correlation_id(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add correlation ID to log entries."""
    event_dict["correlation_id"] = get_correlation_id()
    if not logger and not method_name:
        return event_dict
    return event_dict


def add_service_info(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add service information to log entries."""
    event_dict["service"] = settings.OTEL_SERVICE_NAME
    event_dict["version"] = settings.OTEL_SERVICE_VERSION
    event_dict["environment"] = settings.ENVIRONMENT.value
    if not logger and not method_name:
        return event_dict
    return event_dict


def get_trace_id() -> str | None:
    """Get the current trace ID from OpenTelemetry context."""
    try:
        current_span = trace.get_current_span()
        if current_span and current_span.get_span_context().trace_id != trace.INVALID_TRACE_ID:
            # Convert trace ID to hex string (32 characters, zero-padded)
            return f"{current_span.get_span_context().trace_id:032x}"
    except Exception:
        pass
    return None


def get_span_id() -> str | None:
    """Get the current span ID from OpenTelemetry context."""
    try:
        current_span = trace.get_current_span()
        if current_span and current_span.get_span_context().span_id != trace.INVALID_SPAN_ID:
            # Convert span ID to hex string (16 characters, zero-padded)
            return f"{current_span.get_span_context().span_id:016x}"
    except Exception:
        pass
    return None


def add_trace_context(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add trace and span IDs to log entries."""
    trace_id = get_trace_id()
    span_id = get_span_id()

    if trace_id:
        event_dict["trace_id"] = trace_id
    if span_id:
        event_dict["span_id"] = span_id
    if not logger and not method_name:
        return event_dict

    return event_dict


def add_otel_logging(_logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Send log entries to OpenTelemetry logging."""
    # Get the OpenTelemetry logging handler
    otel_handler = get_logging_handler()
    if not otel_handler:
        return event_dict

    try:
        # Convert structlog level to standard logging level
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "warn": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }

        # Get log level from event_dict or method_name
        log_level = event_dict.get("level", method_name)
        if isinstance(log_level, str):
            log_level = log_level.lower()

        numeric_level = level_map.get(log_level, logging.INFO)

        # Create a log record
        record = logging.LogRecord(
            name=event_dict.get("logger", "structlog"),
            level=numeric_level,
            pathname="",
            lineno=0,
            msg=event_dict.get("event", ""),
            args=(),
            exc_info=None,
        )

        # Add additional attributes to the record
        record.correlation_id = event_dict.get("correlation_id")
        record.service = event_dict.get("service")
        record.version = event_dict.get("version")
        record.environment = event_dict.get("environment")

        # Add any extra fields as attributes
        for key, value in event_dict.items():
            if key not in ["event", "level", "logger", "correlation_id", "service", "version", "environment", "timestamp"]:
                setattr(record, key, value)

        # Send to OpenTelemetry
        otel_handler.emit(record)

    except Exception:
        # Don't let OpenTelemetry errors break regular logging
        pass

    return event_dict


def configure_logging() -> None:
    """Configure structured logging with structlog."""
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.value),
    )

    # Common processors for all loggers
    common_processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        add_correlation_id,
        add_trace_context,  # Add trace and span IDs to logs
        add_service_info,
        add_otel_logging,  # Send logs to OpenTelemetry
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.LOG_FORMAT == "json":
        # JSON formatter for production
        renderer = structlog.processors.JSONRenderer()
    else:
        # Human-readable formatter for development
        renderer = structlog.dev.ConsoleRenderer(colors=settings.ENVIRONMENT == "development")

    # Configure structlog
    structlog.configure(
        processors=common_processors + [renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    # Add OpenTelemetry handler to root logger if available
    # This ensures that direct Python logging calls also get sent to OpenTelemetry
    try:
        otel_handler = get_logging_handler()
        if otel_handler:
            root_logger = logging.getLogger()
            root_logger.addHandler(otel_handler)
    except ImportError:
        # OpenTelemetry may not be fully initialized yet
        pass

    # Configure specific loggers
    # Silence noisy third-party loggers in production
    if settings.ENVIRONMENT == "production":
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("redis").setLevel(logging.WARNING)


class CentralizedLogger:
    """Centralized logger with OpenTelemetry integration for both structured logging and tracing."""

    def __init__(self, name: str = __name__):
        self.name = name
        self.logger = structlog.get_logger(name)
        self.tracer = trace.get_tracer(name)
        # Disable OTEL logging for Jaeger compatibility
        self._otel_logging_enabled = False

    def _log_with_trace(self, level: str, event: str, **kwargs):
        """Log with OpenTelemetry trace context and span attributes."""
        span = trace.get_current_span()

        if span and span.is_recording():
            self._add_span_event(span, level, event, **kwargs)
            self._add_span_attributes(span, **kwargs)
            self._set_span_status_on_error(span, level, event)

        self._try_otel_logging_if_enabled(level, event, **kwargs)
        self._log_with_structlog_fallback(level, event, **kwargs)

    def _add_span_event(self, span, level: str, event: str, **kwargs):
        """Add event to span for Jaeger UI visibility."""
        event_attributes = {
            "level": level.upper(),
            "logger": self.name,
            "timestamp": int(time.time() * 1000),  # milliseconds
            "event_name": event,
        }

        # Add all kwargs as event attributes
        for key, value in kwargs.items():
            if key != "exc_info":  # Skip exc_info for events
                event_attributes[key] = str(value) if isinstance(value, (dict, list)) else value

        span.add_event(f"[{level.upper()}] {event}", attributes=event_attributes)

    def _add_span_attributes(self, span, **kwargs):
        """Add attributes to span for searchability."""
        safe_attributes = self._prepare_safe_span_attributes(**kwargs)
        self._set_span_attributes_safely(span, safe_attributes)

    def _prepare_safe_span_attributes(self, **kwargs):
        """Prepare safe attributes for span."""
        safe_attributes = {}
        for key, value in kwargs.items():
            if key != "exc_info":
                attr_key = f"log.{key}"
                safe_attributes[attr_key] = self._convert_value_to_safe_attribute(value)
        return safe_attributes

    @staticmethod
    def _convert_value_to_safe_attribute(value):
        """Convert value to safe span attribute."""
        try:
            # Handle None values explicitly - OpenTelemetry doesn't accept None
            if value is None:
                return "null"
            elif isinstance(value, (str, int, float, bool)):
                # Check for overly large integers that might cause protobuf issues
                if isinstance(value, int) and (value > 2**63 - 1 or value < -(2**63)):
                    return str(value)
                return value
            elif isinstance(value, (dict, list)):
                import json

                return json.dumps(value, default=str)[:500]  # Limit length
            else:
                return str(value)[:500]  # Limit string length
        except Exception:
            return str(value)[:100]  # Fallback

    @staticmethod
    def _set_span_attributes_safely(span, safe_attributes):
        """Set span attributes safely."""
        for key, value in safe_attributes.items():
            try:
                span.set_attribute(key, value)
            except Exception:
                # If individual attribute fails, skip it
                pass

    @staticmethod
    def _set_span_status_on_error(span, level: str, event: str):
        """Set span status for error levels."""
        if level in ["error", "critical"]:
            try:
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR, event))
            except Exception:
                pass

    def _try_otel_logging_if_enabled(self, level: str, event: str, **kwargs):
        """Try OTEL logging if enabled."""
        if hasattr(self, "_otel_logging_enabled") and self._otel_logging_enabled:
            self._send_to_otel_logging(level, event, **kwargs)

    def _log_with_structlog_fallback(self, level: str, event: str, **kwargs):
        """Log using structlog with fallback."""
        try:
            log_method = getattr(self.logger, level)
            log_method(event, **kwargs)
        except Exception:
            # Fallback if structlog fails

            std_logger = logging.getLogger(self.name)
            getattr(std_logger, level, std_logger.info)(f"{event}: {kwargs}")

    def _send_to_otel_logging(self, level: str, event: str, **kwargs):
        """Send log directly to OpenTelemetry logging handler with proper trace correlation."""
        try:
            self._try_otel_handler_logging(level, event, **kwargs)
            self._try_otel_logger_provider_logging(level, event, **kwargs)
        except Exception:
            # Don't let OpenTelemetry errors break regular logging
            pass

    def _try_otel_handler_logging(self, level: str, event: str, **kwargs):
        """Try logging via OpenTelemetry handler."""
        otel_handler = get_logging_handler()
        if not otel_handler:
            return

        record = self._create_stdlib_log_record(level, event, **kwargs)
        self._add_trace_context_to_record(record)
        self._add_service_info_to_record(record)
        self._add_kwargs_to_record(record, **kwargs)
        otel_handler.emit(record)

    def _try_otel_logger_provider_logging(self, level: str, event: str, **kwargs):
        """Try logging via OpenTelemetry logger provider."""
        try:
            from core.observability import get_logger_provider

            logger_provider = get_logger_provider()
            if not logger_provider or not hasattr(logger_provider, "get_logger"):
                return

            otel_logger = logger_provider.get_logger(name=self.name, version=settings.OTEL_SERVICE_VERSION)

            trace_context = self._get_trace_context()
            attributes = self._prepare_otel_attributes(level, event, **kwargs)

            from opentelemetry.sdk._logs import ReadWriteLogRecord

            otel_logger.emit(
                ReadWriteLogRecord(
                    timestamp=time.time_ns(),
                    trace_id=trace_context["trace_id"],
                    span_id=trace_context["span_id"],
                    trace_flags=trace_context["trace_flags"],
                    severity_text=level.upper(),
                    severity_number=self._get_severity_number(level),  # type: ignore
                    body=event,
                    resource=self._create_otel_resource(),
                    attributes=attributes,
                )
            )
        except Exception:
            # Direct logger approach failed
            pass

    def _create_stdlib_log_record(self, level: str, event: str, **kwargs):
        """Create a standard library log record."""
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "warn": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        numeric_level = level_map.get(level, logging.INFO)

        return logging.LogRecord(
            name=self.name,
            level=numeric_level,
            pathname="",
            lineno=0,
            msg=event,
            args=(),
            exc_info=kwargs.get("exc_info"),
        )

    @staticmethod
    def _add_trace_context_to_record(record):
        """Add trace context to log record."""
        span = trace.get_current_span()
        if span and span.is_recording():
            span_context = span.get_span_context()
            record.otelTraceID = format(span_context.trace_id, "032x")
            record.otelSpanID = format(span_context.span_id, "016x")
            record.trace_id = span_context.trace_id
            record.span_id = span_context.span_id
            record.trace_flags = span_context.trace_flags

    @staticmethod
    def _add_service_info_to_record(record):
        """Add service information to log record."""
        setattr(record, "service.name", settings.OTEL_SERVICE_NAME)
        setattr(record, "service.version", settings.OTEL_SERVICE_VERSION)
        record.environment = settings.ENVIRONMENT.value

    @staticmethod
    def _add_kwargs_to_record(record, **kwargs):
        """Add kwargs as attributes to log record."""
        for key, value in kwargs.items():
            if key != "exc_info":  # exc_info already handled
                try:
                    if isinstance(value, (dict, list)):
                        setattr(record, key, str(value))
                    else:
                        setattr(record, key, value)
                except (TypeError, ValueError):
                    setattr(record, key, str(value))

    @staticmethod
    def _get_trace_context():
        """Get current trace context."""
        span = trace.get_current_span()
        if span and span.is_recording():
            span_context = span.get_span_context()
            return {
                "trace_id": span_context.trace_id,
                "span_id": span_context.span_id,
                "trace_flags": span_context.trace_flags,
            }
        return {"trace_id": None, "span_id": None, "trace_flags": None}

    def _prepare_otel_attributes(self, level: str, event: str, **kwargs):
        """Prepare attributes for OpenTelemetry log record."""
        attributes = {
            "event": event,
            "level": level,
            "logger": self.name,
            "service.name": settings.OTEL_SERVICE_NAME,
            "service.version": settings.OTEL_SERVICE_VERSION,
            "environment": settings.ENVIRONMENT.value,
        }

        for key, value in kwargs.items():
            if key != "exc_info":
                attributes[key] = str(value) if isinstance(value, (dict, list)) else value

        return attributes

    @staticmethod
    def _get_severity_number(level: str):
        """Get severity number for log level."""
        if hasattr(logging, "getLevelName"):
            return logging.getLevelName(level.upper())
        return getattr(logging, level.upper(), 20)

    @staticmethod
    def _create_otel_resource():
        """Create OpenTelemetry resource."""
        return Resource.create(
            {
                "service.name": settings.OTEL_SERVICE_NAME,
                "service.version": settings.OTEL_SERVICE_VERSION,
                "environment": settings.ENVIRONMENT.value,
            }
        )

    def debug(self, event: str, **kwargs):
        """Log debug message with OpenTelemetry integration."""
        self._log_with_trace("debug", event, **kwargs)

    def info(self, event: str, **kwargs):
        """Log info message with OpenTelemetry integration."""
        self._log_with_trace("info", event, **kwargs)

    def warning(self, event: str, **kwargs):
        """Log warning message with OpenTelemetry integration."""
        self._log_with_trace("warning", event, **kwargs)

    def warn(self, event: str, **kwargs):
        """Alias for warning to match standard logging interface."""
        self.warning(event, **kwargs)

    def error(self, event: str, **kwargs):
        """Log error message with OpenTelemetry integration."""
        self._log_with_trace("error", event, **kwargs)

    def critical(self, event: str, **kwargs):
        """Log critical message with OpenTelemetry integration."""
        self._log_with_trace("critical", event, **kwargs)

    def exception(self, event: str, **kwargs):
        """Log exception with traceback and OpenTelemetry integration."""
        # Add exc_info=True to capture the exception traceback
        kwargs["exc_info"] = True
        self._log_with_trace("error", event, **kwargs)

        # Record the exception in the current span
        span = trace.get_current_span()
        if span and span.is_recording():
            _, exc_value, _ = sys.exc_info()
            if exc_value:
                span.record_exception(exc_value)

    def with_context(self, **kwargs):
        """Return a new logger instance with added context."""
        # Create a new bound logger with context
        bound_logger = self.logger.bind(**kwargs)

        # Create new CentralizedLogger instance with the bound logger
        new_logger = CentralizedLogger(self.name)
        new_logger.logger = bound_logger
        return new_logger

    def bind(self, **kwargs):
        """Alias for with_context to match structlog interface."""
        return self.with_context(**kwargs)


def configure_otel_logging() -> None:
    """Configure OpenTelemetry logging integration after OTel is initialized."""
    try:
        from core.observability import get_logging_handler

        otel_handler = get_logging_handler()
        if otel_handler:
            root_logger = logging.getLogger()

            # Check if handler is already added to avoid duplicates
            if otel_handler not in root_logger.handlers:
                root_logger.addHandler(otel_handler)

            # Get the configured logger and log success
            logger = get_logger(__name__)
            logger.info("OpenTelemetry logging integration configured")
    except Exception as e:
        # Don't let OpenTelemetry configuration errors break the application
        logger = get_logger(__name__)
        logger.warning("Failed to configure OpenTelemetry logging integration", error=str(e))


def get_logger(name: str) -> CentralizedLogger:
    """Get a configured centralized logger instance."""
    return CentralizedLogger(name)
