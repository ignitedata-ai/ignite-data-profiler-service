import os
import sys
from typing import Any

# CRITICAL: Set Prometheus multiprocess directory BEFORE any prometheus_client imports
# This must happen at module level for multi-worker compatibility
_PROMETHEUS_MULTIPROC_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR", "/tmp/prometheus_multiproc_dir")  # nosec B108
os.environ["PROMETHEUS_MULTIPROC_DIR"] = _PROMETHEUS_MULTIPROC_DIR

# ruff: noqa: E402
# Imports below must come after environment setup for Prometheus multiprocess mode
from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, LogExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from prometheus_client import CollectorRegistry

from core.config import settings

# Logger will be set after avoiding circular import
logger = None


def _get_logger():
    """Get logger instance, avoiding circular import."""
    global logger
    if logger is None:
        from core.logging import get_logger

        logger = get_logger(__name__)
    return logger


# Global instances (each worker gets its own copy)
tracer: trace.Tracer | None = None
_trace_provider: TracerProvider | None = None
_span_processors: list[BatchSpanProcessor] = []
_meter_provider: MeterProvider | None = None
_logger_provider: LoggerProvider | None = None
_log_processors: list[BatchLogRecordProcessor] = []
_logging_handler: LoggingHandler | None = None
_resource: Resource | None = None
_metrics_registry: CollectorRegistry | None = None


class _BaseConsoleExporter:
    """Base class for console exporters with common error handling."""

    def __init__(self):
        self._shutdown = False

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self._shutdown = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending data.

        Args:
            timeout_millis: Timeout in milliseconds (interface requirement, not used for console).

        """
        if self._shutdown:
            return False
        try:
            if not sys.stdout.closed:
                sys.stdout.flush()
            return True
        except Exception:
            return False

    def _safe_write(self, message: str) -> bool:
        """Safely write to stdout with error handling."""
        try:
            sys.stdout.write(message)
            sys.stdout.flush()
            return True
        except (ValueError, OSError, AttributeError) as e:
            _get_logger().debug("Console exporter I/O error", error=str(e))
            return False
        except Exception as e:
            _get_logger().warning("Unexpected error in console exporter", error=str(e))
            return False


class SafeConsoleSpanExporter(_BaseConsoleExporter, SpanExporter):
    """A console span exporter that handles I/O errors gracefully."""

    def export(self, spans) -> SpanExportResult:
        """Export spans to console with error handling."""
        if self._shutdown:
            return SpanExportResult.FAILURE

        for span in spans:
            span_context = span.get_span_context()
            span_dict = {
                "name": span.name,
                "trace_id": f"{span_context.trace_id:032x}" if span_context else "unknown",
                "span_id": f"{span_context.span_id:016x}" if span_context else "unknown",
                "start_time": span.start_time,
                "end_time": span.end_time,
            }
            if not self._safe_write(f"Span: {span_dict}\n"):
                return SpanExportResult.FAILURE

        return SpanExportResult.SUCCESS


class SafeConsoleLogExporter(_BaseConsoleExporter, LogExporter):
    """A console log exporter that handles I/O errors gracefully."""

    def export(self, batch) -> None:
        """Export log records to console with error handling."""
        if self._shutdown:
            return

        for log_record in batch:
            log_dict = {
                "timestamp": getattr(log_record, "timestamp", None),
                "severity": getattr(log_record, "severity_text", None),
                "body": str(getattr(log_record, "body", "")),
                "trace_id": f"{trace_id:032x}" if (trace_id := getattr(log_record, "trace_id", None)) else None,
                "span_id": f"{span_id:016x}" if (span_id := getattr(log_record, "span_id", None)) else None,
            }
            self._safe_write(f"Log: {log_dict}\n")


def _validate_config() -> None:
    """Validate observability configuration."""
    if settings.TRACE_SAMPLING_RATE < 0 or settings.TRACE_SAMPLING_RATE > 1:
        raise ValueError(f"TRACE_SAMPLING_RATE must be between 0 and 1, got {settings.TRACE_SAMPLING_RATE}")

    if settings.JAEGER_ENABLED and not settings.JAEGER_AGENT_URL:
        _get_logger().warning("JAEGER_ENABLED is True but JAEGER_AGENT_URL is not set")

    if settings.ENABLE_METRICS:
        # Ensure multiprocess directory exists
        try:
            os.makedirs(_PROMETHEUS_MULTIPROC_DIR, exist_ok=True)
        except Exception as e:
            _get_logger().error("Failed to create Prometheus multiprocess directory", error=str(e))
            raise


def _get_or_create_resource() -> Resource:
    """Get or create OpenTelemetry resource (singleton per worker)."""
    global _resource
    if _resource is None:
        _resource = Resource.create(
            {
                "service.name": settings.OTEL_SERVICE_NAME,
                "service.version": settings.OTEL_SERVICE_VERSION,
                "service.environment": settings.ENVIRONMENT.value,
            }
        )
    return _resource


def _create_otlp_endpoint(path: str) -> str:
    """Create OTLP endpoint URL."""
    return f"{settings.JAEGER_AGENT_URL}{path}"


class _LoggingSpanExporter(SpanExporter):
    """Wrapper exporter that logs export results for debugging."""

    def __init__(self, delegate: SpanExporter):
        self._delegate = delegate

    def export(self, spans) -> SpanExportResult:
        """Export spans and log the result."""
        try:
            result = self._delegate.export(spans)
            if result != SpanExportResult.SUCCESS:
                _get_logger().warning(
                    "Span export failed",
                    result=str(result),
                    span_count=len(spans) if spans else 0,
                )
            else:
                _get_logger().debug(
                    "Spans exported successfully",
                    span_count=len(spans) if spans else 0,
                )
            return result
        except Exception as e:
            _get_logger().error(
                "Span export raised exception",
                error=str(e),
                error_type=type(e).__name__,
                span_count=len(spans) if spans else 0,
            )
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Shutdown the delegate exporter."""
        return self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the delegate exporter."""
        return self._delegate.force_flush(timeout_millis)


def _create_otlp_span_exporter() -> SpanExporter | None:
    """Create OTLP span exporter with error handling and logging."""
    try:
        endpoint = _create_otlp_endpoint("/v1/traces")

        exporter = OTLPSpanExporter(
            endpoint=endpoint,
            timeout=10,  # 10 second timeout
        )

        # Wrap with logging exporter to capture export errors
        logging_exporter = _LoggingSpanExporter(exporter)

        _get_logger().info(
            "OTLP HTTP span exporter configured",
            endpoint=endpoint,
        )
        return logging_exporter
    except ImportError:
        _get_logger().debug("OTLP HTTP exporter not available")
    except Exception as e:
        _get_logger().warning("Failed to configure OTLP span exporter", error=str(e))
    return None


def _create_otlp_log_exporter() -> OTLPLogExporter | None:
    """Create OTLP log exporter with error handling."""
    try:
        endpoint = _create_otlp_endpoint("/v1/logs")
        exporter = OTLPLogExporter(endpoint=endpoint)
        _get_logger().info("OTLP HTTP log exporter configured", endpoint=endpoint)
        return exporter
    except ImportError:
        _get_logger().debug("OTLP HTTP log exporter not available")
    except Exception as e:
        _get_logger().warning("Failed to configure OTLP log exporter", error=str(e))
    return None


def _create_batch_processor_config() -> dict:
    """Get common batch processor configuration for console exporters."""
    return {
        "export_timeout_millis": 1000,  # 1 second timeout
        "max_export_batch_size": 64,
        "schedule_delay_millis": 500,
    }


def setup_tracing() -> None:
    """Configure OpenTelemetry tracing with Jaeger support."""
    global tracer, _trace_provider, _span_processors

    resource = _get_or_create_resource()
    sampler = TraceIdRatioBased(rate=settings.TRACE_SAMPLING_RATE)

    # Set up trace provider
    provider = TracerProvider(resource=resource, sampler=sampler)
    trace.set_tracer_provider(provider)
    _trace_provider = provider

    exporters_configured = 0

    # Configure OTLP exporter for Jaeger (if enabled)
    if settings.JAEGER_ENABLED:
        otlp_exporter = _create_otlp_span_exporter()
        if otlp_exporter:
            processor = BatchSpanProcessor(otlp_exporter)
            provider.add_span_processor(processor)
            _span_processors.append(processor)
            exporters_configured += 1
            _get_logger().info("OTLP trace exporter configured successfully")
        else:
            _get_logger().warning("No OTLP trace exporter could be configured")

    # Console exporter for development only
    if exporters_configured == 0 and settings.ENVIRONMENT.value == "development":
        try:
            console_exporter = SafeConsoleSpanExporter()
            processor = BatchSpanProcessor(console_exporter, **_create_batch_processor_config())
            provider.add_span_processor(processor)
            _span_processors.append(processor)
            _get_logger().info("Console trace exporter configured for development")
        except Exception as e:
            _get_logger().warning("Failed to configure console trace exporter", error=str(e))

    # Create global tracer
    tracer = trace.get_tracer(__name__)
    _get_logger().info(
        "OpenTelemetry tracing configured",
        sampling_rate=settings.TRACE_SAMPLING_RATE,
        exporters_count=len(_span_processors),
    )


def setup_metrics() -> None:
    """Configure OpenTelemetry metrics with Prometheus in multiprocess mode."""
    global _meter_provider

    if not settings.ENABLE_METRICS:
        return

    try:
        _get_logger().info("Prometheus multiprocess mode configured", multiproc_dir=_PROMETHEUS_MULTIPROC_DIR)

        resource = _get_or_create_resource()

        # Set up Prometheus metric reader with multiprocess registry
        reader = PrometheusMetricReader()
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        set_meter_provider(provider)
        _meter_provider = provider

        _get_logger().info("Prometheus metrics configured in multiprocess mode")
    except Exception as e:
        _get_logger().error("Failed to configure Prometheus metrics", error=str(e))


def setup_logging() -> None:
    """Configure OpenTelemetry logging with OTLP support."""
    global _logger_provider, _log_processors, _logging_handler

    if not settings.JAEGER_LOGS_ENABLED:
        return

    resource = _get_or_create_resource()

    # Set up logger provider
    provider = LoggerProvider(resource=resource)
    set_logger_provider(provider)
    _logger_provider = provider

    exporters_configured = 0

    # Configure OTLP exporter for logs (if Jaeger is enabled)
    if settings.JAEGER_ENABLED:
        otlp_log_exporter = _create_otlp_log_exporter()
        if otlp_log_exporter:
            processor = BatchLogRecordProcessor(otlp_log_exporter)
            provider.add_log_record_processor(processor)
            _log_processors.append(processor)
            exporters_configured += 1
            _get_logger().info("OTLP log exporter configured successfully")
        else:
            _get_logger().warning("No OTLP log exporter could be configured")

    # Console exporter for development only
    if settings.ENVIRONMENT.value == "development":
        try:
            console_log_exporter = SafeConsoleLogExporter()
            processor = BatchLogRecordProcessor(console_log_exporter, **_create_batch_processor_config())
            provider.add_log_record_processor(processor)
            _log_processors.append(processor)
            exporters_configured += 1
            _get_logger().info("Console log exporter configured for development")
        except Exception as e:
            _get_logger().warning("Failed to configure console log exporter", error=str(e))

    # Create logging handler that can be used by Python logging
    _logging_handler = LoggingHandler(logger_provider=provider)

    _get_logger().info(
        "OpenTelemetry logging configured",
        exporters_count=len(_log_processors),
        logs_enabled=settings.JAEGER_LOGS_ENABLED,
    )


def get_logging_handler() -> LoggingHandler | None:
    """Get the OpenTelemetry logging handler."""
    return _logging_handler


def get_logger_provider():
    """Get the OpenTelemetry logger provider."""
    return _logger_provider


def get_metrics_registry() -> CollectorRegistry:
    """Get the Prometheus collector registry for multiprocess mode.

    This aggregates metrics from all worker processes.
    Cached per worker for efficiency.
    """
    global _metrics_registry

    if _metrics_registry is None:
        from prometheus_client import multiprocess

        _metrics_registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(_metrics_registry)

    return _metrics_registry


def instrument_app(app: Any) -> None:
    """Instrument FastAPI application with OpenTelemetry.

    Note: In multi-worker mode, this instruments once per worker.
    The instrumentors are global singletons per Python process.
    """
    try:
        # FastAPI instrumentation
        if not FastAPIInstrumentor().is_instrumented_by_opentelemetry:
            FastAPIInstrumentor.instrument_app(
                app,
                tracer_provider=trace.get_tracer_provider(),
                excluded_urls="/health,/metrics",
            )
            _get_logger().info("FastAPI instrumentation enabled")
        else:
            _get_logger().debug("FastAPI already instrumented, skipping")

        # SQLAlchemy instrumentation
        sqlalchemy_instrumentor = SQLAlchemyInstrumentor()
        if not sqlalchemy_instrumentor.is_instrumented_by_opentelemetry:
            sqlalchemy_instrumentor.instrument()
            _get_logger().info("SQLAlchemy instrumentation enabled")
        else:
            _get_logger().debug("SQLAlchemy already instrumented, skipping")

    except Exception as e:
        _get_logger().error("Failed to instrument application", error=str(e))


def get_tracer() -> trace.Tracer:
    """Get the configured tracer instance."""
    if tracer is None:
        setup_tracing()
    return tracer or trace.get_tracer(__name__)


def cleanup_metrics_files() -> None:
    """Clean up Prometheus multiprocess metrics files.

    Should be called before starting workers to ensure clean state.
    Note: The startup script handles this - this is for manual cleanup if needed.
    """
    try:
        from prometheus_client import multiprocess

        if os.path.exists(_PROMETHEUS_MULTIPROC_DIR):
            multiprocess.mark_process_dead(os.getpid())
            _get_logger().info("Cleaned up metrics files", multiproc_dir=_PROMETHEUS_MULTIPROC_DIR)
    except Exception as e:
        _get_logger().warning("Failed to clean up metrics files", error=str(e))


def shutdown_observability() -> None:
    """Shutdown all observability components gracefully."""
    global _trace_provider, _meter_provider, _logger_provider, _logging_handler

    _get_logger().info("Shutting down observability components")

    # Shutdown log processors
    for processor in _log_processors:
        try:
            processor.force_flush(timeout_millis=1000)
            processor.shutdown()
            _get_logger().debug("Log processor shutdown completed")
        except Exception as e:
            _get_logger().warning("Error shutting down log processor", error=str(e))

    # Shutdown span processors
    for processor in _span_processors:
        try:
            processor.force_flush(timeout_millis=1000)
            processor.shutdown()
            _get_logger().debug("Span processor shutdown completed")
        except Exception as e:
            _get_logger().warning("Error shutting down span processor", error=str(e))

    # Clean up multiprocess metrics files for this worker
    if settings.ENABLE_METRICS:
        cleanup_metrics_files()

    # Clear processors lists
    _log_processors.clear()
    _span_processors.clear()

    # Reset global providers
    _trace_provider = None
    _meter_provider = None
    _logger_provider = None
    _logging_handler = None

    _get_logger().info("Observability components shutdown completed")


def init_observability() -> None:
    """Initialize all observability components."""
    try:
        # Validate configuration first
        _validate_config()

        # Initialize components
        setup_tracing()
        setup_logging()
        setup_metrics()

        # Configure logging integration after OpenTelemetry is set up
        from core.logging import configure_otel_logging

        configure_otel_logging()

        _get_logger().info("Observability initialized successfully")
    except Exception as e:
        _get_logger().error("Failed to initialize observability", error=str(e))
        # Don't let observability failures prevent app startup
        _get_logger().warning("Application will continue without full observability")
