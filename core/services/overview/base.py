"""BaseOverviewService — abstract base for lightweight metadata overview."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from ignite_data_connectors import (
    ConfigurationError as ConnectorConfigError,
)
from ignite_data_connectors import (
    ConnectorConnectionError,
    QueryError,
    create_connector,
)

from core.api.v1.schemas.overview import DatabaseOverview, OverviewConfig
from core.exceptions import ConfigurationError, DatabaseError, ExternalServiceError
from core.exceptions.base import InternalTimeoutError, ProfilingTimeoutError
from core.logging import get_logger
from core.observability import get_tracer

logger = get_logger(__name__)
tracer = get_tracer()


class BaseOverviewService(ABC):
    """Abstract base for all datasource overview services.

    Subclasses must declare :attr:`service_name` and :attr:`span_name` as
    class-level strings and implement the three abstract methods below.
    """

    service_name: str
    span_name: str

    @abstractmethod
    def _span_attributes(self, conn: Any) -> dict[str, str | int]: ...

    @abstractmethod
    def _log_context(self, conn: Any) -> dict[str, Any]: ...

    @abstractmethod
    async def _fetch_overview(
        self,
        db: Any,
        cfg: OverviewConfig,
        conn: Any,
    ) -> DatabaseOverview: ...

    async def overview(self, body: Any) -> DatabaseOverview:
        """Run the overview flow: connect, fetch counts, return."""
        conn = body.connection
        cfg = body.config
        ctx = self._log_context(conn)
        t_start = time.monotonic()

        with tracer.start_as_current_span(self.span_name) as span:
            for key, value in self._span_attributes(conn).items():
                span.set_attribute(key, value)

            try:
                async with create_connector(conn) as db:
                    await db.test_connection()
                    logger.info("Overview: connection test passed", **ctx)

                    try:
                        result = await asyncio.wait_for(
                            self._fetch_overview(db, cfg, conn),
                            timeout=cfg.timeout_seconds,
                        )
                    except TimeoutError as exc:
                        elapsed = time.monotonic() - t_start
                        if elapsed >= cfg.timeout_seconds * 0.95:
                            raise
                        raise InternalTimeoutError(
                            message=(
                                f"An internal timeout occurred after {elapsed:.0f}s "
                                f"(overall limit is {cfg.timeout_seconds}s). "
                                "This usually means the connection pool is exhausted."
                            ),
                            source="pool_or_query",
                        ) from exc

            except InternalTimeoutError:
                raise

            except TimeoutError:
                logger.warning(
                    "Overview timed out",
                    **ctx,
                    timeout_seconds=cfg.timeout_seconds,
                )
                raise ProfilingTimeoutError(
                    message=f"Overview exceeded the configured timeout of {cfg.timeout_seconds}s",
                    timeout_seconds=cfg.timeout_seconds,
                ) from None

            except ConnectorConfigError as exc:
                logger.error("Configuration error during overview", **ctx, error=str(exc))
                raise ConfigurationError(
                    message=f"Invalid connection configuration: {exc}",
                ) from exc

            except ConnectorConnectionError as exc:
                host = ctx.get("host", "unknown")
                database = ctx.get("database", "unknown")
                logger.error("Connection error during overview", **ctx, error=str(exc))
                raise ExternalServiceError(
                    message=f"Cannot connect to {self.service_name} at {host}/{database}: {exc}",
                    service_name=self.service_name,
                ) from exc

            except QueryError as exc:
                logger.error("Query error during overview", **ctx, error=str(exc))
                raise DatabaseError(
                    message=f"SQL error during overview: {exc}",
                    operation="overview_query",
                ) from exc

        duration_ms = int((time.monotonic() - t_start) * 1000)
        result.duration_ms = duration_ms
        result.profiled_at = datetime.now(UTC)

        logger.info(
            "Overview complete",
            **ctx,
            total_schemas=result.total_schemas,
            total_tables=result.total_tables,
            duration_ms=duration_ms,
        )
        return result
