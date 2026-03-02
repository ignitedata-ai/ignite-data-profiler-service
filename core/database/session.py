from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class DatabaseSessionManager:
    """Manages database sessions with connection pooling and observability."""

    def __init__(self, database_url: str, echo: bool = settings.DATABASE_ECHO) -> None:
        """Initialize the database session manager.

        Args:
            database_url: The database connection URL
            echo: Whether to echo SQL statements for debugging

        """
        self._database_url = database_url
        self._echo = echo
        self._engine = None
        self._session_factory = None
        self._instrumented = False

    async def initialize(self) -> None:
        """Initialize the database engine and session factory."""
        with tracer.start_as_current_span("database_initialize") as span:
            span.set_attribute("database.url", self._database_url.split("@")[0])  # Hide credentials
            span.set_attribute("database.echo", self._echo)

            try:
                # Create async engine with appropriate configuration
                engine_kwargs: dict[str, Any] = {
                    "echo": self._echo,
                    "future": True,
                }

                # Add connection pooling for non-SQLite databases
                if not self._database_url.startswith("sqlite"):
                    engine_kwargs.update(
                        {
                            "pool_size": settings.DATABASE_POOL_SIZE,  # Number of connections to maintain in pool
                            "max_overflow": settings.DATABASE_MAX_OVERFLOW,  # Additional connections beyond pool_size
                            "pool_timeout": settings.DATABASE_POOL_TIMEOUT,  # Seconds to wait for connection
                            "pool_recycle": settings.DATABASE_POOL_RECYCLE,  # Seconds after which connection is recreated
                            "pool_pre_ping": settings.DATABASE_POOL_PRE_PING,  # Validate connections before use
                        }
                    )

                self._engine = create_async_engine(self._database_url, **engine_kwargs)

                # Create session factory
                self._session_factory = async_sessionmaker(
                    bind=self._engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                    autoflush=False,
                    autocommit=False,
                )

                # Instrument SQLAlchemy for observability
                if not self._instrumented:
                    SQLAlchemyInstrumentor().instrument(
                        engine=self._engine.sync_engine,
                        service=settings.OTEL_SERVICE_NAME,
                        version=settings.OTEL_SERVICE_VERSION,
                    )
                    self._instrumented = True

                logger.info(
                    "Database session manager initialized",
                    database_url=self._database_url.split("@")[0],
                    pool_size=20,
                    max_overflow=30,
                )
                span.set_attribute("database.status", "initialized")

            except Exception as e:
                logger.error("Failed to initialize database session manager", error=str(e), exc_info=True)
                span.set_attribute("database.status", "failed")
                span.record_exception(e)
                raise

    async def close(self) -> None:
        """Close the database engine and cleanup resources."""
        with tracer.start_as_current_span("database_close") as span:
            try:
                if self._engine:
                    await self._engine.dispose()
                    logger.info("Database engine closed")
                    span.set_attribute("database.status", "closed")
                else:
                    logger.warning("Attempted to close database engine, but engine was None")
                    span.set_attribute("database.status", "already_closed")

            except Exception as e:
                logger.error("Error closing database engine", error=str(e), exc_info=True)
                span.set_attribute("database.status", "error_during_close")
                span.record_exception(e)
                raise

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session with proper error handling and cleanup.

        Yields:
            AsyncSession: A database session

        Raises:
            RuntimeError: If the session manager is not initialized

        """
        if not self._session_factory:
            raise RuntimeError("Database session manager not initialized. Call initialize() first.")

        session_id = id(asyncio.current_task())
        with tracer.start_as_current_span("database_session") as span:
            span.set_attribute("database.session_id", str(session_id))

            session = None
            try:
                # Create new session
                session = self._session_factory()
                logger.debug("Database session created", session_id=session_id)
                span.set_attribute("database.session_status", "created")

                yield session

                # Commit if no exceptions occurred
                await session.commit()
                logger.debug("Database session committed", session_id=session_id)
                span.set_attribute("database.session_status", "committed")

            except Exception as e:
                logger.error(
                    "Database session error, rolling back",
                    session_id=session_id,
                    error=str(e),
                    exc_info=True,
                )
                span.set_attribute("database.session_status", "error")
                span.record_exception(e)

                if session:
                    try:
                        await session.rollback()
                        logger.debug("Database session rolled back", session_id=session_id)
                        span.set_attribute("database.session_rollback", "success")
                    except Exception as rollback_error:
                        logger.error(
                            "Failed to rollback database session",
                            session_id=session_id,
                            rollback_error=str(rollback_error),
                            exc_info=True,
                        )
                        span.set_attribute("database.session_rollback", "failed")
                        span.record_exception(rollback_error)

                raise
            finally:
                if session:
                    try:
                        await session.close()
                        logger.debug("Database session closed", session_id=session_id)
                        span.set_attribute("database.session_cleanup", "success")
                    except Exception as close_error:
                        logger.error(
                            "Error closing database session",
                            session_id=session_id,
                            close_error=str(close_error),
                            exc_info=True,
                        )
                        span.set_attribute("database.session_cleanup", "failed")
                        span.record_exception(close_error)

    async def health_check(self) -> dict[str, Any]:
        """Perform a health check on the database connection.

        Returns:
            dict: Health check results including connection status and pool info

        """
        with tracer.start_as_current_span("database_health_check") as span:
            try:
                if not self._engine:
                    return {"status": "unhealthy", "error": "Database engine not initialized"}

                # Test connection
                async with self.get_session() as session:
                    from sqlalchemy import text

                    result = await session.execute(text("SELECT 1"))
                    result.scalar()

                # Get pool information
                pool = self._engine.pool
                pool_info = {
                    "size": getattr(pool, "size", lambda: 0)(),
                    "checked_in": getattr(pool, "checkedin", lambda: 0)(),
                    "checked_out": getattr(pool, "checkedout", lambda: 0)(),
                    "overflow": getattr(pool, "overflow", lambda: 0)(),
                    "total": getattr(pool, "size", lambda: 0)() + getattr(pool, "overflow", lambda: 0)(),
                }

                health_data = {
                    "status": "healthy",
                    "pool": pool_info,
                    "database_url": self._database_url.split("@")[0],  # Hide credentials
                }

                logger.debug("Database health check successful", **health_data)
                span.set_attribute("database.health_status", "healthy")
                span.set_attributes({f"database.pool.{k}": v for k, v in pool_info.items()})

                return health_data

            except Exception as e:
                error_data = {"status": "unhealthy", "error": str(e)}
                logger.error("Database health check failed", **error_data, exc_info=True)
                span.set_attribute("database.health_status", "unhealthy")
                span.record_exception(e)
                return error_data


# Global session manager instance
session_manager: DatabaseSessionManager | None = None


def get_session_manager() -> DatabaseSessionManager:
    """Get the global database session manager.

    Returns:
        DatabaseSessionManager: The global session manager instance

    Raises:
        RuntimeError: If the session manager is not initialized

    """
    if session_manager is None:
        raise RuntimeError("Database session manager not initialized")
    return session_manager


def initialize_database() -> DatabaseSessionManager:
    """Initialize the global database session manager.

    Returns:
        DatabaseSessionManager: The initialized session manager

    """
    global session_manager
    session_manager = DatabaseSessionManager(
        database_url=settings.DATABASE_URL,
        echo=settings.DATABASE_ECHO,
    )
    return session_manager


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency function to get a database session.

    This function is intended to be used as a FastAPI dependency.

    Yields:
        AsyncSession: A database session

    """
    manager = get_session_manager()
    async with manager.get_session() as session:
        yield session
