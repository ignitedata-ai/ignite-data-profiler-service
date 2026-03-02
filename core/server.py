from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Response
from fastapi.openapi.utils import get_openapi
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.exc import IntegrityError, OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.api.v1 import api_router
from core.config import settings
from core.database.session import initialize_database
from core.exceptions import (
    BaseException,
    database_exception_handler,
    database_operational_exception_handler,
    exception_handler,
    generic_exception_handler,
    http_exception_handler,
    starlette_http_exception_handler,
    validation_exception_handler,
)
from core.logging import configure_logging, get_logger
from core.middlewares import LoggingMiddleware, SecurityHeadersMiddleware, configure_cors
from core.observability import init_observability, instrument_app, shutdown_observability

# Configure logging before anything else
configure_logging()
logger = get_logger(__name__)


def custom_openapi(app: FastAPI):
    """Custom OpenAPI schema with JWT Bearer security."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add JWT Bearer security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "HTTPBearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter JWT token obtained from login endpoint",
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Backend Services", version=settings.APP_VERSION)

    try:
        # Initialize observability
        init_observability()

        # Instrument the app
        instrument_app(app)

        # Initialize database
        try:
            db_manager = initialize_database()
            await db_manager.initialize()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            # This is critical - we need the database to function
            raise

        # Initialize task manager
        try:
            from core.services.task_manager import task_manager

            await task_manager.startup()
            logger.info("Task manager initialized")
        except Exception as e:
            logger.error("Failed to initialize task manager", error=str(e))
            raise

        logger.info("Application startup completed")
        yield

    except Exception as e:
        logger.error("Failed to start application", error=str(e), exc_info=True)
        raise

    finally:
        # Shutdown
        logger.info("Shutting down Backend Services")

        # Shutdown task manager first (cancel running tasks)
        try:
            from core.services.task_manager import task_manager

            await task_manager.shutdown()
            logger.info("Task manager shut down")
        except Exception as e:
            logger.error("Error shutting down task manager", error=str(e))

        # Shutdown observability components
        try:
            shutdown_observability()
        except Exception as e:
            logger.error("Error shutting down observability", error=str(e))

        # Close database
        try:
            from core.database.session import session_manager

            if session_manager:
                await session_manager.close()
                logger.info("Database closed successfully")
        except Exception as e:
            logger.error("Error closing database", error=str(e))


def create_app() -> FastAPI:
    """Create FastAPI application with all configurations."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="AI-powered platform for data insights and analytics",
        lifespan=lifespan,
        debug=settings.DEBUG,
        docs_url="/profile/docs",
        redoc_url="/profile/redoc",
        openapi_url="/profile/openapi.json",
        swagger_ui_parameters={"persistAuthorization": True},
    )

    # Add CORS middleware (first for preflight handling)
    configure_cors(app)

    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)

    # Add logging middleware (last for complete request/response logging)
    app.add_middleware(LoggingMiddleware)

    # Register exception handlers
    app.add_exception_handler(BaseException, exception_handler)  # type: ignore
    app.add_exception_handler(PydanticValidationError, validation_exception_handler)  # type: ignore
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore
    app.add_exception_handler(StarletteHTTPException, starlette_http_exception_handler)  # type: ignore
    app.add_exception_handler(IntegrityError, database_exception_handler)  # type: ignore
    app.add_exception_handler(OperationalError, database_operational_exception_handler)  # type: ignore
    app.add_exception_handler(Exception, generic_exception_handler)

    # Health check endpoint
    @app.get("/profile/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": settings.OTEL_SERVICE_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT.value,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    # Prometheus metrics endpoint
    @app.get("/profile/metrics", tags=["Monitoring"])
    async def metrics_endpoint():
        """Prometheus metrics endpoint."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Include API routers
    app.include_router(api_router, prefix="/profile")

    logger.info("FastAPI application created")
    return app


# Create the app instance
app = create_app()
