from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Ignite Data Profiler Service"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = Field(default=False)

    # JWT Authentication
    JWT_AUTH_ENABLED: bool = Field(default=True)

    # Server
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8004)

    # Database
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./profiler.db")
    DATABASE_ECHO: bool = Field(default=False)
    DATABASE_POOL_SIZE: int = Field(default=10)
    DATABASE_MAX_OVERFLOW: int = Field(default=5)
    DATABASE_POOL_TIMEOUT: int = Field(default=30)
    DATABASE_POOL_RECYCLE: int = Field(default=1800)
    DATABASE_POOL_PRE_PING: bool = Field(default=True)

    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True)

    # Logging
    LOG_LEVEL: LogLevel = LogLevel.INFO
    LOG_FORMAT: str = Field(default="json")  # json or text

    # OpenTelemetry
    OTEL_SERVICE_NAME: str = Field(default="ignite-data-profiler-service")
    OTEL_SERVICE_VERSION: str = Field(default="0.1.0")

    # Jaeger Configuration
    JAEGER_ENABLED: bool = Field(default=True)
    JAEGER_LOGS_ENABLED: bool = Field(default=False)  # Disabled: Jaeger all-in-one doesn't support OTLP logs properly
    JAEGER_AGENT_URL: str = Field(default="http://localhost:4318")
    TRACE_SAMPLING_RATE: float = Field(default=1.0, ge=0.0, le=1.0)

    # Prometheus
    ENABLE_METRICS: bool = Field(default=True)
    PROMETHEUS_MULTIPROC_DIR: str = Field(default="/tmp/prometheus_multiproc_dir")  # nosec B108

    # Security Headers
    SECURITY_HEADERS_ENABLED: bool = Field(default=True)
    X_FRAME_OPTIONS: str = Field(default="DENY")  # DENY, SAMEORIGIN, or ALLOW-FROM uri
    HSTS_ENABLED: bool = Field(default=True)
    HSTS_MAX_AGE: int = Field(default=31536000)  # 1 year
    HSTS_INCLUDE_SUBDOMAINS: bool = Field(default=True)
    HSTS_PRELOAD: bool = Field(default=False)
    REFERRER_POLICY: str = Field(default="strict-origin-when-cross-origin")
    CSP_ENABLED: bool = Field(default=True)
    CSP_DISABLE_IN_DEVELOPMENT: bool = Field(default=False)
    CONTENT_SECURITY_POLICY: str | None = Field(
        default=(
            "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; "
            "font-src 'self' https:; connect-src 'self' https:; media-src 'self'; "
            "object-src 'none'; child-src 'none'; worker-src 'none'; "
            "frame-ancestors 'none'; form-action 'self'; base-uri 'self';"
        )
    )
    PERMISSIONS_POLICY: str | None = Field(
        default="geolocation=(), microphone=(), camera=(), payment=(), usb=(), magnetometer=(), gyroscope=(), speaker=()"
    )
    REMOVE_SERVER_HEADER: bool = Field(default=True)

    # CORS Settings
    CORS_ALLOWED_ORIGINS: list[str] | None = Field(default=None)
    CORS_ALLOW_ALL_ORIGINS: bool = Field(default=False)
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: list[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
    CORS_ALLOW_HEADERS: list[str] = Field(default=["*"])
    CORS_EXPOSE_HEADERS: list[str] = Field(default=["X-Correlation-ID", "X-Trace-ID"])
    CORS_MAX_AGE: int = Field(default=86400)  # 24 hours

    # LLM Augmentation
    LLM_ENABLED: bool = Field(default=True)
    LLM_PROVIDER: str = Field(default="openai")
    LLM_OPENAI_API_KEY: str | None = Field(default=None)
    LLM_MODEL: str = Field(default="gpt-4o")
    LLM_TEMPERATURE: float = Field(default=0.2, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=500, ge=50, le=2000)
    LLM_REQUEST_TIMEOUT_SECONDS: float = Field(default=30.0, ge=1.0, le=120.0)

    # Portkey Gateway
    PORTKEY_API_KEY: str = Field(default="", description="Portkey API key")
    PORTKEY_VIRTUAL_KEY: str = Field(default="", description="Portkey virtual key for OpenAI routing")

    AWS_ACCESS_KEY_ID: str | None = Field(default=None)
    AWS_SECRET_ACCESS_KEY: str | None = Field(default=None)
    AWS_SESSION_TOKEN: str | None = Field(default=None)
    AWS_REGION: str = Field(default="us-east-1")

    # Task Management
    MAX_CONCURRENT_PROFILE_TASKS: int = Field(
        default=20,
        ge=1,
        le=30,
    )
    TASK_RETENTION_HOURS: int = Field(
        default=24,
        ge=1,
        le=720,
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True)


settings = Settings()
