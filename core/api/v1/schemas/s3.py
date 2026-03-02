"""S3 file profiling request schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr

from core.config import settings


def _default_profiling_config() -> Any:
    """Return a default ProfilingConfig. Lazy import avoids circular dependency."""
    from core.api.v1.schemas.profiler import ProfilingConfig

    return ProfilingConfig()


class S3ConnectionConfig(BaseModel):
    """AWS S3 (or S3-compatible) connection credentials and endpoint configuration."""

    model_config = {"frozen": True}

    aws_access_key_id: str | None = Field(
        default=settings.AWS_ACCESS_KEY_ID,
        description="AWS access key ID. Omit to use IAM role or environment variable credentials.",
    )
    aws_secret_access_key: SecretStr | None = Field(
        default=settings.AWS_SECRET_ACCESS_KEY,
        validate_default=True,
        description="AWS secret access key.",
    )
    aws_session_token: SecretStr | None = Field(
        default=settings.AWS_SESSION_TOKEN,
        validate_default=True,
        description="Temporary session token for assumed roles or SSO.",
    )
    aws_region: str = Field(
        default=settings.AWS_REGION,
        description="AWS region for the S3 bucket.",
    )
    endpoint_url: str | None = Field(
        default=None,
        description=(
            "Custom endpoint URL for S3-compatible storage (e.g. MinIO, LocalStack). "
            "Include scheme: http://host:port or https://host:port."
        ),
    )
    use_ssl: bool = Field(
        default=True,
        description="Use HTTPS when connecting to the endpoint. Set False for local MinIO/LocalStack over HTTP.",
    )


class S3PathConfig(BaseModel):
    """A single S3 path or glob pattern to profile as one logical table."""

    bucket: str = Field(..., description="S3 bucket name.")
    key: str = Field(
        ...,
        description=(
            "Object key or glob pattern within the bucket. "
            "Examples: 'data/orders.csv', 'data/2024/*.csv', 'warehouse/events/**/*.parquet'."
        ),
    )
    name: str | None = Field(
        default=None,
        description=(
            "Logical table name for this path in the profiling response. "
            "Defaults to the final key segment with common file extensions stripped."
        ),
    )
    file_format: Literal["auto", "csv", "parquet", "json"] = Field(
        default="auto",
        description=(
            "'auto' infers format from the file extension (falls back to CSV for unknown). "
            "Use 'csv', 'parquet', or 'json' to override detection."
        ),
    )
    delimiter: str | None = Field(
        default=None,
        description="CSV column delimiter character. None means DuckDB auto-detects (recommended).",
    )
    has_header: bool = Field(
        default=True,
        description="Whether the CSV file includes a header row.",
    )


class S3FileProfilingRequest(BaseModel):
    """Profiling request targeting one or more S3 files or glob patterns via DuckDB."""

    datasource_type: Literal["s3_file"] = Field(..., description="Datasource type discriminator.")
    connection: S3ConnectionConfig = Field(
        default_factory=S3ConnectionConfig,  # type: ignore[call-arg]
        description="AWS credentials and endpoint configuration. All credential fields are optional (supports IAM roles).",
    )
    paths: list[S3PathConfig] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="One or more S3 paths to profile. Each path (or glob pattern) is profiled as a separate logical table.",
    )
    # config is typed as Any to avoid a circular import with profiler.py.
    # At runtime it is always a ProfilingConfig instance.
    config: Any = Field(
        default_factory=_default_profiling_config,
        description="Profiling behaviour flags (column stats, LLM augmentation, etc.). Defaults to ProfilingConfig().",
    )
