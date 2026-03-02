"""Base Pydantic schemas and mixins."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TimestampMixin(BaseModel):
    """Mixin for created/updated timestamps."""

    model_config = ConfigDict(from_attributes=True)

    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime | None = Field(None, description="Last update timestamp")


class PaginationParams(BaseModel):
    """Standard pagination parameters."""

    skip: int = Field(default=0, ge=0, description="Number of records to skip")
    limit: int = Field(default=100, ge=1, le=1000, description="Number of records to return")


class SearchParams(PaginationParams):
    """Standard search parameters."""

    query: str = Field(default="", max_length=100, description="Search query")


class BulkOperationResult(BaseModel):
    """Result of a bulk operation."""

    total_requested: int = Field(..., description="Total number of items requested")
    total_processed: int = Field(..., description="Total number of items processed successfully")
    total_failed: int = Field(..., description="Total number of items that failed")
    errors: list[str] = Field(default=[], description="List of error messages for failed items")


class HealthCheck(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str | None = Field(None, description="Application version")
    environment: str | None = Field(None, description="Environment name")
    services: dict[str, str] | None = Field(None, description="Service health status")
