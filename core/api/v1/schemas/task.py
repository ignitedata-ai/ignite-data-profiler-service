"""Task tracking Pydantic schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskStatusEnum(str, Enum):
    PENDING = "pending"
    CONNECTING = "connecting"
    PROFILING = "profiling"
    AUGMENTING = "augmenting"
    DETECTING_FILTERS = "detecting_filters"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class TaskProgress(BaseModel):
    """Granular progress information for a running task."""

    phase: TaskStatusEnum = Field(..., description="Current execution phase")
    percent: int | None = Field(None, ge=0, le=100, description="Estimated completion percentage")
    detail: dict[str, Any] | None = Field(None, description="Phase-specific progress details")


class TaskErrorDetail(BaseModel):
    """Error details for a failed task."""

    error_code: str
    message: str


class TaskCreateResponse(BaseModel):
    """Response returned when a task is created."""

    model_config = ConfigDict(from_attributes=True)

    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatusEnum = Field(..., description="Initial task status")
    created_at: datetime = Field(..., description="Task creation timestamp")
    status_url: str = Field(..., description="URL to poll for task status")


class TaskStatusResponse(BaseModel):
    """Full task status response."""

    model_config = ConfigDict(from_attributes=True)

    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatusEnum = Field(..., description="Current task status")
    datasource_type: str = Field(..., description="Datasource being profiled")
    progress: TaskProgress | None = Field(None, description="Current progress")
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = Field(None, description="Elapsed time in seconds")
    error: TaskErrorDetail | None = None
    result: dict[str, Any] | None = Field(
        None,
        description="Profiling result (only present when status=completed)",
    )


class TaskSummary(BaseModel):
    """Abbreviated task info for list views (no result payload)."""

    model_config = ConfigDict(from_attributes=True)

    task_id: str
    status: TaskStatusEnum
    datasource_type: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: TaskProgress | None = None
    error: TaskErrorDetail | None = None


class TaskListResponse(BaseModel):
    """Paginated list of tasks."""

    tasks: list[TaskSummary]
    total: int
    skip: int
    limit: int
