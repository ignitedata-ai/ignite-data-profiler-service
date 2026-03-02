"""ProfilingTask database model for background task tracking."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from core.database.session import Base


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    CONNECTING = "connecting"
    PROFILING = "profiling"
    AUGMENTING = "augmenting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class ProfilingTask(Base):
    """Persistent record for a background profiling task."""

    __tablename__ = "profiling_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=TaskStatus.PENDING.value, index=True)
    datasource_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Original request body for auditability
    request_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Progress tracking
    progress_phase: Mapped[str | None] = mapped_column(String(50), nullable=True)
    progress_detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    progress_percent: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Result (full ProfilingResponse as JSON)
    result_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Error info
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, onupdate=func.now())

    __table_args__ = (Index("ix_profiling_tasks_status_created", "status", "created_at"),)
