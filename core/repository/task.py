"""Repository for ProfilingTask CRUD operations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.models.task import ProfilingTask, TaskStatus
from core.repository.base import BaseRepository


class TaskRepository(BaseRepository[ProfilingTask, dict, dict]):
    """Repository for profiling task persistence."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(ProfilingTask, session)

    async def count_running_tasks(self) -> int:
        """Count tasks that are currently running (not just pending)."""
        running_statuses = [
            TaskStatus.CONNECTING.value,
            TaskStatus.PROFILING.value,
            TaskStatus.AUGMENTING.value,
        ]
        return await self.count(filters=[ProfilingTask.status.in_(running_statuses)])

    async def cleanup_stale_tasks(self, retention_hours: int = 24) -> int:
        """Delete completed/failed/cancelled tasks older than retention period."""
        cutoff = datetime.now(UTC) - timedelta(hours=retention_hours)
        terminal_statuses = [
            TaskStatus.COMPLETED.value,
            TaskStatus.FAILED.value,
            TaskStatus.CANCELLED.value,
        ]
        stmt = delete(ProfilingTask).where(ProfilingTask.status.in_(terminal_statuses)).where(ProfilingTask.created_at < cutoff)
        result = await self.session.execute(stmt)
        return result.rowcount

    async def mark_stale_as_failed(self) -> int:
        """Mark any non-terminal tasks as FAILED (for server restart recovery)."""
        active_statuses = [
            TaskStatus.PENDING.value,
            TaskStatus.CONNECTING.value,
            TaskStatus.PROFILING.value,
            TaskStatus.AUGMENTING.value,
            TaskStatus.CANCELLING.value,
        ]
        stmt = (
            update(ProfilingTask)
            .where(ProfilingTask.status.in_(active_statuses))
            .values(
                status=TaskStatus.FAILED.value,
                error_code="SERVER_RESTART",
                error_message="Task was interrupted by server restart",
                completed_at=datetime.now(UTC),
            )
        )
        result = await self.session.execute(stmt)
        return result.rowcount
