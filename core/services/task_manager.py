"""Background task manager for profiling operations."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

from core.config import settings
from core.database.session import get_session_manager
from core.exceptions.base import TaskLimitError
from core.logging import get_logger
from core.models.task import ProfilingTask, TaskStatus
from core.observability import get_tracer
from core.repository.task import TaskRepository

logger = get_logger(__name__)
tracer = get_tracer()


class ProgressReporter:
    """Reports progress from profiler to in-memory state + DB.

    Writes to DB are debounced to at most once per ``flush_interval`` seconds.
    """

    def __init__(self, task_id: str, flush_interval: float = 2.0) -> None:
        self.task_id = task_id
        self.phase: str = TaskStatus.PENDING.value
        self.percent: int | None = None
        self.detail: dict[str, Any] = {}
        self._flush_interval = flush_interval
        self._last_flush: float = 0.0
        self._dirty = False

    def update(
        self,
        phase: str | None = None,
        percent: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Update in-memory progress state."""
        if phase is not None:
            self.phase = phase
        if percent is not None:
            self.percent = percent
        if detail is not None:
            self.detail.update(detail)
        self._dirty = True

    async def flush_if_needed(self) -> None:
        """Write current progress to DB if debounce interval has elapsed."""
        now = time.monotonic()
        if not self._dirty or (now - self._last_flush) < self._flush_interval:
            return
        await self._do_flush()

    async def flush(self) -> None:
        """Force-flush progress to DB regardless of debounce."""
        if self._dirty:
            await self._do_flush()

    async def _do_flush(self) -> None:
        try:
            manager = get_session_manager()
            async with manager.get_session() as session:
                repo = TaskRepository(session)
                await repo.update(
                    self.task_id,
                    {
                        "status": self.phase,
                        "progress_phase": self.phase,
                        "progress_detail": dict(self.detail),
                        "progress_percent": self.percent,
                        "updated_at": datetime.now(UTC),
                    },
                )
            self._last_flush = time.monotonic()
            self._dirty = False
        except Exception:
            logger.warning(
                "Failed to flush task progress to DB",
                task_id=self.task_id,
                exc_info=True,
            )


class TaskManager:
    """Manages background profiling tasks.

    Singleton — import ``task_manager`` from this module.
    """

    def __init__(self) -> None:
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._reporters: dict[str, ProgressReporter] = {}
        self._semaphore: asyncio.Semaphore | None = None
        self._cleanup_task: asyncio.Task[None] | None = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_PROFILE_TASKS)
        return self._semaphore

    @property
    def active_count(self) -> int:
        return len(self._running_tasks)

    # ── Lifecycle ───────────────────────────────────────────────────────────────

    async def startup(self) -> None:
        """Called during app startup to recover from previous instance."""
        manager = get_session_manager()
        async with manager.get_session() as session:
            repo = TaskRepository(session)
            count = await repo.mark_stale_as_failed()
            if count:
                logger.info("Marked stale tasks as failed on startup", count=count)

        self._cleanup_task = asyncio.create_task(self._cleanup_loop(), name="task-cleanup")

    async def shutdown(self) -> None:
        """Graceful shutdown: cancel all running tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()

        for task_id, task in list(self._running_tasks.items()):
            logger.info("Cancelling task on shutdown", task_id=task_id)
            task.cancel()

        if self._running_tasks:
            await asyncio.gather(
                *self._running_tasks.values(),
                return_exceptions=True,
            )

    # ── Public API ──────────────────────────────────────────────────────────────

    async def create_task(self, body: Any) -> ProfilingTask:
        """Create a new profiling task and launch it in the background."""
        if self.active_count >= settings.MAX_CONCURRENT_PROFILE_TASKS:
            raise TaskLimitError(
                message=(
                    f"Maximum concurrent profiling tasks reached "
                    f"({settings.MAX_CONCURRENT_PROFILE_TASKS}). "
                    f"Please wait for a running task to complete."
                ),
                max_tasks=settings.MAX_CONCURRENT_PROFILE_TASKS,
            )

        # Persist task record
        manager = get_session_manager()
        async with manager.get_session() as session:
            repo = TaskRepository(session)
            task_record = await repo.create(
                {
                    "status": TaskStatus.PENDING.value,
                    "datasource_type": body.datasource_type,
                    "request_json": body.model_dump(mode="json"),
                    "progress_phase": TaskStatus.PENDING.value,
                }
            )
            # Capture fields before session closes
            task_id = task_record.id
            created_at = task_record.created_at

        # Create progress reporter and launch background task
        reporter = ProgressReporter(task_id)
        self._reporters[task_id] = reporter

        bg_task = asyncio.create_task(
            self._execute_profiling(task_id, body, reporter),
            name=f"profile-{task_id}",
        )
        self._running_tasks[task_id] = bg_task
        bg_task.add_done_callback(lambda _t: self._on_task_done(task_id))

        # Return a detached-but-populated object for the response
        result = ProfilingTask()
        result.id = task_id
        result.status = TaskStatus.PENDING.value
        result.datasource_type = body.datasource_type
        result.created_at = created_at
        return result

    async def get_task_status(self, task_id: str) -> ProfilingTask | None:
        """Get current task status, overlaying in-memory progress if active."""
        manager = get_session_manager()
        async with manager.get_session() as session:
            repo = TaskRepository(session)
            task = await repo.get(task_id)

        if task is None:
            return None

        # Overlay in-memory progress (more current than DB)
        reporter = self._reporters.get(task_id)
        if reporter:
            task.progress_phase = reporter.phase
            task.progress_detail = dict(reporter.detail)
            task.progress_percent = reporter.percent

        return task

    async def list_tasks(self, skip: int = 0, limit: int = 100) -> tuple[list[ProfilingTask], int]:
        """List tasks with pagination. Returns (tasks, total_count)."""
        manager = get_session_manager()
        async with manager.get_session() as session:
            repo = TaskRepository(session)
            tasks = list(
                await repo.get_multi(
                    skip=skip,
                    limit=limit,
                    order_by=ProfilingTask.created_at.desc(),
                )
            )
            total = await repo.count()
        return tasks, total

    async def cancel_task(self, task_id: str) -> bool:
        """Request cancellation of a running task."""
        bg_task = self._running_tasks.get(task_id)
        if bg_task is None:
            return False

        bg_task.cancel()

        manager = get_session_manager()
        async with manager.get_session() as session:
            repo = TaskRepository(session)
            await repo.update(
                task_id,
                {
                    "status": TaskStatus.CANCELLING.value,
                },
            )
        return True

    # ── Internal ────────────────────────────────────────────────────────────────

    def _on_task_done(self, task_id: str) -> None:
        """Cleanup callback when a background task finishes."""
        self._running_tasks.pop(task_id, None)
        self._reporters.pop(task_id, None)

    async def _execute_profiling(
        self,
        task_id: str,
        body: Any,
        reporter: ProgressReporter,
    ) -> None:
        """The background coroutine that runs profiling."""
        from core.services import PROFILER_REGISTRY

        async with self.semaphore:
            manager = get_session_manager()
            try:
                # Mark as started
                async with manager.get_session() as session:
                    repo = TaskRepository(session)
                    await repo.update(
                        task_id,
                        {
                            "status": TaskStatus.CONNECTING.value,
                            "started_at": datetime.now(UTC),
                        },
                    )

                reporter.update(phase=TaskStatus.CONNECTING.value, percent=0)
                await reporter.flush()

                # Get profiler
                profiler = PROFILER_REGISTRY.get(body.datasource_type)
                if profiler is None:
                    raise ValueError(f"Unsupported datasource: {body.datasource_type}")

                # Run profiling (passes reporter for progress updates)
                result = await profiler.profile(body, progress=reporter)

                # Store result
                async with manager.get_session() as session:
                    repo = TaskRepository(session)
                    await repo.update(
                        task_id,
                        {
                            "status": TaskStatus.COMPLETED.value,
                            "result_json": result.model_dump(mode="json"),
                            "completed_at": datetime.now(UTC),
                            "progress_phase": TaskStatus.COMPLETED.value,
                            "progress_percent": 100,
                        },
                    )

                logger.info("Profiling task completed", task_id=task_id)

            except asyncio.CancelledError:
                async with manager.get_session() as session:
                    repo = TaskRepository(session)
                    await repo.update(
                        task_id,
                        {
                            "status": TaskStatus.CANCELLED.value,
                            "completed_at": datetime.now(UTC),
                            "error_message": "Task was cancelled by user",
                        },
                    )
                logger.info("Profiling task cancelled", task_id=task_id)

            except Exception as exc:
                error_code = type(exc).__name__
                error_message = str(exc)
                async with manager.get_session() as session:
                    repo = TaskRepository(session)
                    await repo.update(
                        task_id,
                        {
                            "status": TaskStatus.FAILED.value,
                            "error_code": error_code,
                            "error_message": error_message,
                            "completed_at": datetime.now(UTC),
                            "progress_phase": TaskStatus.FAILED.value,
                        },
                    )
                logger.error(
                    "Profiling task failed",
                    task_id=task_id,
                    error=error_message,
                    exc_info=True,
                )

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of expired tasks."""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                manager = get_session_manager()
                async with manager.get_session() as session:
                    repo = TaskRepository(session)
                    deleted = await repo.cleanup_stale_tasks(retention_hours=settings.TASK_RETENTION_HOURS)
                    if deleted:
                        logger.info("Cleaned up expired tasks", deleted_count=deleted)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.warning("Task cleanup failed", exc_info=True)


# Global singleton
task_manager = TaskManager()
