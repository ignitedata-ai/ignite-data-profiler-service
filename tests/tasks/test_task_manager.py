"""Unit tests for TaskManager and ProgressReporter."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.exceptions.base import TaskLimitError
from core.models.task import ProfilingTask, TaskStatus
from core.services.task_manager import ProgressReporter, TaskManager


class TestProgressReporter:
    def test_update_state(self):
        reporter = ProgressReporter("task-1")
        assert reporter.phase == TaskStatus.PENDING.value

        reporter.update(phase="profiling", percent=50, detail={"schemas_completed": 2})
        assert reporter.phase == "profiling"
        assert reporter.percent == 50
        assert reporter.detail["schemas_completed"] == 2

    def test_update_partial(self):
        reporter = ProgressReporter("task-1")
        reporter.update(phase="connecting")
        reporter.update(percent=10)
        reporter.update(detail={"step": "test_connection"})
        assert reporter.phase == "connecting"
        assert reporter.percent == 10
        assert reporter.detail["step"] == "test_connection"

    def test_detail_merges(self):
        reporter = ProgressReporter("task-1")
        reporter.update(detail={"a": 1})
        reporter.update(detail={"b": 2})
        assert reporter.detail == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_flush_writes_to_db(self):
        """flush() should update the task record in the DB."""
        reporter = ProgressReporter("task-1")
        reporter.update(phase="profiling", percent=50)

        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_mgr.get_session = MagicMock(return_value=mock_ctx)

        mock_repo = MagicMock()
        mock_repo.update = AsyncMock(return_value=MagicMock())

        with (
            patch("core.services.task_manager.get_session_manager", return_value=mock_mgr),
            patch("core.services.task_manager.TaskRepository", return_value=mock_repo),
        ):
            await reporter.flush()

        mock_repo.update.assert_called_once()
        call_args = mock_repo.update.call_args
        assert call_args[0][0] == "task-1"
        assert call_args[0][1]["status"] == "profiling"
        assert call_args[0][1]["progress_percent"] == 50

    @pytest.mark.asyncio
    async def test_flush_if_needed_debounces(self):
        """flush_if_needed should skip when interval hasn't elapsed."""
        reporter = ProgressReporter("task-1", flush_interval=10.0)
        reporter.update(phase="profiling")

        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_mgr.get_session = MagicMock(return_value=mock_ctx)

        mock_repo = MagicMock()
        mock_repo.update = AsyncMock(return_value=MagicMock())

        with (
            patch("core.services.task_manager.get_session_manager", return_value=mock_mgr),
            patch("core.services.task_manager.TaskRepository", return_value=mock_repo),
        ):
            # First flush should work
            await reporter.flush()
            assert mock_repo.update.call_count == 1

            # Immediate flush_if_needed should be debounced
            reporter.update(percent=60)
            await reporter.flush_if_needed()
            assert mock_repo.update.call_count == 1  # Still 1, debounced


class TestTaskManager:
    @pytest.fixture
    def manager(self):
        return TaskManager()

    @pytest.fixture
    def mock_body(self):
        body = MagicMock()
        body.datasource_type = "postgres"
        body.model_dump = MagicMock(
            return_value={
                "datasource_type": "postgres",
                "connection": {"host": "localhost"},
                "config": {},
            }
        )
        body.connection = MagicMock()
        body.config = MagicMock()
        body.config.timeout_seconds = 60
        return body

    @pytest.mark.asyncio
    async def test_concurrency_limit(self, manager, mock_body):
        """Creating tasks beyond the limit should raise TaskLimitError."""
        # Simulate max tasks running
        manager._running_tasks = {f"task-{i}": MagicMock() for i in range(3)}

        with pytest.raises(TaskLimitError):
            await manager.create_task(mock_body)

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_returns_false(self, manager):
        result = await manager.cancel_task("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_running_task(self, manager):
        """cancel_task should cancel the asyncio.Task and return True."""
        mock_bg_task = MagicMock()
        mock_bg_task.cancel = MagicMock()
        manager._running_tasks["task-1"] = mock_bg_task

        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_mgr.get_session = MagicMock(return_value=mock_ctx)

        mock_repo = MagicMock()
        mock_repo.update = AsyncMock(return_value=MagicMock())

        with (
            patch("core.services.task_manager.get_session_manager", return_value=mock_mgr),
            patch("core.services.task_manager.TaskRepository", return_value=mock_repo),
        ):
            result = await manager.cancel_task("task-1")

        assert result is True
        mock_bg_task.cancel.assert_called_once()

    def test_on_task_done_cleans_up(self, manager):
        manager._running_tasks["task-1"] = MagicMock()
        manager._reporters["task-1"] = MagicMock()

        manager._on_task_done("task-1")

        assert "task-1" not in manager._running_tasks
        assert "task-1" not in manager._reporters

    @pytest.mark.asyncio
    async def test_create_task_persists_and_launches(self, manager, mock_body):
        """create_task should persist a record and launch a background task."""
        now = datetime.now(UTC)
        mock_task = MagicMock(spec=ProfilingTask)
        mock_task.id = "new-task-id"
        mock_task.status = TaskStatus.PENDING.value
        mock_task.datasource_type = "postgres"
        mock_task.created_at = now

        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_mgr.get_session = MagicMock(return_value=mock_ctx)

        mock_repo = MagicMock()
        mock_repo.create = AsyncMock(return_value=mock_task)
        mock_repo.update = AsyncMock(return_value=mock_task)

        mock_profiler = MagicMock()
        mock_profiler.profile = AsyncMock(
            return_value=MagicMock(
                model_dump=MagicMock(return_value={"database": {"name": "testdb"}, "schemas": []}),
            )
        )

        with (
            patch("core.services.task_manager.get_session_manager", return_value=mock_mgr),
            patch("core.services.task_manager.TaskRepository", return_value=mock_repo),
            patch("core.services.PROFILER_REGISTRY", {"postgres": mock_profiler}),
        ):
            result = await manager.create_task(mock_body)

            assert result.id == "new-task-id"
            assert result.status == TaskStatus.PENDING.value
            assert "new-task-id" in manager._running_tasks
            assert "new-task-id" in manager._reporters

            # Wait for background task to finish
            await asyncio.sleep(0.3)

    @pytest.mark.asyncio
    async def test_get_task_status_returns_none_for_unknown(self, manager):
        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_mgr.get_session = MagicMock(return_value=mock_ctx)

        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=None)

        with (
            patch("core.services.task_manager.get_session_manager", return_value=mock_mgr),
            patch("core.services.task_manager.TaskRepository", return_value=mock_repo),
        ):
            result = await manager.get_task_status("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_task_status_overlays_in_memory_progress(self, manager):
        """When a task is active, in-memory progress should overlay DB data."""
        mock_task = MagicMock(spec=ProfilingTask)
        mock_task.progress_phase = "connecting"
        mock_task.progress_percent = 5

        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_mgr.get_session = MagicMock(return_value=mock_ctx)

        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=mock_task)

        # Set up an active reporter with fresher data
        reporter = ProgressReporter("active-task")
        reporter.update(phase="profiling", percent=45, detail={"schemas_completed": 3})
        manager._reporters["active-task"] = reporter

        with (
            patch("core.services.task_manager.get_session_manager", return_value=mock_mgr),
            patch("core.services.task_manager.TaskRepository", return_value=mock_repo),
        ):
            result = await manager.get_task_status("active-task")

        # Should have in-memory values, not DB values
        assert result.progress_phase == "profiling"
        assert result.progress_percent == 45
        assert result.progress_detail == {"schemas_completed": 3}

    @pytest.mark.asyncio
    async def test_startup_marks_stale_tasks(self, manager):
        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_mgr.get_session = MagicMock(return_value=mock_ctx)

        mock_repo = MagicMock()
        mock_repo.mark_stale_as_failed = AsyncMock(return_value=2)

        with (
            patch("core.services.task_manager.get_session_manager", return_value=mock_mgr),
            patch("core.services.task_manager.TaskRepository", return_value=mock_repo),
        ):
            await manager.startup()

        mock_repo.mark_stale_as_failed.assert_called_once()
        # Cleanup task should be running
        assert manager._cleanup_task is not None
        manager._cleanup_task.cancel()
