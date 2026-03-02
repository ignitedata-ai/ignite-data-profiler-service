"""Route-level integration tests for task management endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from core.exceptions.base import TaskLimitError
from core.models.task import ProfilingTask, TaskStatus

TASKS_ENDPOINT = "/profile/v1/profile/tasks"

_CONN = {
    "host": "localhost",
    "port": 5432,
    "database": "testdb",
    "username": "user",
    "password": "pass",
}

VALID_PAYLOAD = {
    "datasource_type": "postgres",
    "connection": _CONN,
    "config": {
        "include_schemas": ["public"],
        "sample_size": 5,
        "timeout_seconds": 60,
    },
}


def _make_task(
    task_id: str = "test-task-id",
    status: str = TaskStatus.PENDING.value,
    datasource_type: str = "postgres",
    **overrides,
) -> ProfilingTask:
    """Create a mock ProfilingTask with sensible defaults."""
    task = ProfilingTask()
    task.id = task_id
    task.status = status
    task.datasource_type = datasource_type
    task.created_at = overrides.get("created_at", datetime.now(UTC))
    task.started_at = overrides.get("started_at")
    task.completed_at = overrides.get("completed_at")
    task.progress_phase = overrides.get("progress_phase")
    task.progress_detail = overrides.get("progress_detail")
    task.progress_percent = overrides.get("progress_percent")
    task.result_json = overrides.get("result_json")
    task.error_code = overrides.get("error_code")
    task.error_message = overrides.get("error_message")
    task.request_json = overrides.get("request_json")
    task.updated_at = overrides.get("updated_at")
    return task


class TestCreateProfilingTask:
    @pytest.mark.asyncio
    async def test_create_task_returns_202(self, async_client: AsyncClient):
        """POST /profile/v1/profile/tasks should return 202 with task_id."""
        mock_task = _make_task()

        with patch(
            "core.api.v1.routes.tasks.task_manager.create_task",
            new_callable=AsyncMock,
            return_value=mock_task,
        ):
            response = await async_client.post(TASKS_ENDPOINT, json=VALID_PAYLOAD)

        assert response.status_code == 202
        data = response.json()
        assert data["task_id"] == "test-task-id"
        assert data["status"] == "pending"
        assert "/profile/v1/profile/tasks/test-task-id" == data["status_url"]

    @pytest.mark.asyncio
    async def test_create_task_missing_datasource_returns_422(self, async_client: AsyncClient):
        response = await async_client.post(
            TASKS_ENDPOINT,
            json={"connection": _CONN},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_create_task_invalid_datasource_returns_422(self, async_client: AsyncClient):
        response = await async_client.post(
            TASKS_ENDPOINT,
            json={"datasource_type": "oracle", "connection": _CONN},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_create_task_concurrency_limit_returns_429(self, async_client: AsyncClient):
        """When max concurrent tasks reached, should return 429."""
        with patch(
            "core.api.v1.routes.tasks.task_manager.create_task",
            new_callable=AsyncMock,
            side_effect=TaskLimitError(max_tasks=3),
        ):
            response = await async_client.post(TASKS_ENDPOINT, json=VALID_PAYLOAD)

        assert response.status_code == 429


class TestGetTaskStatus:
    @pytest.mark.asyncio
    async def test_get_unknown_task_returns_404(self, async_client: AsyncClient):
        with patch(
            "core.api.v1.routes.tasks.task_manager.get_task_status",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = await async_client.get(f"{TASKS_ENDPOINT}/nonexistent-id")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_pending_task(self, async_client: AsyncClient):
        mock_task = _make_task(
            progress_phase=TaskStatus.PENDING.value,
            progress_percent=0,
        )
        with patch(
            "core.api.v1.routes.tasks.task_manager.get_task_status",
            new_callable=AsyncMock,
            return_value=mock_task,
        ):
            response = await async_client.get(f"{TASKS_ENDPOINT}/test-task-id")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test-task-id"
        assert data["status"] == "pending"
        assert data["datasource_type"] == "postgres"

    @pytest.mark.asyncio
    async def test_get_profiling_task_with_progress(self, async_client: AsyncClient):
        now = datetime.now(UTC)
        mock_task = _make_task(
            status=TaskStatus.PROFILING.value,
            started_at=now,
            progress_phase=TaskStatus.PROFILING.value,
            progress_percent=45,
            progress_detail={"schemas_completed": 2, "schema_total": 5},
        )
        with patch(
            "core.api.v1.routes.tasks.task_manager.get_task_status",
            new_callable=AsyncMock,
            return_value=mock_task,
        ):
            response = await async_client.get(f"{TASKS_ENDPOINT}/test-task-id")

        data = response.json()
        assert data["status"] == "profiling"
        assert data["progress"]["phase"] == "profiling"
        assert data["progress"]["percent"] == 45
        assert data["progress"]["detail"]["schemas_completed"] == 2

    @pytest.mark.asyncio
    async def test_get_completed_task_with_result(self, async_client: AsyncClient):
        now = datetime.now(UTC)
        mock_task = _make_task(
            status=TaskStatus.COMPLETED.value,
            started_at=now,
            completed_at=now,
            progress_phase=TaskStatus.COMPLETED.value,
            progress_percent=100,
            result_json={"database": {"name": "testdb"}, "schemas": []},
        )
        with patch(
            "core.api.v1.routes.tasks.task_manager.get_task_status",
            new_callable=AsyncMock,
            return_value=mock_task,
        ):
            response = await async_client.get(f"{TASKS_ENDPOINT}/test-task-id")

        data = response.json()
        assert data["status"] == "completed"
        assert data["result"] is not None
        assert data["result"]["database"]["name"] == "testdb"
        assert data["duration_seconds"] is not None

    @pytest.mark.asyncio
    async def test_get_failed_task_with_error(self, async_client: AsyncClient):
        now = datetime.now(UTC)
        mock_task = _make_task(
            status=TaskStatus.FAILED.value,
            started_at=now,
            completed_at=now,
            error_code="ConnectionError",
            error_message="Cannot connect to database",
        )
        with patch(
            "core.api.v1.routes.tasks.task_manager.get_task_status",
            new_callable=AsyncMock,
            return_value=mock_task,
        ):
            response = await async_client.get(f"{TASKS_ENDPOINT}/test-task-id")

        data = response.json()
        assert data["status"] == "failed"
        assert data["error"]["error_code"] == "ConnectionError"
        assert data["error"]["message"] == "Cannot connect to database"
        assert data["result"] is None


class TestListTasks:
    @pytest.mark.asyncio
    async def test_list_tasks_returns_200(self, async_client: AsyncClient):
        with patch(
            "core.api.v1.routes.tasks.task_manager.list_tasks",
            new_callable=AsyncMock,
            return_value=([], 0),
        ):
            response = await async_client.get(TASKS_ENDPOINT)

        assert response.status_code == 200
        data = response.json()
        assert data["tasks"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_tasks_with_results(self, async_client: AsyncClient):
        tasks = [
            _make_task("id-1", TaskStatus.COMPLETED.value),
            _make_task("id-2", TaskStatus.PROFILING.value),
        ]
        with patch(
            "core.api.v1.routes.tasks.task_manager.list_tasks",
            new_callable=AsyncMock,
            return_value=(tasks, 5),
        ):
            response = await async_client.get(f"{TASKS_ENDPOINT}?skip=0&limit=2")

        data = response.json()
        assert len(data["tasks"]) == 2
        assert data["total"] == 5
        assert data["skip"] == 0
        assert data["limit"] == 2

    @pytest.mark.asyncio
    async def test_list_tasks_pagination_params(self, async_client: AsyncClient):
        with patch(
            "core.api.v1.routes.tasks.task_manager.list_tasks",
            new_callable=AsyncMock,
            return_value=([], 0),
        ) as mock_list:
            response = await async_client.get(f"{TASKS_ENDPOINT}?skip=10&limit=5")

        assert response.status_code == 200
        mock_list.assert_called_once_with(skip=10, limit=5)


class TestCancelTask:
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task_returns_404(self, async_client: AsyncClient):
        with patch(
            "core.api.v1.routes.tasks.task_manager.cancel_task",
            new_callable=AsyncMock,
            return_value=False,
        ):
            response = await async_client.post(f"{TASKS_ENDPOINT}/nonexistent-id/cancel")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_running_task_returns_202(self, async_client: AsyncClient):
        with patch(
            "core.api.v1.routes.tasks.task_manager.cancel_task",
            new_callable=AsyncMock,
            return_value=True,
        ):
            response = await async_client.post(f"{TASKS_ENDPOINT}/test-task-id/cancel")

        assert response.status_code == 202
        data = response.json()
        assert data["task_id"] == "test-task-id"
        assert data["status"] == "cancelling"
