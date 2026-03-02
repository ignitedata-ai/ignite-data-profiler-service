"""Unit tests for task Pydantic schemas."""

from __future__ import annotations

from datetime import UTC, datetime

from core.api.v1.schemas.task import (
    TaskCreateResponse,
    TaskErrorDetail,
    TaskListResponse,
    TaskProgress,
    TaskStatusEnum,
    TaskStatusResponse,
    TaskSummary,
)


class TestTaskProgress:
    def test_valid_progress(self):
        p = TaskProgress(phase=TaskStatusEnum.PROFILING, percent=50, detail={"schemas_completed": 2})
        assert p.phase == TaskStatusEnum.PROFILING
        assert p.percent == 50
        assert p.detail["schemas_completed"] == 2

    def test_progress_with_none_percent(self):
        p = TaskProgress(phase=TaskStatusEnum.CONNECTING)
        assert p.percent is None
        assert p.detail is None

    def test_progress_percent_bounds(self):
        p = TaskProgress(phase=TaskStatusEnum.COMPLETED, percent=100)
        assert p.percent == 100

        p = TaskProgress(phase=TaskStatusEnum.PENDING, percent=0)
        assert p.percent == 0


class TestTaskCreateResponse:
    def test_serialization(self):
        now = datetime.now(UTC)
        resp = TaskCreateResponse(
            task_id="abc-123",
            status=TaskStatusEnum.PENDING,
            created_at=now,
            status_url="/api/v1/profile/tasks/abc-123",
        )
        data = resp.model_dump()
        assert data["task_id"] == "abc-123"
        assert data["status"] == "pending"
        assert data["status_url"] == "/api/v1/profile/tasks/abc-123"


class TestTaskStatusResponse:
    def test_completed_with_result(self):
        now = datetime.now(UTC)
        resp = TaskStatusResponse(
            task_id="abc-123",
            status=TaskStatusEnum.COMPLETED,
            datasource_type="postgres",
            created_at=now,
            started_at=now,
            completed_at=now,
            duration_seconds=12.5,
            result={"database": {"name": "testdb"}},
        )
        assert resp.result is not None
        assert resp.error is None

    def test_failed_with_error(self):
        now = datetime.now(UTC)
        resp = TaskStatusResponse(
            task_id="abc-123",
            status=TaskStatusEnum.FAILED,
            datasource_type="postgres",
            created_at=now,
            error=TaskErrorDetail(error_code="ConnectionError", message="Cannot connect"),
        )
        assert resp.error.error_code == "ConnectionError"
        assert resp.result is None

    def test_pending_minimal(self):
        now = datetime.now(UTC)
        resp = TaskStatusResponse(
            task_id="abc-123",
            status=TaskStatusEnum.PENDING,
            datasource_type="mysql",
            created_at=now,
        )
        assert resp.progress is None
        assert resp.started_at is None


class TestTaskListResponse:
    def test_empty_list(self):
        resp = TaskListResponse(tasks=[], total=0, skip=0, limit=20)
        assert len(resp.tasks) == 0

    def test_with_summaries(self):
        now = datetime.now(UTC)
        resp = TaskListResponse(
            tasks=[
                TaskSummary(
                    task_id="a",
                    status=TaskStatusEnum.COMPLETED,
                    datasource_type="postgres",
                    created_at=now,
                ),
                TaskSummary(
                    task_id="b",
                    status=TaskStatusEnum.PROFILING,
                    datasource_type="mysql",
                    created_at=now,
                    progress=TaskProgress(phase=TaskStatusEnum.PROFILING, percent=40),
                ),
            ],
            total=5,
            skip=0,
            limit=20,
        )
        assert len(resp.tasks) == 2
        assert resp.total == 5
