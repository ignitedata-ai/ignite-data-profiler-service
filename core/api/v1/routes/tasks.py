"""Task management route handlers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Query, Request, Response

from core.api.v1.schemas.profiler import ProfilingRequest
from core.api.v1.schemas.task import (
    TaskCreateResponse,
    TaskErrorDetail,
    TaskListResponse,
    TaskProgress,
    TaskStatusResponse,
    TaskSummary,
)
from core.exceptions import NotFoundError
from core.logging import get_logger
from core.models.task import TaskStatus
from core.services.task_manager import task_manager
from core.utils.rate_limit import limiter

router = APIRouter(prefix="/profile/tasks", tags=["Tasks"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=TaskCreateResponse,
    summary="Start async profiling task",
    description=(
        "Creates a background profiling task and returns immediately with a task ID. Poll the status endpoint to track progress."
    ),
    status_code=202,
)
@limiter.limit("5/minute")
async def create_profiling_task(
    request: Request,
    response: Response,
    body: Annotated[ProfilingRequest, ...],
) -> TaskCreateResponse:
    conn = body.connection
    logger.info(
        "Async profiling task requested",
        datasource_type=body.datasource_type,
        host=getattr(conn, "host", None)
        or getattr(conn, "account", None)
        or getattr(conn, "server_hostname", None)
        or getattr(conn, "project", None)
        or getattr(conn, "endpoint_url", None)
        or "s3",
        database=getattr(conn, "database", None)
        or getattr(conn, "catalog", None)
        or getattr(conn, "project", None)
        or (body.paths[0].bucket if hasattr(body, "paths") and body.paths else "s3"),  # type: ignore
    )

    task = await task_manager.create_task(body)
    return TaskCreateResponse(
        task_id=task.id,
        status=task.status,  # type: ignore
        created_at=task.created_at,
        status_url=f"/profile/v1/profile/tasks/{task.id}",
    )


@router.get(
    "/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get task status and result",
)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    task = await task_manager.get_task_status(task_id)
    if task is None:
        raise NotFoundError(
            message=f"Task {task_id} not found",
            resource_type="ProfilingTask",
            resource_id=task_id,
        )
    return _build_status_response(task)


@router.get(
    "",
    response_model=TaskListResponse,
    summary="List profiling tasks",
)
async def list_tasks(
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> TaskListResponse:
    tasks, total = await task_manager.list_tasks(skip=skip, limit=limit)
    return TaskListResponse(
        tasks=[_build_summary(t) for t in tasks],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.post(
    "/{task_id}/cancel",
    summary="Cancel a running task",
    status_code=202,
)
async def cancel_task(task_id: str) -> dict:
    cancelled = await task_manager.cancel_task(task_id)
    if not cancelled:
        raise NotFoundError(
            message=f"Task {task_id} is not currently running",
            resource_type="ProfilingTask",
            resource_id=task_id,
        )
    return {"task_id": task_id, "status": "cancelling"}


# ── Helpers ─────────────────────────────────────────────────────────────────────


def _build_progress(task) -> TaskProgress | None:
    if not task.progress_phase:
        return None
    return TaskProgress(
        phase=task.progress_phase,
        percent=task.progress_percent,
        detail=task.progress_detail,
    )


def _build_error(task) -> TaskErrorDetail | None:
    if not task.error_code and not task.error_message:
        return None
    return TaskErrorDetail(
        error_code=task.error_code or "UNKNOWN",
        message=task.error_message or "Unknown error",
    )


def _compute_duration(task) -> float | None:
    if not task.started_at:
        return None
    started = task.started_at
    end = task.completed_at or datetime.now(UTC)
    # Normalize: SQLite returns naive datetimes, but datetime.now(UTC) is aware
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    return round((end - started).total_seconds(), 3)


def _build_status_response(task) -> TaskStatusResponse:
    result = None
    if task.status == TaskStatus.COMPLETED.value and task.result_json:
        result = task.result_json

    return TaskStatusResponse(
        task_id=task.id,
        status=task.status,  # type: ignore
        datasource_type=task.datasource_type,
        progress=_build_progress(task),
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        duration_seconds=_compute_duration(task),
        error=_build_error(task),
        result=result,
    )


def _build_summary(task) -> TaskSummary:
    return TaskSummary(
        task_id=task.id,
        status=task.status,
        datasource_type=task.datasource_type,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        progress=_build_progress(task),
        error=_build_error(task),
    )
