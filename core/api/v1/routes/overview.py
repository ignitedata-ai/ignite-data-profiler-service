"""Database overview route handler."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Request, Response

from core.api.v1.schemas.overview import DatabaseOverview, OverviewRequest
from core.logging import get_logger
from core.services.overview import OVERVIEW_REGISTRY
from core.utils.rate_limit import limiter

router = APIRouter(prefix="/profile", tags=["Overview"])
logger = get_logger(__name__)


@router.post(
    "/overview",
    response_model=DatabaseOverview,
    summary="Get database metadata counts",
    description=(
        "Returns a lightweight summary of database metadata: counts of schemas, "
        "tables, views, indexes, relationships, and columns. "
        "Much faster than full profiling — no data scanning, no LLM augmentation."
    ),
    status_code=200,
)
@limiter.limit("10/minute")
async def get_database_overview(
    request: Request,
    response: Response,
    body: Annotated[OverviewRequest, ...],
) -> DatabaseOverview:
    conn = body.connection
    logger.info(
        "Database overview requested",
        datasource_type=body.datasource_type,
        host=getattr(conn, "host", None)
        or getattr(conn, "account", None)
        or getattr(conn, "server_hostname", None)
        or getattr(conn, "project", None),
        database=getattr(conn, "database", None) or getattr(conn, "catalog", None) or getattr(conn, "project", None),
    )

    service = OVERVIEW_REGISTRY[body.datasource_type]
    return await service.overview(body)
