from fastapi import APIRouter

from core.api.v1.routes.overview import router as overview_router
from core.api.v1.routes.tasks import router as tasks_router

api_router = APIRouter(prefix="/v1")

api_router.include_router(tasks_router)
api_router.include_router(overview_router)

__all__ = ["api_router"]
