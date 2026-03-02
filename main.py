import uvicorn

from core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "core.server:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.ENVIRONMENT == "development" else "warning",
    )
