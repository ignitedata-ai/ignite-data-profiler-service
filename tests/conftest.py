import asyncio
from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

from core.config import Environment, LogLevel, Settings
from core.server import create_app


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Create test settings."""
    test_settings = Settings(
        ENVIRONMENT=Environment.TESTING,
        DATABASE_URL="postgresql+asyncpg://postgres:password@localhost:5432/aipal_test",
        DEBUG=True,
        LOG_LEVEL=LogLevel.DEBUG,
        LOG_FORMAT="text",
        ENABLE_METRICS=False,
    )
    return test_settings


@pytest.fixture(scope="session")
def app():
    """Create test FastAPI application."""
    # Override settings for testing
    import core.config

    test_app_settings = Settings(
        ENVIRONMENT=Environment.TESTING,
        DATABASE_URL="postgresql+asyncpg://postgres:password@localhost:5432/aipal_test",
        DEBUG=True,
        LOG_LEVEL=LogLevel.DEBUG,
        LOG_FORMAT="text",
        ENABLE_METRICS=False,
        JWT_AUTH_ENABLED=False,
    )

    # Temporarily override global settings
    original_settings = core.config.settings
    core.config.settings = test_app_settings

    try:
        app = create_app()
        yield app
    finally:
        # Restore original settings
        core.config.settings = original_settings


@pytest.fixture(scope="session")
def client(app) -> Generator[TestClient, None, None]:
    """Create test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest_asyncio.fixture(scope="session")
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    from httpx import ASGITransport

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_test_client:
        yield async_test_client


@pytest.fixture(autouse=True)
def reset_database():
    """Reset database state between tests."""
    # TODO: Implement database reset logic when we have database models
    yield


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "email": "test@example.com",
        "username": "testuser",
        "full_name": "Test User",
        "is_active": True,
    }


@pytest.fixture
def auth_headers():
    """Sample auth headers for testing."""
    return {
        "Authorization": "Bearer test-token",
        "X-Correlation-ID": "test-correlation-id",
    }


@pytest.fixture
def cache_test_data():
    """Sample cache test data."""
    return {
        "user_profile": {
            "id": 123,
            "username": "testuser",
            "email": "test@example.com",
            "preferences": {"theme": "dark", "language": "en"},
        },
        "session_data": {
            "session_id": "abc123",
            "user_id": 123,
            "expires_at": "2024-12-31T23:59:59Z",
            "permissions": ["read", "write"],
        },
        "api_response": {"status": "success", "data": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}], "total": 2},
    }
