"""Profiler test configuration and fixtures."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def disable_rate_limit():
    """Disable slowapi rate limiting for all profiler tests.

    The session-scoped ``async_client`` shares request count across the
    entire test session. With a ``5/minute`` limit, validation and error
    tests consume the budget before success tests can run. Patching
    ``_enabled`` prevents the limiter from enforcing limits while still
    exercising the full route stack.
    """
    with patch("core.api.v1.routes.tasks.limiter.enabled", False):
        yield
