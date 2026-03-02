"""Task test configuration and fixtures."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def disable_rate_limit():
    """Disable slowapi rate limiting for all task tests."""
    with patch("core.api.v1.routes.tasks.limiter.enabled", False):
        yield
