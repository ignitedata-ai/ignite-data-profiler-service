from unittest.mock import patch

import pytest
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

from core.config import settings
from core.middlewares.security import SecurityHeadersMiddleware


def _configure_mock_settings(mock_settings, **overrides):
    """Configure mock settings with defaults and overrides."""
    defaults = {
        "SECURITY_HEADERS_ENABLED": True,
        "X_FRAME_OPTIONS": "DENY",
        "REFERRER_POLICY": "strict-origin-when-cross-origin",
        "ENVIRONMENT": type("Environment", (), {"value": "development"}),
        "HSTS_ENABLED": False,
        "HSTS_MAX_AGE": 31536000,
        "HSTS_INCLUDE_SUBDOMAINS": True,
        "HSTS_PRELOAD": False,
        "CSP_ENABLED": False,
        "CSP_DISABLE_IN_DEVELOPMENT": False,
        "CONTENT_SECURITY_POLICY": None,
        "PERMISSIONS_POLICY": None,
        "REMOVE_SERVER_HEADER": False,
    }

    # Apply overrides
    for key, value in overrides.items():
        defaults[key] = value

    # Set all attributes on mock
    for key, value in defaults.items():
        setattr(mock_settings, key, value)

    return mock_settings


@pytest.fixture
def app():
    """Create a test FastAPI app with security middleware."""
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestSecurityHeadersMiddleware:
    """Test security headers middleware functionality."""

    def test_security_headers_enabled(self, client):
        """Test that security headers are added when enabled."""
        response = client.get("/test")

        assert response.status_code == 200

        # Check basic security headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == settings.X_FRAME_OPTIONS
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert response.headers["Referrer-Policy"] == settings.REFERRER_POLICY
        assert response.headers["X-Permitted-Cross-Domain-Policies"] == "none"

    @patch("core.middlewares.security.settings")
    def test_security_headers_disabled(self, mock_settings, app):
        """Test that security headers are not added when disabled."""
        _configure_mock_settings(mock_settings, SECURITY_HEADERS_ENABLED=False)

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        assert "X-Content-Type-Options" not in response.headers
        assert "X-Frame-Options" not in response.headers

    @patch("core.middlewares.security.settings")
    def test_hsts_header_production(self, mock_settings, app):
        """Test HSTS header is added in production."""
        _configure_mock_settings(
            mock_settings,
            ENVIRONMENT=type("Environment", (), {"value": "production"}),
            HSTS_ENABLED=True,
            HSTS_MAX_AGE=31536000,
            HSTS_INCLUDE_SUBDOMAINS=True,
            HSTS_PRELOAD=False,
        )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        expected_hsts = "max-age=31536000; includeSubDomains"
        assert response.headers["Strict-Transport-Security"] == expected_hsts

    @patch("core.middlewares.security.settings")
    def test_hsts_header_with_preload(self, mock_settings, app):
        """Test HSTS header with preload enabled."""
        _configure_mock_settings(
            mock_settings,
            ENVIRONMENT=type("Environment", (), {"value": "production"}),
            HSTS_ENABLED=True,
            HSTS_MAX_AGE=31536000,
            HSTS_INCLUDE_SUBDOMAINS=True,
            HSTS_PRELOAD=True,
        )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        expected_hsts = "max-age=31536000; includeSubDomains; preload"
        assert response.headers["Strict-Transport-Security"] == expected_hsts

    @patch("core.middlewares.security.settings")
    def test_hsts_header_development(self, mock_settings, app):
        """Test HSTS header is not added in development."""
        _configure_mock_settings(mock_settings, ENVIRONMENT=type("Environment", (), {"value": "development"}), HSTS_ENABLED=True)

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        assert "Strict-Transport-Security" not in response.headers

    @patch("core.middlewares.security.settings")
    def test_csp_header_enabled(self, mock_settings, app):
        """Test CSP header is added when enabled."""
        _configure_mock_settings(mock_settings, CSP_ENABLED=True, CONTENT_SECURITY_POLICY="default-src 'self'")

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        assert response.headers["Content-Security-Policy"] == "default-src 'self'"

    @patch("core.middlewares.security.settings")
    def test_csp_header_disabled(self, mock_settings, app):
        """Test CSP header is not added when disabled."""
        _configure_mock_settings(mock_settings, CSP_ENABLED=False, CONTENT_SECURITY_POLICY="default-src 'self'")

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        assert "Content-Security-Policy" not in response.headers

    @patch("core.middlewares.security.settings")
    def test_permissions_policy_header(self, mock_settings, app):
        """Test Permissions-Policy header is added when configured."""
        _configure_mock_settings(mock_settings, PERMISSIONS_POLICY="geolocation=(), camera=()")

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        assert response.headers["Permissions-Policy"] == "geolocation=(), camera=()"

    @patch("core.middlewares.security.settings")
    def test_permissions_policy_header_none(self, mock_settings, app):
        """Test Permissions-Policy header is not added when None."""
        _configure_mock_settings(mock_settings, PERMISSIONS_POLICY=None)

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        assert "Permissions-Policy" not in response.headers

    def test_x_frame_options_values(self, app):
        """Test different X-Frame-Options values."""
        for value in ["DENY", "SAMEORIGIN"]:
            with patch("core.middlewares.security.settings") as mock_settings:
                _configure_mock_settings(mock_settings, X_FRAME_OPTIONS=value)

                client = TestClient(app)
                response = client.get("/test")

                assert response.status_code == 200
                assert response.headers["X-Frame-Options"] == value

    @patch("core.middlewares.security.settings")
    def test_server_header_removal(self, mock_settings, app):
        """Test server header removal functionality."""
        _configure_mock_settings(mock_settings, REMOVE_SERVER_HEADER=True)

        # Create app with custom response that includes Server header
        app_with_server = FastAPI()
        app_with_server.add_middleware(SecurityHeadersMiddleware)

        @app_with_server.get("/test")
        async def test_endpoint():
            response = Response(content='{"message": "test"}')
            response.headers["Server"] = "TestServer/1.0"
            return response

        client = TestClient(app_with_server)
        response = client.get("/test")

        assert response.status_code == 200
        assert "Server" not in response.headers

    @patch("core.middlewares.security.settings")
    def test_server_header_not_removed_when_disabled(self, mock_settings, app):
        """Test server header is not removed when disabled."""
        _configure_mock_settings(mock_settings, REMOVE_SERVER_HEADER=False)

        # Create app with custom response that includes Server header
        app_with_server = FastAPI()
        app_with_server.add_middleware(SecurityHeadersMiddleware)

        @app_with_server.get("/test")
        async def test_endpoint():
            response = Response(content='{"message": "test"}')
            response.headers["Server"] = "TestServer/1.0"
            return response

        client = TestClient(app_with_server)
        response = client.get("/test")

        assert response.status_code == 200
        assert response.headers["Server"] == "TestServer/1.0"

    def test_middleware_preserves_response_body(self, client):
        """Test that middleware doesn't affect response body."""
        response = client.get("/test")

        assert response.status_code == 200
        assert response.json() == {"message": "test"}

    def test_middleware_with_different_http_methods(self, app):
        """Test middleware works with different HTTP methods."""

        @app.post("/test-post")
        async def post_endpoint():
            return {"method": "post"}

        @app.put("/test-put")
        async def put_endpoint():
            return {"method": "put"}

        client = TestClient(app)

        for method, endpoint in [("post", "/test-post"), ("put", "/test-put")]:
            response = getattr(client, method)(endpoint)

            assert response.status_code == 200
            assert "X-Content-Type-Options" in response.headers
            assert "X-Frame-Options" in response.headers
