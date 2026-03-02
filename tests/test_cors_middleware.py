from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.middlewares.cors import _get_allowed_origins, _is_valid_origin, configure_cors


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    @app.options("/test")
    async def test_options():
        return {"message": "options"}

    return app


class TestCORSConfiguration:
    """Test CORS middleware configuration functionality."""

    @patch("core.middlewares.cors.settings")
    def test_configure_cors_development(self, mock_settings, app):
        """Test CORS configuration in development environment."""
        mock_settings.ENVIRONMENT.value = "development"
        mock_settings.CORS_ALLOWED_ORIGINS = ["http://localhost:3000"]
        mock_settings.CORS_ALLOW_ALL_ORIGINS = False
        mock_settings.CORS_ALLOW_CREDENTIALS = True
        mock_settings.CORS_ALLOW_METHODS = ["GET", "POST"]
        mock_settings.CORS_ALLOW_HEADERS = ["*"]
        mock_settings.CORS_EXPOSE_HEADERS = ["X-Correlation-ID"]
        mock_settings.CORS_MAX_AGE = 3600

        configure_cors(app)
        client = TestClient(app)

        # Test preflight request
        response = client.options("/test", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"})

        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers

    @patch("core.middlewares.cors.settings")
    def test_configure_cors_production(self, mock_settings, app):
        """Test CORS configuration in production environment."""
        mock_settings.ENVIRONMENT.value = "production"
        mock_settings.CORS_ALLOWED_ORIGINS = ["https://example.com", "https://app.example.com"]
        mock_settings.CORS_ALLOW_CREDENTIALS = True
        mock_settings.CORS_ALLOW_METHODS = ["GET", "POST", "PUT", "DELETE"]
        mock_settings.CORS_ALLOW_HEADERS = ["Content-Type", "Authorization"]
        mock_settings.CORS_EXPOSE_HEADERS = ["X-Correlation-ID", "X-Trace-ID"]
        mock_settings.CORS_MAX_AGE = 86400

        configure_cors(app)
        client = TestClient(app)

        # Test allowed origin
        response = client.options("/test", headers={"Origin": "https://example.com", "Access-Control-Request-Method": "POST"})

        assert response.status_code == 200

    @patch("core.middlewares.cors.settings")
    def test_get_allowed_origins_development(self, mock_settings):
        """Test _get_allowed_origins in development environment."""
        mock_settings.ENVIRONMENT.value = "development"
        mock_settings.CORS_ALLOWED_ORIGINS = ["http://localhost:3000"]
        mock_settings.CORS_ALLOW_ALL_ORIGINS = False

        origins = _get_allowed_origins()

        assert "http://localhost:3000" in origins
        assert "http://localhost:3001" in origins  # Added automatically in dev
        assert "http://127.0.0.1:3000" in origins  # Added automatically in dev

    @patch("core.middlewares.cors.settings")
    def test_get_allowed_origins_development_wildcard(self, mock_settings):
        """Test _get_allowed_origins with wildcard in development."""
        mock_settings.ENVIRONMENT.value = "development"
        mock_settings.CORS_ALLOWED_ORIGINS = []
        mock_settings.CORS_ALLOW_ALL_ORIGINS = True

        origins = _get_allowed_origins()

        assert origins == ["*"]

    @patch("core.middlewares.cors.settings")
    def test_get_allowed_origins_production(self, mock_settings):
        """Test _get_allowed_origins in production environment."""
        mock_settings.ENVIRONMENT.value = "production"
        mock_settings.CORS_ALLOWED_ORIGINS = [
            "https://example.com",
            "https://app.example.com",
            "invalid-origin",  # This should be filtered out
        ]

        with patch("core.middlewares.cors.logger") as mock_logger:
            origins = _get_allowed_origins()

            assert "https://example.com" in origins
            assert "https://app.example.com" in origins
            assert "invalid-origin" not in origins
            mock_logger.warning.assert_called_with("Invalid CORS origin skipped: invalid-origin")

    @patch("core.middlewares.cors.settings")
    def test_get_allowed_origins_production_no_origins(self, mock_settings):
        """Test _get_allowed_origins with no origins in production."""
        mock_settings.ENVIRONMENT.value = "production"
        mock_settings.CORS_ALLOWED_ORIGINS = None

        with patch("core.middlewares.cors.logger") as mock_logger:
            origins = _get_allowed_origins()

            assert origins == []
            mock_logger.warning.assert_called_with(
                "No CORS origins configured for production environment. This may block legitimate requests."
            )

    @patch("core.middlewares.cors.settings")
    def test_get_allowed_origins_testing(self, mock_settings):
        """Test _get_allowed_origins in testing environment."""
        mock_settings.ENVIRONMENT.value = "testing"
        mock_settings.CORS_ALLOWED_ORIGINS = ["http://testserver"]

        origins = _get_allowed_origins()

        assert origins == ["http://testserver"]

    @patch("core.middlewares.cors.settings")
    def test_get_allowed_origins_testing_none(self, mock_settings):
        """Test _get_allowed_origins with None in testing environment."""
        mock_settings.ENVIRONMENT.value = "testing"
        mock_settings.CORS_ALLOWED_ORIGINS = None

        origins = _get_allowed_origins()

        assert origins == ["*"]


class TestOriginValidation:
    """Test origin validation functionality."""

    def test_is_valid_origin_valid_https(self):
        """Test validation of valid HTTPS origins."""
        valid_origins = [
            "https://example.com",
            "https://subdomain.example.com",
            "https://app.example.co.uk",
            "https://localhost:8000",
            "https://127.0.0.1:3000",
            "https://api.example.com:8080",
        ]

        for origin in valid_origins:
            assert _is_valid_origin(origin), f"Should be valid: {origin}"

    def test_is_valid_origin_valid_http(self):
        """Test validation of valid HTTP origins."""
        valid_origins = [
            "http://localhost:3000",
            "http://127.0.0.1:8000",
            "http://0.0.0.0:3000",
            "http://example.com",
            "http://subdomain.example.com:8080",
        ]

        for origin in valid_origins:
            assert _is_valid_origin(origin), f"Should be valid: {origin}"

    def test_is_valid_origin_wildcard(self):
        """Test validation of wildcard origin."""
        assert _is_valid_origin("*")

    def test_is_valid_origin_invalid(self):
        """Test validation of invalid origins."""
        invalid_origins = [
            "",  # Empty string
            "example.com",  # Missing protocol
            "ftp://example.com",  # Invalid protocol
            "https://",  # Missing domain
            "https://.com",  # Invalid domain
            "https://example.",  # Incomplete domain
            "https://example..com",  # Double dots
            "https://example.com:99999",  # Invalid port
            "https://example.com:-1",  # Negative port
            "javascript:alert('xss')",  # XSS attempt
            "data:text/html,<script>alert('xss')</script>",  # Data URL
        ]

        for origin in invalid_origins:
            assert not _is_valid_origin(origin), f"Should be invalid: {origin}"

    def test_is_valid_origin_localhost_patterns(self):
        """Test validation of localhost patterns."""
        valid_localhost = [
            "http://localhost",
            "https://localhost",
            "http://localhost:3000",
            "https://localhost:8080",
            "http://127.0.0.1",
            "https://127.0.0.1",
            "http://127.0.0.1:3000",
            "https://127.0.0.1:8080",
            "http://0.0.0.0",
            "https://0.0.0.0",
            "http://0.0.0.0:3000",
            "https://0.0.0.0:8080",
        ]

        for origin in valid_localhost:
            assert _is_valid_origin(origin), f"Should be valid localhost: {origin}"

    def test_is_valid_origin_edge_cases(self):
        """Test validation of edge cases."""
        edge_cases = [
            ("https://a.com", True),  # Minimum valid domain
            ("https://a-b.com", True),  # Hyphen in domain
            ("https://123.com", True),  # Numeric domain
            ("https://example.com:80", True),  # Standard HTTP port
            ("https://example.com:443", True),  # Standard HTTPS port
            ("https://example.com:65535", True),  # Maximum port
            ("https://example.com:0", False),  # Invalid port 0
            ("https://sub.sub.example.com", True),  # Multiple subdomains
            ("https://-example.com", False),  # Leading hyphen
            ("https://example-.com", False),  # Trailing hyphen
        ]

        for origin, expected in edge_cases:
            result = _is_valid_origin(origin)
            assert result == expected, f"Origin {origin} should be {'valid' if expected else 'invalid'}"


class TestCORSIntegration:
    """Test CORS middleware integration."""

    @patch("core.middlewares.cors.settings")
    def test_cors_headers_in_response(self, mock_settings, app):
        """Test that CORS headers are present in responses."""
        mock_settings.ENVIRONMENT.value = "development"
        mock_settings.CORS_ALLOWED_ORIGINS = ["http://localhost:3000"]
        mock_settings.CORS_ALLOW_ALL_ORIGINS = False
        mock_settings.CORS_ALLOW_CREDENTIALS = True
        mock_settings.CORS_ALLOW_METHODS = ["GET", "POST"]
        mock_settings.CORS_ALLOW_HEADERS = ["Content-Type"]
        mock_settings.CORS_EXPOSE_HEADERS = ["X-Correlation-ID"]
        mock_settings.CORS_MAX_AGE = 3600

        configure_cors(app)
        client = TestClient(app)

        # Test actual request with origin
        response = client.get("/test", headers={"Origin": "http://localhost:3000"})

        assert response.status_code == 200
        assert response.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
        assert response.headers.get("Access-Control-Allow-Credentials") == "true"

    @patch("core.middlewares.cors.settings")
    def test_cors_preflight_request(self, mock_settings, app):
        """Test CORS preflight request handling."""
        mock_settings.ENVIRONMENT.value = "production"
        mock_settings.CORS_ALLOWED_ORIGINS = ["https://example.com"]
        mock_settings.CORS_ALLOW_CREDENTIALS = True
        mock_settings.CORS_ALLOW_METHODS = ["GET", "POST", "PUT", "DELETE"]
        mock_settings.CORS_ALLOW_HEADERS = ["Content-Type", "Authorization"]
        mock_settings.CORS_EXPOSE_HEADERS = ["X-Correlation-ID"]
        mock_settings.CORS_MAX_AGE = 86400

        configure_cors(app)
        client = TestClient(app)

        # Test preflight request
        response = client.options(
            "/test",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type,Authorization",
            },
        )

        assert response.status_code == 200
        assert response.headers.get("Access-Control-Allow-Origin") == "https://example.com"
        assert "POST" in response.headers.get("Access-Control-Allow-Methods", "")
        assert response.headers.get("Access-Control-Max-Age") == "86400"

    @patch("core.middlewares.cors.settings")
    def test_cors_blocked_origin(self, mock_settings, app):
        """Test that disallowed origins are blocked."""
        mock_settings.ENVIRONMENT.value = "production"
        mock_settings.CORS_ALLOWED_ORIGINS = ["https://example.com"]
        mock_settings.CORS_ALLOW_CREDENTIALS = True
        mock_settings.CORS_ALLOW_METHODS = ["GET", "POST"]
        mock_settings.CORS_ALLOW_HEADERS = ["Content-Type"]
        mock_settings.CORS_EXPOSE_HEADERS = []
        mock_settings.CORS_MAX_AGE = 3600

        configure_cors(app)
        client = TestClient(app)

        # Test blocked origin
        response = client.options(
            "/test", headers={"Origin": "https://malicious-site.com", "Access-Control-Request-Method": "GET"}
        )

        # FastAPI CORS middleware returns 400 for blocked origins during preflight
        assert response.status_code == 400
