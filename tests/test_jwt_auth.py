"""Tests for JWT authentication middleware and verification logic."""

from __future__ import annotations

import time
from unittest.mock import patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from core.middlewares.jwt_auth import JWTAuthMiddleware, _is_public_path
from core.security.jwt import reset_key_cache, verify_token

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rsa_keypair():
    """Generate a fresh RSA key pair for testing."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()

    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )

    return private_pem, public_pem


@pytest.fixture
def valid_token(rsa_keypair):
    """Create a valid JWT signed with the test private key."""
    private_pem, _ = rsa_keypair
    return pyjwt.encode(
        {"sub": "user-123", "exp": int(time.time()) + 3600},
        private_pem,
        algorithm="RS256",
    )


@pytest.fixture
def expired_token(rsa_keypair):
    """Create an expired JWT."""
    private_pem, _ = rsa_keypair
    return pyjwt.encode(
        {"sub": "user-123", "exp": int(time.time()) - 60},
        private_pem,
        algorithm="RS256",
    )


@pytest.fixture(autouse=True)
def clear_key_cache():
    """Ensure the key cache is cleared between tests."""
    reset_key_cache()
    yield
    reset_key_cache()


@pytest.fixture
def mock_public_key(rsa_keypair, tmp_path):
    """Write the test public key to a temp file and patch the path."""
    _, public_pem = rsa_keypair
    key_file = tmp_path / "public.pem"
    key_file.write_text(public_pem)

    with patch("core.security.jwt.get_public_key_path", return_value=key_file):
        yield key_file


@pytest.fixture
def app_with_auth():
    """Create a minimal FastAPI app with JWTAuthMiddleware."""
    from core.config import settings as _settings

    _orig = _settings.JWT_AUTH_ENABLED
    object.__setattr__(_settings, "JWT_AUTH_ENABLED", True)

    app = FastAPI()
    app.add_middleware(JWTAuthMiddleware)

    @app.get("/profile/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/profile/metrics")
    async def metrics():
        return {"status": "ok"}

    @app.get("/protected")
    async def protected(request: Request):
        user = getattr(request.state, "user", None)
        return {"user": user}

    yield app

    object.__setattr__(_settings, "JWT_AUTH_ENABLED", _orig)


# ── Unit tests for core.security.jwt ──────────────────────────────────────────


class TestVerifyToken:
    def test_valid_token(self, mock_public_key, valid_token):
        payload = verify_token(valid_token)
        assert payload["sub"] == "user-123"

    def test_expired_token_raises(self, mock_public_key, expired_token):
        with pytest.raises(pyjwt.ExpiredSignatureError):
            verify_token(expired_token)

    def test_invalid_signature_raises(self, mock_public_key):
        other_key = rsa.generate_private_key(65537, 2048)
        other_pem = other_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode()
        bad_token = pyjwt.encode(
            {"sub": "x", "exp": int(time.time()) + 3600},
            other_pem,
            algorithm="RS256",
        )
        with pytest.raises(pyjwt.InvalidSignatureError):
            verify_token(bad_token)

    def test_malformed_token_raises(self, mock_public_key):
        with pytest.raises(pyjwt.DecodeError):
            verify_token("not.a.jwt")

    def test_missing_key_file_raises(self, tmp_path):
        missing = tmp_path / "nope.pem"
        with patch("core.security.jwt.get_public_key_path", return_value=missing):
            with pytest.raises(FileNotFoundError):
                verify_token("anything")

    def test_token_without_exp_raises(self, rsa_keypair, mock_public_key):
        private_pem, _ = rsa_keypair
        token = pyjwt.encode(
            {"sub": "user-123"},
            private_pem,
            algorithm="RS256",
        )
        with pytest.raises(pyjwt.MissingRequiredClaimError):
            verify_token(token)


# ── Unit tests for path classification ────────────────────────────────────────


class TestPublicPaths:
    @pytest.mark.parametrize(
        "path",
        [
            "/profile/health",
            "/profile/metrics",
            "/profile/docs",
            "/profile/docs/oauth2-redirect",
            "/profile/redoc",
            "/profile/openapi.json",
        ],
    )
    def test_public_paths(self, path):
        assert _is_public_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "/profile/v1/profile/tasks",
            "/profile/v1/profile/overview",
            "/protected",
        ],
    )
    def test_protected_paths(self, path):
        assert _is_public_path(path) is False


# ── Integration tests for the middleware ──────────────────────────────────────


class TestJWTAuthMiddleware:
    def test_public_path_no_token_required(self, app_with_auth):
        client = TestClient(app_with_auth)
        response = client.get("/profile/health")
        assert response.status_code == 200

    def test_metrics_path_no_token_required(self, app_with_auth):
        client = TestClient(app_with_auth)
        response = client.get("/profile/metrics")
        assert response.status_code == 200

    def test_protected_path_no_token_returns_401(self, app_with_auth):
        client = TestClient(app_with_auth)
        response = client.get("/protected")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "AUTHENTICATION_ERROR"
        assert "WWW-Authenticate" in response.headers

    def test_protected_path_invalid_scheme_returns_401(self, app_with_auth):
        client = TestClient(app_with_auth)
        response = client.get("/protected", headers={"Authorization": "Basic abc123"})
        assert response.status_code == 401

    def test_protected_path_empty_token_returns_401(self, app_with_auth):
        client = TestClient(app_with_auth)
        response = client.get("/protected", headers={"Authorization": "Bearer "})
        assert response.status_code == 401

    def test_protected_path_valid_token(self, app_with_auth, mock_public_key, valid_token):
        client = TestClient(app_with_auth)
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {valid_token}"},
        )
        assert response.status_code == 200
        assert response.json()["user"]["sub"] == "user-123"

    def test_protected_path_expired_token_returns_401(self, app_with_auth, mock_public_key, expired_token):
        client = TestClient(app_with_auth)
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert response.status_code == 401
        assert "expired" in response.json()["error"]["message"].lower()

    def test_protected_path_malformed_token_returns_401(self, app_with_auth, mock_public_key):
        client = TestClient(app_with_auth)
        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer garbage.token.here"},
        )
        assert response.status_code == 401

    def test_auth_disabled_bypasses_verification(self, app_with_auth):
        client = TestClient(app_with_auth)
        with patch("core.middlewares.jwt_auth.settings") as mock_settings:
            mock_settings.JWT_AUTH_ENABLED = False
            response = client.get("/protected")
            assert response.status_code == 200
