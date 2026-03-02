from fastapi.testclient import TestClient

from core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    BusinessLogicError,
    NotFoundError,
    ValidationError,
)


class TestExceptions:
    """Test custom exception classes."""

    def test_validation_error(self):
        """Test ValidationError creation."""
        error = ValidationError(
            message="Invalid input",
            field="email",
            value="invalid-email",
        )

        assert error.message == "Invalid input"
        assert error.error_code == "ValidationError"
        assert error.details["field"] == "email"
        assert error.details["value"] == "invalid-email"

    def test_not_found_error(self):
        """Test NotFoundError creation."""
        error = NotFoundError(
            message="User not found",
            resource_type="User",
            resource_id="123",
        )

        assert error.message == "User not found"
        assert error.error_code == "NotFoundError"
        assert error.details["resource_type"] == "User"
        assert error.details["resource_id"] == "123"

    def test_authentication_error(self):
        """Test AuthenticationError creation."""
        error = AuthenticationError(message="Invalid credentials")

        assert error.message == "Invalid credentials"
        assert error.error_code == "AuthenticationError"

    def test_authorization_error(self):
        """Test AuthorizationError creation."""
        error = AuthorizationError(message="Insufficient permissions")

        assert error.message == "Insufficient permissions"
        assert error.error_code == "AuthorizationError"

    def test_business_logic_error(self):
        """Test BusinessLogicError creation."""
        error = BusinessLogicError(message="Business rule violation")

        assert error.message == "Business rule violation"
        assert error.error_code == "BusinessLogicError"

    def test_exception_to_dict(self):
        """Test exception to_dict method."""
        error = ValidationError(
            message="Test error",
            field="test_field",
        )

        error_dict = error.to_dict()

        assert error_dict["error_code"] == "ValidationError"
        assert error_dict["message"] == "Test error"
        assert error_dict["details"]["field"] == "test_field"


class TestExceptionHandling:
    """Test exception handling in the application."""

    def test_404_endpoint(self, client: TestClient):
        """Test 404 handling for non-existent endpoints."""
        response = client.get("/non-existent-endpoint")

        assert response.status_code == 404
        data = response.json()

        assert "error" in data
        assert data["error"]["code"] == "HTTP_ERROR"
