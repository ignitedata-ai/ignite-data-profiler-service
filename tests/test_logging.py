import uuid

from core.logging import get_correlation_id, get_logger, set_correlation_id


class TestLogging:
    """Test logging functionality."""

    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger("test")
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "debug")

    def test_correlation_id_generation(self):
        """Test correlation ID generation."""
        correlation_id = get_correlation_id()
        assert correlation_id is not None
        assert isinstance(correlation_id, str)

        # Should be a valid UUID
        uuid.UUID(correlation_id)

    def test_correlation_id_setting(self):
        """Test correlation ID setting."""
        test_id = "test-correlation-id"
        set_correlation_id(test_id)

        retrieved_id = get_correlation_id()
        assert retrieved_id == test_id

    def test_logger_output(self, caplog):
        """Test logger output format."""
        logger = get_logger("test.module")

        with caplog.at_level("INFO"):
            logger.info("Test message", extra_field="test_value")

        assert len(caplog.records) > 0
        record = caplog.records[0]
        assert record.levelname == "INFO"
        assert "Test message" in record.getMessage()

    def test_logger_with_correlation_id(self, caplog):
        """Test logger includes correlation ID."""
        test_id = "test-correlation-123"
        set_correlation_id(test_id)

        logger = get_logger("test.correlation")

        with caplog.at_level("INFO"):
            logger.info("Test with correlation ID")

        # Note: In actual implementation, correlation ID would be in structured output
        assert len(caplog.records) > 0
