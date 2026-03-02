from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql.elements import BinaryExpression

from core.repository.base import BaseRepository


# Test Models and Schemas
class Base(DeclarativeBase):
    """Base class for test models."""

    pass


class User(Base):
    """Test user model for repository testing."""

    __tablename__ = "test_users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))


class UserCreateSchema(BaseModel):
    """Schema for creating users."""

    name: str
    email: str


class UserUpdateSchema(BaseModel):
    """Schema for updating users."""

    name: str | None = None
    email: str | None = None


class UserRepository(BaseRepository[User, UserCreateSchema, UserUpdateSchema]):
    """Test repository implementation."""

    pass


@pytest.fixture
def mock_session():
    """Create a mock async session for testing."""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture
def test_repository(mock_session):
    """Create a test repository instance with mocked session."""
    return UserRepository(User, mock_session)


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {"name": "John Doe", "email": "john@example.com"}


@pytest.fixture
def sample_user_schema():
    """Sample user schema for testing."""
    return UserCreateSchema(name="Jane Smith", email="jane@example.com")


@pytest.fixture
def mock_user():
    """Create a mock user object."""
    user = MagicMock()
    user.id = 1
    user.name = "Test User"
    user.email = "test@example.com"
    return user


class TestBaseRepositoryInit:
    """Test BaseRepository initialization."""

    def test_init_with_model_and_session(self, mock_session):
        """Test repository initialization."""
        repo = UserRepository(User, mock_session)

        assert repo.model == User
        assert repo.session == mock_session
        assert repo.model_name == "User"

    def test_init_sets_model_name(self, mock_session):
        """Test that model name is correctly set."""
        repo = UserRepository(User, mock_session)

        assert repo.model_name == User.__name__


class TestBaseRepositoryCreate:
    """Test repository create operations."""

    @pytest.mark.asyncio
    async def test_create_with_dict(self, test_repository, sample_user_data, mock_user):
        """Test creating record with dictionary data."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock the created object
            mock_user.name = sample_user_data["name"]
            mock_user.email = sample_user_data["email"]

            # Configure mocks
            test_repository.session.flush = AsyncMock()
            test_repository.session.refresh = AsyncMock()

            with patch.object(test_repository, "model", return_value=mock_user):
                result = await test_repository.create(sample_user_data)

                test_repository.session.add.assert_called_once()
                test_repository.session.flush.assert_called_once()
                test_repository.session.refresh.assert_called_once()

                mock_span.set_attribute.assert_any_call("repository.model", "User")
                mock_span.set_attribute.assert_any_call("repository.operation", "create")

    @pytest.mark.asyncio
    async def test_create_with_pydantic_schema(self, test_repository, sample_user_schema, mock_user):
        """Test creating record with Pydantic schema."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Configure mocks
            test_repository.session.flush = AsyncMock()
            test_repository.session.refresh = AsyncMock()

            with patch.object(test_repository, "model", return_value=mock_user):
                result = await test_repository.create(sample_user_schema)

                test_repository.session.add.assert_called_once()
                test_repository.session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_with_invalid_input_type(self, test_repository):
        """Test creating record with invalid input type."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            with pytest.raises(ValueError, match="Invalid input type"):
                await test_repository.create("invalid_input")

    @pytest.mark.asyncio
    async def test_create_sets_tracing_attributes(self, test_repository, sample_user_data, mock_user):
        """Test that create operation sets proper tracing attributes."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Configure mocks
            test_repository.session.flush = AsyncMock()
            test_repository.session.refresh = AsyncMock()

            with patch.object(test_repository, "model", return_value=mock_user):
                result = await test_repository.create(sample_user_data)

                mock_span.set_attribute.assert_any_call("repository.model", "User")
                mock_span.set_attribute.assert_any_call("repository.operation", "create")
                mock_span.set_attribute.assert_any_call("repository.success", True)

    @pytest.mark.asyncio
    async def test_create_handles_database_error(self, test_repository):
        """Test error handling during create operation."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock session to raise an exception
            test_repository.session.add.side_effect = Exception("Database error")

            with pytest.raises(Exception, match="Database error"):
                await test_repository.create({"name": "Test", "email": "test@example.com"})

            mock_span.set_attribute.assert_any_call("repository.success", False)
            mock_span.record_exception.assert_called_once()


class TestBaseRepositoryGet:
    """Test repository get operations."""

    @pytest.mark.asyncio
    async def test_get_existing_record(self, test_repository, mock_user):
        """Test getting an existing record."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock result
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = mock_user
            test_repository.session.execute.return_value = mock_result

            result = await test_repository.get(1)

            assert result == mock_user
            test_repository.session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_non_existing_record(self, test_repository):
        """Test getting a non-existing record."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock result
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            test_repository.session.execute.return_value = mock_result

            result = await test_repository.get(99999)

            assert result is None

    @pytest.mark.asyncio
    async def test_get_sets_tracing_attributes(self, test_repository):
        """Test that get operation sets proper tracing attributes."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock result
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            test_repository.session.execute.return_value = mock_result

            await test_repository.get(1)

            mock_span.set_attribute.assert_any_call("repository.model", "User")
            mock_span.set_attribute.assert_any_call("repository.operation", "get")
            mock_span.set_attribute.assert_any_call("repository.id", "1")
            mock_span.set_attribute.assert_any_call("repository.found", False)


class TestBaseRepositoryGetMulti:
    """Test repository get_multi operations."""

    @pytest.mark.asyncio
    async def test_get_multi_without_filters(self, test_repository):
        """Test getting multiple records without filters."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock results
            mock_users = [MagicMock() for _ in range(5)]
            mock_result = MagicMock()
            mock_result.scalars().all.return_value = mock_users
            test_repository.session.execute.return_value = mock_result

            result = await test_repository.get_multi()

            assert len(result) == 5
            test_repository.session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_multi_with_pagination(self, test_repository):
        """Test getting multiple records with pagination."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock results
            mock_users = [MagicMock() for _ in range(3)]
            mock_result = MagicMock()
            mock_result.scalars().all.return_value = mock_users
            test_repository.session.execute.return_value = mock_result

            result = await test_repository.get_multi(skip=2, limit=3)

            assert len(result) == 3
            mock_span.set_attribute.assert_any_call("repository.skip", 2)
            mock_span.set_attribute.assert_any_call("repository.limit", 3)

    @pytest.mark.asyncio
    async def test_get_multi_with_filters(self, test_repository):
        """Test getting multiple records with filters."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock results
            mock_user = MagicMock()
            mock_user.name = "Alice"
            mock_result = MagicMock()
            mock_result.scalars().all.return_value = [mock_user]
            test_repository.session.execute.return_value = mock_result

            filters = [MagicMock(spec=BinaryExpression)]
            result = await test_repository.get_multi(filters=filters)

            assert len(result) == 1
            mock_span.set_attribute.assert_any_call("repository.filters_count", 1)

    @pytest.mark.asyncio
    async def test_get_multi_with_order_by(self, test_repository):
        """Test getting multiple records with ordering."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock results
            mock_users = [MagicMock() for _ in range(3)]
            mock_result = MagicMock()
            mock_result.scalars().all.return_value = mock_users
            test_repository.session.execute.return_value = mock_result

            result = await test_repository.get_multi(order_by=User.name)

            assert len(result) == 3


class TestBaseRepositoryUpdate:
    """Test repository update operations."""

    @pytest.mark.asyncio
    async def test_update_existing_record_with_dict(self, test_repository, mock_user):
        """Test updating existing record with dictionary."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock get method to return user
            with patch.object(test_repository, "get", return_value=mock_user):
                test_repository.session.flush = AsyncMock()
                test_repository.session.refresh = AsyncMock()

                update_data = {"name": "Updated Name"}
                result = await test_repository.update(1, update_data)

                assert result == mock_user
                test_repository.session.flush.assert_called_once()
                test_repository.session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_existing_record_with_schema(self, test_repository, mock_user):
        """Test updating existing record with Pydantic schema."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock get method to return user
            with patch.object(test_repository, "get", return_value=mock_user):
                test_repository.session.flush = AsyncMock()
                test_repository.session.refresh = AsyncMock()

                update_schema = UserUpdateSchema(name="Schema Updated Name")
                result = await test_repository.update(1, update_schema)

                assert result == mock_user

    @pytest.mark.asyncio
    async def test_update_non_existing_record(self, test_repository):
        """Test updating non-existing record."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock get method to return None
            with patch.object(test_repository, "get", return_value=None):
                result = await test_repository.update(99999, {"name": "Non-existing"})

                assert result is None
                mock_span.set_attribute.assert_any_call("repository.found", False)

    @pytest.mark.asyncio
    async def test_update_with_invalid_input_type(self, test_repository, mock_user):
        """Test updating with invalid input type."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            with patch.object(test_repository, "get", return_value=mock_user):
                with pytest.raises(ValueError, match="Invalid input type"):
                    await test_repository.update(1, "invalid_input")


class TestBaseRepositoryDelete:
    """Test repository delete operations."""

    @pytest.mark.asyncio
    async def test_delete_existing_record(self, test_repository):
        """Test deleting existing record."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock result with rowcount > 0
            mock_result = MagicMock()
            mock_result.rowcount = 1
            test_repository.session.execute.return_value = mock_result

            result = await test_repository.delete(1)

            assert result is True
            mock_span.set_attribute.assert_any_call("repository.deleted", True)

    @pytest.mark.asyncio
    async def test_delete_non_existing_record(self, test_repository):
        """Test deleting non-existing record."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock result with rowcount = 0
            mock_result = MagicMock()
            mock_result.rowcount = 0
            test_repository.session.execute.return_value = mock_result

            result = await test_repository.delete(99999)

            assert result is False
            mock_span.set_attribute.assert_any_call("repository.deleted", False)


class TestBaseRepositoryCount:
    """Test repository count operations."""

    @pytest.mark.asyncio
    async def test_count_all_records(self, test_repository):
        """Test counting all records."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock result
            mock_result = MagicMock()
            mock_result.scalar.return_value = 5
            test_repository.session.execute.return_value = mock_result

            count = await test_repository.count()

            assert count == 5
            mock_span.set_attribute.assert_any_call("repository.count", 5)

    @pytest.mark.asyncio
    async def test_count_with_filters(self, test_repository):
        """Test counting records with filters."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock result
            mock_result = MagicMock()
            mock_result.scalar.return_value = 2
            test_repository.session.execute.return_value = mock_result

            filters = [MagicMock(spec=BinaryExpression)]
            count = await test_repository.count(filters=filters)

            assert count == 2
            mock_span.set_attribute.assert_any_call("repository.filters_count", 1)

    @pytest.mark.asyncio
    async def test_count_empty_table(self, test_repository):
        """Test counting records in empty table."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock result
            mock_result = MagicMock()
            mock_result.scalar.return_value = None
            test_repository.session.execute.return_value = mock_result

            count = await test_repository.count()

            assert count == 0


class TestBaseRepositoryExists:
    """Test repository exists operations."""

    @pytest.mark.asyncio
    async def test_exists_with_existing_record(self, test_repository):
        """Test checking existence of existing record."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock result
            mock_result = MagicMock()
            mock_result.scalar.return_value = 1
            test_repository.session.execute.return_value = mock_result

            exists = await test_repository.exists(1)

            assert exists is True
            mock_span.set_attribute.assert_any_call("repository.exists", True)

    @pytest.mark.asyncio
    async def test_exists_with_non_existing_record(self, test_repository):
        """Test checking existence of non-existing record."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock result
            mock_result = MagicMock()
            mock_result.scalar.return_value = None
            test_repository.session.execute.return_value = mock_result

            exists = await test_repository.exists(99999)

            assert exists is False
            mock_span.set_attribute.assert_any_call("repository.exists", False)


class TestBaseRepositoryBulkOperations:
    """Test repository bulk operations."""

    @pytest.mark.asyncio
    async def test_bulk_create_with_dicts(self, test_repository):
        """Test bulk creating records with dictionaries."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Configure mocks
            test_repository.session.add_all = MagicMock()
            test_repository.session.flush = AsyncMock()
            test_repository.session.refresh = AsyncMock()

            users_data = [{"name": f"User {i}", "email": f"user{i}@example.com"} for i in range(3)]

            with patch.object(test_repository, "model") as mock_model:
                mock_users = [MagicMock() for _ in range(3)]
                for i, mock_user in enumerate(mock_users):
                    mock_user.id = i + 1
                    mock_user.name = f"User {i}"
                    mock_user.email = f"user{i}@example.com"

                mock_model.side_effect = mock_users

                results = await test_repository.bulk_create(users_data)

                test_repository.session.add_all.assert_called_once()
                test_repository.session.flush.assert_called_once()
                assert test_repository.session.refresh.call_count == 3

    @pytest.mark.asyncio
    async def test_bulk_create_with_schemas(self, test_repository):
        """Test bulk creating records with Pydantic schemas."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Configure mocks
            test_repository.session.add_all = MagicMock()
            test_repository.session.flush = AsyncMock()
            test_repository.session.refresh = AsyncMock()

            users_schemas = [UserCreateSchema(name=f"Schema User {i}", email=f"schema{i}@example.com") for i in range(2)]

            with patch.object(test_repository, "model") as mock_model:
                mock_users = [MagicMock() for _ in range(2)]
                mock_model.side_effect = mock_users

                results = await test_repository.bulk_create(users_schemas)

                test_repository.session.add_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_bulk_update(self, test_repository):
        """Test bulk updating records."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock results
            mock_result = MagicMock()
            mock_result.rowcount = 1
            test_repository.session.execute.return_value = mock_result

            updates = {
                1: {"name": "Updated User 1"},
                2: {"name": "Updated User 2"},
            }

            updated_count = await test_repository.bulk_update(updates)

            assert updated_count == 2
            assert test_repository.session.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_bulk_delete(self, test_repository):
        """Test bulk deleting records."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock result
            mock_result = MagicMock()
            mock_result.rowcount = 3
            test_repository.session.execute.return_value = mock_result

            ids_to_delete = [1, 2, 3]
            deleted_count = await test_repository.bulk_delete(ids_to_delete)

            assert deleted_count == 3
            mock_span.set_attribute.assert_any_call("repository.deleted_count", 3)


class TestBaseRepositoryErrorHandling:
    """Test repository error handling."""

    @pytest.mark.asyncio
    async def test_database_error_handling(self, test_repository):
        """Test proper error handling for database errors."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock session to raise an exception
            test_repository.session.execute.side_effect = Exception("Database connection error")

            with pytest.raises(Exception, match="Database connection error"):
                await test_repository.get(1)

            mock_span.set_attribute.assert_any_call("repository.success", False)
            mock_span.record_exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_bulk_create_with_invalid_input(self, test_repository):
        """Test bulk create with invalid input types."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            invalid_data = ["invalid", "input", "types"]

            with pytest.raises(ValueError, match="Invalid input type"):
                await test_repository.bulk_create(invalid_data)


class TestBaseRepositoryObservability:
    """Test repository observability features."""

    @pytest.mark.asyncio
    async def test_tracing_span_creation(self, test_repository, sample_user_data, mock_user):
        """Test that tracing spans are created for operations."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Configure mocks for create operation
            test_repository.session.flush = AsyncMock()
            test_repository.session.refresh = AsyncMock()

            with patch.object(test_repository, "model", return_value=mock_user):
                await test_repository.create(sample_user_data)

                mock_tracer.assert_called_with("repository_create")
                mock_span.set_attribute.assert_any_call("repository.model", "User")
                mock_span.set_attribute.assert_any_call("repository.operation", "create")

    @pytest.mark.asyncio
    async def test_logging_on_operations(self, test_repository, sample_user_data, mock_user):
        """Test that logging occurs during operations."""
        with patch("core.repository.base.logger") as mock_logger:
            with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
                mock_span = MagicMock()
                mock_tracer.return_value.__enter__.return_value = mock_span

                # Configure mocks
                test_repository.session.flush = AsyncMock()
                test_repository.session.refresh = AsyncMock()

                with patch.object(test_repository, "model", return_value=mock_user):
                    _ = await test_repository.create(sample_user_data)

                    mock_logger.info.assert_called_with(
                        "Record created successfully",
                        model="User",
                        record_id=mock_user.id,
                    )

    @pytest.mark.asyncio
    async def test_error_logging(self, test_repository):
        """Test that errors are properly logged."""
        with patch("core.repository.base.logger") as mock_logger:
            with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
                mock_span = MagicMock()
                mock_tracer.return_value.__enter__.return_value = mock_span

                # Mock session to raise an exception
                test_repository.session.execute.side_effect = Exception("Test error")

                with pytest.raises(Exception):
                    await test_repository.get(1)

                mock_logger.error.assert_called_once()
                error_call = mock_logger.error.call_args
                assert "Failed to get record" in error_call[0][0]
                assert error_call[1]["model"] == "User"
                assert error_call[1]["record_id"] == "1"


class TestBaseRepositoryIntegration:
    """Test repository integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_crud_workflow(self, test_repository, mock_user):
        """Test complete CRUD workflow."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Configure mocks for each operation
            test_repository.session.flush = AsyncMock()
            test_repository.session.refresh = AsyncMock()

            # Create operation
            with patch.object(test_repository, "model", return_value=mock_user):
                user_data = {"name": "Test User", "email": "test@example.com"}
                created = await test_repository.create(user_data)
                assert created == mock_user

            # Get operation
            with patch.object(test_repository, "get", return_value=mock_user):
                retrieved = await test_repository.get(mock_user.id)
                assert retrieved == mock_user

            # Update operation
            with patch.object(test_repository, "update", return_value=mock_user):
                updated = await test_repository.update(mock_user.id, {"name": "Updated User"})
                assert updated == mock_user

            # Delete operation
            with patch.object(test_repository, "delete", return_value=True):
                deleted = await test_repository.delete(mock_user.id)
                assert deleted is True

            # Verify deletion
            with patch.object(test_repository, "get", return_value=None):
                not_found = await test_repository.get(mock_user.id)
                assert not_found is None

    @pytest.mark.asyncio
    async def test_pagination_and_filtering_workflow(self, test_repository):
        """Test pagination and filtering workflow."""
        with patch("core.repository.base.tracer.start_as_current_span") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.return_value.__enter__.return_value = mock_span

            # Mock bulk create
            mock_users = [MagicMock() for _ in range(5)]
            with patch.object(test_repository, "bulk_create", return_value=mock_users):
                users_data = [{"name": "Alice", "email": f"alice{i}@example.com"} for i in range(3)] + [
                    {"name": "Bob", "email": f"bob{i}@example.com"} for i in range(2)
                ]

                await test_repository.bulk_create(users_data)

            # Test filtering
            alice_users = [u for u in mock_users[:3]]
            with patch.object(test_repository, "get_multi", return_value=alice_users):
                filters = [MagicMock(spec=BinaryExpression)]
                result = await test_repository.get_multi(filters=filters)
                assert len(result) == 3

            # Test pagination
            first_page = mock_users[:2]
            second_page = mock_users[2:4]
            with patch.object(test_repository, "get_multi", side_effect=[first_page, second_page]):
                first = await test_repository.get_multi(skip=0, limit=2)
                second = await test_repository.get_multi(skip=2, limit=2)
                assert len(first) == 2
                assert len(second) == 2

            # Test counting
            with patch.object(test_repository, "count", side_effect=[3, 5]):
                alice_count = await test_repository.count(filters=[MagicMock(spec=BinaryExpression)])
                total_count = await test_repository.count()
                assert alice_count == 3
                assert total_count == 5
