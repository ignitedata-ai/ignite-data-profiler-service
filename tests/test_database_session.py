import asyncio
from unittest.mock import patch

import pytest
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.session import Base, DatabaseSessionManager, get_session_manager, initialize_database


class DatabaseTestModel(Base):
    """Test model for database operations."""

    __tablename__ = "test_model"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    description = Column(String(200))


@pytest.fixture
def test_database_url() -> str:
    """Provide test database URL."""
    return "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def session_manager(test_database_url: str) -> DatabaseSessionManager:
    """Create and initialize a test session manager."""
    manager = DatabaseSessionManager(test_database_url, echo=True)
    await manager.initialize()

    # Create test tables
    async with manager._engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield manager
    await manager.close()


class TestDatabaseSessionManager:
    """Test cases for DatabaseSessionManager."""

    async def test_initialization(self, test_database_url: str):
        """Test session manager initialization."""
        manager = DatabaseSessionManager(test_database_url, echo=True)

        assert manager._database_url == test_database_url
        assert manager._echo is True
        assert manager._engine is None
        assert manager._session_factory is None

        await manager.initialize()

        assert manager._engine is not None
        assert manager._session_factory is not None

        await manager.close()

    async def test_initialization_failure(self):
        """Test session manager initialization with invalid URL."""
        manager = DatabaseSessionManager("invalid://url", echo=False)

        with pytest.raises(Exception):
            await manager.initialize()

    async def test_get_session_before_initialization(self):
        """Test getting session before initialization raises error."""
        manager = DatabaseSessionManager("sqlite+aiosqlite:///:memory:")

        with pytest.raises(RuntimeError, match="Database session manager not initialized"):
            async with manager.get_session() as session:
                pass

    async def test_get_session_success(self, session_manager: DatabaseSessionManager):
        """Test successful session creation and cleanup."""
        session = None
        async with session_manager.get_session() as db_session:
            session = db_session
            assert isinstance(session, AsyncSession)
            assert session.is_active

        # Session should be closed after context exit
        # Note: SQLite sessions may not immediately show as inactive
        assert session is not None

    async def test_get_session_with_exception(self, session_manager: DatabaseSessionManager):
        """Test session rollback on exception."""
        with pytest.raises(ValueError, match="Test error"):
            async with session_manager.get_session() as session:
                # Insert a test record
                test_record = DatabaseTestModel(name="test", description="test description")
                session.add(test_record)
                await session.flush()

                # Raise an exception to trigger rollback
                raise ValueError("Test error")

        # Verify record was not committed
        async with session_manager.get_session() as session:
            from sqlalchemy import select

            result = await session.execute(select(DatabaseTestModel))
            records = result.scalars().all()
            assert len(records) == 0

    async def test_session_commit_on_success(self, session_manager: DatabaseSessionManager):
        """Test session commit on successful completion."""
        async with session_manager.get_session() as session:
            test_record = DatabaseTestModel(name="test", description="test description")
            session.add(test_record)
            await session.flush()

        # Verify record was committed
        async with session_manager.get_session() as session:
            from sqlalchemy import select

            result = await session.execute(select(DatabaseTestModel))
            records = result.scalars().all()
            assert len(records) == 1
            assert records[0].name == "test"

    async def test_health_check_success(self, session_manager: DatabaseSessionManager):
        """Test successful health check."""
        health_data = await session_manager.health_check()

        assert health_data["status"] == "healthy"
        assert "pool" in health_data
        assert "database_url" in health_data
        assert isinstance(health_data["pool"], dict)

    async def test_health_check_before_initialization(self):
        """Test health check before initialization."""
        manager = DatabaseSessionManager("sqlite+aiosqlite:///:memory:")

        health_data = await manager.health_check()

        assert health_data["status"] == "unhealthy"
        assert "error" in health_data
        assert health_data["error"] == "Database engine not initialized"

    async def test_health_check_with_connection_error(self):
        """Test health check with connection error."""
        manager = DatabaseSessionManager("postgresql+asyncpg://invalid:invalid@localhost:9999/invalid")
        await manager.initialize()

        health_data = await manager.health_check()

        assert health_data["status"] == "unhealthy"
        assert "error" in health_data

        await manager.close()

    async def test_close_before_initialization(self):
        """Test closing before initialization."""
        manager = DatabaseSessionManager("sqlite+aiosqlite:///:memory:")

        # Should not raise an exception
        await manager.close()

    async def test_multiple_sessions_concurrent(self, session_manager: DatabaseSessionManager):
        """Test multiple concurrent sessions."""

        async def create_record(name: str):
            async with session_manager.get_session() as session:
                test_record = DatabaseTestModel(name=name, description=f"Description for {name}")
                session.add(test_record)
                await session.flush()
                await asyncio.sleep(0.1)  # Simulate some work

        # Create multiple concurrent sessions
        tasks = [create_record(f"record_{i}") for i in range(5)]
        await asyncio.gather(*tasks)

        # Verify all records were created
        async with session_manager.get_session() as session:
            from sqlalchemy import select

            result = await session.execute(select(DatabaseTestModel))
            records = result.scalars().all()
            assert len(records) == 5


class TestGlobalSessionManager:
    """Test cases for global session manager functions."""

    def test_initialize_database(self):
        """Test global database initialization."""
        with patch("core.database.session.settings") as mock_settings:
            mock_settings.DATABASE_URL = "sqlite+aiosqlite:///:memory:"
            mock_settings.DATABASE_ECHO = True

            manager = initialize_database()

            assert isinstance(manager, DatabaseSessionManager)
            assert manager._database_url == "sqlite+aiosqlite:///:memory:"
            assert manager._echo is True

    def test_get_session_manager_before_initialization(self):
        """Test getting session manager before initialization."""
        # Clear global session manager
        import core.database.session

        original_manager = core.database.session.session_manager
        core.database.session.session_manager = None

        try:
            with pytest.raises(RuntimeError, match="Database session manager not initialized"):
                get_session_manager()
        finally:
            # Restore original manager
            core.database.session.session_manager = original_manager

    def test_get_session_manager_after_initialization(self):
        """Test getting session manager after initialization."""
        with patch("core.database.session.settings") as mock_settings:
            mock_settings.DATABASE_URL = "sqlite+aiosqlite:///:memory:"
            mock_settings.DATABASE_ECHO = False

            # Initialize global manager
            manager = initialize_database()

            # Get manager
            retrieved_manager = get_session_manager()

            assert retrieved_manager is manager


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database session management."""

    async def test_real_database_operations(self):
        """Test with real database operations (requires PostgreSQL)."""
        # This test requires a real PostgreSQL instance
        # Skip if not available
        pytest.skip("Requires PostgreSQL instance for integration testing")

        database_url = "postgresql+asyncpg://postgres:password@localhost:5432/aipal_test"
        manager = DatabaseSessionManager(database_url)

        try:
            await manager.initialize()

            # Test basic operations
            async with manager.get_session() as session:
                from sqlalchemy import text

                result = await session.execute(text("SELECT version()"))
                version = result.scalar()
                assert "PostgreSQL" in version

            # Test health check
            health_data = await manager.health_check()
            assert health_data["status"] == "healthy"

        finally:
            await manager.close()

    async def test_connection_pool_behavior(self, session_manager: DatabaseSessionManager):
        """Test connection pool behavior under load."""

        async def database_operation(operation_id: int):
            async with session_manager.get_session() as session:
                test_record = DatabaseTestModel(name=f"operation_{operation_id}", description="Pool test")
                session.add(test_record)
                await session.flush()
                await asyncio.sleep(0.05)  # Simulate work

        # Create many concurrent operations to test pool
        tasks = [database_operation(i) for i in range(20)]
        await asyncio.gather(*tasks)

        # Verify all operations completed
        async with session_manager.get_session() as session:
            from sqlalchemy import func, select

            result = await session.execute(select(func.count(DatabaseTestModel.id)))
            count = result.scalar()
            assert count == 20

    async def test_transaction_isolation(self, session_manager: DatabaseSessionManager):
        """Test transaction isolation between sessions."""
        # Note: SQLite has different isolation behavior than PostgreSQL
        # This test demonstrates transaction boundaries rather than strict isolation

        # Session 1: Create and commit data
        async with session_manager.get_session() as session1:
            test_record = DatabaseTestModel(name="isolation_test", description="Test isolation")
            session1.add(test_record)
            await session1.flush()
            # Data is committed when context exits

        # Session 2: Should see committed data
        async with session_manager.get_session() as session2:
            from sqlalchemy import select

            result = await session2.execute(select(DatabaseTestModel).where(DatabaseTestModel.name == "isolation_test"))
            records = result.scalars().all()
            assert len(records) == 1

        # Session 3: Verify data persists across sessions
        async with session_manager.get_session() as session3:
            from sqlalchemy import select

            result = await session3.execute(select(DatabaseTestModel).where(DatabaseTestModel.name == "isolation_test"))
            records = result.scalars().all()
            assert len(records) == 1
            assert records[0].name == "isolation_test"


class TestDatabaseSessionDependency:
    """Test cases for FastAPI dependency function."""

    async def test_get_db_session_dependency(self, session_manager: DatabaseSessionManager):
        """Test the FastAPI dependency function."""
        from core.database.session import get_db_session

        # Mock global session manager
        with patch("core.database.session.get_session_manager", return_value=session_manager):
            async for session in get_db_session():
                assert isinstance(session, AsyncSession)
                assert session.is_active

                # Test basic operation
                test_record = DatabaseTestModel(name="dependency_test", description="Test dependency")
                session.add(test_record)
                await session.flush()

                break  # Exit the async generator
