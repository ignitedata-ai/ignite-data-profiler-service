from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from opentelemetry import trace
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql.elements import BinaryExpression

from core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Type variables for generic repository
ModelType = TypeVar("ModelType", bound=DeclarativeBase)
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Base repository class providing common CRUD operations with observability."""

    def __init__(self, model: type[ModelType], session: AsyncSession) -> None:
        """Initialize the repository.

        Args:
            model: The SQLAlchemy model class
            session: The database session

        """
        self.model = model
        self.session = session
        self.model_name = model.__name__

    async def create(self, obj_in: CreateSchemaType | dict[str, Any]) -> ModelType:
        """Create a new record.

        Args:
            obj_in: The data to create the record with

        Returns:
            ModelType: The created record

        Raises:
            Exception: If creation fails

        """
        with tracer.start_as_current_span("repository_create") as span:
            span.set_attribute("repository.model", self.model_name)
            span.set_attribute("repository.operation", "create")

            try:
                # Convert to dict if it's a Pydantic model
                if hasattr(obj_in, "model_dump"):
                    obj_data = obj_in.model_dump(exclude_unset=True)  # type: ignore
                elif isinstance(obj_in, dict):
                    obj_data = obj_in
                else:
                    raise ValueError("Invalid input type")

                db_obj = self.model(**obj_data)
                self.session.add(db_obj)
                await self.session.flush()
                await self.session.refresh(db_obj)

                logger.info(
                    "Record created successfully",
                    model=self.model_name,
                    record_id=getattr(db_obj, "id", None),
                )
                span.set_attribute("repository.success", True)
                span.set_attribute("repository.record_id", str(getattr(db_obj, "id", None)))

                return db_obj

            except Exception as e:
                logger.error(
                    "Failed to create record",
                    model=self.model_name,
                    error=str(e),
                    exc_info=True,
                )
                span.set_attribute("repository.success", False)
                span.record_exception(e)
                raise

    async def get(self, id: Any) -> ModelType | None:
        """Get a record by ID.

        Args:
            id: The record ID

        Returns:
            Optional[ModelType]: The record if found, None otherwise

        """
        with tracer.start_as_current_span("repository_get") as span:
            span.set_attribute("repository.model", self.model_name)
            span.set_attribute("repository.operation", "get")
            span.set_attribute("repository.id", str(id))

            try:
                stmt = select(self.model).where(self.model.id == id)  # type: ignore
                result = await self.session.execute(stmt)
                db_obj = result.scalar_one_or_none()

                if db_obj:
                    logger.debug("Record found", model=self.model_name, record_id=str(id))
                    span.set_attribute("repository.found", True)
                else:
                    logger.debug("Record not found", model=self.model_name, record_id=str(id))
                    span.set_attribute("repository.found", False)

                return db_obj

            except Exception as e:
                logger.error(
                    "Failed to get record",
                    model=self.model_name,
                    record_id=str(id),
                    error=str(e),
                    exc_info=True,
                )
                span.set_attribute("repository.success", False)
                span.record_exception(e)
                raise

    async def get_multi(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: list[BinaryExpression] | None = None,
        order_by: Any | None = None,
    ) -> Sequence[ModelType]:
        """Get multiple records with pagination and filtering.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: List of filter conditions
            order_by: Column to order by

        Returns:
            Sequence[ModelType]: List of records

        """
        with tracer.start_as_current_span("repository_get_multi") as span:
            span.set_attribute("repository.model", self.model_name)
            span.set_attribute("repository.operation", "get_multi")
            span.set_attribute("repository.skip", skip)
            span.set_attribute("repository.limit", limit)

            try:
                stmt = select(self.model)

                # Apply filters
                if filters:
                    for filter_condition in filters:
                        stmt = stmt.where(filter_condition)
                    span.set_attribute("repository.filters_count", len(filters))

                # Apply ordering
                if order_by is not None:
                    stmt = stmt.order_by(order_by)

                # Apply pagination
                stmt = stmt.offset(skip).limit(limit)

                result = await self.session.execute(stmt)
                records = result.scalars().all()

                logger.debug(
                    "Records retrieved",
                    model=self.model_name,
                    count=len(records),
                    skip=skip,
                    limit=limit,
                )
                span.set_attribute("repository.records_count", len(records))

                return records

            except Exception as e:
                logger.error(
                    "Failed to get multiple records",
                    model=self.model_name,
                    error=str(e),
                    exc_info=True,
                )
                span.set_attribute("repository.success", False)
                span.record_exception(e)
                raise

    async def update(self, id: Any, obj_in: UpdateSchemaType | dict[str, Any]) -> ModelType | None:
        """Update a record.

        Args:
            id: The record ID
            obj_in: The data to update the record with

        Returns:
            Optional[ModelType]: The updated record if found, None otherwise

        """
        with tracer.start_as_current_span("repository_update") as span:
            span.set_attribute("repository.model", self.model_name)
            span.set_attribute("repository.operation", "update")
            span.set_attribute("repository.id", str(id))

            try:
                # Get existing record
                db_obj = await self.get(id)
                if not db_obj:
                    logger.warning("Record not found for update", model=self.model_name, record_id=str(id))
                    span.set_attribute("repository.found", False)
                    return None

                # Convert to dict if it's a Pydantic model
                if hasattr(obj_in, "model_dump"):
                    update_data = obj_in.model_dump(exclude_unset=True)  # type: ignore
                elif isinstance(obj_in, dict):
                    update_data = obj_in
                else:
                    raise ValueError("Invalid input type")

                # Update fields
                for field, value in update_data.items():
                    if hasattr(db_obj, field):
                        setattr(db_obj, field, value)

                await self.session.flush()
                await self.session.refresh(db_obj)

                logger.info(
                    "Record updated successfully",
                    model=self.model_name,
                    record_id=str(id),
                    updated_fields=list(update_data.keys()),
                )
                span.set_attribute("repository.success", True)
                span.set_attribute("repository.updated_fields_count", len(update_data))

                return db_obj

            except Exception as e:
                logger.error(
                    "Failed to update record",
                    model=self.model_name,
                    record_id=str(id),
                    error=str(e),
                    exc_info=True,
                )
                span.set_attribute("repository.success", False)
                span.record_exception(e)
                raise

    async def delete(self, id: Any) -> bool:
        """Delete a record.

        Args:
            id: The record ID

        Returns:
            bool: True if record was deleted, False if not found

        """
        with tracer.start_as_current_span("repository_delete") as span:
            span.set_attribute("repository.model", self.model_name)
            span.set_attribute("repository.operation", "delete")
            span.set_attribute("repository.id", str(id))

            try:
                stmt = delete(self.model).where(self.model.id == id)  # type: ignore
                result = await self.session.execute(stmt)

                deleted = result.rowcount > 0
                if deleted:
                    logger.info("Record deleted successfully", model=self.model_name, record_id=str(id))
                    span.set_attribute("repository.deleted", True)
                else:
                    logger.warning("Record not found for deletion", model=self.model_name, record_id=str(id))
                    span.set_attribute("repository.deleted", False)

                return deleted

            except Exception as e:
                logger.error(
                    "Failed to delete record",
                    model=self.model_name,
                    record_id=str(id),
                    error=str(e),
                    exc_info=True,
                )
                span.set_attribute("repository.success", False)
                span.record_exception(e)
                raise

    async def count(self, filters: list[BinaryExpression] | None = None) -> int:
        """Count records matching the given filters.

        Args:
            filters: List of filter conditions

        Returns:
            int: Number of matching records

        """
        with tracer.start_as_current_span("repository_count") as span:
            span.set_attribute("repository.model", self.model_name)
            span.set_attribute("repository.operation", "count")

            try:
                stmt = select(func.count(self.model.id))  # type: ignore

                # Apply filters
                if filters:
                    for filter_condition in filters:
                        stmt = stmt.where(filter_condition)
                    span.set_attribute("repository.filters_count", len(filters))

                result = await self.session.execute(stmt)
                count = result.scalar() or 0

                logger.debug("Records counted", model=self.model_name, count=count)
                span.set_attribute("repository.count", count)

                return count

            except Exception as e:
                logger.error(
                    "Failed to count records",
                    model=self.model_name,
                    error=str(e),
                    exc_info=True,
                )
                span.set_attribute("repository.success", False)
                span.record_exception(e)
                raise

    async def exists(self, id: Any) -> bool:
        """Check if a record exists.

        Args:
            id: The record ID

        Returns:
            bool: True if record exists, False otherwise

        """
        with tracer.start_as_current_span("repository_exists") as span:
            span.set_attribute("repository.model", self.model_name)
            span.set_attribute("repository.operation", "exists")
            span.set_attribute("repository.id", str(id))

            try:
                stmt = select(self.model.id).where(self.model.id == id).limit(1)  # type: ignore
                result = await self.session.execute(stmt)
                exists = result.scalar() is not None

                logger.debug("Record existence check", model=self.model_name, record_id=str(id), exists=exists)
                span.set_attribute("repository.exists", exists)

                return exists

            except Exception as e:
                logger.error(
                    "Failed to check record existence",
                    model=self.model_name,
                    record_id=str(id),
                    error=str(e),
                    exc_info=True,
                )
                span.set_attribute("repository.success", False)
                span.record_exception(e)
                raise

    async def bulk_create(self, objs_in: list[CreateSchemaType | dict[str, Any]]) -> list[ModelType]:
        """Create multiple records in bulk.

        Args:
            objs_in: List of data to create records with

        Returns:
            list[ModelType]: List of created records

        """
        with tracer.start_as_current_span("repository_bulk_create") as span:
            span.set_attribute("repository.model", self.model_name)
            span.set_attribute("repository.operation", "bulk_create")
            span.set_attribute("repository.records_count", len(objs_in))

            try:
                db_objs = []
                for obj_in in objs_in:
                    # Convert to dict if it's a Pydantic model
                    if hasattr(obj_in, "model_dump"):
                        obj_data = obj_in.model_dump(exclude_unset=True)  # type: ignore
                    elif isinstance(obj_in, dict):
                        obj_data = obj_in
                    else:
                        raise ValueError("Invalid input type")

                    db_obj = self.model(**obj_data)
                    db_objs.append(db_obj)

                self.session.add_all(db_objs)
                await self.session.flush()

                # Refresh all objects to get their IDs
                for db_obj in db_objs:
                    await self.session.refresh(db_obj)

                logger.info(
                    "Bulk records created successfully",
                    model=self.model_name,
                    records_count=len(db_objs),
                )
                span.set_attribute("repository.success", True)
                span.set_attribute("repository.created_count", len(db_objs))

                return db_objs

            except Exception as e:
                logger.error(
                    "Failed to bulk create records",
                    model=self.model_name,
                    records_count=len(objs_in),
                    error=str(e),
                    exc_info=True,
                )
                span.set_attribute("repository.success", False)
                span.record_exception(e)
                raise

    async def bulk_update(self, updates: dict[Any, dict[str, Any]]) -> int:
        """Update multiple records in bulk.

        Args:
            updates: Dictionary mapping record IDs to update data

        Returns:
            int: Number of updated records

        """
        with tracer.start_as_current_span("repository_bulk_update") as span:
            span.set_attribute("repository.model", self.model_name)
            span.set_attribute("repository.operation", "bulk_update")
            span.set_attribute("repository.records_count", len(updates))

            try:
                updated_count = 0
                for record_id, update_data in updates.items():
                    stmt = update(self.model).where(self.model.id == record_id).values(**update_data)  # type: ignore
                    result = await self.session.execute(stmt)
                    updated_count += result.rowcount

                logger.info(
                    "Bulk records updated successfully",
                    model=self.model_name,
                    updated_count=updated_count,
                    requested_count=len(updates),
                )
                span.set_attribute("repository.success", True)
                span.set_attribute("repository.updated_count", updated_count)

                return updated_count

            except Exception as e:
                logger.error(
                    "Failed to bulk update records",
                    model=self.model_name,
                    records_count=len(updates),
                    error=str(e),
                    exc_info=True,
                )
                span.set_attribute("repository.success", False)
                span.record_exception(e)
                raise

    async def bulk_delete(self, ids: list[Any]) -> int:
        """Delete multiple records in bulk.

        Args:
            ids: List of record IDs to delete

        Returns:
            int: Number of deleted records

        """
        with tracer.start_as_current_span("repository_bulk_delete") as span:
            span.set_attribute("repository.model", self.model_name)
            span.set_attribute("repository.operation", "bulk_delete")
            span.set_attribute("repository.ids_count", len(ids))

            try:
                stmt = delete(self.model).where(self.model.id.in_(ids))  # type: ignore
                result = await self.session.execute(stmt)

                deleted_count = result.rowcount
                logger.info(
                    "Bulk records deleted successfully",
                    model=self.model_name,
                    deleted_count=deleted_count,
                    requested_count=len(ids),
                )
                span.set_attribute("repository.success", True)
                span.set_attribute("repository.deleted_count", deleted_count)

                return deleted_count

            except Exception as e:
                logger.error(
                    "Failed to bulk delete records",
                    model=self.model_name,
                    ids_count=len(ids),
                    error=str(e),
                    exc_info=True,
                )
                span.set_attribute("repository.success", False)
                span.record_exception(e)
                raise
