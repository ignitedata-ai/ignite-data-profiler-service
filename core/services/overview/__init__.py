"""Lightweight database overview services."""

from .base import BaseOverviewService
from .bigquery import BigQueryOverviewService
from .databricks import DatabricksOverviewService
from .mysql import MySQLOverviewService
from .postgres import PostgresOverviewService
from .redshift import RedshiftOverviewService
from .snowflake import SnowflakeOverviewService

OVERVIEW_REGISTRY: dict[str, BaseOverviewService] = {
    "postgres": PostgresOverviewService(),
    "mysql": MySQLOverviewService(),
    "snowflake": SnowflakeOverviewService(),
    "databricks": DatabricksOverviewService(),
    "bigquery": BigQueryOverviewService(),
    "redshift": RedshiftOverviewService(),
}

__all__ = [
    "OVERVIEW_REGISTRY",
    "BaseOverviewService",
    "BigQueryOverviewService",
    "DatabricksOverviewService",
    "MySQLOverviewService",
    "PostgresOverviewService",
    "RedshiftOverviewService",
    "SnowflakeOverviewService",
]
