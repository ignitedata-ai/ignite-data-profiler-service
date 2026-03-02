from .base import BaseProfilerService
from .bigquery import BigQueryProfiler, BigQueryProfilerService
from .databricks import DatabricksProfiler, DatabricksProfilerService
from .mysql import MySQLProfiler, MySQLProfilerService
from .postgres import PostgresProfiler, PostgresProfilerService
from .redshift import RedshiftProfiler, RedshiftProfilerService
from .s3 import S3FileProfilerService
from .snowflake import SnowflakeProfiler, SnowflakeProfilerService

# Register all supported datasource profilers.
PROFILER_REGISTRY = {
    "postgres": PostgresProfilerService(),
    "mysql": MySQLProfilerService(),
    "snowflake": SnowflakeProfilerService(),
    "databricks": DatabricksProfilerService(),
    "bigquery": BigQueryProfilerService(),
    "redshift": RedshiftProfilerService(),
    "s3_file": S3FileProfilerService(),
}


__all__ = [
    "PROFILER_REGISTRY",
    "BaseProfilerService",
    "BigQueryProfiler",
    "BigQueryProfilerService",
    "DatabricksProfiler",
    "DatabricksProfilerService",
    "MySQLProfiler",
    "MySQLProfilerService",
    "PostgresProfiler",
    "PostgresProfilerService",
    "RedshiftProfiler",
    "RedshiftProfilerService",
    "SnowflakeProfiler",
    "SnowflakeProfilerService",
    "S3FileProfilerService",
]
