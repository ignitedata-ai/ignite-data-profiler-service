"""Profiler request and response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from ignite_data_connectors import BigQueryConfig, DatabricksConfig, MySQLConfig, PostgresConfig, RedshiftConfig, SnowflakeConfig
from pydantic import BaseModel, ConfigDict, Field

# ── Connection / Request ───────────────────────────────────────────────────────


class ProfilingConfig(BaseModel):
    """Profiling behaviour configuration."""

    include_schemas: list[str] | None = Field(
        default=None,
        description="Schemas to profile; None means all non-system schemas",
    )
    exclude_schemas: list[str] = Field(
        default=["pg_catalog", "information_schema", "pg_toast"],
        description="Schemas to always exclude",
    )
    include_tables: list[str] | None = Field(
        default=None,
        description="Tables to include; None means all tables in selected schemas",
    )
    exclude_tables: list[str] = Field(
        default=[],
        description="Table names to exclude",
    )
    sample_size: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of random sample rows to fetch per table",
    )
    include_sample_data: bool = Field(default=True, description="Fetch random sample rows")
    include_row_counts: bool = Field(default=True, description="Fetch row counts from pg_class statistics")
    include_indexes: bool = Field(default=True, description="Extract index metadata")
    include_relationships: bool = Field(default=True, description="Extract foreign key relationships")
    include_data_freshness: bool = Field(default=True, description="Fetch last analyze/vacuum timestamps")
    timeout_seconds: int = Field(
        default=1000,
        ge=1,
        le=3600,
        description="Overall profiling timeout in seconds",
    )
    augment_descriptions: bool = Field(
        default=False,
        description="Call LLM to generate business descriptions for tables. Requires LLM_ENABLED=True server-side. Non-fatal.",
    )
    llm_batch_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of tables to augment concurrently per LLM batch",
    )
    augment_column_descriptions: bool = Field(
        default=False,
        description=(
            "Call LLM to generate business descriptions for each column.Requires LLM_ENABLED=True server-side. Non-fatal."
        ),
    )
    llm_column_batch_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of columns to augment concurrently per LLM batch",
    )
    augment_glossary: bool = Field(
        default=False,
        description="Call LLM to infer business glossary terms for each table. Requires LLM_ENABLED=True server-side. Non-fatal.",
    )
    llm_glossary_batch_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of tables to process concurrently for glossary inference",
    )
    infer_kpis: bool = Field(
        default=False,
        description=(
            "Call LLM to infer business KPIs via Map-Reduce pipeline (cluster → generate → synthesize)."
            "Requires LLM_ENABLED=True server-side. Non-fatal. Best run after augment_descriptions=True for richer context."
        ),
    )
    llm_kpi_max_domains: int = Field(
        default=10,
        ge=1,
        le=50,
        description=(
            "Maximum number of business domain clusters for Phase A (clustering)."
            "Larger values produce finer-grained domains but require more Phase B calls."
        ),
    )
    llm_kpis_per_domain: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of KPIs to generate per business domain in Phase B (generation).",
    )
    include_column_stats: bool = Field(
        default=False,
        description=(
            "Compute column-level statistics (null/distinct counts, "
            "type-specific metrics, distributions). Requires querying actual table data."
        ),
    )
    top_values_limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of top values to return in frequency distributions",
    )
    top_values_cardinality_threshold: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="String columns with distinct_count above this threshold will not have top_values computed",
    )
    max_concurrent_tables: int = Field(
        default=3,
        ge=1,
        le=50,
        description=(
            "Maximum number of tables to profile concurrently per schema. "
            "Lower values reduce connection pool pressure; raise if the pool "
            "is large and latency matters more than connection count."
        ),
    )
    detect_filter_columns: bool = Field(
        default=False,
        description=(
            "Detect columns suitable for analytical filtering (status fields, date ranges, categories, etc.). "
            "Runs a multi-stage pipeline using schema signals and statistical heuristics. "
            "Best results when include_column_stats=True. Non-fatal."
        ),
    )

    # ── Per-request LLM credentials (optional) ────────────────────────────────
    portkey_api_key: str | None = Field(
        default=None,
        description=(
            "Portkey API key for this request. When provided, overrides the server-side "
            "PORTKEY_API_KEY env var. Falls back to server env when null."
        ),
    )
    portkey_virtual_key: str | None = Field(
        default=None,
        description=(
            "Portkey virtual key (LLM routing key) for this request. When provided, overrides the "
            "server-side PORTKEY_VIRTUAL_KEY env var. Falls back to server env when null."
        ),
    )


class PostgresProfilingRequest(BaseModel):
    """Profiling request targeting a PostgreSQL database."""

    datasource_type: Literal["postgres"] = Field(..., description="Datasource type discriminator")
    connection: PostgresConfig
    config: ProfilingConfig = Field(default_factory=ProfilingConfig)


class MySQLProfilingRequest(BaseModel):
    """Profiling request targeting a MySQL database."""

    datasource_type: Literal["mysql"] = Field(..., description="Datasource type discriminator")
    connection: MySQLConfig
    config: ProfilingConfig = Field(default_factory=ProfilingConfig)


class SnowflakeProfilingRequest(BaseModel):
    """Profiling request targeting a Snowflake database."""

    datasource_type: Literal["snowflake"] = Field(..., description="Datasource type discriminator")
    connection: SnowflakeConfig
    config: ProfilingConfig = Field(default_factory=ProfilingConfig)


class DatabricksProfilingRequest(BaseModel):
    """Profiling request targeting a Databricks catalog."""

    datasource_type: Literal["databricks"] = Field(..., description="Datasource type discriminator")
    connection: DatabricksConfig
    config: ProfilingConfig = Field(default_factory=ProfilingConfig)


class BigQueryProfilingRequest(BaseModel):
    """Profiling request targeting a BigQuery project."""

    datasource_type: Literal["bigquery"] = Field(..., description="Datasource type discriminator")
    connection: BigQueryConfig
    config: ProfilingConfig = Field(default_factory=ProfilingConfig)


class RedshiftProfilingRequest(BaseModel):
    """Profiling request targeting an Amazon Redshift database."""

    datasource_type: Literal["redshift"] = Field(..., description="Datasource type discriminator")
    connection: RedshiftConfig
    config: ProfilingConfig = Field(default_factory=ProfilingConfig)


# Discriminated union — add new datasource types here in future phases.
# S3FileProfilingRequest is imported at the bottom of this file to avoid a circular
# import (s3.py lazily imports ProfilingConfig from here).
from core.api.v1.schemas.s3 import S3FileProfilingRequest  # noqa: E402

ProfilingRequest = Annotated[
    PostgresProfilingRequest
    | MySQLProfilingRequest
    | SnowflakeProfilingRequest
    | DatabricksProfilingRequest
    | BigQueryProfilingRequest
    | RedshiftProfilingRequest
    | S3FileProfilingRequest,
    Field(discriminator="datasource_type"),
]


# ── Response ───────────────────────────────────────────────────────────────────


class GlossaryTerm(BaseModel):
    """A single business glossary term inferred for a table."""

    business_term: str
    description: str
    synonyms: list[str] = Field(default_factory=list)


class KPITerm(BaseModel):
    """A single business KPI inferred from cross-table domain analysis."""

    name: str
    description: str
    calculation: str | None = None
    linked_columns: list[str] = Field(default_factory=list)


class FilterColumnInfo(BaseModel):
    """A column identified as suitable for analytical filtering."""

    filter_name: str = Field(description="Name of the filter")
    column_name: str = Field(description="Name of the filter column")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0.0–1.0)")
    confidence_source: str = Field(
        description="How the confidence was determined: heuristic_only, llm_agreed, llm_adjusted, or flagged_for_review",
    )
    filter_type: str = Field(
        description="Classification: temporal, boolean, categorical, or range",
    )
    reasoning: str | None = Field(default=None, description="Brief explanation of why this column was selected")


class TopValueEntry(BaseModel):
    """A single value in a frequency distribution."""

    value: str | None
    count: int
    percentage: float


class NumericColumnStats(BaseModel):
    """Statistical metrics for numeric columns."""

    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    stddev: float | None = None
    variance: float | None = None
    sum: float | None = None
    p5: float | None = None
    p25: float | None = None
    p75: float | None = None
    p95: float | None = None
    zero_count: int | None = None
    negative_count: int | None = None
    outlier_count: int | None = None


class StringColumnStats(BaseModel):
    """Statistical metrics for string/text columns."""

    min_length: int | None = None
    max_length: int | None = None
    avg_length: float | None = None
    empty_count: int | None = None


class BooleanColumnStats(BaseModel):
    """Statistical metrics for boolean columns."""

    true_count: int | None = None
    false_count: int | None = None
    true_percentage: float | None = None


class TemporalColumnStats(BaseModel):
    """Statistical metrics for date/time/timestamp columns."""

    min: str | None = None
    max: str | None = None


class ColumnStatistics(BaseModel):
    """Column-level statistics varying by data type category."""

    total_count: int
    null_count: int
    null_percentage: float
    distinct_count: int
    distinct_percentage: float
    numeric: NumericColumnStats | None = None
    string: StringColumnStats | None = None
    boolean: BooleanColumnStats | None = None
    temporal: TemporalColumnStats | None = None
    top_values: list[TopValueEntry] | None = None


class ColumnMetadata(BaseModel):
    """Metadata for a single table column."""

    name: str
    ordinal_position: int
    data_type: str
    is_nullable: bool
    column_default: str | None
    character_maximum_length: int | None
    numeric_precision: int | None
    numeric_scale: int | None
    is_primary_key: bool = False
    description: str | None = None
    enum_values: list[str] | None = None
    sample_values: list[Any] | None = None
    statistics: ColumnStatistics | None = None


class IndexMetadata(BaseModel):
    """Metadata for a single index."""

    name: str
    columns: list[str]
    is_unique: bool
    is_primary: bool
    index_type: str


class RelationshipMetadata(BaseModel):
    """Metadata for a single foreign key relationship."""

    constraint_name: str
    from_column: str
    to_schema: str
    to_table: str
    to_column: str
    on_update: str
    on_delete: str


class DataFreshnessInfo(BaseModel):
    """Statistics-based data freshness information from pg_stat_user_tables."""

    last_analyze: datetime | None
    last_autoanalyze: datetime | None
    last_vacuum: datetime | None
    last_autovacuum: datetime | None


class TableMetadata(BaseModel):
    """Metadata for a single table.

    Optional fields are ``None`` when the corresponding ``include_*`` flag
    was ``False`` (meaning "not requested"), and an empty list when the flag
    was ``True`` but no data was found.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str
    schema_name: str = Field(alias="schema")
    owner: str
    description: str | None
    row_count: int | None
    size_bytes: int | None
    total_size_bytes: int | None
    data_freshness: DataFreshnessInfo | None
    columns: list[ColumnMetadata]
    indexes: list[IndexMetadata] | None
    relationships: list[RelationshipMetadata] | None
    glossary: list[GlossaryTerm] | None = None
    table_role: str | None = Field(default=None, description="Table classification: fact, dimension, or unknown")
    filter_columns: list[FilterColumnInfo] | None = Field(
        default=None,
        description="Columns identified as suitable for analytical filtering. Present when detect_filter_columns=True.",
    )


class ViewMetadata(BaseModel):
    """Metadata for a single view."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    schema_name: str = Field(alias="schema")
    owner: str
    definition: str | None


class SchemaMetadata(BaseModel):
    """Metadata for a single PostgreSQL schema."""

    name: str
    owner: str
    tables: list[TableMetadata]
    views: list[ViewMetadata]


class DatabaseMetadata(BaseModel):
    """Database-level metadata."""

    name: str
    version: str
    encoding: str
    size_bytes: int


class LLMUsageStats(BaseModel):
    """Aggregated LLM token usage and cost for a profiling run."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(description="Total prompt/input tokens consumed across all LLM calls (from API response).")
    output_tokens: int = Field(description="Total completion/output tokens generated across all LLM calls (from API response).")
    input_cost: float = Field(description="USD cost attributable to input tokens.")
    output_cost: float = Field(description="USD cost attributable to output tokens.")
    total_cost: float = Field(description="Total USD cost (input + output).")
    estimated_text_tokens: int = Field(
        description="Pre-call estimated text token count (tiktoken) — content only, excluding message format overhead."
    )
    estimated_overhead_tokens: int = Field(
        description="Token overhead added by the chat message format (role markers, separators)."
    )
    estimated_total_tokens: int = Field(
        description="Pre-call estimated total tokens (text + overhead) across all LLM calls."
    )
    estimated_message_count: int = Field(
        description="Total number of messages (system + user) sent across all LLM calls."
    )
    total_latency_ms: float = Field(
        description="Total wall-clock latency in milliseconds across all LLM API calls."
    )


class ProfilingResponse(BaseModel):
    """Top-level profiling response."""

    profiled_at: datetime
    database: DatabaseMetadata
    schemas: list[SchemaMetadata]
    kpis: list[KPITerm] | None = None
    llm_usage: LLMUsageStats | None = Field(
        default=None,
        description=(
            "Aggregated LLM token usage and cost for this profiling run "
            "(table descriptions, column descriptions, glossary, KPI inference). "
            "None when LLM augmentation was disabled or cost tracking is unavailable."
        ),
    )
