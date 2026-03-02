"""BigQuery introspection SQL queries for the data profiler.

BigQuery INFORMATION_SCHEMA is scoped per-dataset for table/column metadata.
Dataset-level queries use ``.format()`` to inject the escaped dataset name
into the FROM clause.  Project-level queries (SCHEMATA) use ``%s`` placeholders
for filter parameters.

Column aliases are unquoted lowercase — the BigQuery DB-API connector returns
column names from ``cursor.description`` directly.
"""

# Project-level query to list datasets (schemas).
# The service builds the ``%s, %s, ...`` IN-list placeholder string
# dynamically from the length of ``exclude_schemas`` before executing.
SCHEMAS = """
SELECT
    schema_name     AS schema_name
FROM INFORMATION_SCHEMA.SCHEMATA
WHERE schema_name NOT IN ({placeholders})
ORDER BY schema_name
"""

# Dataset-scoped queries — the ``{dataset}`` placeholder is injected by the
# service layer using backtick-escaped dataset names.

TABLES_IN_DATASET = """
SELECT
    t.table_name                AS table_name,
    t.table_schema              AS table_schema,
    NULL                        AS owner,
    t.size_bytes                AS size_bytes,
    t.size_bytes                AS total_size_bytes,
    opt.option_value            AS description
FROM `{dataset}`.INFORMATION_SCHEMA.TABLES t
LEFT JOIN `{dataset}`.INFORMATION_SCHEMA.TABLE_OPTIONS opt
    ON  t.table_schema  = opt.table_schema
    AND t.table_name    = opt.table_name
    AND opt.option_name = 'description'
WHERE t.table_type = 'BASE TABLE'
ORDER BY t.table_name
"""

ROW_COUNT_FAST = """
SELECT row_count AS row_count
FROM `{dataset}`.INFORMATION_SCHEMA.TABLES
WHERE table_schema = %s
  AND table_name   = %s
"""

# Used as a fallback when row_count is NULL.
# Identifiers are backtick-quoted by the service layer.
ROW_COUNT_EXACT = "SELECT COUNT(1) AS row_count FROM `{dataset}`.`{table}`"

COLUMNS_FOR_TABLE = """
SELECT
    column_name                 AS column_name,
    ordinal_position            AS ordinal_position,
    data_type                   AS data_type,
    (is_nullable = 'YES')       AS is_nullable,
    column_default              AS column_default,
    character_maximum_length    AS character_maximum_length,
    NULL                        AS numeric_precision,
    NULL                        AS numeric_scale
FROM `{dataset}`.INFORMATION_SCHEMA.COLUMNS
WHERE table_schema = %s
  AND table_name   = %s
ORDER BY ordinal_position
"""

COLUMN_DESCRIPTIONS = """
SELECT
    column_name     AS column_name,
    description     AS description
FROM `{dataset}`.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS
WHERE table_schema = %s
  AND table_name   = %s
  AND field_path   = column_name
"""

DATA_FRESHNESS_FOR_TABLE = """
SELECT
    creation_time   AS created
FROM `{dataset}`.INFORMATION_SCHEMA.TABLES
WHERE table_schema = %s
  AND table_name   = %s
"""

VIEWS_IN_DATASET = """
SELECT
    table_name          AS table_name,
    table_schema        AS table_schema,
    NULL                AS owner,
    view_definition     AS definition
FROM `{dataset}`.INFORMATION_SCHEMA.VIEWS
WHERE table_schema = %s
ORDER BY table_name
"""

# Sample data query is constructed at runtime by BigQueryProfiler:
#   f"SELECT * FROM `{safe_dataset}`.`{safe_table}` ORDER BY RAND() LIMIT %s"
# Dataset and table identifiers are backtick-quoted (` → ``) to prevent
# SQL injection from names containing special characters.
