"""Snowflake introspection SQL queries for the data profiler.

All queries use ``%s`` positional placeholders (snowflake-connector-python
pyformat style).  Column aliases are quoted lowercase (e.g. ``AS "db_name"``)
because Snowflake uppercases unquoted identifiers by default.

Table and schema identifiers in sample-data queries are escaped with
double-quotes by the service layer — never interpolated directly from
user input.
"""

DATABASE_METADATA = """
SELECT
    CURRENT_DATABASE()           AS "db_name",
    CURRENT_VERSION()            AS "version",
    'UTF-8'                      AS "encoding",
    COALESCE(SUM(ACTIVE_BYTES), 0) AS "size_bytes"
FROM INFORMATION_SCHEMA.TABLE_STORAGE_METRICS
WHERE TABLE_CATALOG = CURRENT_DATABASE()
"""

# The service builds the ``%s, %s, ...`` IN-list placeholder string
# dynamically from the length of ``exclude_schemas`` before executing.
SCHEMAS = """
SELECT
    SCHEMA_NAME     AS "schema_name",
    CATALOG_NAME    AS "catalog_name"
FROM INFORMATION_SCHEMA.SCHEMATA
WHERE SCHEMA_NAME NOT IN ({placeholders})
ORDER BY SCHEMA_NAME
"""

TABLES_IN_SCHEMA = """
SELECT
    TABLE_NAME              AS "table_name",
    TABLE_SCHEMA            AS "table_schema",
    TABLE_OWNER             AS "owner",
    BYTES                   AS "size_bytes",
    BYTES                   AS "total_size_bytes",
    COMMENT                 AS "description"
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = %s
  AND TABLE_TYPE   = 'BASE TABLE'
ORDER BY TABLE_NAME
"""

ROW_COUNT_FAST = """
SELECT ROW_COUNT AS "row_count"
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = %s
  AND TABLE_NAME   = %s
"""

# Used as a fallback when ROW_COUNT is NULL.
# Identifiers are double-quoted by the service layer — never interpolated directly.
ROW_COUNT_EXACT = 'SELECT COUNT(1) AS "row_count" FROM "{schema}"."{table}"'

COLUMNS_FOR_TABLE = """
SELECT
    COLUMN_NAME                 AS "column_name",
    ORDINAL_POSITION            AS "ordinal_position",
    DATA_TYPE                   AS "data_type",
    (IS_NULLABLE = 'YES')       AS "is_nullable",
    COLUMN_DEFAULT              AS "column_default",
    CHARACTER_MAXIMUM_LENGTH    AS "character_maximum_length",
    NUMERIC_PRECISION           AS "numeric_precision",
    NUMERIC_SCALE               AS "numeric_scale",
    COMMENT                     AS "description"
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = %s
  AND TABLE_NAME   = %s
ORDER BY ORDINAL_POSITION
"""

SHOW_PRIMARY_KEYS_FOR_TABLE = 'SHOW PRIMARY KEYS IN TABLE "{schema}"."{table}"'

SHOW_IMPORTED_KEYS_FOR_TABLE = 'SHOW IMPORTED KEYS IN TABLE "{schema}"."{table}"'

DATA_FRESHNESS_FOR_TABLE = """
SELECT
    LAST_DDL         AS "last_ddl",
    LAST_ALTERED     AS "last_altered",
    CREATED          AS "created"
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = %s
  AND TABLE_NAME   = %s
"""

VIEWS_IN_SCHEMA = """
SELECT
    TABLE_NAME              AS "table_name",
    TABLE_SCHEMA            AS "table_schema",
    TABLE_OWNER             AS "owner",
    VIEW_DEFINITION         AS "definition"
FROM INFORMATION_SCHEMA.VIEWS
WHERE TABLE_SCHEMA = %s
ORDER BY TABLE_NAME
"""

# Sample data query is constructed at runtime by SnowflakeProfiler:
#   f'SELECT * FROM "{safe_schema}"."{safe_table}" ORDER BY RANDOM() LIMIT %s'
# Schema and table identifiers are double-quoted (" → "") to prevent
# SQL injection from names containing special characters.
