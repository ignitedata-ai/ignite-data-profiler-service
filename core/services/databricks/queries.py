"""Databricks introspection SQL queries for the data profiler.

All queries use ``?`` positional placeholders (databricks-sql-connector style).
Column aliases use backtick quoting.  Table and schema identifiers in
sample-data and column-stats queries are escaped with backticks by the
service layer — never interpolated directly from user input.
"""

DATABASE_METADATA = """
SELECT
    CURRENT_CATALOG()   AS `db_name`,
    ''                  AS `version`,
    'UTF-8'             AS `encoding`,
    0                   AS `size_bytes`
"""

# The service builds the ``?, ?, ...`` IN-list placeholder string
# dynamically from the length of ``exclude_schemas`` before executing.
SCHEMAS = """
SELECT
    SCHEMA_NAME     AS `schema_name`,
    CATALOG_NAME    AS `catalog_name`
FROM INFORMATION_SCHEMA.SCHEMATA
WHERE SCHEMA_NAME NOT IN ({placeholders})
ORDER BY SCHEMA_NAME
"""

TABLES_IN_SCHEMA = """
SELECT
    TABLE_NAME      AS `table_name`,
    TABLE_SCHEMA    AS `table_schema`,
    TABLE_OWNER     AS `owner`,
    NULL            AS `size_bytes`,
    NULL            AS `total_size_bytes`,
    COMMENT         AS `description`
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = ?
  AND TABLE_TYPE   = 'BASE TABLE'
ORDER BY TABLE_NAME
"""

# Databricks INFORMATION_SCHEMA.TABLES does not expose a reliable row count
# estimate.  Always use exact COUNT(1).
# Identifiers are backtick-quoted by the service layer.
ROW_COUNT_EXACT = "SELECT COUNT(1) AS `row_count` FROM `{schema}`.`{table}`"

COLUMNS_FOR_TABLE = """
SELECT
    COLUMN_NAME                 AS `column_name`,
    ORDINAL_POSITION            AS `ordinal_position`,
    DATA_TYPE                   AS `data_type`,
    (IS_NULLABLE = 'YES')       AS `is_nullable`,
    COLUMN_DEFAULT              AS `column_default`,
    CHARACTER_MAXIMUM_LENGTH    AS `character_maximum_length`,
    NUMERIC_PRECISION           AS `numeric_precision`,
    NUMERIC_SCALE               AS `numeric_scale`,
    COMMENT                     AS `description`
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = ?
  AND TABLE_NAME   = ?
ORDER BY ORDINAL_POSITION
"""

PRIMARY_KEY_COLUMNS = """
SELECT kcu.COLUMN_NAME AS `column_name`
FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
    ON  kcu.CONSTRAINT_NAME   = tc.CONSTRAINT_NAME
    AND kcu.CONSTRAINT_SCHEMA = tc.CONSTRAINT_SCHEMA
WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
  AND tc.TABLE_SCHEMA    = ?
  AND tc.TABLE_NAME      = ?
"""

FOREIGN_KEYS_FOR_TABLE = """
SELECT
    rc.CONSTRAINT_NAME                  AS `constraint_name`,
    kcu.COLUMN_NAME                     AS `column_name`,
    kcu2.TABLE_SCHEMA                   AS `foreign_table_schema`,
    kcu2.TABLE_NAME                     AS `foreign_table_name`,
    kcu2.COLUMN_NAME                    AS `foreign_column_name`,
    rc.UPDATE_RULE                      AS `update_rule`,
    rc.DELETE_RULE                      AS `delete_rule`
FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
    ON  kcu.CONSTRAINT_NAME   = rc.CONSTRAINT_NAME
    AND kcu.CONSTRAINT_SCHEMA = rc.CONSTRAINT_SCHEMA
JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc2
    ON  tc2.CONSTRAINT_NAME   = rc.UNIQUE_CONSTRAINT_NAME
    AND tc2.CONSTRAINT_SCHEMA = rc.UNIQUE_CONSTRAINT_SCHEMA
JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu2
    ON  kcu2.CONSTRAINT_NAME   = rc.UNIQUE_CONSTRAINT_NAME
    AND kcu2.ORDINAL_POSITION  = kcu.ORDINAL_POSITION
WHERE rc.CONSTRAINT_SCHEMA = ?
  AND kcu.TABLE_NAME       = ?
ORDER BY rc.CONSTRAINT_NAME, kcu.ORDINAL_POSITION
"""

DATA_FRESHNESS_FOR_TABLE = """
SELECT
    LAST_ALTERED    AS `last_altered`,
    CREATED         AS `created`
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = ?
  AND TABLE_NAME   = ?
"""

VIEWS_IN_SCHEMA = """
SELECT
    TABLE_NAME          AS `table_name`,
    TABLE_SCHEMA        AS `table_schema`,
    TABLE_OWNER         AS `owner`,
    VIEW_DEFINITION     AS `definition`
FROM INFORMATION_SCHEMA.VIEWS
WHERE TABLE_SCHEMA = ?
ORDER BY TABLE_NAME
"""

# Sample data query is constructed at runtime by DatabricksProfiler:
#   f"SELECT * FROM `{safe_schema}`.`{safe_table}` ORDER BY RAND() LIMIT ?"
# Schema and table identifiers are backtick-quoted (` → ``) to prevent
# SQL injection from names containing special characters.
