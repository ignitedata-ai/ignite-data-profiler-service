"""MySQL introspection SQL queries for the data profiler.

All queries use DB-API ``%s`` positional placeholders (aiomysql style).
Table and schema identifiers in sample-data queries are escaped with
backticks by the service layer — never interpolated directly from
user input.
"""

DATABASE_METADATA = """
SELECT
    DATABASE()                                              AS db_name,
    VERSION()                                               AS version,
    @@character_set_database                                AS encoding,
    COALESCE(SUM(data_length + index_length), 0)            AS size_bytes
FROM information_schema.tables
WHERE table_schema = DATABASE()
"""

TABLES_IN_SCHEMA = """
SELECT
    table_name                              AS table_name,
    table_schema                            AS table_schema,
    table_comment                           AS description,
    data_length                             AS size_bytes,
    data_length + index_length              AS total_size_bytes
FROM information_schema.tables
WHERE table_schema = %s
  AND table_type   = 'BASE TABLE'
ORDER BY table_name
"""

ROW_COUNT_FAST = """
SELECT table_rows AS row_count
FROM information_schema.tables
WHERE table_schema = %s
  AND table_name   = %s
"""

# Used as a fallback when table_rows is NULL or stale.
# Identifiers are backtick-quoted by the service layer — never interpolated directly.
ROW_COUNT_EXACT = "SELECT COUNT(1) AS row_count FROM `{schema}`.`{table}`"

COLUMNS_FOR_TABLE = """
SELECT
    column_name                             AS column_name,
    ordinal_position                        AS ordinal_position,
    data_type                               AS data_type,
    (is_nullable = 'YES')                   AS is_nullable,
    column_default                          AS column_default,
    character_maximum_length                AS character_maximum_length,
    numeric_precision                       AS numeric_precision,
    numeric_scale                           AS numeric_scale,
    column_comment                          AS description,
    column_type                             AS full_column_type
FROM information_schema.columns
WHERE table_schema  = %s
  AND table_name    = %s
ORDER BY ordinal_position
"""

PRIMARY_KEY_COLUMNS = """
SELECT column_name      AS column_name
FROM information_schema.key_column_usage
WHERE table_schema      = %s
  AND table_name        = %s
  AND constraint_name   = 'PRIMARY'
ORDER BY ordinal_position
"""

# GROUP_CONCAT produces a comma-separated string of column names in index order.
# The service layer splits this string back into a list.
INDEXES_FOR_TABLE = """
SELECT
    index_name                                                      AS index_name,
    MAX(non_unique = 0)                                             AS is_unique,
    MAX(index_name = 'PRIMARY')                                     AS is_primary,
    MIN(index_type)                                                 AS index_type,
    GROUP_CONCAT(column_name ORDER BY seq_in_index SEPARATOR ',')   AS columns_str
FROM information_schema.statistics
WHERE table_schema  = %s
  AND table_name    = %s
GROUP BY index_name
ORDER BY index_name
"""

FOREIGN_KEYS_FOR_TABLE = """
SELECT
    rc.constraint_name                      AS constraint_name,
    kcu.column_name                         AS column_name,
    kcu.referenced_table_schema             AS foreign_table_schema,
    kcu.referenced_table_name              AS foreign_table_name,
    kcu.referenced_column_name             AS foreign_column_name,
    rc.update_rule                          AS update_rule,
    rc.delete_rule                          AS delete_rule
FROM information_schema.referential_constraints rc
JOIN information_schema.key_column_usage kcu
    ON  kcu.constraint_name   = rc.constraint_name
    AND kcu.constraint_schema = rc.constraint_schema
WHERE rc.constraint_schema  = %s
  AND kcu.table_name        = %s
  AND kcu.referenced_table_name IS NOT NULL
ORDER BY rc.constraint_name, kcu.ordinal_position
"""

DATA_FRESHNESS_FOR_TABLE = """
SELECT
    update_time                             AS update_time,
    check_time                              AS check_time,
    create_time                             AS create_time
FROM information_schema.tables
WHERE table_schema  = %s
  AND table_name    = %s
"""

VIEWS_IN_SCHEMA = """
SELECT
    table_name                              AS table_name,
    table_schema                            AS table_schema,
    view_definition                         AS definition
FROM information_schema.views
WHERE table_schema = %s
ORDER BY table_name
"""

# Sample data query is constructed at runtime by MySQLProfiler:
#   f'SELECT * FROM `{safe_schema}`.`{safe_table}` ORDER BY RAND() LIMIT %s'
# Schema and table identifiers are backtick-quoted (` → ``) to prevent
# SQL injection from names containing special characters.
