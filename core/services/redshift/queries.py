"""Redshift introspection SQL queries for the data profiler.

All queries use ``%s`` positional placeholders (pyformat style) as required
by the ``redshift_connector`` driver.  Table and schema identifiers in
sample-data queries are escaped with double-quotes by the service layer —
never interpolated directly from user input.

Key differences from PostgreSQL:
- ``%s`` placeholders instead of ``$1``/``$2``
- ``stv_partitions`` for database size (no ``pg_database_size()``)
- ``pg_class``/``pg_namespace`` for table size (no ``pg_total_relation_size()``)
- ``pg_description`` + ``pg_attribute`` for column comments (no ``col_description()``)
- ``stl_analyze`` for data freshness (no ``pg_stat_user_tables``)
- No index support (Redshift uses sort/distribution keys)
"""

DATABASE_METADATA = """
SELECT
    current_database()                     AS db_name,
    version()                              AS version,
    pg_encoding_to_char(encoding)          AS encoding,
    COALESCE(
        (SELECT SUM(capacity) FROM stv_partitions WHERE part_begin = 0),
        0
    )                                      AS size_bytes
FROM pg_database
WHERE datname = current_database()
"""

# NOTE: The service builds the ``%s, %s, ...`` IN-list placeholder string
# dynamically from the length of ``exclude_schemas`` before executing.
SCHEMAS = """
SELECT
    schema_name,
    schema_owner
FROM information_schema.schemata
WHERE schema_name NOT IN ({placeholders})
ORDER BY schema_name
"""

TABLES_IN_SCHEMA = """
SELECT
    t.table_name,
    t.table_schema,
    u.usename                              AS owner,
    COALESCE(
        (SELECT SUM(b.size_mb) * 1024 * 1024
         FROM (
            SELECT tbl AS id, COUNT(*) AS size_mb
            FROM stv_blocklist
            GROUP BY tbl
         ) b
         WHERE b.id = c.oid),
        0
    )::bigint                              AS size_bytes,
    COALESCE(
        (SELECT SUM(b.size_mb) * 1024 * 1024
         FROM (
            SELECT tbl AS id, COUNT(*) AS size_mb
            FROM stv_blocklist
            GROUP BY tbl
         ) b
         WHERE b.id = c.oid),
        0
    )::bigint                              AS total_size_bytes,
    d.description                          AS description
FROM information_schema.tables t
JOIN pg_class c
    ON  c.relname   = t.table_name
JOIN pg_namespace n
    ON  n.oid       = c.relnamespace
    AND n.nspname   = t.table_schema
JOIN pg_user u
    ON  u.usesysid  = c.relowner
LEFT JOIN pg_description d
    ON  d.objoid    = c.oid
    AND d.objsubid  = 0
WHERE t.table_schema = %s
  AND t.table_type   = 'BASE TABLE'
ORDER BY t.table_name
"""

ROW_COUNT_FAST = """
SELECT reltuples::bigint AS row_count
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relname  = %s
  AND n.nspname  = %s
"""

# Used as a fallback when reltuples = -1 (table never analyzed).
# Identifiers are double-quoted by the service layer — never interpolated directly.
ROW_COUNT_EXACT = 'SELECT COUNT(1) AS row_count FROM "{schema}"."{table}"'

COLUMNS_FOR_TABLE = """
SELECT
    c.column_name,
    c.ordinal_position,
    c.data_type,
    (c.is_nullable = 'YES')               AS is_nullable,
    c.column_default,
    c.character_maximum_length,
    c.numeric_precision,
    c.numeric_scale,
    d.description                          AS description
FROM information_schema.columns c
LEFT JOIN pg_class cls
    ON  cls.relname = c.table_name
LEFT JOIN pg_namespace ns
    ON  ns.oid      = cls.relnamespace
    AND ns.nspname  = c.table_schema
LEFT JOIN pg_attribute a
    ON  a.attrelid  = cls.oid
    AND a.attname   = c.column_name
LEFT JOIN pg_description d
    ON  d.objoid    = cls.oid
    AND d.objsubid  = a.attnum
WHERE c.table_schema = %s
  AND c.table_name   = %s
ORDER BY c.ordinal_position
"""

PRIMARY_KEY_COLUMNS = """
SELECT kcu.column_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
    ON  kcu.constraint_name = tc.constraint_name
    AND kcu.table_schema    = tc.table_schema
    AND kcu.table_name      = tc.table_name
WHERE tc.constraint_type = 'PRIMARY KEY'
  AND tc.table_schema    = %s
  AND tc.table_name      = %s
"""

FOREIGN_KEYS_FOR_TABLE = """
SELECT
    rc.constraint_name,
    kcu.column_name,
    kcu2.table_schema   AS foreign_table_schema,
    kcu2.table_name     AS foreign_table_name,
    kcu2.column_name    AS foreign_column_name,
    rc.update_rule,
    rc.delete_rule
FROM information_schema.referential_constraints rc
JOIN information_schema.key_column_usage kcu
    ON  kcu.constraint_name  = rc.constraint_name
    AND kcu.table_schema     = rc.constraint_schema
JOIN information_schema.key_column_usage kcu2
    ON  kcu2.constraint_name  = rc.unique_constraint_name
    AND kcu2.ordinal_position = kcu.ordinal_position
JOIN information_schema.table_constraints tc
    ON  tc.constraint_name = rc.constraint_name
    AND tc.table_schema    = rc.constraint_schema
WHERE tc.table_schema = %s
  AND tc.table_name   = %s
ORDER BY rc.constraint_name, kcu.ordinal_position
"""

DATA_FRESHNESS_FOR_TABLE = """
SELECT
    MAX(a.run_start_time)   AS last_analyze,
    NULL                    AS last_autoanalyze,
    NULL                    AS last_vacuum,
    NULL                    AS last_autovacuum
FROM stl_analyze a
JOIN pg_class c
    ON  c.oid = a.table_id
JOIN pg_namespace n
    ON  n.oid = c.relnamespace
WHERE n.nspname = %s
  AND c.relname = %s
  AND a.status  = 0
"""

VIEWS_IN_SCHEMA = """
SELECT
    v.table_name,
    v.table_schema,
    u.usename                         AS owner,
    v.view_definition                 AS definition
FROM information_schema.views v
JOIN pg_class     c  ON  c.relname  = v.table_name
JOIN pg_namespace n  ON  n.oid      = c.relnamespace
                     AND n.nspname  = v.table_schema
JOIN pg_user      u  ON  u.usesysid = c.relowner
WHERE v.table_schema = %s
ORDER BY v.table_name
"""

# Sample data query is constructed at runtime by RedshiftProfiler:
#   f'SELECT * FROM "{safe_schema}"."{safe_table}" ORDER BY RANDOM() LIMIT %s'
# The schema and table identifiers are double-quoted (" → "") to prevent
# SQL injection from names containing special characters.
