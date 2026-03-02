"""PostgreSQL introspection SQL queries for the data profiler.

All queries use asyncpg-style ``$1``/``$2``/... positional placeholders.
Table and schema identifiers in sample-data queries are escaped with
double-quotes by the service layer — never interpolated directly from
user input.
"""

DATABASE_METADATA = """
SELECT
    current_database()                     AS db_name,
    version()                              AS version,
    pg_encoding_to_char(encoding)          AS encoding,
    pg_database_size(current_database())   AS size_bytes
FROM pg_database
WHERE datname = current_database()
"""

# NOTE: The service builds the ``$1, $2, ...`` IN-list placeholder string
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
    pg_relation_size(c.oid)               AS size_bytes,
    pg_total_relation_size(c.oid)         AS total_size_bytes,
    obj_description(c.oid, 'pg_class')    AS description
FROM information_schema.tables t
JOIN pg_class c
    ON  c.relname   = t.table_name
JOIN pg_namespace n
    ON  n.oid       = c.relnamespace
    AND n.nspname   = t.table_schema
JOIN pg_user u
    ON  u.usesysid  = c.relowner
WHERE t.table_schema = $1
  AND t.table_type   = 'BASE TABLE'
ORDER BY t.table_name
"""

# Used as a fallback when reltuples = -1 (table never analyzed).
# Identifiers are double-quoted by the service layer — never interpolated directly.
ROW_COUNT_EXACT = 'SELECT COUNT(1) AS row_count FROM "{schema}"."{table}"'

COLUMNS_FOR_TABLE = """
SELECT
    column_name,
    ordinal_position,
    data_type,
    (is_nullable = 'YES')          AS is_nullable,
    column_default,
    character_maximum_length,
    numeric_precision,
    numeric_scale,
    col_description(
        (SELECT c.oid
         FROM pg_class c
         JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE c.relname = $2 AND n.nspname = $1),
        ordinal_position
    )                              AS description
FROM information_schema.columns
WHERE table_schema  = $1
  AND table_name    = $2
ORDER BY ordinal_position
"""

PRIMARY_KEY_COLUMNS = """
SELECT kcu.column_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
    ON  kcu.constraint_name = tc.constraint_name
    AND kcu.table_schema    = tc.table_schema
    AND kcu.table_name      = tc.table_name
WHERE tc.constraint_type = 'PRIMARY KEY'
  AND tc.table_schema    = $1
  AND tc.table_name      = $2
"""

INDEXES_FOR_TABLE = """
SELECT
    i.relname                                      AS index_name,
    ix.indisunique                                 AS is_unique,
    ix.indisprimary                                AS is_primary,
    am.amname                                      AS index_type,
    array_agg(a.attname ORDER BY k.ord)            AS columns
FROM pg_index ix
JOIN pg_class     i   ON  i.oid    = ix.indexrelid
JOIN pg_class     t   ON  t.oid    = ix.indrelid
JOIN pg_am        am  ON  am.oid   = i.relam
JOIN pg_namespace n   ON  n.oid    = t.relnamespace
JOIN pg_attribute a   ON  a.attrelid = t.oid
JOIN LATERAL unnest(ix.indkey) WITH ORDINALITY AS k(num, ord)
                       ON  a.attnum  = k.num
WHERE t.relname  = $1
  AND n.nspname  = $2
  AND a.attnum   > 0
GROUP BY i.relname, ix.indisunique, ix.indisprimary, am.amname
ORDER BY i.relname
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
WHERE tc.table_schema = $1
  AND tc.table_name   = $2
ORDER BY rc.constraint_name, kcu.ordinal_position
"""

DATA_FRESHNESS_FOR_TABLE = """
SELECT
    last_analyze,
    last_autoanalyze,
    last_vacuum,
    last_autovacuum
FROM pg_stat_user_tables
WHERE schemaname = $1
  AND relname    = $2
"""

VIEWS_IN_SCHEMA = """
SELECT
    v.table_name,
    v.table_schema,
    u.usename                         AS owner,
    pg_get_viewdef(c.oid, true)       AS definition
FROM information_schema.views v
JOIN pg_class     c  ON  c.relname  = v.table_name
JOIN pg_namespace n  ON  n.oid      = c.relnamespace
                     AND n.nspname  = v.table_schema
JOIN pg_user      u  ON  u.usesysid = c.relowner
WHERE v.table_schema = $1
  AND NOT EXISTS (
        SELECT 1
        FROM pg_depend d
        WHERE d.classid = 'pg_class'::regclass
          AND d.objid   = c.oid
          AND d.deptype = 'e'
      )
ORDER BY v.table_name
"""

# Sample data query is constructed at runtime by PostgresProfilerService:
#   f'SELECT * FROM "{safe_schema}"."{safe_table}" ORDER BY RANDOM() LIMIT $1'
# The schema and table identifiers are double-quoted (`` " `` → `` "" ``)
# to prevent SQL injection from names containing special characters.

ALLOWED_ENUM_VALUES = """SELECT enumlabel
FROM pg_enum
WHERE enumtypid = (
    SELECT atttypid
    FROM pg_attribute
    WHERE attrelid = $1::regclass
    AND attname = $2
)
ORDER BY enumsortorder;
"""
