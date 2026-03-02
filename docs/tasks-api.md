# Ignite Data Profiler Service — Tasks API

**Base URL:** `http://<host>:8000/api/v1`
**Rate limit:** POST /profile/tasks — 5 requests/minute per IP

---

## Endpoints Overview

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/profile/tasks` | Create async profiling task → 202 |
| `GET` | `/profile/tasks/{task_id}` | Get task status + result |
| `GET` | `/profile/tasks` | List tasks (paginated) |
| `POST` | `/profile/tasks/{task_id}/cancel` | Cancel a running task |

---

## Task Lifecycle

```
pending → connecting → profiling → augmenting → completed
                                              ↘ failed
                  cancelling → cancelled
```

---

## 1. Create Profiling Task

**`POST /profile/tasks`**

Returns immediately with a task ID. The profiling runs in the background.

**Request body** — discriminated by `datasource_type`:

### PostgreSQL

```json
{
  "datasource_type": "postgres",
  "connection": {
    "host": "localhost",
    "port": 5432,
    "database": "mydb",
    "username": "user",
    "password": "secret",
    "connect_timeout": 10.0,
    "pool_min_size": 1,
    "pool_max_size": 10,
    "ssl": {
      "enabled": false,
      "ca_cert": null,
      "client_cert": null,
      "client_key": null,
      "verify": true
    }
  },
  "config": { }
}
```

### MySQL

```json
{
  "datasource_type": "mysql",
  "connection": {
    "host": "localhost",
    "port": 3306,
    "database": "mydb",
    "username": "user",
    "password": "secret",
    "charset": "utf8mb4",
    "autocommit": true,
    "ssl": { "enabled": false }
  },
  "config": { }
}
```

### Snowflake

```json
{
  "datasource_type": "snowflake",
  "connection": {
    "account": "xy12345.us-east-1",
    "username": "user",
    "password": "secret",
    "database": "MYDB",
    "schema_name": "PUBLIC",
    "warehouse": "COMPUTE_WH",
    "role": null,
    "authenticator": null
  },
  "config": { }
}
```

### S3 / S3-compatible

```json
{
  "datasource_type": "s3_file",
  "connection": {
    "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
    "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "aws_session_token": null,
    "aws_region": "us-east-1",
    "endpoint_url": null,
    "use_ssl": true
  },
  "paths": [
    {
      "bucket": "my-bucket",
      "key": "data/orders.csv",
      "name": "orders",
      "file_format": "auto",
      "delimiter": null,
      "has_header": true
    }
  ],
  "config": { }
}
```

> **Notes for S3:**
> - `connection` is optional — omit it entirely to use IAM role / environment credentials
> - `endpoint_url` supports MinIO/LocalStack: `"http://localhost:9000"`
> - `key` supports glob patterns: `"data/2024/*.parquet"`, `"warehouse/**/*.csv"`
> - `file_format`: `"auto"` | `"csv"` | `"parquet"` | `"json"`
> - Up to 50 paths per request; each path is profiled as a separate logical table

---

### `config` object (all datasource types)

All fields are optional. Defaults shown below.

```json
{
  "include_schemas": null,
  "exclude_schemas": ["pg_catalog", "information_schema", "pg_toast"],
  "include_tables": null,
  "exclude_tables": [],
  "sample_size": 10,
  "include_sample_data": true,
  "include_row_counts": true,
  "include_indexes": true,
  "include_relationships": true,
  "include_data_freshness": true,
  "timeout_seconds": 1000,

  "include_column_stats": false,
  "top_values_limit": 10,
  "top_values_cardinality_threshold": 100,

  "augment_descriptions": false,
  "llm_batch_size": 10,
  "augment_column_descriptions": false,
  "llm_column_batch_size": 20,
  "augment_glossary": false,
  "llm_glossary_batch_size": 10,
  "infer_kpis": false,
  "llm_kpi_max_domains": 10,
  "llm_kpis_per_domain": 5
}
```

> All `augment_*` and `infer_kpis` flags require `LLM_ENABLED=true` server-side (non-fatal if disabled).

---

### Response — 202 Accepted

```json
{
  "task_id": "a1b2c3d4-e5f6-...",
  "status": "pending",
  "created_at": "2026-02-22T10:00:00Z",
  "status_url": "/api/v1/profile/tasks/a1b2c3d4-e5f6-..."
}
```

---

## 2. Get Task Status

**`GET /profile/tasks/{task_id}`**

Poll this endpoint until `status` is `completed`, `failed`, or `cancelled`.

### Response — 200 OK

```json
{
  "task_id": "a1b2c3d4-e5f6-...",
  "status": "profiling",
  "datasource_type": "postgres",
  "progress": {
    "phase": "profiling",
    "percent": 45,
    "detail": { "table": "orders", "schema": "public" }
  },
  "created_at": "2026-02-22T10:00:00Z",
  "started_at": "2026-02-22T10:00:01Z",
  "completed_at": null,
  "duration_seconds": 12.4,
  "error": null,
  "result": null
}
```

When `status = "completed"`, `result` contains the full profiling payload (see [Profiling Result](#profiling-result) below).

When `status = "failed"`:

```json
{
  "error": {
    "error_code": "CONNECTION_ERROR",
    "message": "Could not connect to host: timeout after 10s"
  }
}
```

### Status values

| Value | Meaning |
|-------|---------|
| `pending` | Queued, not started |
| `connecting` | Establishing DB connection |
| `profiling` | Collecting schema/table metadata |
| `augmenting` | Running LLM augmentation |
| `completed` | Done — `result` is populated |
| `failed` | Error — `error` is populated |
| `cancelling` | Cancel requested |
| `cancelled` | Cancelled successfully |

### 404 — Task not found

```json
{
  "error": "NOT_FOUND",
  "message": "Task a1b2c3d4 not found"
}
```

---

## 3. List Tasks

**`GET /profile/tasks?skip=0&limit=20`**

| Query param | Type | Default | Max |
|-------------|------|---------|-----|
| `skip` | int ≥ 0 | `0` | — |
| `limit` | int 1–100 | `20` | `100` |

### Response — 200 OK

```json
{
  "tasks": [
    {
      "task_id": "a1b2c3d4-...",
      "status": "completed",
      "datasource_type": "postgres",
      "created_at": "2026-02-22T10:00:00Z",
      "started_at": "2026-02-22T10:00:01Z",
      "completed_at": "2026-02-22T10:02:30Z",
      "progress": null,
      "error": null
    }
  ],
  "total": 42,
  "skip": 0,
  "limit": 20
}
```

> The list response does **not** include `result` — fetch individual tasks for the full payload.

---

## 4. Cancel Task

**`POST /profile/tasks/{task_id}/cancel`**

### Response — 202 Accepted

```json
{
  "task_id": "a1b2c3d4-...",
  "status": "cancelling"
}
```

### 404 — Task not running

Returned when the task is not in a cancellable state (already completed, failed, or not found).

---

## Profiling Result

The `result` field in `TaskStatusResponse` when `status = "completed"`. Structure depends on datasource type.

### PostgreSQL / MySQL / Snowflake

```json
{
  "profiled_at": "2026-02-22T10:02:30Z",
  "database": {
    "name": "mydb",
    "version": "PostgreSQL 15.3",
    "encoding": "UTF8",
    "size_bytes": 104857600
  },
  "schemas": [
    {
      "name": "public",
      "owner": "postgres",
      "tables": [
        {
          "name": "orders",
          "schema": "public",
          "owner": "postgres",
          "description": null,
          "row_count": 50000,
          "size_bytes": 8192000,
          "total_size_bytes": 9216000,
          "data_freshness": {
            "last_analyze": "2026-02-21T08:00:00Z",
            "last_autoanalyze": null,
            "last_vacuum": null,
            "last_autovacuum": null
          },
          "columns": [
            {
              "name": "order_id",
              "ordinal_position": 1,
              "data_type": "integer",
              "is_nullable": false,
              "column_default": null,
              "character_maximum_length": null,
              "numeric_precision": 32,
              "numeric_scale": 0,
              "is_primary_key": true,
              "description": null,
              "enum_values": null,
              "sample_values": [1, 2, 3],
              "statistics": null
            }
          ],
          "indexes": [
            {
              "name": "orders_pkey",
              "columns": ["order_id"],
              "is_unique": true,
              "is_primary": true,
              "index_type": "btree"
            }
          ],
          "relationships": [
            {
              "constraint_name": "orders_customer_id_fkey",
              "from_column": "customer_id",
              "to_schema": "public",
              "to_table": "customers",
              "to_column": "id",
              "on_update": "NO ACTION",
              "on_delete": "NO ACTION"
            }
          ],
          "glossary": null
        }
      ],
      "views": [
        {
          "name": "orders_summary",
          "schema": "public",
          "owner": "postgres",
          "definition": "SELECT ..."
        }
      ]
    }
  ],
  "kpis": null
}
```

### Column statistics (when `include_column_stats: true`)

```json
"statistics": {
  "total_count": 50000,
  "null_count": 120,
  "null_percentage": 0.24,
  "distinct_count": 49880,
  "distinct_percentage": 99.76,
  "numeric": {
    "min": 1.0,
    "max": 99999.0,
    "mean": 45231.5,
    "median": 44800.0,
    "stddev": 15420.3,
    "variance": 237785652.0,
    "sum": 2261575000.0,
    "p5": 5000.0,
    "p25": 32000.0,
    "p75": 58000.0,
    "p95": 88000.0,
    "zero_count": 0,
    "negative_count": 0,
    "outlier_count": 42
  },
  "string": null,
  "boolean": null,
  "temporal": null,
  "top_values": null
}
```

Type-specific stats — only one of `numeric`, `string`, `boolean`, `temporal` is non-null:

| Column type | Stats field |
|-------------|-------------|
| int, float, decimal | `numeric` |
| varchar, text, char | `string` |
| bool | `boolean` |
| date, timestamp, time | `temporal` |

### LLM augmentation fields

When enabled:
- `table.description` — business description string
- `column.description` — business description string
- `table.glossary` — array of `{ business_term, description, synonyms[] }`
- `result.kpis` — array of `{ name, description, calculation, linked_columns[] }`

---

## Recommended Polling Pattern

```python
import httpx, time

BASE = "http://localhost:8000/api/v1"

# 1. Create task
r = httpx.post(f"{BASE}/profile/tasks", json=request_body)
task_id = r.json()["task_id"]

# 2. Poll until terminal state
terminal = {"completed", "failed", "cancelled"}
while True:
    r = httpx.get(f"{BASE}/profile/tasks/{task_id}")
    data = r.json()
    if data["status"] in terminal:
        break
    time.sleep(2)

# 3. Use result
if data["status"] == "completed":
    result = data["result"]
else:
    error = data["error"]
```
