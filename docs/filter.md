# Filter Column Detection Pipeline — Implementation Plan

## 1. Overview

This document outlines the implementation plan for adding **business filter column detection** to the existing Ignite Lens profiling service. The system identifies columns that data analysts typically use to narrow down or slice data during analysis — such as status fields, date ranges, categories, regions, and boolean flags.

The approach is **data-driven first, LLM-verified second**. All core detection logic relies on deterministic schema and statistical signals. An LLM is used only as a final judge to validate pre-computed results, never to generate them from scratch.

---

## 2. Architecture Summary

The pipeline is structured as four sequential stages:

```
┌──────────────────────────────────────────────────────────────────────┐
│                     FILTER COLUMN DETECTION PIPELINE                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Stage 1: Schema Signal Extraction                                   │
│    └─▶ Foreign keys, indexes, constraints, ENUM types                │
│                                                                      │
│  Stage 2: Statistical Signal Computation                             │
│    └─▶ Cardinality analysis, value patterns, type heuristics         │
│                                                                      │
│  Stage 3: Cross-Column Relationship Analysis                         │
│    └─▶ Functional dependencies, ANOVA scoring, hierarchies           │
│                                                                      │
│  Stage 4: LLM Judge & Reconciliation                                 │
│    └─▶ Evidence-based validation, confidence scoring, final output   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---


## 4. Stage 1 — Schema Signal Extraction

### 4.1 Objective

Extract structural metadata from the database schema that encodes human design intent about how columns are meant to be queried.

### 4.2 Data Sources

Query `information_schema` (or equivalent metadata tables) for the connected datasource. The specific queries will vary by database engine (PostgreSQL, MySQL, SQL Server, etc.), so implement a **datasource adapter interface** that normalizes the output.

### 4.3 Signals to Extract

**Foreign key detection** — Query `information_schema.key_column_usage` joined with `information_schema.referential_constraints` to identify FK columns and their referenced (dimension) tables. For each FK, also fetch the row count of the referenced table — a FK pointing to a table with fewer than 500 rows is a very strong filter signal.

**Index analysis** — Query `information_schema.statistics` (MySQL) or `pg_indexes` (PostgreSQL) to identify indexed columns. Distinguish between unique indexes (likely identifiers) and non-unique indexes (likely filters). Extract composite index compositions — columns that appear together in composite indexes are commonly filtered together and should be stored as `composite_index_partners`.

**Constraint detection** — Query for CHECK constraints and ENUM type definitions. Any column with a CHECK constraint that limits values to a finite set (e.g., `status IN ('ACTIVE', 'INACTIVE', 'PENDING')`) is a filter by definition. Parse the constraint expression to extract the allowed values.

**Primary key exclusion** — Identify PK columns and mark them as non-filters. Note that composite PKs may contain columns that are independently useful as filters (e.g., a composite PK of `(year, region, product_id)` — `year` and `region` are filters even though they're part of the PK).

### 4.4 Scoring Logic

Assign a `schema_score` from 0.0 to 1.0 based on the following rules:

- Is a foreign key referencing a small dimension table (< 500 rows) → 0.9
- Is a foreign key referencing a larger table → 0.6
- Has a non-unique index → 0.7
- Has a CHECK constraint or ENUM type → 0.95
- Is a primary key (single-column) → 0.05 (strong negative signal)
- Is part of a composite index → 0.5 (boosted if partners are also filter candidates)
- No schema signals present → 0.0 (neutral, defer to other stages)

When multiple schema signals are present, take the maximum rather than summing — a column that is both a FK and indexed doesn't become "more" of a filter, but the confidence is higher.

### 4.5 Implementation Notes

- Build a `SchemaIntrospector` class with a `get_schema_signals(table_name) -> dict[str, SchemaSignals]` method.
- Implement concrete subclasses per database engine: `PostgresSchemaIntrospector`, `MySQLSchemaIntrospector`, etc.
- Cache schema introspection results per datasource since schema rarely changes between profiling runs.
- Handle the case where the connected user lacks permission to query metadata tables — degrade gracefully by returning empty schema signals and relying on statistical signals.

### 4.6 Table Role Detection

Before scoring individual columns, classify the table itself as either a **fact table** or **dimension table**. Use these heuristics:

- High row count (>10K) + multiple FK columns + numeric measure columns → fact table
- Low row count (<5K) + no outgoing FKs + mostly string columns → dimension table
- Referenced by many FKs from other tables → dimension table

Store the table classification as context — in a fact table, FK columns are almost always filters. In a dimension table, most descriptive columns are potential filters when the table is joined.

---

## 5. Stage 2 — Statistical Signal Computation

### 5.1 Objective

Compute data-level metrics that indicate filter suitability, leveraging the profiling statistics already collected by the existing service.

### 5.2 Cardinality Analysis

This is the single most important statistical signal. Compute `distinct_count` and `repetition_factor` (`row_count / distinct_count`) for every column, then bucket:

| Bucket | Distinct Count | Repetition Factor | Interpretation |
|---|---|---|---|
| `constant` | 1 | = row_count | Useless — exclude entirely |
| `boolean` | 2 | > 1000 | Boolean filter |
| `low_categorical` | 3–50 | high | Strong categorical filter |
| `high_categorical` | 50–500 | moderate | Categorical filter (search/autocomplete UI) |
| `high_cardinality` | 500–10K | low | Possible filter with search UI, or entity column |
| `near_unique` | > 10K, > 10% of rows | ~1 | Identifier — not a filter |

Assign a `cardinality_score`:

- `constant` → 0.0
- `boolean` → 0.85
- `low_categorical` → 0.95
- `high_categorical` → 0.70
- `high_cardinality` → 0.30
- `near_unique` → 0.05

### 5.3 Data Type Scoring

Combine the SQL data type with cardinality to produce a `dtype_score`:

- `DATE`, `TIMESTAMP`, `DATETIME` → 0.90 (almost always a temporal filter)
- `BOOLEAN` → 0.90
- `VARCHAR`/`TEXT` + low cardinality → 0.80
- `VARCHAR`/`TEXT` + high cardinality → 0.15 (likely free-text, not a filter)
- `INTEGER` + low cardinality + not a PK → 0.75 (coded values like status codes)
- `INTEGER` + high cardinality → 0.10 (likely an ID)
- `FLOAT`/`DECIMAL`/`NUMERIC` → 0.05 (measures, not filters — exception for range filters like price bands, but those are rare)

### 5.4 Null Ratio Impact

Apply a penalty multiplier based on null percentage:

- Null ratio < 5% → multiplier 1.0 (no penalty)
- Null ratio 5–30% → multiplier 0.9
- Null ratio 30–60% → multiplier 0.7
- Null ratio 60–80% → multiplier 0.4
- Null ratio > 80% → multiplier 0.1 (heavily penalize — analysts avoid filtering on mostly-empty columns)

This multiplier is applied to the final composite score, not to individual signal scores.

### 5.5 Column Name Pattern Matching

Implement regex-based pattern detection against the column name. Define pattern groups:

**Strong filter patterns (score 0.8):**
`*_status`, `*_type`, `*_category`, `*_code`, `*_class`, `*_group`, `*_tier`, `*_level`, `*_flag`, `*_mode`, `*_state`, `is_*`, `has_*`, `*_region`, `*_country`, `*_city`, `*_department`, `*_channel`

**Temporal patterns (score 0.85):**
`*_date`, `*_time`, `*_at`, `*_on`, `*_year`, `*_month`, `*_quarter`, `*_week`, `date_*`, `time_*`, `fiscal_*`

**Anti-filter patterns (score 0.05):**
`*_id` (when high cardinality), `*_name` (when high cardinality), `*_description`, `*_notes`, `*_comment`, `*_text`, `*_blob`, `*_json`, `*_xml`, `created_by`, `updated_by`, `created_at`, `updated_at`, `modified_*`, `*_hash`, `*_token`, `*_uuid`

**Audit/metadata patterns (score 0.10):**
`created_at`, `updated_at`, `modified_at`, `inserted_at`, `deleted_at`, `last_login`, `*_timestamp` when column name also contains `created`, `updated`, `modified`, `audit`, `log`

These audit columns often look like temporal filters statistically (they're dates with moderate cardinality) but are rarely used analytically. Flag them distinctly.

### 5.6 Value Pattern Analysis

For string columns, sample 100–500 distinct values and run pattern classification without any LLM:

- All values are UPPERCASE short strings (2–5 chars) → likely codes/abbreviations → filter signal 0.8
- Values match ISO country/currency/language codes → filter signal 0.9
- Values are mixed-case natural language phrases → free text → anti-filter signal 0.1
- Values contain numeric patterns like `ORD-001`, `INV-2024-001` → identifiers → anti-filter signal 0.1
- Values are from a small, repeating set with consistent formatting → categorical → filter signal 0.8

Implement this as a `ValuePatternClassifier` utility class with a `classify(sample_values: list[str]) -> str` method returning one of: `code`, `categorical`, `free_text`, `identifier`, `mixed`.

### 5.7 Implementation Notes

- Most of these signals should already be available from the existing profiler output — wire them into the scorer rather than recomputing.
- For large tables, the cardinality computation may already be approximate (HyperLogLog). That's fine — exact counts aren't needed for bucketing.
- Store the raw statistical signals in the `StatisticalSignals` model for debugging and for the LLM judge to consume later.

---

## 6. Stage 3 — Cross-Column Relationship Analysis

### 6.1 Objective

Detect inter-column relationships that reveal filter hierarchies and analytical usefulness — signals that no single-column analysis can provide.

### 6.2 Functional Dependency Detection

For every pair of categorical columns (both in the `low_categorical` or `high_categorical` cardinality bucket), test whether column A functionally determines column B:

**Algorithm:**

1. Group data by column A.
2. For each group, count the distinct values of column B.
3. If the maximum distinct count of B across all groups of A is 1, then A → B (A determines B).

If A → B and B → A, they are equivalent (1:1 mapping). If A → B but not B → A, then A is finer-grained and B is the parent in a hierarchy.

**Build a dependency graph** and extract hierarchy chains. For example, detecting `city → state → country` means all three are filters, and they form a cascading filter hierarchy.

**Performance consideration:** This is O(n²) in the number of categorical columns per table. For tables with many categorical columns (>20), limit testing to columns in the `low_categorical` bucket (distinct < 50) or columns that share naming pattern prefixes.

### 6.3 ANOVA Scoring (Analytical Usefulness)

For each candidate filter column (composite score > 0.3 from stages 1-2), compute a one-way ANOVA F-ratio against each numeric (measure) column in the same table:

**Algorithm:**

1. Group the numeric column values by the candidate filter column.
2. Compute the between-group variance (variance of group means).
3. Compute the within-group variance (mean of group variances).
4. F-ratio = between-group variance / within-group variance.

A high F-ratio means that filtering on this column reveals meaningful differences in the measure — exactly what an analyst wants.

**Scoring:**

- F-ratio > 10 → `anova_score` = 0.9 (highly analytically useful)
- F-ratio 5–10 → `anova_score` = 0.7
- F-ratio 2–5 → `anova_score` = 0.5
- F-ratio 1–2 → `anova_score` = 0.3
- F-ratio < 1 → `anova_score` = 0.1 (filtering on this column reveals nothing interesting)

Take the maximum F-ratio across all measure columns as the column's `anova_score` — a filter only needs to be useful against one measure to be valuable.

**Performance consideration:** For large tables, sample 50K–100K rows before computing ANOVA. The statistical validity holds with samples of this size. Skip this computation entirely for tables with no numeric columns.

### 6.4 Composite Index Co-occurrence

If Stage 1 identified composite indexes, record which columns appear together. These co-occurrence pairs become "recommended filter combinations" in the output — they don't change individual column scores but add metadata about how filters should be presented together in the UI.

### 6.5 Implementation Notes

- Functional dependency detection is the most expensive computation. Implement it with early termination — as soon as a single group of A has >1 distinct value of B, the dependency fails and you can skip the rest.
- Cache the dependency graph per table since it only changes when data changes significantly.
- The ANOVA computation should use the sampled data the profiler already works with — don't trigger a separate full table scan.

---

## 7. Stage 4 — Composite Scoring & LLM Judge

### 7.1 Composite Score Computation

Combine all signal scores into a weighted composite for each column:

**Weights:**

| Signal | Weight | Rationale |
|---|---|---|
| `schema_score` | 0.30 | Highest — reflects explicit human design intent |
| `cardinality_score` | 0.25 | Core statistical signal |
| `dtype_score` | 0.20 | Strong type-based heuristic |
| `anova_score` | 0.15 | Measures actual analytical usefulness |
| `naming_score` | 0.10 | Supplementary pattern signal |

**Formula:**

```
composite = (0.30 × schema_score + 0.25 × cardinality_score + 0.20 × dtype_score +
             0.15 × anova_score + 0.10 × naming_score) × null_penalty_multiplier
```

After computing the composite, assign a preliminary `filter_type` based on the dominant signal:
- Temporal data type → `temporal`
- Boolean type or distinct count = 2 → `boolean`
- Part of a detected hierarchy → `hierarchical`
- Numeric type with moderate cardinality → `range`
- Everything else above threshold → `categorical`

### 7.2 Auto-Accept / Auto-Reject Thresholds

Before invoking the LLM, triage columns into three buckets:

- **Auto-accept** (composite > 0.85 AND has at least one schema signal) — these are definitively filters. No LLM needed. Set `confidence_source = "heuristic_only"`.
- **Auto-reject** (composite < 0.15) — these are definitively not filters. No LLM needed. Exclude from output.
- **LLM review band** (0.15 ≤ composite ≤ 0.85, or composite > 0.85 without schema signals) — send to the LLM judge.

This triage typically eliminates 50–70% of columns from LLM evaluation, significantly reducing cost and latency.

### 7.3 LLM Judge Invocation

**Prompt Design Principles:**

- Frame the LLM as a **judge evaluating evidence**, not a generator.
- Send all columns for a single table in one request for cross-column context.
- Include the full pre-computed evidence (all signal scores, sample values, schema signals) so the LLM doesn't need to guess.
- Explicitly instruct the LLM to be skeptical of audit/metadata columns and low-cardinality ID columns.

**Request Payload Structure:**

Send to the LLM:

- Table name, row count, detected table role (fact/dimension)
- For each column in the review band: column name, data type, all signal scores, sample values (top 10 by frequency), null ratio, detected hierarchy membership
- The composite score and preliminary filter_type classification

**Expected Response Structure:**

Request JSON output with:

- Per-column: `llm_confidence` (0.0–1.0), `llm_filter_type`, `agrees_with_heuristic` (boolean), `reasoning` (1–2 sentences)

**LLM Selection:** Use a cost-efficient model for this task — the judgment is structured and evidence-based, so a smaller model (e.g., Claude Haiku or Sonnet) is sufficient. Reserve larger models for more generative tasks.

### 7.4 Reconciliation Logic

After receiving the LLM verdict, merge it with the heuristic score:

**Agreement (difference < 0.2):**
- Final score = average of heuristic and LLM scores
- Set `confidence_source = "llm_agreed"`

**Mild disagreement (difference 0.2–0.4):**
- If column has strong schema signals → bias 70% toward heuristic, 30% toward LLM
- If column lacks schema signals → bias 40% toward heuristic, 60% toward LLM
- Set `confidence_source = "llm_adjusted"`

**Strong disagreement (difference > 0.4):**
- Final score = average of both (no bias)
- Set `confidence_source = "flagged_for_review"`
- Log the disagreement with full context for pipeline tuning

### 7.5 Final Output Assembly

After reconciliation, apply a final threshold to produce the output list:

- `final_score ≥ 0.5` → include in the filter column list
- `final_score < 0.5` → exclude (but store internally for debugging)

For included columns, also compute the `recommended_ui_control`:

- `boolean` → `toggle`
- `categorical` with distinct ≤ 20 → `dropdown`
- `categorical` with distinct 20–200 → `search_box`
- `temporal` → `date_picker`
- `range` → `slider`
- `hierarchical` → `cascading_dropdown`

---

## Integration with Existing Profiler

The filter detection pipeline should be invoked **after** the statistical profiling stage completes, since it consumes profiling outputs (distinct counts, null ratios, data types, row counts). Add it as an optional post-processing step in the profiling workflow.
