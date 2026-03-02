"""OpenAI LLM client for table and column description augmentation."""

from __future__ import annotations

import json

from openai import AsyncOpenAI
from pydantic import TypeAdapter

from core.api.v1.schemas.profiler import ColumnMetadata, GlossaryTerm, KPITerm, TableMetadata
from core.config import settings
from core.llm.base import BaseLLMClient
from core.logging import get_logger

logger = get_logger(__name__)

# Maximum sample values per column included in the prompt.
_MAX_SAMPLE_VALUES = 3
# Maximum columns described in the table prompt to keep it concise.
_MAX_COLUMNS_IN_PROMPT = 20
# Maximum sibling columns shown in the column prompt for context.
_MAX_SIBLING_COLUMNS = 10
# Max tokens reserved for glossary inference responses (5 terms with descriptions).
_GLOSSARY_MAX_TOKENS = 1000
# Max tokens for the three KPI inference phases.
_KPI_CLUSTER_MAX_TOKENS = 800  # Phase A: domain names + table name lists only
_KPI_GENERATE_MAX_TOKENS = 1500  # Phase B: up to ~15 KPIs with descriptions
_KPI_SYNTHESIZE_MAX_TOKENS = 2000  # Phase C: full deduplication output
# Max tables per domain to include in the Phase B prompt.
_KPI_MAX_TABLES_PER_DOMAIN = 15
# Max columns per table included in the Phase B domain prompt.
_KPI_MAX_COLUMNS_PER_TABLE = 10


def _build_column_summary(col: ColumnMetadata) -> str:
    """Render a single column as a compact prompt line."""
    parts = [f"  - {col.name} ({col.data_type})"]
    if col.is_primary_key:
        parts.append("[PK]")
    if not col.is_nullable:
        parts.append("[NOT NULL]")
    if col.enum_values:
        enum_preview = ", ".join(col.enum_values[:6])
        parts.append(f"enum: [{enum_preview}]")
    if col.sample_values:
        safe_samples = [str(v) for v in col.sample_values[:_MAX_SAMPLE_VALUES] if v is not None]
        if safe_samples:
            parts.append(f"samples: [{', '.join(safe_samples)}]")
    if col.description:
        parts.append(f"pg_desc: {col.description}")
    return " ".join(parts)


def _build_valid_columns_reference(
    tables: list[TableMetadata],
    max_cols_per_table: int,
) -> tuple[str, set[str]]:
    """Build a valid-columns reference block and a lookup set.

    Returns:
        A tuple of ``(prompt_text, valid_set)`` where *prompt_text* is the
        formatted block to embed in the LLM prompt and *valid_set* contains
        all ``table.column`` strings for post-parse validation.

    """
    lines: list[str] = []
    valid: set[str] = set()
    for table in tables:
        for col in table.columns[:max_cols_per_table]:
            ref = f"{table.name}.{col.name}"
            valid.add(ref)
            lines.append(f"  - {ref}")
    return "\n".join(lines), valid


def _strip_invalid_linked_columns(
    kpis: list[KPITerm],
    valid_columns: set[str],
    provider: str,
    context: str,
) -> None:
    """Remove ``linked_columns`` entries not in *valid_columns*.  Mutates in-place."""
    for kpi in kpis:
        original = kpi.linked_columns
        kpi.linked_columns = [col for col in original if col in valid_columns]
        stripped = set(original) - set(kpi.linked_columns)
        if stripped:
            logger.warning(
                "Stripped invalid linked_columns from KPI",
                provider=provider,
                context=context,
                kpi_name=kpi.name,
                invalid_columns=sorted(stripped),
            )


def _build_prompt(table: TableMetadata) -> str:
    """Construct the user message sent to the LLM."""
    columns = table.columns[:_MAX_COLUMNS_IN_PROMPT]
    col_lines = "\n".join(_build_column_summary(c) for c in columns)
    truncation_note = ""
    if len(table.columns) > _MAX_COLUMNS_IN_PROMPT:
        truncation_note = f"\n  ... and {len(table.columns) - _MAX_COLUMNS_IN_PROMPT} more columns (omitted for brevity)"

    existing = table.description or "(none)"
    row_count_str = str(table.row_count) if table.row_count is not None else "unknown"

    return f"""Given the database table metadata below, write a concise 1-3 sentence business description.
Explaining what this table stores, its primary purpose, and any key relationships implied by its structure.

Focus on business meaning, not technical implementation.
Return ONLY the description text — no preamble, no markdown.

Table: {table.schema_name}.{table.name}
Existing pg_description: {existing}
Approximate row count: {row_count_str}

Columns:
{col_lines}{truncation_note}
"""


def _build_column_prompt(column: ColumnMetadata, table: TableMetadata) -> str:
    """Construct the user message for a single column description."""
    parts = [f"  - {column.name} ({column.data_type})"]
    if column.is_primary_key:
        parts.append("[PK]")
    if not column.is_nullable:
        parts.append("[NOT NULL]")
    if column.enum_values:
        enum_preview = ", ".join(column.enum_values[:6])
        parts.append(f"enum: [{enum_preview}]")
    if column.sample_values:
        safe_samples = [str(v) for v in column.sample_values[:_MAX_SAMPLE_VALUES] if v is not None]
        if safe_samples:
            parts.append(f"samples: [{', '.join(safe_samples)}]")
    if column.description:
        parts.append(f"pg_desc: {column.description}")
    column_detail = " ".join(parts)

    table_desc = table.description or "(none)"
    siblings = [c for c in table.columns if c.name != column.name]
    sibling_lines = "\n".join(f"  - {c.name} ({c.data_type})" for c in siblings[:_MAX_SIBLING_COLUMNS])
    truncation_note = ""
    if len(siblings) > _MAX_SIBLING_COLUMNS:
        truncation_note = f"\n  ... and {len(siblings) - _MAX_SIBLING_COLUMNS} more columns (omitted)"

    return f"""Given the database column metadata below, write a concise 1-2 sentence business description.
Explaining what this column stores and its role within the table.

Focus on business meaning, not technical implementation.
Return ONLY the description text — no preamble, no markdown.

Table: {table.schema_name}.{table.name}
Table description: {table_desc}

Column to describe:
{column_detail}

Other columns in this table (for context):
{sibling_lines}{truncation_note}
"""


def _build_glossary_prompt(table: TableMetadata) -> str:
    """Construct the user message for business glossary inference."""
    columns = table.columns[:_MAX_COLUMNS_IN_PROMPT]
    col_lines = "\n".join(_build_column_summary(c) for c in columns)
    truncation_note = ""
    if len(table.columns) > _MAX_COLUMNS_IN_PROMPT:
        truncation_note = f"\n  ... and {len(table.columns) - _MAX_COLUMNS_IN_PROMPT} more columns (omitted for brevity)"

    existing_desc = table.description or "(none)"
    row_count_str = str(table.row_count) if table.row_count is not None else "unknown"

    return f"""Given the database table metadata below, infer up to 5 business glossary terms that domain experts would use
when discussing or working with this data. Make sure the glossary terms are grounded in the actual schema details.
Glossaries term should be linked with the columns.

Table: {table.schema_name}.{table.name}
Table description: {existing_desc}
Approximate row count: {row_count_str}

Columns:
{col_lines}{truncation_note}

For each term provide:
- business_term: the domain-specific term (e.g. "Customer", "Invoice")
- description: 1-2 sentences explaining the term in this business context
- synonyms: common alternative names or abbreviations (may be an empty list)

Respond ONLY with valid JSON in exactly this format (no markdown, no preamble):
{{"terms": [{{"business_term": "...", "description": "...", "synonyms": ["...", "..."]}}]}}
"""


def _build_kpi_cluster_prompt(tables: list[TableMetadata], max_domains: int) -> str:
    """Construct the Phase A clustering prompt.

    Sends only qualified table names — no column metadata — to keep the token
    count minimal and the response fast.
    """
    table_names = "\n".join(f"  - {t.schema_name}.{t.name}" for t in tables)
    return f"""You are a data architect. Analyse the following database table names and group them into business domains.

Rules:
- Create between 1 and {max_domains} domains.
- Every table must appear in exactly one domain.
- Domain names should be short, business-oriented labels (e.g. "Finance", "Customer", "Inventory").
- Use only the information available from table names — do not invent context.

Tables:
{table_names}

Respond ONLY with valid JSON in exactly this format (no markdown, no preamble):
{{"domains": [{{"domain": "Finance", "tables": ["public.invoices", "public.payments"]}}]}}
"""


def _build_kpi_generate_prompt(
    domain_name: str,
    domain_tables: list[TableMetadata],
    max_kpis: int,
) -> str:
    """Construct the Phase B KPI generation prompt for one business domain."""
    tables_section_parts: list[str] = []
    for table in domain_tables[:_KPI_MAX_TABLES_PER_DOMAIN]:
        cols = table.columns[:_KPI_MAX_COLUMNS_PER_TABLE]
        col_lines = "\n".join(_build_column_summary(c) for c in cols)
        col_truncation = ""
        if len(table.columns) > _KPI_MAX_COLUMNS_PER_TABLE:
            col_truncation = f"\n    ... and {len(table.columns) - _KPI_MAX_COLUMNS_PER_TABLE} more columns"
        row_str = str(table.row_count) if table.row_count is not None else "unknown"
        desc_str = table.description or "(none)"
        tables_section_parts.append(
            f"  Table: {table.schema_name}.{table.name} (rows: {row_str}, desc: {desc_str})\n"
            + f"  Columns:\n{col_lines}{col_truncation}"
        )
    tables_section = "\n\n".join(tables_section_parts)

    table_truncation = ""
    if len(domain_tables) > _KPI_MAX_TABLES_PER_DOMAIN:
        table_truncation = (
            f"\n(Note: {len(domain_tables) - _KPI_MAX_TABLES_PER_DOMAIN} additional tables "
            "in this domain were omitted for brevity.)"
        )

    # Build explicit valid-columns reference for prompt and post-parse validation.
    valid_cols_text, _ = _build_valid_columns_reference(domain_tables[:_KPI_MAX_TABLES_PER_DOMAIN], _KPI_MAX_COLUMNS_PER_TABLE)

    return f"""You are a senior business intelligence analyst. \
Given the database tables below belonging to the "{domain_name}" domain, \
infer up to {max_kpis} high-quality, dashboard-ready KPIs that a business user would track. \
Return fewer if high-quality KPIs cannot be reliably inferred from the schema.

Domain: {domain_name}

Tables:
{tables_section}{table_truncation}

## Valid Columns Reference (use ONLY these in linked_columns):
{valid_cols_text}

## KPI Quality Guidelines:
Prioritize QUALITY over QUANTITY. A good KPI should meet at least one of these criteria:
- Ratio or rate (e.g., conversion rate, churn rate, fill rate, utilization rate)
- Period-over-period comparison (e.g., MoM revenue growth, YoY user growth)
- Weighted or normalized metric (e.g., revenue per user, cost per acquisition, average order value)
- Trend-capable metric meaningful when tracked over time on a dashboard
- Multi-table insight that joins two or more tables for a richer business metric

AVOID these trivial patterns:
- Raw COUNT(*) without meaningful filtering or segmentation
- Simple SUM of a single column without normalization or comparison
- Metrics that merely echo a column value (e.g., "Total Amount" = SUM(amount))
- Metrics that do not provide actionable business insight

Each KPI should be something a business analyst would pin to an executive dashboard.
Return 0 KPIs if no meaningful business KPIs can be inferred.

## CRITICAL CONSTRAINT on linked_columns:
- linked_columns MUST contain ONLY identifiers from the "Valid Columns Reference" list above.
- DO NOT invent, infer, or hallucinate column names not in the reference list.
- Use table.column format (e.g., "orders.amount"), NOT schema.table.column.

For each KPI provide:
- name: short, business-oriented KPI name (e.g., "Order Conversion Rate")
- description: 1-2 sentences explaining what this KPI measures and its business value
- calculation: a valid SQL expression using the table and column names above
- linked_columns: list of column names from the Valid Columns Reference directly used in the calculation

Respond ONLY with valid JSON in exactly this format (no markdown, no preamble):
{{"kpis": [{{"name": "...", "description": "...", "calculation": "...",
           "linked_columns": ["orders.amount", "orders.status"]}}]}}
"""


def _build_kpi_synthesize_prompt(all_domain_kpis: list[KPITerm]) -> str:
    """Construct the Phase C synthesis prompt.

    Sends KPI names and descriptions only — linked_columns are omitted to keep
    tokens bounded and are re-attached post-parse via lookup.
    """
    kpi_lines = "\n".join(f"  - {kpi.name}: {kpi.description}" for kpi in all_domain_kpis)
    return f"""You are a business intelligence architect reviewing a company's KPI catalog.

Below are KPIs inferred from a database schema:
{kpi_lines}

Tasks:
Infer up to 5 NEW cross-domain KPIs that combine insights from multiple domains above.
Each cross-domain KPI must:
- Provide a concrete, executable SQL calculation (omit the KPI entirely if none exists)
- Represent a meaningful business ratio, rate, or normalized metric (not a trivial aggregation)
- Be dashboard-ready: trackable over time, visualizable, and actionable
Do NOT include abstract or hollow KPIs that have no actionable SQL expression.
Do NOT generate trivial COUNT(*) or simple SUM metrics.

For each KPI in your output provide:
- name: KPI name
- description: 1-2 sentences
- calculation: a valid SQL expression, or omit the KPI entirely if unreliable
- linked_columns: columns directly referenced in the calculation

## CRITICAL CONSTRAINT on linked_columns:
- linked_columns MUST reference only columns that exist in the KPIs listed above.
- Use table.column format only. DO NOT include schema names.
- If unsure about a column name, omit it from linked_columns rather than guessing.

Respond ONLY with valid JSON in exactly this format (no markdown, no preamble):
{{"kpis": [{{"name": "...", "description": "...", "calculation": "...", "linked_columns": []}}]}}
"""


class OpenAILLMClient(BaseLLMClient):
    """LLM client backed by OpenAI's Chat Completions API (async).

    Args:
        api_key: OpenAI API key.  Defaults to ``settings.LLM_OPENAI_API_KEY``.
        model: Model ID.  Defaults to ``settings.LLM_MODEL``.
        temperature: Sampling temperature.  Defaults to ``settings.LLM_TEMPERATURE``.
        max_tokens: Response length cap.  Defaults to ``settings.LLM_MAX_TOKENS``.
        timeout: Per-call timeout in seconds.  Defaults to ``settings.LLM_REQUEST_TIMEOUT_SECONDS``.

    """

    provider_name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key or settings.LLM_OPENAI_API_KEY,
            timeout=timeout or settings.LLM_REQUEST_TIMEOUT_SECONDS,
        )
        self._model = model or settings.LLM_MODEL
        self._temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self._max_tokens = max_tokens or settings.LLM_MAX_TOKENS

    async def _describe_table(self, table: TableMetadata) -> str | None:
        """Call the OpenAI Chat Completions endpoint for one table."""
        prompt = _build_prompt(table)
        logger.debug(
            "Requesting LLM description",
            provider=self.provider_name,
            model=self._model,
            table=f"{table.schema_name}.{table.name}",
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise technical writer specialising in data "
                        "catalog documentation. Always respond with plain text only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        raw = response.choices[0].message.content
        if raw:
            description = raw.strip()
            logger.debug(
                "LLM description received",
                provider=self.provider_name,
                table=f"{table.schema_name}.{table.name}",
                chars=len(description),
            )
            return description or None
        return None

    async def _describe_column(self, column: ColumnMetadata, table: TableMetadata) -> str | None:
        """Call the OpenAI Chat Completions endpoint for one column."""
        prompt = _build_column_prompt(column, table)
        logger.debug(
            "Requesting LLM column description",
            provider=self.provider_name,
            model=self._model,
            table=f"{table.schema_name}.{table.name}",
            column=column.name,
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise technical writer specialising in data "
                        "catalog documentation. Always respond with plain text only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        raw = response.choices[0].message.content
        if raw:
            description = raw.strip()
            logger.debug(
                "LLM column description received",
                provider=self.provider_name,
                table=f"{table.schema_name}.{table.name}",
                column=column.name,
                chars=len(description),
            )
            return description or None
        return None

    async def _infer_glossary(self, table: TableMetadata) -> list[GlossaryTerm]:
        """Call the OpenAI Chat Completions endpoint to infer glossary terms for one table."""
        prompt = _build_glossary_prompt(table)
        logger.debug(
            "Requesting LLM glossary inference",
            provider=self.provider_name,
            model=self._model,
            table=f"{table.schema_name}.{table.name}",
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a business analyst specialising in data governance and "
                        "enterprise data catalogs. Always respond with valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=_GLOSSARY_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        if not raw:
            return []

        try:
            data = json.loads(raw)
            terms_raw = data.get("terms", [])[:5]
            ta: TypeAdapter[list[GlossaryTerm]] = TypeAdapter(list[GlossaryTerm])
            terms = ta.validate_python(terms_raw)
            logger.debug(
                "LLM glossary terms received",
                provider=self.provider_name,
                table=f"{table.schema_name}.{table.name}",
                term_count=len(terms),
            )
            return terms
        except Exception as exc:
            logger.warning(
                "Failed to parse glossary terms from LLM response",
                provider=self.provider_name,
                table=f"{table.schema_name}.{table.name}",
                error=str(exc),
            )
            return []

    async def _cluster_tables_into_domains(
        self,
        tables: list[TableMetadata],
        max_domains: int,
    ) -> dict[str, list[str]]:
        """Phase A: call OpenAI to cluster table names into business domains."""
        prompt = _build_kpi_cluster_prompt(tables, max_domains)
        logger.debug(
            "Requesting KPI domain clustering",
            provider=self.provider_name,
            model=self._model,
            table_count=len(tables),
            max_domains=max_domains,
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data architect specialising in business domain modelling. Always respond with valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=_KPI_CLUSTER_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        if not raw:
            return {}

        try:
            data = json.loads(raw)
            domains_raw = data.get("domains", [])
            result: dict[str, list[str]] = {}
            for item in domains_raw:
                domain = str(item.get("domain", "")).strip()
                table_list = [str(t) for t in item.get("tables", [])]
                if domain and table_list:
                    result[domain] = table_list
            logger.debug(
                "KPI domain clustering received",
                provider=self.provider_name,
                domain_count=len(result),
            )

            return result
        except Exception as exc:
            logger.warning(
                "Failed to parse domain clusters from LLM response",
                provider=self.provider_name,
                error=str(exc),
            )
            return {}

    async def _generate_domain_kpis(
        self,
        domain_name: str,
        domain_tables: list[TableMetadata],
        max_kpis: int,
    ) -> list[KPITerm]:
        """Phase B: call OpenAI to generate KPIs for one business domain."""
        prompt = _build_kpi_generate_prompt(domain_name, domain_tables, max_kpis)
        logger.debug(
            "Requesting domain KPI generation",
            provider=self.provider_name,
            model=self._model,
            domain=domain_name,
            table_count=len(domain_tables),
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a business intelligence analyst specialising in KPI design. Always respond with valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=_KPI_GENERATE_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        if not raw:
            return []

        try:
            data = json.loads(raw)
            kpis_raw = data.get("kpis", [])[:max_kpis]
            ta: TypeAdapter[list[KPITerm]] = TypeAdapter(list[KPITerm])
            kpis = ta.validate_python(kpis_raw)

            # Strip hallucinated linked_columns that reference non-existent columns.
            _, valid_columns = _build_valid_columns_reference(
                domain_tables[:_KPI_MAX_TABLES_PER_DOMAIN],
                _KPI_MAX_COLUMNS_PER_TABLE,
            )
            _strip_invalid_linked_columns(
                kpis,
                valid_columns,
                self.provider_name,
                f"domain={domain_name}",
            )

            logger.debug(
                "Domain KPIs received",
                provider=self.provider_name,
                domain=domain_name,
                kpi_count=len(kpis),
            )
            return kpis
        except Exception as exc:
            logger.warning(
                "Failed to parse domain KPIs from LLM response",
                provider=self.provider_name,
                domain=domain_name,
                error=str(exc),
            )
            return []

    async def _synthesize_kpis(
        self,
        all_domain_kpis: list[KPITerm],
    ) -> list[KPITerm]:
        """Phase C: call OpenAI to deduplicate KPIs and add cross-domain ones."""
        if not all_domain_kpis:
            return []

        prompt = _build_kpi_synthesize_prompt(all_domain_kpis)
        logger.debug(
            "Requesting KPI synthesis",
            provider=self.provider_name,
            model=self._model,
            input_kpi_count=len(all_domain_kpis),
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a business intelligence architect specialising in enterprise KPI frameworks. "
                        "Always respond with valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=_KPI_SYNTHESIZE_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        if not raw:
            return all_domain_kpis

        try:
            data = json.loads(raw)
            kpis_raw = data.get("kpis", [])
            ta: TypeAdapter[list[KPITerm]] = TypeAdapter(list[KPITerm])
            synthesized = ta.validate_python(kpis_raw)

            # Strip hallucinated linked_columns.  Valid set is the union of
            # all linked_columns that survived Phase B validation.
            valid_columns: set[str] = set()
            for kpi in all_domain_kpis:
                valid_columns.update(kpi.linked_columns)
            _strip_invalid_linked_columns(
                synthesized,
                valid_columns,
                self.provider_name,
                "synthesis",
            )

            logger.debug(
                "KPI synthesis complete",
                provider=self.provider_name,
                output_kpi_count=len(synthesized),
            )
            return all_domain_kpis + synthesized
        except Exception as exc:
            logger.warning(
                "Failed to parse synthesized KPIs from LLM response",
                provider=self.provider_name,
                error=str(exc),
            )
            return all_domain_kpis
