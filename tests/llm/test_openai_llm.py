"""Tests for OpenAILLMClient prompt building and API integration."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.api.v1.schemas.profiler import ColumnMetadata, TableMetadata
from core.llm.openai import (
    _GLOSSARY_MAX_TOKENS,
    OpenAILLMClient,
    _build_column_prompt,
    _build_glossary_prompt,
    _build_prompt,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_column(
    name: str = "id",
    data_type: str = "integer",
    is_primary_key: bool = False,
    is_nullable: bool = True,
    enum_values: list[str] | None = None,
    sample_values: list | None = None,
    description: str | None = None,
) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        ordinal_position=1,
        data_type=data_type,
        is_nullable=is_nullable,
        column_default=None,
        character_maximum_length=None,
        numeric_precision=None,
        numeric_scale=None,
        is_primary_key=is_primary_key,
        enum_values=enum_values,
        sample_values=sample_values,
        description=description,
    )


def _make_table() -> TableMetadata:
    return TableMetadata(
        name="users",
        schema="public",
        owner="postgres",
        description="Existing pg desc",
        row_count=5000,
        size_bytes=None,
        total_size_bytes=None,
        data_freshness=None,
        columns=[
            _make_column("id", "integer", is_primary_key=True, is_nullable=False, sample_values=[1, 2, 3]),
            _make_column("email", "character varying", is_nullable=False, sample_values=["a@b.com", "c@d.com"]),
            _make_column("status", "USER-DEFINED", enum_values=["active", "inactive"]),
        ],
        indexes=None,
        relationships=None,
    )


def _make_client() -> OpenAILLMClient:
    with patch("core.llm.openai.AsyncOpenAI"):
        return OpenAILLMClient(api_key="sk-test", model="gpt-4o-mini")


# ── Prompt building ────────────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_prompt_contains_qualified_table_name(self):
        prompt = _build_prompt(_make_table())
        assert "public.users" in prompt

    def test_prompt_includes_existing_description(self):
        prompt = _build_prompt(_make_table())
        assert "Existing pg desc" in prompt

    def test_prompt_shows_none_when_no_existing_description(self):
        table = _make_table()
        table.description = None
        prompt = _build_prompt(table)
        assert "(none)" in prompt

    def test_prompt_includes_row_count(self):
        prompt = _build_prompt(_make_table())
        assert "5000" in prompt

    def test_prompt_shows_unknown_when_no_row_count(self):
        table = _make_table()
        table.row_count = None
        prompt = _build_prompt(table)
        assert "unknown" in prompt

    def test_prompt_marks_primary_key(self):
        prompt = _build_prompt(_make_table())
        assert "[PK]" in prompt

    def test_prompt_marks_not_null(self):
        prompt = _build_prompt(_make_table())
        assert "[NOT NULL]" in prompt

    def test_prompt_includes_enum_values(self):
        prompt = _build_prompt(_make_table())
        assert "active" in prompt
        assert "inactive" in prompt

    def test_prompt_includes_sample_values(self):
        prompt = _build_prompt(_make_table())
        assert "a@b.com" in prompt

    def test_prompt_truncation_note_for_wide_tables(self):
        table = _make_table()
        extra_cols = [_make_column(f"col_{i}") for i in range(25)]
        table.columns = extra_cols
        prompt = _build_prompt(table)
        assert "more columns" in prompt

    def test_prompt_no_truncation_note_for_narrow_tables(self):
        prompt = _build_prompt(_make_table())
        assert "more columns" not in prompt

    def test_prompt_includes_column_pg_description(self):
        table = _make_table()
        table.columns[0] = _make_column("id", description="Primary identifier")
        prompt = _build_prompt(table)
        assert "Primary identifier" in prompt


# ── OpenAILLMClient._describe_table ───────────────────────────────────────────


class TestOpenAILLMClientDescribeTable:
    async def test_returns_stripped_content(self):
        client = _make_client()
        mock_choice = MagicMock()
        mock_choice.message.content = "  Stores user accounts.  "
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client._describe_table(_make_table())
        assert result == "Stores user accounts."

    async def test_returns_none_on_empty_content(self):
        client = _make_client()
        mock_choice = MagicMock()
        mock_choice.message.content = "   "
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client._describe_table(_make_table())
        assert result is None

    async def test_returns_none_on_none_content(self):
        client = _make_client()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client._describe_table(_make_table())
        assert result is None

    async def test_propagates_api_exceptions(self):
        """API exceptions propagate so BaseLLMClient._augment_single can catch them."""
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(side_effect=Exception("API timeout"))

        with pytest.raises(Exception, match="API timeout"):
            await client._describe_table(_make_table())

    async def test_passes_correct_model_to_api(self):
        client = _make_client()
        mock_choice = MagicMock()
        mock_choice.message.content = "A description."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        create_mock = AsyncMock(return_value=mock_response)
        client._client.chat.completions.create = create_mock

        await client._describe_table(_make_table())

        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"

    async def test_system_message_is_included(self):
        client = _make_client()
        mock_choice = MagicMock()
        mock_choice.message.content = "A description."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        create_mock = AsyncMock(return_value=mock_response)
        client._client.chat.completions.create = create_mock

        await client._describe_table(_make_table())

        messages = create_mock.call_args.kwargs["messages"]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles


# ── Column prompt building ─────────────────────────────────────────────────────


class TestBuildColumnPrompt:
    def _table(self) -> TableMetadata:
        return _make_table()

    def _column(self, **kwargs) -> ColumnMetadata:
        return _make_column(**kwargs)

    def test_prompt_contains_table_qualified_name(self):
        prompt = _build_column_prompt(self._column(), self._table())
        assert "public.users" in prompt

    def test_prompt_contains_column_name_and_type(self):
        col = self._column(name="email", data_type="character varying")
        prompt = _build_column_prompt(col, self._table())
        assert "email" in prompt
        assert "character varying" in prompt

    def test_prompt_marks_primary_key(self):
        col = self._column(name="id", is_primary_key=True)
        prompt = _build_column_prompt(col, self._table())
        assert "[PK]" in prompt

    def test_prompt_marks_not_null(self):
        col = self._column(name="id", is_nullable=False)
        prompt = _build_column_prompt(col, self._table())
        assert "[NOT NULL]" in prompt

    def test_prompt_includes_enum_values(self):
        col = self._column(name="status", enum_values=["active", "inactive"])
        prompt = _build_column_prompt(col, self._table())
        assert "active" in prompt
        assert "inactive" in prompt

    def test_prompt_includes_sample_values(self):
        col = self._column(name="email", sample_values=["a@b.com", "c@d.com"])
        prompt = _build_column_prompt(col, self._table())
        assert "a@b.com" in prompt

    def test_prompt_includes_existing_pg_description(self):
        col = self._column(name="id", description="Primary identifier")
        prompt = _build_column_prompt(col, self._table())
        assert "Primary identifier" in prompt

    def test_prompt_includes_table_description_as_context(self):
        table = self._table()
        table.description = "Stores registered users"
        prompt = _build_column_prompt(self._column(), table)
        assert "Stores registered users" in prompt

    def test_prompt_shows_none_table_description_when_missing(self):
        table = self._table()
        table.description = None
        prompt = _build_column_prompt(self._column(), table)
        assert "(none)" in prompt

    def test_prompt_includes_sibling_columns(self):
        table = self._table()  # has id, email, status columns
        col = _make_column(name="id", is_primary_key=True, is_nullable=False)
        prompt = _build_column_prompt(col, table)
        # siblings (email, status) should appear
        assert "email" in prompt

    def test_prompt_truncates_many_siblings(self):
        table = _make_table()
        extra_cols = [_make_column(f"col_{i}") for i in range(15)]
        table.columns = extra_cols
        col = extra_cols[0]
        prompt = _build_column_prompt(col, table)
        assert "more columns" in prompt


# ── OpenAILLMClient._describe_column ─────────────────────────────────────────


class TestOpenAILLMClientDescribeColumn:
    def _col(self) -> ColumnMetadata:
        return _make_column("email", data_type="character varying", is_nullable=False)

    async def test_returns_stripped_content(self):
        client = _make_client()
        mock_choice = MagicMock()
        mock_choice.message.content = "  Stores the user's email address.  "
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client._describe_column(self._col(), _make_table())
        assert result == "Stores the user's email address."

    async def test_returns_none_on_empty_content(self):
        client = _make_client()
        mock_choice = MagicMock()
        mock_choice.message.content = "   "
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client._describe_column(self._col(), _make_table())
        assert result is None

    async def test_returns_none_on_none_content(self):
        client = _make_client()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client._describe_column(self._col(), _make_table())
        assert result is None

    async def test_propagates_api_exceptions(self):
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(side_effect=Exception("API timeout"))

        with pytest.raises(Exception, match="API timeout"):
            await client._describe_column(self._col(), _make_table())

    async def test_system_message_is_included(self):
        client = _make_client()
        mock_choice = MagicMock()
        mock_choice.message.content = "A description."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        create_mock = AsyncMock(return_value=mock_response)
        client._client.chat.completions.create = create_mock

        await client._describe_column(self._col(), _make_table())

        messages = create_mock.call_args.kwargs["messages"]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles


# ── Glossary prompt building ───────────────────────────────────────────────────


class TestBuildGlossaryPrompt:
    def test_prompt_contains_qualified_table_name(self):
        prompt = _build_glossary_prompt(_make_table())
        assert "public.users" in prompt

    def test_prompt_includes_existing_description(self):
        prompt = _build_glossary_prompt(_make_table())
        assert "Existing pg desc" in prompt

    def test_prompt_shows_none_when_no_description(self):
        table = _make_table()
        table.description = None
        prompt = _build_glossary_prompt(table)
        assert "(none)" in prompt

    def test_prompt_includes_row_count(self):
        prompt = _build_glossary_prompt(_make_table())
        assert "5000" in prompt

    def test_prompt_includes_column_names(self):
        prompt = _build_glossary_prompt(_make_table())
        assert "id" in prompt
        assert "email" in prompt

    def test_prompt_includes_json_format_instruction(self):
        prompt = _build_glossary_prompt(_make_table())
        assert "business_term" in prompt
        assert "synonyms" in prompt

    def test_prompt_truncation_note_for_wide_tables(self):
        table = _make_table()
        table.columns = [_make_column(f"col_{i}") for i in range(25)]
        prompt = _build_glossary_prompt(table)
        assert "more columns" in prompt


# ── OpenAILLMClient._infer_glossary ───────────────────────────────────────────


def _valid_glossary_json(terms: list[dict] | None = None) -> str:
    if terms is None:
        terms = [
            {"business_term": "Customer", "description": "A person who buys goods.", "synonyms": ["client"]},
            {"business_term": "Account", "description": "A registered user record.", "synonyms": []},
        ]
    return json.dumps({"terms": terms})


class TestOpenAILLMClientInferGlossary:
    def _mock_response(self, content: str | None):
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    async def test_returns_parsed_glossary_terms(self):
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(return_value=self._mock_response(_valid_glossary_json()))
        terms = await client._infer_glossary(_make_table())
        assert len(terms) == 2
        assert terms[0].business_term == "Customer"
        assert terms[1].business_term == "Account"
        assert terms[0].synonyms == ["client"]

    async def test_enforces_max_five_terms(self):
        six_terms = [{"business_term": f"Term{i}", "description": "desc", "synonyms": []} for i in range(6)]
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(return_value=self._mock_response(_valid_glossary_json(six_terms)))
        terms = await client._infer_glossary(_make_table())
        assert len(terms) == 5

    async def test_returns_empty_list_on_none_content(self):
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(return_value=self._mock_response(None))
        terms = await client._infer_glossary(_make_table())
        assert terms == []

    async def test_returns_empty_list_on_invalid_json(self):
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(return_value=self._mock_response("not valid json {{"))
        terms = await client._infer_glossary(_make_table())
        assert terms == []

    async def test_returns_empty_list_on_missing_terms_key(self):
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(return_value=self._mock_response(json.dumps({"wrong_key": []})))
        terms = await client._infer_glossary(_make_table())
        assert terms == []

    async def test_uses_glossary_max_tokens(self):
        client = _make_client()
        create_mock = AsyncMock(return_value=self._mock_response(_valid_glossary_json()))
        client._client.chat.completions.create = create_mock

        await client._infer_glossary(_make_table())

        assert create_mock.call_args.kwargs["max_tokens"] == _GLOSSARY_MAX_TOKENS

    async def test_passes_json_object_response_format(self):
        client = _make_client()
        create_mock = AsyncMock(return_value=self._mock_response(_valid_glossary_json()))
        client._client.chat.completions.create = create_mock

        await client._infer_glossary(_make_table())

        assert create_mock.call_args.kwargs["response_format"] == {"type": "json_object"}

    async def test_propagates_api_exceptions(self):
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

        with pytest.raises(Exception, match="API error"):
            await client._infer_glossary(_make_table())

    async def test_system_and_user_messages_included(self):
        client = _make_client()
        create_mock = AsyncMock(return_value=self._mock_response(_valid_glossary_json()))
        client._client.chat.completions.create = create_mock

        await client._infer_glossary(_make_table())

        messages = create_mock.call_args.kwargs["messages"]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles
