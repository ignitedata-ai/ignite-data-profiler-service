"""Unit tests for column type classification utility."""

from __future__ import annotations

import pytest

from core.services.column_stats import ColumnTypeCategory, classify_column_type, classify_columns


class TestClassifyColumnType:
    @pytest.mark.parametrize(
        "data_type",
        [
            "integer",
            "bigint",
            "smallint",
            "decimal",
            "numeric",
            "real",
            "double precision",
            "float",
            "int",
            "tinyint",
            "mediumint",
            "serial",
            "money",
        ],
    )
    def test_numeric_types(self, data_type: str):
        assert classify_column_type(data_type) == ColumnTypeCategory.NUMERIC

    @pytest.mark.parametrize(
        "data_type",
        ["character varying", "varchar", "char", "text", "citext", "bpchar", "tinytext", "mediumtext", "longtext", "name"],
    )
    def test_string_types(self, data_type: str):
        assert classify_column_type(data_type) == ColumnTypeCategory.STRING

    @pytest.mark.parametrize("data_type", ["boolean", "bool"])
    def test_boolean_types(self, data_type: str):
        assert classify_column_type(data_type) == ColumnTypeCategory.BOOLEAN

    @pytest.mark.parametrize(
        "data_type",
        ["date", "time", "timestamp", "timestamp with time zone", "timestamp without time zone", "datetime", "year", "interval"],
    )
    def test_temporal_types(self, data_type: str):
        assert classify_column_type(data_type) == ColumnTypeCategory.TEMPORAL

    @pytest.mark.parametrize("data_type", ["USER-DEFINED", "json", "jsonb", "bytea", "blob", "uuid", "xml"])
    def test_other_types(self, data_type: str):
        assert classify_column_type(data_type) == ColumnTypeCategory.OTHER

    def test_case_insensitive(self):
        assert classify_column_type("INTEGER") == ColumnTypeCategory.NUMERIC
        assert classify_column_type("Boolean") == ColumnTypeCategory.BOOLEAN
        assert classify_column_type("  varchar  ") == ColumnTypeCategory.STRING

    def test_empty_string_is_other(self):
        assert classify_column_type("") == ColumnTypeCategory.OTHER


class TestClassifyColumns:
    def test_groups_mixed_columns(self):
        columns = [
            {"name": "id", "data_type": "integer"},
            {"name": "name", "data_type": "varchar"},
            {"name": "active", "data_type": "boolean"},
            {"name": "created", "data_type": "timestamp"},
            {"name": "meta", "data_type": "jsonb"},
        ]
        result = classify_columns(columns)
        assert len(result[ColumnTypeCategory.NUMERIC]) == 1
        assert result[ColumnTypeCategory.NUMERIC][0]["name"] == "id"
        assert len(result[ColumnTypeCategory.STRING]) == 1
        assert len(result[ColumnTypeCategory.BOOLEAN]) == 1
        assert len(result[ColumnTypeCategory.TEMPORAL]) == 1
        assert len(result[ColumnTypeCategory.OTHER]) == 1

    def test_empty_list(self):
        result = classify_columns([])
        for cat in ColumnTypeCategory:
            assert result[cat] == []

    def test_all_same_type(self):
        columns = [
            {"name": "a", "data_type": "integer"},
            {"name": "b", "data_type": "bigint"},
        ]
        result = classify_columns(columns)
        assert len(result[ColumnTypeCategory.NUMERIC]) == 2
        assert len(result[ColumnTypeCategory.STRING]) == 0
