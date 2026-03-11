"""Tests for the schema introspector (Stage 1)."""

from __future__ import annotations

from core.api.v1.schemas.profiler import (
    ColumnMetadata,
    IndexMetadata,
    RelationshipMetadata,
    TableMetadata,
)
from core.services.filters.schema_introspector import (
    NullSchemaIntrospector,
    _parse_check_values,
)


def _make_table(
    name: str = "orders",
    schema: str = "public",
    row_count: int = 10000,
    columns: list[ColumnMetadata] | None = None,
    indexes: list[IndexMetadata] | None = None,
    relationships: list[RelationshipMetadata] | None = None,
) -> TableMetadata:
    if columns is None:
        columns = [
            ColumnMetadata(
                name="id",
                ordinal_position=1,
                data_type="integer",
                is_nullable=False,
                column_default=None,
                character_maximum_length=None,
                numeric_precision=None,
                numeric_scale=None,
                is_primary_key=True,
            ),
            ColumnMetadata(
                name="status",
                ordinal_position=2,
                data_type="varchar(50)",
                is_nullable=False,
                column_default=None,
                character_maximum_length=50,
                numeric_precision=None,
                numeric_scale=None,
                enum_values=["active", "inactive", "pending"],
            ),
            ColumnMetadata(
                name="customer_id",
                ordinal_position=3,
                data_type="integer",
                is_nullable=False,
                column_default=None,
                character_maximum_length=None,
                numeric_precision=None,
                numeric_scale=None,
            ),
            ColumnMetadata(
                name="total_amount",
                ordinal_position=4,
                data_type="decimal(18,2)",
                is_nullable=True,
                column_default=None,
                character_maximum_length=None,
                numeric_precision=18,
                numeric_scale=2,
            ),
        ]
    return TableMetadata(
        name=name,
        schema=schema,
        owner="admin",
        description=None,
        row_count=row_count,
        size_bytes=None,
        total_size_bytes=None,
        data_freshness=None,
        columns=columns,
        indexes=indexes,
        relationships=relationships,
    )


class TestExtractSignals:
    def test_primary_key_detected(self):
        introspector = NullSchemaIntrospector()
        table = _make_table()
        signals = introspector.extract_signals(table)
        assert signals["id"].is_primary_key is True
        assert signals["status"].is_primary_key is False

    def test_enum_detected(self):
        introspector = NullSchemaIntrospector()
        table = _make_table()
        signals = introspector.extract_signals(table)
        assert signals["status"].has_enum_type is True
        assert signals["id"].has_enum_type is False

    def test_foreign_key_detected(self):
        introspector = NullSchemaIntrospector()
        rels = [
            RelationshipMetadata(
                constraint_name="fk_customer",
                from_column="customer_id",
                to_schema="public",
                to_table="customers",
                to_column="id",
                on_update="NO ACTION",
                on_delete="NO ACTION",
            )
        ]
        table = _make_table(relationships=rels)
        signals = introspector.extract_signals(table)
        assert signals["customer_id"].is_foreign_key is True
        assert signals["customer_id"].fk_referenced_table == "public.customers"

    def test_non_unique_index_detected(self):
        introspector = NullSchemaIntrospector()
        indexes = [
            IndexMetadata(name="idx_status", columns=["status"], is_unique=False, is_primary=False, index_type="btree"),
        ]
        table = _make_table(indexes=indexes)
        signals = introspector.extract_signals(table)
        assert signals["status"].is_non_unique_index is True

    def test_composite_index_partners(self):
        introspector = NullSchemaIntrospector()
        indexes = [
            IndexMetadata(
                name="idx_status_customer",
                columns=["status", "customer_id"],
                is_unique=False,
                is_primary=False,
                index_type="btree",
            ),
        ]
        table = _make_table(indexes=indexes)
        signals = introspector.extract_signals(table)
        assert "customer_id" in signals["status"].composite_index_partners
        assert "status" in signals["customer_id"].composite_index_partners


class TestScoreSignals:
    def test_enum_gets_high_score(self):
        introspector = NullSchemaIntrospector()
        table = _make_table()
        signals = introspector.extract_signals(table)
        introspector.score_signals(signals)
        assert signals["status"].schema_score == 0.95  # has_enum_type

    def test_pk_gets_low_score(self):
        introspector = NullSchemaIntrospector()
        table = _make_table()
        signals = introspector.extract_signals(table)
        introspector.score_signals(signals)
        assert signals["id"].schema_score == 0.05  # single-column PK

    def test_fk_small_dim_table(self):
        introspector = NullSchemaIntrospector()
        rels = [
            RelationshipMetadata(
                constraint_name="fk_customer",
                from_column="customer_id",
                to_schema="public",
                to_table="customers",
                to_column="id",
                on_update="NO ACTION",
                on_delete="NO ACTION",
            )
        ]
        table = _make_table(relationships=rels)
        signals = introspector.extract_signals(table)
        signals["customer_id"].fk_referenced_table_row_count = 100
        introspector.score_signals(signals)
        assert signals["customer_id"].schema_score == 0.9

    def test_fk_large_table(self):
        introspector = NullSchemaIntrospector()
        rels = [
            RelationshipMetadata(
                constraint_name="fk_customer",
                from_column="customer_id",
                to_schema="public",
                to_table="customers",
                to_column="id",
                on_update="NO ACTION",
                on_delete="NO ACTION",
            )
        ]
        table = _make_table(relationships=rels)
        signals = introspector.extract_signals(table)
        signals["customer_id"].fk_referenced_table_row_count = 10000
        introspector.score_signals(signals)
        assert signals["customer_id"].schema_score == 0.6

    def test_check_constraint(self):
        introspector = NullSchemaIntrospector()
        table = _make_table()
        signals = introspector.extract_signals(table)
        check_constraints = {"total_amount": ["positive", "non_negative"]}
        introspector.score_signals(signals, check_constraints)
        assert signals["total_amount"].has_check_constraint is True
        assert signals["total_amount"].schema_score == 0.95

    def test_no_signals_zero_score(self):
        introspector = NullSchemaIntrospector()
        table = _make_table()
        signals = introspector.extract_signals(table)
        introspector.score_signals(signals)
        assert signals["total_amount"].schema_score == 0.0


class TestTableRoleClassification:
    def test_fact_table(self):
        introspector = NullSchemaIntrospector()
        rels = [
            RelationshipMetadata(
                constraint_name="fk1",
                from_column="customer_id",
                to_schema="public",
                to_table="customers",
                to_column="id",
                on_update="NO ACTION",
                on_delete="NO ACTION",
            ),
            RelationshipMetadata(
                constraint_name="fk2",
                from_column="status",
                to_schema="public",
                to_table="statuses",
                to_column="id",
                on_update="NO ACTION",
                on_delete="NO ACTION",
            ),
        ]
        cols = [
            ColumnMetadata(
                name="id",
                ordinal_position=1,
                data_type="integer",
                is_nullable=False,
                column_default=None,
                character_maximum_length=None,
                numeric_precision=None,
                numeric_scale=None,
                is_primary_key=True,
            ),
            ColumnMetadata(
                name="customer_id",
                ordinal_position=2,
                data_type="integer",
                is_nullable=False,
                column_default=None,
                character_maximum_length=None,
                numeric_precision=None,
                numeric_scale=None,
            ),
            ColumnMetadata(
                name="amount",
                ordinal_position=3,
                data_type="decimal",
                is_nullable=True,
                column_default=None,
                character_maximum_length=None,
                numeric_precision=None,
                numeric_scale=None,
            ),
            ColumnMetadata(
                name="tax",
                ordinal_position=4,
                data_type="decimal",
                is_nullable=True,
                column_default=None,
                character_maximum_length=None,
                numeric_precision=None,
                numeric_scale=None,
            ),
        ]
        table = _make_table(row_count=50000, columns=cols, relationships=rels)
        assert introspector.classify_table_role(table) == "fact"

    def test_dimension_table(self):
        introspector = NullSchemaIntrospector()
        cols = [
            ColumnMetadata(
                name="id",
                ordinal_position=1,
                data_type="integer",
                is_nullable=False,
                column_default=None,
                character_maximum_length=None,
                numeric_precision=None,
                numeric_scale=None,
                is_primary_key=True,
            ),
            ColumnMetadata(
                name="name",
                ordinal_position=2,
                data_type="varchar",
                is_nullable=False,
                column_default=None,
                character_maximum_length=None,
                numeric_precision=None,
                numeric_scale=None,
            ),
            ColumnMetadata(
                name="code",
                ordinal_position=3,
                data_type="varchar",
                is_nullable=True,
                column_default=None,
                character_maximum_length=None,
                numeric_precision=None,
                numeric_scale=None,
            ),
        ]
        table = _make_table(row_count=50, columns=cols, relationships=[])
        assert introspector.classify_table_role(table) == "dimension"

    def test_unknown_table(self):
        introspector = NullSchemaIntrospector()
        table = _make_table(row_count=5000)
        assert introspector.classify_table_role(table) == "unknown"


class TestParseCheckValues:
    def test_simple_in_list(self):
        defn = "CHECK ((status = ANY (ARRAY['active'::text, 'inactive'::text, 'pending'::text])))"
        values = _parse_check_values(defn)
        assert values == ["active", "inactive", "pending"]

    def test_no_values(self):
        defn = "CHECK ((amount > 0))"
        values = _parse_check_values(defn)
        assert values == []
