"""Unit tests for PostgresProfilerService observability methods."""

from __future__ import annotations

from unittest.mock import MagicMock

from core.services.postgres.profiler import PostgresProfilerService


def _make_conn(ssh_tunnel=None):
    conn = MagicMock()
    conn.host = "db.internal"
    conn.port = 5432
    conn.database = "mydb"
    conn.username = "dbuser"
    conn.ssh_tunnel = ssh_tunnel
    return conn


def _make_tunnel(host="54.1.2.3", port=22, username="ec2-user"):
    t = MagicMock()
    t.host = host
    t.port = port
    t.username = username
    return t


class TestPostgresProfilerServiceObservability:
    def test_log_context_without_tunnel(self):
        svc = PostgresProfilerService()
        ctx = svc._log_context(_make_conn())
        assert ctx["host"] == "db.internal"
        assert ctx["database"] == "mydb"
        assert ctx["username"] == "dbuser"
        assert "ssh_tunnel_host" not in ctx

    def test_log_context_with_tunnel(self):
        svc = PostgresProfilerService()
        ctx = svc._log_context(_make_conn(ssh_tunnel=_make_tunnel()))
        assert ctx["ssh_tunnel_host"] == "54.1.2.3"
        assert ctx["ssh_tunnel_port"] == 22
        assert ctx["ssh_tunnel_username"] == "ec2-user"

    def test_span_attributes_without_tunnel(self):
        svc = PostgresProfilerService()
        attrs = svc._span_attributes(_make_conn())
        assert attrs["db.system"] == "postgresql"
        assert attrs["db.name"] == "mydb"
        assert "net.ssh_tunnel.host" not in attrs

    def test_span_attributes_with_tunnel(self):
        svc = PostgresProfilerService()
        attrs = svc._span_attributes(_make_conn(ssh_tunnel=_make_tunnel(host="bastion.example.com", port=2222)))
        assert attrs["net.ssh_tunnel.host"] == "bastion.example.com"
        assert attrs["net.ssh_tunnel.port"] == 2222
