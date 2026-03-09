"""Latency tracking package — re-exports from tracker module."""

from core.services.latency_tracking.tracker import get_latency_tracker, init_latency_tracker

__all__ = ["init_latency_tracker", "get_latency_tracker"]
