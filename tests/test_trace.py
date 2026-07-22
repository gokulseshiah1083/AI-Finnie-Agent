"""Tests for the JSONL tracing layer.

Tracing is instrumented across the graph, the LLM client and the tool
registry, so its contract matters: spans nest correctly, failures never
propagate, and disabling it makes it free.
"""

import json

import pytest

from src.core import trace
from src.core.config import CONFIG


@pytest.fixture()
def traced(tmp_path, monkeypatch):
    """Enable tracing into a temp file; restore config afterwards."""
    monkeypatch.setitem(
        CONFIG,
        "tracing",
        {
            "enabled": True,
            "trace_log_path": str(tmp_path / "trace.jsonl"),
            "echo_to_console": False,
            "max_value_chars": 50,
        },
    )
    # trace_log_path resolves relative to the repo root; absolute paths pass through
    monkeypatch.setattr(trace, "resolve_path", lambda p: tmp_path / "trace.jsonl")
    return tmp_path / "trace.jsonl"


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_span_writes_record(traced):
    trace.start_trace()
    with trace.span("unit", tool="lookup_customer") as s:
        s["row_count"] = 3

    records = _read(traced)
    assert len(records) == 1
    record = records[0]
    assert record["name"] == "unit"
    assert record["status"] == "ok"
    assert record["tool"] == "lookup_customer"
    assert record["row_count"] == 3
    assert record["trace_id"] == trace.current_trace_id()
    assert record["duration_ms"] >= 0


def test_nested_spans_link_parent(traced):
    trace.start_trace()
    with trace.span("outer"):
        with trace.span("inner"):
            pass

    records = {r["name"]: r for r in _read(traced)}
    assert records["inner"]["parent_span_id"] == records["outer"]["span_id"]
    assert records["outer"]["parent_span_id"] is None
    assert records["inner"]["trace_id"] == records["outer"]["trace_id"]


def test_exception_marks_error_and_reraises(traced):
    trace.start_trace()
    with pytest.raises(ValueError):
        with trace.span("failing"):
            raise ValueError("boom")

    record = _read(traced)[0]
    assert record["status"] == "error"
    assert "boom" in record["error"]


def test_long_values_are_clipped(traced):
    trace.start_trace()
    with trace.span("clip", payload="x" * 500):
        pass

    record = _read(traced)[0]
    assert len(record["payload"]) < 500
    assert "chars]" in record["payload"]


def test_disabled_tracing_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setitem(CONFIG, "tracing", {"enabled": False})
    with trace.span("ignored") as s:
        s["unused"] = 1
    assert not (tmp_path / "trace.jsonl").exists()


def test_read_trace_entries_newest_first(traced):
    trace.start_trace()
    with trace.span("first"):
        pass
    with trace.span("second"):
        pass

    entries = trace.read_trace_entries(limit=10)
    assert [e["name"] for e in entries] == ["second", "first"]

    only_one = trace.read_trace_entries(limit=1)
    assert len(only_one) == 1


def test_write_failure_does_not_raise(traced, monkeypatch):
    def broken_path():
        raise OSError("disk gone")

    monkeypatch.setattr(trace, "trace_log_path", broken_path)
    trace.start_trace()
    # A broken path must degrade to a warning, never an exception.
    with trace.span("survives"):
        pass
