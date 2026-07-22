"""Lightweight execution tracing for debugging.

WHY NOT OPENTELEMETRY
---------------------
The debugging question this project actually gets is "what did that turn do?" —
which node ran, which tools were called with which filters, what each model call
cost, and where the time went. A dependency-free JSONL span writer answers that
with grep and `jq`, works identically on Streamlit Cloud and locally, and adds
no import weight to the lazy-load path. If the project later ships to real
infrastructure, this module is the single seam where an OTLP exporter would go.

SHAPE
-----
One JSON object per completed span, appended to `tracing.trace_log_path`:

    {"ts": "...", "trace_id": "6f1c...", "span_id": "a3d2...", "parent_span_id":
     null, "name": "turn", "duration_ms": 8123, "status": "ok", ...attributes}

`trace_id` groups every span of one query turn; `parent_span_id` reconstructs
the tree (turn → node → llm/tool). Spans are written on EXIT, so the file is
ordered by completion time, not start time — sort by trace_id then ts to read a
turn top-down.

Tracing must never break the answer: every write is wrapped, and a failure
downgrades to a logger warning.
"""

import contextvars
import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from src.core.config import CONFIG, resolve_path

logger = logging.getLogger(__name__)

_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "trace_id", default=None
)
_parent_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "parent_span", default=None
)


def _cfg() -> dict[str, Any]:
    return CONFIG.get("tracing") or {}


def tracing_enabled() -> bool:
    return bool(_cfg().get("enabled", False))


def trace_log_path():
    return resolve_path(_cfg().get("trace_log_path", "logs/trace.jsonl"))


def current_trace_id() -> str | None:
    return _trace_id.get()


def _clip(value: Any) -> Any:
    """Bound attribute size so one giant tool result can't bloat the file."""
    limit = int(_cfg().get("max_value_chars", 300))
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"…[+{len(value) - limit} chars]"
    if isinstance(value, (dict, list)):
        rendered = json.dumps(value, default=str)
        if len(rendered) > limit:
            return rendered[:limit] + f"…[+{len(rendered) - limit} chars]"
    return value


def _write(record: dict[str, Any]) -> None:
    try:
        path = trace_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        if _cfg().get("echo_to_console", False):
            logger.info(
                "[trace] %s (%sms) %s",
                record.get("name"),
                record.get("duration_ms"),
                record.get("status"),
            )
    except Exception as exc:  # noqa: BLE001 - tracing must never break the answer
        logger.warning("Trace write failed: %s", exc)


def start_trace(name: str = "turn", **attributes: Any) -> str:
    """Open a new trace context and return its id.

    Called once per query turn (see run_query). Nested spans opened after this
    call inherit the trace id automatically via contextvars.
    """
    trace_id = uuid.uuid4().hex[:16]
    _trace_id.set(trace_id)
    _parent_span.set(None)
    return trace_id


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[dict[str, Any]]:
    """Trace one unit of work.

    Yields a mutable dict; anything the caller puts in it is written with the
    span on exit — the idiom for attaching results discovered mid-span:

        with span("llm", agent=self.name) as s:
            response = client.messages.create(...)
            s["output_tokens"] = response.usage.output_tokens
    """
    extra: dict[str, Any] = {}
    if not tracing_enabled():
        yield extra
        return

    span_id = uuid.uuid4().hex[:16]
    parent = _parent_span.get()
    token = _parent_span.set(span_id)
    started = time.perf_counter()
    status, error = "ok", None
    try:
        yield extra
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _parent_span.reset(token)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "trace_id": _trace_id.get(),
            "span_id": span_id,
            "parent_span_id": parent,
            "name": name,
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "status": status,
        }
        if error:
            record["error"] = error
        for key, value in {**attributes, **extra}.items():
            record.setdefault(key, _clip(value))
        _write(record)


def read_trace_entries(
    limit: int = 100, trace_id: str | None = None
) -> list[dict[str, Any]]:
    """Tail of the trace log, newest first. Used by the Request History tab."""
    path = trace_log_path()
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if trace_id and entry.get("trace_id") != trace_id:
                    continue
                entries.append(entry)
    except OSError as exc:
        logger.warning("Trace read failed: %s", exc)
        return []
    return list(reversed(entries))[:limit]
