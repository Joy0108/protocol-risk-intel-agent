"""Minimal span tracer.

Emits OpenTelemetry-shaped spans to a JSONL file so a run can be replayed and
diffed offline. If ``arize-phoenix`` and ``opentelemetry-sdk`` are installed and
``PRIA_PHOENIX_ENDPOINT`` is set, spans are additionally forwarded to Phoenix;
otherwise the local recorder is the only sink and nothing else in the codebase
has to know the difference.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_ACTIVE: list[Tracer] = []


class Tracer:
    def __init__(self, run_id: str | None = None, out_path: Path | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.out_path = out_path
        self.spans: list[dict[str, Any]] = []
        self._stack: list[str] = []
        self._otel = _try_phoenix()

    @contextmanager
    def span(self, name: str, kind: str = "INTERNAL", **attrs: Any) -> Iterator[dict[str, Any]]:
        span_id = uuid.uuid4().hex[:16]
        parent = self._stack[-1] if self._stack else None
        record: dict[str, Any] = {
            "run_id": self.run_id,
            "span_id": span_id,
            "parent_id": parent,
            "name": name,
            "kind": kind,
            "attributes": dict(attrs),
            "events": [],
            "status": "OK",
        }
        self._stack.append(span_id)
        start = time.perf_counter()
        try:
            yield record
        except Exception as exc:  # pragma: no cover - defensive
            record["status"] = "ERROR"
            record["attributes"]["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["duration_ms"] = round((time.perf_counter() - start) * 1000, 3)
            self._stack.pop()
            self.spans.append(record)
            self._forward(record)

    def event(self, message: str, **attrs: Any) -> None:
        if self.spans or self._stack:
            target = next((s for s in reversed(self.spans) if s["span_id"] == (self._stack[-1] if self._stack else None)), None)
            if target is not None:
                target["events"].append({"message": message, **attrs})
                return
        self.spans.append({"run_id": self.run_id, "name": "event", "attributes": {"message": message, **attrs}})

    def _forward(self, record: dict[str, Any]) -> None:
        if self._otel is None:
            return
        try:  # pragma: no cover - only runs with phoenix installed
            with self._otel.start_as_current_span(record["name"]) as s:
                for key, value in record["attributes"].items():
                    s.set_attribute(str(key), value if isinstance(value, (str, int, float, bool)) else json.dumps(value, default=str))
        except Exception:
            self._otel = None

    def flush(self) -> Path | None:
        if self.out_path is None:
            return None
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as fh:
            for record in self.spans:
                fh.write(json.dumps(record, default=str) + "\n")
        return self.out_path


def _try_phoenix():
    endpoint = os.environ.get("PRIA_PHOENIX_ENDPOINT")
    if not endpoint:
        return None
    try:  # pragma: no cover - optional dependency
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        return trace.get_tracer("pria")
    except Exception:
        return None


@contextmanager
def tracer(run_id: str | None = None, out_path: Path | None = None) -> Iterator[Tracer]:
    t = Tracer(run_id=run_id, out_path=out_path)
    _ACTIVE.append(t)
    try:
        yield t
    finally:
        _ACTIVE.pop()
        t.flush()


def current() -> Tracer:
    return _ACTIVE[-1] if _ACTIVE else Tracer()
