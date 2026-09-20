from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def summarize_usage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    by_provider: dict[str, dict[str, Any]] = {}
    total_input = 0
    total_output = 0
    reported_calls = 0

    for call in calls:
        provider = str(call.get("provider") or "unknown")
        usage = call.get("usage") or {}
        reported = bool(usage.get("reported"))
        input_tokens = _safe_int(usage.get("input_tokens"))
        output_tokens = _safe_int(usage.get("output_tokens"))

        bucket = by_provider.setdefault(
            provider,
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "reported_calls": 0,
                "unreported_calls": 0,
            },
        )
        if reported:
            bucket["input_tokens"] += input_tokens
            bucket["output_tokens"] += output_tokens
            bucket["total_tokens"] += input_tokens + output_tokens
            bucket["reported_calls"] += 1
            total_input += input_tokens
            total_output += output_tokens
            reported_calls += 1
        else:
            bucket["unreported_calls"] += 1

    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "reported_calls": reported_calls,
        "unreported_calls": max(0, len(calls) - reported_calls),
        "all_calls_reported": reported_calls == len(calls) if calls else True,
        "by_provider": by_provider,
        "note": (
            "Totals include only usage explicitly reported by provider APIs; "
            "unreported calls are not estimated."
        ),
    }


class ConversationJSONLogger:
    """Append complete turn records to one JSON file per CLI session."""

    def __init__(self, base_dir: str | Path, *, enabled: bool = True):
        self.enabled = bool(enabled)
        self.base_dir = Path(base_dir)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.session_id = f"{stamp}_{uuid.uuid4().hex[:10]}"
        self.path = self.base_dir / f"session_{self.session_id}.json"
        self.document: dict[str, Any] = {
            "schema_version": 1,
            "session_id": self.session_id,
            "started_at": utc_now_iso(),
            "pid": os.getpid(),
            "turns": [],
        }

    def append_turn(
        self,
        *,
        question: str,
        answer: Any,
        pipeline: str,
        calls: list[dict[str, Any]],
        events: list[dict[str, Any]],
        stats: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path | None:
        if not self.enabled:
            return None

        ordered_calls = sorted(calls, key=lambda item: _safe_int(item.get("ts_ns")))
        turn = {
            "turn": len(self.document["turns"]) + 1,
            "timestamp": utc_now_iso(),
            "question": question,
            "answer": answer,
            "pipeline": pipeline,
            "token_usage": summarize_usage(ordered_calls),
            "thought_log": {
                "api_calls": ordered_calls,
                "jevnet_events": events,
            },
            "stats": stats or {},
        }
        if extra:
            turn["extra"] = extra
        self.document["turns"].append(turn)
        self._flush()
        return self.path

    def _flush(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = json.dumps(self.document, ensure_ascii=False, indent=2)
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.path)


def make_call_record(
    *,
    provider: str,
    model: str,
    request: dict[str, Any],
    response: dict[str, Any] | None,
    usage: dict[str, Any],
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ts_ns": time.time_ns(),
        "timestamp": utc_now_iso(),
        "provider": provider,
        "model": model,
        "request": request,
        "response": response,
        "usage": usage,
        "error": error,
    }
