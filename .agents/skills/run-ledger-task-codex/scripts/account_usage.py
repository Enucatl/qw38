#!/usr/bin/env python3
"""Extract final token telemetry from Codex rollout JSONL sessions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _record(path: Path) -> dict[str, Any] | None:
    events: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    events.append(value)
    except OSError:
        return None

    metadata = next(
        (
            event.get("payload")
            for event in events
            if event.get("type") == "session_meta"
            and isinstance(event.get("payload"), dict)
        ),
        {},
    )
    contexts = [
        event.get("payload")
        for event in events
        if event.get("type") == "turn_context"
        and isinstance(event.get("payload"), dict)
    ]
    usage_events = [
        event
        for event in events
        if event.get("type") == "event_msg"
        and isinstance(event.get("payload"), dict)
        and event["payload"].get("type") == "token_count"
        and isinstance(event["payload"].get("info"), dict)
        and isinstance(event["payload"]["info"].get("total_token_usage"), dict)
    ]
    if not usage_events:
        return None

    source = metadata.get("source")
    if isinstance(source, dict):
        subagent = source.get("subagent")
        thread_spawn = (
            subagent.get("thread_spawn") if isinstance(subagent, dict) else None
        )
        agent_path = (
            thread_spawn.get("agent_path", "subagent")
            if isinstance(thread_spawn, dict)
            else "subagent"
        )
    else:
        agent_path = "coordinator"
    context = contexts[-1] if contexts else {}
    usage = usage_events[-1]["payload"]["info"]["total_token_usage"]
    timestamps = [
        _timestamp(event.get("timestamp"))
        for event in events
        if _timestamp(event.get("timestamp")) is not None
    ]
    elapsed = None
    if len(timestamps) >= 2:
        elapsed = round(timestamps[-1] - timestamps[0], 3)
    return {
        "session_id": metadata.get("id"),
        "agent_path": agent_path,
        "model": context.get("model"),
        "reasoning_effort": context.get("effort", context.get("reasoning_effort")),
        "input_tokens": usage.get("input_tokens", 0),
        "cached_input_tokens": usage.get("cached_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "reasoning_output_tokens": usage.get("reasoning_output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "elapsed_seconds": elapsed,
        "source_file": str(path),
    }


def collect(root: Path, agent_path: str | None) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*.jsonl")):
        record = _record(path)
        if record is not None and (
            agent_path is None or record["agent_path"] == agent_path
        ):
            records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions_root", type=Path)
    parser.add_argument("--agent-path")
    args = parser.parse_args()
    json.dump(collect(args.sessions_root, args.agent_path), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
