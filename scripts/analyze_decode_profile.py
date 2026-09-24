# /// script
# requires-python = "==3.12.*"
# ///
"""Summarize an Nsight Systems SQLite trace of qw38-decode --profile."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def category(name: str, vocabulary: bool) -> str:
    """Assign one GPU kernel to the requested decode family."""
    if vocabulary:
        return "vocabulary_readout"
    if name.startswith("decode_mmv"):
        return "projection"
    if name == "gdn_recurrence_kernel":
        return "recurrent"
    if name.startswith("attention_"):
        return "attention"
    return "other"


def summarize(path: Path) -> dict[str, object]:
    """Compute decode-only CPU API and GPU kernel time from the trace."""
    with sqlite3.connect(path) as db:
        ranges = db.execute(
            "SELECT start, end, text FROM NVTX_EVENTS WHERE text IN "
            "('model_load', 'session_bind', 'prompt_ingestion', "
            "'populated_decode', 'decode_token', 'vocabulary_readout') "
            "ORDER BY start"
        ).fetchall()
        by_name: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for start, end, name in ranges:
            if end is None:
                raise ValueError(f"open NVTX range: {name}")
            by_name[name].append((start, end))
        if len(by_name["populated_decode"]) != 1:
            raise ValueError("expected one populated_decode NVTX range")
        for name in ("model_load", "session_bind", "prompt_ingestion"):
            if len(by_name[name]) > 1:
                raise ValueError(f"expected at most one {name} NVTX range")
        decode_start, decode_end = by_name["populated_decode"][0]
        steps = sum(
            decode_start <= start and end <= decode_end
            for start, end in by_name["decode_token"]
        )
        if steps == 0:
            raise ValueError("populated decode contains no complete decode step")
        readout = [
            (start, end)
            for start, end in by_name["vocabulary_readout"]
            if decode_start <= start and end <= decode_end
        ]
        if len(readout) != steps:
            raise ValueError("vocabulary readout count differs from decode steps")

        api_ns: dict[str, int] = defaultdict(int)
        api_count: dict[str, int] = defaultdict(int)
        launches: dict[int, bool] = {}
        for start, end, correlation, name in db.execute(
            "SELECT r.start, r.end, r.correlationId, s.value "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON s.id = r.nameId "
            "WHERE r.start >= ? AND r.start < ?",
            (decode_start, decode_end),
        ):
            if name.startswith("cudaStreamSynchronize"):
                kind = "stream_sync"
            elif name.startswith("cudaLaunchKernel"):
                kind = "kernel_launch"
                launches[correlation] = any(a <= start < b for a, b in readout)
            elif name.startswith("cudaMemcpyAsync"):
                kind = "memcpy_api"
            else:
                kind = "other_cuda_api"
            api_ns[kind] += end - start
            api_count[kind] += 1

        kernel_ns: dict[str, int] = defaultdict(int)
        kernel_count: dict[str, int] = defaultdict(int)
        unmatched = 0
        for start, end, correlation, name in db.execute(
            "SELECT k.start, k.end, k.correlationId, s.value "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id = k.shortName "
            "WHERE k.start >= ? AND k.start < ?",
            (decode_start, decode_end),
        ):
            if correlation not in launches:
                unmatched += 1
                continue
            kind = category(name, launches[correlation])
            kernel_ns[kind] += end - start
            kernel_count[kind] += 1
        if unmatched:
            raise ValueError(f"{unmatched} decode kernels lack a traced launch")

        def ms(ns: int) -> float:
            """Convert Nsight nanoseconds to milliseconds."""
            return round(ns / 1_000_000, 4)

        def metrics(duration: dict[str, int], count: dict[str, int]) -> dict:
            """Format total and per-step durations with event counts."""
            return {
                name: {
                    "count": count[name],
                    "total_ms": ms(duration[name]),
                    "ms_per_step": ms(duration[name] / steps),
                }
                for name in sorted(duration)
            }

        return {
            "source": str(path),
            "decode_steps": steps,
            "phase_wall_ms": {
                name: ms(end - start)
                for name in (
                    "model_load",
                    "session_bind",
                    "prompt_ingestion",
                    "populated_decode",
                )
                for start, end in by_name[name]
            },
            "decode_wall_ms_per_step": ms((decode_end - decode_start) / steps),
            "readout_wall_ms_per_step": ms(
                sum(end - start for start, end in readout) / steps
            ),
            "cpu_cuda_api": metrics(api_ns, api_count),
            "gpu_kernels": metrics(kernel_ns, kernel_count),
            "note": "CPU CUDA API and GPU kernel durations overlap; do not add them.",
        }


def main() -> None:
    """Read a trace export and print one JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path, help="nsys export --type sqlite output")
    args = parser.parse_args()
    print(json.dumps(summarize(args.sqlite), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
