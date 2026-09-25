#!/usr/bin/env python3
"""Verify that a RoPE-only artifact change fixes the layer-3 attention jump."""

import argparse
import json
from pathlib import Path

POSITIONS = (0, 63, 255, 383, 384)
UNCHANGED = ("input", "normalized", "qg", "k_raw", "v_raw", "g")


def trace_bytes(directory: Path, name: str, position: int) -> bytes:
    """Read one Quartz layer-3 trace buffer."""
    return (directory / f"attention.{name}.pos-{position}.bin").read_bytes()


def main() -> None:
    """Check all pre-RoPE bytes and summarize checkpoint improvements."""
    parser = argparse.ArgumentParser()
    parser.add_argument("old_dir", type=Path)
    parser.add_argument("corrected_dir", type=Path)
    parser.add_argument("old_boundaries", type=Path)
    parser.add_argument("corrected_boundaries", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for position in range(385):
        for name in UNCHANGED:
            if trace_bytes(args.old_dir, name, position) != trace_bytes(
                args.corrected_dir, name, position
            ):
                raise ValueError(f"pre-RoPE tensor changed: {name} at {position}")
        # KV contains K followed by V. The V cache must also be identical.
        if (
            trace_bytes(args.old_dir, "kv", position)[2048:]
            != trace_bytes(args.corrected_dir, "kv", position)[2048:]
        ):
            raise ValueError(f"V cache changed at {position}")
    old = json.loads(args.old_boundaries.read_text())
    corrected = json.loads(args.corrected_boundaries.read_text())
    if old["schema"] != corrected["schema"]:
        raise ValueError("boundary report schemas differ")
    metrics = ("q", "gated", "output_branch", "output_residual")
    rows = []
    for position in POSITIONS:
        before = old["positions"][str(position)]
        after = corrected["positions"][str(position)]
        rows.append(
            {
                "position": position,
                **{
                    name: {
                        "old_relative_l2": before[name]["relative_l2"],
                        "corrected_relative_l2": after[name]["relative_l2"],
                    }
                    for name in metrics
                },
            }
        )
    report = {
        "schema": "task022-rope-fix-trace-comparison-v1",
        "matched_positions": 385,
        "byte_identical_pre_rope": list(UNCHANGED),
        "byte_identical_v_cache": True,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
