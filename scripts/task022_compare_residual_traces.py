#!/usr/bin/env python3
# /// script
# requires-python = "==3.12.*"
# dependencies = ["numpy==2.5.3"]
# ///
"""Compare actual llama.cpp and Quartz residuals on one fixed token stream."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_vector(path: Path, offset: int = 0) -> np.ndarray:
    """Read one finite 5120-element residual at a file offset in vectors."""
    with path.open("rb") as file:
        file.seek(offset * 5120 * 4)
        result = np.fromfile(file, dtype="<f4", count=5120)
    if result.size != 5120 or not np.isfinite(result).all():
        raise ValueError(f"missing or invalid residual: {path}")
    return result


def main() -> None:
    """Report residual differences by layer and token position."""
    parser = argparse.ArgumentParser()
    parser.add_argument("llama_dir", type=Path)
    parser.add_argument("quartz_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    mixer_rows = []
    for position in (0, 63, 255, 383, 384):
        phase = "prefill" if position < 384 else "decode"
        offset = (0, 63, 255, 383).index(position) if phase == "prefill" else 0
        for layer in range(64):
            llama_path = args.llama_dir / f"residual.l_out-{layer}.{phase}.bin"
            quartz_path = args.quartz_dir / f"residual.pos-{position}.layer-{layer}.f32"
            llama = read_vector(llama_path, offset)
            quartz = read_vector(quartz_path)
            delta = np.abs(quartz - llama)
            rows.append(
                {
                    "position": position,
                    "layer": layer,
                    "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(llama)),
                    "mean_abs": float(np.mean(delta)),
                    "p99_abs": float(np.percentile(delta, 99)),
                    "max_abs": float(np.max(delta)),
                }
            )
        for layer in (3, 7, 31, 63):
            llama = read_vector(
                args.llama_dir / f"residual.attn_residual-{layer}.{phase}.bin",
                offset,
            )
            quartz = read_vector(
                args.quartz_dir / f"mixer.pos-{position}.layer-{layer}.f32"
            )
            delta = np.abs(quartz - llama)
            mixer_rows.append(
                {
                    "position": position,
                    "layer": layer,
                    "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(llama)),
                    "mean_abs": float(np.mean(delta)),
                    "p99_abs": float(np.percentile(delta, 99)),
                    "max_abs": float(np.max(delta)),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "task022-residual-trace-comparison-v2",
                "rows": rows,
                "mixer_rows": mixer_rows,
            },
            indent=2,
        )
        + "\n"
    )
    for position in (0, 63, 255, 383, 384):
        print(
            "position",
            position,
            [
                (row["layer"], row["relative_l2"])
                for row in rows
                if row["position"] == position
                and row["layer"] in (0, 3, 15, 31, 47, 63)
            ],
        )
        print(
            "mixer",
            position,
            [
                (row["layer"], row["relative_l2"])
                for row in mixer_rows
                if row["position"] == position
            ],
        )


if __name__ == "__main__":
    main()
