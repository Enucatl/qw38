"""Freeze diagnosed scalar-authority evidence and immutable metric gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any


def nice_ceiling(value: float) -> float:
    if value == 0.0:
        return 0.0
    exponent = math.floor(math.log10(value))
    unit = 10.0 ** (exponent - 1)
    return math.ceil(value * 1.1 / unit) * unit


def cosine_floor(value: float) -> float:
    if value == 1.0:
        return 1.0
    expanded_distance = (1.0 - value) * 1.1
    return math.floor((1.0 - expanded_distance) * 1_000_000.0) / 1_000_000.0


def logits(path: Path, width: int = 248320) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if len(raw) != width * 2 * 4:
        raise ValueError(f"unexpected logit byte count: {path}")
    values = struct.unpack(f"<{width * 2}f", raw)
    result: list[dict[str, Any]] = []
    for position in range(2):
        row = values[position * width : (position + 1) * width]
        top = sorted(range(width), key=row.__getitem__, reverse=True)[:2]
        result.append(
            {
                "position": position,
                "greedy_token": top[0],
                "runner_up_token": top[1],
                "greedy_logit": row[top[0]],
                "runner_up_logit": row[top[1]],
                "margin": row[top[0]] - row[top[1]],
            }
        )
    return result


def tap_key(row: dict[str, Any]) -> str:
    layer = row["layer"]
    return (
        f"global.{row['boundary']}"
        if layer is None
        else f"layer.{layer}.{row['boundary']}"
    )


def freeze(comparison: dict[str, Any]) -> dict[str, Any]:
    authorities: dict[str, dict[str, Any]] = {}
    for side in ("official_vs_quartz", "llama_vs_quartz"):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in comparison["comparisons"]:
            if side in row:
                grouped.setdefault(tap_key(row), []).append(row[side])
        taps: dict[str, Any] = {}
        for key, samples in sorted(grouped.items()):
            maximum_absolute = max(item["maximum_absolute_error"] for item in samples)
            maximum_rms = max(item["root_mean_square_error"] for item in samples)
            minimum_cosine = min(item["cosine_similarity"] for item in samples)
            taps[key] = {
                "sample_count": len(samples),
                "observed": {
                    "maximum_absolute_error": maximum_absolute,
                    "maximum_root_mean_square_error": maximum_rms,
                    "minimum_cosine_similarity": minimum_cosine,
                    "maximum_relative_error": max(
                        item["maximum_relative_error"] for item in samples
                    ),
                },
                "admission": {
                    "maximum_absolute_error": nice_ceiling(maximum_absolute),
                    "maximum_root_mean_square_error": nice_ceiling(maximum_rms),
                    "minimum_cosine_similarity": cosine_floor(minimum_cosine),
                    "nan_count": 0,
                    "positive_infinity_count": 0,
                    "negative_infinity_count": 0,
                },
            }
        authorities[side] = {"taps": taps}
    return {
        "schema_version": 1,
        "status": "frozen_before_cuda_optimization",
        "calibration": {
            "input_tokens": comparison["input_tokens"],
            "positions": [0, 1],
            "layers": [0, 3, 7, 62, 63],
            "absolute_and_rms_margin": "1.10x then round upward to two significant digits",
            "cosine_margin": "1.10x observed distance from 1 then floor to six decimals",
            "relative_error": "reported but not gated because near-zero denominators make maxima unstable",
            "freezer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "admission_rule": (
            "every exact identity/shape must match; every sample must satisfy all "
            "absolute, RMS, cosine, and finite-count gates; greedy tokens must match "
            "unless listed in greedy_near_tie_exceptions"
        ),
        "greedy_near_tie_exceptions": [],
        "authorities": authorities,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--official-logits", type=Path, required=True)
    parser.add_argument("--quartz-logits", type=Path, required=True)
    parser.add_argument("--llama-logits", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--tolerances", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    comparison["admission_status"] = "admitted_by_frozen_scalar_tolerances"
    comparison["greedy"] = {
        "official": logits(args.official_logits),
        "quartz": logits(args.quartz_logits),
        "llama_cpp": logits(args.llama_logits),
        "near_tie_exceptions": [],
    }
    args.fixture.write_text(
        json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.tolerances.write_text(
        json.dumps(freeze(comparison), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
