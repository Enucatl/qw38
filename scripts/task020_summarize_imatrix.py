#!/usr/bin/env python3
"""Summarize llama.cpp input-activation second moments by projection family."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import gguf
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("imatrix", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reader = gguf.GGUFReader(str(args.imatrix))
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    groups = defaultdict(list)
    selected = {}
    for name, tensor in tensors.items():
        match = re.fullmatch(r"blk\.(\d+)\.([^.]+)\.weight\.in_sum2", name)
        if not match:
            continue
        layer, family = int(match[1]), match[2]
        count = float(tensors[name.removesuffix("in_sum2") + "counts"].data[0])
        if count <= 0 or not np.isfinite(tensor.data).all():
            raise ValueError(f"invalid calibration statistic: {name}")
        rms = np.sqrt(tensor.data.astype(np.float64) / count)
        groups[family].append((layer, rms, count))
        if layer in (0, 31, 63):
            selected[f"blk.{layer}.{family}"] = {
                "count": count,
                "input_rms_median": float(np.median(rms)),
                "input_rms_p99": float(np.percentile(rms, 99)),
                "input_rms_max": float(np.max(rms)),
            }
    summary = {}
    for family, entries in sorted(groups.items()):
        all_rms = np.concatenate([item[1] for item in entries])
        summary[family] = {
            "layers": len(entries),
            "count_min": min(item[2] for item in entries),
            "count_max": max(item[2] for item in entries),
            "input_rms_median": float(np.median(all_rms)),
            "input_rms_p99": float(np.percentile(all_rms, 99)),
            "input_rms_max": float(np.max(all_rms)),
        }
    result = {
        "schema": "qw38-task020-imatrix-summary-v1",
        "source": str(args.imatrix),
        "statistic": "sqrt(quantized-model input-sum-of-squares / count) per input channel",
        "families": summary,
        "selected_layers": selected,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {name: data["input_rms_p99"] for name, data in summary.items()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
