#!/usr/bin/env python3
"""Summarize TASK-019 raw real-input logs without reading tensor contents."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

FAMILIES = ("mlp_gate", "mlp_up", "mlp_down", "gdn_qkv", "gdn_z",
            "gdn_out", "gdn_a", "gdn_b", "attn_q", "attn_k",
            "attn_v", "attn_out", "lm_head")
M_VALUES = (1, 2, 8, 32, 64, 128, 256, 512, 1024)


def field(line: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}=([^ ]+)", line)
    if not match:
        raise ValueError(f"missing {name}: {line}")
    return match.group(1)


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[max(0, min(len(ordered) - 1,
                              int(len(ordered) * fraction + 0.999999) - 1))]


def summarize_samples(lines: list[str], prefix: str, kernel_name: str) -> dict:
    selected = [line for line in lines if line.startswith(prefix)]
    if len(selected) != 20:
        raise ValueError(f"expected 20 {prefix} samples, got {len(selected)}")
    indices = [int(field(line, "index" if "index=" in line else "sample"))
               for line in selected]
    if indices != list(range(20)):
        raise ValueError(f"unexpected {prefix} sample indices: {indices}")
    kernel = [float(field(line, kernel_name)) for line in selected]
    complete = [float(field(line, "host_complete_ms")) for line in selected]
    return {
        "kernel_median_ms": statistics.median(kernel),
        "kernel_p95_ms": percentile(kernel, 0.95),
        "host_complete_median_ms": statistics.median(complete),
        "host_complete_p95_ms": percentile(complete, 0.95),
        "samples": len(selected),
    }


def read_case(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    descriptor = next(line for line in lines if line.startswith("Descriptor:"))
    base = {name: field(descriptor, name) for name in ("family", "m", "n", "k")}
    base.update({"m": int(base["m"]), "n": int(base["n"]),
                 "k": int(base["k"]), "log": str(path)})
    base["logical_flops"] = 2 * base["m"] * base["n"] * base["k"]
    if sorted(line for line in lines if line.startswith("Result:")) != [
        "Result: bf16_q4_controls PASS", "Result: mxfp4 PASS",
        "Result: nvfp4 PASS"]:
        raise ValueError(f"failed or missing results: {path}")
    rows = []
    for fmt in ("nvfp4", "mxfp4"):
        begin = next(i for i, line in enumerate(lines)
                     if line.startswith("Command:") and f"task019_{fmt}_" in line)
        end = next(i for i in range(begin, len(lines))
                   if lines[i] == f"Result: {fmt} PASS")
        section = lines[begin:end]
        correctness = next(line for line in section
                           if line.startswith("Independent reconstructed contraction:"))
        if " PASS " not in correctness or field(correctness, "mismatches") != "0":
            raise ValueError(f"independent validation failed: {path} {fmt}")
        if not any(line.startswith("Same-weight BF16-activation GEMV: PASS")
                   for line in section):
            raise ValueError(f"GEMV validation failed: {path} {fmt}")
        prepare = next(line for line in section if line.startswith("Weight prepare:"))
        allocation = next(line for line in section if line.startswith("Bytes:"))
        quant = next(line for line in section if line.startswith("Quantization error"))
        common = {**base, "format": fmt,
                  "logical_n": int(field(correctness, "logical_n")),
                  "padded_n": int(field(correctness, "padded_n")),
                  "logical_k": int(field(correctness, "logical_k")),
                  "padded_k": int(field(correctness, "padded_k")),
                  "checked_outputs": int(field(correctness, "checked_outputs")),
                  "workspace_bytes": int(field(correctness, "workspace_bytes")),
                  "workspace_basis": "cutlass_get_workspace_size",
                  "weight_pack_ms": float(field(prepare, "pack_ms")),
                  "weight_upload_ms": float(field(prepare, "upload_ms")),
                  "max_abs_quant_error": float(field(quant, "max_abs")),
                  "operand_b_bytes": int(field(allocation, "operand_b")),
                  "scale_b_bytes": int(field(allocation, "scale_b")),
                  "free_before_bytes": int(field(allocation, "free_before")),
                  "free_after_bytes": int(field(allocation, "free_after")),
                  "output": "BF16"}
        rows.append({**common, "path": "native_gemm", "measured_n": base["n"],
                     "measured_flops": 2 * base["m"] * common["padded_n"] *
                     common["padded_k"],
                     "threadblock_tile_flops_estimate": 2 *
                     ((base["m"] + 127) // 128 * 128) *
                     ((common["padded_n"] + 127) // 128 * 128) *
                     ((common["padded_k"] + 127) // 128 * 128),
                     **summarize_samples(section, "Sample: index=", "kernel_ms")})
        rows.append({**common, "path": "same_weight_bf16_gemv", "measured_n": base["n"],
                     "measured_flops": base["logical_flops"] // base["m"],
                     "workspace_bytes": 0,
                     "workspace_basis": "no_explicit_workspace",
                     **summarize_samples(section, "GEMV Sample:", "kernel_ms")})
    control = lines[next(i for i, line in enumerate(lines)
                         if line.startswith("Command:") and
                         "task019_controls_sm120" in line):]
    q4_prepare = next(line for line in control if line.startswith("Q4G64 weight pack_ms="))
    control_bytes = next(line for line in control if line.startswith("Control bytes:"))
    q4_error = next(line for line in control
                    if line.startswith("Q4G64 quantization error"))
    for fmt, prefix, kernel_name, measured_n in (
        ("bf16", "BF16 sample=", "gemm_ms", base["n"]),
        ("q4g64", "Q4G64 sample=", "unpack_gemm_ms",
         int(field(q4_prepare, "bounded_rows"))),
    ):
        if not any(line.startswith(f"Independent {fmt.upper() if fmt == 'bf16' else 'Q4G64'} contraction: PASS")
                   for line in control):
            raise ValueError(f"control validation failed: {path} {fmt}")
        rows.append({**base, "format": fmt, "path": "library_gemm" if fmt == "bf16"
                     else "bounded_unpack_gemm", "measured_n": measured_n,
                     "measured_flops": 2 * base["m"] * measured_n * base["k"],
                     "logical_n": base["n"], "padded_n": base["n"],
                     "logical_k": base["k"], "padded_k": base["k"],
                     "output": "BF16", "workspace_bytes": 0,
                     "workspace_basis": "no_user_workspace_library_internal_in_free_after",
                     "weight_pack_ms": float(field(q4_prepare, "pack_ms"))
                     if fmt == "q4g64" else 0.0,
                     "operand_b_bytes": int(field(control_bytes, "q4_codes"))
                     if fmt == "q4g64" else int(field(control_bytes, "bf16_weights")),
                     "max_abs_quant_error": float(field(q4_error, "max_abs"))
                     if fmt == "q4g64" else 0.0,
                     "scale_b_bytes": int(field(control_bytes, "q4_scales"))
                     if fmt == "q4g64" else 0,
                     **summarize_samples(control, prefix, kernel_name)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=Path(".cache/task019/real-matrix"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/implementation/task019-real-matrix.csv"))
    args = parser.parse_args()
    paths = sorted(args.logs.glob("core/*.log")) + sorted(args.logs.glob("head/*.log"))
    expected = {(family, m) for family in FAMILIES for m in M_VALUES}
    present = {(path.stem.rsplit("-m", 1)[0], int(path.stem.rsplit("-m", 1)[1]))
               for path in paths}
    if len(paths) != len(expected) or present != expected:
        raise ValueError(f"case matrix differs: missing={expected - present} "
                         f"extra={present - expected}")
    rows = [row for path in paths for row in read_case(path)]
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"cases": len(paths), "rows": len(rows),
                      "samples": sum(row["samples"] for row in rows),
                      "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
