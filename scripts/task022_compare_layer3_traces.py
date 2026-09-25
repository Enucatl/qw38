#!/usr/bin/env python3
# /// script
# requires-python = "==3.12.*"
# dependencies = ["numpy==2.5.3"]
# ///
"""Compare layer-3 attention boundaries and check both attention scans."""

import argparse
import json
from pathlib import Path

import numpy as np

POSITIONS = (0, 63, 255, 383, 384)
HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
HIDDEN = 5120


def proxy_trace(directory: Path, name: str, width: int) -> np.ndarray:
    """Read a complete 384-token prefill and one decode tensor."""
    rows = []
    for phase, count in (("prefill", 384), ("decode", 1)):
        path = directory / f"attention.{name}.{phase}.bin"
        meta = path.with_suffix(".bin.txt").read_text().strip().split("\t")
        if meta[:3] != [name, phase, "f32"] or int(meta[-1]) != count * width * 4:
            raise ValueError(f"unexpected proxy tensor metadata: {path}")
        data = np.fromfile(path, dtype="<f4")
        if data.size != count * width:
            raise ValueError(f"short proxy tensor: {path}")
        rows.append(data.reshape(count, width))
    result = np.concatenate(rows)
    if not np.isfinite(result).all():
        raise ValueError(f"nonfinite proxy tensor: {name}")
    return result


def quartz_trace(directory: Path, name: str, position: int, width: int) -> np.ndarray:
    """Read one captured Quartz buffer in its native dtype."""
    dtype = "<f4" if name in ("input", "output") else "<u2"
    path = directory / f"attention.{name}.pos-{position}.bin"
    data = np.fromfile(path, dtype=dtype)
    if data.size != width:
        raise ValueError(f"unexpected Quartz tensor size: {path}")
    if dtype == "<u2":
        data = (data.astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"nonfinite Quartz tensor: {path}")
    return data


def errors(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Report vector difference relative to the reference norm."""
    a = actual.astype(np.float64).ravel()
    b = reference.astype(np.float64).ravel()
    delta = np.abs(a - b)
    return {
        "relative_l2": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
        "p99_abs": float(np.percentile(delta, 99)),
        "max_abs": float(delta.max()),
    }


def rope_neox(heads: np.ndarray) -> np.ndarray:
    """Apply the model's first-64-coordinate NeoX RoPE to head vectors."""
    result = heads.copy()
    frequency = np.power(10000000.0, -np.arange(32, dtype=np.float64) / 32.0)
    phase = np.arange(385, dtype=np.float64)[:, None] * frequency[None, :]
    cosine = np.cos(phase).astype(np.float32)[:, None, :]
    sine = np.sin(phase).astype(np.float32)[:, None, :]
    left = result[:, :, :32].copy()
    right = result[:, :, 32:64].copy()
    result[:, :, :32] = left * cosine - right * sine
    result[:, :, 32:64] = right * cosine + left * sine
    return result


def attention_values(
    query: np.ndarray, keys: np.ndarray, values: np.ndarray, position: int
) -> np.ndarray:
    """Compute causal GQA values with a stable FP32 softmax."""
    heads = np.arange(HEADS) // 6
    k = keys[: position + 1, heads].transpose(1, 0, 2)
    v = values[: position + 1, heads].transpose(1, 0, 2)
    scores = np.einsum("hd,hkd->hk", query, k, dtype=np.float32) / 16.0
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("hk,hkd->hd", weights, v, dtype=np.float32)


def bf16(data: np.ndarray) -> np.ndarray:
    """Round finite FP32 data to BF16 with ties to even."""
    bits = data.astype(np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.view(np.float32)


def main() -> None:
    """Write checkpoint boundary and scan replay comparisons."""
    parser = argparse.ArgumentParser()
    parser.add_argument("proxy_dir", type=Path)
    parser.add_argument("quartz_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    names = {
        "l_out-2": HIDDEN,
        "attn_norm-3": HIDDEN,
        "Qcur_full-3": 12288,
        "Kcur-3": 1024,
        "Vcur-3": 1024,
        "Kcur_normed-3": 1024,
        "Qcur_normed-3": 6144,
        "Qcur-3": 6144,
        "attn_pregate-3": 6144,
        "gate_sigmoid-3": 6144,
        "attn_gated-3": 6144,
        "attn_output-3": HIDDEN,
    }
    proxy = {
        name: proxy_trace(args.proxy_dir, name, width) for name, width in names.items()
    }
    quartz_kv = np.stack(
        [
            quartz_trace(args.quartz_dir, "kv", p, 2048).reshape(2, 4, 256)
            for p in range(385)
        ]
    )
    proxy_k = rope_neox(proxy["Kcur_normed-3"].reshape(385, 4, 256))
    proxy_q_rotated = rope_neox(proxy["Qcur_normed-3"].reshape(385, 24, 256))
    # The pinned llama.cpp context defaults to F16 K/V; record scan-model error
    # against the actual graph before interpreting arithmetic differences.
    proxy_k = proxy_k.astype(np.float16).astype(np.float32)
    proxy_v = proxy["Vcur-3"].reshape(385, 4, 256).astype(np.float16).astype(np.float32)
    report = {
        "schema": "task022-layer3-boundaries-v2",
        "positions": {},
        "scan_replay": {},
        "rope_check": {},
    }
    mapping = (
        ("input", "l_out-2", HIDDEN),
        ("normalized", "attn_norm-3", HIDDEN),
        ("qg", "Qcur_full-3", 12288),
        ("k_raw", "Kcur-3", 1024),
        ("v_raw", "Vcur-3", 1024),
        ("q", "Qcur-3", 6144),
        ("gated", "attn_gated-3", 6144),
    )
    for position in POSITIONS:
        row = {}
        for quartz_name, proxy_name, width in mapping:
            row[quartz_name] = errors(
                quartz_trace(args.quartz_dir, quartz_name, position, width),
                proxy[proxy_name][position],
            )
        gguf_branch = proxy["attn_output-3"][position]
        quartz_input = quartz_trace(args.quartz_dir, "input", position, HIDDEN)
        quartz_output = quartz_trace(args.quartz_dir, "output", position, HIDDEN)
        row["output_branch"] = errors(quartz_output - quartz_input, gguf_branch)
        row["output_residual"] = errors(
            quartz_output, gguf_branch + proxy["l_out-2"][position]
        )
        report["positions"][str(position)] = row
        q = quartz_trace(args.quartz_dir, "q", position, 6144).reshape(HEADS, HEAD_DIM)
        proxy_q = proxy["Qcur-3"][position].reshape(HEADS, HEAD_DIM)
        report["rope_check"][str(position)] = {
            "proxy_standard_neox_vs_graph": errors(proxy_q_rotated[position], proxy_q),
            "quartz_vs_proxy_rotated_first64": errors(
                q[:, :64], proxy_q_rotated[position, :, :64]
            ),
            "quartz_vs_proxy_rotated_rest": errors(
                q[:, 64:], proxy_q_rotated[position, :, 64:]
            ),
        }
        g = quartz_trace(args.quartz_dir, "g", position, 6144).reshape(HEADS, HEAD_DIM)
        calc = attention_values(q, quartz_kv[:, 0], quartz_kv[:, 1], position)
        calc = bf16(calc / (1.0 + np.exp(-g)))
        actual = quartz_trace(args.quartz_dir, "gated", position, 6144).reshape(
            HEADS, HEAD_DIM
        )
        proxy_values = attention_values(proxy_q, proxy_k, proxy_v, position)
        report["scan_replay"][str(position)] = {
            "quartz_math_vs_kernel_gated": errors(calc, actual),
            "proxy_math_vs_graph_pregate": errors(
                proxy_values, proxy["attn_pregate-3"][position]
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for position in POSITIONS:
        row = report["positions"][str(position)]
        print(
            position,
            {name: round(value["relative_l2"], 6) for name, value in row.items()},
        )
        print("scan", position, report["scan_replay"][str(position)], flush=True)
        print("rope", position, report["rope_check"][str(position)], flush=True)


if __name__ == "__main__":
    main()
