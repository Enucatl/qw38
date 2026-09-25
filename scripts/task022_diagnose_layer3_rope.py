#!/usr/bin/env python3
# /// script
# requires-python = "==3.12.*"
# dependencies = ["gguf", "numpy==2.5.3"]
# ///
"""Reconstruct Quartz layer-3 Q/K normalization and RoPE from saved buffers."""

import argparse
import json
from pathlib import Path

import gguf
import gguf.quants
import numpy as np
from task022_compare_layer3_traces import errors, proxy_trace, quartz_trace

POSITIONS = (0, 63, 255, 383, 384)
FREQUENCY = np.power(10000000.0, -np.arange(32, dtype=np.float64) / 32.0).astype(
    np.float32
)


def normalize(raw: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Apply one FP32 head RMS and the converted additive norm weight."""
    x = raw.astype(np.float32)
    scale = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    return x / scale * gamma


def rotate(heads: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Rotate the first 64 coordinates using NeoX pairs."""
    result = heads.copy()
    cosine = np.cos(phase).astype(np.float32)
    sine = np.sin(phase).astype(np.float32)
    left = heads[:, :32]
    right = heads[:, 32:64]
    result[:, :32] = left * cosine - right * sine
    result[:, 32:64] = right * cosine + left * sine
    return result


def inferred_phase(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Estimate rotation angle for every head and pair."""
    a, b = before[:, :32], before[:, 32:64]
    c, d = after[:, :32], after[:, 32:64]
    return np.arctan2(a * d - b * c, a * c + b * d)


def main() -> None:
    """Compare actual Q/K with standard phase and estimate effective phase."""
    parser = argparse.ArgumentParser()
    parser.add_argument("proxy_gguf", type=Path)
    parser.add_argument("proxy_dir", type=Path)
    parser.add_argument("quartz_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reader = gguf.GGUFReader(str(args.proxy_gguf))
    controls = {tensor.name: tensor for tensor in reader.tensors}
    gamma_q = gguf.quants.dequantize(
        controls["blk.3.attn_q_norm.weight"].data,
        controls["blk.3.attn_q_norm.weight"].tensor_type,
    ).reshape(256)
    gamma_k = gguf.quants.dequantize(
        controls["blk.3.attn_k_norm.weight"].data,
        controls["blk.3.attn_k_norm.weight"].tensor_type,
    ).reshape(256)
    proxy_qnorm = proxy_trace(args.proxy_dir, "Qcur_normed-3", 6144)
    report = {"schema": "task022-layer3-rope-diagnosis-v1", "positions": {}}
    for position in POSITIONS:
        qg = quartz_trace(args.quartz_dir, "qg", position, 12288).reshape(24, 2, 256)
        k_raw = quartz_trace(args.quartz_dir, "k_raw", position, 1024).reshape(4, 256)
        q_before = normalize(qg[:, 0], gamma_q)
        k_before = normalize(k_raw, gamma_k)
        q_actual = quartz_trace(args.quartz_dir, "q", position, 6144).reshape(24, 256)
        k_actual = quartz_trace(args.quartz_dir, "kv", position, 2048).reshape(
            2, 4, 256
        )[0]
        expected_phase = position * FREQUENCY
        q_expected = rotate(q_before, expected_phase)
        k_expected = rotate(k_before, expected_phase)
        proxy_before = proxy_qnorm[position].reshape(24, 256)
        q_phase = inferred_phase(q_before, q_actual)
        k_phase = inferred_phase(k_before, k_actual)
        phase_error_q = np.angle(np.exp(1j * (q_phase - expected_phase)))
        phase_error_k = np.angle(np.exp(1j * (k_phase - expected_phase)))
        report["positions"][str(position)] = {
            "q_standard_vs_actual": errors(q_expected, q_actual),
            "k_standard_vs_actual": errors(k_expected, k_actual),
            "quartz_prerope_vs_proxy": errors(q_before, proxy_before),
            "q_phase_abs_median": float(np.median(np.abs(phase_error_q))),
            "k_phase_abs_median": float(np.median(np.abs(phase_error_k))),
            "q_phase_error_first8": np.median(phase_error_q, axis=0)[:8].tolist(),
            "k_phase_error_first8": np.median(phase_error_k, axis=0)[:8].tolist(),
        }
        print(position, report["positions"][str(position)], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
