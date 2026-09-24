#!/usr/bin/env python3
"""Measure selected QW38 Q4G64/Q8G32 perturbations on traced model inputs."""

import argparse
import json
from pathlib import Path

import gguf
import gguf.quants
import numpy as np
import torch
from task019_real_inputs import read_row, tensor_spec
from task020_component_ablation import errors, verify_source


def quantize_grouped(weights: np.ndarray, group: int) -> np.ndarray:
    """Reconstruct existing QW38 logical groups and BF16 contraction operands."""
    if weights.ndim != 2 or weights.shape[1] % group or group not in (32, 64):
        raise ValueError("invalid grouped matrix geometry")
    if not np.isfinite(weights).all():
        raise ValueError("nonfinite source weight")
    qmax = 127 if group == 32 else 7
    blocks = weights.reshape(weights.shape[0], -1, group)
    peak = np.max(np.abs(blocks), axis=2)
    need = peak / qmax
    if np.any(need > np.finfo(np.float16).max):
        raise ValueError("scale exceeds finite FP16")
    with np.errstate(under="ignore"):
        stored = need.astype(np.float16)
    stored = np.where(
        stored.astype(np.float32) < need,
        np.nextafter(stored, np.float16(np.inf)),
        stored,
    )
    stored = np.where(peak > 0, np.maximum(stored, np.float16(2**-14)), 0)
    scale = stored.astype(np.float32)
    values = np.divide(
        blocks, scale[..., None], out=np.zeros_like(blocks), where=scale[..., None] > 0
    )
    codes = np.clip(np.rint(values), -qmax, qmax)
    reconstructed = (codes * scale[..., None]).reshape(weights.shape)
    return torch.from_numpy(reconstructed).to(torch.bfloat16).float().numpy()


def trace_input(path: Path, name: str, phase: str, width: int) -> np.ndarray:
    """Read a traced node input and retain varied prompt rows and outliers."""
    meta = path.with_suffix(path.suffix + ".txt").read_text().strip().split("\t")
    if meta[:3] != [name, phase, "f32"] or int(meta[3]) != width:
        raise ValueError(f"trace geometry differs: {path}")
    rows = int(meta[4])
    data = np.fromfile(path, dtype="<f4").reshape(rows, width)
    if phase == "prefill" and rows > 1:
        largest = np.argsort(np.max(np.abs(data), axis=1))[-4:]
        data = data[np.unique(np.concatenate((np.arange(0, rows, 48), largest)))].copy()
    if not np.isfinite(data).all():
        raise ValueError(f"nonfinite trace: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bf16_gguf", type=Path)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(20)
    bf16 = {
        tensor.name: tensor for tensor in gguf.GGUFReader(str(args.bf16_gguf)).tensors
    }
    index = json.loads(
        Path(
            ".cache/authorities/qwen3.8-27b-transformers/model.safetensors.index.json"
        ).read_text()
    )
    names = sorted(
        {p.name.split(".prefill.bin")[0] for p in args.trace_dir.glob("*.prefill.bin")}
    )
    results = []
    for name in names:
        full = bf16[name]
        if name == "output.weight":
            if full.data[0].tobytes() != read_row(
                tensor_spec("lm_head.weight", index), 0
            ):
                raise ValueError("BF16 head differs from safetensors source")
            group = 32
        else:
            verify_source(name, full, index)
            group = 64 if ".ffn_" in name else 32
        k, n = map(int, full.shape)
        phases = {}
        for phase in ("prefill", "decode"):
            data = trace_input(args.trace_dir / f"{name}.{phase}.bin", name, phase, k)
            bf16_input = torch.from_numpy(data).to(torch.bfloat16).float().numpy()
            phases[phase] = {
                "input": data,
                "bf16_input": bf16_input,
                "reference": np.empty((len(data), n), dtype=np.float32),
                "weight_only": np.empty((len(data), n), dtype=np.float32),
                "activation_only": np.empty((len(data), n), dtype=np.float32),
                "combined": np.empty((len(data), n), dtype=np.float32),
            }
        weight_delta = weight_reference = 0.0
        for start in range(0, n, 512):
            stop = min(start + 512, n)
            reference = gguf.quants.dequantize(full.data[start:stop], full.tensor_type)
            quantized = quantize_grouped(reference, group)
            weight_delta += float(
                np.sum((quantized.astype(np.float64) - reference) ** 2)
            )
            weight_reference += float(np.sum(reference.astype(np.float64) ** 2))
            wr, wq = torch.from_numpy(reference), torch.from_numpy(quantized)
            with torch.no_grad():
                for phase in phases.values():
                    a = torch.from_numpy(phase["input"])
                    aq = torch.from_numpy(phase["bf16_input"])
                    phase["reference"][:, start:stop] = (a @ wr.T).numpy()
                    phase["weight_only"][:, start:stop] = (a @ wq.T).numpy()
                    phase["activation_only"][:, start:stop] = (aq @ wr.T).numpy()
                    phase["combined"][:, start:stop] = (aq @ wq.T).numpy()
            if name == "output.weight" and start % 32768 == 0:
                print(f"head_rows={stop}/{n}", flush=True)
        for phase_name, phase in phases.items():
            results.append(
                {
                    "tensor": name,
                    "phase": phase_name,
                    "group": group,
                    "input_width": k,
                    "output_width": n,
                    "rows": len(phase["input"]),
                    "input_max_abs": float(np.max(np.abs(phase["input"]))),
                    "weight_relative_l2": (weight_delta / weight_reference) ** 0.5,
                    "activation": errors(phase["bf16_input"], phase["input"]),
                    "weight_only": errors(phase["weight_only"], phase["reference"]),
                    "activation_only": errors(
                        phase["activation_only"], phase["reference"]
                    ),
                    "combined": errors(phase["combined"], phase["reference"]),
                }
            )
            print(name, phase_name, results[-1]["combined"]["relative_l2"], flush=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "qw38-task020-selected-component-ablation-v1",
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
