#!/usr/bin/env python3
"""Full-head FP4 component error using bounded output-row chunks."""

import argparse
import json
from pathlib import Path

import gguf
import gguf.quants
import numpy as np
import torch
from task019_real_inputs import read_row, tensor_spec
from task020_component_ablation import errors, nvfp4_activation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bf16_gguf", type=Path)
    parser.add_argument("fp4_head_gguf", type=Path)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(20)
    bf16 = next(
        t
        for t in gguf.GGUFReader(str(args.bf16_gguf)).tensors
        if t.name == "output.weight"
    )
    fp4 = next(
        t
        for t in gguf.GGUFReader(str(args.fp4_head_gguf)).tensors
        if t.name == "output.weight"
    )
    if (
        fp4.tensor_type != gguf.GGMLQuantizationType.NVFP4
        or bf16.shape.tolist() != fp4.shape.tolist()
    ):
        raise ValueError("expected matching BF16 and NVFP4 full heads")
    index = json.loads(
        Path(
            ".cache/authorities/qwen3.8-27b-transformers/model.safetensors.index.json"
        ).read_text()
    )
    if bf16.data[0].tobytes() != read_row(tensor_spec("lm_head.weight", index), 0):
        raise ValueError("BF16 head differs from safetensors source")
    phases = {}
    for phase in ("prefill", "decode"):
        path = args.trace_dir / f"output.weight.{phase}.bin"
        meta = path.with_suffix(path.suffix + ".txt").read_text().strip().split("\t")
        if meta != ["output.weight", phase, "f32", "5120", "1", "20480"]:
            raise ValueError("head trace geometry differs")
        original = np.fromfile(path, dtype="<f4").reshape(1, 5120)
        phases[phase] = (
            original,
            nvfp4_activation(original),
            {
                key: []
                for key in ("reference", "weight_only", "activation_only", "combined")
            },
        )
    weight_error = weight_reference = 0.0
    for start in range(0, bf16.data.shape[0], 4096):
        stop = min(start + 4096, bf16.data.shape[0])
        w_ref = gguf.quants.dequantize(bf16.data[start:stop], bf16.tensor_type)
        w_fp4 = gguf.quants.dequantize(fp4.data[start:stop], fp4.tensor_type)
        weight_error += float(np.sum((w_fp4.astype(np.float64) - w_ref) ** 2))
        weight_reference += float(np.sum(w_ref.astype(np.float64) ** 2))
        wr, wf = torch.from_numpy(w_ref), torch.from_numpy(w_fp4)
        with torch.no_grad():
            for original, quantized, output in phases.values():
                a, aq = torch.from_numpy(original), torch.from_numpy(quantized)
                output["reference"].append((a @ wr.T).numpy().copy())
                output["weight_only"].append((a @ wf.T).numpy().copy())
                output["activation_only"].append((aq @ wr.T).numpy().copy())
                output["combined"].append((aq @ wf.T).numpy().copy())
        if start % 32768 == 0:
            print(f"head_rows={stop}/{bf16.data.shape[0]}", flush=True)
    result = {
        "schema": "qw38-task020-full-head-ablation-v1",
        "weight_relative_l2": (weight_error / weight_reference) ** 0.5,
        "phases": {},
    }
    for phase, (original, quantized, output) in phases.items():
        merged = {key: np.concatenate(parts, axis=1) for key, parts in output.items()}
        result["phases"][phase] = {
            "activation": errors(quantized, original),
            "weight_only": errors(merged["weight_only"], merged["reference"]),
            "activation_only": errors(merged["activation_only"], merged["reference"]),
            "combined": errors(merged["combined"], merged["reference"]),
            "input_max_abs": float(np.max(np.abs(original))),
            "output_width": merged["reference"].shape[1],
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                phase: data["combined"]["relative_l2"]
                for phase, data in result["phases"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
