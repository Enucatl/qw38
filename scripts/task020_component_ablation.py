#!/usr/bin/env python3
"""Measure FP4 weight and activation perturbations on traced full projections."""

import argparse
import json
import re
from pathlib import Path

import gguf
import gguf.quants
import numpy as np
import torch
from task019_real_inputs import read_row, tensor_spec

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
SOURCE_FAMILY = {
    "attn_qkv": "linear_attn.in_proj_qkv",
    "ssm_out": "linear_attn.out_proj",
    "attn_q": "self_attn.q_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_down": "mlp.down_proj",
}


def nvfp4_activation(data: np.ndarray) -> np.ndarray:
    """Reconstruct per-row/per-16 blocks with pinned gguf UE4M3 scale rounding."""
    if data.ndim != 2 or data.shape[1] % 16 or not np.isfinite(data).all():
        raise ValueError("activation needs finite rows and a multiple-of-16 width")
    blocks = data.reshape(data.shape[0], -1, 16)
    peak = np.max(np.abs(blocks), axis=2)
    # gguf's scale conversion has distinct midpoint and exponent-15 behavior
    # from the TASK-019 native-format reference. This is a GGUF diagnostic.
    codes = gguf.quants.NVFP4.fp32_to_ue4m3(peak / 6.0)
    codes = np.where(peak > 0, np.maximum(codes, 1), codes).astype(np.uint8)
    scale = 2.0 * gguf.quants.NVFP4.ue4m3_to_fp32(codes)
    normalized = np.divide(
        blocks, scale[..., None], out=np.zeros_like(blocks), where=scale[..., None] > 0
    )
    distance = np.abs(np.abs(normalized)[..., None] - E2M1)
    best = distance.min(axis=-1, keepdims=True)
    index = np.where(distance == best, np.arange(8) % 2, 2).argmin(axis=-1)
    reconstructed = np.copysign(E2M1[index] * scale[..., None], blocks)
    return reconstructed.reshape(data.shape)


def relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm((actual - reference).ravel()) / np.linalg.norm(reference.ravel())
    )


def errors(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = np.abs(actual - reference)
    return {
        "relative_l2": relative_l2(actual, reference),
        "p99_abs": float(np.percentile(delta, 99)),
        "max_abs": float(delta.max()),
    }


def source_name(gguf_name: str) -> str:
    match = re.fullmatch(r"blk\.(\d+)\.([^.]+)\.weight", gguf_name)
    if not match or match[2] not in SOURCE_FAMILY:
        raise ValueError(f"unexpected trace tensor {gguf_name}")
    return f"model.language_model.layers.{match[1]}.{SOURCE_FAMILY[match[2]]}.weight"


def verify_source(name: str, full: gguf.ReaderTensor, index: dict) -> None:
    """Check GGUF BF16 rows against HF BF16, including llama.cpp's V reordering."""
    source = tensor_spec(source_name(name), index)
    first = np.frombuffer(read_row(source, 0), dtype="<u2")
    if ".ssm_out." in name:
        first = first.reshape(16, 3, 128).transpose(1, 0, 2).reshape(-1)
    if full.data[0].tobytes() != first.tobytes():
        raise ValueError(
            f"BF16 GGUF differs from transformed safetensors source: {name}"
        )
    # The second 128-row V head in tiled order comes from source V head 3.
    if ".attn_qkv." in name and full.data[4096 + 128].tobytes() != read_row(
        source, 4096 + 384
    ):
        raise ValueError(f"QKV V head permutation differs: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bf16_gguf", type=Path)
    parser.add_argument("fp4_gguf", type=Path)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(20)
    bf16 = {
        tensor.name: tensor for tensor in gguf.GGUFReader(str(args.bf16_gguf)).tensors
    }
    fp4 = {
        tensor.name: tensor for tensor in gguf.GGUFReader(str(args.fp4_gguf)).tensors
    }
    index = json.loads(
        Path(
            ".cache/authorities/qwen3.8-27b-transformers/model.safetensors.index.json"
        ).read_text()
    )
    names = sorted(
        {
            path.name.split(".prefill.bin")[0]
            for path in args.trace_dir.glob("*.prefill.bin")
            # The shared trace directory also feeds task020_head_ablation.py.
            if path.name != "output.weight.prefill.bin"
        }
    )
    results = []
    for name in names:
        full = bf16[name]
        candidate = fp4[name]
        if candidate.tensor_type != gguf.GGMLQuantizationType.NVFP4:
            raise ValueError(f"not NVFP4: {name}")
        verify_source(name, full, index)
        w_ref = gguf.quants.dequantize(full.data, full.tensor_type)
        w_fp4 = gguf.quants.dequantize(candidate.data, candidate.tensor_type)
        if w_ref.shape != w_fp4.shape:
            raise ValueError(f"weight geometry differs: {name}")
        weight_error = relative_l2(w_fp4, w_ref)
        weight_ref = torch.from_numpy(w_ref)
        weight_fp4 = torch.from_numpy(w_fp4)
        for phase in ("prefill", "decode"):
            path = args.trace_dir / f"{name}.{phase}.bin"
            meta = (
                path.with_suffix(path.suffix + ".txt").read_text().strip().split("\t")
            )
            if meta[:3] != [name, phase, "f32"] or int(meta[3]) != w_ref.shape[1]:
                raise ValueError(f"trace geometry differs: {path}")
            rows = int(meta[4])
            data = np.fromfile(path, dtype="<f4").reshape(rows, w_ref.shape[1])
            if phase == "prefill":
                largest = np.argsort(np.max(np.abs(data), axis=1))[-4:]
                data = data[
                    np.unique(np.concatenate((np.arange(0, rows, 48), largest)))
                ].copy()
            if not np.isfinite(data).all():
                raise ValueError(f"nonfinite trace: {path}")
            quantized = nvfp4_activation(data)
            if not np.array_equal(
                quantized, np.vstack([nvfp4_activation(row[None, :]) for row in data])
            ):
                raise ValueError(
                    f"activation scale changes with chunk composition: {path}"
                )
            a_ref = torch.from_numpy(data)
            a_fp4 = torch.from_numpy(quantized)
            with torch.no_grad():
                y_ref = (a_ref @ weight_ref.T).numpy()
                y_weight = (a_ref @ weight_fp4.T).numpy()
                y_activation = (a_fp4 @ weight_ref.T).numpy()
                y_both = (a_fp4 @ weight_fp4.T).numpy()
            result = {
                "tensor": name,
                "phase": phase,
                "rows": len(data),
                "output_width": w_ref.shape[0],
                "input_width": w_ref.shape[1],
                "weight_relative_l2": weight_error,
                "activation": errors(quantized, data),
                "weight_only": errors(y_weight, y_ref),
                "activation_only": errors(y_activation, y_ref),
                "combined": errors(y_both, y_ref),
                "input_max_abs": float(np.max(np.abs(data))),
            }
            results.append(result)
            print(
                f"{name} {phase} rows={len(data)} combined_relative_l2={result['combined']['relative_l2']:.6f}",
                flush=True,
            )
        del w_ref, w_fp4, weight_ref, weight_fp4
    args.output.write_text(
        json.dumps(
            {"schema": "qw38-task020-component-ablation-v1", "results": results},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
