#!/usr/bin/env python3
# /// script
# requires-python = "==3.12.*"
# dependencies = ["gguf", "numpy==2.5.3", "torch==2.9.0"]
# ///
"""Compare GGUF and Quartz weight/operand recipes on saved GDN inputs."""

import argparse
import json
from pathlib import Path

import gguf
import gguf.quants
import numpy as np
import torch
from task020_component_ablation import errors
from task020_selected_ablation import quantize_grouped, trace_input


def project(
    source: gguf.ReaderTensor,
    proxy: gguf.ReaderTensor,
    inputs: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Evaluate both decoded weights and both operand rounding choices."""
    k, n = map(int, source.shape)
    if tuple(map(int, proxy.shape)) != (k, n):
        raise ValueError("source and proxy shapes differ")
    result: dict[str, dict[str, np.ndarray]] = {}
    for phase, x in inputs.items():
        result[phase] = {
            key: np.empty((len(x), n), dtype=np.float32)
            for key in ("source", "proxy", "quartz_full", "quartz_bf16", "proxy_bf16")
        }
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        original = gguf.quants.dequantize(source.data[start:stop], source.tensor_type)
        proxy_weight = gguf.quants.dequantize(proxy.data[start:stop], proxy.tensor_type)
        quartz_weight = quantize_grouped(original, 32)
        w_source = torch.from_numpy(original)
        w_proxy = torch.from_numpy(proxy_weight)
        w_quartz = torch.from_numpy(quartz_weight)
        with torch.no_grad():
            for phase, x in inputs.items():
                x_full = torch.from_numpy(x)
                x_bf16 = x_full.to(torch.bfloat16).float()
                output = result[phase]
                output["source"][:, start:stop] = (x_full @ w_source.T).numpy()
                output["proxy"][:, start:stop] = (x_full @ w_proxy.T).numpy()
                output["quartz_full"][:, start:stop] = (x_full @ w_quartz.T).numpy()
                output["quartz_bf16"][:, start:stop] = (x_bf16 @ w_quartz.T).numpy()
                rounded_proxy = w_proxy.to(torch.bfloat16).float()
                output["proxy_bf16"][:, start:stop] = (x_bf16 @ rounded_proxy.T).numpy()
    return result


def main() -> None:
    """Read frozen traces and write a compact weight/operand attribution report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source_gguf", type=Path)
    parser.add_argument("proxy_gguf", type=Path)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(20)
    source = {t.name: t for t in gguf.GGUFReader(str(args.source_gguf)).tensors}
    proxy = {t.name: t for t in gguf.GGUFReader(str(args.proxy_gguf)).tensors}
    results = []
    for name in ("blk.0.attn_qkv.weight", "blk.0.ssm_out.weight"):
        k, _ = map(int, source[name].shape)
        inputs = {
            phase: trace_input(args.trace_dir / f"{name}.{phase}.bin", name, phase, k)
            for phase in ("prefill", "decode")
        }
        outputs = project(source[name], proxy[name], inputs)
        for phase, output in outputs.items():
            base = output["source"]
            results.append(
                {
                    "tensor": name,
                    "phase": phase,
                    "rows": len(inputs[phase]),
                    "input_max_abs": float(np.max(np.abs(inputs[phase]))),
                    "proxy_weight_vs_source": errors(output["proxy"], base),
                    "quartz_weight_vs_source": errors(output["quartz_full"], base),
                    "quartz_operand_vs_weight": errors(
                        output["quartz_bf16"], output["quartz_full"]
                    ),
                    "quartz_total_vs_source": errors(output["quartz_bf16"], base),
                    "proxy_with_quartz_operands_vs_source": errors(
                        output["proxy_bf16"], base
                    ),
                    "quartz_vs_proxy": errors(output["quartz_bf16"], output["proxy"]),
                }
            )
            print(name, phase, results[-1]["quartz_vs_proxy"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"schema": "task022-gdn-projection-paths-v1", "results": results}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
