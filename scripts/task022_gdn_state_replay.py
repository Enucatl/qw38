#!/usr/bin/env python3
# /// script
# requires-python = "==3.12.*"
# dependencies = ["gguf", "numpy==2.5.3", "torch==2.9.0"]
# ///
"""Replay layer-0 GDN on one saved GGUF input stream with two weight recipes."""

import argparse
import json
from pathlib import Path

import gguf
import gguf.quants
import numpy as np
import torch
from task020_component_ablation import errors
from task020_selected_ablation import quantize_grouped

HIDDEN = 5120
QKV = 10240
VALUE = 6144
HEADS = 48
HEAD_DIM = 128


def load_trace(path: Path, name: str, phase: str, width: int) -> np.ndarray:
    """Read every row of a traced GGUF graph input."""
    meta = path.with_suffix(path.suffix + ".txt").read_text().strip().split("\t")
    if meta[:3] != [name, phase, "f32"] or int(meta[3]) != width:
        raise ValueError(f"trace metadata differs: {path}")
    result = np.fromfile(path, dtype="<f4").reshape(int(meta[4]), width)
    if not np.isfinite(result).all():
        raise ValueError(f"nonfinite trace: {path}")
    return result


def dequant(tensor: gguf.ReaderTensor) -> torch.Tensor:
    """Decode one GGUF tensor into an FP32 CUDA tensor."""
    return torch.from_numpy(
        gguf.quants.dequantize(tensor.data, tensor.tensor_type).copy()
    ).cuda()


def projection(
    x: torch.Tensor,
    source: gguf.ReaderTensor,
    proxy: gguf.ReaderTensor,
    quartz: bool,
    bf16_store: bool,
) -> torch.Tensor:
    """Apply one matrix with either GGUF Q8_0 or Quartz Q8G32 operands."""
    if quartz:
        pieces = []
        for start in range(0, len(source.data), 256):
            raw = gguf.quants.dequantize(
                source.data[start : start + 256], source.tensor_type
            )
            pieces.append(torch.from_numpy(quantize_grouped(raw, 32).copy()))
        weight = torch.cat(pieces).cuda()
        x = x.to(torch.bfloat16).float()
    else:
        weight = dequant(proxy)
    output = x @ weight.T
    if bf16_store:
        output = output.to(torch.bfloat16).float()
    return output


def replay(
    qkv: torch.Tensor,
    z: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv: torch.Tensor,
    decay: torch.Tensor,
    dt: torch.Tensor,
    gamma: torch.Tensor,
    bf16_stores: bool,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Step the documented convolution and GDN state equations."""
    state = torch.zeros((HEADS, HEAD_DIM, HEAD_DIM), device="cuda")
    history = torch.zeros((3, QKV), device="cuda")
    outputs = []
    raw_outputs = []
    convolved_outputs = []
    states = {}
    for t in range(len(qkv)):
        window = torch.stack((history[0], history[1], history[2], qkv[t]))
        conv_t = torch.sum(window * conv.T, dim=0)
        convolved = torch.nn.functional.silu(conv_t)
        raw_outputs.append(conv_t)
        convolved_outputs.append(convolved)
        if bf16_stores:
            convolved = convolved.to(torch.bfloat16).float()
        history = torch.cat((history[1:], qkv[t : t + 1]))
        q = convolved[:2048].reshape(16, HEAD_DIM)
        k = convolved[2048:4096].reshape(16, HEAD_DIM)
        v = convolved[4096:].reshape(HEADS, HEAD_DIM)
        q = torch.nn.functional.normalize(q, dim=-1, eps=0.0)
        k = torch.nn.functional.normalize(k, dim=-1, eps=0.0)
        # GGUF permutes the value heads to match its whole-head repeat order.
        q = q.repeat(3, 1)
        k = k.repeat(3, 1)
        alpha = torch.exp(decay * torch.nn.functional.softplus(a[t] + dt))
        beta = torch.sigmoid(b[t])
        discounted = alpha[:, None, None] * state
        prediction = torch.sum(discounted * k[:, None, :], dim=-1)
        correction = beta[:, None] * (v - prediction)
        state = discounted + correction[:, :, None] * k[:, None, :]
        o = torch.sum(state * q[:, None, :], dim=-1) / (HEAD_DIM**0.5)
        norm = o * torch.rsqrt(torch.mean(o.square(), dim=-1, keepdim=True) + 1e-6)
        u = norm * gamma * torch.nn.functional.silu(z[t].reshape(HEADS, HEAD_DIM))
        if bf16_stores:
            u = u.to(torch.bfloat16).float()
        outputs.append(u.reshape(VALUE))
        if t in (0, 63, 255, 383, 384):
            states[t] = state.clone()
    return (
        torch.stack(outputs),
        states,
        torch.stack(raw_outputs),
        torch.stack(convolved_outputs),
    )


def main() -> None:
    """Compare proxy and Quartz numerical recipes against a real GGUF trace."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source_gguf", type=Path)
    parser.add_argument("proxy_gguf", type=Path)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    source = {t.name: t for t in gguf.GGUFReader(str(args.source_gguf)).tensors}
    proxy = {t.name: t for t in gguf.GGUFReader(str(args.proxy_gguf)).tensors}
    prefix = "blk.0."
    qkv_name = prefix + "attn_qkv.weight"
    out_name = prefix + "ssm_out.weight"
    x = np.concatenate(
        [
            load_trace(
                args.trace_dir / f"{qkv_name}.{phase}.bin", qkv_name, phase, HIDDEN
            )
            for phase in ("prefill", "decode")
        ]
    )
    traced = np.concatenate(
        [
            load_trace(
                args.trace_dir / f"{out_name}.{phase}.bin", out_name, phase, VALUE
            )
            for phase in ("prefill", "decode")
        ]
    )
    if x.shape != (385, HIDDEN) or traced.shape != (385, VALUE):
        raise ValueError("expected 384 prompt rows and one decode row")
    x_gpu = torch.from_numpy(x.copy()).cuda()
    controls = {
        key: dequant(proxy[prefix + key])
        for key in (
            "ssm_conv1d.weight",
            "ssm_a",
            "ssm_dt.bias",
            "ssm_norm.weight",
            "ssm_alpha.weight",
            "ssm_beta.weight",
        )
    }
    conv = controls["ssm_conv1d.weight"]
    decay = controls["ssm_a"]
    dt = controls["ssm_dt.bias"]
    gamma = controls["ssm_norm.weight"]
    runs = {}
    with torch.no_grad():
        for name, quartz in (("proxy", False), ("quartz", True)):
            control_input = x_gpu.to(torch.bfloat16).float() if quartz else x_gpu
            a = control_input @ controls["ssm_alpha.weight"].T
            b = control_input @ controls["ssm_beta.weight"].T
            qkv = projection(x_gpu, source[qkv_name], proxy[qkv_name], quartz, quartz)
            z_name = prefix + "attn_gate.weight"
            z = projection(x_gpu, source[z_name], proxy[z_name], quartz, quartz)
            runs[name] = (*replay(qkv, z, a, b, conv, decay, dt, gamma, quartz), qkv)
            print(name, "replay_complete", flush=True)
    reference = torch.from_numpy(traced.copy()).cuda()
    report = {
        "schema": "task022-gdn-state-replay-v1",
        "positions": {},
        "validation": {},
    }
    for key, calculated in (
        ("linear_attn_qkv_mixed", runs["proxy"][4]),
        ("conv_output_raw", runs["proxy"][2]),
        ("conv_output_silu", runs["proxy"][3]),
    ):
        path = args.trace_dir / f"node.{key}-0.prefill.bin"
        if not path.exists():
            continue
        observed = load_trace(path, f"{key}-0", "prefill", QKV)
        report["validation"][key] = {
            str(p): errors(calculated[p].cpu().numpy(), observed[p])
            for p in (0, 63, 255, 383)
        }
    for position in (0, 63, 255, 383, 384):
        proxy_u, proxy_states = runs["proxy"][:2]
        quartz_u, quartz_states = runs["quartz"][:2]
        report["positions"][str(position)] = {
            "proxy_replay_vs_trace": errors(
                proxy_u[position].cpu().numpy(), reference[position].cpu().numpy()
            ),
            "quartz_vs_proxy_replay_u": errors(
                quartz_u[position].cpu().numpy(), proxy_u[position].cpu().numpy()
            ),
            "quartz_vs_proxy_replay_state": errors(
                quartz_states[position].cpu().numpy(),
                proxy_states[position].cpu().numpy(),
            ),
        }
        print(position, report["positions"][str(position)], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
