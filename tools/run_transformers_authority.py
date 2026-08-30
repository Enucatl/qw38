"""Run the pinned official checkpoint through Transformers eager execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import torch
from transformers import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5

from tools.transformers_taps import TapCapture
from tools.verify_transformers_authority import verify


def device_map() -> dict[str, int | str]:
    mapping: dict[str, int | str] = {
        "model.visual": "disk",
        "model.language_model.embed_tokens": 0,
        "model.language_model.norm": 0,
        "lm_head": 0,
    }
    for layer in range(64):
        if layer <= 24:
            device: int | str = 0
        elif layer <= 44:
            device = "cpu"
        else:
            device = "disk"
        mapping[f"model.language_model.layers.{layer}"] = device
    return mapping


def greedy(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1).item())


def run(
    checkpoint: Path,
    source: Path,
    contract_path: Path,
    offload: Path,
    logits_path: Path,
    taps_path: Path,
) -> dict[str, Any]:
    identity = verify(contract_path, checkpoint, source)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_torch = contract["environment"]["torch"]
    if torch.__version__ != expected_torch:
        raise ValueError(f"torch version differs: {torch.__version__}")
    if torch.version.cuda != "13.0" or not torch.cuda.is_available():
        raise ValueError("CUDA 13.0 PyTorch authority device is unavailable")
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("Transformers authority requires compute capability 12.0")

    offload.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    load_started = time.monotonic()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=device_map(),
        offload_folder=offload,
        offload_buffers=True,
    )
    model.eval()
    load_seconds = time.monotonic() - load_started
    actual_map = dict(model.hf_device_map)
    if actual_map != device_map():
        raise ValueError("loaded Transformers device map differs from contract")

    capture = TapCapture()
    capture.install(model, modeling_qwen3_5)

    tokens = contract["execution"]["input_tokens"]
    rows: list[torch.Tensor] = []
    greedy_tokens: list[int] = []
    past_key_values = None
    execute_started = time.monotonic()
    try:
        with torch.inference_mode():
            for position, token in enumerate(tokens):
                capture.set_position(position)
                input_ids = torch.tensor([[token]], dtype=torch.long, device="cuda:0")
                output = model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                past_key_values = output.past_key_values
                row = output.logits[0, -1].float().cpu()
                capture.add_logits(row)
                rows.append(row)
                greedy_tokens.append(greedy(row))
    finally:
        capture.close()
    torch.cuda.synchronize()
    execute_seconds = time.monotonic() - execute_started

    raw = torch.stack(rows).numpy().astype("<f4", copy=False).tobytes()
    temporary = logits_path.with_name(f"{logits_path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(raw)
    temporary.replace(logits_path)
    taps = capture.write(taps_path)
    return {
        "schema_version": 1,
        "identity": identity,
        "torch": torch.__version__,
        "transformers_attention": model.config.text_config._attn_implementation,
        "device_map": actual_map,
        "input_tokens": tokens,
        "greedy_tokens": greedy_tokens,
        "vocabulary_size": rows[0].numel(),
        "logits_bytes": len(raw),
        "logits_sha256": hashlib.sha256(raw).hexdigest(),
        "taps": taps,
        "load_seconds": load_seconds,
        "execute_seconds": execute_seconds,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--offload", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--taps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = run(
        args.checkpoint,
        args.source,
        args.contract,
        args.offload,
        args.logits,
        args.taps,
    )
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
