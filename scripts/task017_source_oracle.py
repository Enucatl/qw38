# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "accelerate==1.12.0",
#   "numpy==2.5.3",
#   "safetensors==0.8.0",
#   "torch==2.9.0",
#   "transformers==5.17.0",
# ]
# ///
"""Record Qwen3.5 source logits, residuals, and selected cache state."""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers


def write_tensor(path: Path, tensor: torch.Tensor) -> str:
    """Write an activation as contiguous FP32 and return its SHA-256."""
    raw = tensor.float().cpu().contiguous().numpy().tobytes()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    """Run cached source decode and an independent two-token forward check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("tokens", nargs="+", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(
        "torch", torch.__version__, "transformers", transformers.__version__, flush=True
    )
    model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "26GiB", "cpu": "20GiB"},
        offload_folder=args.output / "offload",
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    print("placement", Counter(map(str, model.hf_device_map.values())), flush=True)

    cache = None
    with torch.inference_mode():
        for position, token in enumerate(args.tokens):
            ids = torch.tensor([[token]], dtype=torch.long, device="cuda:0")
            out = model(
                input_ids=ids,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                logits_to_keep=1,
            )
            cache = out.past_key_values
            logits = out.logits[0, -1]
            digest = write_tensor(args.output / f"source-logits-{position}.f32", logits)
            residuals = np.stack(
                [value[0, -1].float().cpu().numpy() for value in out.hidden_states]
            )
            residuals.tofile(args.output / f"source-residuals-{position}.f32")
            for layer in (0, 28, 60):
                state = cache.layers[layer]
                for kind, value in (
                    ("conv", state.conv_states[0]),
                    ("recurrent", state.recurrent_states[0]),
                ):
                    write_tensor(
                        args.output / f"source-{kind}-{position}-{layer}.f32",
                        value,
                    )
            attention = cache.layers[3]
            for kind, value in (("key", attention.keys), ("value", attention.values)):
                write_tensor(args.output / f"source-{kind}-{position}-3.f32", value)
            print(
                "position",
                position,
                "token",
                token,
                "argmax",
                int(logits.argmax()),
                "logits_sha256",
                digest,
                flush=True,
            )

        if len(args.tokens) == 2:
            ids = torch.tensor([args.tokens], dtype=torch.long, device="cuda:0")
            out = model(input_ids=ids, use_cache=False, logits_to_keep=1)
            digest = write_tensor(
                args.output / "source-prefill-logits-1.f32", out.logits[0, -1]
            )
            print("two_token_forward_sha256", digest, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
