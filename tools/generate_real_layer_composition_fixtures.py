from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "build" / "qw38-eval"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
OUTPUT = ROOT / "fixtures" / "real_layer_composition.json"
MODEL_SHA256 = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
FIELDS = (
    "post_mixer_f32_le_hex",
    "ffn_normalized_f32_le_hex",
    "ffn_gate_f32_le_hex",
    "ffn_up_f32_le_hex",
    "ffn_activated_f32_le_hex",
    "ffn_correction_f32_le_hex",
    "residual_output_f32_le_hex",
)


def capture(command: str) -> dict[str, str]:
    result = subprocess.run(
        [str(EVAL), command, str(MODEL), "layer"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines())
    return {field: values[field] for field in FIELDS}


def main() -> None:
    document = {
        "schema_version": 1,
        "model_sha256": MODEL_SHA256,
        "authority": "native composition regression over separately admitted mixer and FFN semantic boundaries",
        "proof_limit": "structural branch-order fixture; not an independent full-layer semantic authority",
        "cases": {
            "gdn_layer_0_position_0": capture("--check-real-gdn-step"),
            "attention_layer_3_position_1": capture("--check-real-attention-step"),
        },
    }
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
