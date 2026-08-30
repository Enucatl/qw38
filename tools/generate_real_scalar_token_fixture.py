from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "build" / "qw38-eval"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
OUTPUT = ROOT / "fixtures" / "real_scalar_token.json"
MODEL_SHA256 = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"


def main() -> None:
    result = subprocess.run(
        [str(EVAL), "--check-real-scalar-token", str(MODEL), "valid"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines())
    document = {
        "schema_version": 1,
        "model_sha256": MODEL_SHA256,
        "input_token": 42,
        "authority": "native structural 64-layer zero-state scalar regression",
        "proof_limit": "not an independent semantic continuation authority; requires TRC-001, TRC-002, and ORA-001",
        "expected": values,
    }
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
