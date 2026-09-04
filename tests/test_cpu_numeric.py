from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "build" / "qw38-eval"
MODEL = ROOT / "models" / "Qwen3.5-2B-Q4_K_M.gguf"
INVENTORY = ROOT / "pins" / "cpu_tensor_inventory.json"
Q5_FIXTURES = ROOT / "fixtures" / "cpu_q5_k_authority.json"
MATVEC = ROOT / "fixtures" / "cpu_avx2_matvec_tolerance.json"
ROWS = ROOT / "fixtures" / "cpu_tensor_rows.json"
TOKENIZER = ROOT / "fixtures" / "cpu_tokenizer_2b.json"
BINDING = ROOT / "fixtures" / "cpu_weight_binding_2b.json"
LLAMA = ROOT / "fixtures" / "cpu_llama_scalar_authority.json"

REAL_STEPS = (
    ("--check-real-gdn-step", ROOT / "fixtures" / "cpu_real_gdn_step_2b.json"),
    (
        "--check-real-attention-step",
        ROOT / "fixtures" / "cpu_real_attention_step_2b.json",
    ),
    ("--check-real-ffn-step", ROOT / "fixtures" / "cpu_real_ffn_step_2b.json"),
    ("--check-real-scalar-token", ROOT / "fixtures" / "cpu_real_scalar_token_2b.json"),
)


def parse_output(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.splitlines() if "=" in line)


def run_eval(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        [str(EVAL), *args],
        check=False,
        capture_output=True,
        text=True,
        env=merged,
    )


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def test_q5_k_scalar_decode_and_dot_match_frozen_hex() -> None:
    fixtures = json.loads(Q5_FIXTURES.read_text())
    for case in fixtures["cases"]:
        result = run_eval("--check-quant", case["kind"], case["block_hex"])
        assert result.returncode == 0, f"{case['name']}: {result.stderr}"
        actual = parse_output(result.stdout)
        assert actual["decoded_f32_le_hex"] == case["decoded_f32_le_hex"], case["name"]
        assert actual["dot_f32_le_hex"] == case["dot_f32_le_hex"], case["name"]


def test_avx2_q8_activation_matvec_stays_inside_frozen_scalar_envelope() -> None:
    document = json.loads(MATVEC.read_text())
    q4 = bytes.fromhex(
        json.loads((ROOT / "fixtures" / "quant_authority.json").read_text())["cases"][
            0
        ]["block_hex"]
    )
    q5 = bytes.fromhex(json.loads(Q5_FIXTURES.read_text())["cases"][0]["block_hex"])
    q6 = bytes.fromhex(
        json.loads((ROOT / "fixtures" / "quant_authority.json").read_text())["cases"][
            2
        ]["block_hex"]
    )
    blocks = {"q4_k": q4, "q5_k": q5, "q6_k": q6}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for case in document["cases"]:
            kind = case["kind"]
            columns = case["columns"]
            rows = case["rows"]
            block = blocks[kind]
            payload = (block * (columns // 256)) * rows
            activation = b"".join(
                struct.pack("<f", f32(((index * 37) % 101 - 50) / 32.0))
                for index in range(columns)
            )
            payload_path = tmp_path / f"{kind}_{columns}.bin"
            act_path = tmp_path / f"{kind}_{columns}.act"
            payload_path.write_bytes(payload)
            act_path.write_bytes(activation)
            avx = run_eval(
                "--check-matvec",
                kind,
                str(columns),
                str(rows),
                f"@{payload_path}",
                f"@{act_path}",
            )
            scalar = run_eval(
                "--check-matvec",
                kind,
                str(columns),
                str(rows),
                f"@{payload_path}",
                f"@{act_path}",
                env={"QW38_SCALAR_MATVEC": "1"},
            )
            assert avx.returncode == 0, avx.stderr
            assert scalar.returncode == 0, scalar.stderr
            left = struct.unpack(
                f"<{rows}f",
                bytes.fromhex(parse_output(avx.stdout)["output_f32_le_hex"]),
            )
            right = struct.unpack(
                f"<{rows}f",
                bytes.fromhex(parse_output(scalar.stdout)["output_f32_le_hex"]),
            )
            errors = [abs(a - b) for a, b in zip(left, right, strict=True)]
            rms = math.sqrt(sum(error * error for error in errors) / len(errors))
            assert max(errors) <= case["maximum_absolute_error"] + 1e-4, kind
            assert rms <= case["root_mean_square_error"] + 1e-4, kind


@pytest.mark.skipif(not MODEL.exists(), reason="the pinned 2B GGUF is required")
def test_inventory_row_sha_and_scalar_dots_match_frozen_2b_samples() -> None:
    inventory = json.loads(INVENTORY.read_text())
    by_name = {tensor["name"]: tensor for tensor in inventory["tensors"]}
    fixtures = json.loads(ROWS.read_text())
    assert fixtures["model_sha256"] == inventory["model_sha256"]
    with MODEL.open("rb") as model:
        for case in fixtures["cases"]:
            tensor = by_name[case["name"]]
            columns, rows = tensor["shape"]
            row_bytes = tensor["storage_bytes"] // rows
            model.seek(tensor["absolute_offset"] + case["row"] * row_bytes)
            payload = model.read(row_bytes)
            assert hashlib.sha256(payload).hexdigest() == case["row_sha256"]
            result = run_eval(
                "--check-tensor-row", str(MODEL), case["name"], str(case["row"])
            )
            assert result.returncode == 0, result.stderr
            actual = parse_output(result.stdout)
            assert actual["dtype"] == case["dtype"]
            assert actual["columns"] == str(columns)
            assert actual["dot_f32_le_hex"] == case["dot_f32_le_hex"]


@pytest.mark.skipif(not MODEL.exists(), reason="the pinned 2B GGUF is required")
def test_2b_weight_binding_is_tied_and_has_320_tensors() -> None:
    fixture = json.loads(BINDING.read_text())["fields"]
    result = run_eval("--check-weight-binding", str(MODEL), "valid")
    assert result.returncode == 0, result.stderr
    actual = parse_output(result.stdout)
    assert actual == fixture
    assert actual["bound_tensors"] == "320"
    assert actual["tied_output"] == "1"
    assert actual["output_shares_embedding"] == "1"
    names = {tensor["name"] for tensor in json.loads(INVENTORY.read_text())["tensors"]}
    assert "output.weight" not in names


@pytest.mark.skipif(not MODEL.exists(), reason="the pinned 2B GGUF is required")
def test_2b_tokenizer_ids_match_this_gguf() -> None:
    fixtures = json.loads(TOKENIZER.read_text())
    for case in fixtures["cases"]:
        result = run_eval("--tokenize-hex", str(MODEL), case["utf8_hex"])
        actual = (
            []
            if not result.stdout.strip()
            else [int(token) for token in result.stdout.strip().split(",")]
        )
        assert result.returncode == 0, f"{case['name']}: {result.stderr}"
        assert actual == case["ids"], case["name"]


@pytest.mark.skipif(not MODEL.exists(), reason="the pinned 2B GGUF is required")
@pytest.mark.parametrize(("flag", "path"), REAL_STEPS)
def test_2b_real_steps_match_frozen_scalar_taps(flag: str, path: Path) -> None:
    fixture = json.loads(path.read_text())["fields"]
    result = run_eval(flag, str(MODEL), "valid", env={"QW38_SCALAR_MATVEC": "1"})
    assert result.returncode == 0, result.stderr
    actual = parse_output(result.stdout)
    assert actual == fixture
    if flag == "--check-real-scalar-token":
        assert actual["layers_completed"] == "24"
        assert actual["logit_count"] == "248320"


def test_cpu_numeric_handbook_names_the_2b_proof_ladder() -> None:
    handbook = (ROOT / "docs" / "66-cpu-laptop-2b.md").read_text()
    for term in (
        "proof ladder",
        "same-GGUF",
        "QW38_SCALAR_MATVEC",
        "skip-if-missing",
        "Transformers",
    ):
        assert term in handbook


@pytest.mark.skipif(
    not LLAMA.exists(),
    reason="cpu_llama_scalar_authority.json is generated by the Darwin CPU llama.cpp oracle",
)
def test_checked_in_2b_llama_metrics_retain_vocab_and_greedy_ids() -> None:
    fixture = json.loads(LLAMA.read_text())
    assert fixture["vocabulary_size"] == 248320
    assert fixture["input_tokens"] == [42, 3649]
    assert len(fixture["comparisons"]) == 2
    assert all(row["metrics"]["count"] == 248320 for row in fixture["comparisons"])
    assert [row["llama_greedy_token"] for row in fixture["comparisons"]] == [
        261,
        8454,
    ]
    assert [row["quartz_greedy_token"] for row in fixture["comparisons"]] == [
        261,
        8454,
    ]
