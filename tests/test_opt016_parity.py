from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
LLAMA_IMAGE = "qw38-llama-authority:cuda-13.0.2"
PREFIX = "QW38_PREFILL_2K_PARITY_RESULT="
CONTRACT = ROOT / "pins/opt016_parity_contract.json"
FIXTURE = ROOT / "fixtures/opt016_parity.json"
REPORT = ROOT / "evidence/optimization/opt016-2k-parity/REPORT.md"
LLAMA_JSON = ROOT / "evidence/optimization/opt016-2k-parity/llama-bench-2k.json"
QUARTZ_JSON = ROOT / "evidence/optimization/opt016-2k-parity/quartz-2k.json"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def validate_result(result: Any) -> None:
    contract = _contract()
    mmq = json.loads((ROOT / "pins" / "cuda_mmq_contract.json").read_text())
    gdn = json.loads((ROOT / "pins" / "cuda_gdn_scan_contract.json").read_text())
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-016"
    assert result["status"] == "measured"
    assert contract["primary_tokens"] == 2048
    assert contract["yardstick"] == "2k_only"
    assert contract["q8_0_production"] == "opt009_byte_exact"
    assert contract["envelopes_unloosened"] is True
    assert contract["cud002_max_abs"] == mmq["admission"]["maximum_absolute_error"]
    assert contract["cud002_max_rms"] == mmq["admission"]["maximum_rms_error"]
    assert contract["gdn_max_abs"] == gdn["admission"]["maximum_absolute_error"]
    assert contract["gdn_max_rms"] == gdn["admission"]["maximum_rms_error"]
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    quartz = result["quartz"]
    assert quartz["prompt_tokens"] == 2048
    assert quartz["replicates"] == contract["quartz_replicates"] == 3
    assert quartz["cold"] is True
    assert quartz["cache_policy"] == "disabled"
    assert quartz["attribution"] is None
    assert len(quartz["wall_ms"]) == 3 and len(quartz["tok_s"]) == 3
    mean = sum(float(v) for v in quartz["tok_s"]) / 3.0
    assert quartz["mean_tok_s"] == pytest.approx(mean, rel=1e-6, abs=1e-6)
    llama = result["llama_cpp"]
    assert llama["n_prompt"] == 2048
    assert result["gate_passed"] is (
        float(quartz["mean_tok_s"]) >= float(llama["avg_ts"])
    )
    assert result["ranks_landed"][0] == "q4k_q6k_mma_mmq"
    assert result["mmq_prompt_tile_2048"] == contract["mmq_prompt_tile_2048"] == 128
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    proof = result["proof_limit"]
    for phrase in contract["proof_limit"]:
        assert phrase in proof
    assert result["report_path"] == contract["report_path"]
    assert REPORT.is_file()
    report = REPORT.read_text()
    for phrase in contract["proof_limit"]:
        assert phrase in report


def test_opt016_contract_and_fixture_are_connected() -> None:
    validate_result(json.loads(FIXTURE.read_text()))
    contract = _contract()
    assert contract["llama_bench_args"] == [
        "-p",
        "2048",
        "-n",
        "0",
        "--no-warmup",
        "-r",
        "3",
        "-ngl",
        "99",
    ]
    assert contract["quartz_warmups"] == 0
    assert contract["quartz_attribution"] == "null_unperturbed"
    assert contract["ranks_required"] == ["q4k_q6k_mma_mmq"]
    assert contract["historical_llama_avg_ts"] == 3114.049476


def test_opt016_validator_rejects_inadmissible_evidence() -> None:
    fixture = json.loads(FIXTURE.read_text())
    mutations = []

    def flip_gate(changed: dict[str, Any]) -> None:
        changed["gate_passed"] = True
        if changed["quartz"]["mean_tok_s"] >= changed["llama_cpp"]["avg_ts"]:
            changed["quartz"]["mean_tok_s"] = changed["llama_cpp"]["avg_ts"] / 2.0
            changed["quartz"]["tok_s"] = [changed["quartz"]["mean_tok_s"]] * 3

    for mutate in (
        lambda x: x["quartz"].__setitem__("prompt_tokens", 2052),
        flip_gate,
        lambda x: x.__setitem__(
            "proof_limit", x["proof_limit"].replace("byte-exact", "approx")
        ),
        lambda x: x["llama_cpp"].__setitem__("n_prompt", 8192),
        lambda x: x.__setitem__("nsight_systems", "/tmp/capture.nsys-rep"),
        lambda x: x.__setitem__("status", "source_inspected"),
        lambda x: x.__setitem__("ranks_landed", ["gdn_fused_token_loop"]),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(AssertionError):
            validate_result(mutation)
    contract = json.loads(CONTRACT.read_text())
    assert contract["q8_0_production"] == "opt009_byte_exact"
    loosened = json.loads(json.dumps(contract))
    loosened["cud002_max_abs"] = 0.05
    loosened["q8_0_production"] = "mma_approx"
    with pytest.raises(AssertionError):
        assert loosened["q8_0_production"] == "opt009_byte_exact"
    with pytest.raises(AssertionError):
        assert loosened["cud002_max_abs"] == 0.0005


def _common(image: str) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{ROOT}:/workspace",
        "-w",
        "/workspace",
        image,
    ]


def _run_llama_bench() -> dict[str, Any]:
    command = [
        *_common(LLAMA_IMAGE),
        "bash",
        "-lc",
        "cd /workspace && .cache/authorities/llama-build/bin/llama-bench "
        "-m /workspace/models/Qwen3.8-27B-Q4_K_M.gguf -p 2048 -n 0 --no-warmup -r 3 -ngl 99 -o json",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    text = completed.stdout + completed.stderr
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list) and payload and payload[0].get("n_prompt") == 2048:
            LLAMA_JSON.write_text(json.dumps(payload, indent=2) + "\n")
            return payload[0]
    raise AssertionError("llama-bench JSON with n_prompt 2048 was not found\n" + text)


def _run_quartz() -> dict[str, Any]:
    nvcc = [
        *_common(IMAGE),
        "nvcc",
        "-std=c++17",
        "-O2",
        "-arch=sm_120",
        "--expt-relaxed-constexpr",
        "--fmad=false",
        "-Xcompiler=-Wall,-Wextra,-Werror,-fno-exceptions,-fno-rtti,-ffp-contract=off,-pthread",
        "-Iinclude",
        "-Isrc",
        "-Ithird_party/utf8proc",
        "-Icuda",
        "-DQW38_DIAGNOSTIC_TRACE",
        "cuda/prefill_2k_parity_test.cu",
        "build/full_scheduler.trace.cuda.o",
        "build/scheduler_primitives.cuda.o",
        "build/quant_mmv.cuda.o",
        "build/gdn_step.cuda.o",
        "build/attention_decode.cuda.o",
        "build/diagnostic/status.o",
        "build/diagnostic/sha256.o",
        "build/diagnostic/model.o",
        "build/diagnostic/tokenizer.o",
        "build/diagnostic/template.o",
        "build/diagnostic/quant.o",
        "build/diagnostic/tensor.o",
        "build/diagnostic/conversion.o",
        "build/diagnostic/projection.o",
        "build/diagnostic/weights.o",
        "build/diagnostic/mixer.o",
        "build/diagnostic/scheduler.o",
        "build/diagnostic/scalar_runtime.o",
        "build/diagnostic/gdn.o",
        "build/diagnostic/attention.o",
        "build/diagnostic/engine.o",
        "build/diagnostic/diagnostic_trace.o",
        "build/utf8proc.o",
        "-o",
        "build/qw38-cuda-prefill-2k-parity-test",
    ]
    commands = [
        [*_common(IMAGE), "make", "build/qw38-cuda-timing-test"],
        nvcc,
        [
            *_common(IMAGE),
            "./build/qw38-cuda-prefill-2k-parity-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ],
    ]
    outputs: list[str] = []
    for command in commands:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        outputs.append(completed.stdout)
    records = [
        json.loads(line.removeprefix(PREFIX))
        for line in outputs[-1].splitlines()
        if line.startswith(PREFIX)
    ]
    assert len(records) == 1
    assert "status=passed" in outputs[-1]
    QUARTZ_JSON.write_text(json.dumps(records[0], indent=2) + "\n")
    return records[0]


def test_opt016_native_2k_gate() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    if not MODEL.exists():
        pytest.skip("the pinned GGUF is required")

    llama = _run_llama_bench()
    quartz = _run_quartz()
    mean_tok_s = float(quartz["mean_tok_s"])
    avg_ts = float(llama["avg_ts"])
    gate_passed = mean_tok_s >= avg_ts
    proof = (
        "2K yardstick only; envelopes unloosened; "
        "no 8K/32K/128K throughput gate; Q8_0 remains byte-exact"
    )
    fixture = {
        "schema_version": 1,
        "task": "OPT-016",
        "status": "measured",
        "measurement_utc": quartz["measurement_utc"],
        "device": quartz["device"],
        "compute_capability": quartz["compute_capability"],
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "ranks_landed": [
            "q4k_q6k_mma_mmq",
            "gdn_fused_token_loop",
            "causal_mma_attention",
        ],
        "mmq_prompt_tile_2048": 128,
        "quartz": {
            "prompt_tokens": 2048,
            "replicates": 3,
            "wall_ms": quartz["wall_ms"],
            "tok_s": quartz["tok_s"],
            "mean_tok_s": mean_tok_s,
            "cold": True,
            "cache_policy": "disabled",
            "attribution": None,
        },
        "llama_cpp": {
            "avg_ts": avg_ts,
            "avg_ns": llama["avg_ns"],
            "n_prompt": 2048,
            "n_ubatch": llama.get("n_ubatch", 512),
            "flash_attn": llama.get("flash_attn", -1),
            "build_commit": llama.get("build_commit", "cc83d7b"),
            "test_time": llama.get("test_time", quartz["measurement_utc"]),
        },
        "llama_cpp_historical_avg_ts": 3114.049476,
        "gate_passed": gate_passed,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": proof,
        "report_path": "evidence/optimization/opt016-2k-parity/REPORT.md",
    }
    validate_result(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    assert gate_passed, (
        f"2K gate failed: quartz mean_tok_s={mean_tok_s} llama avg_ts={avg_ts}"
    )
