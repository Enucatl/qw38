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
PREFIX = "QW38_PREFILL_4K_ORACLE_RESULT="
CONTRACT = ROOT / "pins/opt021_oracle_contract.json"
FIXTURE = ROOT / "fixtures/opt021_oracle.json"
EVIDENCE = ROOT / "evidence/optimization/opt021-4k-oracle"
REPORT = EVIDENCE / "REPORT.md"
LLAMA_JSON = EVIDENCE / "llama-bench-4k.json"
QUARTZ_JSON = EVIDENCE / "quartz-4k.json"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
LLAMA_BENCH = ROOT / ".cache/authorities/llama-build/bin/llama-bench"
PROOF = (
    "4K keep/reject oracle; envelopes unloosened; "
    "does not substitute for the 2K llama.cpp parity gate; "
    "attribution null; graphs created; "
    "scout sitting is not the retained fixture"
)
SCOUT_QUARTZ = 965.204895
SCOUT_LLAMA = 3182.476587


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-021"
    assert result["status"] == "measured"
    assert result["status"] not in ("scout", "source_inspected")
    assert contract["primary_tokens"] == 4096
    assert contract["yardstick"] == "4k_keep_reject_oracle"
    assert contract["q8_0_production"] == "opt017_mma_with_opt009_reference"
    assert contract["envelopes_unloosened"] is True
    assert contract["quartz_graphs"] == "created"
    assert contract["keep_reject_rule"] == "strictly_greater_mean_tok_s"
    assert contract["scout_is_retained_fixture"] is False
    assert contract["owns_opt016_parity_gate"] is False
    assert contract["substitutes_for_opt016"] is False
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["scout_quartz_mean_tok_s"] == contract["scout_quartz_mean_tok_s"]
    assert result["scout_quartz_mean_tok_s"] == SCOUT_QUARTZ
    assert result["scout_llama_avg_ts"] == contract["scout_llama_avg_ts"] == SCOUT_LLAMA
    quartz = result["quartz"]
    assert quartz["prompt_tokens"] == 4096
    assert quartz["replicates"] == contract["quartz_replicates"] == 3
    assert quartz["cold"] is True
    assert quartz["cache_policy"] == "disabled"
    assert quartz["attribution"] is None
    assert quartz["graphs_created"] is True
    assert quartz["prompt_graph_rows"] == 4096
    assert len(quartz["wall_ms"]) == 3 and len(quartz["tok_s"]) == 3
    mean = sum(float(v) for v in quartz["tok_s"]) / 3.0
    assert quartz["mean_tok_s"] == pytest.approx(mean, rel=1e-6, abs=1e-6)
    llama = result["llama_cpp"]
    assert llama["n_prompt"] == 4096
    assert "n_batch" in llama
    assert "n_ubatch" in llama
    assert "flash_attn" in llama
    assert "avg_ts" in llama and "avg_ns" in llama
    assert "build_commit" in llama and "test_time" in llama
    meets = float(quartz["mean_tok_s"]) >= float(llama["avg_ts"])
    assert result["quartz_meets_llama"] is meets
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    proof = result["proof_limit"]
    for phrase in contract["proof_limit"]:
        assert phrase in proof
    assert result["report_path"] == contract["report_path"]
    assert "/tmp" not in result["report_path"]
    assert REPORT.is_file()
    report = REPORT.read_text()
    for phrase in contract["proof_limit"]:
        assert phrase in report
    assert "/tmp/oracle4k" not in report


def test_opt021_contract_and_fixture_are_connected() -> None:
    validate_result(json.loads(FIXTURE.read_text()))
    contract = _contract()
    assert contract["llama_bench_args"] == [
        "-p",
        "4096",
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
    assert contract["quartz_graphs"] == "created"
    assert contract["scout_is_retained_fixture"] is False
    assert contract["keep_reject_rule"] == "strictly_greater_mean_tok_s"


def test_opt021_validator_rejects_inadmissible_evidence() -> None:
    fixture = json.loads(FIXTURE.read_text())
    mutations = []

    def flip_meets(changed: dict[str, Any]) -> None:
        changed["quartz_meets_llama"] = True
        if changed["quartz"]["mean_tok_s"] >= changed["llama_cpp"]["avg_ts"]:
            changed["quartz"]["mean_tok_s"] = changed["llama_cpp"]["avg_ts"] / 2.0
            changed["quartz"]["tok_s"] = [changed["quartz"]["mean_tok_s"]] * 3

    for mutate in (
        lambda x: x["quartz"].__setitem__("prompt_tokens", 2048),
        lambda x: x["quartz"].__setitem__("prompt_tokens", 2052),
        lambda x: x["llama_cpp"].__setitem__("n_prompt", 2048),
        lambda x: x["quartz"].__setitem__("attribution", {"ffn_mmq": 1.0}),
        lambda x: x["quartz"].__setitem__("replicates", 1),
        lambda x: x["quartz"].__setitem__("graphs_created", False),
        lambda x: x.__setitem__("substitutes_for_opt016", True),
        lambda x: x.__setitem__("owns_opt016_parity_gate", True),
        flip_meets,
        lambda x: x.__setitem__("nsight_systems", "/tmp/capture.nsys-rep"),
        lambda x: x.__setitem__("status", "source_inspected"),
        lambda x: x.__setitem__("status", "scout"),
        lambda x: x.__setitem__("report_path", "/tmp/oracle4k/REPORT.md"),
        lambda x: x.__setitem__(
            "proof_limit",
            x["proof_limit"].replace(
                "does not substitute for the 2K llama.cpp parity gate",
                "substitutes for 2K",
            ),
        ),
        lambda x: x.__setitem__("task", "OPT-016"),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(AssertionError):
            validate_result(mutation)


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


def _ensure_llama_bench() -> None:
    if LLAMA_BENCH.is_file():
        return
    configure = [
        *_common(LLAMA_IMAGE),
        "cmake",
        "-S",
        "/workspace/.cache/authorities/llama.cpp",
        "-B",
        "/workspace/.cache/authorities/llama-build",
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_ARCHITECTURES=120",
        "-DGGML_CUDA=ON",
        "-DGGML_NATIVE=OFF",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=ON",
    ]
    build = [
        *_common(LLAMA_IMAGE),
        "cmake",
        "--build",
        "/workspace/.cache/authorities/llama-build",
        "--target",
        "llama-bench",
        "-j",
        "6",
    ]
    for command in (configure, build):
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    assert LLAMA_BENCH.is_file(), "llama-bench was not built"


def _run_llama_bench() -> dict[str, Any]:
    _ensure_llama_bench()
    command = [
        *_common(LLAMA_IMAGE),
        "bash",
        "-lc",
        "cd /workspace && .cache/authorities/llama-build/bin/llama-bench "
        "-m /workspace/models/Qwen3.8-27B-Q4_K_M.gguf "
        "-p 4096 -n 0 --no-warmup -r 3 -ngl 99 -o json",
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
        if isinstance(payload, list) and payload and payload[0].get("n_prompt") == 4096:
            EVIDENCE.mkdir(parents=True, exist_ok=True)
            LLAMA_JSON.write_text(json.dumps(payload, indent=2) + "\n")
            return payload[0]
    raise AssertionError("llama-bench JSON with n_prompt 4096 was not found\n" + text)


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
        "cuda/prefill_4k_oracle_test.cu",
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
        "build/qw38-cuda-prefill-4k-oracle-test",
    ]
    commands = [
        [*_common(IMAGE), "make", "build/qw38-cuda-timing-test"],
        nvcc,
        [
            *_common(IMAGE),
            "./build/qw38-cuda-prefill-4k-oracle-test",
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
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    QUARTZ_JSON.write_text(json.dumps(records[0], indent=2) + "\n")
    return records[0]


def _write_report(fixture: dict[str, Any]) -> None:
    quartz = fixture["quartz"]
    llama = fixture["llama_cpp"]
    walls = ", ".join(str(v) for v in quartz["wall_ms"])
    tok_s = ", ".join(str(v) for v in quartz["tok_s"])
    graphs_created = json.dumps(quartz["graphs_created"])
    meets = json.dumps(fixture["quartz_meets_llama"])
    text = f"""# OPT-021 — Cold 4K prefill keep/reject oracle

## Claim labels and proof limits

This increment pins the **4K keep/reject oracle**. Numeric and exact-state
**envelopes unloosened**. This protocol **does not substitute for the 2K llama.cpp parity gate**.
Timed Quartz walls use **attribution null** and **graphs created**.
The **scout sitting is not the retained fixture**.

Live numbers in `fixtures/opt021_oracle.json`, `llama-bench-4k.json`, and
`quartz-4k.json` are the Measured same-sitting comparison. Scout Quartz
{SCOUT_QUARTZ} tok/s and llama.cpp {SCOUT_LLAMA} tok/s are contract
transparency only.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `{GGUF_SHA}` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `{LLAMA_REV}` |
| Tokens | exact 4096, not chat-rendered |
| Quartz token IDs | `(42 + index * 997) % kVocabularySize` |
| Quartz path | production `sync_tokens`, attribution null, graphs created, production fused GDN, cache_policy disabled, 0 warm-ups, 3 cold replicates |
| llama.cpp | `llama-bench -p 4096 -n 0 --no-warmup -r 3 -ngl 99` |
| Same sitting | llama.cpp first, then Quartz |
| Nsight | not_used |
| Keep/reject | strictly greater Quartz mean tok/s than this retained baseline |

## Measured 4K oracle (implementation sitting)

- Device: {fixture["device"]} compute {fixture["compute_capability"]}
- measurement_utc: {fixture["measurement_utc"]}
- Quartz mean tok/s: {quartz["mean_tok_s"]} (walls {walls} ms; tok/s {tok_s})
- Quartz prompt_tokens: {quartz["prompt_tokens"]}; graphs_created: {graphs_created}; prompt_graph_rows: {quartz["prompt_graph_rows"]}
- llama.cpp live `avg_ts`: {llama["avg_ts"]} (`avg_ns` {llama["avg_ns"]}, `n_prompt` {llama["n_prompt"]}, `n_batch` {llama["n_batch"]}, `n_ubatch` {llama["n_ubatch"]}, `flash_attn` {llama["flash_attn"]}, `build_commit` {llama["build_commit"]}, `test_time` {llama["test_time"]})
- `quartz_meets_llama`: {meets} (informational; not this gate)
- `owns_opt016_parity_gate`: false
- `substitutes_for_opt016`: false

## Scout versus retained

Scout sitting 2026-09-09 Quartz {SCOUT_QUARTZ} tok/s and llama.cpp {SCOUT_LLAMA}
tok/s are recorded on the contract. The scout sitting is not the retained fixture.
This report's live numbers are the retained OPT-022+ baseline.

## Keep/reject rule for later production changes

A later production change is kept only if its cold exact-4096 mean tok/s is
strictly greater than the retained OPT-021 `quartz.mean_tok_s` (or a later
successor published under the same protocol). Equality is a reject.

Ladder stop (not this gate): later tasks may stop when Quartz cold 4K mean
tok/s is at or above same-protocol llama.cpp `avg_ts`.
"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def test_opt021_native_4k_oracle() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    if not MODEL.exists():
        pytest.skip("the pinned GGUF is required")

    llama = _run_llama_bench()
    quartz = _run_quartz()
    mean_tok_s = float(quartz["mean_tok_s"])
    avg_ts = float(llama["avg_ts"])
    quartz_meets_llama = mean_tok_s >= avg_ts
    fixture = {
        "schema_version": 1,
        "task": "OPT-021",
        "status": "measured",
        "measurement_utc": quartz["measurement_utc"],
        "device": quartz["device"],
        "compute_capability": quartz["compute_capability"],
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "quartz": {
            "prompt_tokens": 4096,
            "replicates": 3,
            "wall_ms": quartz["wall_ms"],
            "tok_s": quartz["tok_s"],
            "mean_tok_s": mean_tok_s,
            "cold": True,
            "cache_policy": "disabled",
            "attribution": None,
            "graphs_created": True,
            "prompt_graph_rows": 4096,
        },
        "llama_cpp": {
            "avg_ts": avg_ts,
            "avg_ns": llama["avg_ns"],
            "n_prompt": 4096,
            "n_batch": llama.get("n_batch", 2048),
            "n_ubatch": llama.get("n_ubatch", 512),
            "flash_attn": llama.get("flash_attn", -1),
            "build_commit": llama.get("build_commit", "cc83d7b"),
            "test_time": llama.get("test_time", quartz["measurement_utc"]),
        },
        "scout_quartz_mean_tok_s": SCOUT_QUARTZ,
        "scout_llama_avg_ts": SCOUT_LLAMA,
        "quartz_meets_llama": quartz_meets_llama,
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt021-4k-oracle/REPORT.md",
    }
    assert quartz["graphs_created"] is True
    assert quartz["prompt_graph_rows"] == 4096
    assert quartz["prompt_tokens"] == 4096
    _write_report(fixture)
    validate_result(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
