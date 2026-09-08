from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
LLAMA_IMAGE = "qw38-llama-authority:cuda-13.0.2"
PARITY_PREFIX = "QW38_PREFILL_2K_PARITY_RESULT="
ATTRIBUTION_PREFIX = "QW38_PREFILL_ATTRIBUTION_RESULT="
CONTRACT = ROOT / "pins/opt017_mixer_mma_contract.json"
FIXTURE = ROOT / "fixtures/opt017_mixer_mma.json"
EVIDENCE = ROOT / "evidence/optimization/opt017-mixer-q8-mma"
REPORT = EVIDENCE / "REPORT.md"
LLAMA_JSON = EVIDENCE / "llama-bench-2k.json"
QUARTZ_JSON = EVIDENCE / "quartz-2k.json"
ATTRIBUTION = EVIDENCE / "attribution.json"
SWEEP_RAW = EVIDENCE / "q8-mma-j-sweep-raw.txt"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
CATEGORIES = (
    "embedding",
    "gdn",
    "attention",
    "ffn_mmq",
    "logits",
    "commit_sync",
    "graph",
    "other_idle",
)
PROOF = (
    "envelopes unloosened; OPT-009 Q8_0 reference remains byte-exact; "
    "OPT-016 remains the parity gate owner; not an end-to-end 2K tok/s gate; "
    "no 8K/32K/128K throughput gate"
)


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def _mmq() -> dict[str, Any]:
    return json.loads((ROOT / "pins" / "cuda_mmq_contract.json").read_text())


def _attribution_contract() -> dict[str, Any]:
    return json.loads(
        (ROOT / "pins" / "cuda_prefill_attribution_contract.json").read_text()
    )


def validate_result(
    result: Any,
    *,
    contract: dict[str, Any] | None = None,
    attribution_path: Path = ATTRIBUTION,
    report_path: Path = REPORT,
) -> None:
    if contract is None:
        contract = _contract()
    mmq = _mmq()
    attr_contract = _attribution_contract()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-017"
    assert result["status"] == "measured"
    assert result["status"] != "source_inspected"
    assert contract["primary_tokens"] == 2048
    assert contract["owns_opt016_parity_gate"] is False
    assert result["owns_opt016_parity_gate"] is False
    assert contract["opt016_status"] == "blocked"
    assert result["opt016_status"] == "blocked"
    assert contract["q8_0_production"] == "mma_for_prompt_rows_ge_8"
    assert result["q8_0_production"] == "mma_for_prompt_rows_ge_8"
    assert contract["q8_0_reference"] == (
        "launch_q8_mmq_bf16_variant_byte_exact_to_reference"
    )
    assert contract.get("q8_0_mma_admission_reference") == (
        "launch_quant_mmq_variant_kQ8_0"
    )
    assert contract["q8_0_mma_admission_reference"] != ("launch_q8_mmq_bf16_variant")
    assert contract["envelopes_unloosened"] is True
    assert result["envelopes_unloosened"] is True
    assert contract["opt009_q8_reference_exact"] is True
    assert contract["cud002_max_abs"] == mmq["admission"]["maximum_absolute_error"]
    assert contract["cud002_max_rms"] == mmq["admission"]["maximum_rms_error"]
    assert contract["cud002_max_abs"] == 0.0005
    assert contract["cud002_max_rms"] == 0.00025
    assert contract["mma_threshold_prompt_rows"] == 8
    assert contract["legal_mma_tiles"] == [32, 64, 128]
    assert result["q8_mma_prompt_tile_2048"] == contract["q8_mma_prompt_tile_2048"]
    assert result["q8_mma_prompt_tile_2048"] in contract["legal_mma_tiles"]
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
    would = float(quartz["mean_tok_s"]) >= float(llama["avg_ts"])
    assert result["would_pass_opt016"] is would
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    proof = result["proof_limit"]
    for phrase in contract["proof_limit"]:
        assert phrase in proof
    assert result["report_path"] == contract["report_path"]
    assert report_path.is_file()
    report = report_path.read_text()
    for phrase in contract["proof_limit"]:
        assert phrase in report
    assert attribution_path.is_file()
    attribution = json.loads(attribution_path.read_text())
    nested = result["attribution"]
    assert nested["categories_ms"] == attribution["categories_ms"]
    assert nested["wall_ms"] == attribution["wall_ms"]
    categories = nested["categories_ms"]
    assert set(categories) == set(CATEGORIES)
    attributed = sum(float(categories[name]) for name in CATEGORIES)
    assert math.isclose(
        attributed,
        float(nested["wall_ms"]),
        rel_tol=attr_contract["rel_tol"],
        abs_tol=attr_contract["abs_tol_ms"],
    )


def test_opt017_contract_and_fixture_are_connected() -> None:
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
    assert contract["quartz_attribution_for_remasurement"] == "null_unperturbed"
    assert contract["owns_opt016_parity_gate"] is False
    assert contract["q8_0_mma_admission_reference"] == (
        "launch_quant_mmq_variant_kQ8_0"
    )


def test_opt017_validator_rejects_inadmissible_evidence() -> None:
    fixture = json.loads(FIXTURE.read_text())
    mutations = []
    for mutate in (
        lambda x: x.__setitem__("owns_opt016_parity_gate", True),
        lambda x: x["quartz"].__setitem__("prompt_tokens", 2052),
        lambda x: x.__setitem__("q8_0_production", "opt009_byte_exact"),
        lambda x: x["llama_cpp"].__setitem__("n_prompt", 8192),
        lambda x: x.__setitem__("nsight_systems", "/tmp/capture.nsys-rep"),
        lambda x: x.__setitem__("status", "source_inspected"),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(AssertionError):
            validate_result(mutation)
    with pytest.raises(AssertionError):
        validate_result(fixture, attribution_path=Path("/tmp/missing-opt017.json"))
    loosened = json.loads(json.dumps(_contract()))
    loosened["cud002_max_abs"] = 0.05
    loosened["opt009_q8_reference_exact"] = False
    loosened["q8_0_production"] = "opt009_byte_exact"
    with pytest.raises(AssertionError):
        validate_result(fixture, contract=loosened)
    variant_gate = json.loads(json.dumps(_contract()))
    variant_gate["q8_0_mma_admission_reference"] = "launch_q8_mmq_bf16_variant"
    with pytest.raises(AssertionError):
        validate_result(fixture, contract=variant_gate)
    missing_gate = json.loads(json.dumps(_contract()))
    missing_gate.pop("q8_0_mma_admission_reference")
    with pytest.raises(AssertionError):
        validate_result(fixture, contract=missing_gate)


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


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def _nvcc(sources: list[str], output: str) -> list[str]:
    return [
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
        *sources,
        "-o",
        output,
    ]


def _run_quant_and_prompt() -> str:
    _run([*_common(IMAGE), "make", "cuda-products", "cuda-native"])
    quant = _run([*_common(IMAGE), "./build/qw38-cuda-quant-test"])
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    sweep_lines = [
        line
        for line in quant.stdout.splitlines()
        if line.startswith(("q8_mma_case=", "q8_mma_"))
    ]
    SWEEP_RAW.write_text("\n".join(sweep_lines) + "\n")
    assert "status=passed" in quant.stdout
    assert "q8_mma_prompt_tile_2048=" in quant.stdout
    assert "q8_mma_admission_reference=launch_quant_mmq_variant_kQ8_0" in (quant.stdout)
    _run(
        [
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
            "cuda/prompt_mmq_test.cu",
            "build/quant_mmv.cuda.o",
            "build/quant.o",
            "build/status.o",
            "-o",
            "build/qw38-cuda-prompt-mmq-test",
        ]
    )
    prompt_run = _run([*_common(IMAGE), "./build/qw38-cuda-prompt-mmq-test"])
    records = [
        json.loads(line.removeprefix("QW38_PROMPT_MMQ_RESULT="))
        for line in prompt_run.stdout.splitlines()
        if line.startswith("QW38_PROMPT_MMQ_RESULT=")
    ]
    assert len(records) == 1
    record = records[0]
    assert record["status"] == "measured"
    assert record["semantic"]["q8_production_reference_exact"] is True
    assert record["semantic"]["q8_grid_reuses_weight_tiles"] is True
    return quant.stdout


def _run_llama_bench() -> dict[str, Any]:
    command = [
        *_common(LLAMA_IMAGE),
        "bash",
        "-lc",
        "cd /workspace && .cache/authorities/llama-build/bin/llama-bench "
        "-m /workspace/models/Qwen3.8-27B-Q4_K_M.gguf -p 2048 -n 0 --no-warmup "
        "-r 3 -ngl 99 -o json",
    ]
    completed = _run(command)
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


def _diagnostic_objects() -> list[str]:
    return [
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
    ]


def _run_quartz() -> dict[str, Any]:
    _run([*_common(IMAGE), "make", "build/qw38-cuda-timing-test"])
    _run(
        _nvcc(
            ["cuda/prefill_2k_parity_test.cu", *_diagnostic_objects()],
            "build/qw38-cuda-prefill-2k-parity-test",
        )
    )
    run = _run(
        [
            *_common(IMAGE),
            "./build/qw38-cuda-prefill-2k-parity-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ]
    )
    records = [
        json.loads(line.removeprefix(PARITY_PREFIX))
        for line in run.stdout.splitlines()
        if line.startswith(PARITY_PREFIX)
    ]
    assert len(records) == 1
    assert "status=passed" in run.stdout
    QUARTZ_JSON.write_text(json.dumps(records[0], indent=2) + "\n")
    return records[0]


def _run_attribution() -> dict[str, Any]:
    _run([*_common(IMAGE), "make", "build/qw38-cuda-timing-test"])
    _run(
        _nvcc(
            ["cuda/prefill_attribution_test.cu", *_diagnostic_objects()],
            "build/qw38-cuda-prefill-attribution-test",
        )
    )
    run = _run(
        [
            *_common(IMAGE),
            "./build/qw38-cuda-prefill-attribution-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ]
    )
    records = [
        json.loads(line.removeprefix(ATTRIBUTION_PREFIX))
        for line in run.stdout.splitlines()
        if line.startswith(ATTRIBUTION_PREFIX)
    ]
    assert len(records) == 1
    assert "status=passed" in run.stdout
    ATTRIBUTION.write_text(json.dumps(records[0], indent=2) + "\n")
    return records[0]


def _write_report(
    fixture: dict[str, Any], quant_stdout: str, llama: dict[str, Any]
) -> None:
    quartz = fixture["quartz"]
    attribution = fixture["attribution"]
    categories = attribution["categories_ms"]
    tile = fixture["q8_mma_prompt_tile_2048"]
    winner_line = next(
        line
        for line in quant_stdout.splitlines()
        if line.startswith("q8_mma_prompt_tile_2048=")
    )
    speed_line = next(
        line
        for line in quant_stdout.splitlines()
        if line.startswith("q8_mma_speed_2048_12288x5120")
    )
    text = f"""# OPT-017 — mixer Q8_0 MMA MMQ

## Claim labels and proof limits

This increment admits production mixer Q8_0 prompt MMQ MMA. Numeric
**envelopes unloosened**. **OPT-009 Q8_0 reference remains byte-exact** on
`launch_q8_mmq_bf16_variant` versus `launch_q8_mmq_bf16_reference`.
**OPT-016 remains the parity gate owner**. This remasurement is
**not an end-to-end 2K tok/s gate**. There is **no 8K/32K/128K throughput gate**.

Live MMA envelope, J pin, attribution, and remasurement are **Measured**.
llama.cpp Q8_0 MMQ MMA provenance at `{LLAMA_REV}` is **External**.

## Production path

`launch_q8_mmq_bf16` uses MMA when `prompt_rows >= 8` (J={tile}) and the
OPT-009 tiled variant otherwise. Q8_0 is not routed through
`launch_quant_mmq`. Decode `q8_mmv_bf16` is unchanged.

## Component MMA envelope (**Measured**)

CUD-002 versus `launch_quant_mmq_variant` `kQ8_0` (`gpu_staged_*`): max abs
`5e-4`, RMS `2.5e-4`, zero non-finites. Cases: prompt rows 8/9/64/65 on
17×256 and 1024×5120, plus 48×256 tail. Versus unstaged BF16
`launch_q8_mmq_bf16_variant` is informational only and is not a pass/fail
gate.

## J pin (**Measured**)

Mixer-shape sweep at 2048 prompt rows on 12288×5120, 10240×5120, and
5120×6144, 3 CUDA-event replicates, 0 warm-ups:
`{winner_line}`
`{speed_line}`

Raw samples: `q8-mma-j-sweep-raw.txt`.

## Live 2K remasurement (**Measured**, not the OPT-016 gate)

- Quartz mean tok/s: {quartz["mean_tok_s"]} (walls {quartz["wall_ms"]})
- llama.cpp live avg_ts: {llama["avg_ts"]} (`build_commit` {llama.get("build_commit", "cc83d7b")})
- `would_pass_opt016`: {fixture["would_pass_opt016"]} (informational)
- `owns_opt016_parity_gate`: false; `opt016_status`: blocked

## Post-remasurement attribution (**Measured**)

OPT-014 categories on the same GGUF/GPU after the unperturbed samples:

- ffn_mmq {categories["ffn_mmq"]} ms
- gdn {categories["gdn"]} ms
- attention {categories["attention"]} ms
- wall {attribution["wall_ms"]} ms

Mixer Q8_0 time remains inside `gdn` and `attention` until OPT-020.
"""
    REPORT.write_text(text)


def test_opt017_native_mixer_mma() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    if not MODEL.exists():
        pytest.skip("the pinned GGUF is required")

    quant_stdout = _run_quant_and_prompt()
    named = {}
    for line in quant_stdout.splitlines():
        if not line.startswith("q8_mma_case="):
            continue
        fields = dict(part.split("=", 1) for part in line.split())
        named[fields["q8_mma_case"]] = fields
    required = (
        "q8_0_mma_8x17x256",
        "q8_0_mma_9x17x256",
        "q8_0_mma_64x17x256",
        "q8_0_mma_65x17x256",
        "q8_0_mma_8x1024x5120",
        "q8_0_mma_9x1024x5120",
        "q8_0_mma_64x1024x5120",
        "q8_0_mma_65x1024x5120",
        "q8_0_mma_9x48x256",
    )
    assert all(name in named for name in required)
    for case in named.values():
        assert int(case["gpu_staged_nonfinite"]) == 0
        assert float(case["gpu_staged_max_abs"]) <= 5.0e-4
        assert float(case["gpu_staged_rms"]) <= 2.5e-4
    assert "q8_mma_variant_envelope=failed" not in quant_stdout
    speed_line = next(
        line
        for line in quant_stdout.splitlines()
        if line.startswith("q8_mma_speed_2048_12288x5120")
    )
    speed = dict(part.split("=", 1) for part in speed_line.split()[1:])
    assert float(speed["mma_ms"]) < float(speed["variant_ms"])
    winner_line = next(
        line
        for line in quant_stdout.splitlines()
        if line.startswith("q8_mma_prompt_tile_2048=")
    )
    winner_fields = dict(part.split("=", 1) for part in winner_line.split())
    tile = int(winner_fields["q8_mma_prompt_tile_2048"])
    assert tile == int(winner_fields["pinned"])
    assert tile == _contract()["q8_mma_prompt_tile_2048"]

    llama = _run_llama_bench()
    quartz_raw = _run_quartz()
    attribution = _run_attribution()
    mean_tok_s = float(quartz_raw["mean_tok_s"])
    avg_ts = float(llama["avg_ts"])
    fixture = {
        "schema_version": 1,
        "task": "OPT-017",
        "status": "measured",
        "measurement_utc": quartz_raw["measurement_utc"],
        "device": quartz_raw["device"],
        "compute_capability": quartz_raw["compute_capability"],
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "q8_0_production": "mma_for_prompt_rows_ge_8",
        "q8_mma_prompt_tile_2048": tile,
        "owns_opt016_parity_gate": False,
        "opt016_status": "blocked",
        "envelopes_unloosened": True,
        "quartz": {
            "prompt_tokens": 2048,
            "replicates": 3,
            "wall_ms": quartz_raw["wall_ms"],
            "tok_s": quartz_raw["tok_s"],
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
            "test_time": llama.get("test_time", quartz_raw["measurement_utc"]),
        },
        "would_pass_opt016": mean_tok_s >= avg_ts,
        "attribution": attribution,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt017-mixer-q8-mma/REPORT.md",
    }
    validate_result(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    _write_report(fixture, quant_stdout, llama)
