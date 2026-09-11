from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from cuda_test_support import cuda_test_tier

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
CONTRACT = ROOT / "pins/opt053_mmq_pipeline_contract.json"
FIXTURE = ROOT / "fixtures/opt053_mmq_pipeline.json"
OPT052_FIXTURE = ROOT / "fixtures/opt052_gdn_arithmetic.json"
NUMERICS = ROOT / "pins/production_numerics_contract.json"
EVIDENCE = ROOT / "evidence/optimization/opt053-mmq-pipeline"
REPORT = EVIDENCE / "REPORT.md"
REJECTION = EVIDENCE / "REJECTION.md"
AB_RAW = EVIDENCE / "mmq-pipeline-ab-raw.txt"
SASS = EVIDENCE / "mmq-pipeline-sass.txt"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
AB_PREFIX = "QW38_OPT053_MMQ_PIPELINE_AB_RESULT="
P_PREFIX = "QW38_PREFILL_4K_ORACLE_RESULT="
DECODE_PREFIX = "QW38_DECODE_ORACLE_RESULT="
OPT052_P = 2692.64575
OPT052_D128 = 37.5680695
OPT052_D2048 = 35.6793633
OPT052_D128_P95 = 26.8831482
OPT052_D128_RUN_P95 = 26.7299652
OPT052_D2048_P95 = 28.2002869
OPT052_D2048_RUN_P95 = 28.0617561
LEGAL_PATHS = {"off", "fma", "async_y", "fma_async"}
LEGAL_CANDIDATES = ("off", "fma", "async_y", "fma_async")
LIKE_ARITHMETIC = {"off", "async_y"}
PROOF = (
    "OPT-044 production-numerics budgets; like-arithmetic off control remains production quality MMA I=128/J=128; "
    "complete FFN staging plus gate/up/SwiGLU/down cost; explicit FMA scale accumulation and two-stage packed-Y cp.async; "
    "Q4_K min corrections and Q6_K subscales unchanged; synchronous fallback for tails and misalignment; "
    "no unchanged OPT-024 D2R or OPT-028 stream-K rerun; 95% throughput floors versus OPT-052 keep; "
    "105% p95 ceilings versus OPT-052; does not substitute for the 2K llama.cpp parity gate"
)
NVCC_OBJECTS = [
    "build/quant_mmv.cuda.o",
    "build/q4k_decode_dots.cuda.o",
    "build/q8_decode_dots.cuda.o",
    "build/q6k_decode_dots.cuda.o",
    "build/quant.o",
    "build/status.o",
]


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def pin_from_source() -> str:
    text = (ROOT / "cuda/quant_mmq_mma.cuh").read_text()
    match = re.search(r'kSelectedMmqPipelinePath\[\] = "([^"]+)"', text)
    assert match is not None
    return match.group(1)


def frozen_mmq_pins() -> bool:
    text = (ROOT / "cuda/quant_mmq_mma.cuh").read_text()
    return (
        'kSelectedMmqStreamKPath[] = "off"' in text
        and 'kSelectedFfnPath[] = "shared_y_swiglu_q8"' in text
        and 'kSelectedSkinnyMixerPath[] = "mma_i32_j128"' in text
        and 'kSelectedLargeMixerQ8Path[] = "quality_mma"' in text
        and "kSelectedFfnGateQualityI = 128U" in text
        and "kSelectedFfnGatePromptTile = 128U" in text
        and "kSelectedFfnUpQualityI = 128U" in text
        and "kSelectedFfnUpPromptTile = 128U" in text
        and "kSelectedFfnDownQualityI = 128U" in text
        and "kSelectedFfnDownPromptTile = 128U" in text
        and "unsigned int selected_mma_mmq_prompt_tile() noexcept { return 128U; }"
        in text
    )


def pipeline_kernels_present() -> bool:
    text = (ROOT / "cuda/quant_mmq_mma.cuh").read_text()
    return (
        "mmq_cp_async16" in text
        and "cp.async.cg.shared.global" in text
        and "__fmaf_rn" in text
        and "UseFma" in text
        and "UseAsyncY" in text
        and "launch_quant_mmq_mma_y_pipeline" in text
    )


def _like(ident: str) -> bool:
    return ident in LIKE_ARITHMETIC


def _ab_candidate(block: dict[str, Any], ident: str) -> None:
    cand = block["candidates"][ident]
    assert cand["id"] == ident
    assert cand["path"] in LEGAL_PATHS
    assert cand["occupancy"] >= 1
    assert cand["launch_ok"] is True
    assert len(cand["samples"]) in (1, 3, 30)
    mean = sum(float(v) for v in cand["samples"]) / len(cand["samples"])
    assert cand["mean_ms"] == pytest.approx(mean, rel=1e-6, abs=1e-6)
    assert cand["vs_off"]["nonfinite"] == 0
    assert cand["vs_host"]["nonfinite"] == 0
    if ident == "off":
        assert cand["eligible"] is True
        assert cand["extra_shared_bytes"] == 0
    if ident != "off" and cand["eligible"]:
        if _like(ident):
            assert cand["vs_off"]["max_abs"] == 0.0
        else:
            assert cand["vs_off"]["max_abs"] <= 3.0e-4
            assert cand["vs_off"]["rms"] <= 2.0e-4
        assert cand["vs_host"]["ok"] is True
    if ident in {"async_y", "fma_async"} and cand["eligible"]:
        assert cand["extra_shared_bytes"] > 0


def _select_component_winner(block: dict[str, Any], all_eligible: bool = True) -> str:
    if not all_eligible:
        return "off"
    cand = block["candidates"]
    off_mean = float(cand["off"]["mean_ms"])
    if not cand["off"]["eligible"]:
        return "off"
    best = "off"
    best_mean = off_mean
    for ident in LEGAL_CANDIDATES:
        if ident == "off" or not cand[ident]["eligible"]:
            continue
        mean = float(cand[ident]["mean_ms"])
        if mean < best_mean:
            best = ident
            best_mean = mean
    if best != "off" and best_mean < off_mean:
        return best
    return "off"


def _keep_predicates(result: dict[str, Any]) -> bool:
    if result["ab"]["winner"] == "off":
        return False
    if not result["correctness"]["all_eligible"]:
        return False
    if result["keep_sitting_skipped"]:
        return False
    if result["p"] is None or result["d128"] is None or result["d2048"] is None:
        return False
    if float(result["p"]["quartz"]["mean_tok_s"]) <= OPT052_P:
        return False
    if float(result["d128"]["quartz"]["mean_tok_s"]) < 0.95 * OPT052_D128:
        return False
    if float(result["d2048"]["quartz"]["mean_tok_s"]) < 0.95 * OPT052_D2048:
        return False
    if result["d128"]["quartz"]["token_latency_p95_ms"] > 1.05 * OPT052_D128_P95:
        return False
    if (
        result["d128"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * OPT052_D128_RUN_P95
    ):
        return False
    if result["d2048"]["quartz"]["token_latency_p95_ms"] > 1.05 * OPT052_D2048_P95:
        return False
    if (
        result["d2048"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * OPT052_D2048_RUN_P95
    ):
        return False
    return True


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-053"
    assert result["status"] in ("measured", "rejected")
    assert result["status"] not in ("scout", "source_inspected")
    assert contract["opt052_quartz_p_mean_tok_s"] == OPT052_P
    assert contract["opt052_quartz_d128_mean_tok_s"] == OPT052_D128
    assert contract["opt052_quartz_d2048_mean_tok_s"] == OPT052_D2048
    assert contract["owns_opt016_parity_gate"] is False
    assert contract["substitutes_for_opt016"] is False
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["selected_mmq_pipeline_path"] in LEGAL_PATHS
    assert result["selected_mmq_pipeline_path"] == pin_from_source()
    assert result["selected_mmq_stream_k_path"] == "off"
    assert result["selected_ffn_path"] == "shared_y_swiglu_q8"
    assert frozen_mmq_pins()
    assert pipeline_kernels_present()
    ab = result["ab"]
    assert ab["winner"] in LEGAL_CANDIDATES
    assert set(ab["candidates"]) == set(LEGAL_CANDIDATES)
    for ident in LEGAL_CANDIDATES:
        _ab_candidate(ab, ident)
    expected = _select_component_winner(ab, result["correctness"]["all_eligible"])
    assert ab["winner"] == expected
    assert ab["win"] is (expected != "off")
    if expected == "off":
        assert result["keep_sitting_skipped"] is True
        assert result["reverted"] is False
        assert result["status"] == "measured"
        assert result["selected_mmq_pipeline_path"] == "off"
        assert result["p"] is not None
        assert result["p"]["quartz"]["mean_tok_s"] == OPT052_P
        assert result["d128"]["quartz"]["mean_tok_s"] == OPT052_D128
        assert result["d2048"]["quartz"]["mean_tok_s"] == OPT052_D2048
        assert not REJECTION.is_file()
    else:
        assert result["selected_mmq_pipeline_path"] == expected
        if result["status"] == "measured":
            assert result["reverted"] is False
            assert _keep_predicates(result)
            assert not REJECTION.is_file()
        else:
            assert result["reverted"] is True
            assert REJECTION.is_file()
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    assert result["proof_limit"] == PROOF
    assert (
        result["report_path"] == "evidence/optimization/opt053-mmq-pipeline/REPORT.md"
    )
    assert REPORT.is_file()
    report = REPORT.read_text()
    for phrase in contract["proof_limit"]:
        assert phrase in report
    assert AB_RAW.is_file()
    opt052 = json.loads(OPT052_FIXTURE.read_text())
    assert opt052["p"]["quartz"]["mean_tok_s"] == OPT052_P
    assert result["instruction_mix"]["fma_source"] is True
    assert result["instruction_mix"]["cp_async_source"] is True
    assert isinstance(result["extra_workspace_bytes"], int)
    assert result["extra_workspace_bytes"] >= 0


def _timed(mean: float, ident: str) -> dict[str, Any]:
    like = _like(ident)
    extra = 18432 if ident in {"async_y", "fma_async"} else 0
    return {
        "mean_ms": mean,
        "launch_ok": True,
        "warmup_ms": [mean, mean, mean],
        "samples": [mean] * 30,
        "occupancy": 2,
        "extra_shared_bytes": extra,
        "eligible": True,
        "timed_nonfinite": 0,
        "uses_fma": ident in {"fma", "fma_async"},
        "uses_async": ident in {"async_y", "fma_async"},
        "like_arithmetic": like,
        "vs_off": {
            "max_abs": 0.0 if like else 1.0e-7,
            "rms": 0.0 if like else 1.0e-8,
            "nonfinite": 0,
        },
        "vs_host": {
            "max_abs": 0.01,
            "rms": 0.001,
            "nonfinite": 0,
            "bad": 0,
            "ok": True,
        },
        "mixer_wide_ms": mean,
        "mixer_skinny_ms": mean * 0.1,
        "mixer_wide_occupancy": 2,
        "mixer_skinny_occupancy": 2,
    }


def _ab_stub(winner: str, means: dict[str, float]) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for ident in LEGAL_CANDIDATES:
        slot = _timed(means.get(ident, 8.0), ident)
        slot["id"] = ident
        slot["path"] = ident
        candidates[ident] = slot
    return {
        "token_count": 4096,
        "winner": winner,
        "win": winner != "off",
        "hidden": 5120,
        "ffn": 17408,
        "candidates": candidates,
        "off_tokens_ms": {"512": 1.0, "1024": 2.0, "2048": 4.0},
    }


def _opt052_pd() -> dict[str, Any]:
    opt052 = json.loads(OPT052_FIXTURE.read_text())
    return {
        "p": opt052["p"],
        "d128": opt052["d128"],
        "d2048": opt052["d2048"],
    }


def _no_change_fixture() -> dict[str, Any]:
    copied = _opt052_pd()
    return {
        "schema_version": 1,
        "task": "OPT-053",
        "status": "measured",
        "measurement_utc": "2026-09-10T00:00:00Z",
        "device": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": False,
        "selected_mmq_pipeline_path": "off",
        "selected_mmq_stream_k_path": "off",
        "selected_ffn_path": "shared_y_swiglu_q8",
        "ab": _ab_stub(
            "off", {ident: 8.0 + 0.1 * i for i, ident in enumerate(LEGAL_CANDIDATES)}
        ),
        "correctness": {
            "all_eligible": True,
            "tails_ok": True,
            "case_count": 8,
            "token_counts": [17, 65, 129, 255],
        },
        "production_numerics": {
            "formula": "max(strict_reference_ceiling, 1.05 * measured_llama_error + 1e-6)",
            "budget": {
                "max_abs": 0.0003,
                "rms": 0.0002,
                "one_minus_cosine": 1e-06,
            },
        },
        "keep_sitting_skipped": True,
        "p": copied["p"],
        "d128": copied["d128"],
        "d2048": copied["d2048"],
        "instruction_mix": {
            "fma_source": True,
            "cp_async_source": True,
            "cp_async_sass": "pending",
        },
        "extra_workspace_bytes": 18432,
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt053-mmq-pipeline/REPORT.md",
    }


def test_opt053_contract_and_source_pins() -> None:
    contract = _contract()
    opt052 = json.loads(OPT052_FIXTURE.read_text())
    assert opt052["p"]["quartz"]["mean_tok_s"] == OPT052_P
    assert opt052["d128"]["quartz"]["mean_tok_s"] == OPT052_D128
    assert opt052["d2048"]["quartz"]["mean_tok_s"] == OPT052_D2048
    assert contract["opt052_quartz_p_mean_tok_s"] == OPT052_P
    assert contract["opt052_quartz_d128_mean_tok_s"] == OPT052_D128
    assert contract["opt052_quartz_d2048_mean_tok_s"] == OPT052_D2048
    assert contract["ab"]["candidates"] == list(LEGAL_CANDIDATES)
    assert contract["ab"]["ties_retain"] == "off"
    assert contract["owns_opt016_parity_gate"] is False
    numerics = json.loads(NUMERICS.read_text())
    assert (
        numerics["strict_reference_ceilings"]["cud001_maximum_absolute_error"] == 0.0003
    )
    assert pin_from_source() in LEGAL_PATHS
    assert frozen_mmq_pins()
    assert pipeline_kernels_present()
    header = (ROOT / "cuda/quant_mmv.h").read_text()
    assert "launch_quant_mmq_mma_y_pipeline" in header
    assert "selected_mmq_pipeline_path" in header
    makefile = (ROOT / "Makefile").read_text()
    assert "--fmad=false" in makefile
    assert "opt053_mmq_pipeline_ab_test" not in makefile


def test_opt053_validator_rejects_inadmissible_evidence() -> None:
    fixture = _no_change_fixture()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    report_prev = REPORT.read_text() if REPORT.is_file() else None
    raw_prev = AB_RAW.read_text() if AB_RAW.is_file() else None
    if report_prev is None:
        REPORT.write_text(PROOF + "\n")
    if raw_prev is None:
        AB_RAW.write_text("placeholder\n")
    try:
        mutations: list[dict[str, Any]] = []
        for mutate in (
            lambda x: x.__setitem__("substitutes_for_opt016", True),
            lambda x: x.__setitem__("owns_opt016_parity_gate", True),
            lambda x: x.__setitem__("nsight_systems", "/tmp/capture.nsys-rep"),
            lambda x: x.__setitem__("status", "source_inspected"),
            lambda x: x.__setitem__("task", "OPT-037"),
            lambda x: x["ab"].__setitem__("winner", "fma"),
            lambda x: x.__setitem__("selected_mmq_pipeline_path", "fma"),
            lambda x: x.__setitem__("reverted", True),
        ):
            changed = json.loads(json.dumps(fixture))
            mutate(changed)
            mutations.append(changed)
        for mutation in mutations:
            with pytest.raises(AssertionError):
                validate_result(mutation)
    finally:
        if report_prev is None:
            REPORT.unlink(missing_ok=True)
        else:
            REPORT.write_text(report_prev)
        if raw_prev is None:
            AB_RAW.unlink(missing_ok=True)
        else:
            AB_RAW.write_text(raw_prev)


def test_opt053_fixture_connected() -> None:
    if not FIXTURE.is_file():
        pytest.skip("OPT-053 fixture is written by the exclusive CUDA sitting")
    validate_result(json.loads(FIXTURE.read_text()))


def _common(image: str, tier: str | None = None) -> list[str]:
    command = [
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
    ]
    if tier is not None:
        command.extend(["-e", f"QW38_CUDA_TEST_TIER={tier}"])
    command.append(image)
    return command


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def _nvcc(source: str, output: str, extra: list[str] | None = None) -> list[str]:
    objects = extra if extra is not None else NVCC_OBJECTS
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
        "-DQW38_CUDA_RUNTIME",
        source,
        *objects,
        "-o",
        output,
    ]


def _parse_prefixed(text: str, prefix: str) -> dict[str, Any]:
    for line in text.splitlines():
        if line.startswith(prefix):
            return json.loads(line.removeprefix(prefix))
    raise AssertionError(f"{prefix} was not found\n{text}")


def _set_pin(path: str) -> None:
    source = ROOT / "cuda/quant_mmq_mma.cuh"
    text = source.read_text()
    text = re.sub(
        r'kSelectedMmqPipelinePath\[\] = "[^"]+"',
        f'kSelectedMmqPipelinePath[] = "{path}"',
        text,
    )
    source.write_text(text)
    contract = _contract()
    contract["selected_mmq_pipeline_path"] = path
    CONTRACT.write_text(json.dumps(contract, indent=2) + "\n")


def _write_report(fixture: dict[str, Any]) -> None:
    decision = (
        "keep"
        if fixture["ab"]["winner"] != "off" and fixture["status"] == "measured"
        else "no-change"
    )
    if fixture["status"] == "rejected":
        decision = "reject"
    ab = fixture["ab"]
    sitting = "skipped" if fixture["keep_sitting_skipped"] else "ran"
    quartz_p = fixture["p"]["quartz"]["mean_tok_s"] if fixture["p"] else "n/a"
    quartz_d128 = fixture["d128"]["quartz"]["mean_tok_s"] if fixture["d128"] else "n/a"
    quartz_d2048 = (
        fixture["d2048"]["quartz"]["mean_tok_s"] if fixture["d2048"] else "n/a"
    )
    means = ", ".join(
        f"{ident} {ab['candidates'][ident]['mean_ms']} ms" for ident in LEGAL_CANDIDATES
    )
    text = f"""# OPT-053 — Optimize MMQ scaling and tile staging

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of explicit FMA scale accumulation and two-stage
packed-Y cp.async against production quality MMA. Keep requires
**OPT-044 production-numerics budgets**, **like-arithmetic off control remains production quality MMA I=128/J=128**,
**complete FFN staging plus gate/up/SwiGLU/down cost**, **explicit FMA scale accumulation and two-stage packed-Y cp.async**,
**Q4_K min corrections and Q6_K subscales unchanged**, **synchronous fallback for tails and misalignment**,
**no unchanged OPT-024 D2R or OPT-028 stream-K rerun**, **95% throughput floors versus OPT-052 keep**, **105% p95 ceilings versus OPT-052**, and
**does not substitute for the 2K llama.cpp parity gate**. Copied denominators
are P {OPT052_P}, D128 {OPT052_D128}, D2048 {OPT052_D2048}.

## Decision

**{decision}** — `reverted`={json.dumps(fixture["reverted"])};
`keep_sitting_skipped`={json.dumps(fixture["keep_sitting_skipped"])};
selected_mmq_pipeline_path={fixture["selected_mmq_pipeline_path"]};
4096 complete-FFN A/B winner {ab["winner"]} ({means});
tok/s sitting {sitting}.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `{GGUF_SHA}` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp revision | `{LLAMA_REV}` |
| A/B | complete FFN staging+gate/up/SwiGLU/down, candidates {list(LEGAL_CANDIDATES)} |
| Correctness | tails {contract_tokens()} |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: {fixture["device"]} compute {fixture["compute_capability"]}
- measurement_utc: {fixture["measurement_utc"]}
- extra_workspace_bytes: {fixture["extra_workspace_bytes"]}
- P Quartz mean tok/s: {quartz_p} versus OPT-052 {OPT052_P}
- D128 Quartz mean tok/s: {quartz_d128} versus OPT-052 {OPT052_D128}
- D2048 Quartz mean tok/s: {quartz_d2048} versus OPT-052 {OPT052_D2048}
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false
"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def contract_tokens() -> list[int]:
    return [17, 65, 129, 255]


def _write_rejection(fixture: dict[str, Any]) -> None:
    ab = fixture["ab"]
    text = f"""# OPT-053 rejection

Keep sitting failed P improvement or the cross-workload guard after a
component A/B win. Production MMQ pipeline remains off.

- 4096 A/B winner: {ab["winner"]}
- keep_sitting_skipped: {json.dumps(fixture["keep_sitting_skipped"])}
- measurement_utc: {fixture["measurement_utc"]}
"""
    REJECTION.write_text(text)


def _run_ab(tier: str) -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    commands = [
        [
            *_common(IMAGE),
            "make",
            "build/quant_mmv.cuda.o",
            "build/q4k_decode_dots.cuda.o",
            "build/q8_decode_dots.cuda.o",
            "build/q6k_decode_dots.cuda.o",
            "build/quant.o",
            "build/status.o",
        ],
        _nvcc(
            "cuda/opt053_mmq_pipeline_ab_test.cu",
            "build/qw38-cuda-opt053-mmq-pipeline-ab-test",
        ),
        [
            *_common(IMAGE, tier),
            "./build/qw38-cuda-opt053-mmq-pipeline-ab-test",
            "evidence/optimization/opt053-mmq-pipeline/mmq-pipeline-ab-raw.txt",
        ],
    ]
    outputs: list[str] = []
    for command in commands:
        outputs.append(_run(command).stdout)
    record = _parse_prefixed(outputs[-1], AB_PREFIX)
    assert "status=passed" in outputs[-1]
    sass = _run(
        [
            *_common(IMAGE),
            "cuobjdump",
            "-sass",
            "build/qw38-cuda-opt053-mmq-pipeline-ab-test",
        ]
    )
    SASS.write_text(sass.stdout)
    record.setdefault("instruction_mix", {})
    record["instruction_mix"]["fma_source"] = True
    record["instruction_mix"]["cp_async_source"] = True
    record["instruction_mix"]["cp_async_sass"] = (
        "present"
        if (
            "CP.ASYNC" in sass.stdout
            or "cp.async" in sass.stdout.lower()
            or "LDGSTS" in sass.stdout
        )
        else "absent"
    )
    record["instruction_mix"]["ffma_sass"] = (
        "present" if "FFMA" in sass.stdout else "absent"
    )
    return record


def _run_quartz_p() -> dict[str, Any]:
    commands = [
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
            "-DQW38_DIAGNOSTIC_TRACE",
            "cuda/prefill_4k_oracle_test.cu",
            "build/full_scheduler.trace.cuda.o",
            "build/scheduler_primitives.cuda.o",
            "build/quant_mmv.cuda.o",
            "build/q4k_decode_dots.cuda.o",
            "build/q8_decode_dots.cuda.o",
            "build/q6k_decode_dots.cuda.o",
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
        ],
        [
            *_common(IMAGE),
            "./build/qw38-cuda-prefill-4k-oracle-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ],
    ]
    outputs: list[str] = []
    for command in commands:
        outputs.append(_run(command).stdout)
    record = _parse_prefixed(outputs[-1], P_PREFIX)
    (EVIDENCE / "quartz-p.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _run_quartz_decode(prefix: int) -> dict[str, Any]:
    binary = "build/qw38-cuda-decode-oracle-test"
    commands = [
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
            "-DQW38_DIAGNOSTIC_TRACE",
            "cuda/decode_oracle_test.cu",
            "build/full_scheduler.trace.cuda.o",
            "build/scheduler_primitives.cuda.o",
            "build/quant_mmv.cuda.o",
            "build/q4k_decode_dots.cuda.o",
            "build/q8_decode_dots.cuda.o",
            "build/q6k_decode_dots.cuda.o",
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
            binary,
        ],
        [*_common(IMAGE), f"./{binary}", "models/Qwen3.8-27B-Q4_K_M.gguf", str(prefix)],
    ]
    outputs: list[str] = []
    for command in commands:
        outputs.append(_run(command).stdout)
    record = _parse_prefixed(outputs[-1], DECODE_PREFIX)
    (EVIDENCE / f"quartz-d{prefix}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return record


def _engine_block(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "prefix": record["prefix"],
        "decode_tokens": record["decode_tokens"],
        "warmups": record["warmups"],
        "runs": record["runs"],
        "warmup_tok_s": record["warmup_tok_s"],
        "tok_s": record["tok_s"],
        "run_wall_ms": record["run_wall_ms"],
        "mean_tok_s": record["mean_tok_s"],
        "token_latency_p50_ms": record["token_latency_p50_ms"],
        "token_latency_p95_ms": record["token_latency_p95_ms"],
        "run_mean_token_latency_p95_ms": record["run_mean_token_latency_p95_ms"],
        "graphs_created": record.get("graphs_created", True),
        "attribution": record.get("attribution"),
        "cache_policy": record.get("cache_policy", "disabled"),
    }


def _fixture_from_ab(ab: dict[str, Any]) -> dict[str, Any]:
    winner = ab["ab"]["winner"] if "ab" in ab else ab["winner"]
    copied = _opt052_pd()
    return {
        "schema_version": 1,
        "task": "OPT-053",
        "status": "measured",
        "measurement_utc": ab["measurement_utc"],
        "device": ab["device"],
        "compute_capability": ab["compute_capability"],
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": False,
        "selected_mmq_pipeline_path": "off" if winner == "off" else winner,
        "selected_mmq_stream_k_path": "off",
        "selected_ffn_path": "shared_y_swiglu_q8",
        "ab": ab["ab"] if "ab" in ab else ab,
        "correctness": ab["correctness"],
        "production_numerics": {
            "formula": "max(strict_reference_ceiling, 1.05 * measured_llama_error + 1e-6)",
            "budget": {
                "max_abs": 0.0003,
                "rms": 0.0002,
                "one_minus_cosine": 1e-06,
            },
        },
        "keep_sitting_skipped": winner == "off",
        "p": copied["p"] if winner == "off" else None,
        "d128": copied["d128"] if winner == "off" else None,
        "d2048": copied["d2048"] if winner == "off" else None,
        "instruction_mix": ab.get(
            "instruction_mix",
            {"fma_source": True, "cp_async_source": True, "cp_async_sass": "pending"},
        ),
        "extra_workspace_bytes": int(ab.get("extra_workspace_bytes", 0)),
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt053-mmq-pipeline/REPORT.md",
        "_install": "off" if winner == "off" else winner,
    }


@pytest.mark.skipif(
    os.environ.get("QW38_RUN_CUDA_TESTS") != "1",
    reason="set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate",
)
def test_opt053_exclusive_cuda_sitting() -> None:
    tier = cuda_test_tier()
    if not MODEL.exists() and tier != "smoke":
        pytest.skip("the pinned GGUF is required")
    if FIXTURE.is_file():
        existing = json.loads(FIXTURE.read_text())
        if existing.get("status") in ("measured", "rejected"):
            validate_result(existing)
            return

    ab = _run_ab(tier)
    closed = subprocess.run(
        [*_common(IMAGE), "./build/qw38-cuda-opt053-mmq-pipeline-ab-test"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert closed.returncode != 0
    assert "QW38_CUDA_TEST_TIER must be set" in (closed.stdout + closed.stderr)
    if tier != "acceptance":
        assert ab["task"] == "OPT-053"
        winner = ab["ab"]["winner"] if "ab" in ab else ab["winner"]
        assert winner in LEGAL_CANDIDATES
        return

    wrapper = _fixture_from_ab(ab)
    selected = wrapper.pop("_install")
    if selected == "off":
        _set_pin("off")
        wrapper["selected_mmq_pipeline_path"] = "off"
        _write_report(wrapper)
        FIXTURE.write_text(json.dumps(wrapper, indent=2) + "\n")
        validate_result(wrapper)
        return

    _set_pin(selected)
    _run(
        [
            *_common(IMAGE),
            "make",
            "build/quant_mmv.cuda.o",
            "build/full_scheduler.trace.cuda.o",
            "build/full_scheduler.cuda.o",
        ]
    )
    quartz_p = _run_quartz_p()
    quartz_d128 = _run_quartz_decode(128)
    quartz_d2048 = _run_quartz_decode(2048)
    fixture = wrapper
    fixture["keep_sitting_skipped"] = False
    fixture["reverted"] = False
    fixture["selected_mmq_pipeline_path"] = selected
    fixture["p"] = {
        "quartz": {
            "prompt_tokens": quartz_p.get("prompt_tokens", 4096),
            "replicates": quartz_p.get("replicates", 3),
            "wall_ms": quartz_p.get("wall_ms"),
            "tok_s": quartz_p.get("tok_s"),
            "mean_tok_s": quartz_p["mean_tok_s"],
            "cold": quartz_p.get("cold", True),
            "cache_policy": quartz_p.get("cache_policy", "disabled"),
            "attribution": None,
        }
    }
    fixture["d128"] = {"quartz": _engine_block(quartz_d128)}
    fixture["d2048"] = {"quartz": _engine_block(quartz_d2048)}
    if not _keep_predicates(fixture):
        _set_pin("off")
        fixture["status"] = "rejected"
        fixture["reverted"] = True
        fixture["selected_mmq_pipeline_path"] = "off"
        fixture["keep_sitting_skipped"] = False
        _write_rejection(fixture)
    else:
        fixture["status"] = "measured"
        if REJECTION.is_file():
            REJECTION.unlink()
    _write_report(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    validate_result(fixture)
