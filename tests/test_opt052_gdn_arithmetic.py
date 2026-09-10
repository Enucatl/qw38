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
CONTRACT = ROOT / "pins/opt052_gdn_arithmetic_contract.json"
FIXTURE = ROOT / "fixtures/opt052_gdn_arithmetic.json"
OPT051_FIXTURE = ROOT / "fixtures/opt051_attention_pipeline.json"
NUMERICS = ROOT / "pins/production_numerics_contract.json"
EVIDENCE = ROOT / "evidence/optimization/opt052-gdn-arithmetic"
REPORT = EVIDENCE / "REPORT.md"
REJECTION = EVIDENCE / "REJECTION.md"
AB_RAW = EVIDENCE / "gdn-arithmetic-ab-raw.txt"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
AB_PREFIX = "QW38_OPT052_GDN_ARITHMETIC_AB_RESULT="
P_PREFIX = "QW38_PREFILL_4K_ORACLE_RESULT="
DECODE_PREFIX = "QW38_DECODE_ORACLE_RESULT="
OPT051_P = 2489.33008
OPT051_D128 = 37.6655884
OPT051_D2048 = 35.7582932
OPT051_D128_P95 = 26.8204556
OPT051_D128_RUN_P95 = 26.703701
OPT051_D2048_P95 = 28.0988483
OPT051_D2048_RUN_P95 = 27.979435
LEGAL_PATHS = {"off", "preproc", "preproc_fma", "approx_exp", "transpose"}
LEGAL_CANDIDATES = (
    "shared",
    "preproc",
    "preproc_fma",
    "approx_exp",
    "transpose",
)
LIKE_ARITHMETIC = {"shared", "preproc", "transpose"}
PROOF = (
    "OPT-044 production-numerics budgets; like-arithmetic shared control remains OPT-040 hoisted inverses; "
    "conv+preprocessing+recurrence+gated complete cost; hoisted scaled Q/K and decay; "
    "admitted FMA and approximate exp as separate variants; optional conversion-inclusive state transpose; "
    "nonzero incoming state and outer-chunk restore; same-path transaction/restore exactness; "
    "95% throughput floors versus OPT-051 keep; 105% p95 ceilings versus OPT-051; "
    "does not substitute for the 2K llama.cpp parity gate"
)
NVCC_OBJECTS = ["build/gdn_step.cuda.o"]


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def pin_from_source() -> str:
    text = (ROOT / "cuda/gdn_fused_quality.cuh").read_text()
    match = re.search(r'kSelectedGdnPreprocPath\[\] = "([^"]+)"', text)
    assert match is not None
    return match.group(1)


def frozen_gdn_pins() -> bool:
    text = (ROOT / "cuda/gdn_fused_quality.cuh").read_text()
    return (
        'kSelectedGdnFusePath[] = "off"' in text
        and 'kSelectedGdnInversePath[] = "shared"' in text
        and "rsqrtf(" not in text
        and "rsqrtf_rn" not in text
    )


def decode_gdn_unchanged() -> bool:
    text = (ROOT / "cuda/full_scheduler.cu").read_text()
    return (
        "launch_gdn_prepare_tiled(" in text
        and "launch_gdn_gated_output(" in text
        and "GdnScanPath::kFusedTokenLoop" in text
    )


def preproc_kernels_present() -> bool:
    text = (ROOT / "cuda/gdn_fused_quality.cuh").read_text()
    return (
        "prepare_gdn_scaled_qk" in text
        and "prepare_gdn_decay" in text
        and "prepare_recurrence_fused_warp_column_preproc" in text
        and "gdn_transpose_state_to_column_major" in text
        and "__fmaf_rn" in text
        and "__expf" in text
    )


def _like(ident: str) -> bool:
    return ident in LIKE_ARITHMETIC


def _ab_candidate(block: dict[str, Any], ident: str) -> None:
    cand = block["candidates"][ident]
    assert cand["id"] == ident
    assert cand["path"] in LEGAL_CANDIDATES
    assert cand["occupancy"] >= 1
    assert cand["recurrence_occupancy"] >= 1
    assert cand["launch_ok"] is True
    assert len(cand["samples"]) in (1, 3, 30)
    mean = sum(float(v) for v in cand["samples"]) / len(cand["samples"])
    assert cand["mean_ms"] == pytest.approx(mean, rel=1e-6, abs=1e-6)
    assert cand["vs_shared"]["nonfinite"] == 0
    assert cand["vs_sequential"]["nonfinite"] == 0
    if ident == "shared":
        assert cand["eligible"] is True
    if ident != "shared" and cand["eligible"]:
        assert cand["prepare_preserves_committed"] is True
        assert cand["cancel_isolates"] is True
        assert cand["restore_exact"] is True
        if _like(ident):
            assert cand["recurrent_byte_equal"] is True
            assert cand["state_byte_equal"] is True
            assert cand["vs_sequential"]["max_abs"] <= 5.0e-8
        else:
            assert cand["vs_shared"]["max_abs"] <= 3.0e-4
            assert cand["vs_shared"]["rms"] <= 2.0e-4


def _select_component_winner(block: dict[str, Any], all_eligible: bool = True) -> str:
    if not all_eligible:
        return "shared"
    cand = block["candidates"]
    shared_mean = float(cand["shared"]["mean_ms"])
    if not cand["shared"]["eligible"]:
        return "shared"
    best = "shared"
    best_mean = shared_mean
    like_best = None
    like_mean = shared_mean
    for ident in LEGAL_CANDIDATES:
        if ident == "shared":
            continue
        if not cand[ident]["eligible"]:
            continue
        mean = float(cand[ident]["mean_ms"])
        if not (mean < shared_mean):
            continue
        if mean < best_mean:
            best = ident
            best_mean = mean
        if _like(ident) and (like_best is None or mean < like_mean):
            like_best = ident
            like_mean = mean
    if best == "shared":
        return "shared"
    if like_best is not None and like_mean <= 1.01 * best_mean:
        return like_best
    return best


def _keep_predicates(result: dict[str, Any]) -> bool:
    if result["ab"]["winner"] == "shared":
        return False
    if result["selected_gdn_preproc_path"] in {"off", "shared"}:
        return False
    if not result["correctness"]["all_eligible"]:
        return False
    if result["keep_sitting_skipped"]:
        return False
    if result["p"] is None or result["d128"] is None or result["d2048"] is None:
        return False
    if float(result["p"]["quartz"]["mean_tok_s"]) <= OPT051_P:
        return False
    if float(result["d128"]["quartz"]["mean_tok_s"]) < 0.95 * OPT051_D128:
        return False
    if float(result["d2048"]["quartz"]["mean_tok_s"]) < 0.95 * OPT051_D2048:
        return False
    if result["d128"]["quartz"]["token_latency_p95_ms"] > 1.05 * OPT051_D128_P95:
        return False
    if (
        result["d128"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * OPT051_D128_RUN_P95
    ):
        return False
    if result["d2048"]["quartz"]["token_latency_p95_ms"] > 1.05 * OPT051_D2048_P95:
        return False
    if (
        result["d2048"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * OPT051_D2048_RUN_P95
    ):
        return False
    return True


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-052"
    assert result["status"] in ("measured", "rejected")
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    path = result["selected_gdn_preproc_path"]
    assert path in LEGAL_PATHS
    assert result["selected_gdn_inverse_path"] == "shared"
    assert contract["ab"]["candidates"] == list(LEGAL_CANDIDATES)
    assert contract["ab"]["ties_retain"] == "shared"
    assert (
        _select_component_winner(
            result["ab"], bool(result["correctness"]["all_eligible"])
        )
        == result["ab"]["winner"]
    )
    for ident in LEGAL_CANDIDATES:
        _ab_candidate(result["ab"], ident)
    assert frozen_gdn_pins()
    assert decode_gdn_unchanged()
    assert preproc_kernels_present()
    keep = _keep_predicates(result)
    if result["status"] == "measured":
        assert pin_from_source() == path == result["ab"]["winner"]
        assert keep is True
        assert result["reverted"] is False
        assert REPORT.is_file()
        report = REPORT.read_text()
        for phrase in contract["proof_limit"]:
            assert phrase in report
    else:
        assert path == "off"
        assert keep is False
        assert result["reverted"] is True
        assert REJECTION.is_file()
    assert AB_RAW.is_file()
    assert result["report_path"] == contract["report_path"]
    for phrase in contract["proof_limit"]:
        assert phrase in result["proof_limit"]
        assert phrase in PROOF


def _timed(mean: float, ident: str) -> dict[str, Any]:
    like = _like(ident)
    return {
        "mean_ms": mean,
        "launch_ok": True,
        "warmup_ms": [mean, mean, mean],
        "samples": [mean] * 30,
        "occupancy": 2,
        "recurrence_occupancy": 2,
        "eligible": True,
        "dispatched_preproc": ident != "shared",
        "recurrent_byte_equal": like,
        "conv_byte_equal": like,
        "state_byte_equal": like,
        "gated_byte_equal": like,
        "prepare_preserves_committed": True,
        "cancel_isolates": True,
        "restore_exact": True,
        "chunk_ok": True,
        "vs_shared": {
            "max_abs": 0.0 if like else 1.0e-7,
            "rms": 0.0 if like else 1.0e-8,
            "nonfinite": 0,
        },
        "vs_sequential": {
            "max_abs": 1.0e-9 if like else 1.0e-7,
            "rms": 1.0e-10 if like else 1.0e-8,
            "nonfinite": 0,
        },
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
        "win": winner != "shared",
        "candidates": candidates,
    }


def _reject_fixture() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-052",
        "status": "rejected",
        "measurement_utc": "2026-09-10T00:00:00Z",
        "device": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": True,
        "selected_gdn_preproc_path": "off",
        "selected_gdn_inverse_path": "shared",
        "ab": _ab_stub("shared", {ident: 8.0 for ident in LEGAL_CANDIDATES}),
        "correctness": {
            "all_eligible": True,
            "chunk_ok": True,
            "case_count": 30,
            "token_counts": [1, 2, 3, 4, 63, 64, 65, 512, 2048, 4096],
            "layers": [0, 32, 62],
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
        "p": None,
        "d128": None,
        "d2048": None,
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt052-gdn-arithmetic/REPORT.md",
    }


def test_opt052_contract_and_source_pins() -> None:
    contract = _contract()
    opt051 = json.loads(OPT051_FIXTURE.read_text())
    assert opt051["p"]["quartz"]["mean_tok_s"] == OPT051_P
    assert opt051["d128"]["quartz"]["mean_tok_s"] == OPT051_D128
    assert opt051["d2048"]["quartz"]["mean_tok_s"] == OPT051_D2048
    assert contract["opt051_quartz_p_mean_tok_s"] == OPT051_P
    assert contract["opt051_quartz_d128_mean_tok_s"] == OPT051_D128
    assert contract["opt051_quartz_d2048_mean_tok_s"] == OPT051_D2048
    assert contract["ab"]["candidates"] == list(LEGAL_CANDIDATES)
    assert contract["ab"]["ties_retain"] == "shared"
    assert contract["owns_opt016_parity_gate"] is False
    numerics = json.loads(NUMERICS.read_text())
    assert (
        numerics["strict_reference_ceilings"]["cud001_maximum_absolute_error"] == 0.0003
    )
    assert pin_from_source() in LEGAL_PATHS
    assert frozen_gdn_pins()
    assert decode_gdn_unchanged()
    assert preproc_kernels_present()
    header = (ROOT / "cuda/gdn_step.h").read_text()
    assert "selected_gdn_preproc_path" in header
    assert "gdn_preproc_floats" in header
    assert "preproc_path" in header
    makefile = (ROOT / "Makefile").read_text()
    assert "--fmad=false" in makefile
    assert "opt052_gdn_arithmetic_ab_test" not in makefile


def test_opt052_validator_rejects_inadmissible_evidence() -> None:
    fixture = _reject_fixture()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    report_prev = REPORT.read_text() if REPORT.is_file() else None
    raw_prev = AB_RAW.read_text() if AB_RAW.is_file() else None
    rejection_prev = REJECTION.read_text() if REJECTION.is_file() else None
    if report_prev is None:
        REPORT.write_text(PROOF + "\n")
    if raw_prev is None:
        AB_RAW.write_text("placeholder\n")
    if rejection_prev is None:
        REJECTION.write_text("rejected placeholder shared\n")
    try:
        mutations: list[dict[str, Any]] = []
        for mutate in (
            lambda x: x.__setitem__("substitutes_for_opt016", True),
            lambda x: x.__setitem__("owns_opt016_parity_gate", True),
            lambda x: x.__setitem__("nsight_systems", "/tmp/capture.nsys-rep"),
            lambda x: x.__setitem__("status", "source_inspected"),
            lambda x: x.__setitem__("task", "OPT-040"),
            lambda x: x["ab"].__setitem__("winner", "preproc"),
            lambda x: x.__setitem__("selected_gdn_preproc_path", "preproc"),
            lambda x: x.__setitem__("reverted", False),
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
        if rejection_prev is None:
            REJECTION.unlink(missing_ok=True)
        else:
            REJECTION.write_text(rejection_prev)


def test_opt052_fixture_connected() -> None:
    if not FIXTURE.is_file():
        pytest.skip("OPT-052 fixture is written by the exclusive CUDA sitting")
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
    source = ROOT / "cuda/gdn_fused_quality.cuh"
    text = source.read_text()
    text = re.sub(
        r'kSelectedGdnPreprocPath\[\] = "[^"]+"',
        f'kSelectedGdnPreprocPath[] = "{path}"',
        text,
    )
    source.write_text(text)
    contract = _contract()
    contract["selected_gdn_preproc_path"] = path
    CONTRACT.write_text(json.dumps(contract, indent=2) + "\n")


def _write_report(fixture: dict[str, Any]) -> None:
    decision = "keep" if fixture["status"] == "measured" else "reject"
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
    text = f"""# OPT-052 — Remove redundant GDN arithmetic and state traffic

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of hoisted scaled Q/K and decay against the
OPT-040 shared-inverse warp-column loop. Keep requires
**OPT-044 production-numerics budgets**, **like-arithmetic shared control remains OPT-040 hoisted inverses**,
**conv+preprocessing+recurrence+gated complete cost**, **hoisted scaled Q/K and decay**,
**admitted FMA and approximate exp as separate variants**, **optional conversion-inclusive state transpose**,
**nonzero incoming state and outer-chunk restore**, **same-path transaction/restore exactness**,
**95% throughput floors versus OPT-051 keep**, **105% p95 ceilings versus OPT-051**, and
**does not substitute for the 2K llama.cpp parity gate**. Copied denominators
are P {OPT051_P}, D128 {OPT051_D128}, D2048 {OPT051_D2048}.

## Decision

**{decision}** — `reverted`={json.dumps(fixture["reverted"])};
`keep_sitting_skipped`={json.dumps(fixture["keep_sitting_skipped"])};
selected_gdn_preproc_path={fixture["selected_gdn_preproc_path"]};
4096 A/B winner {ab["winner"]} ({means});
tok/s sitting {sitting}.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `{GGUF_SHA}` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp revision | `{LLAMA_REV}` |
| A/B | complete conv+preproc+recurrence+gated, candidates {list(LEGAL_CANDIDATES)} |
| Correctness | tokens {contract_tokens()}; layers [0, 32, 62] |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: {fixture["device"]} compute {fixture["compute_capability"]}
- measurement_utc: {fixture["measurement_utc"]}
- P Quartz mean tok/s: {quartz_p} versus OPT-051 {OPT051_P}
- D128 Quartz mean tok/s: {quartz_d128} versus OPT-051 {OPT051_D128}
- D2048 Quartz mean tok/s: {quartz_d2048} versus OPT-051 {OPT051_D2048}
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false
"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def contract_tokens() -> list[int]:
    return [1, 2, 3, 4, 63, 64, 65, 512, 2048, 4096]


def _write_rejection(fixture: dict[str, Any]) -> None:
    ab = fixture["ab"]
    reason = (
        "4096 component A/B selected shared"
        if fixture["keep_sitting_skipped"]
        else "keep sitting failed P improvement or the cross-workload guard"
    )
    tok = (
        ""
        if fixture["p"] is None
        else f"\n- P Quartz mean tok/s: {fixture['p']['quartz']['mean_tok_s']}\n"
        f"- OPT-051 P baseline: {OPT051_P}\n"
        f"- D128 Quartz mean tok/s: {fixture['d128']['quartz']['mean_tok_s']}\n"
        f"- D2048 Quartz mean tok/s: {fixture['d2048']['quartz']['mean_tok_s']}"
    )
    text = f"""# OPT-052 rejection

{reason}. Production prompt GDN preprocessing remains off
(`selected_gdn_preproc_path=off`); shared inverse stays installed.

- 4096 A/B winner: {ab["winner"]}
- keep_sitting_skipped: {json.dumps(fixture["keep_sitting_skipped"])}
- measurement_utc: {fixture["measurement_utc"]}{tok}
"""
    REJECTION.write_text(text)


def _run_ab(tier: str) -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    commands = [
        [*_common(IMAGE), "make", "build/gdn_step.cuda.o"],
        _nvcc(
            "cuda/opt052_gdn_arithmetic_ab_test.cu",
            "build/qw38-cuda-opt052-gdn-arithmetic-ab-test",
            ["build/gdn_step.cuda.o"],
        ),
        [
            *_common(IMAGE, tier),
            "./build/qw38-cuda-opt052-gdn-arithmetic-ab-test",
            "evidence/optimization/opt052-gdn-arithmetic/gdn-arithmetic-ab-raw.txt",
        ],
    ]
    outputs: list[str] = []
    for command in commands:
        outputs.append(_run(command).stdout)
    record = _parse_prefixed(outputs[-1], AB_PREFIX)
    assert "status=passed" in outputs[-1]
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
    selected = "off" if winner == "shared" else winner
    return {
        "schema_version": 1,
        "task": "OPT-052",
        "status": "rejected" if selected == "off" else "measured",
        "measurement_utc": ab["measurement_utc"],
        "device": ab["device"],
        "compute_capability": ab["compute_capability"],
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": selected == "off",
        "selected_gdn_preproc_path": selected,
        "selected_gdn_inverse_path": "shared",
        "ab": ab["ab"] if "ab" in ab else ab,
        "correctness": ab["correctness"],
        "production_numerics": ab.get(
            "production_numerics",
            {
                "formula": "max(strict_reference_ceiling, 1.05 * measured_llama_error + 1e-6)",
                "budget": {
                    "max_abs": 0.0003,
                    "rms": 0.0002,
                    "one_minus_cosine": 1e-06,
                },
            },
        ),
        "keep_sitting_skipped": selected == "off",
        "p": None,
        "d128": None,
        "d2048": None,
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt052-gdn-arithmetic/REPORT.md",
        "_install": selected,
    }


@pytest.mark.skipif(
    os.environ.get("QW38_RUN_CUDA_TESTS") != "1",
    reason="set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate",
)
def test_opt052_exclusive_cuda_sitting() -> None:
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
        [*_common(IMAGE), "./build/qw38-cuda-opt052-gdn-arithmetic-ab-test"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert closed.returncode != 0
    assert "QW38_CUDA_TEST_TIER must be set" in (closed.stdout + closed.stderr)
    if tier != "acceptance":
        assert ab["task"] == "OPT-052"
        winner = ab["ab"]["winner"] if "ab" in ab else ab["winner"]
        assert winner in LEGAL_CANDIDATES
        return

    wrapper = _fixture_from_ab(ab)
    selected = wrapper.pop("_install")
    if selected == "off":
        _set_pin("off")
        wrapper["selected_gdn_preproc_path"] = "off"
        _write_report(wrapper)
        _write_rejection(wrapper)
        FIXTURE.write_text(json.dumps(wrapper, indent=2) + "\n")
        validate_result(wrapper)
        return

    _set_pin(selected)
    _run(
        [
            *_common(IMAGE),
            "make",
            "build/gdn_step.cuda.o",
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
    fixture["selected_gdn_preproc_path"] = selected
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
        fixture["selected_gdn_preproc_path"] = "off"
        fixture["keep_sitting_skipped"] = False
        _write_rejection(fixture)
    else:
        fixture["status"] = "measured"
        if REJECTION.is_file():
            REJECTION.unlink()
    _write_report(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    validate_result(fixture)
