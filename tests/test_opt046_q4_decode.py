from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
CONTRACT = ROOT / "pins/opt046_q4_decode_contract.json"
FIXTURE = ROOT / "fixtures/opt046_q4_decode.json"
OPT045_FIXTURE = ROOT / "fixtures/opt045_parallel_norm.json"
OPT041_FIXTURE = ROOT / "fixtures/opt041_fattn_warp_qk.json"
NUMERICS = ROOT / "pins/production_numerics_contract.json"
EVIDENCE = ROOT / "evidence/optimization/opt046-q4-decode"
REPORT = EVIDENCE / "REPORT.md"
REJECTION = EVIDENCE / "REJECTION.md"
AB_RAW = EVIDENCE / "q4-decode-ab-raw.txt"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
AB_PREFIX = "QW38_OPT046_Q4_DECODE_AB_RESULT="
P_PREFIX = "QW38_PREFILL_4K_ORACLE_RESULT="
DECODE_PREFIX = "QW38_DECODE_ORACLE_RESULT="
OPT045_P = 2130.41089
OPT045_D128 = 28.5522804
OPT045_D2048 = 27.5438766
OPT045_D128_P95 = 35.1905479
OPT045_D128_RUN_P95 = 35.0761757
OPT045_D2048_P95 = 36.4285774
OPT045_D2048_RUN_P95 = 36.3359795
OPT041_P = 2076.98315
OPT041_D128 = 26.1599541
OPT041_D2048 = 25.2924843
LEGAL_PATHS = {"packed", "integer_q8", "integer_q8_1"}
LEGAL_CANDIDATES = (
    "packed",
    "integer_q8_w1",
    "integer_q8_w2",
    "integer_q8_w4",
    "integer_q8_w8",
    "integer_q8_1_w1",
    "integer_q8_1_w2",
    "integer_q8_1_w4",
    "integer_q8_1_w8",
)
PROOF = (
    "OPT-044 production-numerics budgets; packed FP32 path retained; "
    "cooperative K warps per row; DP4A packed integer dots; "
    "Q8Block staging control; Q8_1 llama round/scale/sum; "
    "lower complete component time; 95% throughput floors versus OPT-045; "
    "105% p95 ceilings versus OPT-045; does not substitute for the 2K "
    "llama.cpp parity gate"
)
AB_OBJECTS = [
    "build/quant_mmv.cuda.o",
    "build/q4k_decode_dots.cuda.o",
    "build/quant.o",
    "build/status.o",
]
NVCC_OBJECTS = [
    "build/full_scheduler.trace.cuda.o",
    "build/scheduler_primitives.cuda.o",
    "build/quant_mmv.cuda.o",
    "build/q4k_decode_dots.cuda.o",
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


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def pin_from_source() -> str:
    text = (ROOT / "cuda/q4k_decode_path.cuh").read_text()
    match = re.search(r'kSelectedQ4DecodePath\[\] = "([^"]+)"', text)
    assert match is not None
    return match.group(1)


def warps_from_source() -> int:
    text = (ROOT / "cuda/q4k_decode_path.cuh").read_text()
    match = re.search(r"kSelectedQ4DecodeWarpsPerRow = ([0-9]+)", text)
    assert match is not None
    return int(match.group(1))


def packed_path_retained() -> bool:
    text = (ROOT / "cuda/quant_mmv.cu").read_text()
    return (
        'kSelectedMmvLoadPath[] = "packed"' in text
        and "sum = __fadd_rn(sum, __fmul_rn(weight, value));" in text
        and "__global__ void quant_mmv" in text
    )


def cooperative_design_present() -> bool:
    header = (ROOT / "cuda/q4k_decode_dots.cuh").read_text()
    mmv = (ROOT / "cuda/quant_mmv.cu").read_text()
    return (
        "__dp4a" in header
        and "WarpsPerRow" in header
        and "q4k_coop_mmv" in header
        and "quantize_bf16_q8_1" in header
        and "0x0F0F0F0F" in header
        and "launch_q4k_coop_mmv" in mmv
        and "q4_decode_uses_integer()" in mmv
        and "kSelectedMmvLoadPath" in mmv
    )


def q8_block_unchanged() -> bool:
    text = (ROOT / "cuda/quant_mmv.cu").read_text()
    return (
        "__global__ void quantize_bf16_q8" in text
        and "maximum / 127.0F" in text
        and "roundf(value / scale)" in text
    )


def q6k_q8_untouched() -> bool:
    text = (ROOT / "cuda/full_scheduler.cu").read_text()
    execute = text[text.find("cudaError_t execute_ffn") :]
    execute = execute[: execute.find("\ncudaError_t ", 1)]
    matrix = text[text.find("cudaError_t matrix_vector") :]
    matrix = matrix[: matrix.find("\ncudaError_t ", 1)]
    return (
        "matrix_vector(layer.ffn_gate" in execute
        and "launch_quant_mmv(" in matrix
        and "launch_q8_mmq" not in execute
    )


def _ab_candidate(block: dict[str, Any], ident: str) -> None:
    cand = block["candidates"][ident]
    assert cand["id"] == ident
    assert cand["occupancy"] >= 0
    assert cand["path"] in LEGAL_PATHS
    assert "shapes" in cand
    for shape in ("q4_k_17x256", "q4_k_257x512", "q4k_gate_up", "q4k_down"):
        slot = cand["shapes"][shape]
        assert slot["complete"]["launch_ok"] is True
        assert slot["prequant"]["launch_ok"] is True
        assert len(slot["complete"]["samples"]) == 30
        assert len(slot["prequant"]["samples"]) == 30
        assert len(slot["complete"]["warmup_ms"]) == 3
        mean = sum(float(v) for v in slot["complete"]["samples"]) / 30.0
        assert slot["complete"]["mean_ms"] == pytest.approx(mean, rel=1e-6, abs=1e-6)
        assert slot["vs_fp64_original_bf16"]["nonfinite"] == 0
        assert slot["vs_fp64_staged_activation"]["nonfinite"] == 0
    if ident != "packed":
        assert cand["cooperative"] is True
        assert cand["dp4a"] is True
        assert cand["warps_per_row"] in (1, 2, 4, 8)
        assert cand["local_bytes"] == 0


def _select_component_winner(block: dict[str, Any]) -> str:
    packed = block["candidates"]["packed"]
    if not packed["eligible"]:
        return "packed"
    packed_mean = float(packed["weighted_complete_ms"])
    best = "packed"
    best_mean = packed_mean
    for ident in LEGAL_CANDIDATES:
        if ident == "packed":
            continue
        cand = block["candidates"][ident]
        if not cand["eligible"]:
            continue
        mean = float(cand["weighted_complete_ms"])
        if mean < best_mean:
            best = ident
            best_mean = mean
    return best


def _select_install(result: dict[str, Any]) -> str:
    winner = _select_component_winner(result["ab"])
    if winner == "packed":
        return "packed"
    path = result["ab"]["candidates"][winner]["path"]
    if path not in LEGAL_PATHS:
        return "packed"
    return winner


def _keep_predicates(result: dict[str, Any]) -> bool:
    contract = _contract()
    if result["selected_q4_decode_path"] == "packed":
        return False
    if result["reverted"] is True or result["keep_sitting_skipped"] is True:
        return False
    if result["status"] != "measured":
        return False
    if _select_install(result) == "packed":
        return False
    if result["p"] is None or result["d128"] is None or result["d2048"] is None:
        return False
    if (
        float(result["p"]["quartz"]["mean_tok_s"])
        < 0.95 * contract["opt045_quartz_p_mean_tok_s"]
    ):
        return False
    if (
        float(result["d128"]["quartz"]["mean_tok_s"])
        < 0.95 * contract["opt045_quartz_d128_mean_tok_s"]
    ):
        return False
    if (
        float(result["d2048"]["quartz"]["mean_tok_s"])
        < 0.95 * contract["opt045_quartz_d2048_mean_tok_s"]
    ):
        return False
    if (
        result["d128"]["quartz"]["token_latency_p95_ms"]
        > 1.05 * contract["opt045_quartz_d128_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d128"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * contract["opt045_quartz_d128_run_mean_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d2048"]["quartz"]["token_latency_p95_ms"]
        > 1.05 * contract["opt045_quartz_d2048_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d2048"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * contract["opt045_quartz_d2048_run_mean_token_latency_p95_ms"]
    ):
        return False
    return True


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-046"
    assert result["status"] in ("measured", "rejected")
    assert result["status"] not in ("scout", "source_inspected")
    assert contract["yardstick"] == "cooperative_q4k_integer_decode_dots"
    assert contract["opt045_quartz_p_mean_tok_s"] == OPT045_P
    assert contract["opt045_quartz_d2048_mean_tok_s"] == OPT045_D2048
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    path = result["selected_q4_decode_path"]
    assert path in LEGAL_PATHS
    assert result["selected_q4_decode_warps_per_row"] in (1, 2, 4, 8)
    assert contract["ab"]["candidates"] == list(LEGAL_CANDIDATES)
    assert contract["ab"]["ties_retain"] == "packed"
    assert _select_component_winner(result["ab"]) == result["ab"]["winner"]
    for ident in LEGAL_CANDIDATES:
        _ab_candidate(result["ab"], ident)
    numerics = result["production_numerics"]
    assert "budget" in numerics
    packed = result["ab"]["candidates"]["packed"]
    assert packed["path"] == "packed"
    assert packed_path_retained()
    assert cooperative_design_present()
    assert q8_block_unchanged()
    assert pin_from_source() == result["selected_q4_decode_path"]
    assert warps_from_source() == result["selected_q4_decode_warps_per_row"]
    keep = _keep_predicates(result)
    if result["status"] == "measured":
        assert keep is True
        assert result["reverted"] is False
        assert REPORT.is_file()
        assert "OPT-044" in REPORT.read_text()
    else:
        assert keep is False
        assert result["selected_q4_decode_path"] == "packed"
        assert result["reverted"] is True
        if result["keep_sitting_skipped"]:
            assert result["p"] is None
        else:
            assert result["p"] is not None
        assert REJECTION.is_file()
        assert "packed" in REJECTION.read_text()
    assert AB_RAW.is_file()
    assert result["report_path"] == contract["report_path"]
    assert result["proof_limit"] == PROOF


def _timed(mean: float) -> dict[str, Any]:
    return {
        "mean_ms": mean,
        "launch_ok": True,
        "warmup_ms": [mean, mean, mean],
        "samples": [mean] * 30,
    }


def _shape_slot(mean: float, *, eligible: bool = True) -> dict[str, Any]:
    env = {
        "max_abs": 1e-6,
        "rms": 1e-7,
        "one_minus_cosine": 0.0,
        "nonfinite": 0,
    }
    return {
        "staging_bytes": 36 * 8,
        "complete": _timed(mean),
        "prequant": _timed(mean * 0.8),
        "vs_fp64_original_bf16": env,
        "vs_fp64_staged_activation": env,
        "eligible": eligible,
    }


def _ab_stub(
    winner: str, means: dict[str, float], eligible: bool = True
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for ident in LEGAL_CANDIDATES:
        path = (
            "packed"
            if ident == "packed"
            else ("integer_q8_1" if "q8_1" in ident else "integer_q8")
        )
        warps = 0 if ident == "packed" else int(ident.rsplit("_w", 1)[1])
        mean = means.get(ident, 1.0)
        shapes = {
            name: _shape_slot(mean, eligible=eligible)
            for name in ("q4_k_17x256", "q4_k_257x512", "q4k_gate_up", "q4k_down")
        }
        candidates[ident] = {
            "id": ident,
            "path": path,
            "warps_per_row": warps,
            "q8_1": "q8_1" in ident,
            "packed": ident == "packed",
            "occupancy": 1,
            "registers": 40,
            "local_bytes": 0,
            "cooperative": ident != "packed",
            "dp4a": ident != "packed",
            "eligible": eligible,
            "weighted_complete_ms": mean,
            "shapes": shapes,
        }
    return {"winner": winner, "win": winner != "packed", "candidates": candidates}


def _reject_fixture() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-046",
        "status": "rejected",
        "measurement_utc": "2026-09-10T00:00:00Z",
        "device": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": True,
        "selected_q4_decode_path": "packed",
        "selected_q4_decode_warps_per_row": 4,
        "ab": _ab_stub("packed", {ident: 1.0 for ident in LEGAL_CANDIDATES}),
        "production_numerics": {
            "formula": "max(strict_reference_ceiling, 1.05 * measured_llama_error + 1e-6)",
            "budget": {
                "max_abs": 3.27488041,
                "rms": 1.81794829,
                "one_minus_cosine": 1.97e-5,
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
        "report_path": "evidence/optimization/opt046-q4-decode/REPORT.md",
    }


def test_opt046_contract_and_source_pins() -> None:
    contract = _contract()
    opt041 = json.loads(OPT041_FIXTURE.read_text())
    opt045 = json.loads(OPT045_FIXTURE.read_text())
    numerics = json.loads(NUMERICS.read_text())
    assert opt041["p"]["quartz"]["mean_tok_s"] == OPT041_P
    assert opt045["p"]["quartz"]["mean_tok_s"] == OPT045_P
    assert contract["opt045_quartz_p_mean_tok_s"] == OPT045_P
    assert contract["opt045_quartz_d2048_mean_tok_s"] == OPT045_D2048
    assert contract["ab"]["candidates"] == list(LEGAL_CANDIDATES)
    assert contract["ab"]["warps_per_row_tested"] == [1, 2, 4, 8]
    assert contract["owns_opt016_parity_gate"] is False
    assert pin_from_source() in LEGAL_PATHS
    assert packed_path_retained()
    assert cooperative_design_present()
    assert q8_block_unchanged()
    assert q6k_q8_untouched()
    assert numerics["budget_rule"]["do_not_adjust_after_candidate_failure"] is True
    makefile = (ROOT / "Makefile").read_text()
    assert "--fmad=false" in makefile
    assert "opt046_q4_decode_ab_test" not in makefile
    assert "cuda/q4k_decode_dots.cuh" in makefile
    assert "cuda/q4k_decode_path.cuh" in makefile
    mmv = (ROOT / "cuda/quant_mmv.cu").read_text()
    assert "dp4a" not in mmv
    assert "__dp4a" not in mmv


def test_opt046_validator_rejects_inadmissible_evidence() -> None:
    fixture = _reject_fixture()
    mutations: list[dict[str, Any]] = []
    for mutate in (
        lambda x: x.__setitem__("substitutes_for_opt016", True),
        lambda x: x.__setitem__("owns_opt016_parity_gate", True),
        lambda x: x.__setitem__("status", "source_inspected"),
        lambda x: x.__setitem__("task", "OPT-045"),
        lambda x: x["ab"].__setitem__("winner", "integer_q8_1_w4"),
        lambda x: x.__setitem__("selected_q4_decode_path", "integer_q8_1"),
        lambda x: x.__setitem__("keep_sitting_skipped", False),
        lambda x: x.__setitem__("reverted", False),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
    skip_ab = json.loads(json.dumps(fixture))
    skip_ab.pop("ab")
    mutations.append(skip_ab)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    report_prev = REPORT.read_text() if REPORT.is_file() else None
    raw_prev = AB_RAW.read_text() if AB_RAW.is_file() else None
    rejection_prev = REJECTION.read_text() if REJECTION.is_file() else None
    if report_prev is None:
        REPORT.write_text(PROOF + "\n")
    if raw_prev is None:
        AB_RAW.write_text("placeholder\n")
    if rejection_prev is None:
        REJECTION.write_text("rejected placeholder packed\n")
    try:
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


def test_opt046_fixture_connected() -> None:
    if not FIXTURE.is_file():
        pytest.skip("OPT-046 fixture is written by the exclusive CUDA sitting")
    validate_result(json.loads(FIXTURE.read_text()))


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
        "-DQW38_DIAGNOSTIC_TRACE",
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


def _set_pin(path: str, warps_per_row: int) -> None:
    source = ROOT / "cuda/q4k_decode_path.cuh"
    text = source.read_text()
    text = re.sub(
        r'kSelectedQ4DecodePath\[\] = "[^"]+"',
        f'kSelectedQ4DecodePath[] = "{path}"',
        text,
    )
    text = re.sub(
        r"kSelectedQ4DecodeWarpsPerRow = [0-9]+",
        f"kSelectedQ4DecodeWarpsPerRow = {warps_per_row}",
        text,
    )
    source.write_text(text)
    contract = _contract()
    contract["selected_q4_decode_path"] = path
    contract["selected_q4_decode_warps_per_row"] = warps_per_row
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
    packed_ms = ab["candidates"]["packed"]["weighted_complete_ms"]
    winner = ab["winner"]
    winner_ms = ab["candidates"][winner]["weighted_complete_ms"]
    text = f"""# OPT-046 — Cooperative packed Q4_K integer decode dots

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of cooperative Q4_K integer-dot MMV (DP4A packed
products, K distributed across warps per row, independent accumulators until
row completion, Q8Block control staging and llama Q8_1 round/scale/sum) against
the retained packed FP32 path. Keep requires **OPT-044 production-numerics
budgets**, **retained packed FP32 path**, **lower complete component time**,
**95% P/D throughput floors versus OPT-045**, and **105% decode p95 ceilings
versus OPT-045**. This increment does not substitute for the 2K llama.cpp
parity gate. OPT-045 keep denominators are P {OPT045_P}, D128 {OPT045_D128},
D2048 {OPT045_D2048}. Historical OPT-041 copies remain P {OPT041_P}, D128
{OPT041_D128}, D2048 {OPT041_D2048}.

## Decision

**{decision}** — `reverted`={json.dumps(fixture["reverted"])};
`keep_sitting_skipped`={json.dumps(fixture["keep_sitting_skipped"])};
selected_q4_decode_path={fixture["selected_q4_decode_path"]};
warps_per_row={fixture["selected_q4_decode_warps_per_row"]};
A/B winner {winner} (packed {packed_ms} ms, winner {winner_ms} ms);
tok/s sitting {sitting}.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `{GGUF_SHA}` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp revision | `{LLAMA_REV}` |
| A/B | production gate/up 17408×5120 and down 5120×17408, 1/2/4/8 warps per row, 3 warm + 30 |
| Numerics | OPT-044 independently decoded FP64 dots |
| P / D128 / D2048 | OPT-021 / OPT-032 versus OPT-045 keep denominators |
| Nsight | not_used |

## Measured sitting

- Device: {fixture["device"]} compute {fixture["compute_capability"]}
- measurement_utc: {fixture["measurement_utc"]}
- P Quartz mean tok/s: {quartz_p} versus OPT-045 {OPT045_P}
- D128 Quartz mean tok/s: {quartz_d128} versus OPT-045 {OPT045_D128}
- D2048 Quartz mean tok/s: {quartz_d2048} versus OPT-045 {OPT045_D2048}
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false
"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def _write_rejection(fixture: dict[str, Any]) -> None:
    ab = fixture["ab"]
    reason = (
        "component A/B retained packed"
        if fixture["keep_sitting_skipped"]
        else "keep sitting failed OPT-044 numerics or the P/D guard versus OPT-045"
    )
    tok = (
        ""
        if fixture["keep_sitting_skipped"]
        else f"\n- P Quartz mean tok/s: {fixture['p']['quartz']['mean_tok_s']}\n"
        f"- OPT-045 P baseline: {OPT045_P}\n"
        f"- D128 Quartz mean tok/s: {fixture['d128']['quartz']['mean_tok_s']}\n"
        f"- D2048 Quartz mean tok/s: {fixture['d2048']['quartz']['mean_tok_s']}"
    )
    text = f"""# OPT-046 rejection

{reason}. Production Q4_K decode remains the packed FP32 path
(`selected_q4_decode_path=packed`).

- A/B winner: {ab["winner"]}
- keep_sitting_skipped: {json.dumps(fixture["keep_sitting_skipped"])}
- measurement_utc: {fixture["measurement_utc"]}{tok}
"""
    REJECTION.write_text(text)


def _run_ab() -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    commands = [
        [
            *_common(IMAGE),
            "make",
            "build/quant_mmv.cuda.o",
            "build/q4k_decode_dots.cuda.o",
            "build/quant.o",
            "build/status.o",
            "build/full_scheduler.trace.cuda.o",
            "build/scheduler_primitives.cuda.o",
            "build/gdn_step.cuda.o",
            "build/attention_decode.cuda.o",
            "diagnostic",
        ],
        _nvcc(
            "cuda/opt046_q4_decode_ab_test.cu",
            "build/qw38-cuda-opt046-q4-decode-ab-test",
            extra=AB_OBJECTS,
        ),
        [
            *_common(IMAGE),
            "./build/qw38-cuda-opt046-q4-decode-ab-test",
            "evidence/optimization/opt046-q4-decode/q4-decode-ab-raw.txt",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
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
        _nvcc(
            "cuda/prefill_4k_oracle_test.cu", "build/qw38-cuda-prefill-4k-oracle-test"
        ),
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
        _nvcc("cuda/decode_oracle_test.cu", binary),
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
        "prefix": record.get("prefix"),
        "decode_tokens": record.get("decode_tokens"),
        "warmups": record.get("warmups"),
        "runs": record.get("runs"),
        "warmup_tok_s": record.get("warmup_tok_s"),
        "tok_s": record.get("tok_s"),
        "run_wall_ms": record.get("run_wall_ms"),
        "mean_tok_s": record["mean_tok_s"],
        "token_latency_p50_ms": record.get("token_latency_p50_ms"),
        "token_latency_p95_ms": record["token_latency_p95_ms"],
        "run_mean_token_latency_p95_ms": record["run_mean_token_latency_p95_ms"],
        "graphs_created": record.get("graphs_created", True),
        "attribution": record.get("attribution"),
        "cache_policy": record.get("cache_policy", "disabled"),
    }


def _fixture_from_ab(ab: dict[str, Any]) -> dict[str, Any]:
    selected = _select_install({"ab": ab})
    return {
        "schema_version": 1,
        "task": "OPT-046",
        "status": "rejected",
        "measurement_utc": ab["measurement_utc"],
        "device": ab["device"],
        "compute_capability": ab["compute_capability"],
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": True,
        "selected_q4_decode_path": "packed",
        "selected_q4_decode_warps_per_row": 4,
        "ab": {
            "winner": ab["winner"],
            "win": ab["win"],
            "candidates": ab["candidates"],
        },
        "production_numerics": {
            "formula": "max(strict_reference_ceiling, 1.05 * measured_llama_error + 1e-6)",
            "budget": {
                "max_abs": 3.27488041,
                "rms": 1.81794829,
                "one_minus_cosine": 1.97071521e-5,
            },
        },
        "keep_sitting_skipped": selected == "packed",
        "p": None,
        "d128": None,
        "d2048": None,
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt046-q4-decode/REPORT.md",
        "_install": selected,
    }


@pytest.mark.skipif(
    os.environ.get("QW38_RUN_CUDA_TESTS") != "1",
    reason="set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate",
)
def test_opt046_exclusive_cuda_sitting() -> None:
    if not MODEL.exists():
        pytest.skip("the pinned GGUF is required")
    if FIXTURE.is_file():
        existing = json.loads(FIXTURE.read_text())
        if existing.get("status") in ("measured", "rejected"):
            validate_result(existing)
            return

    ab = _run_ab()
    wrapper = _fixture_from_ab(ab)
    selected = wrapper.pop("_install")
    if selected == "packed":
        _set_pin("packed", 4)
        _write_report(wrapper)
        _write_rejection(wrapper)
        FIXTURE.write_text(json.dumps(wrapper, indent=2) + "\n")
        validate_result(wrapper)
        return

    cand = ab["candidates"][selected]
    _set_pin(cand["path"], cand["warps_per_row"])
    _run(
        [
            *_common(IMAGE),
            "make",
            "build/full_scheduler.trace.cuda.o",
            "build/quant_mmv.cuda.o",
            "build/q4k_decode_dots.cuda.o",
        ]
    )
    quartz_p = _run_quartz_p()
    quartz_d128 = _run_quartz_decode(128)
    quartz_d2048 = _run_quartz_decode(2048)
    fixture = wrapper
    fixture["keep_sitting_skipped"] = False
    fixture["reverted"] = False
    fixture["selected_q4_decode_path"] = cand["path"]
    fixture["selected_q4_decode_warps_per_row"] = cand["warps_per_row"]
    fixture["p"] = {
        "quartz": {
            "prompt_tokens": 4096,
            "replicates": 3,
            "wall_ms": quartz_p["wall_ms"],
            "tok_s": quartz_p["tok_s"],
            "mean_tok_s": quartz_p["mean_tok_s"],
            "cold": True,
            "cache_policy": "disabled",
            "attribution": None,
        }
    }
    fixture["d128"] = {"quartz": _engine_block(quartz_d128)}
    fixture["d2048"] = {"quartz": _engine_block(quartz_d2048)}
    if _keep_predicates(fixture):
        fixture["status"] = "measured"
        _write_report(fixture)
        if REJECTION.is_file():
            REJECTION.unlink()
    else:
        fixture["status"] = "rejected"
        fixture["reverted"] = True
        fixture["selected_q4_decode_path"] = "packed"
        fixture["selected_q4_decode_warps_per_row"] = 4
        _set_pin("packed", 4)
        _write_report(fixture)
        _write_rejection(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    validate_result(fixture)
