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
LLAMA_IMAGE = "qw38-llama-authority:cuda-13.0.2"
CONTRACT = ROOT / "pins/opt033_register_vkq_contract.json"
FIXTURE = ROOT / "fixtures/opt033_register_vkq.json"
OPT036_FIXTURE = ROOT / "fixtures/opt036_decode_kv_partition.json"
EVIDENCE = ROOT / "evidence/optimization/opt033-register-vkq"
REPORT = EVIDENCE / "REPORT.md"
REJECTION = EVIDENCE / "REJECTION.md"
AB_RAW = EVIDENCE / "vkq-ab-raw.txt"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
AB_PREFIX = "QW38_FATTN_REGISTER_VKQ_AB_RESULT="
P_PREFIX = "QW38_PREFILL_4K_ORACLE_RESULT="
DECODE_PREFIX = "QW38_DECODE_ORACLE_RESULT="
LLAMA_DECODE_PREFIX = "QW38_LLAMA_DECODE_ORACLE_RESULT="
OPT036_P = 1644.04822
OPT036_D128 = 15.200716
OPT036_D2048 = 13.5282431
OPT036_D128_P95 = 66.3502579
OPT036_D128_RUN_P95 = 65.8343124
OPT036_D2048_P95 = 74.4495544
OPT036_D2048_RUN_P95 = 73.996994
HISTORICAL_P = 1746.71973
OPT032_P = 1637.58594
LEGAL_ACCUM = {"global", "registers"}
PROOF = (
    "byte-equal output; lower component time; improved P; "
    "cross-workload guard; does not substitute for the 2K llama.cpp parity gate; "
    "OPT-036 P D128 D2048 are the keep denominators"
)
NVCC_OBJECTS = [
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


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def selected_vkq_from_source() -> str:
    text = (ROOT / "cuda/fattn_mma_f16.cuh").read_text()
    match = re.search(r'kSelectedVkqAccum\[\] = "([^"]+)"', text)
    assert match is not None
    return match.group(1)


def fattn_path_unchanged() -> bool:
    text = (ROOT / "cuda/fattn_mma_f16.cuh").read_text()
    return (
        'kSelectedFattnPath[] = "stream_k"' in text
        and "kSelectedAttentionMmaQueryRows = 16" in text
        and "kFattnNcols2 = 2" in text
        and "kFattnNbatchFa = 32" in text
        and "fattn_mma_quality_typed<16, true, 2, 2" in text
    )


def decode_16_16_unchanged() -> bool:
    text = (ROOT / "cuda/attention_decode.cu").read_text()
    return (
        "kSelectedDecodeKvPartsLow = 16" in text
        and "kSelectedDecodeKvPartsHigh = 16" in text
    )


def skinny_path_unchanged() -> bool:
    text = (ROOT / "cuda/quant_mmq_mma.cuh").read_text()
    return 'kSelectedSkinnyMixerPath[] = "mma_i32_j128"' in text


def ffn_shared_y_unchanged() -> bool:
    text = (ROOT / "cuda/quant_mmq_mma.cuh").read_text()
    return 'kSelectedFfnPath[] = "shared_y_swiglu_q8"' in text


def scalar_pxv_unchanged() -> bool:
    text = (ROOT / "cuda/fattn_mma_f16.cuh").read_text()
    return (
        "acc[idx] += scores[qcol * kFattnNbatchFa + krow] *" in text
        and "mma_m16n8k16_f16_f32(cfrag, a, b)" in text
        and "fattn_stream_k_combine_kernel" in text
    )


def opt035_not_started() -> bool:
    return not (ROOT / "pins/opt035_mma_pv_contract.json").is_file()


def _ab_candidate(result: dict[str, Any], ident: str) -> None:
    cand = result["ab"]["candidates"][ident]
    assert cand["id"] == ident
    assert len(cand["samples"]) == 30
    assert len(cand["warmup_ms"]) == 3
    mean = sum(float(v) for v in cand["samples"]) / 30.0
    assert cand["mean_ms"] == pytest.approx(mean, rel=1e-6, abs=1e-6)
    assert cand["occupancy"] >= 1
    assert cand["vs_tiled"]["nonfinite"] == 0
    assert cand["include_combine"] is True
    if cand["eligible"]:
        assert cand["vs_tiled"]["max_abs"] <= 5.0e-5
        assert cand["vs_tiled"]["rms"] <= 5.0e-6
        assert cand["scratch_unchanged"] is True
        assert cand["committed_unchanged"] is True
        assert cand["candidate_exact"] is True
        if ident == "registers":
            assert cand["byte_equal"] is True


def _select_winner(result: dict[str, Any]) -> str:
    global_c = result["ab"]["candidates"]["global"]
    registers = result["ab"]["candidates"]["registers"]
    if (
        registers["eligible"]
        and registers["byte_equal"]
        and float(registers["mean_ms"]) < float(global_c["mean_ms"])
    ):
        return "registers"
    return "global"


def _keep_predicates(result: dict[str, Any]) -> bool:
    contract = _contract()
    if result["selected_vkq_accum"] != "registers":
        return False
    if result["reverted"] is True or result["keep_sitting_skipped"] is True:
        return False
    if result["status"] != "measured":
        return False
    if result["ab"]["winner"] != "registers" or result["ab"]["win"] is not True:
        return False
    registers = result["ab"]["candidates"]["registers"]
    global_c = result["ab"]["candidates"]["global"]
    if registers["byte_equal"] is not True:
        return False
    if not (float(registers["mean_ms"]) < float(global_c["mean_ms"])):
        return False
    if result["p"] is None or result["d128"] is None or result["d2048"] is None:
        return False
    if (
        float(result["p"]["quartz"]["mean_tok_s"])
        <= contract["opt036_quartz_p_mean_tok_s"]
    ):
        return False
    quartz_d128 = float(result["d128"]["quartz"]["mean_tok_s"])
    if quartz_d128 < 0.95 * contract["opt036_quartz_d128_mean_tok_s"]:
        return False
    if (
        result["d128"]["quartz"]["token_latency_p95_ms"]
        > 1.05 * contract["opt036_quartz_d128_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d128"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * contract["opt036_quartz_d128_run_mean_token_latency_p95_ms"]
    ):
        return False
    quartz_d2048 = float(result["d2048"]["quartz"]["mean_tok_s"])
    if quartz_d2048 < 0.95 * contract["opt036_quartz_d2048_mean_tok_s"]:
        return False
    if (
        result["d2048"]["quartz"]["token_latency_p95_ms"]
        > 1.05 * contract["opt036_quartz_d2048_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d2048"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * contract["opt036_quartz_d2048_run_mean_token_latency_p95_ms"]
    ):
        return False
    return True


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-033"
    assert result["status"] in ("measured", "rejected")
    assert result["status"] not in ("scout", "source_inspected")
    assert contract["yardstick"] == "prefill_attention_register_vkq"
    assert contract["opt036_quartz_p_mean_tok_s"] == OPT036_P
    assert contract["opt036_quartz_d128_mean_tok_s"] == OPT036_D128
    assert contract["opt036_quartz_d2048_mean_tok_s"] == OPT036_D2048
    assert contract["historical_opt026_quartz_mean_tok_s"] == HISTORICAL_P
    assert contract["opt032_quartz_p_mean_tok_s"] == OPT032_P
    assert contract["use_as_keep_denominator"] is False
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    assert contract["opt005"]["bf16_max_abs"] == 0.00005
    assert contract["opt005"]["bf16_rms"] == 0.000005
    assert contract["ab"]["include_combine"] is True
    assert contract["ab"]["byte_equal_required"] is True
    assert contract["ab"]["candidates"] == ["global", "registers"]
    assert contract["attention_mma_query_rows"] == 16
    assert contract["ncols2"] == 2
    assert contract["kv_tile"] == 32
    assert contract["dual_f16"] is True
    assert contract["grid_z"] == 2
    pin = result["selected_vkq_accum"]
    assert pin in LEGAL_ACCUM
    assert selected_vkq_from_source() == pin == contract["selected_vkq_accum"]
    assert result["ab"]["include_combine"] is True
    assert set(result["ab"]["candidates"]) == {"global", "registers"}
    _ab_candidate(result, "global")
    _ab_candidate(result, "registers")
    assert _select_winner(result) == result["ab"]["winner"]
    assert fattn_path_unchanged()
    assert decode_16_16_unchanged()
    assert skinny_path_unchanged()
    assert ffn_shared_y_unchanged()
    assert scalar_pxv_unchanged()
    assert opt035_not_started()
    if result["ab"]["winner"] == "global":
        assert pin == "global"
        assert result["reverted"] is True
        assert result["keep_sitting_skipped"] is True
        assert result["status"] == "rejected"
    keep = _keep_predicates(result)
    if result["status"] == "measured":
        assert keep
        assert result["reverted"] is False
        assert result["keep_sitting_skipped"] is False
        assert pin == "registers"
        assert result["p"]["quartz"]["mean_tok_s"] > OPT036_P
        assert not REJECTION.is_file()
    else:
        assert not keep
        assert result["reverted"] is True
        assert pin == "global"
        assert REJECTION.is_file()
        rejection = REJECTION.read_text()
        assert "global" in rejection
        if result["keep_sitting_skipped"] is False:
            assert str(result["p"]["quartz"]["mean_tok_s"]) in rejection
        else:
            assert result["p"] is None and result["d128"] is None
            assert result["d2048"] is None
    proof = result["proof_limit"]
    for phrase in contract["proof_limit"]:
        assert phrase in proof
        assert phrase in PROOF
    assert result["report_path"] == contract["report_path"]
    assert "/tmp" not in result["report_path"]
    assert REPORT.is_file()
    report = REPORT.read_text()
    for phrase in contract["proof_limit"]:
        assert phrase in report
    assert AB_RAW.is_file()
    opt036 = json.loads(OPT036_FIXTURE.read_text())
    assert opt036["p"]["quartz"]["mean_tok_s"] == OPT036_P
    assert opt036["d128"]["quartz"]["mean_tok_s"] == OPT036_D128
    assert opt036["d2048"]["quartz"]["mean_tok_s"] == OPT036_D2048


def _candidate_stub(
    ident: str, mean: float, *, eligible: bool = True, byte_equal: bool = True
) -> dict[str, Any]:
    return {
        "id": ident,
        "mean_ms": mean,
        "samples": [mean] * 30,
        "warmup_ms": [mean] * 3,
        "occupancy": 2,
        "launch_ok": True,
        "eligible": eligible,
        "byte_equal": byte_equal,
        "scratch_unchanged": True,
        "committed_unchanged": True,
        "candidate_exact": True,
        "include_combine": True,
        "vs_tiled": {"max_abs": 0.0, "rms": 0.0, "nonfinite": 0},
    }


def _ab_stub(winner: str, global_ms: float, registers_ms: float) -> dict[str, Any]:
    registers_win = winner == "registers"
    return {
        "winner": winner,
        "win": registers_win,
        "include_combine": True,
        "candidates": {
            "global": _candidate_stub("global", global_ms),
            "registers": _candidate_stub(
                "registers",
                registers_ms,
                eligible=registers_win,
                byte_equal=True,
            ),
        },
    }


def test_opt033_contract_and_source_pins() -> None:
    contract = _contract()
    opt036 = json.loads(OPT036_FIXTURE.read_text())
    assert opt036["p"]["quartz"]["mean_tok_s"] == OPT036_P
    assert opt036["d128"]["quartz"]["mean_tok_s"] == OPT036_D128
    assert opt036["d2048"]["quartz"]["mean_tok_s"] == OPT036_D2048
    assert contract["opt036_quartz_p_mean_tok_s"] == OPT036_P
    assert contract["ab"]["candidates"] == ["global", "registers"]
    assert contract["ab"]["include_combine"] is True
    assert contract["owns_opt016_parity_gate"] is False
    tiled = json.loads((ROOT / "pins/cuda_tiled_attention_contract.json").read_text())
    assert tiled["proof_limits"]["bf16_max_abs"] == 0.00005
    assert tiled["proof_limits"]["bf16_rms"] == 0.000005
    assert selected_vkq_from_source() in LEGAL_ACCUM
    assert fattn_path_unchanged()
    assert decode_16_16_unchanged()
    assert scalar_pxv_unchanged()
    assert opt035_not_started()


def test_opt033_validator_rejects_inadmissible_evidence() -> None:
    fixture = {
        "schema_version": 1,
        "task": "OPT-033",
        "status": "rejected",
        "measurement_utc": "2026-09-09T00:00:00Z",
        "device": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": True,
        "selected_vkq_accum": "global",
        "ab": _ab_stub("global", 1.0, 1.5),
        "keep_sitting_skipped": True,
        "p": None,
        "d128": None,
        "d2048": None,
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt033-register-vkq/REPORT.md",
    }
    mutations: list[dict[str, Any]] = []
    for mutate in (
        lambda x: x.__setitem__("substitutes_for_opt016", True),
        lambda x: x.__setitem__("owns_opt016_parity_gate", True),
        lambda x: x.__setitem__("nsight_systems", "/tmp/capture.nsys-rep"),
        lambda x: x.__setitem__("report_path", "/tmp/opt033/REPORT.md"),
        lambda x: x.__setitem__("status", "source_inspected"),
        lambda x: x.__setitem__("task", "OPT-035"),
        lambda x: x["ab"].__setitem__("winner", "registers"),
        lambda x: x.__setitem__("selected_vkq_accum", "registers"),
        lambda x: x.__setitem__("keep_sitting_skipped", False),
        lambda x: x.__setitem__("reverted", False),
        lambda x: x["ab"].__setitem__("include_combine", False),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
    equality = json.loads(json.dumps(fixture))
    equality["status"] = "measured"
    equality["reverted"] = False
    equality["keep_sitting_skipped"] = False
    equality["selected_vkq_accum"] = "registers"
    equality["ab"] = _ab_stub("registers", 2.0, 1.0)
    equality["p"] = {"quartz": {"mean_tok_s": OPT036_P}}
    equality["d128"] = {
        "quartz": {
            "mean_tok_s": OPT036_D128,
            "token_latency_p95_ms": OPT036_D128_P95,
            "run_mean_token_latency_p95_ms": OPT036_D128_RUN_P95,
        }
    }
    equality["d2048"] = {
        "quartz": {
            "mean_tok_s": OPT036_D2048,
            "token_latency_p95_ms": OPT036_D2048_P95,
            "run_mean_token_latency_p95_ms": OPT036_D2048_RUN_P95,
        }
    }
    mutations.append(equality)
    loosened = json.loads(json.dumps(fixture))
    loosened["ab"]["candidates"]["registers"]["vs_tiled"]["max_abs"] = 0.1
    loosened["ab"]["candidates"]["registers"]["eligible"] = True
    loosened["ab"]["winner"] = "registers"
    mutations.append(loosened)
    skip_combine = json.loads(json.dumps(fixture))
    skip_combine["ab"]["candidates"]["global"]["include_combine"] = False
    mutations.append(skip_combine)
    no_byte = json.loads(json.dumps(equality))
    no_byte["ab"]["candidates"]["registers"]["byte_equal"] = False
    mutations.append(no_byte)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    report_prev = REPORT.read_text() if REPORT.is_file() else None
    raw_prev = AB_RAW.read_text() if AB_RAW.is_file() else None
    rejection_prev = REJECTION.read_text() if REJECTION.is_file() else None
    if report_prev is None:
        REPORT.write_text(PROOF + "\n")
    if raw_prev is None:
        AB_RAW.write_text("placeholder\n")
    if rejection_prev is None:
        REJECTION.write_text("rejected placeholder global\n")
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


def test_opt033_fixture_connected() -> None:
    if not FIXTURE.is_file():
        pytest.skip("OPT-033 fixture is written by the exclusive CUDA sitting")
    validate_result(json.loads(FIXTURE.read_text()))


def test_opt033_rejects_mutating_opt036_fixture() -> None:
    opt036 = json.loads(OPT036_FIXTURE.read_text())
    assert opt036["p"]["quartz"]["mean_tok_s"] == OPT036_P
    assert opt036["d128"]["quartz"]["mean_tok_s"] == OPT036_D128
    assert opt036["d2048"]["quartz"]["mean_tok_s"] == OPT036_D2048
    contract = _contract()
    assert contract["opt036_quartz_p_mean_tok_s"] == opt036["p"]["quartz"]["mean_tok_s"]
    assert (
        contract["opt036_quartz_d128_mean_tok_s"]
        == opt036["d128"]["quartz"]["mean_tok_s"]
    )
    assert (
        contract["opt036_quartz_d2048_mean_tok_s"]
        == opt036["d2048"]["quartz"]["mean_tok_s"]
    )


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


def _set_pin(accum: str) -> None:
    source = ROOT / "cuda/fattn_mma_f16.cuh"
    text = source.read_text()
    text = re.sub(
        r'kSelectedVkqAccum\[\] = "[^"]+"',
        f'kSelectedVkqAccum[] = "{accum}"',
        text,
    )
    source.write_text(text)
    contract = _contract()
    contract["selected_vkq_accum"] = accum
    CONTRACT.write_text(json.dumps(contract, indent=2) + "\n")


def _write_report(fixture: dict[str, Any]) -> None:
    decision = "keep" if fixture["status"] == "measured" else "reject"
    sitting = "skipped" if fixture["keep_sitting_skipped"] else "ran"
    quartz_p = fixture["p"]["quartz"]["mean_tok_s"] if fixture["p"] else "n/a"
    quartz_d128 = fixture["d128"]["quartz"]["mean_tok_s"] if fixture["d128"] else "n/a"
    quartz_d2048 = (
        fixture["d2048"]["quartz"]["mean_tok_s"] if fixture["d2048"] else "n/a"
    )
    global_ms = fixture["ab"]["candidates"]["global"]["mean_ms"]
    registers_ms = fixture["ab"]["candidates"]["registers"]["mean_ms"]
    text = f"""# OPT-033 — Retain attention value sums in registers

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of 4096-row production stream-K attention including
combine, candidates `global` versus `registers`. Keep requires
**byte-equal output**, **lower component time**, **improved P**, and the
**cross-workload guard**. This increment does not substitute for the 2K llama.cpp parity gate. The OPT-036 P D128 D2048 are the keep denominators
(P {OPT036_P}, D128 {OPT036_D128}, D2048 {OPT036_D2048}), not
historical OPT-026 {HISTORICAL_P} and not OPT-032 {OPT032_P}.

## Decision

**{decision}** — `reverted`={json.dumps(fixture["reverted"])};
`keep_sitting_skipped`={json.dumps(fixture["keep_sitting_skipped"])};
selected_vkq_accum={fixture["selected_vkq_accum"]};
A/B winner {fixture["ab"]["winner"]} (global mean {global_ms} ms,
registers mean {registers_ms} ms);
tok/s sitting {sitting}.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `{GGUF_SHA}` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `{LLAMA_REV}` |
| A/B | 4096 rows, global vs registers, 3 warm + 30 alternating, include combine |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: {fixture["device"]} compute {fixture["compute_capability"]}
- measurement_utc: {fixture["measurement_utc"]}
- P Quartz mean tok/s: {quartz_p} versus OPT-036 {OPT036_P}
- D128 Quartz mean tok/s: {quartz_d128} versus OPT-036 {OPT036_D128}
- D2048 Quartz mean tok/s: {quartz_d2048} versus OPT-036 {OPT036_D2048}
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false
"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def _write_rejection(fixture: dict[str, Any]) -> None:
    ab = fixture["ab"]
    reason = (
        "component A/B retained global per-tile stores"
        if fixture["keep_sitting_skipped"]
        else "keep sitting failed P improvement or the cross-workload guard"
    )
    tok = (
        ""
        if fixture["keep_sitting_skipped"]
        else f"\n- P Quartz mean tok/s: {fixture['p']['quartz']['mean_tok_s']}\n"
        f"- OPT-036 P baseline: {OPT036_P}\n"
        f"- D128 Quartz mean tok/s: {fixture['d128']['quartz']['mean_tok_s']}\n"
        f"- D2048 Quartz mean tok/s: {fixture['d2048']['quartz']['mean_tok_s']}"
    )
    text = f"""# OPT-033 rejection

{reason}. Production value accumulation is global
(`selected_vkq_accum=global`).

- A/B winner: {ab["winner"]}
- global mean_ms: {ab["candidates"]["global"]["mean_ms"]}
- registers mean_ms: {ab["candidates"]["registers"]["mean_ms"]}
- registers byte_equal: {json.dumps(ab["candidates"]["registers"]["byte_equal"])}
- keep_sitting_skipped: {json.dumps(fixture["keep_sitting_skipped"])}
- measurement_utc: {fixture["measurement_utc"]}{tok}
"""
    REJECTION.write_text(text)


def _flatten_ab(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "winner": record["winner"],
        "win": record["win"],
        "include_combine": record.get("include_combine", True),
        "candidates": record["candidates"],
    }


def _run_ab() -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    commands = [
        [*_common(IMAGE), "make", "build/qw38-cuda-timing-test"],
        _nvcc(
            "cuda/fattn_register_vkq_ab_test.cu",
            "build/qw38-cuda-fattn-register-vkq-ab-test",
            ["build/attention_decode.cuda.o"],
        ),
        [
            *_common(IMAGE),
            "./build/qw38-cuda-fattn-register-vkq-ab-test",
            "evidence/optimization/opt033-register-vkq/vkq-ab-raw.txt",
        ],
    ]
    outputs: list[str] = []
    for command in commands:
        outputs.append(_run(command).stdout)
    record = _parse_prefixed(outputs[-1], AB_PREFIX)
    assert "status=passed" in outputs[-1]
    return record


def _parse_llama_bench(text: str, predicate, what: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list) and payload and predicate(payload[0]):
            return payload
    raise AssertionError(what + "\n" + text)


def _run_llama_bench_p() -> dict[str, Any]:
    command = [
        *_common(LLAMA_IMAGE),
        "bash",
        "-lc",
        "cd /workspace && .cache/authorities/llama-build/bin/llama-bench "
        "-m /workspace/models/Qwen3.8-27B-Q4_K_M.gguf "
        "-p 4096 -n 0 --no-warmup -r 3 -ngl 99 -o json",
    ]
    completed = _run(command)
    payload = _parse_llama_bench(
        completed.stdout + completed.stderr,
        lambda row: row.get("n_prompt") == 4096,
        "llama-bench JSON with n_prompt 4096",
    )
    (EVIDENCE / "llama-bench-4k.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload[0]


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


def _run_llama_decode(prefix: int) -> dict[str, Any]:
    command = [
        *_common(LLAMA_IMAGE),
        "bash",
        "-lc",
        "cd /workspace && .cache/authorities/llama-build/bin/qw38-llama-decode-oracle "
        f"/workspace/models/Qwen3.8-27B-Q4_K_M.gguf {prefix}",
    ]
    completed = _run(command)
    record = _parse_prefixed(completed.stdout + completed.stderr, LLAMA_DECODE_PREFIX)
    (EVIDENCE / f"llama-decode-d{prefix}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return record


def _engine_block(record: dict[str, Any], sidecar: str | None = None) -> dict[str, Any]:
    block = {
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
    if sidecar is not None:
        path = EVIDENCE / sidecar
        path.write_text(json.dumps(record["token_latency_ms"]) + "\n")
        block["token_latency_sidecar"] = (
            f"evidence/optimization/opt033-register-vkq/{sidecar}"
        )
    for key in ("n_gpu_layers", "n_ctx", "n_batch", "n_ubatch"):
        if key in record:
            block[key] = record[key]
    return block


def test_opt033_native_keep_reject() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    if not MODEL.exists():
        pytest.skip("the pinned GGUF is required")
    if FIXTURE.is_file():
        existing = json.loads(FIXTURE.read_text())
        if existing.get("status") in ("measured", "rejected"):
            validate_result(existing)
            return

    ab = _run_ab()
    winner = ab["winner"]
    if winner == "global":
        _set_pin("global")
        fixture = {
            "schema_version": 1,
            "task": "OPT-033",
            "status": "rejected",
            "measurement_utc": ab["measurement_utc"],
            "device": ab["device"],
            "compute_capability": ab["compute_capability"],
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "reverted": True,
            "selected_vkq_accum": "global",
            "ab": _flatten_ab(ab),
            "keep_sitting_skipped": True,
            "p": None,
            "d128": None,
            "d2048": None,
            "owns_opt016_parity_gate": False,
            "substitutes_for_opt016": False,
            "nsight_systems": "not_used",
            "nsight_compute": "not_used",
            "proof_limit": PROOF,
            "report_path": "evidence/optimization/opt033-register-vkq/REPORT.md",
        }
        _write_report(fixture)
        _write_rejection(fixture)
        FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
        validate_result(fixture)
        return

    _set_pin("registers")
    _run([*_common(IMAGE), "make", "build/qw38-cuda-timing-test"])
    llama_p = _run_llama_bench_p()
    quartz_p = _run_quartz_p()
    llama_d128 = _run_llama_decode(128)
    quartz_d128 = _run_quartz_decode(128)
    llama_d2048 = _run_llama_decode(2048)
    quartz_d2048 = _run_quartz_decode(2048)
    fixture = {
        "schema_version": 1,
        "task": "OPT-033",
        "status": "measured",
        "measurement_utc": quartz_d2048.get("measurement_utc", ab["measurement_utc"]),
        "device": ab["device"],
        "compute_capability": ab["compute_capability"],
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": False,
        "selected_vkq_accum": "registers",
        "ab": _flatten_ab(ab),
        "keep_sitting_skipped": False,
        "p": {
            "quartz": {
                "prompt_tokens": 4096,
                "replicates": 3,
                "wall_ms": quartz_p["wall_ms"],
                "tok_s": quartz_p["tok_s"],
                "mean_tok_s": quartz_p["mean_tok_s"],
                "cold": True,
                "cache_policy": "disabled",
                "attribution": None,
                "graphs_created": True,
                "prompt_graph_rows": 4096,
            },
            "llama_cpp": {
                "avg_ts": llama_p["avg_ts"],
                "avg_ns": llama_p.get("avg_ns"),
                "n_prompt": 4096,
                "n_batch": llama_p.get("n_batch", 2048),
                "n_ubatch": llama_p.get("n_ubatch", 512),
                "flash_attn": llama_p.get("flash_attn", -1),
                "build_commit": llama_p.get("build_commit", "cc83d7b"),
                "test_time": llama_p.get("test_time", ab["measurement_utc"]),
            },
        },
        "d128": {
            "quartz": _engine_block(quartz_d128, "quartz-d128-tokens.json"),
            "llama_cpp": _engine_block(llama_d128),
        },
        "d2048": {
            "quartz": _engine_block(quartz_d2048, "quartz-d2048-tokens.json"),
            "llama_cpp": _engine_block(llama_d2048),
        },
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt033-register-vkq/REPORT.md",
    }
    keep = _keep_predicates(fixture)
    if not keep:
        _set_pin("global")
        fixture["status"] = "rejected"
        fixture["reverted"] = True
        fixture["selected_vkq_accum"] = "global"
        _write_report(fixture)
        _write_rejection(fixture)
        _run([*_common(IMAGE), "make", "build/qw38-cuda-timing-test"])
    else:
        if REJECTION.is_file():
            REJECTION.unlink()
        _write_report(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    validate_result(json.loads(FIXTURE.read_text()))
