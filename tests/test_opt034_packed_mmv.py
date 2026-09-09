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
CONTRACT = ROOT / "pins/opt034_packed_mmv_contract.json"
FIXTURE = ROOT / "fixtures/opt034_packed_mmv.json"
OPT035_FIXTURE = ROOT / "fixtures/opt035_pv_mma.json"
EVIDENCE = ROOT / "evidence/optimization/opt034-packed-mmv"
REPORT = EVIDENCE / "REPORT.md"
REJECTION = EVIDENCE / "REJECTION.md"
AB_RAW = EVIDENCE / "mmv-ab-raw.txt"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
AB_PREFIX = "QW38_PACKED_MMV_AB_RESULT="
P_PREFIX = "QW38_PREFILL_4K_ORACLE_RESULT="
DECODE_PREFIX = "QW38_DECODE_ORACLE_RESULT="
LLAMA_DECODE_PREFIX = "QW38_LLAMA_DECODE_ORACLE_RESULT="
OPT035_P = 1865.21155
OPT035_D128 = 15.0562878
OPT035_D2048 = 13.5411425
OPT035_D128_P95 = 66.6085587
OPT035_D128_RUN_P95 = 66.06633
OPT035_D2048_P95 = 74.366951
OPT035_D2048_RUN_P95 = 73.9000702
HISTORICAL_P = 1746.71973
OPT032_P = 1637.58594
OPT033_P = 1745.10315
OPT036_P = 1644.04822
LEGAL_PATHS = {"elementwise", "packed"}
PRODUCTION_SHAPES = ("q4k_gate_up", "q4k_down", "q6k_attn_out", "q6k_logits")
PROBE_SHAPES = ("q4_k_17x256", "q4_k_257x512", "q6_k_17x256", "q6_k_257x512")
PROOF = (
    "byte equality; lower weighted MMV time; improved D2048; "
    "cross-workload guard; does not substitute for the 2K llama.cpp parity gate; "
    "OPT-035 P D128 D2048 are the keep denominators"
)
AB_OBJECTS = [
    "build/quant_mmv.cuda.o",
    "build/quant.o",
    "build/status.o",
]
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


def selected_path_from_source() -> str:
    text = (ROOT / "cuda/quant_mmv.cu").read_text()
    match = re.search(r'kSelectedMmvLoadPath\[\] = "([^"]+)"', text)
    assert match is not None
    return match.group(1)


def warp_buckets_unchanged() -> bool:
    text = (ROOT / "cuda/quant_mmv.cu").read_text()
    return (
        "if (rows <= 48) return 4;" in text
        and "if (rows <= 1024) return 8;" in text
        and "if (rows <= 5120) return 16;" in text
        and "if (rows <= 17408) return 8;" in text
    )


def q8_staging_unchanged() -> bool:
    text = (ROOT / "cuda/quant_mmv.cu").read_text()
    return (
        "__global__ void quantize_bf16_q8" in text
        and "maximum / 127.0F" in text
        and "roundf(value / scale)" in text
    )


def fp32_lane_order_unchanged() -> bool:
    text = (ROOT / "cuda/quant_mmv.cu").read_text()
    return (
        "sum = __fadd_rn(sum, __fmul_rn(weight, value));" in text
        and "__shfl_down_sync(0xFFFFFFFFU, sum, offset, kWarpSize)" in text
        and "d * static_cast<float>(scales[i] * quant)" in text
        and "DP4A" not in text
        and "dp4a" not in text
    )


def decode_16_16_unchanged() -> bool:
    text = (ROOT / "cuda/attention_decode.cu").read_text()
    return (
        "kSelectedDecodeKvPartsLow = 16" in text
        and "kSelectedDecodeKvPartsHigh = 16" in text
    )


def fattn_pins_unchanged() -> bool:
    text = (ROOT / "cuda/fattn_mma_f16.cuh").read_text()
    return (
        'kSelectedFattnPath[] = "stream_k"' in text
        and 'kSelectedVkqAccum[] = "registers"' in text
        and 'kSelectedPvPath[] = "mma"' in text
    )


def skinny_path_unchanged() -> bool:
    text = (ROOT / "cuda/quant_mmq_mma.cuh").read_text()
    return 'kSelectedSkinnyMixerPath[] = "mma_i32_j128"' in text


def ffn_shared_y_unchanged() -> bool:
    text = (ROOT / "cuda/quant_mmq_mma.cuh").read_text()
    return 'kSelectedFfnPath[] = "shared_y_swiglu_q8"' in text


def opt037_not_started() -> bool:
    return not (ROOT / "pins/opt037_ffn_tile_contract.json").is_file()


def _shape_candidate(shape: dict[str, Any], ident: str, *, probe: bool) -> None:
    cand = shape["candidates"][ident]
    assert cand["id"] == ident
    assert len(cand["samples"]) == 30
    assert len(cand["warmup_ms"]) == 3
    mean = sum(float(v) for v in cand["samples"]) / 30.0
    assert cand["mean_ms"] == pytest.approx(mean, rel=1e-6, abs=1e-6)
    assert cand["occupancy"] >= 0
    assert cand["q8_equal"] is True
    if cand["eligible"]:
        assert cand["occupancy"] >= 1
        assert cand["launch_ok"] is True
        if ident == "packed":
            assert cand["byte_equal"] is True
        if probe:
            assert cand["vs_host"]["nonfinite"] == 0
            assert cand["vs_host"]["max_abs"] <= 3.0e-4
            assert cand["vs_host"]["rms"] <= 2.0e-4


def _select_winner(result: dict[str, Any]) -> str:
    packed_ok = True
    weighted = {"elementwise": 0.0, "packed": 0.0}
    weight_sum = 0
    for ident in PRODUCTION_SHAPES:
        shape = result["ab"]["shapes"][ident]
        packed_ok = packed_ok and shape["candidates"]["packed"]["eligible"]
        weight = int(shape["weight"])
        weight_sum += weight
        for path in ("elementwise", "packed"):
            weighted[path] += weight * float(shape["candidates"][path]["mean_ms"])
    for ident in PROBE_SHAPES:
        packed_ok = (
            packed_ok
            and result["ab"]["shapes"][ident]["candidates"]["packed"]["eligible"]
        )
    if not packed_ok or weight_sum == 0:
        return "elementwise"
    packed_mean = weighted["packed"] / weight_sum
    elementwise_mean = weighted["elementwise"] / weight_sum
    if packed_mean < elementwise_mean:
        return "packed"
    return "elementwise"


def _keep_predicates(result: dict[str, Any]) -> bool:
    contract = _contract()
    if result["selected_mmv_load_path"] != "packed":
        return False
    if result["reverted"] is True or result["keep_sitting_skipped"] is True:
        return False
    if result["status"] != "measured":
        return False
    if result["ab"]["winner"] != "packed" or result["ab"]["win"] is not True:
        return False
    if not (
        float(result["ab"]["weighted_mean_ms"]["packed"])
        < float(result["ab"]["weighted_mean_ms"]["elementwise"])
    ):
        return False
    if result["p"] is None or result["d128"] is None or result["d2048"] is None:
        return False
    if (
        float(result["p"]["quartz"]["mean_tok_s"])
        < 0.95 * contract["opt035_quartz_p_mean_tok_s"]
    ):
        return False
    if (
        float(result["d128"]["quartz"]["mean_tok_s"])
        < 0.95 * contract["opt035_quartz_d128_mean_tok_s"]
    ):
        return False
    if (
        result["d128"]["quartz"]["token_latency_p95_ms"]
        > 1.05 * contract["opt035_quartz_d128_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d128"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * contract["opt035_quartz_d128_run_mean_token_latency_p95_ms"]
    ):
        return False
    if (
        float(result["d2048"]["quartz"]["mean_tok_s"])
        <= contract["opt035_quartz_d2048_mean_tok_s"]
    ):
        return False
    if (
        result["d2048"]["quartz"]["token_latency_p95_ms"]
        > 1.05 * contract["opt035_quartz_d2048_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d2048"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * contract["opt035_quartz_d2048_run_mean_token_latency_p95_ms"]
    ):
        return False
    return True


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-034"
    assert result["status"] in ("measured", "rejected")
    assert result["status"] not in ("scout", "source_inspected")
    assert contract["yardstick"] == "d2048_decode_packed_q4k_q6k_mmv"
    assert contract["opt035_quartz_p_mean_tok_s"] == OPT035_P
    assert contract["opt035_quartz_d128_mean_tok_s"] == OPT035_D128
    assert contract["opt035_quartz_d2048_mean_tok_s"] == OPT035_D2048
    assert contract["historical_opt026_quartz_mean_tok_s"] == HISTORICAL_P
    assert contract["opt032_quartz_p_mean_tok_s"] == OPT032_P
    assert contract["opt033_quartz_p_mean_tok_s"] == OPT033_P
    assert contract["opt036_quartz_p_mean_tok_s"] == OPT036_P
    assert contract["use_as_keep_denominator"] is False
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    assert contract["cud001"]["maximum_absolute_error"] == 0.0003
    assert contract["cud001"]["maximum_rms_error"] == 0.0002
    assert contract["ab"]["byte_equal_required"] is True
    assert contract["ab"]["candidates"] == ["elementwise", "packed"]
    pin = result["selected_mmv_load_path"]
    assert pin in LEGAL_PATHS
    assert selected_path_from_source() == pin == contract["selected_mmv_load_path"]
    assert set(result["ab"]["shapes"]) == set(PRODUCTION_SHAPES + PROBE_SHAPES)
    for ident in PRODUCTION_SHAPES:
        _shape_candidate(result["ab"]["shapes"][ident], "elementwise", probe=False)
        _shape_candidate(result["ab"]["shapes"][ident], "packed", probe=False)
    for ident in PROBE_SHAPES:
        _shape_candidate(result["ab"]["shapes"][ident], "elementwise", probe=True)
        _shape_candidate(result["ab"]["shapes"][ident], "packed", probe=True)
    assert _select_winner(result) == result["ab"]["winner"]
    assert warp_buckets_unchanged()
    assert q8_staging_unchanged()
    assert fp32_lane_order_unchanged()
    assert decode_16_16_unchanged()
    assert fattn_pins_unchanged()
    assert skinny_path_unchanged()
    assert ffn_shared_y_unchanged()
    assert opt037_not_started()
    if result["ab"]["winner"] == "elementwise":
        assert pin == "elementwise"
        assert result["reverted"] is True
        assert result["keep_sitting_skipped"] is True
        assert result["status"] == "rejected"
    keep = _keep_predicates(result)
    if result["status"] == "measured":
        assert keep
        assert result["reverted"] is False
        assert result["keep_sitting_skipped"] is False
        assert pin == "packed"
        assert result["d2048"]["quartz"]["mean_tok_s"] > OPT035_D2048
        assert not REJECTION.is_file()
    else:
        assert not keep
        assert result["reverted"] is True
        assert pin == "elementwise"
        assert REJECTION.is_file()
        rejection = REJECTION.read_text()
        assert "elementwise" in rejection
        if result["keep_sitting_skipped"] is False:
            assert str(result["d2048"]["quartz"]["mean_tok_s"]) in rejection
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
    opt035 = json.loads(OPT035_FIXTURE.read_text())
    assert opt035["p"]["quartz"]["mean_tok_s"] == OPT035_P
    assert opt035["d128"]["quartz"]["mean_tok_s"] == OPT035_D128
    assert opt035["d2048"]["quartz"]["mean_tok_s"] == OPT035_D2048


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
        "q8_equal": True,
        "vs_host": {"max_abs": 0.0, "rms": 0.0, "nonfinite": 0},
    }


def _shape_stub(
    ident: str,
    *,
    weight: int,
    probe: bool,
    elementwise_ms: float,
    packed_ms: float,
    packed_equal: bool = True,
) -> dict[str, Any]:
    return {
        "id": ident,
        "weight": weight,
        "probe": probe,
        "candidates": {
            "elementwise": _candidate_stub("elementwise", elementwise_ms),
            "packed": _candidate_stub(
                "packed", packed_ms, eligible=packed_equal, byte_equal=packed_equal
            ),
        },
    }


def _ab_stub(winner: str) -> dict[str, Any]:
    packed_faster = winner == "packed"
    packed_ms = 0.8 if packed_faster else 1.2
    shapes = {}
    for ident, weight in (
        ("q4k_gate_up", 128),
        ("q4k_down", 64),
        ("q6k_attn_out", 16),
        ("q6k_logits", 1),
    ):
        shapes[ident] = _shape_stub(
            ident, weight=weight, probe=False, elementwise_ms=1.0, packed_ms=packed_ms
        )
    for ident in PROBE_SHAPES:
        shapes[ident] = _shape_stub(
            ident, weight=0, probe=True, elementwise_ms=0.01, packed_ms=0.008
        )
    return {
        "winner": winner,
        "win": packed_faster,
        "weighted_mean_ms": {
            "elementwise": 1.0,
            "packed": packed_ms,
        },
        "shapes": shapes,
    }


def test_opt034_contract_and_source_pins() -> None:
    contract = _contract()
    opt035 = json.loads(OPT035_FIXTURE.read_text())
    assert opt035["p"]["quartz"]["mean_tok_s"] == OPT035_P
    assert opt035["d128"]["quartz"]["mean_tok_s"] == OPT035_D128
    assert opt035["d2048"]["quartz"]["mean_tok_s"] == OPT035_D2048
    assert contract["opt035_quartz_p_mean_tok_s"] == OPT035_P
    assert contract["historical_opt026_quartz_mean_tok_s"] == HISTORICAL_P
    assert contract["opt032_quartz_p_mean_tok_s"] == OPT032_P
    assert contract["opt033_quartz_p_mean_tok_s"] == OPT033_P
    assert contract["opt036_quartz_p_mean_tok_s"] == OPT036_P
    assert contract["use_as_keep_denominator"] is False
    assert contract["ab"]["candidates"] == ["elementwise", "packed"]
    assert contract["ab"]["byte_equal_required"] is True
    assert contract["owns_opt016_parity_gate"] is False
    assert contract["q8_0_direct_bf16_out_of_scope"] is True
    assert contract["dp4a_out_of_scope"] is True
    quant = json.loads((ROOT / "pins/cuda_quant_contract.json").read_text())
    assert quant["admission"]["maximum_absolute_error"] == 0.0003
    assert quant["admission"]["maximum_rms_error"] == 0.0002
    assert selected_path_from_source() in LEGAL_PATHS
    assert warp_buckets_unchanged()
    assert q8_staging_unchanged()
    assert fp32_lane_order_unchanged()
    assert decode_16_16_unchanged()
    assert fattn_pins_unchanged()
    assert opt037_not_started()


def test_opt034_validator_rejects_inadmissible_evidence() -> None:
    fixture = {
        "schema_version": 1,
        "task": "OPT-034",
        "status": "rejected",
        "measurement_utc": "2026-09-09T00:00:00Z",
        "device": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": True,
        "selected_mmv_load_path": "elementwise",
        "ab": _ab_stub("elementwise"),
        "keep_sitting_skipped": True,
        "p": None,
        "d128": None,
        "d2048": None,
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt034-packed-mmv/REPORT.md",
    }
    mutations: list[dict[str, Any]] = []
    for mutate in (
        lambda x: x.__setitem__("substitutes_for_opt016", True),
        lambda x: x.__setitem__("owns_opt016_parity_gate", True),
        lambda x: x.__setitem__("nsight_systems", "/tmp/capture.nsys-rep"),
        lambda x: x.__setitem__("report_path", "/tmp/opt034/REPORT.md"),
        lambda x: x.__setitem__("status", "source_inspected"),
        lambda x: x.__setitem__("task", "OPT-037"),
        lambda x: x["ab"].__setitem__("winner", "packed"),
        lambda x: x.__setitem__("selected_mmv_load_path", "packed"),
        lambda x: x.__setitem__("keep_sitting_skipped", False),
        lambda x: x.__setitem__("reverted", False),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
    equality = json.loads(json.dumps(fixture))
    equality["status"] = "measured"
    equality["reverted"] = False
    equality["keep_sitting_skipped"] = False
    equality["selected_mmv_load_path"] = "packed"
    equality["ab"] = _ab_stub("packed")
    equality["p"] = {"quartz": {"mean_tok_s": OPT035_P * 0.96}}
    equality["d128"] = {
        "quartz": {
            "mean_tok_s": OPT035_D128,
            "token_latency_p95_ms": OPT035_D128_P95,
            "run_mean_token_latency_p95_ms": OPT035_D128_RUN_P95,
        }
    }
    equality["d2048"] = {
        "quartz": {
            "mean_tok_s": OPT035_D2048,
            "token_latency_p95_ms": OPT035_D2048_P95,
            "run_mean_token_latency_p95_ms": OPT035_D2048_RUN_P95,
        }
    }
    mutations.append(equality)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    report_prev = REPORT.read_text() if REPORT.is_file() else None
    raw_prev = AB_RAW.read_text() if AB_RAW.is_file() else None
    rejection_prev = REJECTION.read_text() if REJECTION.is_file() else None
    REPORT.write_text(
        "byte equality lower weighted MMV time improved D2048 "
        "cross-workload guard does not substitute for the 2K "
        "llama.cpp parity gate OPT-035 P D128 D2048 are the keep "
        "denominators\n"
    )
    AB_RAW.write_text("raw\n")
    REJECTION.write_text("elementwise\n")
    try:
        for changed in mutations:
            with pytest.raises(AssertionError):
                validate_result(changed)
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


def test_opt034_fixture_connected() -> None:
    if not FIXTURE.is_file():
        pytest.skip("OPT-034 fixture is written by the exclusive CUDA sitting")
    validate_result(json.loads(FIXTURE.read_text()))


def test_opt034_rejects_historical_p_as_keep_denominator() -> None:
    contract = _contract()
    assert contract["opt035_quartz_p_mean_tok_s"] == OPT035_P
    assert contract["historical_opt026_quartz_mean_tok_s"] == HISTORICAL_P
    assert contract["opt032_quartz_p_mean_tok_s"] == OPT032_P
    assert contract["opt033_quartz_p_mean_tok_s"] == OPT033_P
    assert contract["opt036_quartz_p_mean_tok_s"] == OPT036_P
    assert contract["use_as_keep_denominator"] is False
    keep_denominators = {
        contract["opt035_quartz_p_mean_tok_s"],
        contract["opt035_quartz_d128_mean_tok_s"],
        contract["opt035_quartz_d2048_mean_tok_s"],
    }
    assert HISTORICAL_P not in keep_denominators
    assert OPT032_P not in keep_denominators
    assert OPT033_P not in keep_denominators
    assert OPT036_P not in keep_denominators
    opt035 = json.loads(OPT035_FIXTURE.read_text())
    assert contract["opt035_quartz_p_mean_tok_s"] == opt035["p"]["quartz"]["mean_tok_s"]
    assert (
        contract["opt035_quartz_d128_mean_tok_s"]
        == opt035["d128"]["quartz"]["mean_tok_s"]
    )
    assert (
        contract["opt035_quartz_d2048_mean_tok_s"]
        == opt035["d2048"]["quartz"]["mean_tok_s"]
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


def _set_pin(path: str) -> None:
    source = ROOT / "cuda/quant_mmv.cu"
    text = source.read_text()
    text = re.sub(
        r'kSelectedMmvLoadPath\[\] = "[^"]+"',
        f'kSelectedMmvLoadPath[] = "{path}"',
        text,
    )
    source.write_text(text)
    contract = _contract()
    contract["selected_mmv_load_path"] = path
    CONTRACT.write_text(json.dumps(contract, indent=2) + "\n")


def _write_report(fixture: dict[str, Any]) -> None:
    decision = "keep" if fixture["status"] == "measured" else "reject"
    sitting = "skipped" if fixture["keep_sitting_skipped"] else "ran"
    quartz_p = fixture["p"]["quartz"]["mean_tok_s"] if fixture["p"] else "n/a"
    quartz_d128 = fixture["d128"]["quartz"]["mean_tok_s"] if fixture["d128"] else "n/a"
    quartz_d2048 = (
        fixture["d2048"]["quartz"]["mean_tok_s"] if fixture["d2048"] else "n/a"
    )
    elementwise_ms = fixture["ab"]["weighted_mean_ms"]["elementwise"]
    packed_ms = fixture["ab"]["weighted_mean_ms"]["packed"]
    text = f"""# OPT-034 — Packed blockwise Q4_K/Q6_K MMV loads

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of production batch-1 Q4_K/Q6_K decode MMV with
unchanged packed Q8 staging and FP32 lane/reduction order, candidates
`elementwise` versus `packed` blockwise field/scale loads. Keep requires
**byte equality**, **lower weighted MMV time**, **improved D2048**,
and the **cross-workload guard**. This increment does not substitute for the 2K llama.cpp parity gate. The OPT-035 P D128 D2048 are the keep denominators
(P {OPT035_P}, D128 {OPT035_D128}, D2048 {OPT035_D2048}), not
historical OPT-026 {HISTORICAL_P}, not OPT-032 {OPT032_P}, not OPT-033 {OPT033_P},
and not OPT-036 {OPT036_P}.

## Decision

**{decision}** — `reverted`={json.dumps(fixture["reverted"])};
`keep_sitting_skipped`={json.dumps(fixture["keep_sitting_skipped"])};
selected_mmv_load_path={fixture["selected_mmv_load_path"]};
A/B winner {fixture["ab"]["winner"]} (elementwise weighted mean {elementwise_ms} ms,
packed weighted mean {packed_ms} ms);
tok/s sitting {sitting}.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `{GGUF_SHA}` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `{LLAMA_REV}` |
| A/B | production batch-1 Q4_K/Q6_K MMV shapes, elementwise vs packed, 3 warm + 30 alternating, byte-equal, Q8 staging unchanged |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: {fixture["device"]} compute {fixture["compute_capability"]}
- measurement_utc: {fixture["measurement_utc"]}
- P Quartz mean tok/s: {quartz_p} versus OPT-035 {OPT035_P} (retain ≥95%)
- D128 Quartz mean tok/s: {quartz_d128} versus OPT-035 {OPT035_D128}
- D2048 Quartz mean tok/s: {quartz_d2048} versus OPT-035 {OPT035_D2048} (must improve)
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false
"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def _write_rejection(fixture: dict[str, Any]) -> None:
    ab = fixture["ab"]
    reason = (
        "component A/B retained elementwise per-column Q4_K/Q6_K MMV loads"
        if fixture["keep_sitting_skipped"]
        else "keep sitting failed D2048 improvement or the cross-workload guard"
    )
    tok = (
        ""
        if fixture["keep_sitting_skipped"]
        else f"\n- D2048 Quartz mean tok/s: {fixture['d2048']['quartz']['mean_tok_s']}\n"
        f"- OPT-035 D2048 baseline: {OPT035_D2048}\n"
        f"- P Quartz mean tok/s: {fixture['p']['quartz']['mean_tok_s']}\n"
        f"- D128 Quartz mean tok/s: {fixture['d128']['quartz']['mean_tok_s']}"
    )
    text = f"""# OPT-034 rejection

{reason}. Production MMV loads are elementwise
(`selected_mmv_load_path=elementwise`).

- A/B winner: {ab["winner"]}
- elementwise weighted mean_ms: {ab["weighted_mean_ms"]["elementwise"]}
- packed weighted mean_ms: {ab["weighted_mean_ms"]["packed"]}
- keep_sitting_skipped: {json.dumps(fixture["keep_sitting_skipped"])}
- measurement_utc: {fixture["measurement_utc"]}{tok}
"""
    REJECTION.write_text(text)


def _flatten_ab(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "winner": record["winner"],
        "win": record["win"],
        "weighted_mean_ms": record["weighted_mean_ms"],
        "shapes": record["shapes"],
    }


def _run_ab() -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    commands = [
        [*_common(IMAGE), "make", "build/qw38-cuda-quant-test"],
        _nvcc(
            "cuda/packed_mmv_ab_test.cu",
            "build/qw38-cuda-packed-mmv-ab-test",
            AB_OBJECTS,
        ),
        [
            *_common(IMAGE),
            "./build/qw38-cuda-packed-mmv-ab-test",
            "evidence/optimization/opt034-packed-mmv/mmv-ab-raw.txt",
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
            f"evidence/optimization/opt034-packed-mmv/{sidecar}"
        )
    for key in ("n_gpu_layers", "n_ctx", "n_batch", "n_ubatch"):
        if key in record:
            block[key] = record[key]
    return block


def test_opt034_native_keep_reject() -> None:
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
    if winner == "elementwise":
        _set_pin("elementwise")
        fixture = {
            "schema_version": 1,
            "task": "OPT-034",
            "status": "rejected",
            "measurement_utc": ab["measurement_utc"],
            "device": ab["device"],
            "compute_capability": ab["compute_capability"],
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "reverted": True,
            "selected_mmv_load_path": "elementwise",
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
            "report_path": "evidence/optimization/opt034-packed-mmv/REPORT.md",
        }
        _write_report(fixture)
        _write_rejection(fixture)
        FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
        validate_result(fixture)
        return

    _set_pin("packed")
    _run([*_common(IMAGE), "make", "build/qw38-cuda-quant-test"])
    llama_p = _run_llama_bench_p()
    quartz_p = _run_quartz_p()
    llama_d128 = _run_llama_decode(128)
    quartz_d128 = _run_quartz_decode(128)
    llama_d2048 = _run_llama_decode(2048)
    quartz_d2048 = _run_quartz_decode(2048)
    fixture = {
        "schema_version": 1,
        "task": "OPT-034",
        "status": "measured",
        "measurement_utc": quartz_d2048.get("measurement_utc", ab["measurement_utc"]),
        "device": ab["device"],
        "compute_capability": ab["compute_capability"],
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": False,
        "selected_mmv_load_path": "packed",
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
        "report_path": "evidence/optimization/opt034-packed-mmv/REPORT.md",
    }
    keep = _keep_predicates(fixture)
    if not keep:
        _set_pin("elementwise")
        fixture["status"] = "rejected"
        fixture["reverted"] = True
        fixture["selected_mmv_load_path"] = "elementwise"
        _write_report(fixture)
        _write_rejection(fixture)
        _run([*_common(IMAGE), "make", "build/qw38-cuda-quant-test"])
    else:
        if REJECTION.is_file():
            REJECTION.unlink()
        _write_report(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    validate_result(fixture)
