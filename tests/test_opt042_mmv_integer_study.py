from __future__ import annotations

import copy
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
CONTRACT = ROOT / "pins/opt042_mmv_integer_study_contract.json"
FIXTURE = ROOT / "fixtures/opt042_mmv_integer_study.json"
OPT041_FIXTURE = ROOT / "fixtures/opt041_fattn_warp_qk.json"
EVIDENCE = ROOT / "evidence/optimization/opt042-mmv-integer-study"
REPORT = EVIDENCE / "REPORT.md"
AB_RAW = EVIDENCE / "mmv-study-raw.txt"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
PREFIX = "QW38_OPT042_MMV_INTEGER_STUDY_RESULT="
OPT041_P = 2076.98315
OPT041_D128 = 26.1599541
OPT041_D2048 = 25.2924843
OPT041_D128_P95 = 38.3730087
OPT041_D128_RUN_P95 = 38.2578201
OPT041_D2048_P95 = 39.6526222
OPT041_D2048_RUN_P95 = 39.5695038
LEGAL_PATHS = {"elementwise", "packed"}
LEGAL_ADMISSIBILITY = {
    "eligible_and_faster",
    "numeric_reject",
    "performance_reject",
}
SYNTHETIC_IDS = ("q4_k_17x256", "q4_k_257x512", "q4k_gate_up", "q4k_down")
REAL_IDS = (
    "real_L0_gate",
    "real_L0_up",
    "real_L0_down",
    "real_L3_gate",
    "real_L3_up",
    "real_L3_down",
    "real_L63_gate",
    "real_L63_up",
    "real_L63_down",
)
PROOF = (
    "claims no performance improvement; fixed CUD-001 envelope; unchanged "
    "FP32-scale Q8 staging; production dispatch unchanged; accepted keep "
    "denominators remain unchanged; does not substitute for the 2K llama.cpp "
    "parity gate; integer-dot is diagnostic only"
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


def selected_path_from_source() -> str:
    text = (ROOT / "cuda/quant_mmv.cu").read_text()
    match = re.search(r'kSelectedMmvLoadPath\[\] = "([^"]+)"', text)
    assert match is not None
    return match.group(1)


def legal_paths_unchanged() -> bool:
    text = (ROOT / "cuda/quant_mmv.cu").read_text()
    return (
        'std::strcmp(load_path, "elementwise") == 0' in text
        and 'std::strcmp(load_path, "packed") == 0' in text
        and "dp4a" not in text
        and "DP4A" not in text
        and "integer" not in text.split("kSelectedMmvLoadPath", 1)[-1][:400]
    )


def q8_staging_unchanged() -> bool:
    text = (ROOT / "cuda/quant_mmv.cu").read_text()
    return (
        "__global__ void quantize_bf16_q8" in text
        and "maximum / 127.0F" in text
        and "roundf(value / scale)" in text
    )


def decode_ffn_uses_production_mmv() -> bool:
    text = (ROOT / "cuda/full_scheduler.cu").read_text()
    execute = text[text.find("cudaError_t execute_ffn") :]
    execute = execute[: execute.find("\ncudaError_t ", 1)]
    matrix = text[text.find("cudaError_t matrix_vector") :]
    matrix = matrix[: matrix.find("\ncudaError_t ", 1)]
    return (
        "matrix_vector(layer.ffn_gate" in execute
        and "matrix_vector(layer.ffn_up" in execute
        and "matrix_vector(layer.ffn_down" in execute
        and "launch_quant_mmv(" in matrix
        and "launch_quant_mmv_prequant" not in execute
        and "launch_integer_mmv" not in execute
    )


def diagnostic_exports_present() -> bool:
    header = (ROOT / "cuda/quant_mmv.h").read_text()
    source = (ROOT / "cuda/quant_mmv.cu").read_text()
    return (
        "launch_quantize_bf16_q8" in header
        and "launch_quant_mmv_prequant" in header
        and "launch_quantize_bf16_q8" in source
        and "launch_quant_mmv_prequant" in source
        and 'kSelectedMmvLoadPath[] = "packed"' in source
    )


def integer_kernel_isolated() -> bool:
    study = (ROOT / "cuda/opt042_mmv_integer_study.cu").read_text()
    production = (ROOT / "cuda/quant_mmv.cu").read_text()
    return (
        "__dp4a" in study
        and "q4k_integer_mmv" in study
        and "__dp4a" not in production
        and "q4k_integer_mmv" not in production
    )


def keep_denominators() -> dict[str, Any]:
    opt041 = json.loads(OPT041_FIXTURE.read_text())
    return {
        "source_keep_fixture": "fixtures/opt041_fattn_warp_qk.json",
        "p": {"quartz_mean_tok_s": opt041["p"]["quartz"]["mean_tok_s"]},
        "d128": {
            "quartz_mean_tok_s": opt041["d128"]["quartz"]["mean_tok_s"],
            "token_latency_p95_ms": opt041["d128"]["quartz"]["token_latency_p95_ms"],
            "run_mean_token_latency_p95_ms": opt041["d128"]["quartz"][
                "run_mean_token_latency_p95_ms"
            ],
        },
        "d2048": {
            "quartz_mean_tok_s": opt041["d2048"]["quartz"]["mean_tok_s"],
            "token_latency_p95_ms": opt041["d2048"]["quartz"]["token_latency_p95_ms"],
            "run_mean_token_latency_p95_ms": opt041["d2048"]["quartz"][
                "run_mean_token_latency_p95_ms"
            ],
        },
    }


def _timed_view(view: dict[str, Any]) -> None:
    assert len(view["samples"]) == 30
    assert len(view["warmup_ms"]) == 3
    mean = sum(float(v) for v in view["samples"]) / 30.0
    assert view["mean_ms"] == pytest.approx(mean, rel=1e-6, abs=1e-6)


def _case(block: dict[str, Any], ident: str) -> None:
    assert block["id"] == ident
    assert "complete" in block["candidates"]["packed"]
    assert "prequant" in block["candidates"]["packed"]
    assert "complete" in block["candidates"]["integer"]
    assert "prequant" in block["candidates"]["integer"]
    for name in ("packed", "integer"):
        cand = block["candidates"][name]
        assert cand["id"] == name
        assert cand["occupancy"] >= 1
        assert cand["launch_ok"] is True
        _timed_view(cand["complete"])
        _timed_view(cand["prequant"])
        vs = cand["vs_host_cud001"]
        assert "max_abs" in vs and "rms" in vs and "nonfinite" in vs
        assert "cosine" in vs and "first_fail" in vs and "relative" in vs
        if cand["cud001_eligible"]:
            assert vs["nonfinite"] == 0
            assert float(vs["max_abs"]) <= 3.0e-4
            assert float(vs["rms"]) <= 2.0e-4
    integer = block["candidates"]["integer"]
    assert "vs_host_integer_byte_equal" in integer
    assert "vs_packed" in integer
    assert "weights" in block["checksums"]
    assert "activation" in block["checksums"]
    assert int(block["checksums"]["weights_bytes"]) > 0
    assert int(block["checksums"]["activation_bytes"]) > 0


def _admissibility(result: dict[str, Any]) -> str:
    for ident in SYNTHETIC_IDS:
        case = result["synthetic"][ident]
        if (
            not case["candidates"]["packed"]["cud001_eligible"]
            or not case["candidates"]["integer"]["cud001_eligible"]
        ):
            return "numeric_reject"
    for ident in REAL_IDS:
        case = result["real"][ident]
        if (
            not case["candidates"]["packed"]["cud001_eligible"]
            or not case["candidates"]["integer"]["cud001_eligible"]
        ):
            return "numeric_reject"
    syn = result["weighted_complete_ms"]["synthetic"]
    real = result["weighted_complete_ms"]["real"]
    if float(syn["integer"]) < float(syn["packed"]) and float(real["integer"]) < float(
        real["packed"]
    ):
        return "eligible_and_faster"
    return "performance_reject"


def _weighted_synthetic(result: dict[str, Any]) -> dict[str, float]:
    packed = 0.0
    integer = 0.0
    weight_sum = 0
    for ident, weight in (("q4k_gate_up", 128), ("q4k_down", 64)):
        case = result["synthetic"][ident]
        weight_sum += weight
        packed += weight * float(case["candidates"]["packed"]["complete"]["mean_ms"])
        integer += weight * float(case["candidates"]["integer"]["complete"]["mean_ms"])
    return {"packed": packed / weight_sum, "integer": integer / weight_sum}


def _weighted_real(result: dict[str, Any]) -> dict[str, float]:
    packed_layers: list[float] = []
    integer_layers: list[float] = []
    for layer in (0, 3, 63):
        packed = 0.0
        integer = 0.0
        weight_sum = 0
        for proj, weight in (("gate", 128), ("up", 128), ("down", 64)):
            case = result["real"][f"real_L{layer}_{proj}"]
            weight_sum += weight
            packed += weight * float(
                case["candidates"]["packed"]["complete"]["mean_ms"]
            )
            integer += weight * float(
                case["candidates"]["integer"]["complete"]["mean_ms"]
            )
        packed_layers.append(packed / weight_sum)
        integer_layers.append(integer / weight_sum)
    return {
        "packed": sum(packed_layers) / 3.0,
        "integer": sum(integer_layers) / 3.0,
    }


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-042"
    assert result["status"] == "measured"
    assert result["status"] not in ("scout", "source_inspected")
    assert contract["yardstick"] == "q4k_ffn_decode_mmv_integer_admissibility"
    assert contract["claims_performance_improvement"] is False
    assert contract["publishes_successor_oracle"] is False
    assert contract["promote_installs_production_path"] is False
    assert result["claims_performance_improvement"] is False
    assert result["publishes_successor_oracle"] is False
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    assert result["q8_0_mixer_sibling"] == "not_run"
    assert result["q6k_logits_sibling"] == "not_run"
    assert result["reverted"] is False
    assert contract["cud001"]["maximum_absolute_error"] == 0.0003
    assert contract["cud001"]["maximum_rms_error"] == 0.0002
    assert contract["cud001"]["q8_staging_exact"] is True
    assert contract["mmq_association_is_informational"] is True
    assert contract["candidates"] == ["packed", "integer"]
    assert contract["legal_mmv_load_paths"] == ["elementwise", "packed"]
    assert contract["warmups"] == 3 and contract["measured"] == 30
    assert contract["real_layers"] == [0, 3, 63]
    assert result["selected_mmv_load_path"] == "packed"
    assert selected_path_from_source() == "packed"
    assert set(LEGAL_PATHS) == set(contract["legal_mmv_load_paths"])
    assert legal_paths_unchanged()
    assert q8_staging_unchanged()
    assert decode_ffn_uses_production_mmv()
    assert diagnostic_exports_present()
    assert integer_kernel_isolated()
    expected_keep = keep_denominators()
    assert result["accepted_keep_denominators"] == expected_keep
    assert contract["accepted_keep_denominators"] == expected_keep
    assert contract["source_keep_fixture"] == "fixtures/opt041_fattn_warp_qk.json"
    assert expected_keep["p"]["quartz_mean_tok_s"] == OPT041_P
    assert expected_keep["d128"]["quartz_mean_tok_s"] == OPT041_D128
    assert expected_keep["d2048"]["quartz_mean_tok_s"] == OPT041_D2048
    assert expected_keep["d128"]["token_latency_p95_ms"] == OPT041_D128_P95
    assert expected_keep["d128"]["run_mean_token_latency_p95_ms"] == OPT041_D128_RUN_P95
    assert expected_keep["d2048"]["token_latency_p95_ms"] == OPT041_D2048_P95
    assert (
        expected_keep["d2048"]["run_mean_token_latency_p95_ms"] == OPT041_D2048_RUN_P95
    )
    assert set(result["synthetic"]) == set(SYNTHETIC_IDS)
    assert set(result["real"]) == set(REAL_IDS)
    for ident in SYNTHETIC_IDS:
        _case(result["synthetic"][ident], ident)
        source = result["synthetic"][ident]["activation_source"]
        assert source == "unit_normal"
    for ident in REAL_IDS:
        _case(result["real"][ident], ident)
        assert result["real"][ident]["activation_source"] == "post_prefix_hidden"
    syn_w = _weighted_synthetic(result)
    real_w = _weighted_real(result)
    assert result["weighted_complete_ms"]["synthetic"]["packed"] == pytest.approx(
        syn_w["packed"], rel=1e-6, abs=1e-6
    )
    assert result["weighted_complete_ms"]["synthetic"]["integer"] == pytest.approx(
        syn_w["integer"], rel=1e-6, abs=1e-6
    )
    assert result["weighted_complete_ms"]["real"]["packed"] == pytest.approx(
        real_w["packed"], rel=1e-6, abs=1e-6
    )
    assert result["weighted_complete_ms"]["real"]["integer"] == pytest.approx(
        real_w["integer"], rel=1e-6, abs=1e-6
    )
    expected = _admissibility(result)
    assert result["admissibility"] == expected
    assert result["admissibility"] in LEGAL_ADMISSIBILITY
    assert result["promote_to_production_ab"] is (expected == "eligible_and_faster")
    assert result["claims_performance_improvement"] is False
    for warp in ("4", "8", "16"):
        assert int(result["occupancy"]["integer"][warp]) >= 1
        assert int(result["occupancy"]["packed"][warp]) >= 1
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
    assert "OPT-031" not in json.dumps(result)
    assert "OPT-031" not in json.dumps(contract)


def _timed_stub(mean: float) -> dict[str, Any]:
    return {"mean_ms": mean, "samples": [mean] * 30, "warmup_ms": [mean] * 3}


def _candidate_stub(
    ident: str, mean: float, *, eligible: bool = True, byte_equal: bool = True
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": ident,
        "occupancy": 2,
        "launch_ok": True,
        "staging_equal": True,
        "cud001_eligible": eligible,
        "complete": _timed_stub(mean),
        "prequant": _timed_stub(mean * 0.8),
        "vs_host_cud001": {
            "max_abs": 0.0 if eligible else 0.01,
            "rms": 0.0 if eligible else 0.01,
            "nonfinite": 0,
            "cosine": 1.0,
            "first_fail": -1,
            "relative": {
                "max": 0.0,
                "mean": 0.0,
                "gt_1e-3": 0,
                "gt_1e-2": 0,
                "gt_1e-1": 0,
            },
            "mmq_association_fail": 0,
        },
    }
    if ident == "integer":
        payload["vs_host_integer_byte_equal"] = byte_equal
        payload["vs_packed"] = {
            "max_abs": 0.0,
            "rms": 0.0,
            "nonfinite": 0,
            "cosine": 1.0,
            "first_fail": -1,
        }
    return payload


def _case_stub(
    ident: str,
    *,
    rows: int,
    columns: int,
    warps: int,
    weight: int,
    probe: bool,
    source: str,
    packed_ms: float,
    integer_ms: float,
    eligible: bool = True,
) -> dict[str, Any]:
    return {
        "id": ident,
        "rows": rows,
        "columns": columns,
        "warps": warps,
        "weight": weight,
        "probe": probe,
        "activation_source": source,
        "checksums": {
            "weights": "ab" * 32,
            "activation": "cd" * 32,
            "weights_bytes": 144,
            "activation_bytes": columns * 2,
        },
        "staging_equal": True,
        "candidates": {
            "packed": _candidate_stub("packed", packed_ms, eligible=eligible),
            "integer": _candidate_stub("integer", integer_ms, eligible=eligible),
        },
    }


def _fixture_stub(
    *, integer_faster: bool = True, eligible: bool = True
) -> dict[str, Any]:
    packed_ms = 1.0
    integer_ms = 0.8 if integer_faster else 1.2
    synthetic = {
        "q4_k_17x256": _case_stub(
            "q4_k_17x256",
            rows=17,
            columns=256,
            warps=4,
            weight=0,
            probe=True,
            source="unit_normal",
            packed_ms=0.01,
            integer_ms=0.008,
            eligible=eligible,
        ),
        "q4_k_257x512": _case_stub(
            "q4_k_257x512",
            rows=257,
            columns=512,
            warps=8,
            weight=0,
            probe=True,
            source="unit_normal",
            packed_ms=0.02,
            integer_ms=0.016,
            eligible=eligible,
        ),
        "q4k_gate_up": _case_stub(
            "q4k_gate_up",
            rows=17408,
            columns=5120,
            warps=8,
            weight=128,
            probe=False,
            source="unit_normal",
            packed_ms=packed_ms,
            integer_ms=integer_ms,
            eligible=eligible,
        ),
        "q4k_down": _case_stub(
            "q4k_down",
            rows=5120,
            columns=17408,
            warps=16,
            weight=64,
            probe=False,
            source="unit_normal",
            packed_ms=packed_ms,
            integer_ms=integer_ms,
            eligible=eligible,
        ),
    }
    real = {}
    for layer in (0, 3, 63):
        for proj, rows, columns, warps, weight in (
            ("gate", 17408, 5120, 8, 128),
            ("up", 17408, 5120, 8, 128),
            ("down", 5120, 17408, 16, 64),
        ):
            ident = f"real_L{layer}_{proj}"
            real[ident] = _case_stub(
                ident,
                rows=rows,
                columns=columns,
                warps=warps,
                weight=weight,
                probe=False,
                source="post_prefix_hidden",
                packed_ms=packed_ms,
                integer_ms=integer_ms,
                eligible=eligible,
            )
    if not eligible:
        admissibility = "numeric_reject"
        promote = False
    elif integer_faster:
        admissibility = "eligible_and_faster"
        promote = True
    else:
        admissibility = "performance_reject"
        promote = False
    return {
        "schema_version": 1,
        "task": "OPT-042",
        "status": "measured",
        "measurement_utc": "2026-09-10T00:00:00Z",
        "device": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "claims_performance_improvement": False,
        "publishes_successor_oracle": False,
        "promote_to_production_ab": promote,
        "admissibility": admissibility,
        "selected_mmv_load_path": "packed",
        "reverted": False,
        "accepted_keep_denominators": keep_denominators(),
        "synthetic": synthetic,
        "real": real,
        "weighted_complete_ms": {
            "synthetic": {"packed": packed_ms, "integer": integer_ms},
            "real": {"packed": packed_ms, "integer": integer_ms},
        },
        "staging_equal": True,
        "occupancy": {
            "integer": {"4": 2, "8": 2, "16": 2},
            "packed": {"4": 2, "8": 2, "16": 2},
        },
        "q8_0_mixer_sibling": "not_run",
        "q6k_logits_sibling": "not_run",
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt042-mmv-integer-study/REPORT.md",
    }


def _write_report(fixture: dict[str, Any]) -> None:
    syn = fixture["weighted_complete_ms"]["synthetic"]
    real = fixture["weighted_complete_ms"]["real"]
    text = f"""# OPT-042 — Study integer decode MMV admissibility

## Claim labels and proof limits

Live exclusive-RTX-5090 diagnostic of packed FP32 Q4_K decode MMV versus a Q4_K integer-dot (`__dp4a`) candidate. Proof boundary: claims no performance improvement; fixed CUD-001 envelope; unchanged FP32-scale Q8 staging; production dispatch unchanged; accepted keep denominators remain unchanged; does not substitute for the 2K llama.cpp parity gate; integer-dot is diagnostic only.

## Classification

- admissibility: `{fixture["admissibility"]}`
- promote_to_production_ab: {json.dumps(fixture["promote_to_production_ab"])}
- claims_performance_improvement: false
- publishes_successor_oracle: false
- selected_mmv_load_path: `{fixture["selected_mmv_load_path"]}`
- reverted: {json.dumps(fixture["reverted"])}

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `{GGUF_SHA}` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp revision | `{LLAMA_REV}` |
| Candidates | packed then integer, 3 warm + 30 alternating pairs |
| Views | complete (stage+MMV) and prequant (MMV only) |
| Envelope | CUD-001 max abs 3e-4, RMS 2e-4, zero nonfinites |
| Keep denominators | copied from OPT-041; unchanged |
| Nsight | not_used |

## Measured sitting

- Device: {fixture["device"]} compute {fixture["compute_capability"]}
- measurement_utc: {fixture["measurement_utc"]}
- synthetic weighted complete ms packed={syn["packed"]} integer={syn["integer"]}
- real weighted complete ms packed={real["packed"]} integer={real["integer"]}
- staging_equal: {json.dumps(fixture["staging_equal"])}
- occupancy integer: {json.dumps(fixture["occupancy"]["integer"])}
- q8_0_mixer_sibling: not_run
- q6k_logits_sibling: not_run
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false
"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def test_opt042_contract_and_source_pins() -> None:
    contract = _contract()
    expected = keep_denominators()
    assert contract["accepted_keep_denominators"] == expected
    assert contract["cud001"]["maximum_absolute_error"] == 0.0003
    assert contract["cud001"]["maximum_rms_error"] == 0.0002
    assert contract["claims_performance_improvement"] is False
    assert contract["publishes_successor_oracle"] is False
    assert contract["promote_installs_production_path"] is False
    assert contract["mmq_association_is_informational"] is True
    assert contract["legal_mmv_load_paths"] == ["elementwise", "packed"]
    assert contract["selected_mmv_load_path"] == "packed"
    assert selected_path_from_source() == "packed"
    assert legal_paths_unchanged()
    assert q8_staging_unchanged()
    assert decode_ffn_uses_production_mmv()
    assert diagnostic_exports_present()
    assert integer_kernel_isolated()
    quant = json.loads((ROOT / "pins/cuda_quant_contract.json").read_text())
    assert quant["admission"]["maximum_absolute_error"] == 0.0003
    assert quant["admission"]["maximum_rms_error"] == 0.0002


def test_opt042_validator_rejects_inadmissible_evidence() -> None:
    fixture = _fixture_stub(integer_faster=False, eligible=True)
    mutations: list[dict[str, Any]] = []

    def add(mutate) -> None:
        changed = copy.deepcopy(fixture)
        mutate(changed)
        mutations.append(changed)

    add(lambda x: x.__setitem__("substitutes_for_opt016", True))
    add(lambda x: x.__setitem__("nsight_systems", "/tmp/capture.nsys-rep"))
    add(lambda x: x.__setitem__("report_path", "/tmp/opt042/REPORT.md"))
    add(lambda x: x.__setitem__("status", "scout"))
    add(lambda x: x.__setitem__("selected_mmv_load_path", "integer"))
    add(lambda x: x.__setitem__("claims_performance_improvement", True))
    add(lambda x: x.__setitem__("publishes_successor_oracle", True))
    add(lambda x: x["real"].pop("real_L63_down"))
    add(
        lambda x: x["synthetic"]["q4k_gate_up"]["candidates"]["integer"].pop("prequant")
    )
    add(lambda x: x.__setitem__("admissibility", "eligible_and_faster"))
    add(lambda x: x.__setitem__("promote_to_production_ab", True))
    loosened = copy.deepcopy(fixture)
    loosened["synthetic"]["q4_k_17x256"]["candidates"]["integer"]["cud001_eligible"] = (
        True
    )
    loosened["synthetic"]["q4_k_17x256"]["candidates"]["integer"]["vs_host_cud001"][
        "max_abs"
    ] = 0.01
    mutations.append(loosened)
    mmq_gate = copy.deepcopy(fixture)
    mmq_gate["admissibility"] = "eligible_and_faster"
    mmq_gate["promote_to_production_ab"] = True
    mmq_gate["weighted_complete_ms"]["real"]["integer"] = 2.0
    mutations.append(mmq_gate)
    coupled = copy.deepcopy(fixture)
    coupled["coupled"] = "OPT-031"
    mutations.append(coupled)

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    report_prev = REPORT.read_text() if REPORT.is_file() else None
    raw_prev = AB_RAW.read_text() if AB_RAW.is_file() else None
    _write_report(fixture)
    AB_RAW.write_text("raw\n")
    try:
        for changed in mutations:
            with pytest.raises((AssertionError, KeyError)):
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


def test_opt042_admissibility_rule() -> None:
    faster = _fixture_stub(integer_faster=True, eligible=True)
    assert _admissibility(faster) == "eligible_and_faster"
    slower = _fixture_stub(integer_faster=False, eligible=True)
    assert _admissibility(slower) == "performance_reject"
    numeric = _fixture_stub(integer_faster=True, eligible=False)
    assert _admissibility(numeric) == "numeric_reject"
    mixed = _fixture_stub(integer_faster=True, eligible=True)
    mixed["weighted_complete_ms"]["real"]["integer"] = 2.0
    mixed["real"]["real_L0_gate"]["candidates"]["integer"]["complete"] = _timed_stub(
        2.0
    )
    mixed["real"]["real_L0_up"]["candidates"]["integer"]["complete"] = _timed_stub(2.0)
    mixed["real"]["real_L0_down"]["candidates"]["integer"]["complete"] = _timed_stub(
        2.0
    )
    mixed["real"]["real_L3_gate"]["candidates"]["integer"]["complete"] = _timed_stub(
        2.0
    )
    mixed["real"]["real_L3_up"]["candidates"]["integer"]["complete"] = _timed_stub(2.0)
    mixed["real"]["real_L3_down"]["candidates"]["integer"]["complete"] = _timed_stub(
        2.0
    )
    mixed["real"]["real_L63_gate"]["candidates"]["integer"]["complete"] = _timed_stub(
        2.0
    )
    mixed["real"]["real_L63_up"]["candidates"]["integer"]["complete"] = _timed_stub(2.0)
    mixed["real"]["real_L63_down"]["candidates"]["integer"]["complete"] = _timed_stub(
        2.0
    )
    assert _admissibility(mixed) == "performance_reject"


def test_opt042_fixture_connected() -> None:
    if not FIXTURE.is_file():
        pytest.skip("OPT-042 fixture is written by the exclusive CUDA sitting")
    validate_result(json.loads(FIXTURE.read_text()))


def test_opt042_rejects_dp4a_production_path() -> None:
    text = (ROOT / "cuda/quant_mmv.cu").read_text()
    assert 'kSelectedMmvLoadPath[] = "packed"' in text
    assert "dp4a" not in text
    contract = _contract()
    assert "dp4a" not in contract["legal_mmv_load_paths"]
    assert contract["selected_mmv_load_path"] == "packed"


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


def _nvcc(source: str, output: str) -> list[str]:
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
        *NVCC_OBJECTS,
        "-o",
        output,
    ]


def _parse_prefixed(text: str, prefix: str) -> dict[str, Any]:
    for line in text.splitlines():
        if line.startswith(prefix):
            return json.loads(line.removeprefix(prefix))
    raise AssertionError(f"{prefix} was not found\n{text}")


@pytest.mark.skipif(
    os.environ.get("QW38_RUN_CUDA_TESTS") != "1",
    reason="native CUDA sitting is gated on QW38_RUN_CUDA_TESTS=1",
)
def test_opt042_cuda_sitting() -> None:
    assert MODEL.is_file()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    commands = [
        [*_common(IMAGE), "make", "build/qw38-cuda-timing-test"],
        _nvcc(
            "cuda/opt042_mmv_integer_study.cu",
            "build/qw38-cuda-opt042-mmv-integer-study",
        ),
        [
            *_common(IMAGE),
            "./build/qw38-cuda-opt042-mmv-integer-study",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
            "evidence/optimization/opt042-mmv-integer-study/mmv-study-raw.txt",
        ],
    ]
    outputs: list[str] = []
    for command in commands:
        outputs.append(_run(command).stdout)
    record = _parse_prefixed(outputs[-1], PREFIX)
    assert "status=passed" in outputs[-1]
    FIXTURE.write_text(json.dumps(record, indent=2) + "\n")
    _write_report(record)
    validate_result(json.loads(FIXTURE.read_text()))
