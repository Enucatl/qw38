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
CONTRACT = ROOT / "pins/opt041_fattn_warp_qk_contract.json"
FIXTURE = ROOT / "fixtures/opt041_fattn_warp_qk.json"
OPT040_FIXTURE = ROOT / "fixtures/opt040_gdn_shared_inverse.json"
EVIDENCE = ROOT / "evidence/optimization/opt041-fattn-warp-qk"
REPORT = EVIDENCE / "REPORT.md"
REJECTION = EVIDENCE / "REJECTION.md"
AB_RAW = EVIDENCE / "warp-qk-ab-raw.txt"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
AB_PREFIX = "QW38_FATTN_WARP_QK_AB_RESULT="
P_PREFIX = "QW38_PREFILL_4K_ORACLE_RESULT="
DECODE_PREFIX = "QW38_DECODE_ORACLE_RESULT="
LLAMA_DECODE_PREFIX = "QW38_LLAMA_DECODE_ORACLE_RESULT="
OPT040_P = 2060.62183
OPT040_D128 = 26.1689129
OPT040_D2048 = 25.2998886
OPT040_D128_P95 = 38.350193
OPT040_D128_RUN_P95 = 38.2333679
OPT040_D2048_P95 = 39.6363754
OPT040_D2048_RUN_P95 = 39.55233
STALE_OPT034_P = 1869.84412
STALE_OPT034_D128 = 25.3816128
STALE_OPT034_D2048 = 20.169548
LEGAL_PATHS = {"cparts", "warp_microtile"}
CORRECTNESS_TOKENS = [16, 17, 31, 32, 33, 63, 64, 65, 512, 2048, 4096]
CORRECTNESS_STARTS = [0, 1, 31, 128, 2048]
PROOF = (
    "byte-equal quality attention outputs; frozen attention gates; "
    "lower complete component time; improved P; "
    "95% throughput floors; 105% p95 ceilings; "
    "does not substitute for the 2K llama.cpp parity gate"
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


def pin_from_source() -> str:
    text = (ROOT / "cuda/fattn_mma_f16.cuh").read_text()
    match = re.search(r'kSelectedQKPath\[\] = "([^"]+)"', text)
    assert match is not None
    return match.group(1)


def vkq_from_source() -> str:
    text = (ROOT / "cuda/fattn_mma_f16.cuh").read_text()
    match = re.search(r'kSelectedVkqAccum\[\] = "([^"]+)"', text)
    assert match is not None
    return match.group(1)


def pv_from_source() -> str:
    text = (ROOT / "cuda/fattn_mma_f16.cuh").read_text()
    match = re.search(r'kSelectedPvPath\[\] = "([^"]+)"', text)
    assert match is not None
    return match.group(1)


def frozen_fattn_pins() -> bool:
    text = (ROOT / "cuda/fattn_mma_f16.cuh").read_text()
    return (
        'kSelectedFattnPath[] = "stream_k"' in text
        and 'kSelectedPersistentFattnPath[] = "off"' in text
        and "kSelectedAttentionMmaQueryRows = 16" in text
        and "kFattnNcols2 = 2" in text
        and "kFattnNbatchFa = 32" in text
        and 'kSelectedVkqAccum[] = "registers"' in text
        and 'kSelectedPvPath[] = "mma"' in text
        and "fattn_mma_quality_typed<16, true, 2, 2, true, true, true>" in text
        and "fattn_mma_quality_typed<16, true, 2, 2, true, true, false>" in text
        and "fattn_qk_mma_k16" in text
        and "fattn_stream_k_combine_kernel" in text
    )


def cparts_slab_retained() -> bool:
    text = (ROOT / "cuda/fattn_mma_f16.cuh").read_text()
    return (
        "nwarps) * 32ull * 4ull *" in text
        and "return kv + q + scores + stats + scratch + cparts + rescale;" in text
        and "[[maybe_unused]] float* cparts = rescale + kNcols;" in text
    )


def virtual_warp_order_retained() -> bool:
    text = (ROOT / "cuda/fattn_mma_f16.cuh").read_text()
    return (
        "for (int vwarp = 0; vwarp < kNwarps; ++vwarp)" in text
        and "for (int k0 = vwarp * 16; k0 < kWidth; k0 += kNwarps * 16)" in text
        and "for (int k0 = warp * 16; k0 < kWidth; k0 += kNwarps * 16)" in text
        and "if (warp != tile_id % kNwarps) continue;" in text
    )


def decode_vec_unchanged() -> bool:
    text = (ROOT / "cuda/attention_decode.cu").read_text()
    return 'kSelectedDecodeAttentionVec[] = "warp_query"' in text


def gdn_inverse_unchanged() -> bool:
    text = (ROOT / "cuda/gdn_fused_quality.cuh").read_text()
    return (
        'kSelectedGdnInversePath[] = "shared"' in text
        and 'kSelectedGdnFusePath[] = "off"' in text
    )


def skinny_path_unchanged() -> bool:
    text = (ROOT / "cuda/quant_mmq_mma.cuh").read_text()
    return 'kSelectedSkinnyMixerPath[] = "mma_i32_j128"' in text


def ffn_shared_y_unchanged() -> bool:
    text = (ROOT / "cuda/quant_mmq_mma.cuh").read_text()
    return 'kSelectedFfnPath[] = "shared_y_swiglu_q8"' in text


def ncols1_unchanged() -> bool:
    text = (ROOT / "cuda/fattn_mma_f16.cuh").read_text()
    return "kSelectedAttentionMmaQueryRows = 16" in text


def _ab_candidate(block: dict[str, Any], ident: str) -> None:
    cand = block["candidates"][ident]
    assert cand["id"] == ident
    assert cand["occupancy"] >= 1
    assert cand["launch_ok"] is True
    assert len(cand["samples"]) == 30
    assert len(cand["warmup_ms"]) == 3
    mean = sum(float(v) for v in cand["samples"]) / 30.0
    assert cand["mean_ms"] == pytest.approx(mean, rel=1e-6, abs=1e-6)
    assert cand["vs_tiled"]["nonfinite"] == 0
    if cand["eligible"]:
        assert cand["outputs_byte_equal"] is True
        assert cand["candidate_exact"] is True
        assert cand["committed_unchanged"] is True
        assert cand["scratch_unchanged"] is True
        assert cand["vs_tiled"]["max_abs"] <= 5.0e-5
        assert cand["vs_tiled"]["rms"] <= 5.0e-6


def _select_component_winner(block: dict[str, Any]) -> str:
    cparts = block["candidates"]["cparts"]
    warp = block["candidates"]["warp_microtile"]
    if (
        warp["eligible"]
        and cparts["eligible"]
        and float(warp["mean_ms"]) < float(cparts["mean_ms"])
    ):
        return "warp_microtile"
    return "cparts"


def _select_install(result: dict[str, Any]) -> str:
    if not result["correctness"]["all_eligible"]:
        return "cparts"
    if not result["ab_p4096"]["candidates"]["warp_microtile"]["eligible"]:
        return "cparts"
    if _select_component_winner(result["ab_p4096"]) != "warp_microtile":
        return "cparts"
    return "warp_microtile"


def _keep_predicates(result: dict[str, Any]) -> bool:
    contract = _contract()
    if result["selected_qk_path"] != "warp_microtile":
        return False
    if result["selected_vkq_accum"] != "registers":
        return False
    if result["selected_pv_path"] != "mma":
        return False
    if result["reverted"] is True or result["keep_sitting_skipped"] is True:
        return False
    if result["status"] != "measured":
        return False
    if _select_install(result) != "warp_microtile":
        return False
    if result["p"] is None or result["d128"] is None or result["d2048"] is None:
        return False
    if (
        float(result["p"]["quartz"]["mean_tok_s"])
        <= contract["opt040_quartz_p_mean_tok_s"]
    ):
        return False
    if (
        float(result["d128"]["quartz"]["mean_tok_s"])
        < 0.95 * contract["opt040_quartz_d128_mean_tok_s"]
    ):
        return False
    if (
        float(result["d2048"]["quartz"]["mean_tok_s"])
        < 0.95 * contract["opt040_quartz_d2048_mean_tok_s"]
    ):
        return False
    if (
        result["d128"]["quartz"]["token_latency_p95_ms"]
        > 1.05 * contract["opt040_quartz_d128_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d128"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * contract["opt040_quartz_d128_run_mean_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d2048"]["quartz"]["token_latency_p95_ms"]
        > 1.05 * contract["opt040_quartz_d2048_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d2048"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * contract["opt040_quartz_d2048_run_mean_token_latency_p95_ms"]
    ):
        return False
    return True


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-041"
    assert result["status"] in ("measured", "rejected")
    assert result["status"] not in ("scout", "source_inspected")
    assert contract["yardstick"] == "prefill_attention_warp_qk_microtiles"
    assert contract["opt040_quartz_p_mean_tok_s"] == OPT040_P
    assert contract["opt040_quartz_d128_mean_tok_s"] == OPT040_D128
    assert contract["opt040_quartz_d2048_mean_tok_s"] == OPT040_D2048
    assert contract["opt040_quartz_d128_token_latency_p95_ms"] == OPT040_D128_P95
    assert (
        contract["opt040_quartz_d128_run_mean_token_latency_p95_ms"]
        == OPT040_D128_RUN_P95
    )
    assert contract["opt040_quartz_d2048_token_latency_p95_ms"] == OPT040_D2048_P95
    assert (
        contract["opt040_quartz_d2048_run_mean_token_latency_p95_ms"]
        == OPT040_D2048_RUN_P95
    )
    assert contract["historical_opt034_quartz_p_mean_tok_s"] == STALE_OPT034_P
    assert contract["historical_opt034_quartz_d128_mean_tok_s"] == STALE_OPT034_D128
    assert contract["historical_opt034_quartz_d2048_mean_tok_s"] == STALE_OPT034_D2048
    assert contract["use_as_keep_denominator"] is False
    assert contract["opt005"]["bf16_max_abs"] == 5.0e-5
    assert contract["opt005"]["bf16_rms"] == 5.0e-6
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    assert "fattn_path" not in result
    assert "starts_opt042" not in result
    assert "integer_decode_mmv" not in result
    assert "selected_gdn_fuse_path" not in result
    assert "mixer_gdn_graphs" not in result
    path = result["selected_qk_path"]
    assert path in LEGAL_PATHS
    assert result["selected_vkq_accum"] == "registers"
    assert result["selected_pv_path"] == "mma"
    assert contract["ab"]["candidates"] == ["cparts", "warp_microtile"]
    assert contract["ab"]["ties_retain"] == "cparts"
    assert contract["ab"]["byte_equal_required"] is True
    assert contract["ab"]["envelope_vs_tiled_required"] is True
    assert contract["attention_mma_query_rows"] == 16
    assert contract["ncols2"] == 2
    assert contract["kv_tile"] == 32
    assert contract["dual_f16_q"] is True
    assert contract["virtual_warps"] == 4
    assert contract["microtiles"] == 8
    assert contract["cparts_shared_bytes_retained"] is True
    assert contract["grid_z"] == 2
    assert _select_component_winner(result["ab_p4096"]) == result["ab_p4096"]["winner"]
    _ab_candidate(result["ab_p4096"], "cparts")
    _ab_candidate(result["ab_p4096"], "warp_microtile")
    assert result["ab_p4096"]["token_count"] == 4096
    assert result["correctness"]["token_counts"] == CORRECTNESS_TOKENS
    assert result["correctness"]["starts"] == CORRECTNESS_STARTS
    expected_keys = {
        f"{start}:{token}"
        for start in CORRECTNESS_STARTS
        for token in CORRECTNESS_TOKENS
    }
    assert set(result["correctness"]["cases"]) == expected_keys
    for start in CORRECTNESS_STARTS:
        for token in CORRECTNESS_TOKENS:
            case = result["correctness"]["cases"][f"{start}:{token}"]
            assert case["start_position"] == start
            assert case["token_count"] == token
            if case["eligible"]:
                assert case["outputs_byte_equal"] is True
                assert case["candidate_exact"] is True
                assert case["committed_unchanged"] is True
                assert case["scratch_unchanged"] is True
                assert case["vs_cparts"]["nonfinite"] == 0
    installed = _select_install(result)
    assert pin_from_source() == path == contract["selected_qk_path"]
    assert vkq_from_source() == "registers" == contract["kSelectedVkqAccum"]
    assert pv_from_source() == "mma" == contract["kSelectedPvPath"]
    assert frozen_fattn_pins()
    assert cparts_slab_retained()
    assert virtual_warp_order_retained()
    assert decode_vec_unchanged()
    assert gdn_inverse_unchanged()
    assert skinny_path_unchanged()
    assert ffn_shared_y_unchanged()
    assert ncols1_unchanged()
    keep = _keep_predicates(result)
    if result["status"] == "measured":
        assert keep
        assert result["reverted"] is False
        assert result["keep_sitting_skipped"] is False
        assert path == "warp_microtile" == installed
        assert result["p"]["quartz"]["mean_tok_s"] > OPT040_P
        assert not REJECTION.is_file()
    else:
        assert not keep
        assert result["reverted"] is True
        assert path == "cparts"
        assert REJECTION.is_file()
        rejection = REJECTION.read_text()
        assert "cparts" in rejection
        if result["keep_sitting_skipped"]:
            assert installed == "cparts"
            assert result["p"] is None and result["d128"] is None
            assert result["d2048"] is None
        else:
            assert installed == "warp_microtile"
            assert result["p"] is not None
            assert str(result["p"]["quartz"]["mean_tok_s"]) in rejection
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


def _candidate_stub(
    ident: str, mean: float, *, eligible: bool = True
) -> dict[str, Any]:
    envelope = {"max_abs": 1.0e-9, "rms": 1.0e-10, "nonfinite": 0}
    return {
        "id": ident,
        "mean_ms": mean,
        "samples": [mean] * 30,
        "warmup_ms": [mean] * 3,
        "occupancy": 2,
        "launch_ok": True,
        "eligible": eligible,
        "scratch_unchanged": True,
        "committed_unchanged": True,
        "candidate_exact": True,
        "outputs_byte_equal": True,
        "include_combine": True,
        "vs_tiled": dict(envelope),
        "vs_cparts": dict(envelope),
    }


def _ab_stub(winner: str, means: dict[str, float]) -> dict[str, Any]:
    return {
        "token_count": 4096,
        "winner": winner,
        "win": winner == "warp_microtile",
        "include_combine": True,
        "candidates": {
            ident: _candidate_stub(ident, means[ident])
            for ident in ("cparts", "warp_microtile")
        },
    }


def _correctness_stub(*, warp_ok: bool = True) -> dict[str, Any]:
    cases = {}
    for start in CORRECTNESS_STARTS:
        for token in CORRECTNESS_TOKENS:
            cases[f"{start}:{token}"] = {
                "start_position": start,
                "token_count": token,
                "launch_ok": True,
                "outputs_byte_equal": True,
                "scratch_unchanged": True,
                "committed_unchanged": True,
                "candidate_exact": True,
                "eligible": warp_ok,
                "vs_cparts": {"max_abs": 1.0e-9, "rms": 1.0e-10, "nonfinite": 0},
            }
    return {
        "token_counts": CORRECTNESS_TOKENS,
        "starts": CORRECTNESS_STARTS,
        "cases": cases,
        "all_eligible": warp_ok,
    }


def _reject_fixture() -> dict[str, Any]:
    means = {"cparts": 1.0, "warp_microtile": 2.0}
    return {
        "schema_version": 1,
        "task": "OPT-041",
        "status": "rejected",
        "measurement_utc": "2026-09-10T00:00:00Z",
        "device": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": True,
        "selected_qk_path": "cparts",
        "selected_vkq_accum": "registers",
        "selected_pv_path": "mma",
        "ab_p4096": _ab_stub("cparts", means),
        "correctness": _correctness_stub(),
        "keep_sitting_skipped": True,
        "p": None,
        "d128": None,
        "d2048": None,
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt041-fattn-warp-qk/REPORT.md",
    }


def _keep_tok_s() -> dict[str, Any]:
    return {
        "p": {"quartz": {"mean_tok_s": OPT040_P + 1.0}},
        "d128": {
            "quartz": {
                "mean_tok_s": OPT040_D128,
                "token_latency_p95_ms": OPT040_D128_P95,
                "run_mean_token_latency_p95_ms": OPT040_D128_RUN_P95,
            }
        },
        "d2048": {
            "quartz": {
                "mean_tok_s": OPT040_D2048,
                "token_latency_p95_ms": OPT040_D2048_P95,
                "run_mean_token_latency_p95_ms": OPT040_D2048_RUN_P95,
            }
        },
    }


def test_opt041_contract_and_source_pins() -> None:
    contract = _contract()
    opt040 = json.loads(OPT040_FIXTURE.read_text())
    opt034 = json.loads((ROOT / "fixtures/opt034_packed_mmv.json").read_text())
    assert opt040["p"]["quartz"]["mean_tok_s"] == OPT040_P
    assert opt040["d128"]["quartz"]["mean_tok_s"] == OPT040_D128
    assert opt040["d2048"]["quartz"]["mean_tok_s"] == OPT040_D2048
    assert opt034["p"]["quartz"]["mean_tok_s"] == STALE_OPT034_P
    assert contract["opt040_quartz_p_mean_tok_s"] == OPT040_P
    assert contract["opt040_quartz_d128_mean_tok_s"] == OPT040_D128
    assert contract["opt040_quartz_d2048_mean_tok_s"] == OPT040_D2048
    assert contract["historical_opt034_quartz_p_mean_tok_s"] == STALE_OPT034_P
    assert contract["use_as_keep_denominator"] is False
    assert contract["ab"]["candidates"] == ["cparts", "warp_microtile"]
    assert contract["ab"]["timed_token_counts"] == [4096]
    assert contract["ab"]["correctness_token_counts"] == CORRECTNESS_TOKENS
    assert contract["ab"]["correctness_starts"] == CORRECTNESS_STARTS
    assert contract["ab"]["ties_retain"] == "cparts"
    assert contract["ab"]["byte_equal_required"] is True
    assert contract["owns_opt016_parity_gate"] is False
    assert contract["substitutes_for_opt016"] is False
    assert contract["llama_bench_decode_is_informational"] is True
    tiled = json.loads((ROOT / "pins/cuda_tiled_attention_contract.json").read_text())
    assert tiled["proof_limits"]["bf16_max_abs"] == 5.0e-5
    assert tiled["proof_limits"]["bf16_rms"] == 5.0e-6
    assert pin_from_source() in LEGAL_PATHS
    assert vkq_from_source() == "registers"
    assert pv_from_source() == "mma"
    assert frozen_fattn_pins()
    assert cparts_slab_retained()
    assert virtual_warp_order_retained()
    assert decode_vec_unchanged()
    assert gdn_inverse_unchanged()
    header = (ROOT / "cuda/attention_decode.h").read_text()
    assert "selected_qk_path" in header
    assert "fattn_uses_warp_qk" in header
    assert "fattn_warp_qk_occupancy" in header
    assert "launch_attention_prepare_chunk_stream_k_qk" in header
    makefile = (ROOT / "Makefile").read_text()
    assert "--fmad=false" in makefile
    assert "opt041_fattn_warp_qk_ab_test" not in makefile
    source = (ROOT / "cuda/attention_decode.cu").read_text()
    assert source.count("launch_attention_prepare_chunk_stream_k(") >= 1


def test_opt041_validator_rejects_inadmissible_evidence() -> None:
    fixture = _reject_fixture()
    mutations: list[dict[str, Any]] = []
    for mutate in (
        lambda x: x.__setitem__("substitutes_for_opt016", True),
        lambda x: x.__setitem__("owns_opt016_parity_gate", True),
        lambda x: x.__setitem__("nsight_systems", "/tmp/capture.nsys-rep"),
        lambda x: x.__setitem__("report_path", "/tmp/opt041/REPORT.md"),
        lambda x: x.__setitem__("status", "source_inspected"),
        lambda x: x.__setitem__("task", "OPT-035"),
        lambda x: x["ab_p4096"].__setitem__("winner", "warp_microtile"),
        lambda x: x.__setitem__("selected_qk_path", "warp_microtile"),
        lambda x: x.__setitem__("keep_sitting_skipped", False),
        lambda x: x.__setitem__("reverted", False),
        lambda x: x.__setitem__("starts_opt042", True),
        lambda x: x.__setitem__("integer_decode_mmv", True),
        lambda x: x.__setitem__("mixer_gdn_graphs", True),
        lambda x: x.__setitem__("selected_gdn_fuse_path", "fuse_both"),
        lambda x: x.__setitem__("selected_vkq_accum", "global"),
        lambda x: x.__setitem__("selected_pv_path", "scalar"),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
    equality = json.loads(json.dumps(fixture))
    equality["status"] = "measured"
    equality["reverted"] = False
    equality["keep_sitting_skipped"] = False
    equality["selected_qk_path"] = "warp_microtile"
    equality["ab_p4096"] = _ab_stub(
        "warp_microtile", {"cparts": 2.0, "warp_microtile": 2.0}
    )
    equality.update(_keep_tok_s())
    mutations.append(equality)
    stale_p = json.loads(json.dumps(fixture))
    stale_p["status"] = "measured"
    stale_p["reverted"] = False
    stale_p["keep_sitting_skipped"] = False
    stale_p["selected_qk_path"] = "warp_microtile"
    stale_p["ab_p4096"] = _ab_stub(
        "warp_microtile", {"cparts": 2.0, "warp_microtile": 1.0}
    )
    stale_p.update(_keep_tok_s())
    stale_p["p"]["quartz"]["mean_tok_s"] = STALE_OPT034_P
    mutations.append(stale_p)
    loosened = json.loads(json.dumps(fixture))
    loosened["ab_p4096"]["candidates"]["warp_microtile"]["vs_tiled"]["max_abs"] = 5.0e-4
    loosened["ab_p4096"]["candidates"]["warp_microtile"]["eligible"] = True
    loosened["ab_p4096"]["winner"] = "warp_microtile"
    loosened["ab_p4096"]["candidates"]["warp_microtile"]["mean_ms"] = 0.5
    mutations.append(loosened)
    skip_ab = json.loads(json.dumps(fixture))
    skip_ab.pop("ab_p4096")
    mutations.append(skip_ab)
    skip_correctness = json.loads(json.dumps(fixture))
    skip_correctness["correctness"]["cases"].pop("0:4096")
    mutations.append(skip_correctness)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    report_prev = REPORT.read_text() if REPORT.is_file() else None
    raw_prev = AB_RAW.read_text() if AB_RAW.is_file() else None
    rejection_prev = REJECTION.read_text() if REJECTION.is_file() else None
    source_path = ROOT / "cuda/fattn_mma_f16.cuh"
    source_prev = source_path.read_text()
    contract_prev = CONTRACT.read_text()
    if report_prev is None:
        REPORT.write_text(PROOF + "\n")
    if raw_prev is None:
        AB_RAW.write_text("placeholder\n")
    if rejection_prev is None:
        REJECTION.write_text("rejected placeholder cparts\n")
    try:
        for mutation in mutations:
            with pytest.raises(AssertionError):
                validate_result(mutation)
        shrink = json.loads(json.dumps(fixture))
        source_path.write_text(
            source_prev.replace(
                "return kv + q + scores + stats + scratch + cparts + rescale;",
                "return kv + q + scores + stats + scratch + rescale;",
                1,
            )
        )
        with pytest.raises(AssertionError):
            validate_result(shrink)
        ncols = json.loads(json.dumps(fixture))
        source_path.write_text(
            source_prev.replace(
                "kSelectedAttentionMmaQueryRows = 16",
                "kSelectedAttentionMmaQueryRows = 8",
                1,
            )
        )
        with pytest.raises(AssertionError):
            validate_result(ncols)
        order = json.loads(json.dumps(fixture))
        source_path.write_text(
            source_prev.replace(
                "for (int k0 = vwarp * 16; k0 < kWidth; k0 += kNwarps * 16)",
                "for (int k0 = 0; k0 < kWidth; k0 += 16)",
                1,
            )
        )
        with pytest.raises(AssertionError):
            validate_result(order)
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
        source_path.write_text(source_prev)
        CONTRACT.write_text(contract_prev)


def test_opt041_fixture_connected() -> None:
    if not FIXTURE.is_file():
        pytest.skip("OPT-041 fixture is written by the exclusive CUDA sitting")
    validate_result(json.loads(FIXTURE.read_text()))


def test_opt041_rejects_historical_opt034_as_keep_denominator() -> None:
    contract = _contract()
    assert contract["opt040_quartz_p_mean_tok_s"] == OPT040_P
    assert contract["historical_opt034_quartz_p_mean_tok_s"] == STALE_OPT034_P
    assert contract["use_as_keep_denominator"] is False
    keep_denominators = {
        contract["opt040_quartz_p_mean_tok_s"],
        contract["opt040_quartz_d128_mean_tok_s"],
        contract["opt040_quartz_d2048_mean_tok_s"],
    }
    assert STALE_OPT034_P not in keep_denominators
    assert STALE_OPT034_D128 not in keep_denominators
    assert STALE_OPT034_D2048 not in keep_denominators
    opt040 = json.loads(OPT040_FIXTURE.read_text())
    assert contract["opt040_quartz_p_mean_tok_s"] == opt040["p"]["quartz"]["mean_tok_s"]
    assert (
        contract["opt040_quartz_d128_mean_tok_s"]
        == opt040["d128"]["quartz"]["mean_tok_s"]
    )
    assert (
        contract["opt040_quartz_d2048_mean_tok_s"]
        == opt040["d2048"]["quartz"]["mean_tok_s"]
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
    source = ROOT / "cuda/fattn_mma_f16.cuh"
    text = source.read_text()
    text = re.sub(
        r'kSelectedQKPath\[\] = "[^"]+"',
        f'kSelectedQKPath[] = "{path}"',
        text,
    )
    source.write_text(text)
    contract = _contract()
    contract["selected_qk_path"] = path
    CONTRACT.write_text(json.dumps(contract, indent=2) + "\n")


def _write_report(fixture: dict[str, Any]) -> None:
    decision = "keep" if fixture["status"] == "measured" else "reject"
    ab = fixture["ab_p4096"]
    sitting = "skipped" if fixture["keep_sitting_skipped"] else "ran"
    quartz_p = fixture["p"]["quartz"]["mean_tok_s"] if fixture["p"] else "n/a"
    quartz_d128 = fixture["d128"]["quartz"]["mean_tok_s"] if fixture["d128"] else "n/a"
    quartz_d2048 = (
        fixture["d2048"]["quartz"]["mean_tok_s"] if fixture["d2048"] else "n/a"
    )
    text = f"""# OPT-041 — Give prompt QK microtiles warp ownership

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of warp-owned 16×8 prompt QK microtiles against
shared `cparts` reduction. Keep requires **byte-equal quality attention outputs**,
**frozen attention gates**, **lower complete component time**, **improved P**
versus OPT-040, **95% throughput floors**, and **105% p95 ceilings**. This
increment does not substitute for the 2K llama.cpp parity gate. Copied
denominators are P {OPT040_P}, D128 {OPT040_D128}, D2048 {OPT040_D2048}.

## Decision

**{decision}** — `reverted`={json.dumps(fixture["reverted"])};
`keep_sitting_skipped`={json.dumps(fixture["keep_sitting_skipped"])};
selected_qk_path={fixture["selected_qk_path"]};
4096 A/B winner {ab["winner"]} (cparts {ab["candidates"]["cparts"]["mean_ms"]} ms,
warp_microtile {ab["candidates"]["warp_microtile"]["mean_ms"]} ms);
tok/s sitting {sitting}.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `{GGUF_SHA}` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `{LLAMA_REV}` |
| A/B | token_count 4096, candidates cparts then warp_microtile, 3 warm + 30 alternating |
| Correctness | tokens {CORRECTNESS_TOKENS}; starts {CORRECTNESS_STARTS} |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: {fixture["device"]} compute {fixture["compute_capability"]}
- measurement_utc: {fixture["measurement_utc"]}
- P Quartz mean tok/s: {quartz_p} versus OPT-040 {OPT040_P}
- D128 Quartz mean tok/s: {quartz_d128} versus OPT-040 {OPT040_D128}
- D2048 Quartz mean tok/s: {quartz_d2048} versus OPT-040 {OPT040_D2048}
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false
"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def _write_rejection(fixture: dict[str, Any]) -> None:
    ab = fixture["ab_p4096"]
    reason = (
        "4096 component A/B selected cparts"
        if fixture["keep_sitting_skipped"]
        else "keep sitting failed P improvement or the cross-workload guard"
    )
    tok = (
        ""
        if fixture["keep_sitting_skipped"]
        else f"\n- P Quartz mean tok/s: {fixture['p']['quartz']['mean_tok_s']}\n"
        f"- OPT-040 P baseline: {OPT040_P}\n"
        f"- D128 Quartz mean tok/s: {fixture['d128']['quartz']['mean_tok_s']}\n"
        f"- D2048 Quartz mean tok/s: {fixture['d2048']['quartz']['mean_tok_s']}"
    )
    text = f"""# OPT-041 rejection

{reason}. Production prompt QK remains the shared cparts reduction
(`selected_qk_path=cparts`).

- 4096 A/B winner: {ab["winner"]} cparts={ab["candidates"]["cparts"]["mean_ms"]} warp_microtile={ab["candidates"]["warp_microtile"]["mean_ms"]}
- keep_sitting_skipped: {json.dumps(fixture["keep_sitting_skipped"])}
- measurement_utc: {fixture["measurement_utc"]}{tok}
"""
    REJECTION.write_text(text)


def _run_ab() -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    commands = [
        [*_common(IMAGE), "make", "build/qw38-cuda-timing-test"],
        _nvcc(
            "cuda/opt041_fattn_warp_qk_ab_test.cu",
            "build/qw38-cuda-opt041-fattn-warp-qk-ab-test",
            ["build/attention_decode.cuda.o"],
        ),
        [
            *_common(IMAGE),
            "./build/qw38-cuda-opt041-fattn-warp-qk-ab-test",
            "evidence/optimization/opt041-fattn-warp-qk/warp-qk-ab-raw.txt",
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
            f"evidence/optimization/opt041-fattn-warp-qk/{sidecar}"
        )
    for key in ("n_gpu_layers", "n_ctx", "n_batch", "n_ubatch"):
        if key in record:
            block[key] = record[key]
    return block


def test_opt041_native_keep_reject() -> None:
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
    selected = _select_install(ab)
    if selected == "cparts":
        _set_pin("cparts")
        fixture = {
            "schema_version": 1,
            "task": "OPT-041",
            "status": "rejected",
            "measurement_utc": ab["measurement_utc"],
            "device": ab["device"],
            "compute_capability": ab["compute_capability"],
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "reverted": True,
            "selected_qk_path": "cparts",
            "selected_vkq_accum": "registers",
            "selected_pv_path": "mma",
            "ab_p4096": ab["ab_p4096"],
            "correctness": ab["correctness"],
            "keep_sitting_skipped": True,
            "p": None,
            "d128": None,
            "d2048": None,
            "owns_opt016_parity_gate": False,
            "substitutes_for_opt016": False,
            "nsight_systems": "not_used",
            "nsight_compute": "not_used",
            "proof_limit": PROOF,
            "report_path": "evidence/optimization/opt041-fattn-warp-qk/REPORT.md",
        }
        _write_report(fixture)
        _write_rejection(fixture)
        FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
        validate_result(fixture)
        return

    _set_pin("warp_microtile")
    _run([*_common(IMAGE), "make", "build/qw38-cuda-timing-test"])
    llama_p = _run_llama_bench_p()
    quartz_p = _run_quartz_p()
    llama_d128 = _run_llama_decode(128)
    quartz_d128 = _run_quartz_decode(128)
    llama_d2048 = _run_llama_decode(2048)
    quartz_d2048 = _run_quartz_decode(2048)
    fixture = {
        "schema_version": 1,
        "task": "OPT-041",
        "status": "measured",
        "measurement_utc": quartz_p.get("measurement_utc", ab["measurement_utc"]),
        "device": ab["device"],
        "compute_capability": ab["compute_capability"],
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": False,
        "selected_qk_path": "warp_microtile",
        "selected_vkq_accum": "registers",
        "selected_pv_path": "mma",
        "ab_p4096": ab["ab_p4096"],
        "correctness": ab["correctness"],
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
        "report_path": "evidence/optimization/opt041-fattn-warp-qk/REPORT.md",
    }
    keep = _keep_predicates(fixture)
    if not keep:
        _set_pin("cparts")
        fixture["status"] = "rejected"
        fixture["reverted"] = True
        fixture["selected_qk_path"] = "cparts"
        _write_report(fixture)
        _write_rejection(fixture)
        _run([*_common(IMAGE), "make", "build/qw38-cuda-timing-test"])
    else:
        if REJECTION.is_file():
            REJECTION.unlink()
        _write_report(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    validate_result(json.loads(FIXTURE.read_text()))
