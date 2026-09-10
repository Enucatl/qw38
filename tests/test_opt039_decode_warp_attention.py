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
CONTRACT = ROOT / "pins/opt039_decode_warp_contract.json"
FIXTURE = ROOT / "fixtures/opt039_decode_warp.json"
EVIDENCE = ROOT / "evidence/optimization/opt039-decode-warp"
REPORT = EVIDENCE / "REPORT.md"
REJECTION = EVIDENCE / "REJECTION.md"
AB_RAW = EVIDENCE / "warp-ab-raw.txt"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
AB_PREFIX = "QW38_DECODE_WARP_AB_RESULT="
P_PREFIX = "QW38_PREFILL_4K_ORACLE_RESULT="
DECODE_PREFIX = "QW38_DECODE_ORACLE_RESULT="
LLAMA_DECODE_PREFIX = "QW38_LLAMA_DECODE_ORACLE_RESULT="
OPT034_P = 1869.84412
OPT034_D128 = 25.3816128
OPT034_D2048 = 20.169548
OPT034_D128_P95 = 39.9736366
OPT034_D128_RUN_P95 = 39.4155655
OPT034_D2048_P95 = 50.2872772
OPT034_D2048_RUN_P95 = 49.590683
LEGAL_VECS = {"cta_group", "warp_query"}
CORRECTNESS_POSITIONS = [0, 1, 15, 16, 31, 32, 127, 128, 2047, 2048, 2303, 131071]
PROOF = (
    "frozen attention envelopes; exact candidate KV and state isolation; "
    "lower D2048 component time; improved D2048; "
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
    text = (ROOT / "cuda/attention_decode.cu").read_text()
    match = re.search(r'kSelectedDecodeAttentionVec\[\] = "([^"]+)"', text)
    assert match is not None
    return match.group(1)


def decode_16_16_unchanged() -> bool:
    text = (ROOT / "cuda/attention_decode.cu").read_text()
    return (
        "kSelectedDecodeKvPartsLow = 16" in text
        and "kSelectedDecodeKvPartsHigh = 16" in text
    )


def scheduler_dispatches_partitioned() -> bool:
    text = (ROOT / "cuda/full_scheduler.cu").read_text()
    return (
        "launch_attention_prepare_partitioned" in text
        and "decode_kv_parts_for_position" in text
        and "launch_attention_prepare_partitioned_vec" not in text
    )


def fattn_path_unchanged() -> bool:
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


def packed_mmv_unchanged() -> bool:
    text = (ROOT / "cuda/quant_mmv.cu").read_text()
    return 'kSelectedMmvLoadPath[] = "packed"' in text


def _ab_candidate(block: dict[str, Any], ident: str, *, timed: bool) -> None:
    cand = block["candidates"][ident] if "candidates" in block else block[ident]
    assert cand["id"] == ident
    assert cand["occupancy"] >= 1
    assert cand["merge_occupancy"] >= 1
    assert cand["launch_ok"] is True
    assert cand["scratch_unchanged"] is True
    assert cand["committed_unchanged"] is True
    assert cand["candidate_exact"] is True
    assert cand["vs_cta"]["nonfinite"] == 0
    if timed:
        assert len(cand["samples"]) == 30
        assert len(cand["warmup_ms"]) == 3
        mean = sum(float(v) for v in cand["samples"]) / 30.0
        assert cand["mean_ms"] == pytest.approx(mean, rel=1e-6, abs=1e-6)
    if cand["eligible"]:
        assert cand["vs_cta"]["max_abs"] <= 5.0e-5
        assert cand["vs_cta"]["rms"] <= 5.0e-6
        if not cand.get("skip_tiled_reference"):
            assert cand["vs_tiled"]["nonfinite"] == 0
            assert cand["vs_reference"]["nonfinite"] == 0
            assert cand["vs_tiled"]["max_abs"] <= 5.0e-5
            assert cand["vs_tiled"]["rms"] <= 5.0e-6
            assert cand["vs_reference"]["max_abs"] <= 5.0e-5
            assert cand["vs_reference"]["rms"] <= 5.0e-6


def _select_component_winner(block: dict[str, Any]) -> str:
    cta = block["candidates"]["cta_group"]
    warp = block["candidates"]["warp_query"]
    if (
        warp["eligible"]
        and cta["eligible"]
        and float(warp["mean_ms"]) < float(cta["mean_ms"])
    ):
        return "warp_query"
    return "cta_group"


def _select_install(result: dict[str, Any]) -> str:
    if not result["correctness"]["all_eligible"]:
        return "cta_group"
    if not result["ab_d128"]["candidates"]["warp_query"]["eligible"]:
        return "cta_group"
    if _select_component_winner(result["ab_d2048"]) != "warp_query":
        return "cta_group"
    return "warp_query"


def _keep_predicates(result: dict[str, Any]) -> bool:
    contract = _contract()
    if result["selected_decode_attention_vec"] != "warp_query":
        return False
    if result["reverted"] is True or result["keep_sitting_skipped"] is True:
        return False
    if result["status"] != "measured":
        return False
    if _select_install(result) != "warp_query":
        return False
    if result["p"] is None or result["d128"] is None or result["d2048"] is None:
        return False
    quartz_d2048 = float(result["d2048"]["quartz"]["mean_tok_s"])
    if quartz_d2048 <= contract["opt034_quartz_d2048_mean_tok_s"]:
        return False
    if (
        float(result["d128"]["quartz"]["mean_tok_s"])
        < 0.95 * contract["opt034_quartz_d128_mean_tok_s"]
    ):
        return False
    if (
        float(result["p"]["quartz"]["mean_tok_s"])
        < 0.95 * contract["opt034_quartz_p_mean_tok_s"]
    ):
        return False
    if (
        result["d128"]["quartz"]["token_latency_p95_ms"]
        > 1.05 * contract["opt034_quartz_d128_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d128"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * contract["opt034_quartz_d128_run_mean_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d2048"]["quartz"]["token_latency_p95_ms"]
        > 1.05 * contract["opt034_quartz_d2048_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d2048"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * contract["opt034_quartz_d2048_run_mean_token_latency_p95_ms"]
    ):
        return False
    return True


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-039"
    assert result["status"] in ("measured", "rejected")
    assert result["status"] not in ("scout", "source_inspected")
    assert contract["yardstick"] == "d2048_decode_attention_warp_query"
    assert contract["opt034_quartz_p_mean_tok_s"] == OPT034_P
    assert contract["opt034_quartz_d128_mean_tok_s"] == OPT034_D128
    assert contract["opt034_quartz_d2048_mean_tok_s"] == OPT034_D2048
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    assert "fattn_path" not in result
    assert "starts_integer_decode_mmv" not in result
    assert contract["atn001"]["bf16_max_abs"] == 0.00005
    assert contract["atn001"]["bf16_rms"] == 0.000005
    vec = result["selected_decode_attention_vec"]
    assert vec in LEGAL_VECS
    assert contract["ab"]["candidates"] == ["cta_group", "warp_query"]
    assert _select_component_winner(result["ab_d128"]) == result["ab_d128"]["winner"]
    assert _select_component_winner(result["ab_d2048"]) == result["ab_d2048"]["winner"]
    _ab_candidate(result["ab_d128"], "cta_group", timed=True)
    _ab_candidate(result["ab_d128"], "warp_query", timed=True)
    _ab_candidate(result["ab_d2048"], "cta_group", timed=True)
    _ab_candidate(result["ab_d2048"], "warp_query", timed=True)
    assert result["ab_d128"]["position"] == 128
    assert result["ab_d2048"]["position"] == 2048
    assert result["correctness"]["positions"] == CORRECTNESS_POSITIONS
    assert set(result["correctness"]["cases"]) == {
        str(position) for position in CORRECTNESS_POSITIONS
    }
    for position in CORRECTNESS_POSITIONS:
        case = result["correctness"]["cases"][str(position)]
        assert case["position"] == position
        skip = position == 131071
        for ident in ("cta_group", "warp_query"):
            cand = case[ident]
            assert cand["id"] == ident
            assert cand["skip_tiled_reference"] is skip
            if cand["eligible"]:
                assert cand["vs_cta"]["max_abs"] <= 5.0e-5
                assert cand["vs_cta"]["rms"] <= 5.0e-6
                assert cand["scratch_unchanged"] is True
                assert cand["committed_unchanged"] is True
                assert cand["candidate_exact"] is True
                if not skip:
                    assert cand["vs_tiled"]["max_abs"] <= 5.0e-5
                    assert cand["vs_tiled"]["rms"] <= 5.0e-6
                    assert cand["vs_reference"]["max_abs"] <= 5.0e-5
                    assert cand["vs_reference"]["rms"] <= 5.0e-6
            else:
                assert skip or cand["vs_tiled"] is not None
    warp_cases_ok = all(
        result["correctness"]["cases"][str(position)]["warp_query"]["eligible"]
        for position in CORRECTNESS_POSITIONS
    )
    assert result["correctness"]["all_eligible"] is warp_cases_ok
    installed = _select_install(result)
    assert pin_from_source() == vec == contract["selected_decode_attention_vec"]
    assert decode_16_16_unchanged()
    assert scheduler_dispatches_partitioned()
    assert fattn_path_unchanged()
    assert skinny_path_unchanged()
    assert ffn_shared_y_unchanged()
    assert packed_mmv_unchanged()
    keep = _keep_predicates(result)
    if result["status"] == "measured":
        assert keep
        assert result["reverted"] is False
        assert result["keep_sitting_skipped"] is False
        assert vec == "warp_query" == installed
        assert result["p"]["quartz"]["mean_tok_s"] > 0
        assert not REJECTION.is_file()
    else:
        assert not keep
        assert result["reverted"] is True
        assert vec == "cta_group"
        assert REJECTION.is_file()
        rejection = REJECTION.read_text()
        assert "cta_group" in rejection
        if result["keep_sitting_skipped"]:
            assert installed == "cta_group"
            assert result["p"] is None and result["d128"] is None
            assert result["d2048"] is None
        else:
            assert installed == "warp_query"
            assert result["d2048"] is not None and result["p"] is not None
            assert str(result["d2048"]["quartz"]["mean_tok_s"]) in rejection
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
    ident: str, mean: float, *, eligible: bool = True, skip: bool = False
) -> dict[str, Any]:
    envelope = {"max_abs": 0.0, "rms": 0.0, "nonfinite": 0}
    return {
        "id": ident,
        "mean_ms": mean,
        "samples": [mean] * 30,
        "warmup_ms": [mean] * 3,
        "occupancy": 1,
        "merge_occupancy": 1,
        "launch_ok": True,
        "eligible": eligible,
        "scratch_unchanged": True,
        "committed_unchanged": True,
        "candidate_exact": True,
        "skip_tiled_reference": skip,
        "vs_cta": dict(envelope),
        "vs_tiled": None if skip else dict(envelope),
        "vs_reference": None if skip else dict(envelope),
    }


def _ab_stub(position: int, winner: str, means: dict[str, float]) -> dict[str, Any]:
    return {
        "position": position,
        "winner": winner,
        "win": winner == "warp_query",
        "candidates": {
            ident: _candidate_stub(ident, means[ident])
            for ident in ("cta_group", "warp_query")
        },
    }


def _correctness_stub(*, warp_ok: bool = True) -> dict[str, Any]:
    cases = {}
    for position in CORRECTNESS_POSITIONS:
        skip = position == 131071
        cases[str(position)] = {
            "position": position,
            "cta_group": _candidate_stub("cta_group", 0.0, skip=skip),
            "warp_query": _candidate_stub(
                "warp_query", 0.0, eligible=warp_ok, skip=skip
            ),
        }
        del cases[str(position)]["cta_group"]["samples"]
        del cases[str(position)]["cta_group"]["warmup_ms"]
        del cases[str(position)]["warp_query"]["samples"]
        del cases[str(position)]["warp_query"]["warmup_ms"]
    return {
        "positions": CORRECTNESS_POSITIONS,
        "cases": cases,
        "all_eligible": warp_ok,
    }


def test_opt039_contract_and_source_pins() -> None:
    contract = _contract()
    opt034 = json.loads((ROOT / "fixtures/opt034_packed_mmv.json").read_text())
    assert opt034["p"]["quartz"]["mean_tok_s"] == OPT034_P
    assert opt034["d128"]["quartz"]["mean_tok_s"] == OPT034_D128
    assert opt034["d2048"]["quartz"]["mean_tok_s"] == OPT034_D2048
    assert contract["opt034_quartz_p_mean_tok_s"] == OPT034_P
    assert contract["opt034_quartz_d128_mean_tok_s"] == OPT034_D128
    assert contract["opt034_quartz_d2048_mean_tok_s"] == OPT034_D2048
    assert contract["ab"]["candidates"] == ["cta_group", "warp_query"]
    assert contract["ab"]["timed_positions"] == [128, 2048]
    assert contract["ab"]["correctness_positions"] == CORRECTNESS_POSITIONS
    assert contract["ab"]["ties_retain"] == "cta_group"
    assert contract["owns_opt016_parity_gate"] is False
    assert contract["substitutes_for_opt016"] is False
    assert contract["llama_bench_decode_is_informational"] is True
    tiled = json.loads((ROOT / "pins/cuda_tiled_attention_contract.json").read_text())
    assert tiled["proof_limits"]["bf16_max_abs"] == 0.00005
    assert tiled["proof_limits"]["bf16_rms"] == 0.000005
    assert pin_from_source() in LEGAL_VECS
    assert decode_16_16_unchanged()
    assert scheduler_dispatches_partitioned()
    assert fattn_path_unchanged()
    assert packed_mmv_unchanged()
    source = (ROOT / "cuda/attention_decode.cu").read_text()
    assert "warp_query_decode_attention" in source
    assert "__syncthreads" in source
    assert "fattn-vec.cuh" in source
    header = (ROOT / "cuda/attention_decode.h").read_text()
    assert "launch_attention_prepare_partitioned_vec" in header
    assert "selected_decode_attention_vec" in header
    makefile = (ROOT / "Makefile").read_text()
    assert "--fmad=false" in makefile
    assert "opt039_decode_warp_ab_test" not in makefile


def test_opt039_validator_rejects_inadmissible_evidence() -> None:
    means = {"cta_group": 1.0, "warp_query": 2.0}
    fixture = {
        "schema_version": 1,
        "task": "OPT-039",
        "status": "rejected",
        "measurement_utc": "2026-09-10T00:00:00Z",
        "device": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": True,
        "selected_decode_attention_vec": "cta_group",
        "ab_d128": _ab_stub(128, "cta_group", means),
        "ab_d2048": _ab_stub(2048, "cta_group", means),
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
        "report_path": "evidence/optimization/opt039-decode-warp/REPORT.md",
    }
    mutations: list[dict[str, Any]] = []
    for mutate in (
        lambda x: x.__setitem__("substitutes_for_opt016", True),
        lambda x: x.__setitem__("owns_opt016_parity_gate", True),
        lambda x: x.__setitem__("nsight_systems", "/tmp/capture.nsys-rep"),
        lambda x: x.__setitem__("report_path", "/tmp/opt039/REPORT.md"),
        lambda x: x.__setitem__("status", "source_inspected"),
        lambda x: x.__setitem__("task", "OPT-036"),
        lambda x: x["ab_d2048"].__setitem__("winner", "warp_query"),
        lambda x: x.__setitem__("selected_decode_attention_vec", "warp_query"),
        lambda x: x.__setitem__("keep_sitting_skipped", False),
        lambda x: x.__setitem__("reverted", False),
        lambda x: x.__setitem__("fattn_path", "occupancy2"),
        lambda x: x.__setitem__("starts_integer_decode_mmv", True),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
    equality = json.loads(json.dumps(fixture))
    equality["status"] = "measured"
    equality["reverted"] = False
    equality["keep_sitting_skipped"] = False
    equality["selected_decode_attention_vec"] = "warp_query"
    equality["ab_d2048"] = _ab_stub(
        2048, "warp_query", {"cta_group": 2.0, "warp_query": 1.0}
    )
    equality["ab_d128"] = _ab_stub(
        128, "warp_query", {"cta_group": 2.0, "warp_query": 1.0}
    )
    equality["p"] = {"quartz": {"mean_tok_s": OPT034_P}}
    equality["d128"] = {
        "quartz": {
            "mean_tok_s": OPT034_D128,
            "token_latency_p95_ms": OPT034_D128_P95,
            "run_mean_token_latency_p95_ms": OPT034_D128_RUN_P95,
        }
    }
    equality["d2048"] = {
        "quartz": {
            "mean_tok_s": OPT034_D2048,
            "token_latency_p95_ms": OPT034_D2048_P95,
            "run_mean_token_latency_p95_ms": OPT034_D2048_RUN_P95,
        }
    }
    mutations.append(equality)
    historical = json.loads(json.dumps(equality))
    historical["p"]["quartz"]["mean_tok_s"] = 1746.71973
    mutations.append(historical)
    loosened = json.loads(json.dumps(fixture))
    loosened["ab_d2048"]["candidates"]["warp_query"]["vs_cta"]["max_abs"] = 0.1
    loosened["ab_d2048"]["candidates"]["warp_query"]["eligible"] = True
    loosened["ab_d2048"]["winner"] = "warp_query"
    loosened["ab_d2048"]["candidates"]["warp_query"]["mean_ms"] = 0.5
    mutations.append(loosened)
    skip_d2048 = json.loads(json.dumps(fixture))
    skip_d2048.pop("ab_d2048")
    mutations.append(skip_d2048)
    skip_correctness = json.loads(json.dumps(fixture))
    skip_correctness["correctness"]["cases"].pop("131071")
    mutations.append(skip_correctness)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    report_prev = REPORT.read_text() if REPORT.is_file() else None
    raw_prev = AB_RAW.read_text() if AB_RAW.is_file() else None
    rejection_prev = REJECTION.read_text() if REJECTION.is_file() else None
    source_prev = (ROOT / "cuda/attention_decode.cu").read_text()
    contract_prev = CONTRACT.read_text()
    if report_prev is None:
        REPORT.write_text(PROOF + "\n")
    if raw_prev is None:
        AB_RAW.write_text("placeholder\n")
    if rejection_prev is None:
        REJECTION.write_text("rejected placeholder cta_group\n")
    try:
        for mutation in mutations:
            with pytest.raises(AssertionError):
                validate_result(mutation)
        sixteen = json.loads(json.dumps(fixture))
        (ROOT / "cuda/attention_decode.cu").write_text(
            source_prev.replace(
                "kSelectedDecodeKvPartsLow = 16",
                "kSelectedDecodeKvPartsLow = 8",
            )
        )
        with pytest.raises(AssertionError):
            validate_result(sixteen)
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
        (ROOT / "cuda/attention_decode.cu").write_text(source_prev)
        CONTRACT.write_text(contract_prev)


def test_opt039_fixture_connected() -> None:
    if not FIXTURE.is_file():
        pytest.skip("OPT-039 fixture is written by the exclusive CUDA sitting")
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


def _set_pin(vec: str) -> None:
    source = ROOT / "cuda/attention_decode.cu"
    text = source.read_text()
    text = re.sub(
        r'kSelectedDecodeAttentionVec\[\] = "[^"]+"',
        f'kSelectedDecodeAttentionVec[] = "{vec}"',
        text,
    )
    source.write_text(text)
    contract = _contract()
    contract["selected_decode_attention_vec"] = vec
    CONTRACT.write_text(json.dumps(contract, indent=2) + "\n")


def _write_report(fixture: dict[str, Any]) -> None:
    decision = "keep" if fixture["status"] == "measured" else "reject"
    d2048_ab = fixture["ab_d2048"]
    d128_ab = fixture["ab_d128"]
    sitting = "skipped" if fixture["keep_sitting_skipped"] else "ran"
    quartz_p = fixture["p"]["quartz"]["mean_tok_s"] if fixture["p"] else "n/a"
    quartz_d128 = fixture["d128"]["quartz"]["mean_tok_s"] if fixture["d128"] else "n/a"
    quartz_d2048 = (
        fixture["d2048"]["quartz"]["mean_tok_s"] if fixture["d2048"] else "n/a"
    )
    text = f"""# OPT-039 — Warp-owned vector decode attention

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of warp-owned one-token query-head attention
against the accepted 16-partition CTA kernel. Keep requires
**frozen attention envelopes**, **exact candidate KV and state isolation**,
**lower D2048 component time**, **improved D2048** Quartz tok/s versus OPT-034,
**95% throughput floors**, and **105% p95 ceilings**. This increment
does not substitute for the 2K llama.cpp parity gate. Copied OPT-034
denominators are P {OPT034_P}, D128 {OPT034_D128}, D2048 {OPT034_D2048}.

## Decision

**{decision}** — `reverted`={json.dumps(fixture["reverted"])};
`keep_sitting_skipped`={json.dumps(fixture["keep_sitting_skipped"])};
selected_decode_attention_vec={fixture["selected_decode_attention_vec"]};
D128 A/B winner {d128_ab["winner"]} (cta {d128_ab["candidates"]["cta_group"]["mean_ms"]} ms,
warp {d128_ab["candidates"]["warp_query"]["mean_ms"]} ms);
D2048 A/B winner {d2048_ab["winner"]} (cta {d2048_ab["candidates"]["cta_group"]["mean_ms"]} ms,
warp {d2048_ab["candidates"]["warp_query"]["mean_ms"]} ms);
tok/s sitting {sitting}.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `{GGUF_SHA}` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `{LLAMA_REV}` |
| A/B | positions 128 and 2048, candidates cta_group then warp_query, 3 warm + 30 alternating |
| Correctness | {CORRECTNESS_POSITIONS} |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: {fixture["device"]} compute {fixture["compute_capability"]}
- measurement_utc: {fixture["measurement_utc"]}
- P Quartz mean tok/s: {quartz_p} versus OPT-034 {OPT034_P}
- D128 Quartz mean tok/s: {quartz_d128} versus OPT-034 {OPT034_D128}
- D2048 Quartz mean tok/s: {quartz_d2048} versus OPT-034 {OPT034_D2048}
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false
"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def _write_rejection(fixture: dict[str, Any]) -> None:
    d2048 = fixture["ab_d2048"]
    reason = (
        "D2048 component A/B selected cta_group"
        if fixture["keep_sitting_skipped"]
        else "keep sitting failed D2048 improvement or the cross-workload guard"
    )
    tok = (
        ""
        if fixture["keep_sitting_skipped"]
        else f"\n- D2048 Quartz mean tok/s: {fixture['d2048']['quartz']['mean_tok_s']}\n"
        f"- OPT-034 D2048 baseline: {OPT034_D2048}\n"
        f"- P Quartz mean tok/s: {fixture['p']['quartz']['mean_tok_s']}\n"
        f"- D128 Quartz mean tok/s: {fixture['d128']['quartz']['mean_tok_s']}"
    )
    text = f"""# OPT-039 rejection

{reason}. Production decode attention remains the 16-partition CTA path
(`selected_decode_attention_vec=cta_group`).

- D128 A/B winner: {fixture["ab_d128"]["winner"]} cta={fixture["ab_d128"]["candidates"]["cta_group"]["mean_ms"]} warp={fixture["ab_d128"]["candidates"]["warp_query"]["mean_ms"]}
- D2048 A/B winner: {d2048["winner"]} cta={d2048["candidates"]["cta_group"]["mean_ms"]} warp={d2048["candidates"]["warp_query"]["mean_ms"]}
- keep_sitting_skipped: {json.dumps(fixture["keep_sitting_skipped"])}
- measurement_utc: {fixture["measurement_utc"]}{tok}
"""
    REJECTION.write_text(text)


def _run_ab() -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    commands = [
        [*_common(IMAGE), "make", "build/qw38-cuda-timing-test"],
        _nvcc(
            "cuda/opt039_decode_warp_ab_test.cu",
            "build/qw38-cuda-opt039-decode-warp-ab-test",
            ["build/attention_decode.cuda.o"],
        ),
        [*_common(IMAGE), "make", "build/qw38-cuda-attention-test"],
        [
            *_common(IMAGE),
            "./build/qw38-cuda-opt039-decode-warp-ab-test",
            "evidence/optimization/opt039-decode-warp/warp-ab-raw.txt",
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
            f"evidence/optimization/opt039-decode-warp/{sidecar}"
        )
    for key in ("n_gpu_layers", "n_ctx", "n_batch", "n_ubatch"):
        if key in record:
            block[key] = record[key]
    return block


def test_opt039_native_keep_reject() -> None:
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
    if selected == "cta_group":
        _set_pin("cta_group")
        fixture = {
            "schema_version": 1,
            "task": "OPT-039",
            "status": "rejected",
            "measurement_utc": ab["measurement_utc"],
            "device": ab["device"],
            "compute_capability": ab["compute_capability"],
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "reverted": True,
            "selected_decode_attention_vec": "cta_group",
            "ab_d128": ab["ab_d128"],
            "ab_d2048": ab["ab_d2048"],
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
            "report_path": "evidence/optimization/opt039-decode-warp/REPORT.md",
        }
        _write_report(fixture)
        _write_rejection(fixture)
        FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
        validate_result(fixture)
        return

    _set_pin("warp_query")
    _run([*_common(IMAGE), "make", "build/qw38-cuda-timing-test"])
    llama_p = _run_llama_bench_p()
    quartz_p = _run_quartz_p()
    llama_d128 = _run_llama_decode(128)
    quartz_d128 = _run_quartz_decode(128)
    llama_d2048 = _run_llama_decode(2048)
    quartz_d2048 = _run_quartz_decode(2048)
    fixture = {
        "schema_version": 1,
        "task": "OPT-039",
        "status": "measured",
        "measurement_utc": quartz_d2048.get("measurement_utc", ab["measurement_utc"]),
        "device": ab["device"],
        "compute_capability": ab["compute_capability"],
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": False,
        "selected_decode_attention_vec": "warp_query",
        "ab_d128": ab["ab_d128"],
        "ab_d2048": ab["ab_d2048"],
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
        "report_path": "evidence/optimization/opt039-decode-warp/REPORT.md",
    }
    keep = _keep_predicates(fixture)
    if not keep:
        _set_pin("cta_group")
        fixture["status"] = "rejected"
        fixture["reverted"] = True
        fixture["selected_decode_attention_vec"] = "cta_group"
        _write_report(fixture)
        _write_rejection(fixture)
        _run([*_common(IMAGE), "make", "build/qw38-cuda-timing-test"])
    else:
        if REJECTION.is_file():
            REJECTION.unlink()
        _write_report(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    validate_result(json.loads(FIXTURE.read_text()))
