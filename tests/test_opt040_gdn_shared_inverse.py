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
CONTRACT = ROOT / "pins/opt040_gdn_shared_inverse_contract.json"
FIXTURE = ROOT / "fixtures/opt040_gdn_shared_inverse.json"
EVIDENCE = ROOT / "evidence/optimization/opt040-gdn-shared-inverse"
REPORT = EVIDENCE / "REPORT.md"
REJECTION = EVIDENCE / "REJECTION.md"
AB_RAW = EVIDENCE / "gdn-ab-raw.txt"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
AB_PREFIX = "QW38_GDN_SHARED_INVERSE_AB_RESULT="
P_PREFIX = "QW38_PREFILL_4K_ORACLE_RESULT="
DECODE_PREFIX = "QW38_DECODE_ORACLE_RESULT="
LLAMA_DECODE_PREFIX = "QW38_LLAMA_DECODE_ORACLE_RESULT="
OPT034_P = 1869.84412
OPT039_D128 = 26.1887932
OPT039_D2048 = 25.3357754
OPT039_D128_P95 = 38.3246689
OPT039_D128_RUN_P95 = 38.2041283
OPT039_D2048_P95 = 39.5857964
OPT039_D2048_RUN_P95 = 39.4975739
STALE_OPT034_D2048 = 20.169548
LEGAL_PATHS = {"repeated", "shared"}
CORRECTNESS_TOKENS = [1, 2, 3, 4, 63, 64, 65, 512, 2048, 4096]
CORRECTNESS_LAYERS = [0, 1, 62]
PROOF = (
    "byte-equal quality outputs/state; frozen sequential GDN gates; "
    "lower complete GDN component time; improved P; "
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
    text = (ROOT / "cuda/gdn_fused_quality.cuh").read_text()
    match = re.search(r'kSelectedGdnInversePath\[\] = "([^"]+)"', text)
    assert match is not None
    return match.group(1)


def fuse_path_unchanged() -> bool:
    text = (ROOT / "cuda/gdn_fused_quality.cuh").read_text()
    return 'kSelectedGdnFusePath[] = "off"' in text


def decode_gdn_unchanged() -> bool:
    text = (ROOT / "cuda/full_scheduler.cu").read_text()
    return (
        "launch_gdn_prepare_tiled(" in text
        and "launch_gdn_gated_output(" in text
        and "GdnScanPath::kFusedTokenLoop" in text
    )


def decode_vec_unchanged() -> bool:
    text = (ROOT / "cuda/attention_decode.cu").read_text()
    return 'kSelectedDecodeAttentionVec[] = "warp_query"' in text


def rsqrtf_absent() -> bool:
    text = (ROOT / "cuda/gdn_fused_quality.cuh").read_text()
    return "rsqrtf(" not in text and "rsqrtf_rn" not in text


def skinny_path_unchanged() -> bool:
    text = (ROOT / "cuda/quant_mmq_mma.cuh").read_text()
    return 'kSelectedSkinnyMixerPath[] = "mma_i32_j128"' in text


def ffn_shared_y_unchanged() -> bool:
    text = (ROOT / "cuda/quant_mmq_mma.cuh").read_text()
    return 'kSelectedFfnPath[] = "shared_y_swiglu_q8"' in text


def fattn_path_unchanged() -> bool:
    text = (ROOT / "cuda/fattn_mma_f16.cuh").read_text()
    return (
        'kSelectedFattnPath[] = "stream_k"' in text
        and 'kSelectedVkqAccum[] = "registers"' in text
        and 'kSelectedPvPath[] = "mma"' in text
    )


def _ab_candidate(block: dict[str, Any], ident: str) -> None:
    cand = block["candidates"][ident]
    assert cand["id"] == ident
    assert cand["occupancy"] >= 1
    assert cand["recurrence_occupancy"] >= 1
    assert cand["launch_ok"] is True
    assert len(cand["samples"]) == 30
    assert len(cand["warmup_ms"]) == 3
    mean = sum(float(v) for v in cand["samples"]) / 30.0
    assert cand["mean_ms"] == pytest.approx(mean, rel=1e-6, abs=1e-6)
    assert cand["vs_sequential"]["nonfinite"] == 0
    if cand["eligible"]:
        assert cand["inverses_byte_equal"] is True
        assert cand["recurrent_byte_equal"] is True
        assert cand["conv_byte_equal"] is True
        assert cand["state_byte_equal"] is True
        assert cand["gated_byte_equal"] is True
        assert cand["prepare_preserves_committed"] is True
        assert cand["cancel_isolates"] is True
        assert cand["vs_sequential"]["max_abs"] <= 5.0e-8
        assert cand["vs_sequential"]["rms"] <= 5.0e-9


def _select_component_winner(block: dict[str, Any]) -> str:
    repeated = block["candidates"]["repeated"]
    shared = block["candidates"]["shared"]
    if (
        shared["eligible"]
        and repeated["eligible"]
        and float(shared["mean_ms"]) < float(repeated["mean_ms"])
    ):
        return "shared"
    return "repeated"


def _select_install(result: dict[str, Any]) -> str:
    if not result["correctness"]["all_eligible"]:
        return "repeated"
    if not result["ab_p4096"]["candidates"]["shared"]["eligible"]:
        return "repeated"
    if _select_component_winner(result["ab_p4096"]) != "shared":
        return "repeated"
    return "shared"


def _keep_predicates(result: dict[str, Any]) -> bool:
    contract = _contract()
    if result["selected_gdn_inverse_path"] != "shared":
        return False
    if result["reverted"] is True or result["keep_sitting_skipped"] is True:
        return False
    if result["status"] != "measured":
        return False
    if _select_install(result) != "shared":
        return False
    if result["p"] is None or result["d128"] is None or result["d2048"] is None:
        return False
    if (
        float(result["p"]["quartz"]["mean_tok_s"])
        <= contract["opt034_quartz_p_mean_tok_s"]
    ):
        return False
    if (
        float(result["d128"]["quartz"]["mean_tok_s"])
        < 0.95 * contract["opt039_quartz_d128_mean_tok_s"]
    ):
        return False
    if (
        float(result["d2048"]["quartz"]["mean_tok_s"])
        < 0.95 * contract["opt039_quartz_d2048_mean_tok_s"]
    ):
        return False
    if (
        result["d128"]["quartz"]["token_latency_p95_ms"]
        > 1.05 * contract["opt039_quartz_d128_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d128"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * contract["opt039_quartz_d128_run_mean_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d2048"]["quartz"]["token_latency_p95_ms"]
        > 1.05 * contract["opt039_quartz_d2048_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d2048"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * contract["opt039_quartz_d2048_run_mean_token_latency_p95_ms"]
    ):
        return False
    return True


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-040"
    assert result["status"] in ("measured", "rejected")
    assert result["status"] not in ("scout", "source_inspected")
    assert contract["yardstick"] == "prompt_gdn_shared_inverse"
    assert contract["opt034_quartz_p_mean_tok_s"] == OPT034_P
    assert contract["opt039_quartz_d128_mean_tok_s"] == OPT039_D128
    assert contract["opt039_quartz_d2048_mean_tok_s"] == OPT039_D2048
    assert contract["opt039_quartz_d2048_mean_tok_s"] != STALE_OPT034_D2048
    assert contract["gdn002"]["max_abs"] == 5.0e-8
    assert contract["gdn002"]["rms"] == 5.0e-9
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    assert "fattn_path" not in result
    assert "starts_opt041" not in result
    assert "starts_opt042" not in result
    assert "selected_gdn_fuse_path" not in result
    path = result["selected_gdn_inverse_path"]
    assert path in LEGAL_PATHS
    assert contract["ab"]["candidates"] == ["repeated", "shared"]
    assert contract["ab"]["ties_retain"] == "repeated"
    assert _select_component_winner(result["ab_p4096"]) == result["ab_p4096"]["winner"]
    _ab_candidate(result["ab_p4096"], "repeated")
    _ab_candidate(result["ab_p4096"], "shared")
    assert result["ab_p4096"]["token_count"] == 4096
    assert result["correctness"]["token_counts"] == CORRECTNESS_TOKENS
    assert result["correctness"]["layers"] == CORRECTNESS_LAYERS
    expected_keys = {
        f"{layer}:{token}"
        for layer in CORRECTNESS_LAYERS
        for token in CORRECTNESS_TOKENS
    }
    assert set(result["correctness"]["cases"]) == expected_keys
    for layer in CORRECTNESS_LAYERS:
        for token in CORRECTNESS_TOKENS:
            case = result["correctness"]["cases"][f"{layer}:{token}"]
            assert case["layer"] == layer
            assert case["token_count"] == token
            if case["eligible"]:
                assert case["inverses_byte_equal"] is True
                assert case["recurrent_byte_equal"] is True
                assert case["conv_byte_equal"] is True
                assert case["state_byte_equal"] is True
                assert case["gated_byte_equal"] is True
                assert case["prepare_preserves_committed"] is True
                assert case["cancel_isolates"] is True
                assert case["vs_sequential"]["max_abs"] <= 5.0e-8
                assert case["vs_sequential"]["rms"] <= 5.0e-9
                assert case["vs_sequential"]["nonfinite"] == 0
                if token == 1:
                    assert case["dispatched_shared"] is False
                else:
                    assert case["dispatched_shared"] is True
    installed = _select_install(result)
    assert pin_from_source() == path == contract["selected_gdn_inverse_path"]
    assert fuse_path_unchanged()
    assert decode_gdn_unchanged()
    assert decode_vec_unchanged()
    assert rsqrtf_absent()
    assert skinny_path_unchanged()
    assert ffn_shared_y_unchanged()
    assert fattn_path_unchanged()
    keep = _keep_predicates(result)
    if result["status"] == "measured":
        assert keep
        assert result["reverted"] is False
        assert result["keep_sitting_skipped"] is False
        assert path == "shared" == installed
        assert result["p"]["quartz"]["mean_tok_s"] > OPT034_P
        assert not REJECTION.is_file()
    else:
        assert not keep
        assert result["reverted"] is True
        assert path == "repeated"
        assert REJECTION.is_file()
        rejection = REJECTION.read_text()
        assert "repeated" in rejection
        if result["keep_sitting_skipped"]:
            assert installed == "repeated"
            assert result["p"] is None and result["d128"] is None
            assert result["d2048"] is None
        else:
            assert installed == "shared"
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
    ident: str, mean: float, *, eligible: bool = True, dispatched: bool = True
) -> dict[str, Any]:
    envelope = {"max_abs": 1.0e-9, "rms": 1.0e-10, "nonfinite": 0}
    return {
        "id": ident,
        "mean_ms": mean,
        "samples": [mean] * 30,
        "warmup_ms": [mean] * 3,
        "occupancy": 1,
        "recurrence_occupancy": 1,
        "launch_ok": True,
        "eligible": eligible,
        "dispatched_shared": dispatched if ident == "shared" else False,
        "inverses_byte_equal": True,
        "recurrent_byte_equal": True,
        "conv_byte_equal": True,
        "state_byte_equal": True,
        "gated_byte_equal": True,
        "prepare_preserves_committed": True,
        "cancel_isolates": True,
        "vs_sequential": dict(envelope),
    }


def _ab_stub(winner: str, means: dict[str, float]) -> dict[str, Any]:
    return {
        "token_count": 4096,
        "winner": winner,
        "win": winner == "shared",
        "candidates": {
            ident: _candidate_stub(ident, means[ident], dispatched=ident == "shared")
            for ident in ("repeated", "shared")
        },
    }


def _correctness_stub(*, shared_ok: bool = True) -> dict[str, Any]:
    cases = {}
    for layer in CORRECTNESS_LAYERS:
        for token in CORRECTNESS_TOKENS:
            case = {
                "layer": layer,
                "token_count": token,
                "launch_ok": True,
                "dispatched_shared": False if token == 1 else shared_ok,
                "inverses_byte_equal": True,
                "recurrent_byte_equal": True,
                "conv_byte_equal": True,
                "state_byte_equal": True,
                "gated_byte_equal": True,
                "prepare_preserves_committed": True,
                "cancel_isolates": True,
                "eligible": shared_ok,
                "vs_sequential": {"max_abs": 1.0e-9, "rms": 1.0e-10, "nonfinite": 0},
            }
            cases[f"{layer}:{token}"] = case
    return {
        "token_counts": CORRECTNESS_TOKENS,
        "layers": CORRECTNESS_LAYERS,
        "cases": cases,
        "all_eligible": shared_ok,
    }


def test_opt040_contract_and_source_pins() -> None:
    contract = _contract()
    opt034 = json.loads((ROOT / "fixtures/opt034_packed_mmv.json").read_text())
    opt039 = json.loads((ROOT / "fixtures/opt039_decode_warp.json").read_text())
    assert opt034["p"]["quartz"]["mean_tok_s"] == OPT034_P
    assert opt039["d128"]["quartz"]["mean_tok_s"] == OPT039_D128
    assert opt039["d2048"]["quartz"]["mean_tok_s"] == OPT039_D2048
    assert contract["opt034_quartz_p_mean_tok_s"] == OPT034_P
    assert contract["opt039_quartz_d128_mean_tok_s"] == OPT039_D128
    assert contract["opt039_quartz_d2048_mean_tok_s"] == OPT039_D2048
    assert contract["ab"]["candidates"] == ["repeated", "shared"]
    assert contract["ab"]["timed_token_counts"] == [4096]
    assert contract["ab"]["correctness_token_counts"] == CORRECTNESS_TOKENS
    assert contract["ab"]["correctness_layers"] == CORRECTNESS_LAYERS
    assert contract["ab"]["ties_retain"] == "repeated"
    assert contract["owns_opt016_parity_gate"] is False
    assert contract["substitutes_for_opt016"] is False
    assert contract["llama_bench_decode_is_informational"] is True
    gdn = json.loads((ROOT / "pins/cuda_gdn_chunk_contract.json").read_text())
    assert gdn["admission"]["maximum_absolute_error"] == 5.0e-8
    assert gdn["admission"]["maximum_rms_error"] == 5.0e-9
    assert pin_from_source() in LEGAL_PATHS
    assert fuse_path_unchanged()
    assert decode_gdn_unchanged()
    assert decode_vec_unchanged()
    assert rsqrtf_absent()
    source = (ROOT / "cuda/gdn_fused_quality.cuh").read_text()
    assert "prepare_gdn_shared_inverses" in source
    assert "prepare_recurrence_fused_warp_column_shared" in source
    assert "gdn_quality_load_qk_and_inverses" in source
    header = (ROOT / "cuda/gdn_step.h").read_text()
    assert "selected_gdn_inverse_path" in header
    assert "gdn_uses_shared_inverse" in header
    assert "gdn_shared_inverse_floats" in header
    assert "gdn_shared_inverse_occupancy" in header
    assert "inverse_path" in header
    makefile = (ROOT / "Makefile").read_text()
    assert "--fmad=false" in makefile
    assert "opt040_gdn_shared_inverse_ab_test" not in makefile
    assert (ROOT / "cuda/full_scheduler.cu").read_text().count(
        "launch_gdn_prepare_tiled("
    ) >= 1


def test_opt040_validator_rejects_inadmissible_evidence() -> None:
    means = {"repeated": 1.0, "shared": 2.0}
    fixture = {
        "schema_version": 1,
        "task": "OPT-040",
        "status": "rejected",
        "measurement_utc": "2026-09-10T00:00:00Z",
        "device": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": True,
        "selected_gdn_inverse_path": "repeated",
        "ab_p4096": _ab_stub("repeated", means),
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
        "report_path": "evidence/optimization/opt040-gdn-shared-inverse/REPORT.md",
    }
    mutations: list[dict[str, Any]] = []
    for mutate in (
        lambda x: x.__setitem__("substitutes_for_opt016", True),
        lambda x: x.__setitem__("owns_opt016_parity_gate", True),
        lambda x: x.__setitem__("nsight_systems", "/tmp/capture.nsys-rep"),
        lambda x: x.__setitem__("report_path", "/tmp/opt040/REPORT.md"),
        lambda x: x.__setitem__("status", "source_inspected"),
        lambda x: x.__setitem__("task", "OPT-029"),
        lambda x: x["ab_p4096"].__setitem__("winner", "shared"),
        lambda x: x.__setitem__("selected_gdn_inverse_path", "shared"),
        lambda x: x.__setitem__("keep_sitting_skipped", False),
        lambda x: x.__setitem__("reverted", False),
        lambda x: x.__setitem__("starts_opt041", True),
        lambda x: x.__setitem__("starts_opt042", True),
        lambda x: x.__setitem__("selected_gdn_fuse_path", "fuse_both"),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
    equality = json.loads(json.dumps(fixture))
    equality["status"] = "measured"
    equality["reverted"] = False
    equality["keep_sitting_skipped"] = False
    equality["selected_gdn_inverse_path"] = "shared"
    equality["ab_p4096"] = _ab_stub("shared", {"repeated": 2.0, "shared": 2.0})
    equality["p"] = {"quartz": {"mean_tok_s": OPT034_P + 1.0}}
    equality["d128"] = {
        "quartz": {
            "mean_tok_s": OPT039_D128,
            "token_latency_p95_ms": OPT039_D128_P95,
            "run_mean_token_latency_p95_ms": OPT039_D128_RUN_P95,
        }
    }
    equality["d2048"] = {
        "quartz": {
            "mean_tok_s": OPT039_D2048,
            "token_latency_p95_ms": OPT039_D2048_P95,
            "run_mean_token_latency_p95_ms": OPT039_D2048_RUN_P95,
        }
    }
    mutations.append(equality)
    stale_decode = json.loads(json.dumps(equality))
    stale_decode["ab_p4096"] = _ab_stub("shared", {"repeated": 2.0, "shared": 1.0})
    stale_decode["d2048"]["quartz"]["mean_tok_s"] = STALE_OPT034_D2048
    mutations.append(stale_decode)
    loosened = json.loads(json.dumps(fixture))
    loosened["ab_p4096"]["candidates"]["shared"]["vs_sequential"]["max_abs"] = 5.0e-7
    loosened["ab_p4096"]["candidates"]["shared"]["eligible"] = True
    loosened["ab_p4096"]["winner"] = "shared"
    loosened["ab_p4096"]["candidates"]["shared"]["mean_ms"] = 0.5
    mutations.append(loosened)
    skip_ab = json.loads(json.dumps(fixture))
    skip_ab.pop("ab_p4096")
    mutations.append(skip_ab)
    skip_correctness = json.loads(json.dumps(fixture))
    skip_correctness["correctness"]["cases"].pop("62:4096")
    mutations.append(skip_correctness)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    report_prev = REPORT.read_text() if REPORT.is_file() else None
    raw_prev = AB_RAW.read_text() if AB_RAW.is_file() else None
    rejection_prev = REJECTION.read_text() if REJECTION.is_file() else None
    source_prev = (ROOT / "cuda/gdn_fused_quality.cuh").read_text()
    contract_prev = CONTRACT.read_text()
    if report_prev is None:
        REPORT.write_text(PROOF + "\n")
    if raw_prev is None:
        AB_RAW.write_text("placeholder\n")
    if rejection_prev is None:
        REJECTION.write_text("rejected placeholder repeated\n")
    try:
        for mutation in mutations:
            with pytest.raises(AssertionError):
                validate_result(mutation)
        rsqrt = json.loads(json.dumps(fixture))
        (ROOT / "cuda/gdn_fused_quality.cuh").write_text(
            source_prev.replace(
                "1.0F / sqrtf(query_squares + kGdnQualityL2Epsilon)",
                "rsqrtf(query_squares + kGdnQualityL2Epsilon)",
                1,
            )
        )
        with pytest.raises(AssertionError):
            validate_result(rsqrt)
        fuse = json.loads(json.dumps(fixture))
        (ROOT / "cuda/gdn_fused_quality.cuh").write_text(source_prev)
        (ROOT / "cuda/gdn_fused_quality.cuh").write_text(
            source_prev.replace(
                'kSelectedGdnFusePath[] = "off"',
                'kSelectedGdnFusePath[] = "fuse_both"',
                1,
            )
        )
        with pytest.raises(AssertionError):
            validate_result(fuse)
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
        (ROOT / "cuda/gdn_fused_quality.cuh").write_text(source_prev)
        CONTRACT.write_text(contract_prev)


def test_opt040_fixture_connected() -> None:
    if not FIXTURE.is_file():
        pytest.skip("OPT-040 fixture is written by the exclusive CUDA sitting")
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


def _set_pin(path: str) -> None:
    source = ROOT / "cuda/gdn_fused_quality.cuh"
    text = source.read_text()
    text = re.sub(
        r'kSelectedGdnInversePath\[\] = "[^"]+"',
        f'kSelectedGdnInversePath[] = "{path}"',
        text,
    )
    source.write_text(text)
    contract = _contract()
    contract["selected_gdn_inverse_path"] = path
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
    text = f"""# OPT-040 — Hoist prompt GDN inverse normalization

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of shared per-token/key-head Q/K inverse
norms against repeated warp-column L2. Keep requires
**byte-equal quality outputs/state**, **frozen sequential GDN gates**,
**lower complete GDN component time**, **improved P** versus OPT-034,
**95% throughput floors**, and **105% p95 ceilings**. This increment
does not substitute for the 2K llama.cpp parity gate. Copied denominators
are P {OPT034_P}, D128 {OPT039_D128}, D2048 {OPT039_D2048}.

## Decision

**{decision}** — `reverted`={json.dumps(fixture["reverted"])};
`keep_sitting_skipped`={json.dumps(fixture["keep_sitting_skipped"])};
selected_gdn_inverse_path={fixture["selected_gdn_inverse_path"]};
4096 A/B winner {ab["winner"]} (repeated {ab["candidates"]["repeated"]["mean_ms"]} ms,
shared {ab["candidates"]["shared"]["mean_ms"]} ms);
tok/s sitting {sitting}.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `{GGUF_SHA}` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `{LLAMA_REV}` |
| A/B | token_count 4096, candidates repeated then shared, 3 warm + 30 alternating |
| Correctness | tokens {CORRECTNESS_TOKENS}; layers {CORRECTNESS_LAYERS} |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: {fixture["device"]} compute {fixture["compute_capability"]}
- measurement_utc: {fixture["measurement_utc"]}
- P Quartz mean tok/s: {quartz_p} versus OPT-034 {OPT034_P}
- D128 Quartz mean tok/s: {quartz_d128} versus OPT-039 {OPT039_D128}
- D2048 Quartz mean tok/s: {quartz_d2048} versus OPT-039 {OPT039_D2048}
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false
"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def _write_rejection(fixture: dict[str, Any]) -> None:
    ab = fixture["ab_p4096"]
    reason = (
        "4096 component A/B selected repeated"
        if fixture["keep_sitting_skipped"]
        else "keep sitting failed P improvement or the cross-workload guard"
    )
    tok = (
        ""
        if fixture["keep_sitting_skipped"]
        else f"\n- P Quartz mean tok/s: {fixture['p']['quartz']['mean_tok_s']}\n"
        f"- OPT-034 P baseline: {OPT034_P}\n"
        f"- D128 Quartz mean tok/s: {fixture['d128']['quartz']['mean_tok_s']}\n"
        f"- D2048 Quartz mean tok/s: {fixture['d2048']['quartz']['mean_tok_s']}"
    )
    text = f"""# OPT-040 rejection

{reason}. Production prompt GDN inverse remains the repeated warp-column
path (`selected_gdn_inverse_path=repeated`).

- 4096 A/B winner: {ab["winner"]} repeated={ab["candidates"]["repeated"]["mean_ms"]} shared={ab["candidates"]["shared"]["mean_ms"]}
- keep_sitting_skipped: {json.dumps(fixture["keep_sitting_skipped"])}
- measurement_utc: {fixture["measurement_utc"]}{tok}
"""
    REJECTION.write_text(text)


def _run_ab() -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    commands = [
        [*_common(IMAGE), "make", "build/qw38-cuda-timing-test"],
        [*_common(IMAGE), "make", "build/qw38-cuda-gdn-test"],
        [*_common(IMAGE), "make", "build/qw38-cuda-gdn-chunk-test"],
        _nvcc(
            "cuda/opt040_gdn_shared_inverse_ab_test.cu",
            "build/qw38-cuda-opt040-gdn-shared-inverse-ab-test",
            ["build/gdn_step.cuda.o"],
        ),
        [
            *_common(IMAGE),
            "./build/qw38-cuda-opt040-gdn-shared-inverse-ab-test",
            "evidence/optimization/opt040-gdn-shared-inverse/gdn-ab-raw.txt",
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
            f"evidence/optimization/opt040-gdn-shared-inverse/{sidecar}"
        )
    for key in ("n_gpu_layers", "n_ctx", "n_batch", "n_ubatch"):
        if key in record:
            block[key] = record[key]
    return block


def test_opt040_native_keep_reject() -> None:
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
    if selected == "repeated":
        _set_pin("repeated")
        fixture = {
            "schema_version": 1,
            "task": "OPT-040",
            "status": "rejected",
            "measurement_utc": ab["measurement_utc"],
            "device": ab["device"],
            "compute_capability": ab["compute_capability"],
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "reverted": True,
            "selected_gdn_inverse_path": "repeated",
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
            "report_path": "evidence/optimization/opt040-gdn-shared-inverse/REPORT.md",
        }
        _write_report(fixture)
        _write_rejection(fixture)
        FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
        validate_result(fixture)
        return

    _set_pin("shared")
    _run([*_common(IMAGE), "make", "build/qw38-cuda-timing-test"])
    llama_p = _run_llama_bench_p()
    quartz_p = _run_quartz_p()
    llama_d128 = _run_llama_decode(128)
    quartz_d128 = _run_quartz_decode(128)
    llama_d2048 = _run_llama_decode(2048)
    quartz_d2048 = _run_quartz_decode(2048)
    fixture = {
        "schema_version": 1,
        "task": "OPT-040",
        "status": "measured",
        "measurement_utc": quartz_p.get("measurement_utc", ab["measurement_utc"]),
        "device": ab["device"],
        "compute_capability": ab["compute_capability"],
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": False,
        "selected_gdn_inverse_path": "shared",
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
        "report_path": "evidence/optimization/opt040-gdn-shared-inverse/REPORT.md",
    }
    keep = _keep_predicates(fixture)
    if not keep:
        _set_pin("repeated")
        fixture["status"] = "rejected"
        fixture["reverted"] = True
        fixture["selected_gdn_inverse_path"] = "repeated"
        _write_report(fixture)
        _write_rejection(fixture)
        _run([*_common(IMAGE), "make", "build/qw38-cuda-timing-test"])
    else:
        if REJECTION.is_file():
            REJECTION.unlink()
        _write_report(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    validate_result(json.loads(FIXTURE.read_text()))
