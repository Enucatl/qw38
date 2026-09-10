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
CONTRACT = ROOT / "pins/opt045_parallel_norm_contract.json"
FIXTURE = ROOT / "fixtures/opt045_parallel_norm.json"
OPT041_FIXTURE = ROOT / "fixtures/opt041_fattn_warp_qk.json"
EVIDENCE = ROOT / "evidence/optimization/opt045-parallel-norm"
REPORT = EVIDENCE / "REPORT.md"
REJECTION = EVIDENCE / "REJECTION.md"
AB_RAW = EVIDENCE / "parallel-norm-ab-raw.txt"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
AB_PREFIX = "QW38_OPT045_PARALLEL_NORM_AB_RESULT="
P_PREFIX = "QW38_PREFILL_4K_ORACLE_RESULT="
DECODE_PREFIX = "QW38_DECODE_ORACLE_RESULT="
LLAMA_DECODE_PREFIX = "QW38_LLAMA_DECODE_ORACLE_RESULT="
OPT041_P = 2076.98315
OPT041_D128 = 26.1599541
OPT041_D2048 = 25.2924843
OPT041_D128_P95 = 38.3730087
OPT041_D128_RUN_P95 = 38.2578201
OPT041_D2048_P95 = 39.6526222
OPT041_D2048_RUN_P95 = 39.5695038
LEGAL_PATHS = {"serial", "parallel_fma", "parallel_rsqrt"}
LEGAL_CANDIDATES = (
    "serial",
    "parallel_fma_t128",
    "parallel_fma_t256",
    "parallel_rsqrt_t128",
    "parallel_rsqrt_t256",
)
PROOF = (
    "OPT-044 production-numerics budgets; serial strict reference retained; "
    "cooperative fmaf reduction; sqrt/div vs rsqrt A/B; "
    "lower complete component time; 95% throughput floors; "
    "105% p95 ceilings; does not substitute for the 2K llama.cpp parity gate"
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
    text = (ROOT / "cuda/rms_norm.cuh").read_text()
    match = re.search(r'kSelectedRmsNormPath\[\] = "([^"]+)"', text)
    assert match is not None
    return match.group(1)


def threads_from_source() -> int:
    text = (ROOT / "cuda/rms_norm.cuh").read_text()
    match = re.search(r"kSelectedRmsNormThreads = ([0-9]+)", text)
    assert match is not None
    return int(match.group(1))


def gdn_threads_from_source() -> int:
    text = (ROOT / "cuda/rms_norm.cuh").read_text()
    match = re.search(r"kSelectedGdnNormThreads = ([0-9]+)", text)
    assert match is not None
    return int(match.group(1))


def serial_reference_retained() -> bool:
    text = (ROOT / "cuda/full_scheduler.cu").read_text()
    primitives = (ROOT / "cuda/scheduler_primitives.cu").read_text()
    return (
        "if (threadIdx.x == 0)" in text
        and "__fadd_rn(sum," in text
        and "__fmul_rn(row_input[index], row_input[index])" in text
        and "if (threadIdx.x == 0)" in primitives
        and "rms_norm_fp32_to_bf16<<<1, kThreads, 0, stream>>>(" in text
    )


def parallel_design_present() -> bool:
    header = (ROOT / "cuda/rms_norm.cuh").read_text()
    sched = (ROOT / "cuda/full_scheduler.cu").read_text()
    primitives = (ROOT / "cuda/scheduler_primitives.cu").read_text()
    return (
        "fmaf(value, value, sum)" in header
        and "__shfl_down_sync" in header
        and "cooperative_rms_inverse" in header
        and "rms_norm_fp32_to_bf16_parallel" in sched
        and "residual_add_norm_rows_fp32_to_bf16_parallel" in sched
        and "gdn_gated_output_rows_parallel" in sched
        and "rms_norm_parallel" in primitives
        and "gated_output_parallel" in primitives
        and "rsqrtf(mean)" in header
        and "1.0F / sqrtf(mean)" in header
        and "__fmul_rn(__fmul_rn(value, inverse), scale)" in header
    )


def mmv_qk_untouched() -> bool:
    mmv = (ROOT / "cuda/quant_mmv.cu").read_text()
    fattn = (ROOT / "cuda/fattn_mma_f16.cuh").read_text()
    return (
        'kSelectedMmvLoadPath[] = "packed"' in mmv
        and 'kSelectedQKPath[] = "warp_microtile"' in fattn
    )


def _ab_candidate(block: dict[str, Any], ident: str) -> None:
    cand = block["candidates"][ident]
    assert cand["id"] == ident
    assert cand["occupancy"] >= 0
    assert cand["decode_rms"]["launch_ok"] is True
    assert cand["prompt_norm"]["launch_ok"] is True
    assert len(cand["decode_rms"]["samples"]) == 30
    assert len(cand["prompt_norm"]["samples"]) == 30
    assert len(cand["decode_rms"]["warmup_ms"]) == 3
    mean = sum(float(v) for v in cand["prompt_norm"]["samples"]) / 30.0
    assert cand["prompt_norm"]["mean_ms"] == pytest.approx(mean, rel=1e-6, abs=1e-6)
    assert cand["vs_fp64"]["nonfinite"] == 0
    assert cand["zeros"]["nonfinite"] == 0
    assert cand["extremes"]["nonfinite"] == 0
    assert cand["residual_fp32_exact"] is True
    if cand["eligible"] and ident != "serial":
        assert cand["removed_serial_dependency"] is True
        assert cand["registers"] >= 1


def _select_component_winner(block: dict[str, Any]) -> str:
    serial = block["candidates"]["serial"]
    if not serial["eligible"]:
        return "serial"
    serial_mean = float(serial["prompt_norm"]["mean_ms"])
    fast: list[str] = []
    for ident in LEGAL_CANDIDATES:
        if ident == "serial":
            continue
        cand = block["candidates"][ident]
        if cand["eligible"] and float(cand["prompt_norm"]["mean_ms"]) < serial_mean:
            fast.append(ident)
    if not fast:
        return "serial"
    best = fast[0]
    best_mean = float(block["candidates"][best]["prompt_norm"]["mean_ms"])
    for ident in fast[1:]:
        cand = block["candidates"][ident]
        mean = float(cand["prompt_norm"]["mean_ms"])
        current = block["candidates"][best]
        if (
            cand["path"] == "parallel_rsqrt"
            and current["path"] == "parallel_fma"
            and cand["residual_threads"] == current["residual_threads"]
        ):
            if mean < 0.98 * best_mean:
                best = ident
                best_mean = mean
            continue
        if mean < best_mean:
            best = ident
            best_mean = mean
    return best


def _select_install(result: dict[str, Any]) -> str:
    if not result["ab"]["candidates"]["serial"]["eligible"]:
        return "serial"
    winner = _select_component_winner(result["ab"])
    if winner == "serial":
        return "serial"
    path = result["ab"]["candidates"][winner]["path"]
    if path not in LEGAL_PATHS:
        return "serial"
    return winner


def _keep_predicates(result: dict[str, Any]) -> bool:
    contract = _contract()
    if result["selected_rms_norm_path"] == "serial":
        return False
    if result["reverted"] is True or result["keep_sitting_skipped"] is True:
        return False
    if result["status"] != "measured":
        return False
    if _select_install(result) == "serial":
        return False
    if result["p"] is None or result["d128"] is None or result["d2048"] is None:
        return False
    if (
        float(result["p"]["quartz"]["mean_tok_s"])
        < 0.95 * contract["opt041_quartz_p_mean_tok_s"]
    ):
        return False
    if (
        float(result["d128"]["quartz"]["mean_tok_s"])
        < 0.95 * contract["opt041_quartz_d128_mean_tok_s"]
    ):
        return False
    if (
        float(result["d2048"]["quartz"]["mean_tok_s"])
        < 0.95 * contract["opt041_quartz_d2048_mean_tok_s"]
    ):
        return False
    if (
        result["d128"]["quartz"]["token_latency_p95_ms"]
        > 1.05 * contract["opt041_quartz_d128_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d128"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * contract["opt041_quartz_d128_run_mean_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d2048"]["quartz"]["token_latency_p95_ms"]
        > 1.05 * contract["opt041_quartz_d2048_token_latency_p95_ms"]
    ):
        return False
    if (
        result["d2048"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * contract["opt041_quartz_d2048_run_mean_token_latency_p95_ms"]
    ):
        return False
    return True


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-045"
    assert result["status"] in ("measured", "rejected")
    assert result["status"] not in ("scout", "source_inspected")
    assert contract["yardstick"] == "cooperative_parallel_rmsnorm"
    assert contract["opt041_quartz_p_mean_tok_s"] == OPT041_P
    assert contract["opt041_quartz_d128_mean_tok_s"] == OPT041_D128
    assert contract["opt041_quartz_d2048_mean_tok_s"] == OPT041_D2048
    assert contract["opt044"]["cross_path_exact_bits_required"] is False
    assert contract["opt044"]["residual_fp32_exact"] is True
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    path = result["selected_rms_norm_path"]
    assert path in LEGAL_PATHS
    assert result["selected_rms_norm_threads"] in (128, 256)
    assert result["selected_gdn_norm_threads"] in (32, 128, 256)
    assert contract["ab"]["candidates"] == list(LEGAL_CANDIDATES)
    assert contract["ab"]["ties_retain"] == "serial"
    assert _select_component_winner(result["ab"]) == result["ab"]["winner"]
    for ident in LEGAL_CANDIDATES:
        _ab_candidate(result["ab"], ident)
    numerics = result["production_numerics"]
    assert "budget" in numerics and "strict_vs_fp64" in numerics
    assert numerics["budget"]["max_abs"] >= 9.9e-7
    serial = result["ab"]["candidates"]["serial"]
    assert serial["path"] == "serial"
    for ident in LEGAL_CANDIDATES:
        cand = result["ab"]["candidates"][ident]
        if cand["eligible"]:
            assert cand["vs_fp64"]["max_abs"] <= numerics["budget"]["max_abs"]
            assert cand["vs_fp64"]["rms"] <= numerics["budget"]["rms"]
    assert pin_from_source() == result["selected_rms_norm_path"]
    assert threads_from_source() == result["selected_rms_norm_threads"]
    assert gdn_threads_from_source() == result["selected_gdn_norm_threads"]
    assert serial_reference_retained()
    assert parallel_design_present()
    keep = _keep_predicates(result)
    if result["status"] == "measured":
        assert keep is True
        assert result["reverted"] is False
        assert REPORT.is_file()
        assert (
            PROOF.split(";")[0].strip()[:12] in REPORT.read_text()
            or "OPT-044" in REPORT.read_text()
        )
    else:
        assert keep is False
        assert result["selected_rms_norm_path"] == "serial"
        assert result["reverted"] is True
        if result["keep_sitting_skipped"]:
            assert result["p"] is None
        else:
            assert result["p"] is not None
        assert REJECTION.is_file()
        assert "serial" in REJECTION.read_text()
    assert AB_RAW.is_file()
    assert result["report_path"] == contract["report_path"]
    assert result["proof_limit"] == PROOF


def _ab_stub(
    winner: str, means: dict[str, float], eligible: bool = True
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for ident in LEGAL_CANDIDATES:
        path = (
            "serial"
            if ident == "serial"
            else ("parallel_rsqrt" if "rsqrt" in ident else "parallel_fma")
        )
        threads = 256 if ident == "serial" or ident.endswith("t256") else 128
        gdn = 256 if ident == "serial" else (32 if ident.endswith("t256") else 128)
        mean = means.get(ident, 1.0)
        samples = [mean] * 30
        timed = {
            "mean_ms": mean,
            "launch_ok": True,
            "warmup_ms": [mean, mean, mean],
            "samples": samples,
        }
        candidates[ident] = {
            "id": ident,
            "path": path,
            "residual_threads": threads,
            "gdn_threads": gdn,
            "occupancy": 1,
            "registers": 32,
            "local_bytes": 0,
            "eligible": eligible,
            "removed_serial_dependency": ident != "serial",
            "decode_rms": timed,
            "prompt_norm": timed,
            "decode_residual": timed,
            "gdn": timed,
            "complete_ffn": timed,
            "vs_serial": {
                "max_abs": 0.0,
                "rms": 0.0,
                "one_minus_cosine": 0.0,
                "nonfinite": 0,
            },
            "vs_fp64": {
                "max_abs": 1e-6,
                "rms": 1e-7,
                "one_minus_cosine": 0.0,
                "nonfinite": 0,
            },
            "residual_fp32_exact": True,
            "gdn_vs_serial": {"max_abs": 0.0, "rms": 0.0, "nonfinite": 0},
            "zeros": {"max_abs": 0.0, "rms": 0.0, "nonfinite": 0},
            "extremes": {"max_abs": 0.001, "rms": 0.0001, "nonfinite": 0},
            "opt043": {"max_abs": 1e-6, "rms": 1e-7, "nonfinite": 0},
            "primitives": {"max_abs": 0.0, "rms": 0.0, "nonfinite": 0},
        }
    return {
        "winner": winner,
        "win": winner != "serial",
        "candidates": candidates,
    }


def _keep_tok_s() -> dict[str, Any]:
    return {
        "p": {"quartz": {"mean_tok_s": OPT041_P * 1.01}},
        "d128": {
            "quartz": {
                "mean_tok_s": OPT041_D128,
                "token_latency_p95_ms": OPT041_D128_P95,
                "run_mean_token_latency_p95_ms": OPT041_D128_RUN_P95,
            }
        },
        "d2048": {
            "quartz": {
                "mean_tok_s": OPT041_D2048,
                "token_latency_p95_ms": OPT041_D2048_P95,
                "run_mean_token_latency_p95_ms": OPT041_D2048_RUN_P95,
            }
        },
    }


def _reject_fixture() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-045",
        "status": "rejected",
        "measurement_utc": "2026-09-10T00:00:00Z",
        "device": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": True,
        "selected_rms_norm_path": "serial",
        "selected_rms_norm_threads": 256,
        "selected_gdn_norm_threads": 256,
        "ab": _ab_stub("serial", {ident: 1.0 for ident in LEGAL_CANDIDATES}),
        "production_numerics": {
            "formula": "max(strict_reference_ceiling, 1.05 * measured_llama_error + 1e-6)",
            "strict_vs_fp64": {
                "max_abs": 1e-6,
                "rms": 1e-7,
                "one_minus_cosine": 0.0,
                "nonfinite": 0,
            },
            "llama_vs_fp64": {
                "max_abs": 1e-6,
                "rms": 1e-7,
                "one_minus_cosine": 0.0,
                "nonfinite": 0,
            },
            "budget": {"max_abs": 2.05e-6, "rms": 1.05e-6, "one_minus_cosine": 1e-6},
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
        "report_path": "evidence/optimization/opt045-parallel-norm/REPORT.md",
    }


def test_opt045_contract_and_source_pins() -> None:
    contract = _contract()
    opt041 = json.loads(OPT041_FIXTURE.read_text())
    assert opt041["p"]["quartz"]["mean_tok_s"] == OPT041_P
    assert contract["opt041_quartz_p_mean_tok_s"] == OPT041_P
    assert contract["opt041_quartz_d2048_mean_tok_s"] == OPT041_D2048
    assert contract["ab"]["candidates"] == list(LEGAL_CANDIDATES)
    assert contract["ab"]["geometries_tested"] == [128, 256]
    assert contract["ab"]["second_ab"] == "sqrt_div_vs_rsqrt"
    assert contract["owns_opt016_parity_gate"] is False
    assert pin_from_source() in LEGAL_PATHS
    assert serial_reference_retained()
    assert parallel_design_present()
    assert mmv_qk_untouched()
    makefile = (ROOT / "Makefile").read_text()
    assert "--fmad=false" in makefile
    assert "opt045_parallel_norm_ab_test" not in makefile
    assert "cuda/rms_norm.cuh" in makefile


def test_opt045_validator_rejects_inadmissible_evidence() -> None:
    fixture = _reject_fixture()
    mutations: list[dict[str, Any]] = []
    for mutate in (
        lambda x: x.__setitem__("substitutes_for_opt016", True),
        lambda x: x.__setitem__("owns_opt016_parity_gate", True),
        lambda x: x.__setitem__("status", "source_inspected"),
        lambda x: x.__setitem__("task", "OPT-044"),
        lambda x: x["ab"].__setitem__("winner", "parallel_fma_t256"),
        lambda x: x.__setitem__("selected_rms_norm_path", "parallel_fma"),
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
        REJECTION.write_text("rejected placeholder serial\n")
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


def test_opt045_fixture_connected() -> None:
    if not FIXTURE.is_file():
        pytest.skip("OPT-045 fixture is written by the exclusive CUDA sitting")
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


def _set_pin(path: str, residual_threads: int, gdn_threads: int) -> None:
    source = ROOT / "cuda/rms_norm.cuh"
    text = source.read_text()
    text = re.sub(
        r'kSelectedRmsNormPath\[\] = "[^"]+"',
        f'kSelectedRmsNormPath[] = "{path}"',
        text,
    )
    text = re.sub(
        r"kSelectedRmsNormThreads = [0-9]+",
        f"kSelectedRmsNormThreads = {residual_threads}",
        text,
    )
    text = re.sub(
        r"kSelectedGdnNormThreads = [0-9]+",
        f"kSelectedGdnNormThreads = {gdn_threads}",
        text,
    )
    source.write_text(text)
    contract = _contract()
    contract["selected_rms_norm_path"] = path
    contract["selected_rms_norm_threads"] = residual_threads
    contract["selected_gdn_norm_threads"] = gdn_threads
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
    serial_ms = ab["candidates"]["serial"]["prompt_norm"]["mean_ms"]
    winner = ab["winner"]
    winner_ms = ab["candidates"][winner]["prompt_norm"]["mean_ms"]
    text = f"""# OPT-045 — Parallelize normalization and use admitted FMA

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of cooperative RMSNorm (fmaf squares, warp shuffle
plus shared warp-sum, one inverse broadcast, single BF16 rounding) against the
serial thread-0 strict reference. Sqrt/divide versus admitted rsqrt is a second
A/B. Keep requires **OPT-044 production-numerics budgets**, **retained serial
reference**, **lower complete component time**, **95% P/D throughput floors**,
and **105% decode p95 ceilings**. This increment does not substitute for the 2K
llama.cpp parity gate. Copied denominators are P {OPT041_P}, D128 {OPT041_D128},
D2048 {OPT041_D2048}.

## Decision

**{decision}** — `reverted`={json.dumps(fixture["reverted"])};
`keep_sitting_skipped`={json.dumps(fixture["keep_sitting_skipped"])};
selected_rms_norm_path={fixture["selected_rms_norm_path"]};
threads={fixture["selected_rms_norm_threads"]}/gdn={fixture["selected_gdn_norm_threads"]};
A/B winner {winner} (serial {serial_ms} ms, winner {winner_ms} ms);
tok/s sitting {sitting}.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `{GGUF_SHA}` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp revision | `{LLAMA_REV}` |
| A/B | 4096-row residual+norm, candidates {list(LEGAL_CANDIDATES)}, 3 warm + 30 alternating |
| Numerics | OPT-044 formula vs FP64 and llama cooperative replica |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols |
| Nsight | not_used |

## Measured sitting

- Device: {fixture["device"]} compute {fixture["compute_capability"]}
- measurement_utc: {fixture["measurement_utc"]}
- P Quartz mean tok/s: {quartz_p} versus OPT-041 {OPT041_P}
- D128 Quartz mean tok/s: {quartz_d128} versus OPT-041 {OPT041_D128}
- D2048 Quartz mean tok/s: {quartz_d2048} versus OPT-041 {OPT041_D2048}
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false
"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def _write_rejection(fixture: dict[str, Any]) -> None:
    ab = fixture["ab"]
    reason = (
        "component A/B retained serial"
        if fixture["keep_sitting_skipped"]
        else "keep sitting failed OPT-044 numerics or the P/D guard"
    )
    tok = (
        ""
        if fixture["keep_sitting_skipped"]
        else f"\n- P Quartz mean tok/s: {fixture['p']['quartz']['mean_tok_s']}\n"
        f"- OPT-041 P baseline: {OPT041_P}\n"
        f"- D128 Quartz mean tok/s: {fixture['d128']['quartz']['mean_tok_s']}\n"
        f"- D2048 Quartz mean tok/s: {fixture['d2048']['quartz']['mean_tok_s']}"
    )
    text = f"""# OPT-045 rejection

{reason}. Production RMSNorm remains the serial thread-0 reference
(`selected_rms_norm_path=serial`).

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
            "build/full_scheduler.trace.cuda.o",
            "build/scheduler_primitives.cuda.o",
            "build/quant_mmv.cuda.o",
            "build/gdn_step.cuda.o",
            "build/attention_decode.cuda.o",
            "diagnostic",
        ],
        _nvcc(
            "cuda/opt045_parallel_norm_ab_test.cu",
            "build/qw38-cuda-opt045-parallel-norm-ab-test",
        ),
        [
            *_common(IMAGE),
            "./build/qw38-cuda-opt045-parallel-norm-ab-test",
            "evidence/optimization/opt045-parallel-norm/parallel-norm-ab-raw.txt",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ],
    ]
    outputs: list[str] = []
    for command in commands:
        outputs.append(_run(command).stdout)
    record = _parse_prefixed(outputs[-1], AB_PREFIX)
    assert "status=passed" in outputs[-1]
    return record


def _parse_llama_bench(text: str, predicate: Any, what: str) -> list[dict[str, Any]]:
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
    winner = ab["winner"]
    cand = (
        ab["candidates"][winner] if selected != "serial" else ab["candidates"]["serial"]
    )
    return {
        "schema_version": 1,
        "task": "OPT-045",
        "status": "rejected",
        "measurement_utc": ab["measurement_utc"],
        "device": ab["device"],
        "compute_capability": ab["compute_capability"],
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": True,
        "selected_rms_norm_path": "serial",
        "selected_rms_norm_threads": 256,
        "selected_gdn_norm_threads": 256,
        "ab": {
            "winner": ab["winner"],
            "win": ab["win"],
            "candidates": ab["candidates"],
        },
        "production_numerics": ab["production_numerics"],
        "keep_sitting_skipped": selected == "serial",
        "p": None,
        "d128": None,
        "d2048": None,
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt045-parallel-norm/REPORT.md",
        "_install": selected,
        "_cand": cand,
    }


@pytest.mark.skipif(
    os.environ.get("QW38_RUN_CUDA_TESTS") != "1",
    reason="set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate",
)
def test_opt045_exclusive_cuda_sitting() -> None:
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
    wrapper.pop("_cand")
    if selected == "serial":
        _set_pin("serial", 256, 256)
        _write_report(wrapper)
        _write_rejection(wrapper)
        FIXTURE.write_text(json.dumps(wrapper, indent=2) + "\n")
        validate_result(wrapper)
        return

    cand = ab["candidates"][selected]
    _set_pin(cand["path"], cand["residual_threads"], cand["gdn_threads"])
    _run([*_common(IMAGE), "make", "build/full_scheduler.trace.cuda.o"])
    llama_p = _run_llama_bench_p()
    quartz_p = _run_quartz_p()
    llama_d128 = _run_llama_decode(128)
    quartz_d128 = _run_quartz_decode(128)
    llama_d2048 = _run_llama_decode(2048)
    quartz_d2048 = _run_quartz_decode(2048)
    fixture = wrapper
    fixture["keep_sitting_skipped"] = False
    fixture["reverted"] = False
    fixture["selected_rms_norm_path"] = cand["path"]
    fixture["selected_rms_norm_threads"] = cand["residual_threads"]
    fixture["selected_gdn_norm_threads"] = cand["gdn_threads"]
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
        },
        "llama": {"mean_tok_s": llama_p.get("avg_ts", llama_p.get("mean_tok_s"))},
    }
    fixture["d128"] = {
        "quartz": _engine_block(quartz_d128),
        "llama": _engine_block(llama_d128),
    }
    fixture["d2048"] = {
        "quartz": _engine_block(quartz_d2048),
        "llama": _engine_block(llama_d2048),
    }
    if _keep_predicates(fixture):
        fixture["status"] = "measured"
        _write_report(fixture)
        if REJECTION.is_file():
            REJECTION.unlink()
    else:
        fixture["status"] = "rejected"
        fixture["reverted"] = True
        fixture["selected_rms_norm_path"] = "serial"
        fixture["selected_rms_norm_threads"] = 256
        fixture["selected_gdn_norm_threads"] = 256
        _set_pin("serial", 256, 256)
        _write_report(fixture)
        _write_rejection(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    validate_result(fixture)
