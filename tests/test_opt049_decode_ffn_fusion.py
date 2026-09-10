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
CONTRACT = ROOT / "pins/opt049_decode_ffn_fusion_contract.json"
FIXTURE = ROOT / "fixtures/opt049_decode_ffn_fusion.json"
OPT045_FIXTURE = ROOT / "fixtures/opt045_parallel_norm.json"
OPT046_FIXTURE = ROOT / "fixtures/opt046_q4_decode.json"
OPT047_FIXTURE = ROOT / "fixtures/opt047_q8_decode.json"
OPT048_FIXTURE = ROOT / "fixtures/opt048_q6_logits.json"
NUMERICS = ROOT / "pins/production_numerics_contract.json"
EVIDENCE = ROOT / "evidence/optimization/opt049-decode-ffn-fusion"
REPORT = EVIDENCE / "REPORT.md"
REJECTION = EVIDENCE / "REJECTION.md"
AB_RAW = EVIDENCE / "decode-ffn-fusion-ab-raw.txt"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
AB_PREFIX = "QW38_OPT049_DECODE_FFN_FUSION_AB_RESULT="
P_PREFIX = "QW38_PREFILL_4K_ORACLE_RESULT="
DECODE_PREFIX = "QW38_DECODE_ORACLE_RESULT="
OPT045_P = 2130.41089
OPT045_D128 = 28.5522804
OPT045_D2048 = 27.5438766
OPT045_D128_P95 = 35.1905479
OPT045_D128_RUN_P95 = 35.0761757
OPT045_D2048_P95 = 36.4285774
OPT045_D2048_RUN_P95 = 36.3359795
OPT048_P = 2129.85938
OPT048_D128 = 35.9651642
OPT048_D2048 = 34.3371162
LEGAL_PATHS = {"separate", "shared_stage", "paired", "paired_staged"}
LEGAL_CANDIDATES = ("separate", "shared_stage", "paired", "paired_staged")
SHAPES = ("ffn_probe", "ffn_layer0", "ffn_layer3", "ffn_layer31")
PROOF = (
    "OPT-044 production-numerics budgets; separate-leg packed Q4_K control "
    "retained; shared staging of normalized FFN input; two-pointer paired "
    "gate/up Q4_K kernel; admitted SwiGLU and single BF16 rounding; "
    "down/residual/norm stay on accepted kernels; graph/eager equality "
    "within each path; complete FFN cost includes stage/SwiGLU/down/norm; "
    "95% throughput floors versus OPT-048 keep; 105% p95 ceilings versus "
    "OPT-045; does not substitute for the 2K llama.cpp parity gate"
)
NVCC_OBJECTS = [
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
]


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def pin_from_source() -> str:
    text = (ROOT / "cuda/ffn_decode_path.cuh").read_text()
    match = re.search(r'kSelectedFfnDecodePath\[\] = "([^"]+)"', text)
    assert match is not None
    return match.group(1)


def packed_q4_retained() -> bool:
    q4 = (ROOT / "cuda/q4k_decode_path.cuh").read_text()
    mmv = (ROOT / "cuda/quant_mmv.cu").read_text()
    return (
        'kSelectedQ4DecodePath[] = "packed"' in q4
        and 'kSelectedMmvLoadPath[] = "packed"' in mmv
        and "sum = __fadd_rn(sum, __fmul_rn(weight, value));" in mmv
    )


def fusion_design_present() -> bool:
    mmv = (ROOT / "cuda/quant_mmv.cu").read_text()
    sched = (ROOT / "cuda/full_scheduler.cu").read_text()
    execute = sched[sched.find("cudaError_t execute_ffn") :]
    execute = execute[: execute.find("\ncudaError_t ", 1)]
    return (
        "quant_mmv_q4k_gate_up_swiglu" in mmv
        and "admitted_swiglu" in mmv
        and "launch_q4k_gate_up_swiglu_prequant" in mmv
        and "launch_quantize_bf16_q8" in execute
        and "matrix_vector(layer.ffn_gate" in execute
        and "matrix_vector(layer.ffn_down" in execute
        and "launch_swiglu_bf16" in execute
        and "concatenat" not in mmv.casefold()
    )


def prior_pins_untouched() -> bool:
    q8 = (ROOT / "cuda/q8_decode_path.cuh").read_text()
    q6 = (ROOT / "cuda/q6k_decode_path.cuh").read_text()
    return (
        'kSelectedQ8DecodePath[] = "dp4a_q8_1"' in q8
        and 'kSelectedQ6DecodePath[] = "integer_q8_1"' in q6
    )


def _ab_candidate(block: dict[str, Any], ident: str) -> None:
    cand = block["candidates"][ident]
    assert cand["id"] == ident
    assert cand["path"] in LEGAL_PATHS
    assert cand["occupancy"] >= 0
    assert "shapes" in cand
    for shape in cand["shapes"]:
        slot = cand["shapes"][shape]
        assert slot["complete"]["launch_ok"] is True
        assert len(slot["complete"]["samples"]) in (1, 3, 30)
        mean = sum(float(v) for v in slot["complete"]["samples"]) / len(
            slot["complete"]["samples"]
        )
        assert slot["complete"]["mean_ms"] == pytest.approx(mean, rel=1e-6, abs=1e-6)
        assert slot["vs_separate"]["nonfinite"] == 0
        if ident in ("shared_stage", "paired_staged") and shape != "ffn_probe":
            assert slot["vs_separate"]["max_abs"] <= 1.0e-4
        if ident == "separate":
            assert slot["graph_eager_equal"] is True
        if ident in ("shared_stage", "paired_staged", "paired"):
            assert "graph_eager_equal" in slot


def _select_component_winner(block: dict[str, Any]) -> str:
    separate = block["candidates"]["separate"]
    if not separate["eligible"]:
        return "separate"
    best = "separate"
    best_mean = float(separate["weighted_complete_ms"])
    for ident in LEGAL_CANDIDATES:
        if ident == "separate":
            continue
        cand = block["candidates"][ident]
        if not cand["eligible"]:
            continue
        mean = float(cand["weighted_complete_ms"])
        if mean < best_mean:
            best = ident
            best_mean = mean
    return best


def _keep_predicates(result: dict[str, Any]) -> bool:
    contract = _contract()
    if result["selected_ffn_decode_path"] == "separate":
        return False
    if result["reverted"] is True or result["keep_sitting_skipped"] is True:
        return False
    if _select_component_winner(result["ab"]) == "separate":
        return False
    if result["p"] is None or result["d128"] is None or result["d2048"] is None:
        return False
    if (
        float(result["p"]["quartz"]["mean_tok_s"])
        < 0.95 * contract["opt048_quartz_p_mean_tok_s"]
    ):
        return False
    if (
        float(result["d128"]["quartz"]["mean_tok_s"])
        < 0.95 * contract["opt048_quartz_d128_mean_tok_s"]
    ):
        return False
    if (
        float(result["d2048"]["quartz"]["mean_tok_s"])
        <= contract["opt048_quartz_d2048_mean_tok_s"]
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
    assert result["schema_version"] == 1 and result["task"] == "OPT-049"
    assert result["status"] in ("measured", "rejected")
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    assert result["opt046_selected_q4_decode_path"] == "packed"
    assert result["opt047_selected_q8_decode_path"] == "dp4a_q8_1"
    assert result["opt048_selected_q6_decode_path"] == "integer_q8_1"
    assert result["selected_ffn_decode_path"] in LEGAL_PATHS
    assert contract["ab"]["candidates"] == list(LEGAL_CANDIDATES)
    assert contract["ab"]["ties_retain"] == "separate"
    assert _select_component_winner(result["ab"]) == result["ab"]["winner"]
    for ident in LEGAL_CANDIDATES:
        _ab_candidate(result["ab"], ident)
    assert packed_q4_retained()
    assert fusion_design_present()
    assert prior_pins_untouched()
    assert pin_from_source() == result["selected_ffn_decode_path"]
    keep = _keep_predicates(result)
    if result["status"] == "measured":
        assert keep is True
        assert result["reverted"] is False
        assert REPORT.is_file()
        assert "OPT-044" in REPORT.read_text()
        assert "SwiGLU" in REPORT.read_text()
    else:
        assert keep is False
        assert result["selected_ffn_decode_path"] == "separate"
        assert result["reverted"] is True
        assert REJECTION.is_file()
        assert "separate" in REJECTION.read_text()
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


def _shape_slot(mean: float, *, probe: bool = False) -> dict[str, Any]:
    env = {
        "max_abs": 0.0 if not probe else 1e-6,
        "rms": 0.0,
        "one_minus_cosine": 0.0,
        "nonfinite": 0,
    }
    return {
        "role": "probe" if probe else "layer",
        "staging_bytes": 36 * (256 if probe else 160),
        "graph_eager_equal": True,
        "complete": _timed(mean),
        "vs_separate": env,
        "vs_residual": env,
    }


def _ab_stub(
    winner: str, means: dict[str, float], eligible: bool = True
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for ident in LEGAL_CANDIDATES:
        mean = means.get(ident, 1.0)
        candidates[ident] = {
            "id": ident,
            "path": ident,
            "paired": ident.startswith("paired"),
            "staged": ident in ("shared_stage", "paired_staged"),
            "occupancy": 8,
            "registers": 40,
            "local_bytes": 0,
            "eligible": eligible,
            "weighted_complete_ms": mean,
            "shapes": {
                "ffn_probe": _shape_slot(mean, probe=True),
                "ffn_layer0": _shape_slot(mean),
                "ffn_layer3": _shape_slot(mean),
                "ffn_layer31": _shape_slot(mean),
            },
        }
    return {"winner": winner, "win": winner != "separate", "candidates": candidates}


def _reject_fixture() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-049",
        "status": "rejected",
        "measurement_utc": "2026-09-10T00:00:00Z",
        "device": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": True,
        "selected_ffn_decode_path": "separate",
        "opt046_selected_q4_decode_path": "packed",
        "opt047_selected_q8_decode_path": "dp4a_q8_1",
        "opt048_selected_q6_decode_path": "integer_q8_1",
        "ab": _ab_stub("separate", {ident: 1.0 for ident in LEGAL_CANDIDATES}),
        "production_numerics": {
            "formula": "max(strict_reference_ceiling, 1.05 * measured_llama_error + 1e-6)",
            "budget": {"max_abs": 0.0003, "rms": 0.0002, "one_minus_cosine": 1.0e-6},
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
        "report_path": "evidence/optimization/opt049-decode-ffn-fusion/REPORT.md",
    }


def test_opt049_contract_and_source_pins() -> None:
    contract = _contract()
    opt048 = json.loads(OPT048_FIXTURE.read_text())
    opt046 = json.loads(OPT046_FIXTURE.read_text())
    opt047 = json.loads(OPT047_FIXTURE.read_text())
    numerics = json.loads(NUMERICS.read_text())
    assert opt048["p"]["quartz"]["mean_tok_s"] == OPT048_P
    assert opt048["d2048"]["quartz"]["mean_tok_s"] == OPT048_D2048
    assert opt046["selected_q4_decode_path"] == "packed"
    assert opt047["selected_q8_decode_path"] == "dp4a_q8_1"
    assert contract["opt048_quartz_p_mean_tok_s"] == OPT048_P
    assert contract["opt048_quartz_d2048_mean_tok_s"] == OPT048_D2048
    assert contract["ffn_gate_rows"] == 17408
    assert contract["ab"]["candidates"] == list(LEGAL_CANDIDATES)
    assert pin_from_source() in LEGAL_PATHS
    assert packed_q4_retained()
    assert fusion_design_present()
    assert prior_pins_untouched()
    assert numerics["budget_rule"]["do_not_adjust_after_candidate_failure"] is True
    makefile = (ROOT / "Makefile").read_text()
    assert "--fmad=false" in makefile
    assert "opt049_decode_ffn_fusion_ab_test" not in makefile
    assert "cuda/ffn_decode_path.cuh" in makefile
    ab = (ROOT / "cuda/opt049_decode_ffn_fusion_ab_test.cu").read_text()
    assert "QW38_CUDA_TEST_TIER must be set" in ab
    assert (
        "two-pointer" in (ROOT / "tasks/OPT-049.md").read_text() or "two-pointer" in ab
    )
    assert "launch_q4k_gate_up_swiglu" in ab
    dossier = (ROOT / "tasks/OPT-049.md").read_text()
    assert "QW38_CUDA_TEST_TIER" in dossier or "Acceptance" in dossier
    assert OPT045_FIXTURE.is_file()


def test_opt049_validator_rejects_inadmissible_evidence() -> None:
    fixture = _reject_fixture()
    mutations: list[dict[str, Any]] = []
    for mutate in (
        lambda x: x.__setitem__("substitutes_for_opt016", True),
        lambda x: x.__setitem__("owns_opt016_parity_gate", True),
        lambda x: x.__setitem__("status", "source_inspected"),
        lambda x: x.__setitem__("task", "OPT-048"),
        lambda x: x["ab"].__setitem__("winner", "paired"),
        lambda x: x.__setitem__("selected_ffn_decode_path", "paired"),
        lambda x: x.__setitem__("gguf_sha256", "0" * 64),
        lambda x: x.__setitem__("reverted", False),
        lambda x: x.__setitem__("opt046_selected_q4_decode_path", "integer_q8"),
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
        REPORT.write_text(PROOF + "\nSwiGLU\nOPT-044\n")
    if raw_prev is None:
        AB_RAW.write_text("placeholder\n")
    if rejection_prev is None:
        REJECTION.write_text("rejected placeholder separate\n")
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


def test_opt049_fixture_connected() -> None:
    if not FIXTURE.is_file():
        pytest.skip("OPT-049 fixture is written by the exclusive CUDA sitting")
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
    source = ROOT / "cuda/ffn_decode_path.cuh"
    text = source.read_text()
    text = re.sub(
        r'kSelectedFfnDecodePath\[\] = "[^"]+"',
        f'kSelectedFfnDecodePath[] = "{path}"',
        text,
    )
    source.write_text(text)
    contract = _contract()
    contract["selected_ffn_decode_path"] = path
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
    separate_ms = ab["candidates"]["separate"]["weighted_complete_ms"]
    winner = ab["winner"]
    winner_ms = ab["candidates"][winner]["weighted_complete_ms"]
    text = f"""# OPT-049 — Share decode FFN staging and fuse gate/up

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of decode FFN shared Q8 staging and a
two-pointer paired Q4_K gate/up kernel with the admitted SwiGLU
(`gate / (1 + expf(-gate)) * up`) and a single BF16 rounding before the
down projection. Separate-leg packed Q4_K remains the control. Down,
residual, and RMSNorm stay on already accepted kernels. Complete cost
includes norm, stage, gate/up/SwiGLU, down, and residual-add-norm.
Keep requires **OPT-044 production-numerics budgets**, **graph/eager
equality**, **lower complete FFN time**, **95% P/D128 floors versus the
OPT-048 keep**, and a **strict D2048 improvement** versus OPT-048
**{OPT048_D2048} tok/s**. Decode p95 stays inside 105% of OPT-045. Does
not substitute for the 2K llama.cpp parity gate. Nsight is not used.

Proof limit: {PROOF}

## Decision

**{decision}**. Production pin `{fixture["selected_ffn_decode_path"]}`.
A/B winner `{winner}` ({winner_ms} ms vs separate {separate_ms} ms).
Keep sitting {sitting}. Quartz P {quartz_p}, D128 {quartz_d128}, D2048
{quartz_d2048} tok/s versus OPT-048 keep P {OPT048_P}, D128 {OPT048_D128},
D2048 {OPT048_D2048}. OPT-046 packed Q4, OPT-047 DP4A Q8, and OPT-048
integer Q6 remain.

## Quality

Shared-stage and paired-staged paths must match the separate-leg control
on real layer activations. Paired-only BF16 activation loads are compared
under OPT-044 budgets. Graph replay equals eager within the selected path.

## Throughput

Baseline is the OPT-048 keep sitting. On reject, speedup is 0 and
production stays separate.
"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def _write_rejection(fixture: dict[str, Any]) -> None:
    winner = fixture["ab"]["winner"]
    text = f"""# OPT-049 rejection

Production decode FFN stays **separate**. A/B winner `{winner}` was not
installed. Keep sitting skipped={fixture["keep_sitting_skipped"]};
reverted={fixture["reverted"]}. Packed Q4_K gate/up/down and
`launch_swiglu_bf16` remain the production path.
"""
    REJECTION.write_text(text)


def _run_ab(tier: str) -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    make_targets = [
        "build/quant_mmv.cuda.o",
        "build/q4k_decode_dots.cuda.o",
        "build/q8_decode_dots.cuda.o",
        "build/q6k_decode_dots.cuda.o",
        "build/scheduler_primitives.cuda.o",
        "build/full_scheduler.trace.cuda.o",
        "build/gdn_step.cuda.o",
        "build/attention_decode.cuda.o",
        "build/quant.o",
        "build/status.o",
        "build/utf8proc.o",
        "diagnostic",
    ]
    commands = [
        [*_common(IMAGE, tier), "make", *make_targets],
        _nvcc(
            "cuda/opt049_decode_ffn_fusion_ab_test.cu",
            "build/qw38-cuda-opt049-decode-ffn-fusion-ab-test",
            extra=NVCC_OBJECTS,
        ),
        [
            *_common(IMAGE, tier),
            "./build/qw38-cuda-opt049-decode-ffn-fusion-ab-test",
            "evidence/optimization/opt049-decode-ffn-fusion/decode-ffn-fusion-ab-raw.txt",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ],
    ]
    outputs: list[str] = []
    for command in commands:
        completed = _run(command)
        outputs.append(completed.stdout + completed.stderr)
    combined = outputs[-1]
    record = _parse_prefixed(combined, AB_PREFIX)
    assert "status=passed" in combined
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
    block = ab["ab"] if "ab" in ab else ab
    selected = _select_component_winner(block)
    block = {
        "winner": selected,
        "win": selected != "separate",
        "candidates": block["candidates"],
    }
    return {
        "schema_version": 1,
        "task": "OPT-049",
        "status": "rejected",
        "measurement_utc": ab["measurement_utc"],
        "device": ab["device"],
        "compute_capability": ab["compute_capability"],
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": True,
        "selected_ffn_decode_path": "separate",
        "opt046_selected_q4_decode_path": "packed",
        "opt047_selected_q8_decode_path": "dp4a_q8_1",
        "opt048_selected_q6_decode_path": "integer_q8_1",
        "ab": {
            "winner": block["winner"],
            "win": block["win"],
            "candidates": block["candidates"],
        },
        "production_numerics": {
            "formula": "max(strict_reference_ceiling, 1.05 * measured_llama_error + 1e-6)",
            "budget": {"max_abs": 0.0003, "rms": 0.0002, "one_minus_cosine": 1.0e-6},
        },
        "keep_sitting_skipped": selected == "separate",
        "p": None,
        "d128": None,
        "d2048": None,
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt049-decode-ffn-fusion/REPORT.md",
        "_install": selected,
    }


@pytest.mark.skipif(
    os.environ.get("QW38_RUN_CUDA_TESTS") != "1",
    reason="set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate",
)
def test_opt049_exclusive_cuda_sitting() -> None:
    tier = cuda_test_tier()
    if not MODEL.exists():
        pytest.skip("the pinned GGUF is required")
    if FIXTURE.is_file():
        existing = json.loads(FIXTURE.read_text())
        if existing.get("status") in ("measured", "rejected"):
            validate_result(existing)
            return

    ab = _run_ab(tier)
    closed = subprocess.run(
        [*_common(IMAGE), "./build/qw38-cuda-opt049-decode-ffn-fusion-ab-test"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert closed.returncode != 0
    assert "QW38_CUDA_TEST_TIER must be set" in (closed.stdout + closed.stderr)
    if tier != "acceptance":
        assert ab["task"] == "OPT-049"
        winner = ab["ab"]["winner"] if "ab" in ab else ab["winner"]
        assert winner in LEGAL_CANDIDATES
        return

    wrapper = _fixture_from_ab(ab)
    selected = wrapper.pop("_install")
    if selected == "separate":
        _set_pin("separate")
        wrapper["selected_ffn_decode_path"] = "separate"
        _write_report(wrapper)
        _write_rejection(wrapper)
        FIXTURE.write_text(json.dumps(wrapper, indent=2) + "\n")
        validate_result(wrapper)
        return

    cand = (
        ab["ab"]["candidates"][selected] if "ab" in ab else ab["candidates"][selected]
    )
    _set_pin(cand["path"])
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
    fixture["selected_ffn_decode_path"] = cand["path"]
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
        fixture["selected_ffn_decode_path"] = "separate"
        _set_pin("separate")
        _write_report(fixture)
        _write_rejection(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    validate_result(fixture)
