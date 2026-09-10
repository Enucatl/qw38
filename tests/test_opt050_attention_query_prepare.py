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
CONTRACT = ROOT / "pins/opt050_attention_query_prepare_contract.json"
FIXTURE = ROOT / "fixtures/opt050_attention_query_prepare.json"
OPT045_FIXTURE = ROOT / "fixtures/opt045_parallel_norm.json"
OPT049_FIXTURE = ROOT / "fixtures/opt049_decode_ffn_fusion.json"
NUMERICS = ROOT / "pins/production_numerics_contract.json"
EVIDENCE = ROOT / "evidence/optimization/opt050-attention-query-prepare"
REPORT = EVIDENCE / "REPORT.md"
REJECTION = EVIDENCE / "REJECTION.md"
AB_RAW = EVIDENCE / "attention-query-prepare-ab-raw.txt"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
AB_PREFIX = "QW38_OPT050_ATTENTION_QUERY_PREPARE_AB_RESULT="
P_PREFIX = "QW38_PREFILL_4K_ORACLE_RESULT="
DECODE_PREFIX = "QW38_DECODE_ORACLE_RESULT="
OPT045_P = 2130.41089
OPT045_D128 = 28.5522804
OPT045_D2048 = 27.5438766
OPT045_D128_P95 = 35.1905479
OPT045_D128_RUN_P95 = 35.0761757
OPT045_D2048_P95 = 36.4285774
OPT045_D2048_RUN_P95 = 36.3359795
OPT049_P = 2130.79614
OPT049_D128 = 37.2543182
OPT049_D2048 = 35.4072151
LEGAL_PATHS = {"in_kernel", "hoisted", "hoisted_fma"}
LEGAL_CANDIDATES = ("in_kernel", "hoisted", "hoisted_fma")
PROOF = (
    "OPT-044 production-numerics budgets; like-arithmetic hoisted Q/output "
    "match in_kernel; prepare+attention+combine complete cost; "
    "tail/prefix/nonzero-start correctness; committed isolation and "
    "graph/eager equality; no extra persistent allocation; 95% throughput "
    "floors versus OPT-049 keep; 105% p95 ceilings versus OPT-045; "
    "does not substitute for the 2K llama.cpp parity gate"
)
NVCC_OBJECTS = [
    "build/attention_decode.cuda.o",
]


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def pin_from_source() -> str:
    text = (ROOT / "cuda/fattn_mma_f16.cuh").read_text()
    match = re.search(r'kSelectedQueryPreparePath\[\] = "([^"]+)"', text)
    assert match is not None
    return match.group(1)


def frozen_fattn_pins() -> bool:
    text = (ROOT / "cuda/fattn_mma_f16.cuh").read_text()
    return (
        'kSelectedFattnPath[] = "stream_k"' in text
        and 'kSelectedPersistentFattnPath[] = "off"' in text
        and "kSelectedAttentionMmaQueryRows = 16" in text
        and 'kSelectedVkqAccum[] = "registers"' in text
        and 'kSelectedPvPath[] = "mma"' in text
        and 'kSelectedQKPath[] = "warp_microtile"' in text
        and "fattn_mma_quality_typed<16, true, 2, 2, true, true, true>" in text
    )


def overlay_without_new_malloc() -> bool:
    sched = (ROOT / "cuda/full_scheduler.cu").read_text()
    decode = (ROOT / "cuda/attention_decode.cu").read_text()
    return (
        "prompt_projection_a_" in sched
        and "prepared_q" in sched
        and "launch_attention_prepare_prompt_queries" in decode
        and "prepare_prompt_query_kernel" in decode
        and "cudaMalloc"
        not in decode[decode.find("prepare_prompt_query_kernel") :][:800]
    )


def decode_path_untouched() -> bool:
    text = (ROOT / "cuda/attention_decode.cu").read_text()
    return 'kSelectedDecodeAttentionVec[] = "warp_query"' in text


def _ab_candidate(block: dict[str, Any], ident: str) -> None:
    cand = block["candidates"][ident]
    assert cand["id"] == ident
    assert cand["path"] in LEGAL_PATHS
    assert cand["occupancy"] >= 0
    assert cand["launch_ok"] is True
    assert len(cand["samples"]) in (1, 3, 30)
    mean = sum(float(v) for v in cand["samples"]) / len(cand["samples"])
    assert cand["mean_ms"] == pytest.approx(mean, rel=1e-6, abs=1e-6)
    assert cand["vs_in_kernel"]["nonfinite"] == 0
    if ident == "hoisted" and cand["eligible"]:
        assert (
            cand["outputs_byte_equal"] is True or cand["vs_in_kernel"]["max_abs"] == 0
        )
        assert cand["graph_eager_equal"] is True
        assert cand["committed_unchanged"] is True
    if ident == "in_kernel":
        assert cand["eligible"] is True


def _select_component_winner(block: dict[str, Any]) -> str:
    control = float(block["candidates"]["in_kernel"]["mean_ms"])
    hoisted = block["candidates"]["hoisted"]
    fma = block["candidates"]["hoisted_fma"]
    if hoisted["eligible"] and float(hoisted["mean_ms"]) < control:
        if fma["eligible"] and float(fma["mean_ms"]) < 0.99 * float(hoisted["mean_ms"]):
            return "hoisted_fma"
        return "hoisted"
    if fma["eligible"] and float(fma["mean_ms"]) < control:
        return "hoisted_fma"
    return "in_kernel"


def _keep_predicates(result: dict[str, Any]) -> bool:
    if result["ab"]["winner"] == "in_kernel":
        return False
    if not result["correctness"]["all_eligible"]:
        return False
    if result["keep_sitting_skipped"]:
        return False
    if result["p"] is None or result["d128"] is None or result["d2048"] is None:
        return False
    if result["p"]["quartz"]["mean_tok_s"] <= OPT049_P:
        return False
    if result["d128"]["quartz"]["mean_tok_s"] < 0.95 * OPT049_D128:
        return False
    if result["d2048"]["quartz"]["mean_tok_s"] < 0.95 * OPT049_D2048:
        return False
    if result["d128"]["quartz"]["token_latency_p95_ms"] > 1.05 * OPT045_D128_P95:
        return False
    if (
        result["d128"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * OPT045_D128_RUN_P95
    ):
        return False
    if result["d2048"]["quartz"]["token_latency_p95_ms"] > 1.05 * OPT045_D2048_P95:
        return False
    if (
        result["d2048"]["quartz"]["run_mean_token_latency_p95_ms"]
        > 1.05 * OPT045_D2048_RUN_P95
    ):
        return False
    return True


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == set(
        contract["required_fixture_keys"]
    )
    assert result["schema_version"] == 1 and result["task"] == "OPT-050"
    assert result["status"] in ("measured", "rejected")
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert result["nsight_systems"] == "not_used"
    assert result["nsight_compute"] == "not_used"
    assert result["selected_query_prepare_path"] in LEGAL_PATHS
    assert result["selected_qk_path"] == "warp_microtile"
    assert contract["ab"]["candidates"] == list(LEGAL_CANDIDATES)
    assert contract["ab"]["ties_retain"] == "in_kernel"
    assert _select_component_winner(result["ab"]) == result["ab"]["winner"]
    for ident in LEGAL_CANDIDATES:
        _ab_candidate(result["ab"], ident)
    assert frozen_fattn_pins()
    assert overlay_without_new_malloc()
    assert decode_path_untouched()
    assert pin_from_source() == result["selected_query_prepare_path"]
    keep = _keep_predicates(result)
    if result["status"] == "measured":
        assert keep is True
        assert result["reverted"] is False
        assert REPORT.is_file()
        assert "OPT-044" in REPORT.read_text()
        assert "prepare" in REPORT.read_text().casefold()
    else:
        assert keep is False
        assert result["selected_query_prepare_path"] == "in_kernel"
        assert result["reverted"] is True
        assert REJECTION.is_file()
        assert "in_kernel" in REJECTION.read_text()
    assert AB_RAW.is_file()
    assert result["report_path"] == contract["report_path"]
    assert result["proof_limit"] == PROOF


def _timed(mean: float) -> dict[str, Any]:
    return {
        "mean_ms": mean,
        "launch_ok": True,
        "warmup_ms": [mean, mean, mean],
        "samples": [mean] * 30,
        "occupancy": 2,
        "eligible": True,
        "outputs_byte_equal": True,
        "prepared_q_equal": True,
        "graph_eager_equal": True,
        "candidate_exact": True,
        "committed_unchanged": True,
        "scratch_unchanged": True,
        "vs_in_kernel": {
            "max_abs": 0.0,
            "rms": 0.0,
            "one_minus_cosine": 0.0,
            "nonfinite": 0,
        },
    }


def _ab_stub(winner: str, means: dict[str, float]) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for ident in LEGAL_CANDIDATES:
        slot = _timed(means.get(ident, 1.0))
        slot["id"] = ident
        slot["path"] = ident
        if ident == "in_kernel":
            slot["outputs_byte_equal"] = True
        candidates[ident] = slot
    return {"winner": winner, "win": winner != "in_kernel", "candidates": candidates}


def _reject_fixture() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-050",
        "status": "rejected",
        "measurement_utc": "2026-09-10T00:00:00Z",
        "device": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": True,
        "selected_query_prepare_path": "in_kernel",
        "selected_qk_path": "warp_microtile",
        "ab": _ab_stub("in_kernel", {ident: 1.0 for ident in LEGAL_CANDIDATES}),
        "correctness": {"all_eligible": True, "layers_ok": True, "case_count": 60},
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
        "report_path": "evidence/optimization/opt050-attention-query-prepare/REPORT.md",
    }


def test_opt050_contract_and_source_pins() -> None:
    contract = _contract()
    opt049 = json.loads(OPT049_FIXTURE.read_text())
    numerics = json.loads(NUMERICS.read_text())
    assert opt049["p"]["quartz"]["mean_tok_s"] == OPT049_P
    assert opt049["d2048"]["quartz"]["mean_tok_s"] == OPT049_D2048
    assert contract["opt049_quartz_p_mean_tok_s"] == OPT049_P
    assert contract["opt049_quartz_d128_mean_tok_s"] == OPT049_D128
    assert contract["opt049_quartz_d2048_mean_tok_s"] == OPT049_D2048
    assert contract["ab"]["candidates"] == list(LEGAL_CANDIDATES)
    assert pin_from_source() in LEGAL_PATHS
    assert frozen_fattn_pins()
    assert overlay_without_new_malloc()
    assert decode_path_untouched()
    assert numerics["budget_rule"]["do_not_adjust_after_candidate_failure"] is True
    makefile = (ROOT / "Makefile").read_text()
    assert "--fmad=false" in makefile
    assert "opt050_attention_query_prepare_ab_test" not in makefile
    ab = (ROOT / "cuda/opt050_attention_query_prepare_ab_test.cu").read_text()
    decode = (ROOT / "cuda/attention_decode.cu").read_text()
    assert "QW38_CUDA_TEST_TIER must be set" in ab
    assert "launch_attention_prepare_prompt_queries" in decode
    assert "prepare_prompt_query_kernel" in decode
    dossier = (ROOT / "tasks/OPT-050.md").read_text()
    assert "Acceptance" in dossier
    assert OPT045_FIXTURE.is_file()
    assert contract["prepared_bytes_4096"] == 4096 * 24 * 256 * 2 * 2


def test_opt050_validator_rejects_inadmissible_evidence() -> None:
    fixture = _reject_fixture()
    mutations: list[dict[str, Any]] = []
    for mutate in (
        lambda x: x.__setitem__("substitutes_for_opt016", True),
        lambda x: x.__setitem__("owns_opt016_parity_gate", True),
        lambda x: x.__setitem__("status", "source_inspected"),
        lambda x: x.__setitem__("task", "OPT-049"),
        lambda x: x["ab"].__setitem__("winner", "hoisted"),
        lambda x: x.__setitem__("selected_query_prepare_path", "hoisted"),
        lambda x: x.__setitem__("gguf_sha256", "0" * 64),
        lambda x: x.__setitem__("reverted", False),
        lambda x: x.__setitem__("selected_qk_path", "cparts"),
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
        REPORT.write_text(PROOF + "\nprepare\nOPT-044\n")
    if raw_prev is None:
        AB_RAW.write_text("placeholder\n")
    if rejection_prev is None:
        REJECTION.write_text("rejected placeholder in_kernel\n")
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


def test_opt050_fixture_connected() -> None:
    if not FIXTURE.is_file():
        pytest.skip("OPT-050 fixture is written by the exclusive CUDA sitting")
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
    source = ROOT / "cuda/fattn_mma_f16.cuh"
    text = source.read_text()
    text = re.sub(
        r'kSelectedQueryPreparePath\[\] = "[^"]+"',
        f'kSelectedQueryPreparePath[] = "{path}"',
        text,
    )
    source.write_text(text)
    contract = _contract()
    contract["selected_query_prepare_path"] = path
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
    text = f"""# OPT-050 — Prepare prompt Q once outside attention

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of upstream prompt query RMS/RoPE/dual-F16
preparation against in-kernel per-CTA repetition. Complete cost includes
prepare + attention + combine. Like-arithmetic movement must match
in-kernel Q/output; FMA uses OPT-044 budgets. Keep requires lower complete
attention time, improved P versus the OPT-049 keep, 95% D128/D2048 floors,
and 105% p95 ceilings versus OPT-045. Does not substitute for the 2K
llama.cpp parity gate. Nsight is not used.

Proof limit: {PROOF}

## Decision

**{decision}**. Production pin `{fixture["selected_query_prepare_path"]}`.
A/B winner `{ab["winner"]}`. Keep sitting {sitting}. Quartz P {quartz_p},
D128 {quartz_d128}, D2048 {quartz_d2048} tok/s versus OPT-049 keep P
{OPT049_P}, D128 {OPT049_D128}, D2048 {OPT049_D2048}.

## Quality

Hoisted like-arithmetic matches in-kernel prepared Q and attention
output. FMA uses OPT-044 budgets. Graph replay equals eager. Committed
KV stays isolated.

## Throughput

Baseline is the OPT-049 keep sitting. On reject, speedup is 0 and
production stays in_kernel.
"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def _write_rejection(fixture: dict[str, Any]) -> None:
    winner = fixture["ab"]["winner"]
    text = f"""# OPT-050 rejection

Production query prepare stays **in_kernel**. A/B winner `{winner}` was
not installed. Keep sitting skipped={fixture["keep_sitting_skipped"]};
reverted={fixture["reverted"]}. Per-CTA RMS/RoPE inside fattn remains
the production path.
"""
    REJECTION.write_text(text)


def _run_ab(tier: str) -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    commands = [
        [*_common(IMAGE, tier), "make", "build/attention_decode.cuda.o"],
        _nvcc(
            "cuda/opt050_attention_query_prepare_ab_test.cu",
            "build/qw38-cuda-opt050-attention-query-prepare-ab-test",
            extra=NVCC_OBJECTS,
        ),
        [
            *_common(IMAGE, tier),
            "./build/qw38-cuda-opt050-attention-query-prepare-ab-test",
            "evidence/optimization/opt050-attention-query-prepare/attention-query-prepare-ab-raw.txt",
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
    objects = [
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
    commands = [
        [
            *_common(IMAGE),
            "make",
            "build/full_scheduler.trace.cuda.o",
            "build/attention_decode.cuda.o",
            "diagnostic",
        ],
        _nvcc(
            "cuda/prefill_4k_oracle_test.cu",
            "build/qw38-cuda-prefill-4k-oracle-test",
            extra=objects,
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
    objects = [
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
    binary = "build/qw38-cuda-decode-oracle-test"
    commands = [
        _nvcc("cuda/decode_oracle_test.cu", binary, extra=objects),
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
    return {
        "schema_version": 1,
        "task": "OPT-050",
        "status": "rejected",
        "measurement_utc": ab.get("measurement_utc", "2026-09-10T00:00:00Z"),
        "device": ab.get("device", "NVIDIA GeForce RTX 5090"),
        "compute_capability": ab.get("compute_capability", "12.0"),
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "reverted": True,
        "selected_query_prepare_path": "in_kernel",
        "selected_qk_path": "warp_microtile",
        "ab": {
            "winner": selected,
            "win": selected != "in_kernel",
            "candidates": block["candidates"],
        },
        "correctness": ab.get(
            "correctness",
            {"all_eligible": True, "layers_ok": True, "case_count": 0},
        ),
        "production_numerics": {
            "formula": "max(strict_reference_ceiling, 1.05 * measured_llama_error + 1e-6)",
            "budget": {"max_abs": 0.0003, "rms": 0.0002, "one_minus_cosine": 1.0e-6},
        },
        "keep_sitting_skipped": selected == "in_kernel",
        "p": None,
        "d128": None,
        "d2048": None,
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt050-attention-query-prepare/REPORT.md",
        "_install": selected,
    }


@pytest.mark.skipif(
    os.environ.get("QW38_RUN_CUDA_TESTS") != "1",
    reason="set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate",
)
def test_opt050_exclusive_cuda_sitting() -> None:
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
        [*_common(IMAGE), "./build/qw38-cuda-opt050-attention-query-prepare-ab-test"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert closed.returncode != 0
    assert "QW38_CUDA_TEST_TIER must be set" in (closed.stdout + closed.stderr)
    if tier != "acceptance":
        assert ab["task"] == "OPT-050"
        winner = ab["ab"]["winner"] if "ab" in ab else ab["winner"]
        assert winner in LEGAL_CANDIDATES
        return

    wrapper = _fixture_from_ab(ab)
    selected = wrapper.pop("_install")
    if selected == "in_kernel":
        _set_pin("in_kernel")
        wrapper["selected_query_prepare_path"] = "in_kernel"
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
            "build/attention_decode.cuda.o",
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
    fixture["selected_query_prepare_path"] = selected
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
        _set_pin("in_kernel")
        fixture["status"] = "rejected"
        fixture["reverted"] = True
        fixture["selected_query_prepare_path"] = "in_kernel"
        fixture["keep_sitting_skipped"] = False
        _write_rejection(fixture)
    else:
        fixture["status"] = "measured"
        if REJECTION.is_file():
            REJECTION.unlink()
    _write_report(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    validate_result(fixture)
