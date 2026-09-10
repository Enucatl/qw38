"""OPT-044 production arithmetic policy, budgets, and quality-suite helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

ABS_HEADROOM = 1.05
ABS_FLOOR = 1e-6
COSINE_FLOOR = 1e-7
STRICT_CUD001_ABS = 3.0e-4
STRICT_CUD001_RMS = 2.0e-4
PRODUCTION_PPL_RATIO = 1.01
LEGACY_PPL_RATIO = 1.05
RECURRENCE_INCREMENTAL_NLL = 0.02
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
DATASET_SHA = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"
TOKENIZER_SHA = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"

GATE_INVENTORY: tuple[dict[str, str], ...] = (
    {
        "gate": "CUD-001",
        "class": "retained-reference-arithmetic",
        "notes": (
            "Original 3e-4/2e-4 envelope stays on the strict MMV reference; "
            "production optimized kernels use this contract."
        ),
    },
    {
        "gate": "CUD-003",
        "class": "retained-reference-arithmetic",
        "notes": (
            "Q8_0 MMV and pointwise numeric envelopes remain strict-reference; "
            "embedding/layout exactness is structural."
        ),
    },
    {
        "gate": "CUD-003-embedding-layout-gdn-map",
        "class": "structural-exactness",
        "notes": (
            "BF16 embedding equality, attention split, and GDN tiled/grouped "
            "maps stay bit-exact."
        ),
    },
    {
        "gate": "GDN-001",
        "class": "retained-reference-arithmetic",
        "notes": "Strict GDN step numeric envelope plus structural prepare/commit isolation.",
    },
    {
        "gate": "GDN-001-commit",
        "class": "structural-exactness",
        "notes": "Candidate/commit isolation and frontier publication remain exact.",
    },
    {
        "gate": "GDN-002",
        "class": "retained-reference-arithmetic",
        "notes": "Chunked GDN numeric envelope vs sequential reference.",
    },
    {
        "gate": "GDN-002-chunk-equivalence",
        "class": "structural-exactness",
        "notes": "Chunk vs repeated-token CUDA equality stays exact.",
    },
    {
        "gate": "ATN-001",
        "class": "retained-reference-arithmetic",
        "notes": "Decode attention numeric envelope vs scalar/BF16-aware references.",
    },
    {
        "gate": "ATN-001-commit",
        "class": "structural-exactness",
        "notes": "Candidate KV bytes and committed-row isolation stay exact.",
    },
    {
        "gate": "ATN-002",
        "class": "retained-reference-arithmetic",
        "notes": "Prefill attention numeric envelope vs tiled reference.",
    },
    {
        "gate": "ATN-002-chunk-equivalence",
        "class": "structural-exactness",
        "notes": "Token-wise vs chunked attention commit equality stays exact.",
    },
    {
        "gate": "OPT-003-graph",
        "class": "structural-exactness",
        "notes": "Graph vs eager equality is same-arithmetic-path only.",
    },
    {
        "gate": "SES-001-prefix",
        "class": "structural-exactness",
        "notes": "Token identity, prefix reuse, and checkpoint round-trips stay exact.",
    },
    {
        "gate": "TRC-004",
        "class": "retained-reference-arithmetic",
        "notes": "CUDA diagnostic taps use frozen scalar tolerances; identities stay exact.",
    },
    {
        "gate": "ORA-004",
        "class": "retained-reference-arithmetic",
        "notes": "Scalar-oracle tolerances remain frozen pre-CUDA reference contracts.",
    },
    {
        "gate": "QLT-001",
        "class": "end-to-end-quality",
        "notes": "Legacy 1.05 PPL suite is retained; production-optimization uses 1.01.",
    },
    {
        "gate": "OPT-044-primitive",
        "class": "production-primitive-quality",
        "notes": "Family/shape ceilings vs FP64 dequant and pinned llama error.",
    },
    {
        "gate": "OPT-044-e2e",
        "class": "end-to-end-quality",
        "notes": (
            "Production-optimization suite: 1.01 PPL, 0.02 recurrence drift, "
            "functional answers."
        ),
    },
)

PRODUCTION_CASES = (
    "wikitext_nll",
    "continuation_2048",
    "continuation_4096",
    "recurrence_short",
    "recurrence_long",
    "task_arithmetic",
    "task_python_len",
    "task_inference",
    "task_minutes",
    "task_sort",
    "task_json",
    "task_reading",
    "task_sequence",
    "held_out_wikitext_1024",
)

PERMITTED_PRODUCTION = (
    "Q8_1 activation quantization",
    "reordered FP32 reductions",
    "explicit FMA / fmaf",
    "F16 MMA operands with FP32 accumulation",
    "individually validated approximate transcendental ops (rsqrtf, __expf)",
)

FORBIDDEN = (
    "global host or JSON fast-math",
    "fast-math on strict reference objects",
    "sparse attention",
    "weight requantization",
    "reduced vocabulary",
    "speculative decoding",
    "unvalidated fast kernel in production",
)


def sha256_json_tokens(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def production_ceiling(
    strict: float,
    llama_error: float | None,
    *,
    nonfinite: int = 0,
) -> dict[str, Any]:
    """Freeze one abs/RMS/1-cosine ceiling. Pathological llama output stays strict."""
    if llama_error is None or nonfinite > 0 or not math.isfinite(float(llama_error)):
        return {
            "ceiling": strict,
            "source": "strict_reference",
            "path": "strict",
            "llama_error": llama_error,
            "nonfinite": nonfinite,
        }
    value = max(strict, ABS_HEADROOM * float(llama_error) + ABS_FLOOR)
    return {
        "ceiling": value,
        "source": "max(strict, 1.05*llama + 1e-6)",
        "path": "optimized_eligible",
        "llama_error": float(llama_error),
        "nonfinite": 0,
    }


def cosine_ceiling(
    strict_one_minus: float,
    llama_one_minus: float | None,
    *,
    nonfinite: int = 0,
) -> dict[str, Any]:
    floor = max(strict_one_minus, COSINE_FLOOR)
    budget = production_ceiling(floor, llama_one_minus, nonfinite=nonfinite)
    budget["ceiling"] = max(float(budget["ceiling"]), COSINE_FLOOR)
    return budget


def zero_vector_policy(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule": "explicit_norm_and_abs",
        "abs_ok": float(metrics.get("max_abs", 1.0)) == 0.0,
        "candidate_norm": metrics.get("candidate_norm"),
        "reference_norm": metrics.get("reference_norm"),
        "cosine_not_used": True,
    }


def parse_host_payload(text: str) -> dict[str, Any]:
    prefix = "QW38_OPT044_PRODUCTION_NUMERICS_RESULT="
    for line in text.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    raise ValueError("missing OPT-044 host diagnostic payload")


def family_budgets(host: dict[str, Any]) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for case in host["cases"]:
        ident = str(case["id"])
        comps = case["comparisons"]
        llama_orig = comps["llama_q8_1_vs_fp64_original_bf16"]
        llama_staged = comps["llama_q8_1_vs_fp64_llama_staged"]
        host_staged = comps["fp32_host_vs_fp64_quartz_staged"]
        nonfinite = int(llama_orig["nonfinite"]) + int(case["llama_q8_1_nonfinite"])
        vs_original = {
            "abs": production_ceiling(
                STRICT_CUD001_ABS, llama_orig["max_abs"], nonfinite=nonfinite
            ),
            "rms": production_ceiling(
                STRICT_CUD001_RMS, llama_orig["rms"], nonfinite=nonfinite
            ),
            "one_minus_cosine": cosine_ceiling(
                COSINE_FLOOR, llama_orig["one_minus_cosine"], nonfinite=nonfinite
            ),
        }
        vs_staged = {
            "abs": production_ceiling(
                STRICT_CUD001_ABS, llama_staged["max_abs"], nonfinite=nonfinite
            ),
            "rms": production_ceiling(
                STRICT_CUD001_RMS, llama_staged["rms"], nonfinite=nonfinite
            ),
            "one_minus_cosine": cosine_ceiling(
                COSINE_FLOOR, llama_staged["one_minus_cosine"], nonfinite=nonfinite
            ),
        }
        if case["activation_source"] == "zero":
            vs_original["zero_vector"] = zero_vector_policy(llama_orig)
            vs_staged["zero_vector"] = zero_vector_policy(llama_staged)
            vs_original["one_minus_cosine"]["path"] = "strict"
            vs_staged["one_minus_cosine"]["path"] = "strict"
        existing_staged_miss = (
            float(host_staged["max_abs"]) > STRICT_CUD001_ABS
            or float(host_staged["rms"]) > STRICT_CUD001_RMS
        )
        families[ident] = {
            "family": case["family"],
            "rows": case["rows"],
            "columns": case["columns"],
            "activation_source": case["activation_source"],
            "checksums": case["checksums"],
            "vs_fp64_original_bf16": vs_original,
            "vs_fp64_staged_activation": vs_staged,
            "existing_strict_staged_miss": existing_staged_miss,
            "production_dispatch": "strict",
            "nonfinite": nonfinite,
            "pathological_llama": nonfinite > 0,
        }
    return families


def ppl_ratio(quartz_nll: float, llama_nll: float) -> float:
    return math.exp(float(quartz_nll) - float(llama_nll))


def teacher_forced_metrics(
    quartz: dict[str, Any], llama: dict[str, Any]
) -> dict[str, Any]:
    q_steps = quartz.get("steps", [])
    l_steps = llama.get("steps", [])
    count = min(len(q_steps), len(l_steps))
    if count == 0:
        return {"positions": 0, "top1_agreement": None}
    agree = 0
    abs_err = 0.0
    sq = 0.0
    first_fail = -1
    for index in range(count):
        qg = int(q_steps[index]["greedy_token"])
        lg = int(l_steps[index]["greedy_token"])
        if qg == lg:
            agree += 1
        q_logit = float(q_steps[index]["greedy_logit"])
        l_logit = float(l_steps[index]["greedy_logit"])
        err = abs(q_logit - l_logit)
        abs_err = max(abs_err, err)
        sq += err * err
        if first_fail < 0 and qg != lg:
            first_fail = index
    return {
        "positions": count,
        "top1_agreement": agree / count,
        "greedy_logit_max_abs": abs_err,
        "greedy_logit_rms": math.sqrt(sq / count),
        "first_disagreement": first_fail,
        "kl_full_logits": "not_captured_full_vocab_at_qlt001_positions",
        "full_logit_cosine": "not_captured_full_vocab_at_qlt001_positions",
    }


def validate_contract(contract: dict[str, Any], fixture: dict[str, Any]) -> None:
    if contract.get("schema") != "qw38.production-numerics-contract":
        raise ValueError("unexpected production numerics schema")
    if contract.get("claims_performance_improvement") is not False:
        raise ValueError("speedup claim is forbidden")
    if contract["path_selection"]["selected"] != "strict":
        raise ValueError("unvalidated optimized production path")
    if contract["path_selection"]["optimized_admitted"] is not False:
        raise ValueError("optimized path must not be admitted yet")
    if contract["quality_suite"]["nll_ppl_ratio"] != PRODUCTION_PPL_RATIO:
        raise ValueError("production PPL ratio must stay 1.01")
    if contract["quality_suite"]["legacy_nll_ppl_ratio"] != LEGACY_PPL_RATIO:
        raise ValueError("legacy QLT-001 ratio must remain 1.05")
    expected = family_budgets(fixture["host_fp64"])
    for ident, family in expected.items():
        got = contract["families"][ident]
        for side in ("vs_fp64_original_bf16", "vs_fp64_staged_activation"):
            for metric in ("abs", "rms", "one_minus_cosine"):
                left = float(got[side][metric]["ceiling"])
                right = float(family[side][metric]["ceiling"])
                if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-18):
                    raise ValueError(f"budget mutated for {ident} {side} {metric}")
        if got["production_dispatch"] != "strict":
            raise ValueError(f"{ident} production dispatch is not strict")
        if family["nonfinite"] and got.get("pathological_llama") is not True:
            raise ValueError(f"{ident} dropped a nonfinite llama case")
        if (
            family["nonfinite"]
            and got["vs_fp64_original_bf16"]["abs"]["path"] != "strict"
        ):
            raise ValueError(f"{ident} inflated a nonfinite llama case")
    inventory = {row["gate"] for row in contract["gate_inventory"]}
    required = {row["gate"] for row in GATE_INVENTORY}
    if inventory != required:
        raise ValueError("gate inventory mismatch")
    if fixture["opt042_numeric_reject_preserved"] is not True:
        raise ValueError("OPT-042 numeric_reject evidence must stay unchanged")
    held = contract["quality_suite"]["held_out_wikitext_1024"]
    cal = contract["quality_suite"]["calibration_wikitext_1024"]
    if held["continuation_sha256"] == cal["continuation_sha256"]:
        raise ValueError("held-out token hash equals calibration")
    if int(held["stream_start"]) < int(cal["stream_end"]) and int(
        held["stream_end"]
    ) > int(cal["stream_start"]):
        raise ValueError("held-out span overlaps calibration")
