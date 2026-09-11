"""OPT-059 v2 production arithmetic policy: GPU llama error and admission."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any

from tools.production_numerics import (
    ABS_FLOOR,
    COSINE_FLOOR,
    FORBIDDEN,
    GATE_INVENTORY,
    GGUF_SHA,
    LLAMA_REV,
    PERMITTED_PRODUCTION,
    PRODUCTION_PPL_RATIO,
    RECURRENCE_INCREMENTAL_NLL,
    ROOT,
    STRICT_CUD001_ABS,
    STRICT_CUD001_RMS,
    production_ceiling,
    zero_vector_policy,
)

V2_SCHEMA = "qw38.production-numerics-v2-contract"
V2_ABS_HEADROOM = 1.25
V2_COSINE_HEADROOM = 1.25
INT32_OVERFLOW_MAX = 2_147_483_647
Q4K_GROUP_INT_PRODUCT_MAX = 15 * 127 * 32  # 60_960; dp4a int32 is safe
FP16_INTEGER_EXACT_MAX = 2048
STAGING_QUARTZ_SUM_Q = "quartz_q8_1_sum_q"
STAGING_LLAMA_SUM_X = "llama_q8_1_sum_x"
STAGING_Q8_FP32 = "q8_fp32"

CALIBRATION_LAYERS = (0, 3, 31, 32)
HELD_OUT_LAYERS = (62, 63)
CALIBRATION_TOKEN_POSITIONS = (128, 512)
HELD_OUT_TOKEN_POSITIONS = (2048, 4095)
MAX_ROWS = 16
MAX_ACTIVATIONS = 4
MAX_PRODUCTION_FP64_DOTS = 64

FAMILY_SHAPES: tuple[dict[str, Any], ...] = (
    {
        "id": "q4_gate_up_k5120",
        "family": "q4_k",
        "rows": 17408,
        "columns": 5120,
        "role": "ffn_gate_up",
        "activation": "ffn_preprojection",
    },
    {
        "id": "q4_down_k17408",
        "family": "q4_k",
        "rows": 5120,
        "columns": 17408,
        "role": "ffn_down",
        "activation": "swiglu_down_input",
    },
    {
        "id": "q8_mixer_k5120",
        "family": "q8_0",
        "rows": 10240,
        "columns": 5120,
        "role": "mixer_qkv",
        "activation": "mixer_preprojection",
    },
    {
        "id": "q8_mixer_k6144",
        "family": "q8_0",
        "rows": 5120,
        "columns": 6144,
        "role": "mixer_output",
        "activation": "gdn_or_attn_mix_6144",
    },
    {
        "id": "q6_vocab_k5120",
        "family": "q6_k",
        "rows": 248320,
        "columns": 5120,
        "role": "vocab",
        "activation": "final_norm",
    },
)

SMALL_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "q4_k_17x256", "family": "q4_k", "rows": 17, "columns": 256},
    {"id": "q6_k_17x256", "family": "q6_k", "rows": 17, "columns": 256},
    {"id": "q8_0_17x32", "family": "q8_0", "rows": 17, "columns": 32},
)

SMALL_PATTERNS = ("zero", "alternating_cancellation", "finite_scales", "random")
INTEGER_SUM_CASES = (2047, 2049, 4064, -2047, -2049, -4064)


def float_to_half_bits(value: float) -> int:
    if not math.isfinite(value):
        sign = 0x8000 if math.copysign(1.0, value) < 0.0 else 0
        if math.isnan(value):
            return sign | 0x7E00
        return sign | 0x7C00
    packed = float(value)
    raw = struct.unpack(">I", struct.pack(">f", packed))[0]
    sign = (raw >> 16) & 0x8000
    exponent = int((raw >> 23) & 0xFF) - 127
    mantissa = raw & 0x7FFFFF
    if ((raw >> 23) & 0xFF) == 0xFF:
        return sign | 0x7C00 | (0x200 if mantissa else 0)
    if exponent > 15:
        return sign | 0x7C00
    if exponent > -15:
        rounded = mantissa + 0x1000 + ((mantissa >> 13) & 1)
        if rounded & 0x800000:
            exponent += 1
            mantissa = 0
        else:
            mantissa = rounded
        if exponent > 15:
            return sign | 0x7C00
        return sign | ((exponent + 15) << 10) | (mantissa >> 13)
    if exponent < -25:
        return sign
    mantissa |= 0x800000
    shift = -14 - exponent
    rounded = (mantissa + (1 << (shift + 12)) + ((mantissa >> (shift + 13)) & 1)) >> (
        shift + 13
    )
    return sign | rounded


def half_bits_to_float(half: int) -> float:
    sign = (half & 0x8000) << 16
    exponent = (half >> 10) & 0x1F
    fraction = half & 0x03FF
    if exponent == 0:
        if fraction == 0:
            bits = sign
        else:
            shifts = 0
            while (fraction & 0x0400) == 0:
                fraction <<= 1
                shifts += 1
            fraction &= 0x03FF
            bits = sign | ((113 - shifts) << 23) | (fraction << 13)
    elif exponent == 0x1F:
        bits = sign | 0x7F800000 | (fraction << 13)
    else:
        bits = sign | ((exponent + 112) << 23) | (fraction << 13)
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def half_round(value: float) -> float:
    return half_bits_to_float(float_to_half_bits(value))


def integer_lost_in_half(total: int) -> bool:
    return half_round(float(total)) != float(total)


def v2_abs_rms_ceiling(
    strict: float,
    llama_gpu_error: float | None,
    *,
    nonfinite: int = 0,
) -> dict[str, Any]:
    """Freeze max abs/RMS from reference-only llama GPU error."""
    if (
        llama_gpu_error is None
        or nonfinite > 0
        or not math.isfinite(float(llama_gpu_error))
    ):
        return {
            "ceiling": strict,
            "source": "strict_reference",
            "path": "strict",
            "llama_gpu_error": llama_gpu_error,
            "nonfinite": nonfinite,
            "admitted": False,
        }
    value = max(strict, V2_ABS_HEADROOM * float(llama_gpu_error) + ABS_FLOOR)
    return {
        "ceiling": value,
        "source": "max(strict, 1.25*llama_gpu + 1e-6)",
        "path": "optimized_eligible",
        "llama_gpu_error": float(llama_gpu_error),
        "nonfinite": 0,
        "admitted": True,
    }


def v2_cosine_ceiling(
    llama_one_minus: float | None,
    *,
    nonfinite: int = 0,
) -> dict[str, Any]:
    if (
        llama_one_minus is None
        or nonfinite > 0
        or not math.isfinite(float(llama_one_minus))
    ):
        return {
            "ceiling": COSINE_FLOOR,
            "source": "strict_reference",
            "path": "strict",
            "llama_gpu_error": llama_one_minus,
            "nonfinite": nonfinite,
            "admitted": False,
        }
    value = max(
        COSINE_FLOOR,
        V2_COSINE_HEADROOM * float(llama_one_minus) + COSINE_FLOOR,
    )
    return {
        "ceiling": value,
        "source": "max(1e-7, 1.25*(1-cos_llama)+1e-7)",
        "path": "optimized_eligible",
        "llama_gpu_error": float(llama_one_minus),
        "nonfinite": 0,
        "admitted": True,
    }


def v1_diagnostic_ceiling(
    strict: float, llama_error: float | None, *, nonfinite: int = 0
) -> dict[str, Any]:
    return production_ceiling(strict, llama_error, nonfinite=nonfinite)


def staged_rounding_guard(
    *,
    fp32_scale_reductions: int,
    sum_abs_products: float,
) -> dict[str, Any]:
    """Conservative FP32 accumulation bound. Cannot waive a bad total-error."""
    unit = 1.1920928955078125e-07  # fp32 epsilon
    bound = unit * max(1, fp32_scale_reductions) * max(0.0, float(sum_abs_products))
    return {
        "fp32_scale_reductions": fp32_scale_reductions,
        "sum_abs_products": float(sum_abs_products),
        "conservative_abs_bound": bound,
        "int32_overflow_maximum": INT32_OVERFLOW_MAX,
        "q4k_group_int_product_max": Q4K_GROUP_INT_PRODUCT_MAX,
        "waives_total_error": False,
        "counts": "floating scale/group reductions, not integer multiplies",
    }


def q4k_fp32_reductions(columns: int) -> int:
    blocks = columns // 256
    # 8 groups × (scale*dot, min*sum, sub) plus block add into the row acc.
    return blocks * (8 * 3 + 1)


def freeze_case_ids() -> dict[str, Any]:
    calibration = []
    held_out = []
    for shape in FAMILY_SHAPES:
        for layer in CALIBRATION_LAYERS:
            for token in CALIBRATION_TOKEN_POSITIONS:
                calibration.append(
                    {
                        **shape,
                        "id": f"{shape['id']}_L{layer}_t{token}",
                        "split": "calibration",
                        "layer": layer,
                        "token_position": token,
                        "stage": "decode",
                        "max_rows": MAX_ROWS,
                        "max_activations": MAX_ACTIVATIONS,
                    }
                )
        for layer in HELD_OUT_LAYERS:
            for token in HELD_OUT_TOKEN_POSITIONS:
                held_out.append(
                    {
                        **shape,
                        "id": f"{shape['id']}_L{layer}_t{token}",
                        "split": "held_out",
                        "layer": layer,
                        "token_position": token,
                        "stage": "decode",
                        "max_rows": MAX_ROWS,
                        "max_activations": MAX_ACTIVATIONS,
                    }
                )
    probes = []
    for probe in SMALL_PROBES:
        for pattern in SMALL_PATTERNS:
            probes.append(
                {
                    "id": f"{probe['id']}_{pattern}",
                    "split": "small_probe",
                    "pattern": pattern,
                    **probe,
                }
            )
    payload = json.dumps(
        {
            "calibration": [row["id"] for row in calibration],
            "held_out": [row["id"] for row in held_out],
            "small_probes": [row["id"] for row in probes],
        },
        separators=(",", ":"),
    ).encode("ascii")
    return {
        "calibration_layers": list(CALIBRATION_LAYERS),
        "held_out_layers": list(HELD_OUT_LAYERS),
        "calibration_token_positions": list(CALIBRATION_TOKEN_POSITIONS),
        "held_out_token_positions": list(HELD_OUT_TOKEN_POSITIONS),
        "max_rows": MAX_ROWS,
        "max_activations": MAX_ACTIVATIONS,
        "max_production_fp64_dots_per_phase": MAX_PRODUCTION_FP64_DOTS,
        "opt043_preprojection_insufficient": True,
        "required_extra_captures": ["swiglu_down_input", "final_norm"],
        "id_manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "calibration": calibration,
        "held_out": held_out,
        "small_probes": probes,
        "disjoint": True,
    }


def teacher_forced_margin(admitted_max_logit_error: float | None) -> dict[str, Any]:
    if admitted_max_logit_error is None or not math.isfinite(
        float(admitted_max_logit_error)
    ):
        return {
            "rule": "twice_admitted_max_logit_error",
            "admitted_max_logit_error": None,
            "per_position_margin": None,
            "admit_swaps": "top1_runner_up_within_margin_only",
            "status": "unadmitted_until_reference_logits",
        }
    margin = 2.0 * float(admitted_max_logit_error)
    return {
        "rule": "twice_admitted_max_logit_error",
        "admitted_max_logit_error": float(admitted_max_logit_error),
        "per_position_margin": margin,
        "admit_swaps": "top1_runner_up_within_margin_only",
        "status": "frozen",
    }


def admit_top1_swap(
    quartz_top1: int,
    llama_top1: int,
    llama_runner_up: int | None,
    logit_gap: float | None,
    margin: float | None,
) -> bool:
    if quartz_top1 == llama_top1:
        return True
    if margin is None or logit_gap is None or llama_runner_up is None:
        return False
    return quartz_top1 == llama_runner_up and float(logit_gap) <= float(margin)


def staging_semantics() -> dict[str, Any]:
    lost = {str(value): integer_lost_in_half(value) for value in INTEGER_SUM_CASES}
    return {
        "quartz_q8_1_sum_q": {
            "id": STAGING_QUARTZ_SUM_Q,
            "field": "Q8_1Block::q8_sum",
            "stores": "half(sum(integer quants))",
            "kernel": "q4k_decode_dots.cuh::quantize_bf16_q8_1",
            "do_not_reinterpret": True,
        },
        "llama_gpu_q8_1_sum_x": {
            "id": STAGING_LLAMA_SUM_X,
            "field": "block_q8_1.ds.y",
            "stores": "half(sum(original x))",
            "kernel": "pinned llama ggml-cuda/quantize.cu::quantize_q8_1",
            "revision": LLAMA_REV,
        },
        "llama_cpu_q8_1_sum_qd": {
            "id": "llama_cpu_q8_1_sum_q_times_d",
            "stores": "half(sum(q)*d)",
            "kernel": "ggml-quants.c::quantize_row_q8_1_ref",
            "not_gpu_authority": True,
        },
        "opt044_host_replica": {
            "stores": "half(sum(q)*d)",
            "not_gpu_authority": True,
            "role": "v1_diagnostic_only",
        },
        "pinned_q4_mmv": {
            "uses_stored_sum": False,
            "recomputes": "integer sum of q8 values via dp4a 0x01010101",
            "source": "vecdotq.cuh::vec_dot_q4_K_q8_1_impl_vmmq",
        },
        "half_integer_exact_max": FP16_INTEGER_EXACT_MAX,
        "integer_sum_cases": list(INTEGER_SUM_CASES),
        "units_lost_when_stored_as_half": lost,
        "consumer_change": "separately_measured_candidate_not_fixture_repair",
    }


def compare_metrics(
    candidate: list[float],
    reference: list[float],
) -> dict[str, Any]:
    if len(candidate) != len(reference) or not candidate:
        return {
            "max_abs": math.inf,
            "rms": math.inf,
            "normalized_rms": math.inf,
            "cosine": 0.0,
            "one_minus_cosine": 1.0,
            "nonfinite": 1,
            "candidate_norm": 0.0,
            "reference_norm": 0.0,
        }
    max_abs = 0.0
    squared = 0.0
    dot = 0.0
    cand_sq = 0.0
    ref_sq = 0.0
    nonfinite = 0
    for left, right in zip(candidate, reference, strict=True):
        if not math.isfinite(left) or not math.isfinite(right):
            nonfinite += 1
            continue
        err = abs(left - right)
        max_abs = max(max_abs, err)
        squared += err * err
        dot += left * right
        cand_sq += left * left
        ref_sq += right * right
    n = len(candidate)
    cand_norm = math.sqrt(cand_sq)
    ref_norm = math.sqrt(ref_sq)
    rms = math.sqrt(squared / n)
    denom = cand_norm * ref_norm
    if denom == 0.0:
        cosine = 0.0
        one_minus = 1.0
    else:
        cosine = dot / denom
        one_minus = 1.0 - cosine
    return {
        "max_abs": max_abs,
        "rms": rms,
        "normalized_rms": rms / max(ref_norm, 1e-12),
        "cosine": cosine,
        "one_minus_cosine": one_minus,
        "nonfinite": nonfinite,
        "candidate_norm": cand_norm,
        "reference_norm": ref_norm,
    }


def apply_zero_vector(
    metrics: dict[str, Any], budgets: dict[str, Any]
) -> dict[str, Any]:
    policy = zero_vector_policy(metrics)
    budgets["zero_vector"] = policy
    budgets["one_minus_cosine"]["path"] = "strict"
    budgets["one_minus_cosine"]["cosine_not_used"] = True
    return budgets


def family_v2_budgets(
    llama_gpu: dict[str, Any] | None,
    *,
    family: str,
    ident: str,
    activation_source: str,
    nonfinite: int = 0,
) -> dict[str, Any]:
    gpu = llama_gpu or {}
    abs_budget = v2_abs_rms_ceiling(
        STRICT_CUD001_ABS, gpu.get("max_abs"), nonfinite=nonfinite
    )
    rms_budget = v2_abs_rms_ceiling(
        STRICT_CUD001_RMS, gpu.get("rms"), nonfinite=nonfinite
    )
    cos_budget = v2_cosine_ceiling(gpu.get("one_minus_cosine"), nonfinite=nonfinite)
    vs_original = {
        "abs": abs_budget,
        "rms": rms_budget,
        "one_minus_cosine": cos_budget,
    }
    if activation_source == "zero":
        apply_zero_vector(
            {
                "max_abs": gpu.get("max_abs", 0.0) or 0.0,
                "candidate_norm": gpu.get("candidate_norm", 0.0),
                "reference_norm": gpu.get("reference_norm", 0.0),
            },
            vs_original,
        )
    return {
        "id": ident,
        "family": family,
        "activation_source": activation_source,
        "vs_fp64_original_bf16": vs_original,
        "v1_1_05_diagnostic": {
            "abs": v1_diagnostic_ceiling(
                STRICT_CUD001_ABS, gpu.get("max_abs"), nonfinite=nonfinite
            ),
            "rms": v1_diagnostic_ceiling(
                STRICT_CUD001_RMS, gpu.get("rms"), nonfinite=nonfinite
            ),
        },
        "nonfinite": nonfinite,
        "pathological_llama": nonfinite > 0,
        "coverage": "admitted" if abs_budget["admitted"] else "unadmitted",
    }


def lookup_admission(
    manifest: dict[str, Any],
    *,
    family: str,
    columns: int | None = None,
    staging: str | None = None,
    variant: str | None = None,
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for entry in manifest.get("entries", []):
        if entry.get("family") != family:
            continue
        if columns is not None and int(entry.get("columns", -1)) not in {
            int(columns),
            0,
        }:
            if int(entry.get("columns", -1)) != int(columns):
                continue
        if staging is not None and entry.get("staging") != staging:
            continue
        if variant is not None and entry.get("variant") != variant:
            continue
        matches.append(entry)
    if not matches:
        return None
    if columns is not None:
        exact = [row for row in matches if int(row.get("columns", -1)) == int(columns)]
        if exact:
            return exact[0]
    return matches[0]


def unrepresented_fallback() -> dict[str, Any]:
    return {
        "family": "*",
        "columns": 0,
        "staging": "*",
        "variant": "*",
        "production_dispatch": "strict",
        "currently_selected": False,
        "v2_admitted": False,
        "testing_admitted": False,
        "reason": "unrepresented_shape_strict_fallback",
    }


def engine_summary_path(manifest: dict[str, Any]) -> str:
    selected = [
        row
        for row in manifest.get("entries", [])
        if row.get("currently_selected") is True
    ]
    dispatches = {str(row.get("production_dispatch")) for row in selected}
    if not selected or dispatches == {"strict"}:
        return "strict"
    if dispatches == {"optimized"}:
        return "optimized"
    return "mixed"


def any_optimized_selected(manifest: dict[str, Any]) -> bool:
    return any(
        row.get("currently_selected") is True
        and row.get("production_dispatch") == "optimized"
        for row in manifest.get("entries", [])
    )


def opt046_independent_verdict(
    opt046: dict[str, Any],
    v2_ceiling_abs: float,
    *,
    llama_gpu_error: float | None = None,
) -> dict[str, Any]:
    winner = opt046.get("ab", {}).get("candidates", {}).get("integer_q8_w4", {})
    probe = winner.get("shapes", {}).get("q4_k_17x256", {})
    orig = probe.get("vs_fp64_original_bf16", {})
    staged = probe.get("vs_fp64_staged_activation", {})
    orig_abs = float(orig.get("max_abs", math.inf))
    staged_abs = float(staged.get("max_abs", math.inf))
    historical_cud001 = 0.000305175781
    gpu_known = llama_gpu_error is not None and math.isfinite(float(llama_gpu_error))
    vs_v2_orig = gpu_known and orig_abs <= v2_ceiling_abs
    vs_staged_strict = staged_abs <= STRICT_CUD001_ABS
    vs_historical = historical_cud001 <= STRICT_CUD001_ABS
    installed = (
        opt046.get("selected_q4_decode_path") == "integer_q8"
        and opt046.get("reverted") is not True
    )
    return {
        "candidate": "integer_q8_w4",
        "installed": bool(installed),
        "production_path": opt046.get("selected_q4_decode_path", "packed"),
        "vs_fp64_original_max_abs": orig_abs,
        "vs_fp64_staged_max_abs": staged_abs,
        "historical_cud001_max_abs": historical_cud001,
        "llama_gpu_error": llama_gpu_error if gpu_known else None,
        "v2_ceiling_abs": float(v2_ceiling_abs),
        "meets_v2_original": vs_v2_orig,
        "meets_staged_strict": vs_staged_strict,
        "meets_historical_cud001": vs_historical,
        "keep_strict_q4_dispatch": True,
        "reason": (
            "OPT-046 remains uninstalled. Historical CUD-001 vs scalar host "
            "MMV is 3.05175781e-4 and still fails the 3e-4 envelope. Missing "
            "llama GPU error leaves v2 total-error unadmitted; do not retune "
            "that budget to pass this candidate. Later Q4 work owns any keep."
        ),
    }


def parse_opt059_payload(text: str) -> dict[str, Any]:
    prefix = "QW38_OPT059_NUMERICS_RESULT="
    for line in text.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    raise ValueError("missing OPT-059 native diagnostic payload")


def validate_v2_contract(contract: dict[str, Any], fixture: dict[str, Any]) -> None:
    if contract.get("schema") != V2_SCHEMA:
        raise ValueError("unexpected production numerics v2 schema")
    if contract.get("claims_performance_improvement") is not False:
        raise ValueError("speedup claim is forbidden")
    if contract.get("llama_revision") != LLAMA_REV:
        raise ValueError("llama revision must stay pinned")
    if contract.get("gguf_sha256") != GGUF_SHA:
        raise ValueError("GGUF hash must stay pinned")
    if contract["quality"]["nll_ppl_ratio"] != PRODUCTION_PPL_RATIO:
        raise ValueError("production PPL ratio must stay 1.01")
    if contract["quality"]["recurrence_incremental_nll"] != RECURRENCE_INCREMENTAL_NLL:
        raise ValueError("recurrence drift must stay 0.02")
    frozen = freeze_case_ids()
    got_cal = [row["id"] for row in contract["case_ids"]["calibration"]]
    exp_cal = [row["id"] for row in frozen["calibration"]]
    if got_cal != exp_cal:
        raise ValueError("calibration case IDs mutated")
    got_held = [row["id"] for row in contract["case_ids"]["held_out"]]
    exp_held = [row["id"] for row in frozen["held_out"]]
    if got_held != exp_held:
        raise ValueError("held-out case IDs mutated")
    if contract["case_ids"]["id_manifest_sha256"] != frozen["id_manifest_sha256"]:
        raise ValueError("case ID manifest hash mutated")
    overlap = set(got_cal) & set(got_held)
    if overlap:
        raise ValueError("held-out overlaps calibration")
    for ident, family in contract["families"].items():
        for metric in ("abs", "rms"):
            budget = family["vs_fp64_original_bf16"][metric]
            recomputed = v2_abs_rms_ceiling(
                STRICT_CUD001_ABS if metric == "abs" else STRICT_CUD001_RMS,
                budget.get("llama_gpu_error"),
                nonfinite=int(family.get("nonfinite", 0)),
            )
            if not math.isclose(
                float(budget["ceiling"]),
                float(recomputed["ceiling"]),
                rel_tol=0.0,
                abs_tol=1e-18,
            ):
                raise ValueError(f"v2 budget mutated for {ident} {metric}")
        if (
            family.get("pathological_llama")
            and family["vs_fp64_original_bf16"]["abs"]["path"] != "strict"
        ):
            raise ValueError(f"{ident} inflated a nonfinite llama GPU case")
        if (
            family.get("coverage") == "unadmitted"
            and family.get("production_dispatch_override") == "optimized"
        ):
            raise ValueError(f"{ident} optimized without coverage")
    if fixture.get("opt046_installed") is True:
        raise ValueError("OPT-046 must not be installed by OPT-059")
    if fixture.get("opt042_numeric_reject_preserved") is not True:
        raise ValueError("OPT-042 numeric_reject evidence must stay unchanged")
    inventory = {row["gate"] for row in contract["gate_inventory"]}
    required = {row["gate"] for row in GATE_INVENTORY}
    if not required.issubset(inventory):
        raise ValueError("v1 gate inventory was dropped")
    if "OPT-059-v2" not in inventory:
        raise ValueError("missing OPT-059 v2 gate")
    if (
        contract["staging"]["quartz_q8_1_sum_q"]["stores"]
        != "half(sum(integer quants))"
    ):
        raise ValueError("Quartz Q8_1 staging mislabeled")
    if contract["staging"]["llama_gpu_q8_1_sum_x"]["stores"] != "half(sum(original x))":
        raise ValueError("llama GPU Q8_1 staging mislabeled")
    forbidden = set(contract.get("forbidden", []))
    if not set(FORBIDDEN).issubset(forbidden):
        raise ValueError("forbidden list shrank")
    if not set(PERMITTED_PRODUCTION).issubset(
        set(contract.get("permitted_production_approximations", []))
    ):
        raise ValueError("permitted approximations shrank")
    v1 = json.loads((ROOT / "pins/production_numerics_contract.json").read_text())
    if v1["path_selection"]["selected"] != "strict":
        raise ValueError("v1 contract selected path mutated")
    if v1["path_selection"]["optimized_admitted"] is not False:
        raise ValueError("v1 contract admission mutated")
