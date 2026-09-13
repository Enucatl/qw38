"""Two-layer correctness for approximate formats. Not a bitwise same-math gate."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from tools.quality.errors import QualityFrameworkError

LAYER_KERNEL = "declared_quantizer_reference"
LAYER_QUALITY = "model_quality_admission"


def approximate_format_contract() -> dict[str, Any]:
    return {
        "layers": [
            {
                "id": LAYER_KERNEL,
                "requirement": (
                    "kernel implements its declared quantizer/dequantizer/"
                    "arithmetic against an independent reference"
                ),
                "checks": [
                    "scales",
                    "clipping",
                    "ties",
                    "tails",
                    "zeros",
                    "extreme_values",
                ],
            },
            {
                "id": LAYER_QUALITY,
                "requirement": (
                    "the complete changed operation is admitted by model quality"
                ),
                "contract_id": "opt116_generated_v1",
            },
        ],
        "do_not_apply_old_same_math_bitwise_gate_to_new_representation": True,
        "corruption_is_not_acceptable_approximation": True,
        "same_path_graph_eager_exact_equivalence": True,
        "checkpoint_restore_exact_equivalence": True,
        "cross_arithmetic_token_identity_not_required": True,
        "ppl_and_recurrence_bounds_unchanged": True,
    }


def evaluate_kernel_reference(
    *,
    declared: Mapping[str, Any],
    independent: Mapping[str, Any],
    bitwise_same_math_required: bool = False,
) -> dict[str, Any]:
    """Layer 1. A new representation must not be judged by the old bitwise gate."""
    if bitwise_same_math_required:
        raise QualityFrameworkError(
            "do not apply an old same-math bitwise gate to a deliberately "
            "different representation"
        )
    checks = (
        "scales",
        "clipping",
        "ties",
        "tails",
        "zeros",
        "extreme_values",
    )
    missing = [name for name in checks if name not in independent]
    if missing:
        raise QualityFrameworkError(
            "independent reference missing " + ",".join(missing)
        )
    mismatches = [
        name
        for name in checks
        if independent.get(name) != declared.get(name) and name in declared
    ]
    corruption = bool(independent.get("corruption")) or bool(declared.get("corruption"))
    if corruption:
        raise QualityFrameworkError("corruption is not an acceptable approximation")
    return {
        "layer": LAYER_KERNEL,
        "pass": not mismatches,
        "mismatches": mismatches,
        "bitwise_same_math_applied": False,
        "checks": list(checks),
    }


def evaluate_quality_layer(
    *,
    successor_quality_pass: bool,
    incomplete: bool = False,
) -> dict[str, Any]:
    if incomplete:
        raise QualityFrameworkError("incomplete quality cannot admit a format")
    return {
        "layer": LAYER_QUALITY,
        "pass": bool(successor_quality_pass),
        "contract_id": "opt116_generated_v1",
    }


def evaluate_same_path_exact(
    *,
    graph_logits: Sequence[float],
    eager_logits: Sequence[float],
    restore_tokens: Sequence[int],
    live_tokens: Sequence[int],
) -> dict[str, Any]:
    if list(graph_logits) != list(eager_logits):
        raise QualityFrameworkError("graph/eager same-path must remain exact")
    if list(restore_tokens) != list(live_tokens):
        raise QualityFrameworkError("checkpoint restore must remain exact")
    return {
        "graph_eager_exact": True,
        "checkpoint_restore_exact": True,
        "pass": True,
    }
