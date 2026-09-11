"""Explicit --quality mode: disable shortcuts that would misrepresent production arithmetic."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from tools.quality.errors import QualityFrameworkError

QUALITY_FLAG = "--quality"
QUALITY_SHORTCUTS: tuple[str, ...] = (
    "diagnostic_fallback",
    "experimental_cuda",
    "fmad_true",
    "fast_math",
    "incomplete_vocab_logits",
    "logit_masking",
    "teacher_forced_functional_answers",
    "graph_kernel_divergence",
    "speculative_decoding",
    "numeric_bypass",
    "ssd_streaming_approx",
    "truncated_softmax",
)

SHIPPING_SELECTORS: dict[str, Any] = {
    "q4_decode": "packed",
    "q4_staging": "paired_staged",
    "q8_decode": "r2_w2",
    "q8_path": "dp4a_q8_1",
    "prompt_mmq": "fma_async_x",
    "prompt_mmq_tile": "i128_j128",
    "prompt_attention": "kv_once",
    "decode_gdn": "sequential",
    "decode_attention": "warp_query",
    "prompt_pair": "off",
    "nvccflags": "-O2 --fmad=false",
    "production_numerics": "strict",
    "execution_graphs": "ffn_only",
    "chat_template": "no_thinking",
    "enable_thinking": False,
    "logit_masking": False,
}

QUALITY_DELTA: dict[str, Any] = {
    "flag": QUALITY_FLAG,
    "selector": "production_arithmetic_under_test",
    "precision": {
        "nvccflags": "-O2 --fmad=false",
        "fmad": False,
        "fast_math": False,
        "fp_contract": "off",
    },
    "disables": list(QUALITY_SHORTCUTS),
    "unknown_shortcut": "fail_closed",
    "graph_vs_eager": "same_math_equivalence_not_different_kernel",
    "diagnostic_fallbacks": "disabled",
    "speed_oriented_numeric_bypass": "disabled",
    "does_not_install_a_different_kernel": True,
}


def quality_argv(*, enabled: bool) -> list[str]:
    return [QUALITY_FLAG] if enabled else []


def apply_quality_mode(
    *,
    enabled: bool,
    requested_shortcuts: Sequence[str] | None = None,
    selectors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return effective selectors and the documented shortcut delta.

    Unknown shortcuts fail closed. Quality mode does not swap in a different
    production kernel; it disables shortcuts that would fail to represent the
    shipping arithmetic configuration under test.
    """
    extra = list(requested_shortcuts or ())
    unknown = [name for name in extra if name not in QUALITY_SHORTCUTS]
    if unknown:
        raise QualityFrameworkError(f"unknown quality shortcut: {unknown[0]}")
    active = {name: (not enabled) for name in QUALITY_SHORTCUTS}
    if enabled:
        for name in QUALITY_SHORTCUTS:
            active[name] = False
    effective = dict(SHIPPING_SELECTORS)
    if selectors:
        effective.update(dict(selectors))
    if enabled:
        effective["nvccflags"] = "-O2 --fmad=false"
        effective["logit_masking"] = False
        effective["enable_thinking"] = False
        effective["chat_template"] = "no_thinking"
        if effective.get("fmad") is True:
            raise QualityFrameworkError("quality mode forbids fmad=true")
        if effective.get("nvccflags") != "-O2 --fmad=false":
            raise QualityFrameworkError(
                "quality mode requires production NVCCFLAGS -O2 --fmad=false"
            )
    return {
        "quality": enabled,
        "flag": QUALITY_FLAG,
        "argv": quality_argv(enabled=enabled),
        "shortcuts": active,
        "disabled_shortcuts": [name for name, on in active.items() if not on],
        "enabled_shortcuts": [name for name, on in active.items() if on],
        "selectors": effective,
        "delta": dict(QUALITY_DELTA),
        "graph_vs_eager": "same_math_equivalence",
        "same_math_equivalence_not_different_kernel": True,
    }
