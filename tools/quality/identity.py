"""Prove Quartz and llama share tokenization, context, targets, and scoring."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from tools.opt058_quality_baseline import (
    NO_THINKING_SUFFIX,
    render_no_thinking_user_turn,
)
from tools.quality.errors import QualityFrameworkError
from tools.quality.scoring import teacher_forced_nll

LLAMA_ADAPTER = "tools/run_llama_quality_reference.py"
QUARTZ_NATIVE = "build/qw38-cuda-opt058-quality-baseline-test"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
VOCAB_SIZE = 248320
SCORING = "teacher_forced_nll"


def render_qwen_prompt(user: str) -> str:
    rendered = render_no_thinking_user_turn(user)
    if not rendered.endswith(NO_THINKING_SUFFIX):
        raise QualityFrameworkError("missing no-thinking assistant boundary")
    return rendered


def case_plan(
    *,
    case_id: str,
    user: str,
    context: Sequence[int],
    targets: Sequence[int],
    engine: str,
    gguf_sha256: str = GGUF_SHA,
    llama_revision: str = LLAMA_REV,
    vocab_size: int = VOCAB_SIZE,
) -> dict[str, Any]:
    rendered = render_qwen_prompt(user)
    return {
        "id": case_id,
        "engine": engine,
        "user": user,
        "rendered": rendered,
        "assistant_answer_boundary": NO_THINKING_SUFFIX,
        "chat_template": "no_thinking",
        "enable_thinking": False,
        "context": [int(token) for token in context],
        "targets": [int(token) for token in targets],
        "scoring": SCORING,
        "logit_masking": False,
        "gguf_sha256": gguf_sha256,
        "llama_revision": llama_revision,
        "vocab_size": vocab_size,
        "tokenizer": "quartz/llama_identical",
        "llama_adapter": LLAMA_ADAPTER,
        "quartz_native": QUARTZ_NATIVE,
    }


def scoring_identity(
    quartz: Mapping[str, Any],
    llama: Mapping[str, Any],
) -> dict[str, Any]:
    fields = (
        "rendered",
        "context",
        "targets",
        "scoring",
        "chat_template",
        "enable_thinking",
        "logit_masking",
        "gguf_sha256",
        "llama_revision",
        "vocab_size",
        "assistant_answer_boundary",
    )
    mismatches = [name for name in fields if quartz.get(name) != llama.get(name)]
    if mismatches:
        raise QualityFrameworkError(
            "quartz/llama identity mismatch: " + ",".join(mismatches)
        )
    if quartz.get("scoring") != SCORING:
        raise QualityFrameworkError("scoring definition is not teacher_forced_nll")
    if int(quartz.get("vocab_size") or 0) != VOCAB_SIZE:
        raise QualityFrameworkError("vocab_size must be the pinned Qwen 248320")
    return {
        "pass": True,
        "identical_tokenization": True,
        "identical_context_construction": True,
        "identical_target_tokens": True,
        "identical_scoring_definition": True,
        "llama_adapter": LLAMA_ADAPTER,
        "quartz_native": QUARTZ_NATIVE,
        "mismatches": [],
    }


def score_both_engines(
    *,
    case_id: str,
    targets: Sequence[int],
    quartz_logits: Sequence[Sequence[float]],
    llama_logits: Sequence[Sequence[float]],
    vocab_size: int | None = None,
) -> dict[str, Any]:
    quartz = teacher_forced_nll(
        quartz_logits, targets, case_id=case_id, engine="quartz", vocab_size=vocab_size
    )
    llama = teacher_forced_nll(
        llama_logits, targets, case_id=case_id, engine="llama", vocab_size=vocab_size
    )
    return {"quartz": quartz, "llama": llama}
