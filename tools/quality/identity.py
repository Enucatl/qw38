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
GRAPH_PATH = "ffn_only"
# Post-113 production encodings. --quality must not restore packed/r2.
POST113_WEIGHT_ENCODING: dict[str, Any] = {
    "q4_decode": "llama_q4k_mmvq",
    "q8_decode": "r1_w4",
    "q8_path": "dp4a_q8_1",
    "q6_decode": "integer_q8_1",
    "ffn_decode": "paired_integer",
}
POST113_KV_ENCODING: dict[str, Any] = {
    "attention_key": "committed_bf16_prefix",
    "attention_value": "committed_bf16_prefix",
    "logical_rows": "below_frontier_only",
}
POST113_STATE_ENCODING: dict[str, Any] = {
    "gdn_conv": "fp32_rings",
    "gdn_recurrent": "fp32_matrices",
    "gdn_decode": "sequential",
}
POST113_QUANTIZER_PARAMETERS: dict[str, Any] = {
    "q4_staging": "paired_integer",
    "q8_grouping": "grouped_r1_w4",
    "q8_device_layout": "raw_gguf",
    "q6_device_layout": "raw_gguf",
}
ENCODING_FIELDS: tuple[str, ...] = (
    "weight_encoding",
    "kv_encoding",
    "state_encoding",
    "quantizer_parameters",
    "graph_path",
)


def path_encodings(
    *,
    weight_encoding: Mapping[str, Any] | None = None,
    kv_encoding: Mapping[str, Any] | None = None,
    state_encoding: Mapping[str, Any] | None = None,
    quantizer_parameters: Mapping[str, Any] | None = None,
    graph_path: str = GRAPH_PATH,
) -> dict[str, Any]:
    """Identity record for encodings so --quality cannot silent-revert."""
    return {
        "weight_encoding": dict(weight_encoding or POST113_WEIGHT_ENCODING),
        "kv_encoding": dict(kv_encoding or POST113_KV_ENCODING),
        "state_encoding": dict(state_encoding or POST113_STATE_ENCODING),
        "quantizer_parameters": dict(
            quantizer_parameters or POST113_QUANTIZER_PARAMETERS
        ),
        "graph_path": str(graph_path),
    }


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
    encodings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rendered = render_qwen_prompt(user)
    extra = dict(encodings or {})
    path = path_encodings(
        weight_encoding=extra.get("weight_encoding"),
        kv_encoding=extra.get("kv_encoding"),
        state_encoding=extra.get("state_encoding"),
        quantizer_parameters=extra.get("quantizer_parameters"),
        graph_path=str(extra.get("graph_path") or GRAPH_PATH),
    )
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
        **path,
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
        *ENCODING_FIELDS,
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
    encodings = encoding_identity(quartz, llama)
    return {
        "pass": True,
        "identical_tokenization": True,
        "identical_context_construction": True,
        "identical_target_tokens": True,
        "identical_scoring_definition": True,
        "identical_encodings": encodings["pass"],
        "identical_graph_path": encodings["identical_graph_path"],
        "llama_adapter": LLAMA_ADAPTER,
        "quartz_native": QUARTZ_NATIVE,
        "mismatches": [],
        "encodings": encodings,
    }


def encoding_identity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed when weight/KV/state encodings or graph path diverge."""
    mismatches = [name for name in ENCODING_FIELDS if left.get(name) != right.get(name)]
    if mismatches:
        raise QualityFrameworkError(
            "encoding/graph-path identity mismatch: " + ",".join(mismatches)
        )
    graph = left.get("graph_path") or GRAPH_PATH
    if graph != GRAPH_PATH:
        raise QualityFrameworkError(f"graph_path {graph!r} is not {GRAPH_PATH}")
    return {
        "pass": True,
        "identical_weight_encoding": True,
        "identical_kv_encoding": True,
        "identical_state_encoding": True,
        "identical_quantizer_parameters": True,
        "identical_graph_path": True,
        "graph_path": graph,
        "weight_encoding": dict(left.get("weight_encoding") or POST113_WEIGHT_ENCODING),
        "mismatches": [],
    }


def require_contract_encodings(
    record: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject a wrong-path score that silently used the old high-precision kernel."""
    extra = dict(expected or {})
    want = path_encodings(
        weight_encoding=extra.get("weight_encoding"),
        kv_encoding=extra.get("kv_encoding"),
        state_encoding=extra.get("state_encoding"),
        quantizer_parameters=extra.get("quantizer_parameters"),
        graph_path=str(extra.get("graph_path") or GRAPH_PATH),
    )
    got = path_encodings(
        weight_encoding=record.get("weight_encoding"),
        kv_encoding=record.get("kv_encoding"),
        state_encoding=record.get("state_encoding"),
        quantizer_parameters=record.get("quantizer_parameters"),
        graph_path=str(record.get("graph_path") or GRAPH_PATH),
    )
    packed = dict(got["weight_encoding"])
    if packed.get("q4_decode") == "packed" and want["weight_encoding"].get(
        "q4_decode"
    ) not in {None, "packed"}:
        raise QualityFrameworkError(
            "--quality restored packed Q4 over the candidate encoding"
        )
    return encoding_identity(got, want)


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
