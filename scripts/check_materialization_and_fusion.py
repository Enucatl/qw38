#!/usr/bin/env python3
"""Check Qwen3.8-27B materialization-and-fusion contracts against sitting text_config.

Reads ``text_config`` from a Transformers ``config.json`` (no safetensor
payloads) and either prints the materialization-and-fusion summary object or
checks that ``docs/architecture/materialization-and-fusion.md`` matches it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

AUTHORITY = ".cache/authorities/qwen3.8-27b-transformers"

CANONICAL_SENTENCE_LOGICAL = (
    "Logical values in this document are graph nodes. They do not imply "
    "physical allocation, materialization, buffer reuse, or kernel fusion."
)

CANONICAL_SENTENCE_PHYSICAL = (
    "Physical-materialization need in this document is a mathematical "
    "survival class for evaluating the map, not a CUDA allocation, cache "
    "layout, or selected fusion."
)

CANONICAL_SENTENCE_HYPOTHESIS = (
    "Every fusion, split, reuse, and recomputation proposal in this "
    "document is a HYPOTHESIS; none is a selected winner."
)

CANONICAL_SENTENCE_OPEN_QUESTION = (
    "Apparent fusions are enumerated with working-set and synchronization "
    "tradeoffs; which of them improve total behavior remains a HYPOTHESIS."
)

CATALOG_IDS: tuple[str, ...] = (
    "token_id",
    "e",
    "h",
    "h_tilde",
    "h_mid",
    "h_post",
    "h_64",
    "h_final",
    "logits_0",
    "u_q",
    "q_prime",
    "g",
    "k_raw",
    "v_full",
    "q_n",
    "k_n",
    "q_rope",
    "k_rope",
    "attn",
    "y_gate",
    "mix_full",
    "K_state",
    "V_state",
    "qkv",
    "z",
    "a",
    "b",
    "c_tilde",
    "c",
    "q_lin",
    "k_lin",
    "v_lin",
    "q_hat",
    "k_hat",
    "alpha",
    "beta",
    "S",
    "o",
    "u_gdn",
    "mix_lin",
    "C_state",
    "g_mlp",
    "up",
    "swiglu",
    "mlp_out",
    "e_next",
    "e_next_n",
    "h64_n",
    "mtp_cat",
    "mtp_u",
    "h_mtp",
    "logits_1",
)

PHYSICAL_NEED_CLASS_IDS: tuple[str, ...] = (
    "forced_token_store",
    "forced_output",
    "forced_input",
    "boundary_tradeoff",
    "live_across_tradeoff",
    "reuse_tradeoff",
    "fuse_default",
)

PHYSICAL_NEED_RULE_IDS: tuple[str, ...] = (
    "rule_token_store",
    "rule_output",
    "rule_input",
    "rule_boundary",
    "rule_live_across",
    "rule_reuse",
    "rule_fuse_default",
)

DEFAULT_TACTIC_IDS: tuple[str, ...] = (
    "must_store",
    "must_expose",
    "must_present",
    "keep_live_or_fuse",
    "keep_inside_or_split",
    "reuse_or_recompute",
    "fuse_or_recompute",
)

FORCED_TOKEN_STORE_IDS: tuple[str, ...] = (
    "K_state",
    "V_state",
    "C_state",
    "S",
)

FORCED_OUTPUT_IDS: tuple[str, ...] = ("logits_0", "logits_1")
FORCED_INPUT_IDS: tuple[str, ...] = ("token_id",)

BOUNDARY_TRADEOFF_IDS: tuple[str, ...] = (
    "e",
    "h",
    "h_mid",
    "h_64",
    "e_next",
    "mtp_u",
    "h_mtp",
)

LIVE_ACROSS_TRADEOFF_IDS: tuple[str, ...] = ("g", "z")

REUSE_TRADEOFF_IDS: tuple[str, ...] = (
    "h_tilde",
    "h_post",
    "k_rope",
    "v_full",
    "qkv",
)

FUSE_DEFAULT_IDS: tuple[str, ...] = (
    "h_final",
    "u_q",
    "q_prime",
    "k_raw",
    "q_n",
    "k_n",
    "q_rope",
    "attn",
    "y_gate",
    "mix_full",
    "a",
    "b",
    "c_tilde",
    "c",
    "q_lin",
    "k_lin",
    "v_lin",
    "q_hat",
    "k_hat",
    "alpha",
    "beta",
    "o",
    "u_gdn",
    "mix_lin",
    "g_mlp",
    "up",
    "swiglu",
    "mlp_out",
    "e_next_n",
    "h64_n",
    "mtp_cat",
)

CLASS_TO_IDS: dict[str, tuple[str, ...]] = {
    "forced_token_store": FORCED_TOKEN_STORE_IDS,
    "forced_output": FORCED_OUTPUT_IDS,
    "forced_input": FORCED_INPUT_IDS,
    "boundary_tradeoff": BOUNDARY_TRADEOFF_IDS,
    "live_across_tradeoff": LIVE_ACROSS_TRADEOFF_IDS,
    "reuse_tradeoff": REUSE_TRADEOFF_IDS,
    "fuse_default": FUSE_DEFAULT_IDS,
}

CLASS_TO_DEFAULT_TACTIC: dict[str, str] = {
    "forced_token_store": "must_store",
    "forced_output": "must_expose",
    "forced_input": "must_present",
    "boundary_tradeoff": "keep_live_or_fuse",
    "live_across_tradeoff": "keep_inside_or_split",
    "reuse_tradeoff": "reuse_or_recompute",
    "fuse_default": "fuse_or_recompute",
}

IMPORTANT_IDS: tuple[str, ...] = (
    "token_id",
    "e",
    "h",
    "h_tilde",
    "h_mid",
    "h_post",
    "h_64",
    "logits_0",
    "g",
    "v_full",
    "k_rope",
    "K_state",
    "V_state",
    "qkv",
    "z",
    "S",
    "C_state",
    "e_next",
    "mtp_u",
    "h_mtp",
    "logits_1",
)

HIGH_FANOUT_BOUNDARY_IDS: tuple[str, ...] = ("h", "h_mid", "h_64")

TRADEOFF_TACTIC_IDS: tuple[str, ...] = (
    "reuse",
    "recompute",
    "local_working_set",
    "synchronize",
)

REQUIRES_PRIOR_STATE_IDS: tuple[str, ...] = (
    "K_state",
    "V_state",
    "C_state",
    "S",
)

REUSE_OPPORTUNITY_IDS: tuple[str, ...] = (
    "h",
    "h_tilde",
    "h_mid",
    "h_post",
    "h_64",
    "k_rope",
    "v_full",
    "qkv",
    "S",
    "E",
    "W_lm",
)

INTRA_EQUATION_REUSE_IDS: tuple[str, ...] = ("k_hat",)
SHARED_WEIGHT_IDS: tuple[str, ...] = ("E", "W_lm")

FORBIDDEN_DROP_CATALOG_IDS: tuple[str, ...] = (
    "K_state",
    "V_state",
    "C_state",
    "S",
    "token_id",
    "logits_0",
    "logits_1",
)

FORBIDDEN_DROP_OPS: tuple[str, ...] = (
    "residual_add_mix",
    "residual_add_mlp",
)

FORBIDDEN_DROP_IDS: tuple[str, ...] = (
    *FORBIDDEN_DROP_CATALOG_IDS,
    *FORBIDDEN_DROP_OPS,
)

SYNC_EDGE_IDS: tuple[str, ...] = (
    "identity_e_h0",
    "residual_h",
    "residual_h_mid",
    "fanout_h64",
    "embed_e_next",
    "mtp_u_to_block",
    "h_mtp_to_logits",
    "output_logits_0",
    "output_logits_1",
    "state_kv",
    "state_c",
    "state_s",
    "shared_E",
    "shared_W_lm",
)

FUSIBLE_SYNC_EDGE_IDS: tuple[str, ...] = (
    "identity_e_h0",
    "residual_h",
    "residual_h_mid",
    "fanout_h64",
    "embed_e_next",
    "mtp_u_to_block",
    "h_mtp_to_logits",
)

NONFUSIBLE_SYNC_EDGE_IDS: tuple[str, ...] = (
    "state_kv",
    "state_c",
    "state_s",
    "output_logits_0",
    "output_logits_1",
)

SHARED_WEIGHT_SYNC_EDGE_IDS: tuple[str, ...] = ("shared_E", "shared_W_lm")

FUSION_KIND_IDS: tuple[str, ...] = (
    "fuse_internals",
    "split_at_internal",
    "fuse_across_edge",
)

FUSION_HYPOTHESIS_IDS: tuple[str, ...] = (
    "fuse_gated_attn_internals",
    "fuse_gated_delta_net_internals",
    "fuse_mlp_internals",
    "fuse_lm_head_internals",
    "fuse_mtp_mix_internals",
    "split_g",
    "split_z",
    "split_h_tilde",
    "split_k_rope",
    "split_v_full",
    "split_qkv",
    "split_h_post",
    "split_swiglu",
    "split_h_final",
    "split_mtp_cat",
    "fuse_across_identity_e_h0",
    "fuse_across_residual_h",
    "fuse_across_residual_h_mid",
    "fuse_across_fanout_h64",
    "fuse_across_embed_e_next",
    "fuse_across_mtp_u_to_block",
    "fuse_across_h_mtp_to_logits",
)

FUSION_HYPOTHESIS_KINDS: tuple[str, ...] = (
    "fuse_internals",
    "fuse_internals",
    "fuse_internals",
    "fuse_internals",
    "fuse_internals",
    "split_at_internal",
    "split_at_internal",
    "split_at_internal",
    "split_at_internal",
    "split_at_internal",
    "split_at_internal",
    "split_at_internal",
    "split_at_internal",
    "split_at_internal",
    "split_at_internal",
    "fuse_across_edge",
    "fuse_across_edge",
    "fuse_across_edge",
    "fuse_across_edge",
    "fuse_across_edge",
    "fuse_across_edge",
    "fuse_across_edge",
)

LOCKED_FUSION_EXTRA_SYNC_EDGES: tuple[int, ...] = (
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
)

LOCKED_FUSION_WORKING_SET_BYTES: tuple[int, ...] = (
    0,
    0,
    0,
    0,
    0,
    12288,
    12288,
    10240,
    2048,
    2048,
    20480,
    10240,
    34816,
    10240,
    20480,
    10240,
    10240,
    10240,
    10240,
    10240,
    10240,
    10240,
)

SPLIT_CANDIDATE_IDS: tuple[str, ...] = (
    "g",
    "z",
    "h_tilde",
    "k_rope",
    "v_full",
    "qkv",
    "h_post",
    "swiglu",
    "h_final",
    "mtp_cat",
)

DIAGRAM_REQUIRED_IDS: tuple[str, ...] = (
    "token_id",
    "h",
    "h_mid",
    "h_64",
    "h_tilde",
    "g",
    "z",
    "k_rope",
    "v_full",
    "qkv",
    "K_state",
    "V_state",
    "C_state",
    "S",
    "logits_0",
    "logits_1",
)

LOCKED_FULL_ATTENTION_INDICES: tuple[int, ...] = (
    3,
    7,
    11,
    15,
    19,
    23,
    27,
    31,
    35,
    39,
    43,
    47,
    51,
    55,
    59,
    63,
)

BYTES_BF16 = 2
BYTES_F32 = 4

LOCKED_DOCUMENT_NUMBERS: tuple[int, ...] = (
    5120,
    17408,
    248320,
    10240,
    12288,
    2048,
    20480,
    34816,
    496640,
    6144,
    69632,
    150994944,
    153944064,
    3145728,
    61440,
    4096,
    22,
    21,
    31,
)

FORBIDDEN_WINNER_PHRASES: tuple[str, ...] = (
    "selected fusion",
    "selected winner",
    "winning fusion",
    "should fuse",
    "recommend fusion",
    "must fuse",
    "chosen fusion",
    "CUDA kernel fusion is required",
    "framework module is the node",
    "GGUF is the graph",
    "Quartz graph",
    "llama.cpp graph",
)

ALLOWED_EXCEPTION_PHRASES: tuple[str, ...] = (
    "not a selected fusion",
    "not a CUDA allocation, cache layout, or selected fusion",
    "none is a selected winner",
)

SCHEMA_KEYS: tuple[str, ...] = (
    "authority",
    "hidden_size",
    "intermediate_size",
    "vocab_size",
    "n_decoder_layers",
    "n_linear_layers",
    "n_full_layers",
    "n_mtp_blocks",
    "n_full_layers_with_kv",
    "full_attention_indices",
    "n_attn_heads",
    "n_kv_heads",
    "head_dim",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "bytes_bf16",
    "bytes_f32",
    "n_catalog_nodes",
    "catalog_ids",
    "physical_need_class_ids",
    "n_physical_need_classes",
    "physical_need_rule_ids",
    "n_physical_need_rules",
    "default_tactic_ids",
    "n_default_tactics",
    "forced_token_store_ids",
    "forced_output_ids",
    "forced_input_ids",
    "boundary_tradeoff_ids",
    "live_across_tradeoff_ids",
    "reuse_tradeoff_ids",
    "fuse_default_ids",
    "n_forced_token_store",
    "n_forced_output",
    "n_forced_input",
    "n_boundary_tradeoff",
    "n_live_across_tradeoff",
    "n_reuse_tradeoff",
    "n_fuse_default",
    "physical_need_partition_complete",
    "physical_need_by_id",
    "default_tactic_by_id",
    "important_ids",
    "n_important_ids",
    "high_fanout_boundary_ids",
    "n_high_fanout_boundary_ids",
    "tradeoff_tactic_ids",
    "n_tradeoff_tactics",
    "recomputable_ids",
    "n_recomputable_ids",
    "requires_prior_state_ids",
    "reuse_opportunity_ids",
    "n_reuse_opportunity_ids",
    "intra_equation_reuse_ids",
    "shared_weight_ids",
    "forbidden_drop_catalog_ids",
    "n_forbidden_drop_catalog_ids",
    "forbidden_drop_ops",
    "n_forbidden_drop_ops",
    "forbidden_drop_ids",
    "n_forbidden_drops",
    "sync_edge_ids",
    "n_sync_edges",
    "fusible_sync_edge_ids",
    "n_fusible_sync_edges",
    "nonfusible_sync_edge_ids",
    "n_nonfusible_sync_edges",
    "shared_weight_sync_edge_ids",
    "n_shared_weight_sync_edges",
    "n_baseline_sync_edges",
    "fusion_kind_ids",
    "n_fusion_kinds",
    "fusion_hypothesis_ids",
    "fusion_hypothesis_kinds",
    "fusion_extra_sync_edges",
    "fusion_working_set_bytes",
    "n_fusion_hypotheses",
    "fusion_selected",
    "n_fusion_hypotheses_selected",
    "fusion_usefulness_label",
    "fusion_winner_selected",
    "split_candidate_ids",
    "n_split_candidates",
    "split_selected",
    "n_split_candidates_selected",
    "residual_elems",
    "residual_bytes",
    "g_elems",
    "g_bytes",
    "z_elems",
    "z_bytes",
    "k_rope_elems",
    "k_rope_bytes",
    "v_full_elems",
    "v_full_bytes",
    "qkv_elems",
    "qkv_bytes",
    "swiglu_elems",
    "swiglu_bytes",
    "mtp_cat_elems",
    "mtp_cat_bytes",
    "logits_elems",
    "logits_bytes",
    "kv_bytes_per_full_layer_per_token",
    "kv_bytes_all_per_token",
    "c_bytes_per_layer",
    "s_bytes_per_layer",
    "s_bytes_all",
    "storage_fixed_bytes",
    "storage_kv_bytes_coeff_T",
    "decode_prefill_share_taxonomy",
    "fanout_neq_must_store",
    "node_io_neq_must_store",
    "must_survive_neq_cuda_malloc",
    "live_across_are_internal",
    "residual_add_inside_mixer",
    "residual_add_inside_mlp",
    "residual_input_live_until_add",
    "rms_roles_not_collapsed",
    "gdn_primary_is_recurrent_eq_17",
    "chunkwise_not_zero_s_traffic",
    "fanout_h64_cannot_hide_from_one_consumer",
    "state_write_not_optional",
    "hardware_independent",
    "cuda_mapping_deferred",
    "schedule_selected",
    "layout_selected",
    "activation_dtype_decided",
    "primary_includes_mtp",
    "vision_interface_is_not_a_node",
    "ledger_open_question_fusions_remain_hypothesis",
    "diagram_ids",
    "n_diagrams",
    "canonical_sentence_logical",
    "canonical_sentence_physical",
    "canonical_sentence_hypothesis",
    "canonical_sentence_open_question",
)

REQUIRED_HEADINGS: tuple[str, ...] = (
    "Authority",
    "Physical versus mathematical",
    "Physical-need taxonomy",
    "Catalog physical-need table",
    "Forced stores and forbidden drops",
    "Reuse, recompute, local-working-set, and synchronization tradeoffs",
    "Fusion hypotheses",
    "Working-set and synchronization cost identities",
    "Deferred vision",
    "Machine-checkable summary JSON",
)

REQUIRED_ID_GROUPS: tuple[tuple[str, ...], ...] = (
    PHYSICAL_NEED_CLASS_IDS,
    FUSION_HYPOTHESIS_IDS,
    FUSION_KIND_IDS,
    TRADEOFF_TACTIC_IDS,
    CATALOG_IDS,
    IMPORTANT_IDS,
    SPLIT_CANDIDATE_IDS,
    SYNC_EDGE_IDS,
    FUSIBLE_SYNC_EDGE_IDS,
    NONFUSIBLE_SYNC_EDGE_IDS,
)

JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)
MERMAID_FENCE_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)
FORBIDDEN_RE = re.compile(r"TBD|TODO|\?\?\?")
UNKNOWN_RE = re.compile(r"UNKNOWN")


class MissingConfig(Exception):
    """Raised when the config path is absent or unreadable."""

    def __init__(self, path: Path) -> None:
        super().__init__(str(path))
        self.path = path


class MaterializationMismatch(Exception):
    """Raised when the materialization-and-fusion markdown fails a content check."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def dumps_summary(summary: dict) -> str:
    """Pretty-print the materialization-and-fusion summary object.

    Args:
        summary: Object produced by :func:`instantiate_materialization_and_fusion`.

    Returns:
        Pretty-printed JSON including a trailing newline.
    """

    return json.dumps(summary, indent=2) + "\n"


def first_json_fence(text: str) -> dict:
    """Parse the first fenced ``json`` code block in ``text``.

    Args:
        text: Markdown document containing a `` ```json `` fence.

    Returns:
        Decoded JSON object.

    Raises:
        AssertionError: If no fence exists or the payload is not an object.
    """

    match = JSON_FENCE_RE.search(text)
    assert match is not None, "no fenced json block found"
    payload = json.loads(match.group(1))
    assert isinstance(payload, dict), "fenced json block is not an object"
    return payload


def _require_int(mapping: dict, key: str) -> int:
    """Return ``mapping[key]`` as an ``int``.

    Args:
        mapping: Config object.
        key: Required field name.

    Returns:
        Integer field value.

    Raises:
        AssertionError: If the field is missing or not an integer value.
    """

    assert key in mapping, f"missing text_config field: {key}"
    value = mapping[key]
    assert isinstance(value, int) and not isinstance(value, bool), (
        f"text_config.{key} is not an int: {value!r}"
    )
    return value


def _require_str(mapping: dict, key: str) -> str:
    """Return ``mapping[key]`` as a ``str``.

    Args:
        mapping: Config object.
        key: Required field name.

    Returns:
        String field value.

    Raises:
        AssertionError: If the field is missing or not a string.
    """

    assert key in mapping, f"missing text_config field: {key}"
    value = mapping[key]
    assert isinstance(value, str), f"text_config.{key} is not a str: {value!r}"
    return value


def _physical_need_by_id() -> dict[str, str]:
    """Return the locked physical-need class for every catalog ID.

    Returns:
        Mapping keyed in ``CATALOG_IDS`` order.
    """

    class_by_id: dict[str, str] = {}
    for class_id, member_ids in CLASS_TO_IDS.items():
        for catalog_id in member_ids:
            assert catalog_id not in class_by_id, (
                f"catalog id {catalog_id!r} assigned to multiple classes"
            )
            class_by_id[catalog_id] = class_id
    return {catalog_id: class_by_id[catalog_id] for catalog_id in CATALOG_IDS}


def instantiate_materialization_and_fusion(config: dict) -> dict:
    """Compute the TASK-12 materialization-and-fusion summary from ``config``.

    Args:
        config: Parsed Transformers ``config.json`` object.

    Returns:
        Ordered summary dict matching the dossier JSON schema.

    Raises:
        AssertionError: If required fields are missing or identities fail.
    """

    assert isinstance(config, dict), "config.json root is not an object"
    assert "text_config" in config, "config.json has no text_config"
    text = config["text_config"]
    assert isinstance(text, dict), "text_config is not an object"

    hidden_size = _require_int(text, "hidden_size")
    intermediate_size = _require_int(text, "intermediate_size")
    vocab_size = _require_int(text, "vocab_size")
    n_decoder_layers = _require_int(text, "num_hidden_layers")
    n_attn_heads = _require_int(text, "num_attention_heads")
    n_kv_heads = _require_int(text, "num_key_value_heads")
    head_dim = _require_int(text, "head_dim")
    linear_num_key_heads = _require_int(text, "linear_num_key_heads")
    linear_num_value_heads = _require_int(text, "linear_num_value_heads")
    linear_key_head_dim = _require_int(text, "linear_key_head_dim")
    linear_value_head_dim = _require_int(text, "linear_value_head_dim")
    linear_conv_kernel_dim = _require_int(text, "linear_conv_kernel_dim")
    n_mtp_blocks = _require_int(text, "mtp_num_hidden_layers")
    full_attention_interval = _require_int(text, "full_attention_interval")
    param_dtype = _require_str(text, "dtype")
    mamba_ssm_dtype = _require_str(text, "mamba_ssm_dtype")

    assert "layer_types" in text, "missing layer_types"
    layer_types = text["layer_types"]
    assert isinstance(layer_types, list), "layer_types is not a list"
    assert len(layer_types) == n_decoder_layers, (
        f"layer_types length {len(layer_types)} != num_hidden_layers "
        f"{n_decoder_layers}"
    )
    full_from_types = [
        index for index, name in enumerate(layer_types) if name == "full_attention"
    ]
    linear_from_types = [
        index
        for index, name in enumerate(layer_types)
        if name == "linear_attention"
    ]
    assert len(full_from_types) + len(linear_from_types) == n_decoder_layers, (
        "layer_types contains names other than linear_attention/full_attention"
    )
    full_from_interval = [
        index
        for index in range(n_decoder_layers)
        if index % full_attention_interval == full_attention_interval - 1
    ]
    assert full_from_types == full_from_interval, (
        f"layer_types full indices {full_from_types} != interval {full_from_interval}"
    )
    n_full_layers = len(full_from_types)
    n_linear_layers = len(linear_from_types)
    n_full_layers_with_kv = n_full_layers + n_mtp_blocks

    residual_elems = hidden_size
    residual_bytes = residual_elems * BYTES_BF16
    g_elems = n_attn_heads * head_dim
    g_bytes = g_elems * BYTES_BF16
    z_elems = linear_num_value_heads * linear_value_head_dim
    z_bytes = z_elems * BYTES_BF16
    k_rope_elems = n_kv_heads * head_dim
    k_rope_bytes = k_rope_elems * BYTES_BF16
    v_full_elems = k_rope_elems
    v_full_bytes = k_rope_bytes
    qkv_elems = (
        2 * linear_num_key_heads + linear_num_value_heads
    ) * linear_key_head_dim
    qkv_bytes = qkv_elems * BYTES_BF16
    swiglu_elems = intermediate_size
    swiglu_bytes = swiglu_elems * BYTES_BF16
    mtp_cat_elems = 2 * hidden_size
    mtp_cat_bytes = mtp_cat_elems * BYTES_BF16
    logits_elems = vocab_size
    logits_bytes = logits_elems * BYTES_BF16

    linear_conv_delay = linear_conv_kernel_dim - 1
    kv_bytes_per_full_layer_per_token = 2 * n_kv_heads * head_dim * BYTES_BF16
    kv_bytes_all_per_token = n_full_layers_with_kv * kv_bytes_per_full_layer_per_token
    c_bytes_per_layer = linear_conv_delay * qkv_elems * BYTES_BF16
    s_bytes_per_layer = (
        linear_num_value_heads
        * linear_key_head_dim
        * linear_value_head_dim
        * BYTES_F32
    )
    s_bytes_all = n_linear_layers * s_bytes_per_layer
    storage_fixed_bytes = n_linear_layers * c_bytes_per_layer + s_bytes_all
    storage_kv_bytes_coeff_t = kv_bytes_all_per_token

    physical_need_by_id = _physical_need_by_id()
    default_tactic_by_id = {
        catalog_id: CLASS_TO_DEFAULT_TACTIC[physical_need_by_id[catalog_id]]
        for catalog_id in CATALOG_IDS
    }
    recomputable_ids = [
        catalog_id
        for catalog_id in CATALOG_IDS
        if catalog_id not in REQUIRES_PRIOR_STATE_IDS
    ]
    fusion_working_set_bytes = [
        0,
        0,
        0,
        0,
        0,
        g_bytes,
        z_bytes,
        residual_bytes,
        k_rope_bytes,
        v_full_bytes,
        qkv_bytes,
        residual_bytes,
        swiglu_bytes,
        residual_bytes,
        mtp_cat_bytes,
        residual_bytes,
        residual_bytes,
        residual_bytes,
        residual_bytes,
        residual_bytes,
        residual_bytes,
        residual_bytes,
    ]
    fusion_selected = {hypothesis_id: False for hypothesis_id in FUSION_HYPOTHESIS_IDS}
    split_selected = {split_id: False for split_id in SPLIT_CANDIDATE_IDS}

    summary = {
        "authority": AUTHORITY,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "n_decoder_layers": n_decoder_layers,
        "n_linear_layers": n_linear_layers,
        "n_full_layers": n_full_layers,
        "n_mtp_blocks": n_mtp_blocks,
        "n_full_layers_with_kv": n_full_layers_with_kv,
        "full_attention_indices": full_from_types,
        "n_attn_heads": n_attn_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "linear_num_value_heads": linear_num_value_heads,
        "linear_key_head_dim": linear_key_head_dim,
        "linear_value_head_dim": linear_value_head_dim,
        "bytes_bf16": BYTES_BF16,
        "bytes_f32": BYTES_F32,
        "n_catalog_nodes": len(CATALOG_IDS),
        "catalog_ids": list(CATALOG_IDS),
        "physical_need_class_ids": list(PHYSICAL_NEED_CLASS_IDS),
        "n_physical_need_classes": len(PHYSICAL_NEED_CLASS_IDS),
        "physical_need_rule_ids": list(PHYSICAL_NEED_RULE_IDS),
        "n_physical_need_rules": len(PHYSICAL_NEED_RULE_IDS),
        "default_tactic_ids": list(DEFAULT_TACTIC_IDS),
        "n_default_tactics": len(DEFAULT_TACTIC_IDS),
        "forced_token_store_ids": list(FORCED_TOKEN_STORE_IDS),
        "forced_output_ids": list(FORCED_OUTPUT_IDS),
        "forced_input_ids": list(FORCED_INPUT_IDS),
        "boundary_tradeoff_ids": list(BOUNDARY_TRADEOFF_IDS),
        "live_across_tradeoff_ids": list(LIVE_ACROSS_TRADEOFF_IDS),
        "reuse_tradeoff_ids": list(REUSE_TRADEOFF_IDS),
        "fuse_default_ids": list(FUSE_DEFAULT_IDS),
        "n_forced_token_store": len(FORCED_TOKEN_STORE_IDS),
        "n_forced_output": len(FORCED_OUTPUT_IDS),
        "n_forced_input": len(FORCED_INPUT_IDS),
        "n_boundary_tradeoff": len(BOUNDARY_TRADEOFF_IDS),
        "n_live_across_tradeoff": len(LIVE_ACROSS_TRADEOFF_IDS),
        "n_reuse_tradeoff": len(REUSE_TRADEOFF_IDS),
        "n_fuse_default": len(FUSE_DEFAULT_IDS),
        "physical_need_partition_complete": True,
        "physical_need_by_id": physical_need_by_id,
        "default_tactic_by_id": default_tactic_by_id,
        "important_ids": list(IMPORTANT_IDS),
        "n_important_ids": len(IMPORTANT_IDS),
        "high_fanout_boundary_ids": list(HIGH_FANOUT_BOUNDARY_IDS),
        "n_high_fanout_boundary_ids": len(HIGH_FANOUT_BOUNDARY_IDS),
        "tradeoff_tactic_ids": list(TRADEOFF_TACTIC_IDS),
        "n_tradeoff_tactics": len(TRADEOFF_TACTIC_IDS),
        "recomputable_ids": recomputable_ids,
        "n_recomputable_ids": len(recomputable_ids),
        "requires_prior_state_ids": list(REQUIRES_PRIOR_STATE_IDS),
        "reuse_opportunity_ids": list(REUSE_OPPORTUNITY_IDS),
        "n_reuse_opportunity_ids": len(REUSE_OPPORTUNITY_IDS),
        "intra_equation_reuse_ids": list(INTRA_EQUATION_REUSE_IDS),
        "shared_weight_ids": list(SHARED_WEIGHT_IDS),
        "forbidden_drop_catalog_ids": list(FORBIDDEN_DROP_CATALOG_IDS),
        "n_forbidden_drop_catalog_ids": len(FORBIDDEN_DROP_CATALOG_IDS),
        "forbidden_drop_ops": list(FORBIDDEN_DROP_OPS),
        "n_forbidden_drop_ops": len(FORBIDDEN_DROP_OPS),
        "forbidden_drop_ids": list(FORBIDDEN_DROP_IDS),
        "n_forbidden_drops": len(FORBIDDEN_DROP_IDS),
        "sync_edge_ids": list(SYNC_EDGE_IDS),
        "n_sync_edges": len(SYNC_EDGE_IDS),
        "fusible_sync_edge_ids": list(FUSIBLE_SYNC_EDGE_IDS),
        "n_fusible_sync_edges": len(FUSIBLE_SYNC_EDGE_IDS),
        "nonfusible_sync_edge_ids": list(NONFUSIBLE_SYNC_EDGE_IDS),
        "n_nonfusible_sync_edges": len(NONFUSIBLE_SYNC_EDGE_IDS),
        "shared_weight_sync_edge_ids": list(SHARED_WEIGHT_SYNC_EDGE_IDS),
        "n_shared_weight_sync_edges": len(SHARED_WEIGHT_SYNC_EDGE_IDS),
        "n_baseline_sync_edges": len(SYNC_EDGE_IDS),
        "fusion_kind_ids": list(FUSION_KIND_IDS),
        "n_fusion_kinds": len(FUSION_KIND_IDS),
        "fusion_hypothesis_ids": list(FUSION_HYPOTHESIS_IDS),
        "fusion_hypothesis_kinds": list(FUSION_HYPOTHESIS_KINDS),
        "fusion_extra_sync_edges": list(LOCKED_FUSION_EXTRA_SYNC_EDGES),
        "fusion_working_set_bytes": fusion_working_set_bytes,
        "n_fusion_hypotheses": len(FUSION_HYPOTHESIS_IDS),
        "fusion_selected": fusion_selected,
        "n_fusion_hypotheses_selected": 0,
        "fusion_usefulness_label": "HYPOTHESIS",
        "fusion_winner_selected": False,
        "split_candidate_ids": list(SPLIT_CANDIDATE_IDS),
        "n_split_candidates": len(SPLIT_CANDIDATE_IDS),
        "split_selected": split_selected,
        "n_split_candidates_selected": 0,
        "residual_elems": residual_elems,
        "residual_bytes": residual_bytes,
        "g_elems": g_elems,
        "g_bytes": g_bytes,
        "z_elems": z_elems,
        "z_bytes": z_bytes,
        "k_rope_elems": k_rope_elems,
        "k_rope_bytes": k_rope_bytes,
        "v_full_elems": v_full_elems,
        "v_full_bytes": v_full_bytes,
        "qkv_elems": qkv_elems,
        "qkv_bytes": qkv_bytes,
        "swiglu_elems": swiglu_elems,
        "swiglu_bytes": swiglu_bytes,
        "mtp_cat_elems": mtp_cat_elems,
        "mtp_cat_bytes": mtp_cat_bytes,
        "logits_elems": logits_elems,
        "logits_bytes": logits_bytes,
        "kv_bytes_per_full_layer_per_token": kv_bytes_per_full_layer_per_token,
        "kv_bytes_all_per_token": kv_bytes_all_per_token,
        "c_bytes_per_layer": c_bytes_per_layer,
        "s_bytes_per_layer": s_bytes_per_layer,
        "s_bytes_all": s_bytes_all,
        "storage_fixed_bytes": storage_fixed_bytes,
        "storage_kv_bytes_coeff_T": storage_kv_bytes_coeff_t,
        "decode_prefill_share_taxonomy": True,
        "fanout_neq_must_store": True,
        "node_io_neq_must_store": True,
        "must_survive_neq_cuda_malloc": True,
        "live_across_are_internal": True,
        "residual_add_inside_mixer": True,
        "residual_add_inside_mlp": True,
        "residual_input_live_until_add": True,
        "rms_roles_not_collapsed": True,
        "gdn_primary_is_recurrent_eq_17": True,
        "chunkwise_not_zero_s_traffic": True,
        "fanout_h64_cannot_hide_from_one_consumer": True,
        "state_write_not_optional": True,
        "hardware_independent": True,
        "cuda_mapping_deferred": True,
        "schedule_selected": False,
        "layout_selected": False,
        "activation_dtype_decided": False,
        "primary_includes_mtp": True,
        "vision_interface_is_not_a_node": True,
        "ledger_open_question_fusions_remain_hypothesis": True,
        "diagram_ids": list(DIAGRAM_REQUIRED_IDS),
        "n_diagrams": 1,
        "canonical_sentence_logical": CANONICAL_SENTENCE_LOGICAL,
        "canonical_sentence_physical": CANONICAL_SENTENCE_PHYSICAL,
        "canonical_sentence_hypothesis": CANONICAL_SENTENCE_HYPOTHESIS,
        "canonical_sentence_open_question": CANONICAL_SENTENCE_OPEN_QUESTION,
    }
    _assert_locked_identities(summary, param_dtype, mamba_ssm_dtype)
    return summary


def _assert_locked_identities(
    summary: dict, param_dtype: str, mamba_ssm_dtype: str
) -> None:
    """Fail closed when live arithmetic disagrees with locked TASK-12 identities.

    Args:
        summary: Instantiated summary object.
        param_dtype: Sitting ``text_config.dtype``.
        mamba_ssm_dtype: Sitting ``text_config.mamba_ssm_dtype``.

    Raises:
        AssertionError: If a locked count, partition, or byte identity fails.
    """

    assert list(summary) == list(SCHEMA_KEYS), "summary key order != SCHEMA_KEYS"
    assert summary["n_linear_layers"] == 48
    assert summary["n_full_layers"] == 16
    assert summary["n_mtp_blocks"] == 1
    assert summary["n_full_layers_with_kv"] == 17
    assert summary["full_attention_indices"] == list(LOCKED_FULL_ATTENTION_INDICES)
    assert param_dtype == "bfloat16"
    assert mamba_ssm_dtype == "float32"
    assert summary["n_catalog_nodes"] == 52
    assert len(summary["catalog_ids"]) == 52
    assert summary["n_physical_need_classes"] == 7
    assert summary["n_physical_need_rules"] == 7
    assert summary["n_default_tactics"] == 7
    assert summary["n_forced_token_store"] == 4
    assert summary["n_forced_output"] == 2
    assert summary["n_forced_input"] == 1
    assert summary["n_boundary_tradeoff"] == 7
    assert summary["n_live_across_tradeoff"] == 2
    assert summary["n_reuse_tradeoff"] == 5
    assert summary["n_fuse_default"] == 31
    class_count_sum = (
        summary["n_forced_token_store"]
        + summary["n_forced_output"]
        + summary["n_forced_input"]
        + summary["n_boundary_tradeoff"]
        + summary["n_live_across_tradeoff"]
        + summary["n_reuse_tradeoff"]
        + summary["n_fuse_default"]
    )
    assert class_count_sum == 52
    class_arrays = [
        summary["forced_token_store_ids"],
        summary["forced_output_ids"],
        summary["forced_input_ids"],
        summary["boundary_tradeoff_ids"],
        summary["live_across_tradeoff_ids"],
        summary["reuse_tradeoff_ids"],
        summary["fuse_default_ids"],
    ]
    seen: set[str] = set()
    for member_ids in class_arrays:
        overlap = seen.intersection(member_ids)
        assert not overlap, f"physical-need class overlap: {sorted(overlap)}"
        seen.update(member_ids)
    assert seen == set(summary["catalog_ids"]), (
        "physical-need class union != catalog_ids"
    )
    assert summary["forced_token_store_ids"] == ["K_state", "V_state", "C_state", "S"]
    assert summary["live_across_tradeoff_ids"] == ["g", "z"]
    assert summary["n_important_ids"] == 21
    assert summary["n_tradeoff_tactics"] == 4
    assert summary["n_recomputable_ids"] == 48
    assert summary["n_fusion_kinds"] == 3
    assert summary["n_fusion_hypotheses"] == 22
    assert summary["n_fusion_hypotheses_selected"] == 0
    assert summary["n_split_candidates"] == 10
    assert summary["n_split_candidates_selected"] == 0
    assert summary["n_sync_edges"] == 14
    assert summary["n_fusible_sync_edges"] == 7
    assert summary["n_nonfusible_sync_edges"] == 5
    assert summary["n_shared_weight_sync_edges"] == 2
    assert (
        summary["n_fusible_sync_edges"]
        + summary["n_nonfusible_sync_edges"]
        + summary["n_shared_weight_sync_edges"]
        == 14
    )
    assert summary["n_diagrams"] == 1
    assert summary["n_forbidden_drop_catalog_ids"] == 7
    assert summary["residual_elems"] == summary["hidden_size"] == 5120
    assert summary["residual_bytes"] == 10240
    assert summary["g_elems"] == 24 * 256 == 6144
    assert summary["g_bytes"] == 12288
    assert summary["z_elems"] == 48 * 128 == 6144
    assert summary["z_bytes"] == 12288
    assert summary["k_rope_elems"] == 4 * 256 == 1024
    assert summary["k_rope_bytes"] == 2048
    assert summary["qkv_elems"] == 10240
    assert summary["qkv_bytes"] == 20480
    assert summary["swiglu_elems"] == summary["intermediate_size"] == 17408
    assert summary["swiglu_bytes"] == 34816
    assert summary["mtp_cat_elems"] == 2 * summary["hidden_size"] == 10240
    assert summary["mtp_cat_bytes"] == 20480
    assert summary["logits_elems"] == summary["vocab_size"] == 248320
    assert summary["logits_bytes"] == 496640
    assert summary["kv_bytes_all_per_token"] == 69632
    assert summary["s_bytes_all"] == 150994944
    assert summary["storage_fixed_bytes"] == 153944064
    assert summary["fusion_working_set_bytes"] == list(LOCKED_FUSION_WORKING_SET_BYTES)
    assert summary["fusion_extra_sync_edges"] == list(LOCKED_FUSION_EXTRA_SYNC_EDGES)
    assert all(value is False for value in summary["fusion_selected"].values())
    assert all(value is False for value in summary["split_selected"].values())
    assert summary["fusion_winner_selected"] is False
    assert summary["ledger_open_question_fusions_remain_hypothesis"] is True
    assert summary["decode_prefill_share_taxonomy"] is True
    assert summary["hardware_independent"] is True
    assert summary["live_across_are_internal"] is True
    assert summary["fanout_h64_cannot_hide_from_one_consumer"] is True
    assert summary["state_write_not_optional"] is True
    assert summary["chunkwise_not_zero_s_traffic"] is True
    assert summary["vision_interface_is_not_a_node"] is True
    assert summary["schedule_selected"] is False
    assert summary["layout_selected"] is False
    assert summary["physical_need_class_ids"] == list(PHYSICAL_NEED_CLASS_IDS)
    assert summary["fusion_hypothesis_ids"] == list(FUSION_HYPOTHESIS_IDS)
    assert "g" in summary["live_across_tradeoff_ids"]
    assert "S" in summary["forced_token_store_ids"]
    assert "h_mid" in summary["boundary_tradeoff_ids"]
    assert "h_tilde" in summary["reuse_tradeoff_ids"]
    assert summary["n_attn_heads"] * summary["head_dim"] == summary["g_elems"]
    assert (
        summary["linear_num_value_heads"] * summary["linear_value_head_dim"]
        == summary["z_elems"]
    )


def load_config(config_path: Path) -> dict:
    """Load Transformers ``config.json`` from ``config_path``.

    Args:
        config_path: Path to the sitting authority config file.

    Returns:
        Parsed JSON object.

    Raises:
        MissingConfig: If the file is absent or unreadable.
        AssertionError: If the file is not a JSON object.
    """

    if not config_path.is_file():
        raise MissingConfig(config_path)
    try:
        payload_text = config_path.read_text(encoding="utf-8")
        payload = json.loads(payload_text)
    except OSError as exc:
        raise MissingConfig(config_path) from exc
    assert isinstance(payload, dict), "config.json root is not an object"
    return payload


def _deferred_vision_span(text: str) -> tuple[int, int]:
    """Return the character span of the Deferred vision section.

    Args:
        text: Full markdown document.

    Returns:
        Inclusive-start exclusive-end indices of that section.

    Raises:
        AssertionError: If the heading is missing.
    """

    match = re.search(r"^## Deferred vision\s*$", text, flags=re.MULTILINE)
    assert match is not None, "missing ## Deferred vision heading"
    start = match.start()
    next_heading = HEADING_RE.search(text, match.end())
    end = next_heading.start() if next_heading is not None else len(text)
    return start, end


def _exception_spans(text: str) -> list[tuple[int, int]]:
    """Return character spans of allowed exception phrases.

    Args:
        text: Full markdown document.

    Returns:
        Inclusive-start exclusive-end spans that may contain otherwise
        forbidden winner phrases.
    """

    spans: list[tuple[int, int]] = []
    for phrase in ALLOWED_EXCEPTION_PHRASES:
        start = 0
        while True:
            index = text.find(phrase, start)
            if index < 0:
                break
            spans.append((index, index + len(phrase)))
            start = index + 1
    return spans


def _inside_exception(index: int, end: int, spans: list[tuple[int, int]]) -> bool:
    """Return whether ``[index, end)`` sits inside an allowed exception span.

    Args:
        index: Match start.
        end: Match end.
        spans: Allowed exception spans.

    Returns:
        True when the match is covered by an exception phrase.
    """

    return any(
        span_start <= index and end <= span_end for span_start, span_end in spans
    )


def diff_summary(live: dict, documented: dict) -> list[str]:
    """Compare documented JSON against a live materialization-and-fusion object.

    Args:
        live: Fresh script object.
        documented: Object parsed from a markdown JSON fence.

    Returns:
        Human-readable difference strings; empty when the documents match.
    """

    diffs: list[str] = []
    live_keys = list(live)
    doc_keys = list(documented)
    if doc_keys != live_keys:
        extra = [key for key in doc_keys if key not in live]
        missing = [key for key in live_keys if key not in documented]
        if extra:
            diffs.append(f"documented extra keys: {extra}")
        if missing:
            diffs.append(f"documented missing keys: {missing}")
        if not extra and not missing and doc_keys != live_keys:
            diffs.append(f"documented key order {doc_keys} != live {live_keys}")
    for key in SCHEMA_KEYS:
        if key not in documented or key not in live:
            continue
        if documented[key] != live[key]:
            diffs.append(
                f"{key}: documented {documented[key]!r} != live {live[key]!r}"
            )
    return diffs


def check_materialization_and_fusion(live: dict, materialization_path: Path) -> None:
    """Check a materialization-and-fusion markdown file against a live summary.

    Args:
        live: Materialization-and-fusion summary object from sitting config.
        materialization_path: Path to ``materialization-and-fusion.md``.

    Raises:
        MaterializationMismatch: On heading, JSON, diagram, or token mismatches.
        OSError: If the file cannot be read.
    """

    text = materialization_path.read_text(encoding="utf-8")
    differences: list[str] = []

    headings = HEADING_RE.findall(text)
    if headings != list(REQUIRED_HEADINGS):
        differences.append(
            "heading mismatch:\n"
            f"  documented: {headings}\n"
            f"  required:   {list(REQUIRED_HEADINGS)}"
        )

    try:
        documented = first_json_fence(text)
    except (AssertionError, json.JSONDecodeError) as exc:
        differences.append(f"json fence: {exc}")
        documented = None

    if documented is not None:
        differences.extend(diff_summary(live, documented))

    mermaid_fences = MERMAID_FENCE_RE.findall(text)
    if len(mermaid_fences) != 1:
        differences.append(f"mermaid fence count {len(mermaid_fences)} != 1")
    else:
        body = mermaid_fences[0]
        stripped = body.lstrip()
        if not (
            stripped.startswith("flowchart TB") or stripped.startswith("flowchart LR")
        ):
            differences.append(
                "mermaid fence body does not start with flowchart TB or flowchart LR"
            )
        if "flowchart" not in body:
            differences.append("mermaid fence does not contain flowchart")
        missing_ids = [
            node_id for node_id in DIAGRAM_REQUIRED_IDS if node_id not in body
        ]
        if missing_ids:
            differences.append(f"diagram 1: missing node ids {missing_ids}")

    if CANONICAL_SENTENCE_LOGICAL not in text:
        differences.append("canonical_sentence_logical not present verbatim")
    if CANONICAL_SENTENCE_PHYSICAL not in text:
        differences.append("canonical_sentence_physical not present verbatim")
    if CANONICAL_SENTENCE_HYPOTHESIS not in text:
        differences.append("canonical_sentence_hypothesis not present verbatim")
    if CANONICAL_SENTENCE_OPEN_QUESTION not in text:
        differences.append("canonical_sentence_open_question not present verbatim")

    for group in REQUIRED_ID_GROUPS:
        for item_id in group:
            if item_id not in text:
                differences.append(f"required id {item_id!r} missing as substring")

    if "HYPOTHESIS" not in text:
        differences.append("word HYPOTHESIS missing")
    if "hardware-independent" not in text:
        differences.append("word hardware-independent missing")

    for match in FORBIDDEN_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        differences.append(f"line {line}: forbidden token {match.group(0)!r}")

    try:
        vis_start, vis_end = _deferred_vision_span(text)
    except AssertionError as exc:
        differences.append(str(exc))
        vis_start, vis_end = 0, 0

    for match in UNKNOWN_RE.finditer(text):
        if vis_start <= match.start() < vis_end:
            continue
        line = text.count("\n", 0, match.start()) + 1
        differences.append(f"line {line}: UNKNOWN outside Deferred vision section")

    exception_spans = _exception_spans(text)
    for phrase in FORBIDDEN_WINNER_PHRASES:
        start = 0
        while True:
            index = text.find(phrase, start)
            if index < 0:
                break
            end = index + len(phrase)
            if not _inside_exception(index, end, exception_spans):
                line = text.count("\n", 0, index) + 1
                differences.append(
                    f"line {line}: forbidden winner phrase {phrase!r}"
                )
            start = index + 1

    for number in LOCKED_DOCUMENT_NUMBERS:
        token = str(number)
        if token not in text:
            differences.append(f"locked number {token} missing as decimal substring")

    if differences:
        raise MaterializationMismatch(differences)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the materialization-and-fusion checker."""

    parser = argparse.ArgumentParser(
        description=(
            "Instantiate Qwen3.8-27B language+MTP materialization-and-fusion "
            "summary from text_config and check "
            "docs/architecture/materialization-and-fusion.md."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to Transformers config.json (text_config authority).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write materialization-and-fusion summary JSON to stdout.",
    )
    parser.add_argument(
        "--materialization",
        type=Path,
        default=None,
        help="Markdown path whose first json fence must match live summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the materialization-and-fusion checker CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch, 2 missing
        config.
    """

    args = _parse_args(argv)
    try:
        config = load_config(args.config)
        summary = instantiate_materialization_and_fusion(config)
        if args.materialization is not None:
            if not args.materialization.is_file():
                print(
                    f"materialization file not found: {args.materialization}",
                    file=sys.stderr,
                )
                return 1
            check_materialization_and_fusion(summary, args.materialization)
    except MissingConfig as exc:
        print(str(exc.path), file=sys.stderr)
        return 2
    except MaterializationMismatch as exc:
        print("\n".join(exc.differences), file=sys.stderr)
        return 1
    except AssertionError as orig:
        print(f"assert failed: {orig}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as orig:
        print(f"assert failed: invalid config JSON: {orig}", file=sys.stderr)
        return 1
    except OSError as exc:
        missing = Path(getattr(exc, "filename", args.config))
        print(str(missing), file=sys.stderr)
        return 2

    if args.json or args.materialization is None:
        sys.stdout.write(dumps_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
