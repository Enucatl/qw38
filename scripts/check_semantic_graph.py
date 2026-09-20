#!/usr/bin/env python3
"""Check Qwen3.8-27B semantic-graph contracts against sitting text_config.

Reads ``text_config`` from a Transformers ``config.json`` (no safetensor
payloads) and either prints the semantic-graph summary object or checks that
``docs/architecture/semantic-graph.md`` matches it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

AUTHORITY = ".cache/authorities/qwen3.8-27b-transformers"

CANONICAL_SENTENCE_LOGICAL = (
    "Logical values in this document are graph nodes. They do not imply "
    "physical allocation, materialization, buffer reuse, or kernel fusion."
)

CANONICAL_SENTENCE_CONTRACTS = (
    "Semantic nodes in this document are hardware-independent execution "
    "contracts, not kernels, CUDA graphs, or framework modules."
)

CANONICAL_SENTENCE_INTERNALS = (
    "Internal values and flexible splits are TASK-12 materialization and "
    "fusion candidates; listing them is not a fusion or schedule decision."
)

BOUNDARY_QUESTION_SENTENCE = (
    "Natural node boundaries follow mixer kind and state, residual-add I/O, "
    "embed gather, vocabulary projection, and MTP mix; internal values stay "
    "inside nodes; synchronization edges are the declared node I/O, state, "
    "and shared weights."
)

NODE_TYPE_IDS: tuple[str, ...] = (
    "embed",
    "gated_attn",
    "gated_delta_net",
    "mlp",
    "lm_head",
    "mtp_mix",
)

BOUNDARY_RULE_IDS: tuple[str, ...] = (
    "cut_mixer_kind",
    "cut_mixer_mlp",
    "cut_embed_gather",
    "cut_vocab",
    "cut_mtp_mix",
    "cut_not_module",
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

BOUNDARY_IDS: tuple[str, ...] = (
    "token_id",
    "e",
    "h",
    "h_mid",
    "h_64",
    "logits_0",
    "e_next",
    "mtp_u",
    "h_mtp",
    "logits_1",
)

STATE_IDS: tuple[str, ...] = ("K_state", "V_state", "C_state", "S")

INTERNAL_IDS: tuple[str, ...] = (
    "h_tilde",
    "h_post",
    "h_final",
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

LIVE_ACROSS_IDS: tuple[str, ...] = ("g", "z")
INTRA_EQUATION_REUSE_IDS: tuple[str, ...] = ("k_hat",)
SHARED_WEIGHT_IDS: tuple[str, ...] = ("E", "W_lm")

NODE_INPUTS: dict[str, tuple[str, ...]] = {
    "embed": ("token_id",),
    "gated_attn": ("h",),
    "gated_delta_net": ("h",),
    "mlp": ("h_mid",),
    "lm_head": ("h_64",),
    "mtp_mix": ("h_64", "e_next"),
}

NODE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "embed": ("e",),
    "gated_attn": ("h_mid",),
    "gated_delta_net": ("h_mid",),
    "mlp": ("h",),
    "lm_head": ("logits_0",),
    "mtp_mix": ("mtp_u",),
}

NODE_STATE: dict[str, tuple[str, ...]] = {
    "embed": (),
    "gated_attn": ("K_state", "V_state"),
    "gated_delta_net": ("C_state", "S"),
    "mlp": (),
    "lm_head": (),
    "mtp_mix": (),
}

NODE_INTERNALS: dict[str, tuple[str, ...]] = {
    "embed": (),
    "gated_attn": (
        "h_tilde",
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
    ),
    "gated_delta_net": (
        "h_tilde",
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
        "o",
        "u_gdn",
        "mix_lin",
    ),
    "mlp": ("h_post", "g_mlp", "up", "swiglu", "mlp_out"),
    "lm_head": ("h_final",),
    "mtp_mix": ("e_next_n", "h64_n", "mtp_cat"),
}

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

SYNC_EDGE_CLASSES: tuple[str, ...] = (
    "identity",
    "residual",
    "residual",
    "fanout",
    "mtp",
    "mtp",
    "mtp",
    "output",
    "output",
    "state",
    "state",
    "state",
    "shared_weight",
    "shared_weight",
)

SYNC_EDGE_CLASS_IDS: tuple[str, ...] = (
    "identity",
    "residual",
    "fanout",
    "mtp",
    "output",
    "state",
    "shared_weight",
)

BOTTLENECK_LABELS: tuple[str, ...] = (
    "weight_memory",
    "vocab_memory",
    "state_memory",
    "kv_memory",
    "quadratic_attn",
    "compute",
)

PRECISION_ROLES: tuple[str, ...] = ("param", "activation", "accum", "state")

NODE_SENSITIVE_OPS: dict[str, tuple[str, ...]] = {
    "embed": ("param_bf16",),
    "gated_attn": (
        "param_bf16",
        "residual_stream",
        "live_across_gates",
        "silu_sigmoid",
        "rms_hidden",
        "rms_head",
        "softmax_over_T",
        "attn_av_over_T",
        "gemm_k5120",
        "rope_phase",
        "state_kv_bf16",
    ),
    "gated_delta_net": (
        "param_bf16",
        "residual_stream",
        "live_across_gates",
        "silu_sigmoid",
        "rms_hidden",
        "rms_head",
        "l2_gdn",
        "gemm_k5120",
        "gdn_S_recurrent",
        "gdn_inner_d128",
        "gdn_alpha_beta",
        "state_c_bf16",
        "s_below_f32",
        "conv_fir",
    ),
    "mlp": (
        "param_bf16",
        "residual_stream",
        "silu_sigmoid",
        "rms_hidden",
        "gemm_k5120",
        "gemm_k17408",
    ),
    "lm_head": ("param_bf16", "rms_hidden", "gemm_lm_head"),
    "mtp_mix": ("param_bf16", "rms_hidden", "gemm_k5120"),
}

FLEXIBILITY_KIND_IDS: tuple[str, ...] = (
    "algebraic_equivalent",
    "fuse_internals",
    "split_at_internal",
    "recompute_ephemeral",
    "shared_weight_reuse",
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
    "embed",
    "gated_attn",
    "gated_delta_net",
    "mlp",
    "lm_head",
    "mtp_mix",
    "h",
    "h_mid",
    "h_64",
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

LOCKED_DOCUMENT_NUMBERS: tuple[int | float, ...] = (
    5120,
    17408,
    248320,
    135,
    130,
    17,
    65,
    104857600,
    118235136,
    267386880,
    1271398400,
    52428800,
    2359296,
    12288,
    40960,
    10240,
    69632,
    150994944,
    2542796800,
    3145728,
    61440,
    4096,
    0.75,
)

FORBIDDEN_WINNER_PHRASES: tuple[str, ...] = (
    "selected fusion",
    "selected schedule",
    "selected kernel",
    "selected split",
    "should fuse",
    "recommend fusion",
    "framework module is the node",
    "GGUF is the graph",
    "Quartz graph",
    "llama.cpp graph",
)

ALLOWED_EXCEPTION_PHRASES: tuple[str, ...] = (
    "not a selected fusion",
    "not a fusion or schedule decision",
    "not kernels",
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
    "node_type_ids",
    "n_node_types",
    "n_embed_instances",
    "n_gated_attn_instances",
    "n_gated_delta_net_instances",
    "n_mlp_instances",
    "n_lm_head_instances",
    "n_mtp_mix_instances",
    "n_node_instances_complete",
    "n_node_instances_language",
    "boundary_rule_ids",
    "n_boundary_rules",
    "n_catalog_nodes",
    "catalog_ids",
    "boundary_ids",
    "internal_ids",
    "state_ids",
    "n_boundary_ids",
    "n_internal_ids",
    "n_state_ids",
    "live_across_ids",
    "intra_equation_reuse_ids",
    "shared_weight_ids",
    "catalog_partition_complete",
    "node_inputs",
    "node_outputs",
    "node_state",
    "node_internals",
    "sync_edge_ids",
    "sync_edge_classes",
    "sync_edge_class_ids",
    "n_sync_edges",
    "n_sync_edge_classes",
    "mac_embed",
    "mac_full_proj_per_layer",
    "mac_attn_coeff_per_full_layer",
    "mac_lin_token_per_layer",
    "mac_lin_conv_per_layer",
    "mac_gdn_per_layer",
    "mac_mlp_per_layer",
    "mac_lm_head",
    "mac_mtp_fc",
    "weight_gather_bytes_per_row",
    "kv_bytes_per_full_layer_per_token",
    "kv_bytes_all_per_token",
    "c_bytes_per_layer",
    "s_bytes_per_layer",
    "s_bytes_all",
    "weight_bytes_lm_head",
    "i_mlp_weight_only",
    "i_lm_head_weight_only",
    "i_gdn_vs_s_rw",
    "i_attn_core_vs_kv",
    "bottleneck_labels",
    "n_bottleneck_labels",
    "precision_roles",
    "node_sensitive_ops",
    "param_dtype",
    "kv_conceptual_dtype",
    "s_conceptual_dtype",
    "flexibility_kind_ids",
    "n_flexibility_kinds",
    "split_candidate_ids",
    "n_split_candidates",
    "split_selected",
    "n_split_candidates_selected",
    "decode_prefill_share_graph",
    "residual_add_inside_mixer",
    "residual_add_inside_mlp",
    "residual_input_live_until_add",
    "rms_inside_consumer",
    "rms_roles_not_collapsed",
    "live_across_are_internal",
    "hardware_independent",
    "cuda_mapping_deferred",
    "fusion_selected",
    "schedule_selected",
    "layout_selected",
    "framework_primitives_rejected",
    "fanout_neq_must_store",
    "node_io_neq_must_store",
    "gdn_primary_is_recurrent_eq_17",
    "chunkwise_fp_gap_is_hypothesis",
    "activation_dtype_decided",
    "vision_interface_is_not_a_node",
    "primary_includes_mtp",
    "mtp_omission_is_algebraic_equivalent",
    "mtp_norm_has_no_extra_catalog_id",
    "mixer_xor_by_layer_types",
    "ledger_open_question_boundaries_closed",
    "diagram_ids",
    "n_diagrams",
    "canonical_sentence_logical",
    "canonical_sentence_contracts",
    "canonical_sentence_internals",
    "boundary_question_sentence",
)

REQUIRED_HEADINGS: tuple[str, ...] = (
    "Authority",
    "Semantic contract convention",
    "Boundary derivation",
    "Node types",
    "Per-node contracts",
    "Inter-node edges",
    "Work, traffic, and precision citations",
    "Flexibilities and non-decisions",
    "Deferred vision",
    "Machine-checkable summary JSON",
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


class SemanticGraphMismatch(Exception):
    """Raised when the semantic-graph markdown fails a content check."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def dumps_summary(summary: dict) -> str:
    """Pretty-print the semantic-graph summary object.

    Args:
        summary: Object produced by :func:`instantiate_semantic_graph`.

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


def _ratio_number(numerator: int, denominator: int) -> int | float:
    """Return ``numerator / denominator`` as an int when the quotient is exact.

    Args:
        numerator: Intensity numerator (FLOPs or a FLOP-like product).
        denominator: Intensity denominator in bytes.

    Returns:
        Integer quotient when it divides evenly, otherwise a float.

    Raises:
        AssertionError: If ``denominator`` is zero.
    """

    assert denominator != 0, "intensity denominator is 0"
    if numerator % denominator == 0:
        return numerator // denominator
    return numerator / denominator


def _ordered_mapping(ids: tuple[str, ...], values: dict[str, tuple[str, ...]]) -> dict:
    """Return a JSON object keyed in ``ids`` order with list values.

    Args:
        ids: Key order (node-type ids).
        values: Mapping from id to a tuple of catalog or op ids.

    Returns:
        Ordered dict with JSON-array values.
    """

    return {node_id: list(values[node_id]) for node_id in ids}


def instantiate_semantic_graph(config: dict) -> dict:
    """Compute the TASK-11 semantic-graph summary from ``config``.

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

    n_embed_instances = 1 + n_mtp_blocks
    n_gated_attn_instances = n_full_layers + n_mtp_blocks
    n_gated_delta_net_instances = n_linear_layers
    n_mlp_instances = n_decoder_layers + n_mtp_blocks
    n_lm_head_instances = 1 + n_mtp_blocks
    n_mtp_mix_instances = n_mtp_blocks
    n_node_instances_complete = (
        n_embed_instances
        + n_gated_attn_instances
        + n_gated_delta_net_instances
        + n_mlp_instances
        + n_lm_head_instances
        + n_mtp_mix_instances
    )
    n_node_instances_language = (
        1 + n_full_layers + n_linear_layers + n_decoder_layers + 1
    )

    linear_qkv_width = (
        2 * linear_num_key_heads + linear_num_value_heads
    ) * linear_key_head_dim
    linear_z_width = linear_num_value_heads * linear_value_head_dim
    linear_conv_delay = linear_conv_kernel_dim - 1
    q_proj_out = 2 * n_attn_heads * head_dim
    kv_proj_out = n_kv_heads * head_dim
    o_proj_in = n_attn_heads * head_dim

    mac_embed = 0
    mac_full_proj_per_layer = (
        q_proj_out * hidden_size
        + kv_proj_out * hidden_size
        + kv_proj_out * hidden_size
        + hidden_size * o_proj_in
    )
    mac_attn_coeff_per_full_layer = 2 * n_attn_heads * head_dim
    mac_lin_proj_per_layer = (
        linear_qkv_width * hidden_size
        + linear_z_width * hidden_size
        + linear_num_value_heads * hidden_size
        + linear_num_value_heads * hidden_size
    )
    mac_lin_conv_per_layer = linear_qkv_width * linear_conv_kernel_dim
    mac_gdn_per_layer = (
        3 * linear_num_value_heads * linear_key_head_dim * linear_value_head_dim
    )
    mac_lin_out_per_layer = hidden_size * linear_z_width
    mac_lin_token_per_layer = (
        mac_lin_proj_per_layer
        + mac_lin_conv_per_layer
        + mac_gdn_per_layer
        + mac_lin_out_per_layer
    )
    mac_mlp_per_layer = 3 * intermediate_size * hidden_size
    mac_lm_head = vocab_size * hidden_size
    mac_mtp_fc = hidden_size * (2 * hidden_size)

    weight_gather_bytes_per_row = hidden_size * BYTES_BF16
    kv_bytes_per_full_layer_per_token = 2 * n_kv_heads * head_dim * BYTES_BF16
    kv_bytes_all_per_token = n_full_layers_with_kv * kv_bytes_per_full_layer_per_token
    c_bytes_per_layer = linear_conv_delay * linear_qkv_width * BYTES_BF16
    s_bytes_per_layer = (
        linear_num_value_heads
        * linear_key_head_dim
        * linear_value_head_dim
        * BYTES_F32
    )
    s_bytes_all = n_linear_layers * s_bytes_per_layer
    weight_bytes_lm_head = vocab_size * hidden_size * BYTES_BF16
    mlp_weight_bytes_one_layer = 3 * intermediate_size * hidden_size * BYTES_BF16

    i_mlp_weight_only = _ratio_number(2 * mac_mlp_per_layer, mlp_weight_bytes_one_layer)
    i_lm_head_weight_only = _ratio_number(2 * mac_lm_head, weight_bytes_lm_head)
    i_gdn_vs_s_rw = _ratio_number(2 * mac_gdn_per_layer, 2 * s_bytes_per_layer)
    i_attn_core_vs_kv = _ratio_number(
        2 * mac_attn_coeff_per_full_layer, kv_bytes_per_full_layer_per_token
    )

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
        "node_type_ids": list(NODE_TYPE_IDS),
        "n_node_types": len(NODE_TYPE_IDS),
        "n_embed_instances": n_embed_instances,
        "n_gated_attn_instances": n_gated_attn_instances,
        "n_gated_delta_net_instances": n_gated_delta_net_instances,
        "n_mlp_instances": n_mlp_instances,
        "n_lm_head_instances": n_lm_head_instances,
        "n_mtp_mix_instances": n_mtp_mix_instances,
        "n_node_instances_complete": n_node_instances_complete,
        "n_node_instances_language": n_node_instances_language,
        "boundary_rule_ids": list(BOUNDARY_RULE_IDS),
        "n_boundary_rules": len(BOUNDARY_RULE_IDS),
        "n_catalog_nodes": len(CATALOG_IDS),
        "catalog_ids": list(CATALOG_IDS),
        "boundary_ids": list(BOUNDARY_IDS),
        "internal_ids": list(INTERNAL_IDS),
        "state_ids": list(STATE_IDS),
        "n_boundary_ids": len(BOUNDARY_IDS),
        "n_internal_ids": len(INTERNAL_IDS),
        "n_state_ids": len(STATE_IDS),
        "live_across_ids": list(LIVE_ACROSS_IDS),
        "intra_equation_reuse_ids": list(INTRA_EQUATION_REUSE_IDS),
        "shared_weight_ids": list(SHARED_WEIGHT_IDS),
        "catalog_partition_complete": True,
        "node_inputs": _ordered_mapping(NODE_TYPE_IDS, NODE_INPUTS),
        "node_outputs": _ordered_mapping(NODE_TYPE_IDS, NODE_OUTPUTS),
        "node_state": _ordered_mapping(NODE_TYPE_IDS, NODE_STATE),
        "node_internals": _ordered_mapping(NODE_TYPE_IDS, NODE_INTERNALS),
        "sync_edge_ids": list(SYNC_EDGE_IDS),
        "sync_edge_classes": list(SYNC_EDGE_CLASSES),
        "sync_edge_class_ids": list(SYNC_EDGE_CLASS_IDS),
        "n_sync_edges": len(SYNC_EDGE_IDS),
        "n_sync_edge_classes": len(SYNC_EDGE_CLASS_IDS),
        "mac_embed": mac_embed,
        "mac_full_proj_per_layer": mac_full_proj_per_layer,
        "mac_attn_coeff_per_full_layer": mac_attn_coeff_per_full_layer,
        "mac_lin_token_per_layer": mac_lin_token_per_layer,
        "mac_lin_conv_per_layer": mac_lin_conv_per_layer,
        "mac_gdn_per_layer": mac_gdn_per_layer,
        "mac_mlp_per_layer": mac_mlp_per_layer,
        "mac_lm_head": mac_lm_head,
        "mac_mtp_fc": mac_mtp_fc,
        "weight_gather_bytes_per_row": weight_gather_bytes_per_row,
        "kv_bytes_per_full_layer_per_token": kv_bytes_per_full_layer_per_token,
        "kv_bytes_all_per_token": kv_bytes_all_per_token,
        "c_bytes_per_layer": c_bytes_per_layer,
        "s_bytes_per_layer": s_bytes_per_layer,
        "s_bytes_all": s_bytes_all,
        "weight_bytes_lm_head": weight_bytes_lm_head,
        "i_mlp_weight_only": i_mlp_weight_only,
        "i_lm_head_weight_only": i_lm_head_weight_only,
        "i_gdn_vs_s_rw": i_gdn_vs_s_rw,
        "i_attn_core_vs_kv": i_attn_core_vs_kv,
        "bottleneck_labels": list(BOTTLENECK_LABELS),
        "n_bottleneck_labels": len(BOTTLENECK_LABELS),
        "precision_roles": list(PRECISION_ROLES),
        "node_sensitive_ops": _ordered_mapping(NODE_TYPE_IDS, NODE_SENSITIVE_OPS),
        "param_dtype": param_dtype,
        "kv_conceptual_dtype": "bfloat16",
        "s_conceptual_dtype": mamba_ssm_dtype,
        "flexibility_kind_ids": list(FLEXIBILITY_KIND_IDS),
        "n_flexibility_kinds": len(FLEXIBILITY_KIND_IDS),
        "split_candidate_ids": list(SPLIT_CANDIDATE_IDS),
        "n_split_candidates": len(SPLIT_CANDIDATE_IDS),
        "split_selected": split_selected,
        "n_split_candidates_selected": 0,
        "decode_prefill_share_graph": True,
        "residual_add_inside_mixer": True,
        "residual_add_inside_mlp": True,
        "residual_input_live_until_add": True,
        "rms_inside_consumer": True,
        "rms_roles_not_collapsed": True,
        "live_across_are_internal": True,
        "hardware_independent": True,
        "cuda_mapping_deferred": True,
        "fusion_selected": False,
        "schedule_selected": False,
        "layout_selected": False,
        "framework_primitives_rejected": True,
        "fanout_neq_must_store": True,
        "node_io_neq_must_store": True,
        "gdn_primary_is_recurrent_eq_17": True,
        "chunkwise_fp_gap_is_hypothesis": True,
        "activation_dtype_decided": False,
        "vision_interface_is_not_a_node": True,
        "primary_includes_mtp": True,
        "mtp_omission_is_algebraic_equivalent": True,
        "mtp_norm_has_no_extra_catalog_id": True,
        "mixer_xor_by_layer_types": True,
        "ledger_open_question_boundaries_closed": True,
        "diagram_ids": list(DIAGRAM_REQUIRED_IDS),
        "n_diagrams": 1,
        "canonical_sentence_logical": CANONICAL_SENTENCE_LOGICAL,
        "canonical_sentence_contracts": CANONICAL_SENTENCE_CONTRACTS,
        "canonical_sentence_internals": CANONICAL_SENTENCE_INTERNALS,
        "boundary_question_sentence": BOUNDARY_QUESTION_SENTENCE,
    }
    _assert_identities(summary, text)
    return summary


def _assert_identities(summary: dict, text: dict) -> None:
    """Assert dossier identities on a live semantic-graph object.

    Args:
        summary: Object produced by :func:`instantiate_semantic_graph`.
        text: The ``text_config`` mapping used to build ``summary``.
    """

    assert list(summary) == list(SCHEMA_KEYS), (
        f"schema key order mismatch: {list(summary)} vs {list(SCHEMA_KEYS)}"
    )
    assert summary["authority"] == AUTHORITY
    assert summary["n_linear_layers"] == 48
    assert summary["n_full_layers"] == 16
    assert summary["n_mtp_blocks"] == 1
    assert summary["n_full_layers_with_kv"] == 17
    assert summary["n_full_layers_with_kv"] == (
        summary["n_full_layers"] + summary["n_mtp_blocks"]
    )
    assert summary["full_attention_indices"] == list(LOCKED_FULL_ATTENTION_INDICES)
    assert text.get("dtype") == "bfloat16"
    assert text.get("mamba_ssm_dtype") == "float32"
    assert summary["param_dtype"] == "bfloat16"
    assert summary["s_conceptual_dtype"] == "float32"
    assert summary["kv_conceptual_dtype"] == "bfloat16"
    assert summary["n_node_types"] == 6
    assert summary["n_node_instances_complete"] == 135
    assert summary["n_node_instances_language"] == 130
    assert summary["n_embed_instances"] == 2
    assert summary["n_gated_attn_instances"] == 17
    assert summary["n_gated_delta_net_instances"] == 48
    assert summary["n_mlp_instances"] == 65
    assert summary["n_lm_head_instances"] == 2
    assert summary["n_mtp_mix_instances"] == 1
    assert summary["n_node_instances_complete"] == (
        summary["n_embed_instances"]
        + summary["n_gated_attn_instances"]
        + summary["n_gated_delta_net_instances"]
        + summary["n_mlp_instances"]
        + summary["n_lm_head_instances"]
        + summary["n_mtp_mix_instances"]
    )
    assert summary["n_catalog_nodes"] == 52
    assert summary["n_boundary_ids"] == 10
    assert summary["n_internal_ids"] == 38
    assert summary["n_state_ids"] == 4
    assert (
        summary["n_boundary_ids"]
        + summary["n_internal_ids"]
        + summary["n_state_ids"]
        == 52
    )
    boundary_set = set(summary["boundary_ids"])
    internal_set = set(summary["internal_ids"])
    state_set = set(summary["state_ids"])
    catalog_set = set(summary["catalog_ids"])
    assert boundary_set.isdisjoint(internal_set)
    assert boundary_set.isdisjoint(state_set)
    assert internal_set.isdisjoint(state_set)
    assert boundary_set | internal_set | state_set == catalog_set
    assert summary["catalog_ids"] == list(CATALOG_IDS)
    assert summary["boundary_ids"] == list(BOUNDARY_IDS)
    assert summary["internal_ids"] == list(INTERNAL_IDS)
    assert summary["state_ids"] == list(STATE_IDS)
    assert summary["live_across_ids"] == ["g", "z"]
    assert set(summary["live_across_ids"]).issubset(internal_set)
    assert summary["intra_equation_reuse_ids"] == ["k_hat"]
    assert set(summary["intra_equation_reuse_ids"]).issubset(internal_set)
    assert summary["n_sync_edges"] == 14
    assert summary["n_sync_edge_classes"] == 7
    assert summary["n_flexibility_kinds"] == 5
    assert summary["n_split_candidates"] == 10
    assert summary["n_split_candidates_selected"] == 0
    assert summary["n_boundary_rules"] == 6
    assert summary["n_diagrams"] == 1
    assert summary["n_bottleneck_labels"] == 6
    assert summary["mac_embed"] == 0
    assert summary["mac_full_proj_per_layer"] == 104857600
    assert summary["mac_attn_coeff_per_full_layer"] == 12288
    assert summary["mac_lin_token_per_layer"] == 118235136
    assert summary["mac_mlp_per_layer"] == 267386880
    assert summary["mac_lm_head"] == 1271398400
    assert summary["mac_gdn_per_layer"] == 2359296
    assert summary["mac_mtp_fc"] == 52428800
    assert summary["mac_lin_conv_per_layer"] == 40960
    hidden_size = summary["hidden_size"]
    n_attn_heads = summary["n_attn_heads"]
    n_kv_heads = summary["n_kv_heads"]
    head_dim = summary["head_dim"]
    assert summary["mac_full_proj_per_layer"] == (
        2 * n_attn_heads * head_dim * hidden_size
        + 2 * n_kv_heads * head_dim * hidden_size
        + hidden_size * (n_attn_heads * head_dim)
    )
    assert summary["mac_full_proj_per_layer"] == (
        24 * 256 * hidden_size
        + 2 * 4 * 256 * hidden_size
        + hidden_size * (24 * 256)
        + n_attn_heads * head_dim * hidden_size
    )
    assert summary["mac_mlp_per_layer"] == (
        3 * summary["intermediate_size"] * hidden_size
    )
    assert summary["mac_lm_head"] == summary["vocab_size"] * hidden_size
    assert summary["mac_gdn_per_layer"] == 3 * 48 * 128 * 128
    assert summary["i_mlp_weight_only"] == 1
    assert summary["i_lm_head_weight_only"] == 1
    assert math.isclose(summary["i_gdn_vs_s_rw"], 0.75)
    assert summary["i_gdn_vs_s_rw"] == 0.75
    assert summary["i_attn_core_vs_kv"] == 6
    assert summary["kv_bytes_all_per_token"] == 69632
    assert summary["s_bytes_all"] == 150994944
    assert summary["weight_gather_bytes_per_row"] == 10240
    assert summary["kv_bytes_per_full_layer_per_token"] == 4096
    assert summary["c_bytes_per_layer"] == 61440
    assert summary["s_bytes_per_layer"] == 3145728
    assert summary["weight_bytes_lm_head"] == 2542796800
    assert summary["decode_prefill_share_graph"] is True
    assert summary["hardware_independent"] is True
    assert summary["fusion_selected"] is False
    assert summary["schedule_selected"] is False
    assert summary["layout_selected"] is False
    assert summary["live_across_are_internal"] is True
    assert summary["rms_inside_consumer"] is True
    assert summary["rms_roles_not_collapsed"] is True
    assert summary["residual_add_inside_mixer"] is True
    assert summary["residual_add_inside_mlp"] is True
    assert summary["residual_input_live_until_add"] is True
    assert summary["framework_primitives_rejected"] is True
    assert summary["ledger_open_question_boundaries_closed"] is True
    assert summary["gdn_primary_is_recurrent_eq_17"] is True
    assert summary["chunkwise_fp_gap_is_hypothesis"] is True
    assert summary["activation_dtype_decided"] is False
    assert summary["vision_interface_is_not_a_node"] is True
    assert summary["primary_includes_mtp"] is True
    assert summary["mixer_xor_by_layer_types"] is True
    assert summary["mtp_omission_is_algebraic_equivalent"] is True
    assert summary["mtp_norm_has_no_extra_catalog_id"] is True
    assert summary["fanout_neq_must_store"] is True
    assert summary["node_io_neq_must_store"] is True
    assert summary["cuda_mapping_deferred"] is True
    assert summary["catalog_partition_complete"] is True
    assert all(value is False for value in summary["split_selected"].values())
    assert list(summary["split_selected"]) == list(SPLIT_CANDIDATE_IDS)
    assert summary["node_type_ids"] == list(NODE_TYPE_IDS)
    assert summary["sync_edge_ids"] == list(SYNC_EDGE_IDS)
    assert summary["sync_edge_classes"] == list(SYNC_EDGE_CLASSES)
    assert summary["sync_edge_class_ids"] == list(SYNC_EDGE_CLASS_IDS)
    assert summary["boundary_rule_ids"] == list(BOUNDARY_RULE_IDS)
    assert summary["flexibility_kind_ids"] == list(FLEXIBILITY_KIND_IDS)
    assert summary["split_candidate_ids"] == list(SPLIT_CANDIDATE_IDS)
    assert summary["bottleneck_labels"] == list(BOTTLENECK_LABELS)
    assert summary["diagram_ids"] == list(DIAGRAM_REQUIRED_IDS)
    internals = summary["node_internals"]
    assert "h_tilde" in internals["gated_attn"]
    assert "h_tilde" in internals["gated_delta_net"]
    assert "g" in internals["gated_attn"]
    assert "z" in internals["gated_delta_net"]
    assert "mix_full" in internals["gated_attn"]
    assert "mlp_out" in internals["mlp"]
    owned: set[str] = set()
    for node_id in NODE_TYPE_IDS:
        owned.update(summary["node_internals"][node_id])
    owned.update(summary["boundary_ids"])
    owned.update(summary["state_ids"])
    assert owned == set(CATALOG_IDS)
    assert summary["canonical_sentence_logical"] == CANONICAL_SENTENCE_LOGICAL
    assert summary["canonical_sentence_contracts"] == CANONICAL_SENTENCE_CONTRACTS
    assert summary["canonical_sentence_internals"] == CANONICAL_SENTENCE_INTERNALS
    assert summary["boundary_question_sentence"] == BOUNDARY_QUESTION_SENTENCE
    assert list(summary["node_inputs"]) == list(NODE_TYPE_IDS)
    assert list(summary["node_outputs"]) == list(NODE_TYPE_IDS)
    assert list(summary["node_state"]) == list(NODE_TYPE_IDS)
    assert list(summary["node_internals"]) == list(NODE_TYPE_IDS)
    assert list(summary["node_sensitive_ops"]) == list(NODE_TYPE_IDS)


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

    return any(span_start <= index and end <= span_end for span_start, span_end in spans)


def diff_summary(live: dict, documented: dict) -> list[str]:
    """Compare documented JSON against a live semantic-graph object.

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


def check_semantic_graph(live: dict, semantic_path: Path) -> None:
    """Check a semantic-graph markdown file against a live summary.

    Args:
        live: Semantic-graph summary object from sitting config.
        semantic_path: Path to ``semantic-graph.md``.

    Raises:
        SemanticGraphMismatch: On heading, JSON, diagram, or token mismatches.
        OSError: If the file cannot be read.
    """

    text = semantic_path.read_text(encoding="utf-8")
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
    if CANONICAL_SENTENCE_CONTRACTS not in text:
        differences.append("canonical_sentence_contracts not present verbatim")
    if CANONICAL_SENTENCE_INTERNALS not in text:
        differences.append("canonical_sentence_internals not present verbatim")
    if BOUNDARY_QUESTION_SENTENCE not in text:
        differences.append("boundary_question_sentence not present verbatim")

    id_groups = (
        NODE_TYPE_IDS,
        SYNC_EDGE_IDS,
        FLEXIBILITY_KIND_IDS,
        SPLIT_CANDIDATE_IDS,
        BOUNDARY_RULE_IDS,
        CATALOG_IDS,
        BOUNDARY_IDS,
        INTERNAL_IDS,
        STATE_IDS,
        BOTTLENECK_LABELS,
    )
    for group in id_groups:
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
        differences.append(
            f"line {line}: UNKNOWN outside Deferred vision section"
        )

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
            differences.append(
                f"locked number {token} missing as decimal substring"
            )

    if differences:
        raise SemanticGraphMismatch(differences)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the semantic-graph checker."""

    parser = argparse.ArgumentParser(
        description=(
            "Instantiate Qwen3.8-27B language+MTP semantic-graph summary "
            "from text_config and check docs/architecture/semantic-graph.md."
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
        help="Write semantic-graph summary JSON to stdout.",
    )
    parser.add_argument(
        "--semantic-graph",
        type=Path,
        default=None,
        help="Markdown path whose first json fence must match live summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the semantic-graph checker CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch, 2 missing
        config.
    """

    args = _parse_args(argv)
    try:
        config = load_config(args.config)
        summary = instantiate_semantic_graph(config)
        if args.semantic_graph is not None:
            if not args.semantic_graph.is_file():
                print(
                    f"semantic-graph file not found: {args.semantic_graph}",
                    file=sys.stderr,
                )
                return 1
            check_semantic_graph(summary, args.semantic_graph)
    except MissingConfig as exc:
        print(str(exc.path), file=sys.stderr)
        return 2
    except SemanticGraphMismatch as exc:
        print("\n".join(exc.differences), file=sys.stderr)
        return 1
    except AssertionError as exc:
        print(f"assert failed: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as orig:
        print(f"assert failed: invalid config JSON: {orig}", file=sys.stderr)
        return 1
    except OSError as exc:
        missing = Path(getattr(exc, "filename", args.config))
        print(str(missing), file=sys.stderr)
        return 2

    if args.json or args.semantic_graph is None:
        sys.stdout.write(dumps_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
