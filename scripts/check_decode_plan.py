#!/usr/bin/env python3
"""Check Qwen3.8-27B decode-plan contracts against sitting text_config.

Reads ``text_config`` from a Transformers ``config.json`` (no safetensor
payloads) and either prints the decode-plan summary object or checks that
``docs/architecture/decode-plan.md`` matches it.
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

CANONICAL_SENTENCE_SCHEDULE = (
    "The decode schedule in this document is a hardware-independent order "
    "of TASK-11 nodes for one-token inference, not a CUDA graph, kernel "
    "launch sequence, or selected fusion."
)

CANONICAL_SENTENCE_TRAFFIC = (
    "Unavoidable traffic is the TASK-06 unique-weight, state, and "
    "forced-activation minimum; proposed-boundary traffic is this "
    "schedule's named stage-cut channel."
)

CANONICAL_SENTENCE_OPEN_QUESTION = (
    "Boundary-added traffic relative to the mathematical minimum is the "
    "named stage-cut channel compared with TASK-06 unique-weight, state, "
    "and activation views; which packing or fusion hypotheses change that "
    "extra remains a HYPOTHESIS."
)

NODE_TYPE_IDS: tuple[str, ...] = (
    "embed",
    "gated_attn",
    "gated_delta_net",
    "mlp",
    "lm_head",
    "mtp_mix",
)

STAGE_KIND_IDS: tuple[str, ...] = (
    "embed_current",
    "language_mixer",
    "language_mlp",
    "lm_head_primary",
    "embed_next",
    "mtp_mix",
    "mtp_mixer",
    "mtp_mlp",
    "lm_head_mtp",
)

STAGE_MULTIPLICITIES: tuple[int, ...] = (1, 64, 64, 1, 1, 1, 1, 1, 1)

STAGE_NODE_TYPES: tuple[str, ...] = (
    "embed",
    "mixer_xor",
    "mlp",
    "lm_head",
    "embed",
    "mtp_mix",
    "gated_attn",
    "mlp",
    "lm_head",
)

STAGE_PRIMARY_SEQUENCE_IDS: tuple[str, ...] = (
    "seq_gather_row",
    "seq_gemm_codes_then_scales",
    "seq_gemm_codes_then_scales",
    "seq_lm_head_full",
    "seq_gather_row",
    "seq_gemm_codes_then_scales",
    "seq_gemm_codes_then_scales",
    "seq_gemm_codes_then_scales",
    "seq_lm_head_full",
)

STAGE_VISIBILITY_IN: dict[str, tuple[str, ...]] = {
    "embed_current": ("token_id",),
    "language_mixer": ("h",),
    "language_mlp": ("h_mid",),
    "lm_head_primary": ("h_64",),
    "embed_next": ("token_id",),
    "mtp_mix": ("h_64", "e_next"),
    "mtp_mixer": ("h",),
    "mtp_mlp": ("h_mid",),
    "lm_head_mtp": ("h_mtp",),
}

STAGE_VISIBILITY_OUT: dict[str, tuple[str, ...]] = {
    "embed_current": ("e",),
    "language_mixer": ("h_mid",),
    "language_mlp": ("h",),
    "lm_head_primary": ("logits_0",),
    "embed_next": ("e_next",),
    "mtp_mix": ("mtp_u",),
    "mtp_mixer": ("h_mid",),
    "mtp_mlp": ("h_mtp",),
    "lm_head_mtp": ("logits_1",),
}

STAGE_STATE_READ: dict[str, tuple[str, ...]] = {
    "embed_current": (),
    "language_mixer": ("K_state", "V_state", "C_state", "S"),
    "language_mlp": (),
    "lm_head_primary": (),
    "embed_next": (),
    "mtp_mix": (),
    "mtp_mixer": ("K_state", "V_state"),
    "mtp_mlp": (),
    "lm_head_mtp": (),
}

STAGE_STATE_WRITE: dict[str, tuple[str, ...]] = {
    "embed_current": (),
    "language_mixer": ("K_state", "V_state", "C_state", "S"),
    "language_mlp": (),
    "lm_head_primary": (),
    "embed_next": (),
    "mtp_mix": (),
    "mtp_mixer": ("K_state", "V_state"),
    "mtp_mlp": (),
    "lm_head_mtp": (),
}

READY_CONSTRAINT_IDS: tuple[str, ...] = (
    "ready_embed_current",
    "ready_language_mixer_0",
    "ready_language_mlp",
    "ready_language_mixer_next",
    "ready_lm_head_primary",
    "ready_embed_next",
    "ready_mtp_mix",
    "ready_mtp_mixer",
    "ready_mtp_mlp",
    "ready_lm_head_mtp",
)

HOIST_HYPOTHESIS_IDS: tuple[str, ...] = (
    "hoist_embed_next",
    "overlap_fanout_h64",
    "reuse_E",
    "reuse_W_lm",
    "reuse_h64",
)

CONSUMER_SEQUENCE_IDS: tuple[str, ...] = (
    "seq_gemm_codes_then_scales",
    "seq_gemm_interleaved_group",
    "seq_gather_row",
    "seq_lm_head_full",
    "seq_outlier_extra",
    "seq_state_s_dense",
    "seq_specialized_tile",
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

BOTTLENECK_LABELS: tuple[str, ...] = (
    "weight_memory",
    "vocab_memory",
    "state_memory",
    "kv_memory",
    "quadratic_attn",
    "compute",
)

DIAGRAM_REQUIRED_IDS: tuple[str, ...] = (
    "embed_current",
    "language_mixer",
    "language_mlp",
    "lm_head_primary",
    "embed_next",
    "mtp_mix",
    "mtp_mixer",
    "mtp_mlp",
    "lm_head_mtp",
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

EXAMPLE_T: tuple[int, ...] = (1, 4096)

BYTES_BF16 = 2
BYTES_F32 = 4

LOCKED_DOCUMENT_NUMBERS: tuple[int | float, ...] = (
    5120,
    17408,
    248320,
    135,
    130,
    133,
    2355200,
    1361920,
    993280,
    3123200,
    5847040,
    665600,
    798720,
    10240,
    20480,
    496640,
    52098598912,
    52098619392,
    152047616,
    153944064,
    439087104,
    69632,
    150994944,
    2542796800,
    27433238528,
    208896,
    27433447424,
    28288876544,
    104857600,
    118235136,
    267386880,
    1271398400,
    0.75,
)

FORBIDDEN_WINNER_PHRASES: tuple[str, ...] = (
    "selected fusion",
    "selected packing",
    "selected kernel",
    "winning fusion",
    "should fuse",
    "recommend fusion",
    "ideal byte sequence is",
    "artifact boundary is",
    "CUDA kernel fusion is required",
    "thread block",
    "warp shuffle",
    "Quartz graph",
    "llama.cpp graph",
    "GGUF is the schedule",
)

ALLOWED_EXCEPTION_PHRASES: tuple[str, ...] = (
    "not a selected fusion",
    "not a CUDA graph, kernel launch sequence, or selected fusion",
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
    "stage_kind_ids",
    "n_stage_kinds",
    "stage_multiplicities",
    "stage_node_types",
    "n_stage_instances",
    "serial_stage_kind_order",
    "stage_primary_sequence_ids",
    "gated_delta_net_state_sequence_id",
    "stage_visibility_in",
    "stage_visibility_out",
    "stage_state_read",
    "stage_state_write",
    "language_mixer_state_is_xor",
    "ready_constraint_ids",
    "n_ready_constraints",
    "hoist_hypothesis_ids",
    "hoist_selected",
    "n_hoist_hypotheses",
    "n_hoist_hypotheses_selected",
    "hoist_usefulness_label",
    "n_token_presentations",
    "example_T",
    "decode_T_new",
    "T_is_stored_length_after_append",
    "incoming_state_populated",
    "primary_includes_mtp",
    "mac_C_language",
    "mac_C_complete",
    "mac_A_language",
    "mac_A_complete",
    "mac_decode_complete_at_example_T",
    "mac_full_proj_per_layer",
    "mac_lin_token_per_layer",
    "mac_mlp_per_layer",
    "mac_lm_head",
    "mac_mtp_fc",
    "mac_gdn_per_layer",
    "i_mlp_weight_only",
    "i_lm_head_weight_only",
    "i_gdn_vs_s_rw",
    "i_attn_core_vs_kv",
    "bottleneck_labels",
    "n_bottleneck_labels",
    "weight_bytes_unique_non_embed",
    "weight_gather_bytes_per_row",
    "weight_gather_bytes_decode_complete",
    "weight_unique_plus_gather_decode_complete",
    "weight_boundary_added_unique_bytes",
    "weight_second_w_lm_read_bytes",
    "weight_bytes_lm_head",
    "kv_bytes_per_full_layer_per_token",
    "kv_bytes_all_per_token",
    "c_bytes_per_layer",
    "c_write_bytes_all",
    "c_read_bytes_all",
    "s_bytes_per_layer",
    "s_bytes_all",
    "decode_write_bytes",
    "decode_read_fixed_bytes",
    "decode_read_kv_bytes_coeff_Tm1",
    "decode_read_bytes_at_example_T",
    "storage_kv_bytes_coeff_T",
    "storage_fixed_bytes",
    "storage_bytes_at_example_T",
    "state_boundary_added_bytes",
    "residual_bytes",
    "g_bytes",
    "z_bytes",
    "logits_bytes",
    "n_logit_outputs",
    "n_identity_e_h0_crossings",
    "n_h_mid_crossings",
    "n_h_interlayer_crossings",
    "n_fanout_h64_crossings",
    "n_embed_e_next_crossings",
    "n_mtp_u_crossings",
    "n_h_mtp_crossings",
    "n_residual_stage_crossings",
    "act_residual_stage_cut_bytes",
    "act_logits_stage_cut_bytes",
    "act_stage_cut_decode_complete_bytes",
    "act_forced_decode_complete_bytes",
    "act_region_cut_decode_complete_bytes",
    "act_boundary_added_vs_forced",
    "act_boundary_added_vs_region_cut",
    "act_gz_local_bytes",
    "act_h_mid_stage_cut_bytes",
    "act_h64_second_consumer_bytes",
    "act_h64_second_consumer_is_hypothesis",
    "consumer_sequence_ids",
    "n_consumer_sequences",
    "fusion_hypothesis_ids",
    "n_fusion_hypotheses",
    "fusion_selected",
    "n_fusion_hypotheses_selected",
    "fusion_usefulness_label",
    "sync_edge_ids",
    "n_sync_edges",
    "decode_prefill_share_graph",
    "residual_add_inside_mixer",
    "residual_add_inside_mlp",
    "residual_input_live_until_add",
    "rms_inside_consumer",
    "live_across_are_internal",
    "hardware_independent",
    "cuda_mapping_deferred",
    "thread_geometry_absent",
    "prefill_schedule_deferred",
    "layout_selected",
    "fusion_winner_selected",
    "ideal_byte_sequence_selected",
    "artifact_boundary_selected",
    "schedule_serial_selected",
    "gdn_primary_is_recurrent_eq_17",
    "chunkwise_not_zero_s_traffic",
    "state_write_not_optional",
    "fanout_h64_cannot_hide_from_one_consumer",
    "weight_unique_counted_once",
    "weight_second_w_lm_read_is_hypothesis",
    "mixer_xor_by_layer_types",
    "activation_dtype_decided",
    "vision_interface_is_not_a_node",
    "mtp_omission_is_algebraic_equivalent",
    "ledger_open_question_boundary_traffic_closed",
    "diagram_ids",
    "n_diagrams",
    "canonical_sentence_logical",
    "canonical_sentence_schedule",
    "canonical_sentence_traffic",
    "canonical_sentence_open_question",
)

REQUIRED_HEADINGS: tuple[str, ...] = (
    "Authority",
    "Decode schedule convention",
    "One-token setting",
    "Stage kinds and serial order",
    "Per-stage loads, state, visibility, and reuse",
    "Unavoidable versus proposed-boundary traffic",
    "Packing and fusion hypotheses",
    "Work citations and non-decisions",
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


class DecodePlanMismatch(Exception):
    """Raised when the decode-plan markdown fails a content check."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def dumps_summary(summary: dict) -> str:
    """Pretty-print the decode-plan summary object.

    Args:
        summary: Object produced by :func:`instantiate_decode_plan`.

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


def _ordered_stage_mapping(
    values: dict[str, tuple[str, ...]],
) -> dict[str, list[str]]:
    """Return a JSON object keyed in ``STAGE_KIND_IDS`` order.

    Args:
        values: Mapping from stage-kind id to a tuple of catalog ids.

    Returns:
        Ordered dict with JSON-array values.
    """

    return {stage_id: list(values[stage_id]) for stage_id in STAGE_KIND_IDS}


def instantiate_decode_plan(config: dict) -> dict:
    """Compute the TASK-13 decode-plan summary from ``config``.

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
    _require_str(text, "dtype")
    _require_str(text, "mamba_ssm_dtype")

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

    mac_c_linear_attn = n_linear_layers * mac_lin_token_per_layer
    mac_c_full_attn = n_full_layers * mac_full_proj_per_layer
    mac_c_mlp = n_decoder_layers * mac_mlp_per_layer
    mac_c_lm_head = mac_lm_head
    mac_c_mtp = (
        mac_mtp_fc + mac_full_proj_per_layer + mac_mlp_per_layer + mac_lm_head
    )
    mac_c_language = (
        mac_c_linear_attn + mac_c_full_attn + mac_c_mlp + mac_c_lm_head
    )
    mac_c_complete = mac_c_language + mac_c_mtp
    mac_a_language = n_full_layers * mac_attn_coeff_per_full_layer
    mac_a_complete = mac_a_language + mac_attn_coeff_per_full_layer
    example_t = list(EXAMPLE_T)
    mac_decode_complete = [
        mac_c_complete + mac_a_complete * t_value for t_value in example_t
    ]

    lin_params_per_layer = (
        linear_qkv_width * hidden_size
        + linear_z_width * hidden_size
        + linear_num_value_heads * hidden_size
        + linear_num_value_heads * hidden_size
        + linear_qkv_width * 1 * linear_conv_kernel_dim
        + linear_num_value_heads
        + linear_num_value_heads
        + linear_value_head_dim
        + hidden_size * linear_z_width
    )
    full_params_per_layer = (
        q_proj_out * hidden_size
        + kv_proj_out * hidden_size
        + kv_proj_out * hidden_size
        + hidden_size * o_proj_in
        + head_dim
        + head_dim
    )
    mtp_params = (
        hidden_size * (2 * hidden_size)
        + 5 * hidden_size
        + full_params_per_layer
        + 3 * intermediate_size * hidden_size
    )
    weight_bytes_language_linear_attn = (
        n_linear_layers * lin_params_per_layer * BYTES_BF16
    )
    weight_bytes_language_self_attn = (
        n_full_layers * full_params_per_layer * BYTES_BF16
    )
    weight_bytes_language_mlp = (
        n_decoder_layers * 3 * intermediate_size * hidden_size * BYTES_BF16
    )
    weight_bytes_language_layer_norms = (
        2 * n_decoder_layers * hidden_size * BYTES_BF16
    )
    weight_bytes_language_final_norm = hidden_size * BYTES_BF16
    weight_bytes_lm_head = vocab_size * hidden_size * BYTES_BF16
    weight_bytes_mtp = mtp_params * BYTES_BF16
    weight_bytes_embed_table = vocab_size * hidden_size * BYTES_BF16
    weight_bytes_language_mtp_excl_vision = (
        weight_bytes_embed_table
        + weight_bytes_language_final_norm
        + weight_bytes_lm_head
        + weight_bytes_language_layer_norms
        + weight_bytes_language_linear_attn
        + weight_bytes_language_self_attn
        + weight_bytes_language_mlp
        + weight_bytes_mtp
    )
    weight_bytes_unique_non_embed = (
        weight_bytes_language_mtp_excl_vision - weight_bytes_embed_table
    )
    weight_gather_bytes_per_row = hidden_size * BYTES_BF16
    weight_gather_bytes_decode_complete = 2 * weight_gather_bytes_per_row
    weight_unique_plus_gather_decode_complete = (
        weight_bytes_unique_non_embed + weight_gather_bytes_decode_complete
    )

    kv_bytes_per_full_layer_per_token = 2 * n_kv_heads * head_dim * BYTES_BF16
    kv_bytes_all_per_token = n_full_layers_with_kv * kv_bytes_per_full_layer_per_token
    c_bytes_per_layer = linear_conv_delay * linear_qkv_width * BYTES_BF16
    c_write_bytes_all = n_linear_layers * linear_qkv_width * BYTES_BF16
    c_read_bytes_all = n_linear_layers * c_bytes_per_layer
    s_bytes_per_layer = (
        linear_num_value_heads
        * linear_key_head_dim
        * linear_value_head_dim
        * BYTES_F32
    )
    s_bytes_all = n_linear_layers * s_bytes_per_layer
    storage_kv_bytes_coeff_t = kv_bytes_all_per_token
    storage_fixed_bytes = c_read_bytes_all + s_bytes_all
    decode_write_bytes = (
        kv_bytes_all_per_token + c_write_bytes_all + s_bytes_all
    )
    decode_read_kv_bytes_coeff_tm1 = kv_bytes_all_per_token
    decode_read_fixed_bytes = storage_fixed_bytes
    storage_bytes_at_example_t = [
        storage_kv_bytes_coeff_t * t_value + storage_fixed_bytes
        for t_value in example_t
    ]
    decode_read_bytes_at_example_t = [
        decode_read_kv_bytes_coeff_tm1 * (t_value - 1) + decode_read_fixed_bytes
        for t_value in example_t
    ]

    residual_bytes = hidden_size * BYTES_BF16
    g_bytes = n_attn_heads * head_dim * BYTES_BF16
    z_bytes = linear_z_width * BYTES_BF16
    logits_bytes = vocab_size * BYTES_BF16
    n_logit_outputs = 2
    n_identity_e_h0_crossings = 1
    n_h_mid_crossings = n_decoder_layers + n_mtp_blocks
    n_h_interlayer_crossings = n_decoder_layers - 1
    n_fanout_h64_crossings = 1
    n_embed_e_next_crossings = 1
    n_mtp_u_crossings = 1
    n_h_mtp_crossings = 1
    n_residual_stage_crossings = (
        n_identity_e_h0_crossings
        + n_h_mid_crossings
        + n_h_interlayer_crossings
        + n_fanout_h64_crossings
        + n_embed_e_next_crossings
        + n_mtp_u_crossings
        + n_h_mtp_crossings
    )
    act_residual_stage_cut_bytes = n_residual_stage_crossings * residual_bytes
    act_logits_stage_cut_bytes = n_logit_outputs * logits_bytes
    act_stage_cut_decode_complete_bytes = (
        act_residual_stage_cut_bytes + act_logits_stage_cut_bytes
    )
    act_forced_decode_language_bytes = (
        n_decoder_layers * residual_bytes * 2
        + n_full_layers * g_bytes
        + n_linear_layers * z_bytes
        + logits_bytes
    )
    act_forced_decode_complete_bytes = (
        act_forced_decode_language_bytes
        + 2 * residual_bytes
        + g_bytes
        + logits_bytes
    )
    act_region_cut_decode_language_bytes = (
        act_forced_decode_language_bytes
        + residual_bytes
        + n_decoder_layers * residual_bytes
        + n_decoder_layers * residual_bytes
        + n_decoder_layers * residual_bytes
        + n_decoder_layers * residual_bytes
        + residual_bytes
        + residual_bytes
    )
    act_region_cut_decode_complete_bytes = (
        act_region_cut_decode_language_bytes
        + 2 * residual_bytes
        + g_bytes
        + logits_bytes
        + 4 * residual_bytes
        + 3 * residual_bytes
    )
    act_boundary_added_vs_forced = (
        act_stage_cut_decode_complete_bytes - act_forced_decode_complete_bytes
    )
    act_boundary_added_vs_region_cut = (
        act_stage_cut_decode_complete_bytes - act_region_cut_decode_complete_bytes
    )
    act_gz_local_bytes = (
        n_full_layers * g_bytes + n_linear_layers * z_bytes + n_mtp_blocks * g_bytes
    )
    act_h_mid_stage_cut_bytes = n_h_mid_crossings * residual_bytes

    mlp_weight_bytes_one_layer = 3 * intermediate_size * hidden_size * BYTES_BF16
    i_mlp_weight_only = _ratio_number(2 * mac_mlp_per_layer, mlp_weight_bytes_one_layer)
    i_lm_head_weight_only = _ratio_number(2 * mac_lm_head, weight_bytes_lm_head)
    i_gdn_vs_s_rw = _ratio_number(2 * mac_gdn_per_layer, 2 * s_bytes_per_layer)
    i_attn_core_vs_kv = _ratio_number(
        2 * mac_attn_coeff_per_full_layer, kv_bytes_per_full_layer_per_token
    )

    fusion_selected = {hyp_id: False for hyp_id in FUSION_HYPOTHESIS_IDS}
    hoist_selected = {hyp_id: False for hyp_id in HOIST_HYPOTHESIS_IDS}
    n_stage_instances = sum(STAGE_MULTIPLICITIES)

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
        "stage_kind_ids": list(STAGE_KIND_IDS),
        "n_stage_kinds": len(STAGE_KIND_IDS),
        "stage_multiplicities": list(STAGE_MULTIPLICITIES),
        "stage_node_types": list(STAGE_NODE_TYPES),
        "n_stage_instances": n_stage_instances,
        "serial_stage_kind_order": list(STAGE_KIND_IDS),
        "stage_primary_sequence_ids": list(STAGE_PRIMARY_SEQUENCE_IDS),
        "gated_delta_net_state_sequence_id": "seq_state_s_dense",
        "stage_visibility_in": _ordered_stage_mapping(STAGE_VISIBILITY_IN),
        "stage_visibility_out": _ordered_stage_mapping(STAGE_VISIBILITY_OUT),
        "stage_state_read": _ordered_stage_mapping(STAGE_STATE_READ),
        "stage_state_write": _ordered_stage_mapping(STAGE_STATE_WRITE),
        "language_mixer_state_is_xor": True,
        "ready_constraint_ids": list(READY_CONSTRAINT_IDS),
        "n_ready_constraints": len(READY_CONSTRAINT_IDS),
        "hoist_hypothesis_ids": list(HOIST_HYPOTHESIS_IDS),
        "hoist_selected": hoist_selected,
        "n_hoist_hypotheses": len(HOIST_HYPOTHESIS_IDS),
        "n_hoist_hypotheses_selected": 0,
        "hoist_usefulness_label": "HYPOTHESIS",
        "n_token_presentations": 2,
        "example_T": example_t,
        "decode_T_new": 1,
        "T_is_stored_length_after_append": True,
        "incoming_state_populated": True,
        "primary_includes_mtp": True,
        "mac_C_language": mac_c_language,
        "mac_C_complete": mac_c_complete,
        "mac_A_language": mac_a_language,
        "mac_A_complete": mac_a_complete,
        "mac_decode_complete_at_example_T": mac_decode_complete,
        "mac_full_proj_per_layer": mac_full_proj_per_layer,
        "mac_lin_token_per_layer": mac_lin_token_per_layer,
        "mac_mlp_per_layer": mac_mlp_per_layer,
        "mac_lm_head": mac_lm_head,
        "mac_mtp_fc": mac_mtp_fc,
        "mac_gdn_per_layer": mac_gdn_per_layer,
        "i_mlp_weight_only": i_mlp_weight_only,
        "i_lm_head_weight_only": i_lm_head_weight_only,
        "i_gdn_vs_s_rw": i_gdn_vs_s_rw,
        "i_attn_core_vs_kv": i_attn_core_vs_kv,
        "bottleneck_labels": list(BOTTLENECK_LABELS),
        "n_bottleneck_labels": len(BOTTLENECK_LABELS),
        "weight_bytes_unique_non_embed": weight_bytes_unique_non_embed,
        "weight_gather_bytes_per_row": weight_gather_bytes_per_row,
        "weight_gather_bytes_decode_complete": weight_gather_bytes_decode_complete,
        "weight_unique_plus_gather_decode_complete": (
            weight_unique_plus_gather_decode_complete
        ),
        "weight_boundary_added_unique_bytes": 0,
        "weight_second_w_lm_read_bytes": weight_bytes_lm_head,
        "weight_bytes_lm_head": weight_bytes_lm_head,
        "kv_bytes_per_full_layer_per_token": kv_bytes_per_full_layer_per_token,
        "kv_bytes_all_per_token": kv_bytes_all_per_token,
        "c_bytes_per_layer": c_bytes_per_layer,
        "c_write_bytes_all": c_write_bytes_all,
        "c_read_bytes_all": c_read_bytes_all,
        "s_bytes_per_layer": s_bytes_per_layer,
        "s_bytes_all": s_bytes_all,
        "decode_write_bytes": decode_write_bytes,
        "decode_read_fixed_bytes": decode_read_fixed_bytes,
        "decode_read_kv_bytes_coeff_Tm1": decode_read_kv_bytes_coeff_tm1,
        "decode_read_bytes_at_example_T": decode_read_bytes_at_example_t,
        "storage_kv_bytes_coeff_T": storage_kv_bytes_coeff_t,
        "storage_fixed_bytes": storage_fixed_bytes,
        "storage_bytes_at_example_T": storage_bytes_at_example_t,
        "state_boundary_added_bytes": 0,
        "residual_bytes": residual_bytes,
        "g_bytes": g_bytes,
        "z_bytes": z_bytes,
        "logits_bytes": logits_bytes,
        "n_logit_outputs": n_logit_outputs,
        "n_identity_e_h0_crossings": n_identity_e_h0_crossings,
        "n_h_mid_crossings": n_h_mid_crossings,
        "n_h_interlayer_crossings": n_h_interlayer_crossings,
        "n_fanout_h64_crossings": n_fanout_h64_crossings,
        "n_embed_e_next_crossings": n_embed_e_next_crossings,
        "n_mtp_u_crossings": n_mtp_u_crossings,
        "n_h_mtp_crossings": n_h_mtp_crossings,
        "n_residual_stage_crossings": n_residual_stage_crossings,
        "act_residual_stage_cut_bytes": act_residual_stage_cut_bytes,
        "act_logits_stage_cut_bytes": act_logits_stage_cut_bytes,
        "act_stage_cut_decode_complete_bytes": act_stage_cut_decode_complete_bytes,
        "act_forced_decode_complete_bytes": act_forced_decode_complete_bytes,
        "act_region_cut_decode_complete_bytes": act_region_cut_decode_complete_bytes,
        "act_boundary_added_vs_forced": act_boundary_added_vs_forced,
        "act_boundary_added_vs_region_cut": act_boundary_added_vs_region_cut,
        "act_gz_local_bytes": act_gz_local_bytes,
        "act_h_mid_stage_cut_bytes": act_h_mid_stage_cut_bytes,
        "act_h64_second_consumer_bytes": residual_bytes,
        "act_h64_second_consumer_is_hypothesis": True,
        "consumer_sequence_ids": list(CONSUMER_SEQUENCE_IDS),
        "n_consumer_sequences": len(CONSUMER_SEQUENCE_IDS),
        "fusion_hypothesis_ids": list(FUSION_HYPOTHESIS_IDS),
        "n_fusion_hypotheses": len(FUSION_HYPOTHESIS_IDS),
        "fusion_selected": fusion_selected,
        "n_fusion_hypotheses_selected": 0,
        "fusion_usefulness_label": "HYPOTHESIS",
        "sync_edge_ids": list(SYNC_EDGE_IDS),
        "n_sync_edges": len(SYNC_EDGE_IDS),
        "decode_prefill_share_graph": True,
        "residual_add_inside_mixer": True,
        "residual_add_inside_mlp": True,
        "residual_input_live_until_add": True,
        "rms_inside_consumer": True,
        "live_across_are_internal": True,
        "hardware_independent": True,
        "cuda_mapping_deferred": True,
        "thread_geometry_absent": True,
        "prefill_schedule_deferred": True,
        "layout_selected": False,
        "fusion_winner_selected": False,
        "ideal_byte_sequence_selected": False,
        "artifact_boundary_selected": False,
        "schedule_serial_selected": True,
        "gdn_primary_is_recurrent_eq_17": True,
        "chunkwise_not_zero_s_traffic": True,
        "state_write_not_optional": True,
        "fanout_h64_cannot_hide_from_one_consumer": True,
        "weight_unique_counted_once": True,
        "weight_second_w_lm_read_is_hypothesis": True,
        "mixer_xor_by_layer_types": True,
        "activation_dtype_decided": False,
        "vision_interface_is_not_a_node": True,
        "mtp_omission_is_algebraic_equivalent": True,
        "ledger_open_question_boundary_traffic_closed": True,
        "diagram_ids": list(DIAGRAM_REQUIRED_IDS),
        "n_diagrams": 1,
        "canonical_sentence_logical": CANONICAL_SENTENCE_LOGICAL,
        "canonical_sentence_schedule": CANONICAL_SENTENCE_SCHEDULE,
        "canonical_sentence_traffic": CANONICAL_SENTENCE_TRAFFIC,
        "canonical_sentence_open_question": CANONICAL_SENTENCE_OPEN_QUESTION,
    }
    _assert_identities(summary, text)
    return summary


def _assert_identities(summary: dict, text: dict) -> None:
    """Assert dossier identities on a live decode-plan object.

    Args:
        summary: Object produced by :func:`instantiate_decode_plan`.
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
    assert summary["full_attention_indices"] == list(LOCKED_FULL_ATTENTION_INDICES)
    assert text.get("dtype") == "bfloat16"
    assert text.get("mamba_ssm_dtype") == "float32"
    assert summary["n_stage_kinds"] == 9
    assert summary["n_stage_instances"] == 135
    assert sum(summary["stage_multiplicities"]) == 135
    assert summary["stage_multiplicities"] == list(STAGE_MULTIPLICITIES)
    assert summary["n_node_instances_complete"] == 135
    assert summary["n_node_instances_language"] == 130
    assert summary["n_embed_instances"] == 2
    assert summary["n_gated_attn_instances"] == 17
    assert summary["n_gated_delta_net_instances"] == 48
    assert summary["n_mlp_instances"] == 65
    assert summary["n_lm_head_instances"] == 2
    assert summary["n_mtp_mix_instances"] == 1
    assert summary["n_ready_constraints"] == 10
    assert summary["n_hoist_hypotheses"] == 5
    assert summary["n_hoist_hypotheses_selected"] == 0
    assert summary["n_consumer_sequences"] == 7
    assert summary["n_fusion_hypotheses"] == 22
    assert summary["n_fusion_hypotheses_selected"] == 0
    assert summary["n_sync_edges"] == 14
    assert summary["n_diagrams"] == 1
    assert summary["n_token_presentations"] == 2
    assert summary["n_residual_stage_crossings"] == 133
    assert summary["n_h_mid_crossings"] == 65
    assert summary["n_h_interlayer_crossings"] == 63
    assert summary["residual_bytes"] == 10240
    assert summary["act_residual_stage_cut_bytes"] == 1361920
    assert summary["logits_bytes"] == 496640
    assert summary["act_logits_stage_cut_bytes"] == 993280
    assert summary["act_stage_cut_decode_complete_bytes"] == 2355200
    assert summary["act_forced_decode_complete_bytes"] == 3123200
    assert summary["act_region_cut_decode_complete_bytes"] == 5847040
    assert summary["act_boundary_added_vs_forced"] == -768000
    assert summary["act_boundary_added_vs_region_cut"] == -3491840
    assert summary["act_gz_local_bytes"] == 798720
    assert summary["act_h_mid_stage_cut_bytes"] == 665600
    assert summary["weight_bytes_unique_non_embed"] == 52098598912
    assert summary["weight_gather_bytes_decode_complete"] == 20480
    assert summary["weight_unique_plus_gather_decode_complete"] == 52098619392
    assert summary["weight_boundary_added_unique_bytes"] == 0
    assert summary["decode_write_bytes"] == 152047616
    assert summary["state_boundary_added_bytes"] == 0
    assert summary["mac_C_complete"] == 27433238528
    assert summary["mac_A_complete"] == 208896
    assert summary["mac_decode_complete_at_example_T"] == [27433447424, 28288876544]
    for index, t_value in enumerate(summary["example_T"]):
        assert summary["mac_decode_complete_at_example_T"][index] == (
            summary["mac_C_complete"] + summary["mac_A_complete"] * t_value
        )
    assert summary["i_mlp_weight_only"] == 1
    assert summary["i_lm_head_weight_only"] == 1
    assert math.isclose(summary["i_gdn_vs_s_rw"], 0.75)
    assert summary["i_attn_core_vs_kv"] == 6
    assert summary["schedule_serial_selected"] is True
    assert summary["fusion_winner_selected"] is False
    assert summary["ideal_byte_sequence_selected"] is False
    assert summary["thread_geometry_absent"] is True
    assert summary["hardware_independent"] is True
    assert summary["incoming_state_populated"] is True
    assert summary["prefill_schedule_deferred"] is True
    assert summary["weight_unique_counted_once"] is True
    assert summary["weight_second_w_lm_read_is_hypothesis"] is True
    assert summary["live_across_are_internal"] is True
    assert summary["state_write_not_optional"] is True
    assert summary["fanout_h64_cannot_hide_from_one_consumer"] is True
    assert summary["chunkwise_not_zero_s_traffic"] is True
    assert summary["ledger_open_question_boundary_traffic_closed"] is True
    assert all(value is False for value in summary["fusion_selected"].values())
    assert all(value is False for value in summary["hoist_selected"].values())
    assert summary["stage_kind_ids"] == list(STAGE_KIND_IDS)
    assert summary["fusion_hypothesis_ids"] == list(FUSION_HYPOTHESIS_IDS)
    assert summary["consumer_sequence_ids"] == list(CONSUMER_SEQUENCE_IDS)
    for mapping_key in ("stage_visibility_in", "stage_visibility_out"):
        for names in summary[mapping_key].values():
            assert "g" not in names
            assert "z" not in names
    assert "language_mixer" in summary["stage_kind_ids"]
    assert "seq_lm_head_full" in summary["stage_primary_sequence_ids"]
    assert summary["n_residual_stage_crossings"] == (
        summary["n_identity_e_h0_crossings"]
        + summary["n_h_mid_crossings"]
        + summary["n_h_interlayer_crossings"]
        + summary["n_fanout_h64_crossings"]
        + summary["n_embed_e_next_crossings"]
        + summary["n_mtp_u_crossings"]
        + summary["n_h_mtp_crossings"]
    )
    assert summary["act_residual_stage_cut_bytes"] == (
        summary["n_residual_stage_crossings"] * summary["residual_bytes"]
    )
    assert summary["act_stage_cut_decode_complete_bytes"] == (
        summary["act_residual_stage_cut_bytes"] + summary["act_logits_stage_cut_bytes"]
    )
    assert summary["serial_stage_kind_order"] == summary["stage_kind_ids"]
    assert summary["n_stage_instances"] == summary["n_node_instances_complete"]


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
    """Compare documented JSON against a live decode-plan object.

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


def check_decode_plan(live: dict, decode_plan_path: Path) -> None:
    """Check a decode-plan markdown file against a live summary.

    Args:
        live: Decode-plan summary object from sitting config.
        decode_plan_path: Path to ``decode-plan.md``.

    Raises:
        DecodePlanMismatch: On heading, JSON, diagram, or token mismatches.
        OSError: If the file cannot be read.
    """

    text = decode_plan_path.read_text(encoding="utf-8")
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
    if CANONICAL_SENTENCE_SCHEDULE not in text:
        differences.append("canonical_sentence_schedule not present verbatim")
    if CANONICAL_SENTENCE_TRAFFIC not in text:
        differences.append("canonical_sentence_traffic not present verbatim")
    if CANONICAL_SENTENCE_OPEN_QUESTION not in text:
        differences.append("canonical_sentence_open_question not present verbatim")

    id_groups = (
        STAGE_KIND_IDS,
        CONSUMER_SEQUENCE_IDS,
        FUSION_HYPOTHESIS_IDS,
        HOIST_HYPOTHESIS_IDS,
        READY_CONSTRAINT_IDS,
        SYNC_EDGE_IDS,
        NODE_TYPE_IDS,
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
        raise DecodePlanMismatch(differences)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the decode-plan checker."""

    parser = argparse.ArgumentParser(
        description=(
            "Instantiate Qwen3.8-27B language+MTP decode-plan summary "
            "from text_config and check docs/architecture/decode-plan.md."
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
        help="Write decode-plan summary JSON to stdout.",
    )
    parser.add_argument(
        "--decode-plan",
        type=Path,
        default=None,
        help="Markdown path whose first json fence must match live summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the decode-plan checker CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch, 2 missing
        config.
    """

    args = _parse_args(argv)
    try:
        config = load_config(args.config)
        summary = instantiate_decode_plan(config)
        if args.decode_plan is not None:
            if not args.decode_plan.is_file():
                print(
                    f"decode-plan file not found: {args.decode_plan}",
                    file=sys.stderr,
                )
                return 1
            check_decode_plan(summary, args.decode_plan)
    except MissingConfig as exc:
        print(str(exc.path), file=sys.stderr)
        return 2
    except DecodePlanMismatch as exc:
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

    if args.json or args.decode_plan is None:
        sys.stdout.write(dumps_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
