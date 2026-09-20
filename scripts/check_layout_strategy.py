#!/usr/bin/env python3
"""Check Qwen3.8-27B layout-strategy contracts against sitting text_config.

Reads ``text_config`` from a Transformers ``config.json`` (no safetensor
payloads) and either prints the layout-strategy summary object or checks
that ``docs/architecture/layout-strategy.md`` matches it.
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

CANONICAL_SENTENCE_CANDIDATES = (
    "Candidate layouts in this document are consumer-driven orderings and "
    "tiles, not selected winners and not CUDA mappings."
)

CANONICAL_SENTENCE_CONVERSION = (
    "Alignment and conversion in this document are named capabilities and "
    "hypotheses, not a selected pack-to-layout pipeline."
)

CANONICAL_SENTENCE_OPEN_QUESTION = (
    "Which planned parallel decompositions justify each candidate ordering "
    "and tile remains open until CUDA analysis."
)

CANONICAL_SENTENCE_OPTIMALITY = (
    "This document makes no optimality claim before CUDA analysis."
)

CANONICAL_SENTENCE_JUSTIFICATION = (
    "Listing a justification hypothesis is not selecting a parallel "
    "decomposition and does not justify a candidate ordering or tile."
)

LAYOUT_OBJECT_IDS: tuple[str, ...] = (
    "gather_row",
    "dense_gemm",
    "depthwise_conv",
    "vector_param",
    "state_kv",
    "state_c",
    "state_s",
)

LAYOUT_OBJECT_RANKS: tuple[int, ...] = (2, 2, 2, 1, 3, 2, 3)

LAYOUT_OBJECT_AXES: dict[str, tuple[str, ...]] = {
    "gather_row": ("vocab", "hidden"),
    "dense_gemm": ("d_out", "d_in"),
    "depthwise_conv": ("channel", "tap"),
    "vector_param": ("width",),
    "state_kv": ("n_kv", "T", "d_h"),
    "state_c": ("delay", "channel"),
    "state_s": ("n_v", "d_k", "d_v"),
}

LAYOUT_OBJECT_PRIMARY_SEQUENCE_IDS: tuple[str | None, ...] = (
    "seq_gather_row",
    "seq_gemm_codes_then_scales",
    None,
    None,
    None,
    None,
    "seq_state_s_dense",
)

ACCESS_CLASS_IDS: tuple[str, ...] = LAYOUT_OBJECT_IDS

POLICY_FAMILIES: tuple[str, ...] = (
    "norm_gamma",
    "gdn_time_param",
    "gdn_gate_proj",
    "conv1d",
    "linear_large_proj",
    "attn_qkv",
    "attn_out",
    "mlp_up_gate",
    "mlp_down",
    "embed_table",
    "lm_head",
    "mtp_fc",
    "vision_deferred",
    "state_kv",
    "state_c",
    "state_s",
)

FAMILY_ACCESS_CLASS: dict[str, str | None] = {
    "norm_gamma": "vector_param",
    "gdn_time_param": "vector_param",
    "gdn_gate_proj": "dense_gemm",
    "conv1d": "depthwise_conv",
    "linear_large_proj": "dense_gemm",
    "attn_qkv": "dense_gemm",
    "attn_out": "dense_gemm",
    "mlp_up_gate": "dense_gemm",
    "mlp_down": "dense_gemm",
    "embed_table": "gather_row",
    "lm_head": "dense_gemm",
    "mtp_fc": "dense_gemm",
    "vision_deferred": None,
    "state_kv": "state_kv",
    "state_c": "state_c",
    "state_s": "state_s",
}

CONSUMER_SEQUENCE_IDS: tuple[str, ...] = (
    "seq_gemm_codes_then_scales",
    "seq_gemm_interleaved_group",
    "seq_gather_row",
    "seq_lm_head_full",
    "seq_outlier_extra",
    "seq_state_s_dense",
    "seq_specialized_tile",
)

ANALYSIS_DIMENSION_IDS: tuple[str, ...] = (
    "logical_dimensions",
    "consumers",
    "access",
    "tiling",
    "alignment",
    "conversion",
)

PERSISTENT_STATE_COVERAGE_IDS: tuple[str, ...] = (
    "gdn",
    "convolution",
    "kv",
)

CONSUMER_MODE_IDS: tuple[str, ...] = (
    "decode_gemv",
    "prefill_gemm",
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

BOTTLENECK_LABELS: tuple[str, ...] = (
    "weight_memory",
    "vocab_memory",
    "state_memory",
    "kv_memory",
    "quadratic_attn",
    "compute",
)

ORDERING_IDS: tuple[str, ...] = (
    "ord_gemm_out_major",
    "ord_gemm_in_major",
    "ord_embed_vocab_major",
    "ord_embed_hidden_major",
    "ord_conv_channel_tap",
    "ord_conv_tap_channel",
    "ord_vec_width",
    "ord_kv_n_t_dh",
    "ord_kv_n_dh_t",
    "ord_kv_t_n_dh",
    "ord_c_delay_channel",
    "ord_c_channel_delay",
    "ord_s_n_dk_dv",
    "ord_s_n_dv_dk",
)

ORDERING_OBJECT_IDS: tuple[str, ...] = (
    "dense_gemm",
    "dense_gemm",
    "gather_row",
    "gather_row",
    "depthwise_conv",
    "depthwise_conv",
    "vector_param",
    "state_kv",
    "state_kv",
    "state_kv",
    "state_c",
    "state_c",
    "state_s",
    "state_s",
)

TILE_FAMILY_IDS: tuple[str, ...] = (
    "tile_none",
    "tile_2d_mn",
    "tile_1d_row",
    "tile_conv_channel",
    "tile_kv_t",
    "tile_kv_dh",
    "tile_s_head",
    "tile_s_block",
    "tile_mma_shaped",
)

TILE_FAMILY_OBJECTS: dict[str, tuple[str, ...]] = {
    "tile_none": LAYOUT_OBJECT_IDS,
    "tile_2d_mn": ("dense_gemm",),
    "tile_1d_row": ("gather_row",),
    "tile_conv_channel": ("depthwise_conv", "state_c"),
    "tile_kv_t": ("state_kv",),
    "tile_kv_dh": ("state_kv",),
    "tile_s_head": ("state_s",),
    "tile_s_block": ("state_s",),
    "tile_mma_shaped": ("dense_gemm",),
}

PARALLEL_DECOMPOSITION_IDS: tuple[str, ...] = (
    "par_gemm_d_out",
    "par_gemm_d_in",
    "par_gemm_T",
    "par_attn_head",
    "par_attn_T",
    "par_gdn_head",
    "par_conv_channel",
    "par_kv_head",
    "par_embed_row",
)

CONVERSION_HYPOTHESIS_IDS: tuple[str, ...] = (
    "conv_compile_pack",
    "conv_load_repack",
    "conv_inkernel_unpack",
    "conv_dual_view",
)

JUSTIFICATION_HYPOTHESIS_IDS: tuple[str, ...] = (
    "j_gemm_out_major_par_d_out",
    "j_gemm_in_major_par_d_in",
    "j_gemm_tile_2d_par_T",
    "j_embed_vocab_major_par_row",
    "j_conv_channel_tap_par_channel",
    "j_kv_n_t_dh_par_head",
    "j_kv_n_dh_t_par_T",
    "j_s_n_dk_dv_par_head",
    "j_s_n_dv_dk_par_head",
    "j_mma_shaped_unselected_par",
)

LAYOUT_RISK_IDS: tuple[str, ...] = (
    "l_gemm_vs_gather",
    "l_decode_vs_prefill_view",
    "l_s_transpose",
    "l_kv_append",
    "l_conv_fir_stride",
    "l_convert_cost",
    "l_int3_grain",
    "l_mma_sku_unknown",
)

LAYOUT_RISK_SEVERITIES: tuple[str, ...] = (
    "medium",
    "medium",
    "medium",
    "medium",
    "medium",
    "high",
    "medium",
    "low",
)

LAYOUT_HIGH_IDS: tuple[str, ...] = ("l_convert_cost",)

LAYOUT_MEDIUM_IDS: tuple[str, ...] = (
    "l_gemm_vs_gather",
    "l_decode_vs_prefill_view",
    "l_s_transpose",
    "l_kv_append",
    "l_conv_fir_stride",
    "l_int3_grain",
)

LAYOUT_LOW_IDS: tuple[str, ...] = ("l_mma_sku_unknown",)

DIAGRAM_REQUIRED_IDS: tuple[str, ...] = (
    "gemm",
    "gather",
    "conv",
    "kv",
    "gdn",
    "c_state",
    "portable",
    "specialized",
    "open",
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
TILE_EXTENT_CANDIDATES: tuple[int, ...] = (16, 32, 64, 128, 256)
ALIGNMENT_GRAIN_CANDIDATES: tuple[int, ...] = (1, 16, 32, 128, 256)
SCALE_STORAGE_BYTES_CANDIDATES: tuple[int, ...] = (2, 4)
SCALE_PLACEMENT_CANDIDATES: tuple[str, ...] = (
    "sidecar_array",
    "interleaved_group",
)
CODE_BIT_ORDER_CANDIDATES: tuple[str, ...] = ("lsb_first", "msb_first")

BYTES_BF16 = 2
BYTES_F32 = 4
SCALE_STORAGE_ILLUSTRATION_BYTES = 2
DUAL_VIEW_INT4_G128_UNIQUE_NON_EMBED_BYTES = 26863340064
I_MLP_WEIGHT_ONLY = 1
I_LM_HEAD_WEIGHT_ONLY = 1
I_GDN_VS_S_RW = 0.75
I_ATTN_CORE_VS_KV = 6
DEFAULT_LAYOUT_STRATEGY = Path("docs/architecture/layout-strategy.md")

LOCKED_DOCUMENT_NUMBERS: tuple[int | float, ...] = (
    5120,
    17408,
    248320,
    10240,
    6144,
    256,
    128,
    48,
    24,
    4,
    3,
    6,
    16,
    17,
    64,
    4096,
    69632,
    61440,
    2949120,
    3145728,
    150994944,
    75497472,
    2542796800,
    2562,
    2560,
    3840,
    12,
    8,
    26863340064,
    0.75,
)

FORBIDDEN_WINNER_PHRASES: tuple[str, ...] = (
    "optimal layout",
    "best tile",
    "winning layout",
    "selected tile is",
    "selected ordering is",
    "this layout is optimal",
    "CUDA occupancy selects",
    "should use this tile",
    "recommend this layout",
    "parallel decomposition justifies",
    "MMA shape is required",
    "ideal byte sequence is",
    "artifact boundary is",
    "selected winner",
    "thread block",
    "warp shuffle",
    "Quartz layout",
    "llama.cpp layout",
    "GGUF is the layout",
    "selected distinct views",
    "prefill requires a distinct view",
)

ALLOWED_EXCEPTION_PHRASES: tuple[str, ...] = (
    "not a selected winner",
    "not selected winners",
    "not a CUDA mapping",
    "makes no optimality claim",
    "does not justify a candidate",
)

REQUIRED_HEADINGS: tuple[str, ...] = (
    "Authority",
    "Layout convention",
    "Logical dimensions",
    "Consumers",
    "Access",
    "Candidate orderings and tiling",
    "Alignment and conversion",
    "GDN, convolution, and KV persistent state",
    "Parallel decompositions",
    "Work citations and non-decisions",
    "Deferred vision",
    "Machine-checkable summary JSON",
)

REQUIRED_H3_HEADINGS: tuple[str, ...] = (
    "### GDN",
    "### Convolution",
    "### KV persistent state",
)

JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)
MERMAID_FENCE_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
HEADING_RE = re.compile(r"^## (?!#)(.+)$", re.MULTILINE)
FORBIDDEN_RE = re.compile(r"TBD|TODO|\?\?\?")
UNKNOWN_RE = re.compile(r"UNKNOWN")

SUBSTRING_ID_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("layout object", LAYOUT_OBJECT_IDS),
    ("ordering", ORDERING_IDS),
    ("tile family", TILE_FAMILY_IDS),
    ("parallel decomposition", PARALLEL_DECOMPOSITION_IDS),
    ("conversion hypothesis", CONVERSION_HYPOTHESIS_IDS),
    ("justification hypothesis", JUSTIFICATION_HYPOTHESIS_IDS),
    ("layout risk", LAYOUT_RISK_IDS),
    ("analysis dimension", ANALYSIS_DIMENSION_IDS),
    ("persistent state coverage", PERSISTENT_STATE_COVERAGE_IDS),
    ("consumer sequence", CONSUMER_SEQUENCE_IDS),
    ("access class", ACCESS_CLASS_IDS),
    ("policy family", POLICY_FAMILIES),
    ("stage kind", STAGE_KIND_IDS),
    ("bottleneck label", BOTTLENECK_LABELS),
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
    "head_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "g_qa",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_num_value_heads",
    "linear_conv_kernel_dim",
    "linear_conv_delay",
    "d_qkv",
    "bytes_bf16",
    "bytes_f32",
    "kv_bytes_per_full_layer_per_token",
    "kv_bytes_all_per_token",
    "c_bytes_per_layer",
    "c_bytes_all",
    "s_bytes_per_layer",
    "s_f32_bytes",
    "s_bf16_bytes",
    "embed_gather_bf16_bytes",
    "embed_gather_int4_row_bytes",
    "weight_bytes_lm_head",
    "int4_g32_group_payload_bytes",
    "int3_g32_group_payload_bytes",
    "int2_g32_group_payload_bytes",
    "int4_row_hidden_payload_bytes",
    "int6_row_hidden_payload_bytes",
    "dual_view_int4_g128_unique_non_embed_bytes",
    "tile_extent_candidates",
    "n_tile_extent_candidates",
    "alignment_grain_candidates",
    "scale_storage_bytes_candidates",
    "scale_placement_candidates",
    "code_bit_order_candidates",
    "tile_extent_divides_hidden",
    "tile_extent_divides_intermediate",
    "tile_extent_divides_d_qkv",
    "tile_extent_divides_head_dim",
    "tile_extent_divides_dk",
    "n_v_divides_none_of_extents_as_head_count",
    "layout_object_ids",
    "n_layout_objects",
    "layout_object_ranks",
    "layout_object_axes",
    "layout_object_primary_sequence_ids",
    "n_layout_object_primary_sequences",
    "lm_head_sequence_id",
    "gated_delta_net_state_sequence_id",
    "access_class_ids",
    "n_access_classes",
    "policy_families",
    "family_access_class",
    "family_access_class_values",
    "family_layout_object",
    "consumer_sequence_ids",
    "n_consumer_sequences",
    "analysis_dimension_ids",
    "n_analysis_dimensions",
    "persistent_state_coverage_ids",
    "n_persistent_state_coverages",
    "consumer_mode_ids",
    "n_consumer_modes",
    "n_consumer_modes_selected",
    "stage_kind_ids",
    "n_stage_kinds",
    "bottleneck_labels",
    "n_bottleneck_labels",
    "i_mlp_weight_only",
    "i_lm_head_weight_only",
    "i_gdn_vs_s_rw",
    "i_attn_core_vs_kv",
    "ordering_ids",
    "ordering_object_ids",
    "n_orderings",
    "n_orderings_selected",
    "ordering_usefulness_label",
    "tile_family_ids",
    "n_tile_families",
    "tile_family_objects",
    "tile_size_selected",
    "mma_tile_extents_selected",
    "tile_usefulness_label",
    "parallel_decomposition_ids",
    "n_parallel_decompositions",
    "n_parallel_decompositions_selected",
    "conversion_hypothesis_ids",
    "n_conversion_hypotheses",
    "n_conversion_hypotheses_selected",
    "conversion_usefulness_label",
    "justification_hypothesis_ids",
    "n_justification_hypotheses",
    "n_justification_hypotheses_selected",
    "justification_usefulness_label",
    "layout_risk_ids",
    "layout_risk_severities",
    "layout_high_ids",
    "layout_medium_ids",
    "layout_low_ids",
    "n_layout_risks",
    "n_layout_high",
    "n_layout_medium",
    "n_layout_low",
    "example_T",
    "T_is_stored_length_after_append",
    "primary_includes_mtp",
    "decode_prefill_share_artifact",
    "decode_prefill_share_graph",
    "decode_prefill_distinct_views_selected",
    "activations_in_layout_scope",
    "hardware_independent",
    "cuda_mapping_deferred",
    "thread_geometry_absent",
    "layout_winner_selected",
    "ordering_selected",
    "alignment_grain_selected",
    "specialized_alignment_grain_selected",
    "specialized_view_may_constrain_alignment",
    "conversion_pipeline_selected",
    "parallel_decomposition_selected",
    "ideal_byte_sequence_selected",
    "artifact_boundary_selected",
    "seq_specialized_tile_candidates_named",
    "tile_layout_deferred_to_task15",
    "ledger_open_question_parallel_decomposition_closed",
    "parallel_decomposition_justifies_layout_selected",
    "optimality_claim_absent",
    "gdn_primary_is_recurrent_eq_17",
    "chunkwise_not_zero_s_traffic",
    "paper_s_transpose_same_map",
    "state_write_not_optional",
    "kv_rope_baked_into_k",
    "gqa_repeat_not_stored",
    "conv_z_does_not_enter_conv",
    "portable_payload_little_endian",
    "gguf_is_not_the_runtime_format",
    "safetensors_is_source_not_runtime",
    "embed_lm_head_tied",
    "lm_head_uses_dense_gemm_object",
    "embed_uses_gather_row_object",
    "weight_unique_counted_once",
    "analyzes_logical_dimensions",
    "analyzes_consumers",
    "analyzes_access",
    "analyzes_tiling",
    "analyzes_alignment",
    "analyzes_conversion",
    "includes_gdn",
    "includes_convolution",
    "includes_kv",
    "vision_interface_is_not_a_node",
    "payloads_restreamed",
    "fusion_winner_selected",
    "state_payload_in_artifact_selected",
    "diagram_ids",
    "n_diagrams",
    "canonical_sentence_logical",
    "canonical_sentence_candidates",
    "canonical_sentence_conversion",
    "canonical_sentence_open_question",
    "canonical_sentence_optimality",
    "canonical_sentence_justification",
)


class MissingConfig(Exception):
    """Raised when the config path is absent or unreadable."""

    def __init__(self, path: Path) -> None:
        super().__init__(str(path))
        self.path = path


class LayoutStrategyMismatch(Exception):
    """Raised when the layout-strategy markdown fails a content check."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def dumps_summary(summary: dict) -> str:
    """Pretty-print the layout-strategy summary object.

    Args:
        summary: Object produced by :func:`instantiate_layout_strategy`.

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


def _payload_pack_bytes(n_elements: int, code_bits: int) -> int:
    """Return ceil(n*b/8) packed payload bytes at tensor grain.

    Args:
        n_elements: Element count n.
        code_bits: Integer code width b.

    Returns:
        Packed payload byte count.

    Raises:
        AssertionError: If n or b is not positive.
    """

    assert n_elements > 0 and code_bits > 0, "n and b must be positive"
    return math.ceil(n_elements * code_bits / 8)


def _divides(width: int, extent: int) -> bool:
    """Return whether ``extent`` divides ``width``.

    Args:
        width: Instantiated axis width.
        extent: Tile-extent candidate.

    Returns:
        True when ``width % extent == 0``.
    """

    return extent > 0 and width % extent == 0


def _ordered_str_mapping(
    keys: tuple[str, ...], values: dict[str, tuple[str, ...] | str | None]
) -> dict[str, list[str] | str | None]:
    """Copy ``values`` in ``keys`` order, converting tuples to lists.

    Args:
        keys: Required key order.
        values: Mapping keyed by those ids.

    Returns:
        Ordered mapping with list values where the source used tuples.
    """

    ordered: dict[str, list[str] | str | None] = {}
    for key in keys:
        value = values[key]
        if isinstance(value, tuple):
            ordered[key] = list(value)
        else:
            ordered[key] = value
    return ordered


def instantiate_layout_strategy(config: dict) -> dict:
    """Compute the TASK-15 layout-strategy summary from ``config``.

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
    num_attention_heads = _require_int(text, "num_attention_heads")
    num_key_value_heads = _require_int(text, "num_key_value_heads")
    head_dim = _require_int(text, "head_dim")
    linear_num_key_heads = _require_int(text, "linear_num_key_heads")
    linear_num_value_heads = _require_int(text, "linear_num_value_heads")
    linear_key_head_dim = _require_int(text, "linear_key_head_dim")
    linear_value_head_dim = _require_int(text, "linear_value_head_dim")
    linear_conv_kernel_dim = _require_int(text, "linear_conv_kernel_dim")
    n_mtp_blocks = _require_int(text, "mtp_num_hidden_layers")
    full_attention_interval = _require_int(text, "full_attention_interval")
    layer_types = text.get("layer_types")
    assert isinstance(layer_types, list), "text_config.layer_types is not a list"
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
        f"layer_types full indices {full_from_types} != interval "
        f"{full_from_interval}"
    )
    n_full_layers = len(full_from_types)
    n_linear_layers = len(linear_from_types)
    n_full_layers_with_kv = n_full_layers + n_mtp_blocks
    assert num_attention_heads % num_key_value_heads == 0, (
        "num_attention_heads is not divisible by num_key_value_heads"
    )
    g_qa = num_attention_heads // num_key_value_heads
    linear_conv_delay = linear_conv_kernel_dim - 1
    d_qkv = (
        2 * linear_num_key_heads + linear_num_value_heads
    ) * linear_key_head_dim

    kv_bytes_per_full_layer_per_token = (
        2 * num_key_value_heads * head_dim * BYTES_BF16
    )
    kv_bytes_all_per_token = (
        kv_bytes_per_full_layer_per_token * n_full_layers_with_kv
    )
    c_bytes_per_layer = linear_conv_delay * d_qkv * BYTES_BF16
    c_bytes_all = c_bytes_per_layer * n_linear_layers
    s_bytes_per_layer = (
        linear_num_value_heads
        * linear_key_head_dim
        * linear_value_head_dim
        * BYTES_F32
    )
    s_f32_bytes = s_bytes_per_layer * n_linear_layers
    s_bf16_bytes = (
        n_linear_layers
        * linear_num_value_heads
        * linear_key_head_dim
        * linear_value_head_dim
        * BYTES_BF16
    )
    embed_gather_bf16_bytes = hidden_size * BYTES_BF16
    int4_g32_group_payload_bytes = _payload_pack_bytes(32, 4)
    int3_g32_group_payload_bytes = _payload_pack_bytes(32, 3)
    int2_g32_group_payload_bytes = _payload_pack_bytes(32, 2)
    int4_row_hidden_payload_bytes = _payload_pack_bytes(hidden_size, 4)
    int6_row_hidden_payload_bytes = _payload_pack_bytes(hidden_size, 6)
    embed_gather_int4_row_bytes = (
        int4_row_hidden_payload_bytes + SCALE_STORAGE_ILLUSTRATION_BYTES
    )
    weight_bytes_lm_head = vocab_size * hidden_size * BYTES_BF16

    tile_extent_divides_hidden = [
        _divides(hidden_size, extent) for extent in TILE_EXTENT_CANDIDATES
    ]
    tile_extent_divides_intermediate = [
        _divides(intermediate_size, extent) for extent in TILE_EXTENT_CANDIDATES
    ]
    tile_extent_divides_d_qkv = [
        _divides(d_qkv, extent) for extent in TILE_EXTENT_CANDIDATES
    ]
    tile_extent_divides_head_dim = [
        _divides(head_dim, extent) for extent in TILE_EXTENT_CANDIDATES
    ]
    tile_extent_divides_dk = [
        _divides(linear_key_head_dim, extent) for extent in TILE_EXTENT_CANDIDATES
    ]
    n_v_divides_none_of_extents_as_head_count = (
        linear_num_value_heads not in TILE_EXTENT_CANDIDATES
    )

    family_access_class = {
        family_id: FAMILY_ACCESS_CLASS[family_id] for family_id in POLICY_FAMILIES
    }
    family_access_class_values = [
        FAMILY_ACCESS_CLASS[family_id] for family_id in POLICY_FAMILIES
    ]
    layout_object_axes = _ordered_str_mapping(LAYOUT_OBJECT_IDS, LAYOUT_OBJECT_AXES)
    tile_family_objects = _ordered_str_mapping(TILE_FAMILY_IDS, TILE_FAMILY_OBJECTS)
    n_layout_object_primary_sequences = sum(
        1 for seq_id in LAYOUT_OBJECT_PRIMARY_SEQUENCE_IDS if seq_id is not None
    )

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
        "head_dim": head_dim,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "g_qa": g_qa,
        "linear_key_head_dim": linear_key_head_dim,
        "linear_value_head_dim": linear_value_head_dim,
        "linear_num_value_heads": linear_num_value_heads,
        "linear_conv_kernel_dim": linear_conv_kernel_dim,
        "linear_conv_delay": linear_conv_delay,
        "d_qkv": d_qkv,
        "bytes_bf16": BYTES_BF16,
        "bytes_f32": BYTES_F32,
        "kv_bytes_per_full_layer_per_token": kv_bytes_per_full_layer_per_token,
        "kv_bytes_all_per_token": kv_bytes_all_per_token,
        "c_bytes_per_layer": c_bytes_per_layer,
        "c_bytes_all": c_bytes_all,
        "s_bytes_per_layer": s_bytes_per_layer,
        "s_f32_bytes": s_f32_bytes,
        "s_bf16_bytes": s_bf16_bytes,
        "embed_gather_bf16_bytes": embed_gather_bf16_bytes,
        "embed_gather_int4_row_bytes": embed_gather_int4_row_bytes,
        "weight_bytes_lm_head": weight_bytes_lm_head,
        "int4_g32_group_payload_bytes": int4_g32_group_payload_bytes,
        "int3_g32_group_payload_bytes": int3_g32_group_payload_bytes,
        "int2_g32_group_payload_bytes": int2_g32_group_payload_bytes,
        "int4_row_hidden_payload_bytes": int4_row_hidden_payload_bytes,
        "int6_row_hidden_payload_bytes": int6_row_hidden_payload_bytes,
        "dual_view_int4_g128_unique_non_embed_bytes": (
            DUAL_VIEW_INT4_G128_UNIQUE_NON_EMBED_BYTES
        ),
        "tile_extent_candidates": list(TILE_EXTENT_CANDIDATES),
        "n_tile_extent_candidates": len(TILE_EXTENT_CANDIDATES),
        "alignment_grain_candidates": list(ALIGNMENT_GRAIN_CANDIDATES),
        "scale_storage_bytes_candidates": list(SCALE_STORAGE_BYTES_CANDIDATES),
        "scale_placement_candidates": list(SCALE_PLACEMENT_CANDIDATES),
        "code_bit_order_candidates": list(CODE_BIT_ORDER_CANDIDATES),
        "tile_extent_divides_hidden": tile_extent_divides_hidden,
        "tile_extent_divides_intermediate": tile_extent_divides_intermediate,
        "tile_extent_divides_d_qkv": tile_extent_divides_d_qkv,
        "tile_extent_divides_head_dim": tile_extent_divides_head_dim,
        "tile_extent_divides_dk": tile_extent_divides_dk,
        "n_v_divides_none_of_extents_as_head_count": (
            n_v_divides_none_of_extents_as_head_count
        ),
        "layout_object_ids": list(LAYOUT_OBJECT_IDS),
        "n_layout_objects": len(LAYOUT_OBJECT_IDS),
        "layout_object_ranks": list(LAYOUT_OBJECT_RANKS),
        "layout_object_axes": layout_object_axes,
        "layout_object_primary_sequence_ids": list(
            LAYOUT_OBJECT_PRIMARY_SEQUENCE_IDS
        ),
        "n_layout_object_primary_sequences": n_layout_object_primary_sequences,
        "lm_head_sequence_id": "seq_lm_head_full",
        "gated_delta_net_state_sequence_id": "seq_state_s_dense",
        "access_class_ids": list(ACCESS_CLASS_IDS),
        "n_access_classes": len(ACCESS_CLASS_IDS),
        "policy_families": list(POLICY_FAMILIES),
        "family_access_class": family_access_class,
        "family_access_class_values": family_access_class_values,
        "family_layout_object": dict(family_access_class),
        "consumer_sequence_ids": list(CONSUMER_SEQUENCE_IDS),
        "n_consumer_sequences": len(CONSUMER_SEQUENCE_IDS),
        "analysis_dimension_ids": list(ANALYSIS_DIMENSION_IDS),
        "n_analysis_dimensions": len(ANALYSIS_DIMENSION_IDS),
        "persistent_state_coverage_ids": list(PERSISTENT_STATE_COVERAGE_IDS),
        "n_persistent_state_coverages": len(PERSISTENT_STATE_COVERAGE_IDS),
        "consumer_mode_ids": list(CONSUMER_MODE_IDS),
        "n_consumer_modes": len(CONSUMER_MODE_IDS),
        "n_consumer_modes_selected": 0,
        "stage_kind_ids": list(STAGE_KIND_IDS),
        "n_stage_kinds": len(STAGE_KIND_IDS),
        "bottleneck_labels": list(BOTTLENECK_LABELS),
        "n_bottleneck_labels": len(BOTTLENECK_LABELS),
        "i_mlp_weight_only": I_MLP_WEIGHT_ONLY,
        "i_lm_head_weight_only": I_LM_HEAD_WEIGHT_ONLY,
        "i_gdn_vs_s_rw": I_GDN_VS_S_RW,
        "i_attn_core_vs_kv": I_ATTN_CORE_VS_KV,
        "ordering_ids": list(ORDERING_IDS),
        "ordering_object_ids": list(ORDERING_OBJECT_IDS),
        "n_orderings": len(ORDERING_IDS),
        "n_orderings_selected": 0,
        "ordering_usefulness_label": "HYPOTHESIS",
        "tile_family_ids": list(TILE_FAMILY_IDS),
        "n_tile_families": len(TILE_FAMILY_IDS),
        "tile_family_objects": tile_family_objects,
        "tile_size_selected": False,
        "mma_tile_extents_selected": False,
        "tile_usefulness_label": "HYPOTHESIS",
        "parallel_decomposition_ids": list(PARALLEL_DECOMPOSITION_IDS),
        "n_parallel_decompositions": len(PARALLEL_DECOMPOSITION_IDS),
        "n_parallel_decompositions_selected": 0,
        "conversion_hypothesis_ids": list(CONVERSION_HYPOTHESIS_IDS),
        "n_conversion_hypotheses": len(CONVERSION_HYPOTHESIS_IDS),
        "n_conversion_hypotheses_selected": 0,
        "conversion_usefulness_label": "HYPOTHESIS",
        "justification_hypothesis_ids": list(JUSTIFICATION_HYPOTHESIS_IDS),
        "n_justification_hypotheses": len(JUSTIFICATION_HYPOTHESIS_IDS),
        "n_justification_hypotheses_selected": 0,
        "justification_usefulness_label": "HYPOTHESIS",
        "layout_risk_ids": list(LAYOUT_RISK_IDS),
        "layout_risk_severities": list(LAYOUT_RISK_SEVERITIES),
        "layout_high_ids": list(LAYOUT_HIGH_IDS),
        "layout_medium_ids": list(LAYOUT_MEDIUM_IDS),
        "layout_low_ids": list(LAYOUT_LOW_IDS),
        "n_layout_risks": len(LAYOUT_RISK_IDS),
        "n_layout_high": len(LAYOUT_HIGH_IDS),
        "n_layout_medium": len(LAYOUT_MEDIUM_IDS),
        "n_layout_low": len(LAYOUT_LOW_IDS),
        "example_T": list(EXAMPLE_T),
        "T_is_stored_length_after_append": True,
        "primary_includes_mtp": True,
        "decode_prefill_share_artifact": True,
        "decode_prefill_share_graph": True,
        "decode_prefill_distinct_views_selected": False,
        "activations_in_layout_scope": False,
        "hardware_independent": True,
        "cuda_mapping_deferred": True,
        "thread_geometry_absent": True,
        "layout_winner_selected": False,
        "ordering_selected": False,
        "alignment_grain_selected": False,
        "specialized_alignment_grain_selected": False,
        "specialized_view_may_constrain_alignment": True,
        "conversion_pipeline_selected": False,
        "parallel_decomposition_selected": False,
        "ideal_byte_sequence_selected": False,
        "artifact_boundary_selected": False,
        "seq_specialized_tile_candidates_named": True,
        "tile_layout_deferred_to_task15": False,
        "ledger_open_question_parallel_decomposition_closed": False,
        "parallel_decomposition_justifies_layout_selected": False,
        "optimality_claim_absent": True,
        "gdn_primary_is_recurrent_eq_17": True,
        "chunkwise_not_zero_s_traffic": True,
        "paper_s_transpose_same_map": True,
        "state_write_not_optional": True,
        "kv_rope_baked_into_k": True,
        "gqa_repeat_not_stored": True,
        "conv_z_does_not_enter_conv": True,
        "portable_payload_little_endian": True,
        "gguf_is_not_the_runtime_format": True,
        "safetensors_is_source_not_runtime": True,
        "embed_lm_head_tied": False,
        "lm_head_uses_dense_gemm_object": True,
        "embed_uses_gather_row_object": True,
        "weight_unique_counted_once": True,
        "analyzes_logical_dimensions": True,
        "analyzes_consumers": True,
        "analyzes_access": True,
        "analyzes_tiling": True,
        "analyzes_alignment": True,
        "analyzes_conversion": True,
        "includes_gdn": True,
        "includes_convolution": True,
        "includes_kv": True,
        "vision_interface_is_not_a_node": True,
        "payloads_restreamed": False,
        "fusion_winner_selected": False,
        "state_payload_in_artifact_selected": False,
        "diagram_ids": list(DIAGRAM_REQUIRED_IDS),
        "n_diagrams": 1,
        "canonical_sentence_logical": CANONICAL_SENTENCE_LOGICAL,
        "canonical_sentence_candidates": CANONICAL_SENTENCE_CANDIDATES,
        "canonical_sentence_conversion": CANONICAL_SENTENCE_CONVERSION,
        "canonical_sentence_open_question": CANONICAL_SENTENCE_OPEN_QUESTION,
        "canonical_sentence_optimality": CANONICAL_SENTENCE_OPTIMALITY,
        "canonical_sentence_justification": CANONICAL_SENTENCE_JUSTIFICATION,
    }
    _assert_identities(summary, text)
    return summary


def _assert_identities(summary: dict, text: dict) -> None:
    """Assert dossier identities on a live layout-strategy object.

    Args:
        summary: Object produced by :func:`instantiate_layout_strategy`.
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
    assert summary["hidden_size"] == 5120
    assert summary["intermediate_size"] == 17408
    assert summary["vocab_size"] == 248320
    assert summary["head_dim"] == 256
    assert summary["num_attention_heads"] == 24
    assert summary["num_key_value_heads"] == 4
    assert summary["g_qa"] == 6
    assert summary["linear_num_value_heads"] == 48
    assert summary["linear_key_head_dim"] == 128
    assert summary["linear_value_head_dim"] == 128
    assert summary["linear_conv_kernel_dim"] == 4
    assert summary["linear_conv_delay"] == 3
    assert summary["d_qkv"] == 10240
    assert summary["kv_bytes_per_full_layer_per_token"] == 4096
    assert summary["kv_bytes_all_per_token"] == 69632
    assert summary["c_bytes_per_layer"] == 61440
    assert summary["c_bytes_all"] == 2949120
    assert summary["s_bytes_per_layer"] == 3145728
    assert summary["s_f32_bytes"] == 150994944
    assert summary["s_bf16_bytes"] == 75497472
    assert summary["embed_gather_bf16_bytes"] == 10240
    assert summary["embed_gather_int4_row_bytes"] == 2562
    assert summary["weight_bytes_lm_head"] == 2542796800
    assert summary["int4_g32_group_payload_bytes"] == 16
    assert summary["int3_g32_group_payload_bytes"] == 12
    assert summary["int2_g32_group_payload_bytes"] == 8
    assert summary["int4_row_hidden_payload_bytes"] == 2560
    assert summary["int6_row_hidden_payload_bytes"] == 3840
    assert summary["dual_view_int4_g128_unique_non_embed_bytes"] == 26863340064
    assert summary["n_layout_objects"] == 7
    assert summary["layout_object_ids"] == summary["access_class_ids"]
    assert summary["n_orderings"] == 14
    assert summary["n_orderings_selected"] == 0
    assert summary["n_tile_families"] == 9
    assert summary["tile_size_selected"] is False
    assert summary["mma_tile_extents_selected"] is False
    assert summary["n_tile_extent_candidates"] == 5
    assert summary["tile_extent_divides_dk"][-1] is False
    assert summary["tile_extent_divides_dk"][:4] == [True, True, True, True]
    assert summary["n_parallel_decompositions"] == 9
    assert summary["n_parallel_decompositions_selected"] == 0
    assert summary["n_conversion_hypotheses"] == 4
    assert summary["n_conversion_hypotheses_selected"] == 0
    assert summary["n_justification_hypotheses"] == 10
    assert summary["n_justification_hypotheses_selected"] == 0
    assert summary["n_layout_risks"] == 8
    assert summary["n_layout_high"] == 1
    assert summary["n_layout_medium"] == 6
    assert summary["n_layout_low"] == 1
    assert summary["n_analysis_dimensions"] == 6
    assert summary["n_persistent_state_coverages"] == 3
    assert summary["n_consumer_sequences"] == 7
    assert summary["n_access_classes"] == 7
    assert summary["n_stage_kinds"] == 9
    assert summary["n_diagrams"] == 1
    assert summary["n_consumer_modes"] == 2
    assert summary["n_consumer_modes_selected"] == 0
    assert summary["family_access_class"]["vision_deferred"] is None
    assert summary["family_access_class"]["embed_table"] == "gather_row"
    assert summary["family_access_class"]["lm_head"] == "dense_gemm"
    assert summary["family_access_class"]["conv1d"] == "depthwise_conv"
    assert summary["family_access_class"]["state_s"] == "state_s"
    assert summary["layout_object_primary_sequence_ids"][0] == "seq_gather_row"
    assert (
        summary["layout_object_primary_sequence_ids"][1]
        == "seq_gemm_codes_then_scales"
    )
    assert summary["layout_object_primary_sequence_ids"][6] == "seq_state_s_dense"
    assert summary["layout_object_primary_sequence_ids"][2:6] == [
        None,
        None,
        None,
        None,
    ]
    assert summary["lm_head_sequence_id"] == "seq_lm_head_full"
    assert summary["gated_delta_net_state_sequence_id"] == "seq_state_s_dense"
    assert summary["ideal_byte_sequence_selected"] is False
    assert summary["artifact_boundary_selected"] is False
    assert summary["decode_prefill_distinct_views_selected"] is False
    assert summary["ledger_open_question_parallel_decomposition_closed"] is False
    assert summary["layout_winner_selected"] is False
    assert summary["optimality_claim_absent"] is True
    assert summary["activations_in_layout_scope"] is False
    assert summary["thread_geometry_absent"] is True
    assert summary["hardware_independent"] is True
    assert summary["gguf_is_not_the_runtime_format"] is True
    assert summary["payloads_restreamed"] is False
    assert summary["seq_specialized_tile_candidates_named"] is True
    assert summary["tile_layout_deferred_to_task15"] is False
    assert summary["includes_gdn"] is True
    assert summary["includes_convolution"] is True
    assert summary["includes_kv"] is True
    assert summary["analyzes_logical_dimensions"] is True
    assert summary["analyzes_consumers"] is True
    assert summary["analyzes_access"] is True
    assert summary["analyzes_tiling"] is True
    assert summary["analyzes_alignment"] is True
    assert summary["analyzes_conversion"] is True
    assert summary["gqa_repeat_not_stored"] is True
    assert summary["paper_s_transpose_same_map"] is True
    assert summary["chunkwise_not_zero_s_traffic"] is True
    assert summary["n_layout_object_primary_sequences"] == 3
    assert summary["family_layout_object"] == summary["family_access_class"]
    assert summary["ordering_usefulness_label"] == "HYPOTHESIS"
    assert summary["tile_usefulness_label"] == "HYPOTHESIS"
    assert summary["conversion_usefulness_label"] == "HYPOTHESIS"
    assert summary["justification_usefulness_label"] == "HYPOTHESIS"
    assert math.isclose(summary["i_gdn_vs_s_rw"], 0.75)
    assert summary["i_mlp_weight_only"] == 1
    assert summary["i_lm_head_weight_only"] == 1
    assert summary["i_attn_core_vs_kv"] == 6
    assert set(summary["ordering_object_ids"]) == set(summary["layout_object_ids"])
    assert summary["layout_object_ranks"] == list(LAYOUT_OBJECT_RANKS)
    assert summary["n_v_divides_none_of_extents_as_head_count"] is True
    assert all(summary["tile_extent_divides_hidden"])
    assert all(summary["tile_extent_divides_intermediate"])
    assert all(summary["tile_extent_divides_d_qkv"])
    assert all(summary["tile_extent_divides_head_dim"])


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
    """Compare documented JSON against a live layout-strategy object.

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


def check_layout_strategy(live: dict, layout_strategy_path: Path) -> None:
    """Check a layout-strategy markdown file against a live summary.

    Args:
        live: Layout-strategy summary object from sitting config.
        layout_strategy_path: Path to ``layout-strategy.md``.

    Raises:
        LayoutStrategyMismatch: On heading, JSON, diagram, or token
            mismatches.
        OSError: If the file cannot be read.
    """

    text = layout_strategy_path.read_text(encoding="utf-8")
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

    mermaid_fences = list(MERMAID_FENCE_RE.finditer(text))
    if len(mermaid_fences) != 1:
        differences.append(f"mermaid fence count {len(mermaid_fences)} != 1")
    else:
        match = mermaid_fences[0]
        body = match.group(1)
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
        par_match = re.search(
            r"^## Parallel decompositions\s*$", text, flags=re.MULTILINE
        )
        work_match = re.search(
            r"^## Work citations and non-decisions\s*$",
            text,
            flags=re.MULTILINE,
        )
        if par_match is None or work_match is None:
            differences.append("missing Parallel decompositions or Work citations heading")
        elif not (par_match.end() <= match.start() < work_match.start()):
            differences.append(
                "mermaid fence is not under ## Parallel decompositions"
            )
        caption = text[par_match.end() : match.start()] if par_match else ""
        if "HYPOTHESIS" not in caption or "unresolved" not in caption:
            differences.append(
                "mermaid caption missing HYPOTHESIS or unresolved"
            )

    if CANONICAL_SENTENCE_LOGICAL not in text:
        differences.append("canonical_sentence_logical not present verbatim")
    if CANONICAL_SENTENCE_CANDIDATES not in text:
        differences.append("canonical_sentence_candidates not present verbatim")
    if CANONICAL_SENTENCE_CONVERSION not in text:
        differences.append("canonical_sentence_conversion not present verbatim")
    if CANONICAL_SENTENCE_OPEN_QUESTION not in text:
        differences.append("canonical_sentence_open_question not present verbatim")
    if CANONICAL_SENTENCE_OPTIMALITY not in text:
        differences.append("canonical_sentence_optimality not present verbatim")
    if CANONICAL_SENTENCE_JUSTIFICATION not in text:
        differences.append(
            "canonical_sentence_justification not present verbatim"
        )

    for label, ids in SUBSTRING_ID_GROUPS:
        for item_id in ids:
            if item_id not in text:
                differences.append(f"{label} {item_id!r} missing as substring")

    for heading in REQUIRED_H3_HEADINGS:
        if heading not in text:
            differences.append(f"required subsection {heading!r} missing")

    if "HYPOTHESIS" not in text:
        differences.append("word HYPOTHESIS missing")
    if "hardware-independent" not in text:
        differences.append("word hardware-independent missing")
    if "unresolved" not in text:
        differences.append("word unresolved missing")
    if "no optimality claim" not in text:
        differences.append("phrase 'no optimality claim' missing")

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
        raise LayoutStrategyMismatch(differences)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the layout-strategy checker.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Instantiate Qwen3.8-27B language+MTP layout-strategy summary "
            "from text_config and check docs/architecture/layout-strategy.md."
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
        help="Write layout-strategy summary JSON to stdout.",
    )
    parser.add_argument(
        "--layout-strategy",
        type=Path,
        default=None,
        help="Markdown path whose first json fence must match live summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the layout-strategy checker CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch, 2 missing
        config.
    """

    args = _parse_args(argv)
    check_path = args.layout_strategy
    if check_path is None and not args.json:
        check_path = DEFAULT_LAYOUT_STRATEGY
    try:
        config = load_config(args.config)
        summary = instantiate_layout_strategy(config)
        if check_path is not None:
            if not check_path.is_file():
                print(
                    f"layout-strategy file not found: {check_path}",
                    file=sys.stderr,
                )
                return 1
            check_layout_strategy(summary, check_path)
    except MissingConfig as exc:
        print(str(exc.path), file=sys.stderr)
        return 2
    except LayoutStrategyMismatch as exc:
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

    if args.json or check_path is None:
        sys.stdout.write(dumps_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
