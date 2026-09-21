#!/usr/bin/env python3
"""Check Qwen3.8-27B runtime-format design against sitting text_config.

Reads ``text_config`` from a Transformers ``config.json`` (no safetensor
payloads) and either prints the runtime-format-design summary object or
checks that ``docs/architecture/runtime-format-design.md`` matches it.
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

CANONICAL_SENTENCE_CAPABILITIES = (
    "Representation and packing capabilities in this document are a research "
    "space, not selected winners."
)

CANONICAL_SENTENCE_PORTABLE = (
    "Portable versus backend-specialized artifact boundaries remain open "
    "absent compelling evidence."
)

CANONICAL_SENTENCE_PACKED = (
    "Packed byte counts in this document are DERIVED illustrations, not "
    "selected consumer layouts."
)

CANONICAL_SENTENCE_SEQUENCES = (
    "Ideal consumer byte sequences remain an open question."
)

BYTES_BF16 = 2
BYTES_F32 = 4
BYTES_FP8 = 1
SCALE_STORAGE_ILLUSTRATION_BYTES = 2
ALIGNMENT_ILLUSTRATION_GRAIN = 1
SCALE_PLACEMENT_ILLUSTRATION = "sidecar_array"
CONTAINER_HEADER_ILLUSTRATION_BYTES_PER_TENSOR = 64
N_LANGUAGE_MTP_TENSORS = 866

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

CANDIDATE_RECIPE_IDS: tuple[str, ...] = (
    "keep_source",
    "narrow_bf16",
    "fp8_tensor",
    "i8_tensor",
    "i8_row",
    "i8_row_asym",
    "i8_g32",
    "i6_row",
    "i4_row",
    "i4_col",
    "i4_g32",
    "i4_g64",
    "i4_g128",
    "i4_g128_p99",
    "i4_g128_rms",
    "i4_g128_extract",
    "i4_g32_mixed",
    "i4_row_extract",
    "i4_clip",
    "i3_g32",
    "i3_g32_extract",
    "i2_g32_extract",
)

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

ARTIFACT_KINDS: tuple[str, ...] = (
    "manifest",
    "payload",
    "metadata_blob",
    "sidecar",
    "view",
    "schema_state",
)

REPRESENTATION_CAPABILITY_IDS: tuple[str, ...] = (
    "tensor_identity",
    "element_encoding",
    "group_metadata",
    "outlier_sidecar",
    "mixed_width_map",
    "shared_binding",
    "access_class",
    "state_schema",
)

PACKING_CAPABILITY_IDS: tuple[str, ...] = (
    "bit_pack",
    "scale_storage",
    "scale_placement",
    "alignment_pad",
    "endian_le",
    "row_addressable",
    "integrity_record",
    "view_binding",
)

ACCESS_CLASS_IDS: tuple[str, ...] = (
    "gather_row",
    "dense_gemm",
    "depthwise_conv",
    "vector_param",
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

FAMILY_ACCESS_CLASS_VALUES: tuple[str | None, ...] = tuple(
    FAMILY_ACCESS_CLASS[family_id] for family_id in POLICY_FAMILIES
)

SHARED_WEIGHT_IDS: tuple[str, ...] = ("E", "W_lm")

CONSUMER_SEQUENCE_IDS: tuple[str, ...] = (
    "seq_gemm_codes_then_scales",
    "seq_gemm_interleaved_group",
    "seq_gather_row",
    "seq_lm_head_full",
    "seq_outlier_extra",
    "seq_state_s_dense",
    "seq_specialized_tile",
)

CONSUMER_SEQUENCE_ACCESS_CLASSES: tuple[str, ...] = (
    "dense_gemm",
    "dense_gemm",
    "gather_row",
    "dense_gemm",
    "dense_gemm",
    "state_s",
    "dense_gemm",
)

ARTIFACT_APPROACH_IDS: tuple[str, ...] = (
    "portable_only",
    "backend_specialized_only",
    "portable_plus_specialized_views",
    "manifest_plus_backend_blobs",
)

COMPARISON_DIMENSION_IDS: tuple[str, ...] = (
    "compile_once",
    "load_convert",
    "byte_sequence",
    "tile_freedom",
    "store_amplification",
    "consumer_portability",
)

OPEN_DECISION_IDS: tuple[str, ...] = (
    "artifact_boundary",
    "ideal_byte_sequence",
    "scale_storage_bytes",
    "scale_placement",
    "alignment_grain",
    "code_bit_order",
    "integrity_algorithm",
    "state_payload_inclusion",
)

FORMAT_RISK_IDS: tuple[str, ...] = (
    "f_unpack_portable",
    "f_repack_specialized",
    "f_dual_view_size",
    "f_gather_stride",
    "f_3bit_shift",
    "f_outlier_index",
)

FORMAT_RISK_SEVERITIES: tuple[str, ...] = (
    "high",
    "medium",
    "low",
    "medium",
    "medium",
    "medium",
)

FORMAT_HIGH_IDS: tuple[str, ...] = ("f_unpack_portable",)

FORMAT_MEDIUM_IDS: tuple[str, ...] = (
    "f_repack_specialized",
    "f_gather_stride",
    "f_3bit_shift",
    "f_outlier_index",
)

FORMAT_LOW_IDS: tuple[str, ...] = ("f_dual_view_size",)

CODE_BIT_ORDER_CANDIDATES: tuple[str, ...] = ("lsb_first", "msb_first")
SCALE_PLACEMENT_CANDIDATES: tuple[str, ...] = (
    "sidecar_array",
    "interleaved_group",
)
ALIGNMENT_GRAIN_CANDIDATES: tuple[int, ...] = (1, 16, 32, 128, 256)
SCALE_STORAGE_BYTES_CANDIDATES: tuple[int, ...] = (2, 4)
INTEGRITY_ALGORITHM_CANDIDATES: tuple[str, ...] = ("none", "checksum")

DIAGRAM_REQUIRED_IDS: tuple[str, ...] = (
    "representation",
    "packing",
    "portable",
    "specialized",
    "gather",
    "gemm",
    "state",
    "open",
)

LOCKED_DOCUMENT_TOKENS: tuple[str, ...] = (
    "5120",
    "17408",
    "248320",
    "866",
    "27320697856",
    "17112760320",
    "34225520640",
    "2542796800",
    "54641395712",
    "52098598912",
    "150994944",
    "69632",
    "2949120",
    "10240",
    "2562",
    "2560",
    "3840",
    "12",
    "8",
    "64",
    "55424",
    "8556380160",
    "9625927680",
    "8823767040",
    "13431670032",
    "26863340064",
    "0.28125",
    "0.2578125",
)

FORBIDDEN_WINNER_PHRASES: tuple[str, ...] = (
    "portable is better",
    "specialized is better",
    "recommend portable",
    "recommend specialized",
    "selected artifact",
    "ideal byte sequence is",
    "artifact boundary is",
    "should use GGUF",
    "GGUF is the runtime format",
    "selected winner",
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
    "full_attention_indices",
    "bytes_bf16",
    "bytes_f32",
    "bytes_fp8",
    "n_language_mtp_tensors",
    "n_language_mtp_parameters",
    "mlp_n",
    "embed_n",
    "mlp_bf16_bytes",
    "mlp_int4_payload_bytes",
    "mlp_int4_g32_meta_bytes",
    "mlp_int4_g32_total_bytes",
    "mlp_int4_g32_over_bf16",
    "mlp_int4_g128_meta_bytes",
    "mlp_int4_g128_total_bytes",
    "mlp_int4_g128_over_bf16",
    "weight_bytes_language_mlp",
    "weight_bytes_lm_head",
    "weight_bytes_embed_table",
    "weight_bytes_language_linear_attn",
    "weight_bytes_language_self_attn",
    "weight_bytes_language_mtp_excl_vision",
    "weight_bytes_unique_non_embed",
    "unique_non_embed_n",
    "unique_non_embed_bf16_bytes",
    "unique_non_embed_int4_g128_total_bytes",
    "unique_non_embed_int4_g128_over_bf16",
    "embed_gather_bf16_bytes",
    "embed_gather_int4_row_bytes",
    "int4_row_hidden_payload_bytes",
    "int6_row_hidden_payload_bytes",
    "int3_g32_group_payload_bytes",
    "int2_g32_group_payload_bytes",
    "s_f32_bytes",
    "s_bf16_bytes",
    "s_fp8_bytes",
    "kv_bytes_all_per_token",
    "c_bytes_all",
    "scale_storage_illustration_bytes",
    "scale_storage_bytes_candidates",
    "alignment_illustration_grain",
    "alignment_grain_candidates",
    "scale_placement_illustration",
    "scale_placement_candidates",
    "code_bit_order_candidates",
    "integrity_algorithm_candidates",
    "container_header_illustration_bytes_per_tensor",
    "container_header_illustration_total_bytes",
    "dual_view_int4_g128_unique_non_embed_bytes",
    "ragged_last_group",
    "group_clips_to_axis",
    "payload_formula_uses_ceil",
    "packing_illustration_matches_task08_lower_bound",
    "portable_payload_little_endian",
    "outlier_sidecar_bytes_instantiated",
    "artifact_kinds",
    "n_artifact_kinds",
    "representation_capability_ids",
    "n_representation_capabilities",
    "logical_descriptors",
    "view_descriptor",
    "packing_capability_ids",
    "n_packing_capabilities",
    "access_class_ids",
    "n_access_classes",
    "policy_families",
    "family_access_class",
    "family_access_class_values",
    "candidate_recipe_ids",
    "n_candidate_recipes",
    "keep_source_packable_on_all_defined_families",
    "shared_weight_ids",
    "consumer_sequence_ids",
    "consumer_sequence_access_classes",
    "n_consumer_sequences",
    "artifact_approach_ids",
    "n_artifact_approaches",
    "comparison_dimension_ids",
    "n_comparison_dimensions",
    "open_decision_ids",
    "open_decision_selected",
    "n_open_decisions",
    "n_open_decisions_selected",
    "format_risk_ids",
    "format_risk_severities",
    "format_high_ids",
    "format_medium_ids",
    "format_low_ids",
    "n_format_risks",
    "n_format_high",
    "n_format_medium",
    "n_format_low",
    "activations_in_artifact",
    "kernels_in_artifact",
    "tokenizer_in_artifact",
    "gguf_is_not_the_runtime_format",
    "safetensors_is_source_not_runtime",
    "embed_lm_head_tied",
    "decode_prefill_share_artifact",
    "decode_prefill_distinct_views_selected",
    "tile_layout_deferred_to_task15",
    "artifact_boundary_selected",
    "ideal_byte_sequence_selected",
    "state_payload_in_artifact_selected",
    "state_schema_required",
    "learned_codebooks_required",
    "compiler_profile_selected",
    "pareto_frontier_selected",
    "payloads_restreamed",
    "diagram_ids",
    "n_diagrams",
    "canonical_sentence_logical",
    "canonical_sentence_capabilities",
    "canonical_sentence_portable",
    "canonical_sentence_packed",
    "canonical_sentence_sequences",
)

REQUIRED_HEADINGS: tuple[str, ...] = (
    "Authority",
    "Format convention",
    "Artifact object model",
    "Representation capabilities",
    "Packing capabilities",
    "Consumer access and byte sequences",
    "Portable versus backend-specific artifacts",
    "Open decisions",
    "Deferred vision",
    "Machine-checkable summary JSON",
)

JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)
MERMAID_FENCE_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)
FORBIDDEN_RE = re.compile(r"TBD|TODO|\?\?\?")
UNKNOWN_RE = re.compile(r"UNKNOWN")

SUBSTRING_ID_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("representation capability", REPRESENTATION_CAPABILITY_IDS),
    ("packing capability", PACKING_CAPABILITY_IDS),
    ("artifact approach", ARTIFACT_APPROACH_IDS),
    ("consumer sequence", CONSUMER_SEQUENCE_IDS),
    ("open decision", OPEN_DECISION_IDS),
    ("format risk", FORMAT_RISK_IDS),
    ("access class", ACCESS_CLASS_IDS),
    ("policy family", POLICY_FAMILIES),
    ("recipe id", CANDIDATE_RECIPE_IDS),
)


class MissingConfig(Exception):
    """Raised when the config path is absent or unreadable."""

    def __init__(self, path: Path) -> None:
        super().__init__(str(path))
        self.path = path


class RuntimeFormatDesignMismatch(Exception):
    """Raised when the runtime-format-design markdown fails a content check."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def dumps_summary(summary: dict) -> str:
    """Pretty-print the runtime-format-design summary object.

    Args:
        summary: Object produced by :func:`instantiate_runtime_format_design`.

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


def _n_groups(n_elements: int, group_size: int) -> int:
    """Return ceil(n/g) groups, allowing a ragged last group.

    Args:
        n_elements: Element count n.
        group_size: Group size g.

    Returns:
        Number of groups.

    Raises:
        AssertionError: If n or g is not positive.
    """

    assert n_elements > 0 and group_size > 0, "n and g must be positive"
    return math.ceil(n_elements / group_size)


def _aligned_bytes(payload_bytes: int, alignment_grain: int) -> int:
    """Return ceil(B_payload,pack / A) * A aligned payload bytes.

    Args:
        payload_bytes: Packed payload byte count.
        alignment_grain: Alignment grain A.

    Returns:
        Aligned payload byte count.

    Raises:
        AssertionError: If payload is negative or grain is not positive.
    """

    assert payload_bytes >= 0 and alignment_grain > 0, "B and A must be valid"
    return math.ceil(payload_bytes / alignment_grain) * alignment_grain


def _integer_storage_bytes(
    n_elements: int,
    code_bits: int,
    group_size: int,
    scale_bytes: int,
    zero_point_bytes: int,
    alignment_grain: int,
) -> tuple[int, int, int, int]:
    """Return payload, group count, metadata, and aligned packed totals.

    Args:
        n_elements: Element count n.
        code_bits: Integer code width b.
        group_size: Group size g.
        scale_bytes: Scale storage bytes s.
        zero_point_bytes: Zero-point bytes z.
        alignment_grain: Alignment grain A.

    Returns:
        ``(B_payload,pack, n_g, B_meta, B)`` with ``B`` the aligned payload
        plus sidecar metadata.
    """

    payload_bytes = _payload_pack_bytes(n_elements, code_bits)
    n_g = _n_groups(n_elements, group_size)
    meta_bytes = n_g * (scale_bytes + zero_point_bytes)
    aligned = _aligned_bytes(payload_bytes, alignment_grain)
    return payload_bytes, n_g, meta_bytes, aligned + meta_bytes


def instantiate_runtime_format_design(config: dict) -> dict:
    """Compute the TASK-09 runtime-format-design summary from ``config``.

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

    assert param_dtype == "bfloat16", f"dtype is not bfloat16: {param_dtype!r}"
    assert mamba_ssm_dtype == "float32", (
        f"mamba_ssm_dtype is not float32: {mamba_ssm_dtype!r}"
    )
    assert text.get("attn_output_gate") is True, "attn_output_gate is not true"
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

    linear_qkv_width = (
        2 * linear_num_key_heads + linear_num_value_heads
    ) * linear_key_head_dim
    linear_z_width = linear_num_value_heads * linear_value_head_dim
    q_proj_out = 2 * n_attn_heads * head_dim
    kv_proj_out = n_kv_heads * head_dim
    o_proj_in = n_attn_heads * head_dim
    linear_conv_delay = linear_conv_kernel_dim - 1

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

    mlp_n = 3 * n_decoder_layers * intermediate_size * hidden_size
    embed_n = vocab_size * hidden_size
    mlp_bf16_bytes = BYTES_BF16 * mlp_n
    weight_bytes_language_mlp = mlp_bf16_bytes
    weight_bytes_lm_head = embed_n * BYTES_BF16
    weight_bytes_embed_table = embed_n * BYTES_BF16
    weight_bytes_language_linear_attn = (
        n_linear_layers * lin_params_per_layer * BYTES_BF16
    )
    weight_bytes_language_self_attn = (
        n_full_layers * full_params_per_layer * BYTES_BF16
    )
    weight_bytes_language_layer_norms = (
        2 * n_decoder_layers * hidden_size * BYTES_BF16
    )
    weight_bytes_language_final_norm = hidden_size * BYTES_BF16
    weight_bytes_mtp = mtp_params * BYTES_BF16
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
    n_language_mtp_parameters = weight_bytes_language_mtp_excl_vision // BYTES_BF16
    unique_non_embed_n = weight_bytes_unique_non_embed // BYTES_BF16
    unique_non_embed_bf16_bytes = weight_bytes_unique_non_embed

    scale_s = SCALE_STORAGE_ILLUSTRATION_BYTES
    align_a = ALIGNMENT_ILLUSTRATION_GRAIN
    (
        mlp_int4_payload_bytes,
        _mlp_g32_n_g,
        mlp_int4_g32_meta_bytes,
        mlp_int4_g32_total_bytes,
    ) = _integer_storage_bytes(mlp_n, 4, 32, scale_s, 0, align_a)
    (
        _mlp_g128_payload,
        _mlp_g128_n_g,
        mlp_int4_g128_meta_bytes,
        mlp_int4_g128_total_bytes,
    ) = _integer_storage_bytes(mlp_n, 4, 128, scale_s, 0, align_a)
    mlp_int4_g32_over_bf16 = mlp_int4_g32_total_bytes / mlp_bf16_bytes
    mlp_int4_g128_over_bf16 = mlp_int4_g128_total_bytes / mlp_bf16_bytes

    (
        _une_payload,
        _une_n_g,
        _une_meta,
        unique_non_embed_int4_g128_total_bytes,
    ) = _integer_storage_bytes(unique_non_embed_n, 4, 128, scale_s, 0, align_a)
    unique_non_embed_int4_g128_over_bf16 = (
        unique_non_embed_int4_g128_total_bytes / unique_non_embed_bf16_bytes
    )

    embed_gather_bf16_bytes = hidden_size * BYTES_BF16
    int4_row_hidden_payload_bytes = _payload_pack_bytes(hidden_size, 4)
    int6_row_hidden_payload_bytes = _payload_pack_bytes(hidden_size, 6)
    int3_g32_group_payload_bytes = _payload_pack_bytes(32, 3)
    int2_g32_group_payload_bytes = _payload_pack_bytes(32, 2)
    embed_gather_int4_row_bytes = (
        int4_row_hidden_payload_bytes + SCALE_STORAGE_ILLUSTRATION_BYTES
    )

    s_elems_per_layer = (
        linear_num_value_heads * linear_key_head_dim * linear_value_head_dim
    )
    s_f32_bytes = n_linear_layers * s_elems_per_layer * BYTES_F32
    s_bf16_bytes = n_linear_layers * s_elems_per_layer * BYTES_BF16
    s_fp8_bytes = n_linear_layers * s_elems_per_layer * BYTES_FP8

    kv_elems_per_full_layer_per_token = 2 * n_kv_heads * head_dim
    kv_bytes_all_per_token = (
        n_full_layers_with_kv * kv_elems_per_full_layer_per_token * BYTES_BF16
    )
    c_bytes_all = (
        n_linear_layers * linear_conv_delay * linear_qkv_width * BYTES_BF16
    )

    container_header_illustration_total_bytes = (
        N_LANGUAGE_MTP_TENSORS * CONTAINER_HEADER_ILLUSTRATION_BYTES_PER_TENSOR
    )
    dual_view_int4_g128_unique_non_embed_bytes = (
        2 * unique_non_embed_int4_g128_total_bytes
    )

    family_access_class = {
        family_id: FAMILY_ACCESS_CLASS[family_id] for family_id in POLICY_FAMILIES
    }
    open_decision_selected = {decision_id: False for decision_id in OPEN_DECISION_IDS}

    summary = {
        "authority": AUTHORITY,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "n_decoder_layers": n_decoder_layers,
        "n_linear_layers": n_linear_layers,
        "n_full_layers": n_full_layers,
        "n_mtp_blocks": n_mtp_blocks,
        "full_attention_indices": full_from_types,
        "bytes_bf16": BYTES_BF16,
        "bytes_f32": BYTES_F32,
        "bytes_fp8": BYTES_FP8,
        "n_language_mtp_tensors": N_LANGUAGE_MTP_TENSORS,
        "n_language_mtp_parameters": n_language_mtp_parameters,
        "mlp_n": mlp_n,
        "embed_n": embed_n,
        "mlp_bf16_bytes": mlp_bf16_bytes,
        "mlp_int4_payload_bytes": mlp_int4_payload_bytes,
        "mlp_int4_g32_meta_bytes": mlp_int4_g32_meta_bytes,
        "mlp_int4_g32_total_bytes": mlp_int4_g32_total_bytes,
        "mlp_int4_g32_over_bf16": mlp_int4_g32_over_bf16,
        "mlp_int4_g128_meta_bytes": mlp_int4_g128_meta_bytes,
        "mlp_int4_g128_total_bytes": mlp_int4_g128_total_bytes,
        "mlp_int4_g128_over_bf16": mlp_int4_g128_over_bf16,
        "weight_bytes_language_mlp": weight_bytes_language_mlp,
        "weight_bytes_lm_head": weight_bytes_lm_head,
        "weight_bytes_embed_table": weight_bytes_embed_table,
        "weight_bytes_language_linear_attn": weight_bytes_language_linear_attn,
        "weight_bytes_language_self_attn": weight_bytes_language_self_attn,
        "weight_bytes_language_mtp_excl_vision": (
            weight_bytes_language_mtp_excl_vision
        ),
        "weight_bytes_unique_non_embed": weight_bytes_unique_non_embed,
        "unique_non_embed_n": unique_non_embed_n,
        "unique_non_embed_bf16_bytes": unique_non_embed_bf16_bytes,
        "unique_non_embed_int4_g128_total_bytes": (
            unique_non_embed_int4_g128_total_bytes
        ),
        "unique_non_embed_int4_g128_over_bf16": (
            unique_non_embed_int4_g128_over_bf16
        ),
        "embed_gather_bf16_bytes": embed_gather_bf16_bytes,
        "embed_gather_int4_row_bytes": embed_gather_int4_row_bytes,
        "int4_row_hidden_payload_bytes": int4_row_hidden_payload_bytes,
        "int6_row_hidden_payload_bytes": int6_row_hidden_payload_bytes,
        "int3_g32_group_payload_bytes": int3_g32_group_payload_bytes,
        "int2_g32_group_payload_bytes": int2_g32_group_payload_bytes,
        "s_f32_bytes": s_f32_bytes,
        "s_bf16_bytes": s_bf16_bytes,
        "s_fp8_bytes": s_fp8_bytes,
        "kv_bytes_all_per_token": kv_bytes_all_per_token,
        "c_bytes_all": c_bytes_all,
        "scale_storage_illustration_bytes": SCALE_STORAGE_ILLUSTRATION_BYTES,
        "scale_storage_bytes_candidates": list(SCALE_STORAGE_BYTES_CANDIDATES),
        "alignment_illustration_grain": ALIGNMENT_ILLUSTRATION_GRAIN,
        "alignment_grain_candidates": list(ALIGNMENT_GRAIN_CANDIDATES),
        "scale_placement_illustration": SCALE_PLACEMENT_ILLUSTRATION,
        "scale_placement_candidates": list(SCALE_PLACEMENT_CANDIDATES),
        "code_bit_order_candidates": list(CODE_BIT_ORDER_CANDIDATES),
        "integrity_algorithm_candidates": list(INTEGRITY_ALGORITHM_CANDIDATES),
        "container_header_illustration_bytes_per_tensor": (
            CONTAINER_HEADER_ILLUSTRATION_BYTES_PER_TENSOR
        ),
        "container_header_illustration_total_bytes": (
            container_header_illustration_total_bytes
        ),
        "dual_view_int4_g128_unique_non_embed_bytes": (
            dual_view_int4_g128_unique_non_embed_bytes
        ),
        "ragged_last_group": True,
        "group_clips_to_axis": True,
        "payload_formula_uses_ceil": True,
        "packing_illustration_matches_task08_lower_bound": True,
        "portable_payload_little_endian": True,
        "outlier_sidecar_bytes_instantiated": False,
        "artifact_kinds": list(ARTIFACT_KINDS),
        "n_artifact_kinds": len(ARTIFACT_KINDS),
        "representation_capability_ids": list(REPRESENTATION_CAPABILITY_IDS),
        "n_representation_capabilities": len(REPRESENTATION_CAPABILITY_IDS),
        "logical_descriptors": {
            "i8_row_asym": {"zero_point_encoding": "signed_int8", "count": "d_out", "range": [-128, 127], "binding": "logical_output_row_ordinal", "storage": "metadata_blob_span", "scale_field": "separate_per_row"},
            "extract_high": {"count": "explicit", "value_encoding": "bf16_le", "index_encoding_candidates": ["flat_u32", "flat_u64"], "selected_index_encoding_per_view": True, "binding": "logical_flat_index", "index_range": "0 <= index < numel", "value_span": "sidecar_payload_span", "index_span": "sidecar_index_span"},
            "mixed_group": {"width_code_encoding": "u8", "count": "group_count", "binding": "logical_group_ordinal", "allowed_values": "recipe_narrow_and_wide_formats", "storage": "sidecar_span"},
        },
        "view_descriptor": {"logical_tensor_id": "required", "view_id": "required", "view_role": "required", "backend_tag": "required", "payload_span": ["offset", "length"], "metadata_span": ["offset", "length"], "sidecar_span": ["offset", "length"], "access_consumer_binding": "required", "ordering_ref": None, "tile_ref": None, "integrity_ref": "required"},
        "packing_capability_ids": list(PACKING_CAPABILITY_IDS),
        "n_packing_capabilities": len(PACKING_CAPABILITY_IDS),
        "access_class_ids": list(ACCESS_CLASS_IDS),
        "n_access_classes": len(ACCESS_CLASS_IDS),
        "policy_families": list(POLICY_FAMILIES),
        "family_access_class": family_access_class,
        "family_access_class_values": list(FAMILY_ACCESS_CLASS_VALUES),
        "candidate_recipe_ids": list(CANDIDATE_RECIPE_IDS),
        "n_candidate_recipes": len(CANDIDATE_RECIPE_IDS),
        "keep_source_packable_on_all_defined_families": True,
        "shared_weight_ids": list(SHARED_WEIGHT_IDS),
        "consumer_sequence_ids": list(CONSUMER_SEQUENCE_IDS),
        "consumer_sequence_access_classes": list(CONSUMER_SEQUENCE_ACCESS_CLASSES),
        "n_consumer_sequences": len(CONSUMER_SEQUENCE_IDS),
        "artifact_approach_ids": list(ARTIFACT_APPROACH_IDS),
        "n_artifact_approaches": len(ARTIFACT_APPROACH_IDS),
        "comparison_dimension_ids": list(COMPARISON_DIMENSION_IDS),
        "n_comparison_dimensions": len(COMPARISON_DIMENSION_IDS),
        "open_decision_ids": list(OPEN_DECISION_IDS),
        "open_decision_selected": open_decision_selected,
        "n_open_decisions": len(OPEN_DECISION_IDS),
        "n_open_decisions_selected": 0,
        "format_risk_ids": list(FORMAT_RISK_IDS),
        "format_risk_severities": list(FORMAT_RISK_SEVERITIES),
        "format_high_ids": list(FORMAT_HIGH_IDS),
        "format_medium_ids": list(FORMAT_MEDIUM_IDS),
        "format_low_ids": list(FORMAT_LOW_IDS),
        "n_format_risks": len(FORMAT_RISK_IDS),
        "n_format_high": len(FORMAT_HIGH_IDS),
        "n_format_medium": len(FORMAT_MEDIUM_IDS),
        "n_format_low": len(FORMAT_LOW_IDS),
        "activations_in_artifact": False,
        "kernels_in_artifact": False,
        "tokenizer_in_artifact": False,
        "gguf_is_not_the_runtime_format": True,
        "safetensors_is_source_not_runtime": True,
        "embed_lm_head_tied": False,
        "decode_prefill_share_artifact": True,
        "decode_prefill_distinct_views_selected": False,
        "tile_layout_deferred_to_task15": True,
        "artifact_boundary_selected": False,
        "ideal_byte_sequence_selected": False,
        "state_payload_in_artifact_selected": False,
        "state_schema_required": True,
        "learned_codebooks_required": False,
        "compiler_profile_selected": False,
        "pareto_frontier_selected": False,
        "payloads_restreamed": False,
        "diagram_ids": list(DIAGRAM_REQUIRED_IDS),
        "n_diagrams": 1,
        "canonical_sentence_logical": CANONICAL_SENTENCE_LOGICAL,
        "canonical_sentence_capabilities": CANONICAL_SENTENCE_CAPABILITIES,
        "canonical_sentence_portable": CANONICAL_SENTENCE_PORTABLE,
        "canonical_sentence_packed": CANONICAL_SENTENCE_PACKED,
        "canonical_sentence_sequences": CANONICAL_SENTENCE_SEQUENCES,
    }
    _assert_identities(summary, text)
    return summary


def _assert_identities(summary: dict, text: dict) -> None:
    """Assert dossier identities on a live runtime-format-design object.

    Args:
        summary: Object produced by :func:`instantiate_runtime_format_design`.
        text: The ``text_config`` mapping used to build ``summary``.
    """

    assert list(summary) == list(SCHEMA_KEYS), (
        f"schema key order mismatch: {list(summary)} vs {list(SCHEMA_KEYS)}"
    )
    assert summary["authority"] == AUTHORITY
    assert summary["n_linear_layers"] == 48
    assert summary["n_full_layers"] == 16
    assert summary["n_mtp_blocks"] == 1
    assert summary["full_attention_indices"] == list(LOCKED_FULL_ATTENTION_INDICES)
    assert text.get("dtype") == "bfloat16"
    assert text.get("mamba_ssm_dtype") == "float32"
    assert summary["mlp_n"] == 17112760320
    assert summary["mlp_n"] == (
        3
        * summary["n_decoder_layers"]
        * summary["intermediate_size"]
        * summary["hidden_size"]
    )
    assert summary["embed_n"] == 1271398400
    assert summary["embed_n"] == summary["vocab_size"] * summary["hidden_size"]
    assert summary["n_language_mtp_tensors"] == 866
    assert summary["n_language_mtp_parameters"] == 27320697856
    assert summary["mlp_bf16_bytes"] == 34225520640
    assert summary["mlp_bf16_bytes"] == 2 * summary["mlp_n"]
    assert summary["mlp_int4_g32_over_bf16"] == 0.28125
    assert summary["mlp_int4_g128_over_bf16"] == 0.2578125
    assert summary["mlp_int4_g32_total_bytes"] == 9625927680
    assert summary["mlp_int4_g128_total_bytes"] == 8823767040
    assert summary["unique_non_embed_int4_g128_total_bytes"] == 13431670032
    assert summary["unique_non_embed_int4_g128_over_bf16"] == 0.2578125
    assert summary["weight_bytes_language_mtp_excl_vision"] == 54641395712
    assert summary["weight_bytes_unique_non_embed"] == 52098598912
    assert summary["embed_gather_bf16_bytes"] == 10240
    assert summary["embed_gather_int4_row_bytes"] == 2562
    assert summary["s_f32_bytes"] == 150994944
    assert summary["kv_bytes_all_per_token"] == 69632
    assert summary["c_bytes_all"] == 2949120
    assert summary["int3_g32_group_payload_bytes"] == 12
    assert summary["int2_g32_group_payload_bytes"] == 8
    assert summary["int6_row_hidden_payload_bytes"] == 3840
    assert summary["container_header_illustration_total_bytes"] == 55424
    assert summary["container_header_illustration_total_bytes"] == 866 * 64
    assert summary["dual_view_int4_g128_unique_non_embed_bytes"] == 26863340064
    assert summary["dual_view_int4_g128_unique_non_embed_bytes"] == 2 * 13431670032
    assert summary["n_representation_capabilities"] == 8
    assert summary["n_packing_capabilities"] == 8
    assert summary["n_artifact_approaches"] == 4
    assert summary["n_consumer_sequences"] == 7
    assert summary["n_open_decisions"] == 8
    assert summary["n_open_decisions_selected"] == 0
    assert summary["n_format_risks"] == 6
    assert summary["n_format_high"] == 1
    assert summary["n_format_medium"] == 4
    assert summary["n_format_low"] == 1
    assert summary["n_access_classes"] == 7
    assert summary["n_artifact_kinds"] == 6
    assert summary["n_diagrams"] == 1
    assert summary["n_comparison_dimensions"] == 6
    assert summary["candidate_recipe_ids"] == list(CANDIDATE_RECIPE_IDS)
    assert summary["policy_families"] == list(POLICY_FAMILIES)
    assert summary["family_access_class"]["vision_deferred"] is None
    assert summary["family_access_class"]["embed_table"] == "gather_row"
    assert summary["family_access_class"]["lm_head"] == "dense_gemm"
    assert all(
        value is False for value in summary["open_decision_selected"].values()
    )
    assert summary["artifact_boundary_selected"] is False
    assert summary["ideal_byte_sequence_selected"] is False
    assert summary["gguf_is_not_the_runtime_format"] is True
    assert summary["safetensors_is_source_not_runtime"] is True
    assert summary["activations_in_artifact"] is False
    assert summary["kernels_in_artifact"] is False
    assert summary["learned_codebooks_required"] is False
    assert summary["decode_prefill_share_artifact"] is True
    assert summary["state_schema_required"] is True
    assert summary["embed_lm_head_tied"] is False
    assert summary["packing_illustration_matches_task08_lower_bound"] is True
    assert summary["payload_formula_uses_ceil"] is True
    assert summary["portable_payload_little_endian"] is True
    assert summary["outlier_sidecar_bytes_instantiated"] is False
    assert summary["payloads_restreamed"] is False
    assert summary["mlp_int4_payload_bytes"] == 8556380160
    assert summary["mlp_int4_g32_meta_bytes"] == 1069547520
    assert summary["mlp_int4_g128_meta_bytes"] == 267386880
    assert summary["unique_non_embed_n"] == 26049299456
    assert summary["unique_non_embed_bf16_bytes"] == 52098598912
    assert summary["int4_row_hidden_payload_bytes"] == 2560
    assert summary["s_bf16_bytes"] == 75497472
    assert summary["s_fp8_bytes"] == 37748736
    assert summary["weight_bytes_language_mlp"] == 34225520640
    assert summary["weight_bytes_lm_head"] == 2542796800
    assert summary["weight_bytes_embed_table"] == 2542796800
    assert summary["artifact_kinds"] == list(ARTIFACT_KINDS)
    assert summary["representation_capability_ids"] == list(
        REPRESENTATION_CAPABILITY_IDS
    )
    assert summary["packing_capability_ids"] == list(PACKING_CAPABILITY_IDS)
    assert summary["access_class_ids"] == list(ACCESS_CLASS_IDS)
    assert summary["family_access_class_values"] == list(FAMILY_ACCESS_CLASS_VALUES)
    assert summary["shared_weight_ids"] == list(SHARED_WEIGHT_IDS)
    assert summary["consumer_sequence_ids"] == list(CONSUMER_SEQUENCE_IDS)
    assert summary["consumer_sequence_access_classes"] == list(
        CONSUMER_SEQUENCE_ACCESS_CLASSES
    )
    assert summary["artifact_approach_ids"] == list(ARTIFACT_APPROACH_IDS)
    assert summary["comparison_dimension_ids"] == list(COMPARISON_DIMENSION_IDS)
    assert summary["open_decision_ids"] == list(OPEN_DECISION_IDS)
    assert summary["open_decision_selected"] == {
        decision_id: False for decision_id in OPEN_DECISION_IDS
    }
    assert summary["format_risk_ids"] == list(FORMAT_RISK_IDS)
    assert summary["format_risk_severities"] == list(FORMAT_RISK_SEVERITIES)
    assert summary["format_high_ids"] == list(FORMAT_HIGH_IDS)
    assert summary["format_medium_ids"] == list(FORMAT_MEDIUM_IDS)
    assert summary["format_low_ids"] == list(FORMAT_LOW_IDS)
    high_from_ops = [
        risk_id
        for risk_id, severity in zip(
            summary["format_risk_ids"], summary["format_risk_severities"]
        )
        if severity == "high"
    ]
    medium_from_ops = [
        risk_id
        for risk_id, severity in zip(
            summary["format_risk_ids"], summary["format_risk_severities"]
        )
        if severity == "medium"
    ]
    low_from_ops = [
        risk_id
        for risk_id, severity in zip(
            summary["format_risk_ids"], summary["format_risk_severities"]
        )
        if severity == "low"
    ]
    assert high_from_ops == summary["format_high_ids"]
    assert medium_from_ops == summary["format_medium_ids"]
    assert low_from_ops == summary["format_low_ids"]
    assert summary["scale_storage_bytes_candidates"] == list(
        SCALE_STORAGE_BYTES_CANDIDATES
    )
    assert summary["alignment_grain_candidates"] == list(ALIGNMENT_GRAIN_CANDIDATES)
    assert summary["scale_placement_candidates"] == list(SCALE_PLACEMENT_CANDIDATES)
    assert summary["code_bit_order_candidates"] == list(CODE_BIT_ORDER_CANDIDATES)
    assert summary["integrity_algorithm_candidates"] == list(
        INTEGRITY_ALGORITHM_CANDIDATES
    )
    assert summary["scale_storage_illustration_bytes"] == 2
    assert summary["alignment_illustration_grain"] == 1
    assert summary["scale_placement_illustration"] == "sidecar_array"
    assert summary["ragged_last_group"] is True
    assert summary["group_clips_to_axis"] is True
    assert summary["bytes_bf16"] == 2
    assert summary["bytes_f32"] == 4
    assert summary["bytes_fp8"] == 1
    assert summary["keep_source_packable_on_all_defined_families"] is True
    assert summary["n_candidate_recipes"] == 22
    assert summary["tokenizer_in_artifact"] is False
    assert summary["decode_prefill_distinct_views_selected"] is False
    assert summary["tile_layout_deferred_to_task15"] is True
    assert summary["state_payload_in_artifact_selected"] is False
    assert summary["compiler_profile_selected"] is False
    assert summary["pareto_frontier_selected"] is False
    assert summary["diagram_ids"] == list(DIAGRAM_REQUIRED_IDS)
    assert summary["canonical_sentence_logical"] == CANONICAL_SENTENCE_LOGICAL
    assert summary["canonical_sentence_capabilities"] == CANONICAL_SENTENCE_CAPABILITIES
    assert summary["canonical_sentence_portable"] == CANONICAL_SENTENCE_PORTABLE
    assert summary["canonical_sentence_packed"] == CANONICAL_SENTENCE_PACKED
    assert summary["canonical_sentence_sequences"] == CANONICAL_SENTENCE_SEQUENCES


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


def diff_summary(live: dict, documented: dict) -> list[str]:
    """Compare documented JSON against a live runtime-format-design object.

    Args:
        live: Fresh script object.
        documented: Parsed first JSON fence from the markdown document.

    Returns:
        Human-readable difference strings; empty when objects match.
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


def _winner_phrase_allowed(text: str, start: int, phrase: str) -> bool:
    """Return whether a forbidden-winner match is the required example wording.

    The locked canonical sentence must contain ``not selected winners``, which
    includes the otherwise-forbidden substring ``selected winner``.

    Args:
        text: Full markdown document.
        start: Match start index.
        phrase: Forbidden phrase that matched.

    Returns:
        True when this occurrence is allowed.
    """

    if phrase != "selected winner":
        return False
    end = start + len(phrase)
    example = text[max(0, start - 6) : end]
    canonical = text[max(0, start - 4) : end + 1]
    return example == "not a selected winner" or canonical == "not selected winners"


def check_runtime_format_design(live: dict, design_path: Path) -> None:
    """Check a runtime-format-design markdown file against a live summary.

    Args:
        live: Runtime-format-design summary object from sitting config.
        design_path: Path to ``runtime-format-design.md``.

    Raises:
        RuntimeFormatDesignMismatch: On heading, JSON, diagram, or token
            mismatches.
        OSError: If the file cannot be read.
    """

    text = design_path.read_text(encoding="utf-8")
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
    elif "flowchart" not in mermaid_fences[0]:
        differences.append("mermaid fence does not contain flowchart")
    if mermaid_fences:
        body = mermaid_fences[0]
        missing_ids = [
            node_id for node_id in DIAGRAM_REQUIRED_IDS if node_id not in body
        ]
        if missing_ids:
            differences.append(f"diagram 1: missing node ids {missing_ids}")

    if CANONICAL_SENTENCE_LOGICAL not in text:
        differences.append("canonical_sentence_logical not present verbatim")
    if CANONICAL_SENTENCE_CAPABILITIES not in text:
        differences.append("canonical_sentence_capabilities not present verbatim")
    if CANONICAL_SENTENCE_PORTABLE not in text:
        differences.append("canonical_sentence_portable not present verbatim")
    if CANONICAL_SENTENCE_PACKED not in text:
        differences.append("canonical_sentence_packed not present verbatim")
    if CANONICAL_SENTENCE_SEQUENCES not in text:
        differences.append("canonical_sentence_sequences not present verbatim")

    for label, ids in SUBSTRING_ID_GROUPS:
        for item_id in ids:
            if item_id not in text:
                differences.append(f"{label} {item_id!r} missing as substring")

    if "HYPOTHESIS" not in text:
        differences.append("word HYPOTHESIS missing")
    if "not selected winners" not in text:
        differences.append("phrase 'not selected winners' missing")

    for match in FORBIDDEN_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        differences.append(f"line {line}: forbidden token {match.group(0)!r}")

    for phrase in FORBIDDEN_WINNER_PHRASES:
        search_from = 0
        while True:
            found = text.find(phrase, search_from)
            if found < 0:
                break
            if not _winner_phrase_allowed(text, found, phrase):
                line = text.count("\n", 0, found) + 1
                differences.append(
                    f"line {line}: forbidden winner phrase {phrase!r}"
                )
            search_from = found + len(phrase)

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

    for token in LOCKED_DOCUMENT_TOKENS:
        if token not in text:
            differences.append(
                f"locked number {token} missing as decimal substring"
            )

    if differences:
        raise RuntimeFormatDesignMismatch(differences)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the runtime-format-design checker.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Instantiate Qwen3.8-27B language+MTP runtime-format-design "
            "summary from text_config and check "
            "docs/architecture/runtime-format-design.md."
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
        help="Write runtime-format-design summary JSON to stdout.",
    )
    parser.add_argument(
        "--runtime-format-design",
        type=Path,
        default=None,
        help="Markdown path whose first json fence must match live summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the runtime-format-design checker CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch, 2 missing
        config.
    """

    args = _parse_args(argv)
    try:
        config = load_config(args.config)
        summary = instantiate_runtime_format_design(config)
        if args.runtime_format_design is not None:
            if not args.runtime_format_design.is_file():
                print(
                    f"runtime-format-design file not found: "
                    f"{args.runtime_format_design}",
                    file=sys.stderr,
                )
                return 1
            check_runtime_format_design(summary, args.runtime_format_design)
    except MissingConfig as exc:
        print(str(exc.path), file=sys.stderr)
        return 2
    except RuntimeFormatDesignMismatch as exc:
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

    if args.json or args.runtime_format_design is None:
        sys.stdout.write(dumps_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
