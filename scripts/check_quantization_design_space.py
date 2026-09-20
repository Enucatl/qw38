#!/usr/bin/env python3
"""Check Qwen3.8-27B quantization design space against sitting text_config.

Reads ``text_config`` from a Transformers ``config.json`` (no safetensor
payloads) and either prints the quantization-design-space summary object or
checks that ``docs/architecture/quantization-design-space.md`` matches it.
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
    "Candidate policies in this document are a research space, not selected "
    "winners."
)

CANONICAL_SENTENCE_HYPOTHESIS = (
    "Quality and decode-complexity risks in this document are hypotheses, "
    "not measurements."
)

CANONICAL_SENTENCE_METADATA = (
    "Metadata byte counts in this document are DERIVED lower bounds on scale "
    "and code storage, not a packed runtime layout."
)

BYTES_BF16 = 2
BYTES_F32 = 4
BYTES_FP8 = 1
SCALE_STORAGE_ILLUSTRATION_BYTES = 2

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

LEVEL2_IDS: tuple[str, ...] = (
    "embed",
    "final_norm",
    "lm_head",
    "input_layernorm",
    "post_attention_layernorm",
    "linear_attn.A_log",
    "linear_attn.conv1d",
    "linear_attn.dt_bias",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.norm",
    "linear_attn.out_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "mtp.fc",
    "mtp.norm",
    "mtp.pre_fc_norm_embedding",
    "mtp.pre_fc_norm_hidden",
    "mtp.input_layernorm",
    "mtp.post_attention_layernorm",
    "mtp.self_attn.q_proj",
    "mtp.self_attn.k_proj",
    "mtp.self_attn.v_proj",
    "mtp.self_attn.o_proj",
    "mtp.self_attn.q_norm",
    "mtp.self_attn.k_norm",
    "mtp.mlp.gate_proj",
    "mtp.mlp.up_proj",
    "mtp.mlp.down_proj",
    "vision.patch_embed",
    "vision.pos_embed",
    "vision.blocks",
    "vision.merger",
)

STATE_IDS: tuple[str, ...] = ("K_state", "V_state", "C_state", "S")

DESIGN_DIMENSIONS: tuple[str, ...] = (
    "bit_width",
    "grouping",
    "scale",
    "outlier_policy",
)

FORMATS: tuple[str, ...] = (
    "source",
    "bf16",
    "fp8_e4m3",
    "int8",
    "int6",
    "int4",
    "int3",
    "int2",
)

IEEE_LIKE_FORMATS: tuple[str, ...] = ("source", "bf16", "fp8_e4m3")
INTEGER_FORMATS: tuple[str, ...] = ("int8", "int6", "int4", "int3", "int2")

GROUPING_IDS: tuple[str, ...] = (
    "none",
    "per_tensor",
    "per_row",
    "per_col",
    "block_g32",
    "block_g64",
    "block_g128",
)

BLOCK_GROUP_SIZES: tuple[int, ...] = (32, 64, 128)

SCALE_IDS: tuple[str, ...] = (
    "none",
    "symmetric_absmax",
    "symmetric_rms",
    "asymmetric_minmax",
    "percentile_p99",
)

OUTLIER_IDS: tuple[str, ...] = ("none", "clip", "extract_high", "mixed_group")

SCALE_STORAGE_BYTES_CANDIDATES: tuple[int, ...] = (2, 4)

RECIPES: tuple[tuple[str, str, str, str, str], ...] = (
    ("keep_source", "source", "none", "none", "none"),
    ("narrow_bf16", "bf16", "none", "none", "none"),
    ("fp8_tensor", "fp8_e4m3", "per_tensor", "none", "none"),
    ("i8_tensor", "int8", "per_tensor", "symmetric_absmax", "none"),
    ("i8_row", "int8", "per_row", "symmetric_absmax", "none"),
    ("i8_row_asym", "int8", "per_row", "asymmetric_minmax", "none"),
    ("i8_g32", "int8", "block_g32", "symmetric_absmax", "none"),
    ("i6_row", "int6", "per_row", "symmetric_absmax", "none"),
    ("i4_row", "int4", "per_row", "symmetric_absmax", "none"),
    ("i4_col", "int4", "per_col", "symmetric_absmax", "none"),
    ("i4_g32", "int4", "block_g32", "symmetric_absmax", "none"),
    ("i4_g64", "int4", "block_g64", "symmetric_absmax", "none"),
    ("i4_g128", "int4", "block_g128", "symmetric_absmax", "none"),
    ("i4_g128_p99", "int4", "block_g128", "percentile_p99", "none"),
    ("i4_g128_rms", "int4", "block_g128", "symmetric_rms", "none"),
    ("i4_g128_extract", "int4", "block_g128", "symmetric_absmax", "extract_high"),
    ("i4_g32_mixed", "int4", "block_g32", "symmetric_absmax", "mixed_group"),
    ("i4_row_extract", "int4", "per_row", "symmetric_absmax", "extract_high"),
    ("i4_clip", "int4", "block_g128", "symmetric_absmax", "clip"),
    ("i3_g32", "int3", "block_g32", "symmetric_absmax", "none"),
    ("i3_g32_extract", "int3", "block_g32", "symmetric_absmax", "extract_high"),
    ("i2_g32_extract", "int2", "block_g32", "symmetric_absmax", "extract_high"),
)

CANDIDATE_RECIPE_IDS: tuple[str, ...] = tuple(row[0] for row in RECIPES)
RECIPE_FORMATS: tuple[str, ...] = tuple(row[1] for row in RECIPES)
RECIPE_GROUPINGS: tuple[str, ...] = tuple(row[2] for row in RECIPES)
RECIPE_SCALES: tuple[str, ...] = tuple(row[3] for row in RECIPES)
RECIPE_OUTLIERS: tuple[str, ...] = tuple(row[4] for row in RECIPES)

C_GEMM_LARGE: tuple[str, ...] = (
    "keep_source",
    "fp8_tensor",
    "i8_row",
    "i8_row_asym",
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
    "i3_g32",
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

POLICY_FAMILY_MEMBERS: dict[str, tuple[str, ...]] = {
    "norm_gamma": (
        "input_layernorm",
        "post_attention_layernorm",
        "final_norm",
        "linear_attn.norm",
        "self_attn.q_norm",
        "self_attn.k_norm",
        "mtp.norm",
        "mtp.pre_fc_norm_embedding",
        "mtp.pre_fc_norm_hidden",
        "mtp.input_layernorm",
        "mtp.post_attention_layernorm",
        "mtp.self_attn.q_norm",
        "mtp.self_attn.k_norm",
    ),
    "gdn_time_param": ("linear_attn.A_log", "linear_attn.dt_bias"),
    "gdn_gate_proj": ("linear_attn.in_proj_a", "linear_attn.in_proj_b"),
    "conv1d": ("linear_attn.conv1d",),
    "linear_large_proj": (
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
    ),
    "attn_qkv": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "mtp.self_attn.q_proj",
        "mtp.self_attn.k_proj",
        "mtp.self_attn.v_proj",
    ),
    "attn_out": ("self_attn.o_proj", "mtp.self_attn.o_proj"),
    "mlp_up_gate": (
        "mlp.gate_proj",
        "mlp.up_proj",
        "mtp.mlp.gate_proj",
        "mtp.mlp.up_proj",
    ),
    "mlp_down": ("mlp.down_proj", "mtp.mlp.down_proj"),
    "embed_table": ("embed",),
    "lm_head": ("lm_head",),
    "mtp_fc": ("mtp.fc",),
    "vision_deferred": (
        "vision.patch_embed",
        "vision.pos_embed",
        "vision.blocks",
        "vision.merger",
    ),
    "state_kv": ("K_state", "V_state"),
    "state_c": ("C_state",),
    "state_s": ("S",),
}

FAMILY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "norm_gamma": ("keep_source", "i8_tensor", "i8_g32"),
    "gdn_time_param": ("keep_source", "i8_tensor"),
    "gdn_gate_proj": (
        "keep_source",
        "i8_tensor",
        "i8_row",
        "i6_row",
        "i4_row",
        "i4_g128",
        "i4_g128_extract",
    ),
    "conv1d": (
        "keep_source",
        "i8_row",
        "i4_row",
        "i4_g32",
        "i4_g128_extract",
        "i4_clip",
    ),
    "linear_large_proj": C_GEMM_LARGE,
    "attn_qkv": C_GEMM_LARGE,
    "attn_out": C_GEMM_LARGE,
    "mlp_up_gate": C_GEMM_LARGE + ("i3_g32_extract", "i2_g32_extract"),
    "mlp_down": C_GEMM_LARGE,
    "embed_table": (
        "keep_source",
        "i8_row",
        "i4_row",
        "i4_g128",
        "i4_row_extract",
    ),
    "lm_head": (
        "keep_source",
        "i8_row",
        "i8_row_asym",
        "i6_row",
        "i4_row",
        "i4_g128",
        "i4_g128_extract",
    ),
    "mtp_fc": ("keep_source", "i8_row", "i4_row", "i4_col", "i4_g128"),
    "vision_deferred": (),
    "state_kv": ("keep_source", "fp8_tensor", "i8_tensor"),
    "state_c": ("keep_source", "i8_tensor"),
    "state_s": ("keep_source", "narrow_bf16", "fp8_tensor", "i8_tensor"),
}

FAMILY_CANDIDATE_COUNTS: tuple[int, ...] = (3, 2, 7, 6, 15, 15, 15, 17, 15, 5, 7, 5, 0, 3, 2, 4)

QUALITY_RISK_IDS: tuple[str, ...] = (
    "q_norm_gamma",
    "q_gdn_time",
    "q_gdn_gate",
    "q_conv",
    "q_linear_proj",
    "q_attn_qkv",
    "q_attn_out",
    "q_mlp_up",
    "q_mlp_down",
    "q_embed",
    "q_lm_head",
    "q_mtp_fc",
    "q_state_kv",
    "q_state_c",
    "q_state_s",
    "q_clip_tail",
    "q_int2_mass",
)

QUALITY_RISK_FAMILIES: tuple[str, ...] = (
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
    "state_kv",
    "state_c",
    "state_s",
    "clip",
    "mlp_up_gate",
)

QUALITY_RISK_SENSITIVE_OPS: tuple[str, ...] = (
    "rms_hidden",
    "gdn_alpha_beta",
    "gdn_alpha_beta",
    "conv_fir",
    "gemm_k5120",
    "gemm_k5120",
    "residual_stream",
    "gemm_k5120",
    "residual_stream",
    "param_bf16",
    "gemm_lm_head",
    "gemm_k5120",
    "state_kv_bf16",
    "state_c_bf16",
    "s_below_f32",
    "param_bf16",
    "param_bf16",
)

QUALITY_RISK_SEVERITIES: tuple[str, ...] = (
    "high",
    "high",
    "high",
    "low",
    "medium",
    "medium",
    "high",
    "medium",
    "high",
    "medium",
    "medium",
    "medium",
    "high",
    "low",
    "high",
    "medium",
    "high",
)

QUALITY_HIGH_IDS: tuple[str, ...] = (
    "q_norm_gamma",
    "q_gdn_time",
    "q_gdn_gate",
    "q_attn_out",
    "q_mlp_down",
    "q_state_kv",
    "q_state_s",
    "q_int2_mass",
)

QUALITY_MEDIUM_IDS: tuple[str, ...] = (
    "q_linear_proj",
    "q_attn_qkv",
    "q_mlp_up",
    "q_embed",
    "q_lm_head",
    "q_mtp_fc",
    "q_clip_tail",
)

QUALITY_LOW_IDS: tuple[str, ...] = ("q_conv", "q_state_c")

DECODE_RISK_IDS: tuple[str, ...] = (
    "d_weight_dequant",
    "d_fine_group_meta",
    "d_outlier_gather",
    "d_mixed_width",
    "d_embed_gather",
    "d_lm_head_unpack",
    "d_3bit_pack",
    "d_state_s_narrow",
    "d_kv_narrow",
)

DECODE_RISK_SEVERITIES: tuple[str, ...] = (
    "high",
    "medium",
    "medium",
    "low",
    "medium",
    "high",
    "medium",
    "high",
    "medium",
)

DECODE_HIGH_IDS: tuple[str, ...] = (
    "d_weight_dequant",
    "d_lm_head_unpack",
    "d_state_s_narrow",
)

DECODE_MEDIUM_IDS: tuple[str, ...] = (
    "d_fine_group_meta",
    "d_outlier_gather",
    "d_embed_gather",
    "d_3bit_pack",
    "d_kv_narrow",
)

DECODE_LOW_IDS: tuple[str, ...] = ("d_mixed_width",)

DIAGRAM_REQUIRED_IDS: tuple[str, ...] = (
    "bit_width",
    "grouping",
    "scale",
    "outlier",
    "param",
    "state",
    "quality",
    "decode",
)

CIT_LANGUAGE_MTP_ABSMAX = 19.25
CIT_DT_BIAS_ABSMAX = 19.25
CIT_A_LOG_ABSMAX = 5.5625
CIT_CONV1D_FRAC_OUT_6X = 0.1805953979492
CIT_EMBED_ROW_RATIO = 11.74384236453
CIT_LM_HEAD_ROW_RATIO = 5.223880597015
CIT_OUT_PROJ_MEAN_ROW_RATIO = 15.3244687679
CIT_O_PROJ_MEAN_ROW_RATIO = 13.59971022497
CIT_DOWN_PROJ_MAX_ROW_RATIO = 25.73544973545
CIT_MLP_GATE_FRAC_OUT_6X = 0.0003818664480658
CIT_EMBED_N_ZERO = 7528

LOCKED_DOCUMENT_TOKENS: tuple[str, ...] = (
    "5120",
    "17408",
    "248320",
    "17112760320",
    "34225520640",
    "2542796800",
    "54641395712",
    "52098598912",
    "150994944",
    "69632",
    "2949120",
    "19.25",
    "0.28125",
    "0.2578125",
    "8556380160",
    "9625927680",
    "8823767040",
    "10240",
    "2562",
    "0.1805953979492",
    "11.74384236453",
    "25.73544973545",
    "15.3244687679",
    "7528",
)

FORBIDDEN_WINNER_PHRASES: tuple[str, ...] = (
    "should be 4-bit",
    "should be 8-bit",
    "quantization winner",
    "recommend 4-bit",
    "recommend 8-bit",
    "selected winner",
    "Pareto frontier is",
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
    "mlp_n",
    "embed_n",
    "n_language_mtp_parameters",
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
    "s_f32_bytes",
    "s_bf16_bytes",
    "s_fp8_bytes",
    "kv_bytes_all_per_token",
    "c_bytes_all",
    "scale_storage_illustration_bytes",
    "scale_storage_bytes_candidates",
    "ragged_last_group",
    "group_clips_to_axis",
    "payload_formula_allows_fractional_bytes",
    "cit_language_mtp_absmax",
    "cit_dt_bias_absmax",
    "cit_a_log_absmax",
    "cit_conv1d_frac_out_6x",
    "cit_embed_row_ratio",
    "cit_lm_head_row_ratio",
    "cit_out_proj_mean_row_ratio",
    "cit_o_proj_mean_row_ratio",
    "cit_down_proj_max_row_ratio",
    "cit_mlp_gate_frac_out_6x",
    "cit_embed_n_zero",
    "design_dimensions",
    "formats",
    "ieee_like_formats",
    "integer_formats",
    "grouping_ids",
    "block_group_sizes",
    "scale_ids",
    "outlier_ids",
    "candidate_recipe_ids",
    "recipe_formats",
    "recipe_groupings",
    "recipe_scales",
    "recipe_outliers",
    "c_gemm_large",
    "n_candidate_recipes",
    "n_design_dimensions",
    "policy_families",
    "policy_family_members",
    "family_candidates",
    "family_candidate_counts",
    "n_policy_families",
    "n_level2_mapped",
    "n_state_mapped",
    "keep_source_on_all_defined_families",
    "quality_risk_ids",
    "quality_risk_families",
    "quality_risk_sensitive_ops",
    "quality_risk_severities",
    "quality_high_ids",
    "quality_medium_ids",
    "quality_low_ids",
    "n_quality_risks",
    "n_quality_high",
    "n_quality_medium",
    "n_quality_low",
    "decode_risk_ids",
    "decode_risk_severities",
    "decode_high_ids",
    "decode_medium_ids",
    "decode_low_ids",
    "n_decode_risks",
    "n_decode_high",
    "n_decode_medium",
    "n_decode_low",
    "gguf_is_not_a_recipe",
    "pareto_frontier_selected",
    "activation_quant_in_family_policies",
    "payloads_restreamed",
    "diagram_ids",
    "n_diagrams",
    "canonical_sentence_logical",
    "canonical_sentence_candidates",
    "canonical_sentence_hypothesis",
    "canonical_sentence_metadata",
)

REQUIRED_HEADINGS: tuple[str, ...] = (
    "Authority",
    "Research-space convention",
    "Design dimensions",
    "Metadata and compute implications",
    "Source evidence",
    "Policy families",
    "Family-specific candidate policies",
    "Quality and decode-complexity risks",
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


class QuantizationDesignSpaceMismatch(Exception):
    """Raised when the quantization-design-space markdown fails a content check."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def dumps_summary(summary: dict) -> str:
    """Pretty-print the quantization-design-space summary object.

    Args:
        summary: Object produced by :func:`instantiate_quantization_design_space`.

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


def _integer_storage_bytes(
    n_elements: int,
    code_bits: int,
    group_size: int,
    scale_bytes: int,
    zero_point_bytes: int,
) -> tuple[int, int, int, int]:
    """Return payload, group count, metadata, and total storage lower bounds.

    Args:
        n_elements: Element count n.
        code_bits: Integer code width b.
        group_size: Group size g.
        scale_bytes: Scale storage bytes s.
        zero_point_bytes: Zero-point bytes z.

    Returns:
        ``(B_payload, n_g, B_meta, B)`` as ints when the payload is an integer
        number of bytes; payload may be a float for 3-bit/6-bit formulas, but
        this helper is used only for even-byte illustrations.

    Raises:
        AssertionError: If the payload is not an integer byte count.
    """

    payload = n_elements * code_bits / 8
    assert payload == int(payload), f"payload bytes not integral: {payload}"
    payload_bytes = int(payload)
    n_g = _n_groups(n_elements, group_size)
    meta_bytes = n_g * (scale_bytes + zero_point_bytes)
    return payload_bytes, n_g, meta_bytes, payload_bytes + meta_bytes


def instantiate_quantization_design_space(config: dict) -> dict:
    """Compute the TASK-08 quantization-design-space summary from ``config``.

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
    (
        mlp_int4_payload_bytes,
        _mlp_g32_n_g,
        mlp_int4_g32_meta_bytes,
        mlp_int4_g32_total_bytes,
    ) = _integer_storage_bytes(mlp_n, 4, 32, scale_s, 0)
    (
        _mlp_g128_payload,
        _mlp_g128_n_g,
        mlp_int4_g128_meta_bytes,
        mlp_int4_g128_total_bytes,
    ) = _integer_storage_bytes(mlp_n, 4, 128, scale_s, 0)
    mlp_int4_g32_over_bf16 = mlp_int4_g32_total_bytes / mlp_bf16_bytes
    mlp_int4_g128_over_bf16 = mlp_int4_g128_total_bytes / mlp_bf16_bytes

    (
        _une_payload,
        _une_n_g,
        _une_meta,
        unique_non_embed_int4_g128_total_bytes,
    ) = _integer_storage_bytes(unique_non_embed_n, 4, 128, scale_s, 0)
    unique_non_embed_int4_g128_over_bf16 = (
        unique_non_embed_int4_g128_total_bytes / unique_non_embed_bf16_bytes
    )

    embed_gather_bf16_bytes = hidden_size * BYTES_BF16
    embed_gather_int4_row_bytes = (
        hidden_size * 4 // 8
    ) + SCALE_STORAGE_ILLUSTRATION_BYTES

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

    policy_family_members = {
        family_id: list(POLICY_FAMILY_MEMBERS[family_id])
        for family_id in POLICY_FAMILIES
    }
    family_candidates = {
        family_id: list(FAMILY_CANDIDATES[family_id])
        for family_id in POLICY_FAMILIES
    }
    keep_source_on_all_defined = all(
        "keep_source" in FAMILY_CANDIDATES[family_id]
        for family_id in POLICY_FAMILIES
        if family_id != "vision_deferred"
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
        "full_attention_indices": full_from_types,
        "bytes_bf16": BYTES_BF16,
        "bytes_f32": BYTES_F32,
        "bytes_fp8": BYTES_FP8,
        "mlp_n": mlp_n,
        "embed_n": embed_n,
        "n_language_mtp_parameters": n_language_mtp_parameters,
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
        "s_f32_bytes": s_f32_bytes,
        "s_bf16_bytes": s_bf16_bytes,
        "s_fp8_bytes": s_fp8_bytes,
        "kv_bytes_all_per_token": kv_bytes_all_per_token,
        "c_bytes_all": c_bytes_all,
        "scale_storage_illustration_bytes": SCALE_STORAGE_ILLUSTRATION_BYTES,
        "scale_storage_bytes_candidates": list(SCALE_STORAGE_BYTES_CANDIDATES),
        "ragged_last_group": True,
        "group_clips_to_axis": True,
        "payload_formula_allows_fractional_bytes": True,
        "cit_language_mtp_absmax": CIT_LANGUAGE_MTP_ABSMAX,
        "cit_dt_bias_absmax": CIT_DT_BIAS_ABSMAX,
        "cit_a_log_absmax": CIT_A_LOG_ABSMAX,
        "cit_conv1d_frac_out_6x": CIT_CONV1D_FRAC_OUT_6X,
        "cit_embed_row_ratio": CIT_EMBED_ROW_RATIO,
        "cit_lm_head_row_ratio": CIT_LM_HEAD_ROW_RATIO,
        "cit_out_proj_mean_row_ratio": CIT_OUT_PROJ_MEAN_ROW_RATIO,
        "cit_o_proj_mean_row_ratio": CIT_O_PROJ_MEAN_ROW_RATIO,
        "cit_down_proj_max_row_ratio": CIT_DOWN_PROJ_MAX_ROW_RATIO,
        "cit_mlp_gate_frac_out_6x": CIT_MLP_GATE_FRAC_OUT_6X,
        "cit_embed_n_zero": CIT_EMBED_N_ZERO,
        "design_dimensions": list(DESIGN_DIMENSIONS),
        "formats": list(FORMATS),
        "ieee_like_formats": list(IEEE_LIKE_FORMATS),
        "integer_formats": list(INTEGER_FORMATS),
        "grouping_ids": list(GROUPING_IDS),
        "block_group_sizes": list(BLOCK_GROUP_SIZES),
        "scale_ids": list(SCALE_IDS),
        "outlier_ids": list(OUTLIER_IDS),
        "candidate_recipe_ids": list(CANDIDATE_RECIPE_IDS),
        "recipe_formats": list(RECIPE_FORMATS),
        "recipe_groupings": list(RECIPE_GROUPINGS),
        "recipe_scales": list(RECIPE_SCALES),
        "recipe_outliers": list(RECIPE_OUTLIERS),
        "c_gemm_large": list(C_GEMM_LARGE),
        "n_candidate_recipes": len(CANDIDATE_RECIPE_IDS),
        "n_design_dimensions": len(DESIGN_DIMENSIONS),
        "policy_families": list(POLICY_FAMILIES),
        "policy_family_members": policy_family_members,
        "family_candidates": family_candidates,
        "family_candidate_counts": list(FAMILY_CANDIDATE_COUNTS),
        "n_policy_families": len(POLICY_FAMILIES),
        "n_level2_mapped": len(LEVEL2_IDS),
        "n_state_mapped": len(STATE_IDS),
        "keep_source_on_all_defined_families": keep_source_on_all_defined,
        "quality_risk_ids": list(QUALITY_RISK_IDS),
        "quality_risk_families": list(QUALITY_RISK_FAMILIES),
        "quality_risk_sensitive_ops": list(QUALITY_RISK_SENSITIVE_OPS),
        "quality_risk_severities": list(QUALITY_RISK_SEVERITIES),
        "quality_high_ids": list(QUALITY_HIGH_IDS),
        "quality_medium_ids": list(QUALITY_MEDIUM_IDS),
        "quality_low_ids": list(QUALITY_LOW_IDS),
        "n_quality_risks": len(QUALITY_RISK_IDS),
        "n_quality_high": len(QUALITY_HIGH_IDS),
        "n_quality_medium": len(QUALITY_MEDIUM_IDS),
        "n_quality_low": len(QUALITY_LOW_IDS),
        "decode_risk_ids": list(DECODE_RISK_IDS),
        "decode_risk_severities": list(DECODE_RISK_SEVERITIES),
        "decode_high_ids": list(DECODE_HIGH_IDS),
        "decode_medium_ids": list(DECODE_MEDIUM_IDS),
        "decode_low_ids": list(DECODE_LOW_IDS),
        "n_decode_risks": len(DECODE_RISK_IDS),
        "n_decode_high": len(DECODE_HIGH_IDS),
        "n_decode_medium": len(DECODE_MEDIUM_IDS),
        "n_decode_low": len(DECODE_LOW_IDS),
        "gguf_is_not_a_recipe": True,
        "pareto_frontier_selected": False,
        "activation_quant_in_family_policies": False,
        "payloads_restreamed": False,
        "diagram_ids": list(DIAGRAM_REQUIRED_IDS),
        "n_diagrams": 1,
        "canonical_sentence_logical": CANONICAL_SENTENCE_LOGICAL,
        "canonical_sentence_candidates": CANONICAL_SENTENCE_CANDIDATES,
        "canonical_sentence_hypothesis": CANONICAL_SENTENCE_HYPOTHESIS,
        "canonical_sentence_metadata": CANONICAL_SENTENCE_METADATA,
    }
    _assert_identities(summary, text)
    return summary


def _assert_identities(summary: dict, text: dict) -> None:
    """Assert dossier identities on a live quantization-design-space object.

    Args:
        summary: Object produced by :func:`instantiate_quantization_design_space`.
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
    assert summary["mlp_bf16_bytes"] == 34225520640
    assert summary["mlp_bf16_bytes"] == 2 * summary["mlp_n"]
    assert summary["mlp_int4_g32_over_bf16"] == 0.28125
    assert summary["mlp_int4_g128_over_bf16"] == 0.2578125
    assert summary["unique_non_embed_int4_g128_over_bf16"] == 0.2578125
    assert summary["weight_bytes_language_mtp_excl_vision"] == 54641395712
    assert summary["weight_bytes_unique_non_embed"] == 52098598912
    assert summary["s_f32_bytes"] == 150994944
    assert summary["kv_bytes_all_per_token"] == 69632
    assert summary["c_bytes_all"] == 2949120
    assert summary["n_design_dimensions"] == 4
    assert summary["n_candidate_recipes"] == 22
    assert summary["n_policy_families"] == 16
    assert summary["n_quality_risks"] == 17
    assert summary["n_quality_high"] == 8
    assert summary["n_quality_medium"] == 7
    assert summary["n_quality_low"] == 2
    assert summary["n_decode_risks"] == 9
    assert summary["n_decode_high"] == 3
    assert summary["n_decode_medium"] == 5
    assert summary["n_decode_low"] == 1
    assert summary["n_level2_mapped"] == 42
    assert summary["n_state_mapped"] == 4
    assert summary["n_diagrams"] == 1
    assert summary["candidate_recipe_ids"] == list(CANDIDATE_RECIPE_IDS)
    assert summary["recipe_formats"] == list(RECIPE_FORMATS)
    assert summary["recipe_groupings"] == list(RECIPE_GROUPINGS)
    assert summary["recipe_scales"] == list(RECIPE_SCALES)
    assert summary["recipe_outliers"] == list(RECIPE_OUTLIERS)
    assert summary["c_gemm_large"] == list(C_GEMM_LARGE)
    assert summary["policy_families"] == list(POLICY_FAMILIES)
    assert summary["family_candidate_counts"] == list(FAMILY_CANDIDATE_COUNTS)
    for family_id in POLICY_FAMILIES:
        assert summary["family_candidates"][family_id] == list(
            FAMILY_CANDIDATES[family_id]
        )
        assert summary["policy_family_members"][family_id] == list(
            POLICY_FAMILY_MEMBERS[family_id]
        )
    flattened_param: list[str] = []
    flattened_state: list[str] = []
    for family_id in POLICY_FAMILIES:
        members = POLICY_FAMILY_MEMBERS[family_id]
        if family_id in {"state_kv", "state_c", "state_s"}:
            flattened_state.extend(members)
        else:
            flattened_param.extend(members)
    assert len(flattened_param) == 42
    assert set(flattened_param) == set(LEVEL2_IDS)
    assert flattened_state == list(STATE_IDS)
    assert summary["family_candidates"]["vision_deferred"] == []
    for family_id, candidates in summary["family_candidates"].items():
        if family_id == "vision_deferred":
            continue
        assert "keep_source" in candidates, f"{family_id} missing keep_source"
    assert summary["keep_source_on_all_defined_families"] is True
    narrow_families = [
        family_id
        for family_id, candidates in summary["family_candidates"].items()
        if "narrow_bf16" in candidates
    ]
    assert narrow_families == ["state_s"]
    i2_families = [
        family_id
        for family_id, candidates in summary["family_candidates"].items()
        if "i2_g32_extract" in candidates
    ]
    assert i2_families == ["mlp_up_gate"]
    clip_families = [
        family_id
        for family_id, candidates in summary["family_candidates"].items()
        if "i4_clip" in candidates
    ]
    assert clip_families == ["conv1d"]
    assert summary["gguf_is_not_a_recipe"] is True
    assert summary["pareto_frontier_selected"] is False
    assert summary["activation_quant_in_family_policies"] is False
    assert summary["payloads_restreamed"] is False
    assert summary["cit_language_mtp_absmax"] == 19.25
    assert summary["cit_dt_bias_absmax"] == 19.25
    assert summary["cit_conv1d_frac_out_6x"] == 0.1805953979492
    assert summary["cit_a_log_absmax"] == 5.5625
    assert summary["weight_bytes_language_mlp"] == 34225520640
    assert summary["weight_bytes_lm_head"] == 2542796800
    assert summary["weight_bytes_embed_table"] == 2542796800
    assert summary["weight_bytes_language_linear_attn"] == 11124102144
    assert summary["weight_bytes_language_self_attn"] == 3355459584
    assert summary["n_language_mtp_parameters"] == 27320697856
    assert summary["mlp_int4_payload_bytes"] == 8556380160
    assert summary["mlp_int4_g32_meta_bytes"] == 1069547520
    assert summary["mlp_int4_g32_total_bytes"] == 9625927680
    assert summary["mlp_int4_g128_meta_bytes"] == 267386880
    assert summary["mlp_int4_g128_total_bytes"] == 8823767040
    assert summary["unique_non_embed_n"] == 26049299456
    assert summary["unique_non_embed_bf16_bytes"] == 52098598912
    assert summary["unique_non_embed_int4_g128_total_bytes"] == 13431670032
    assert summary["embed_gather_bf16_bytes"] == 10240
    assert summary["embed_gather_int4_row_bytes"] == 2562
    assert summary["s_bf16_bytes"] == 75497472
    assert summary["s_fp8_bytes"] == 37748736
    assert summary["quality_risk_ids"] == list(QUALITY_RISK_IDS)
    assert summary["quality_risk_families"] == list(QUALITY_RISK_FAMILIES)
    assert summary["quality_risk_sensitive_ops"] == list(QUALITY_RISK_SENSITIVE_OPS)
    assert summary["quality_risk_severities"] == list(QUALITY_RISK_SEVERITIES)
    assert summary["quality_high_ids"] == list(QUALITY_HIGH_IDS)
    assert summary["quality_medium_ids"] == list(QUALITY_MEDIUM_IDS)
    assert summary["quality_low_ids"] == list(QUALITY_LOW_IDS)
    assert summary["decode_risk_ids"] == list(DECODE_RISK_IDS)
    assert summary["decode_risk_severities"] == list(DECODE_RISK_SEVERITIES)
    assert summary["decode_high_ids"] == list(DECODE_HIGH_IDS)
    assert summary["decode_medium_ids"] == list(DECODE_MEDIUM_IDS)
    assert summary["decode_low_ids"] == list(DECODE_LOW_IDS)
    high_from_ops = [
        risk_id
        for risk_id, severity in zip(
            summary["quality_risk_ids"], summary["quality_risk_severities"]
        )
        if severity == "high"
    ]
    medium_from_ops = [
        risk_id
        for risk_id, severity in zip(
            summary["quality_risk_ids"], summary["quality_risk_severities"]
        )
        if severity == "medium"
    ]
    low_from_ops = [
        risk_id
        for risk_id, severity in zip(
            summary["quality_risk_ids"], summary["quality_risk_severities"]
        )
        if severity == "low"
    ]
    assert high_from_ops == summary["quality_high_ids"]
    assert medium_from_ops == summary["quality_medium_ids"]
    assert low_from_ops == summary["quality_low_ids"]
    decode_high = [
        risk_id
        for risk_id, severity in zip(
            summary["decode_risk_ids"], summary["decode_risk_severities"]
        )
        if severity == "high"
    ]
    decode_medium = [
        risk_id
        for risk_id, severity in zip(
            summary["decode_risk_ids"], summary["decode_risk_severities"]
        )
        if severity == "medium"
    ]
    decode_low = [
        risk_id
        for risk_id, severity in zip(
            summary["decode_risk_ids"], summary["decode_risk_severities"]
        )
        if severity == "low"
    ]
    assert decode_high == summary["decode_high_ids"]
    assert decode_medium == summary["decode_medium_ids"]
    assert decode_low == summary["decode_low_ids"]
    assert summary["design_dimensions"] == list(DESIGN_DIMENSIONS)
    assert summary["formats"] == list(FORMATS)
    assert summary["ieee_like_formats"] == list(IEEE_LIKE_FORMATS)
    assert summary["integer_formats"] == list(INTEGER_FORMATS)
    assert summary["grouping_ids"] == list(GROUPING_IDS)
    assert summary["block_group_sizes"] == list(BLOCK_GROUP_SIZES)
    assert summary["scale_ids"] == list(SCALE_IDS)
    assert summary["outlier_ids"] == list(OUTLIER_IDS)
    assert summary["scale_storage_bytes_candidates"] == list(
        SCALE_STORAGE_BYTES_CANDIDATES
    )
    assert summary["ragged_last_group"] is True
    assert summary["group_clips_to_axis"] is True
    assert summary["payload_formula_allows_fractional_bytes"] is True
    assert summary["bytes_bf16"] == 2
    assert summary["bytes_f32"] == 4
    assert summary["bytes_fp8"] == 1
    assert summary["scale_storage_illustration_bytes"] == 2
    assert summary["diagram_ids"] == list(DIAGRAM_REQUIRED_IDS)
    assert summary["canonical_sentence_logical"] == CANONICAL_SENTENCE_LOGICAL
    assert summary["canonical_sentence_candidates"] == CANONICAL_SENTENCE_CANDIDATES
    assert summary["canonical_sentence_hypothesis"] == CANONICAL_SENTENCE_HYPOTHESIS
    assert summary["canonical_sentence_metadata"] == CANONICAL_SENTENCE_METADATA
    assert len(C_GEMM_LARGE) == 15
    assert len(FAMILY_CANDIDATES["mlp_up_gate"]) == 17
    for family_id, count in zip(POLICY_FAMILIES, FAMILY_CANDIDATE_COUNTS):
        assert len(FAMILY_CANDIDATES[family_id]) == count


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
    """Compare documented JSON against a live quantization-design-space object.

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


def _winner_phrase_allowed(text: str, start: int, phrase: str) -> bool:
    """Return whether a forbidden-winner match is the required example wording.

    The locked example cell must contain ``not a selected winner``, which
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


def check_quantization_design_space(live: dict, design_path: Path) -> None:
    """Check a quantization-design-space markdown file against a live summary.

    Args:
        live: Quantization-design-space summary object from sitting config.
        design_path: Path to ``quantization-design-space.md``.

    Raises:
        QuantizationDesignSpaceMismatch: On heading, JSON, diagram, or token
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
    if CANONICAL_SENTENCE_CANDIDATES not in text:
        differences.append("canonical_sentence_candidates not present verbatim")
    if CANONICAL_SENTENCE_HYPOTHESIS not in text:
        differences.append("canonical_sentence_hypothesis not present verbatim")
    if CANONICAL_SENTENCE_METADATA not in text:
        differences.append("canonical_sentence_metadata not present verbatim")

    for family_id in POLICY_FAMILIES:
        if family_id not in text:
            differences.append(f"policy family {family_id!r} missing as substring")
    for recipe_id in CANDIDATE_RECIPE_IDS:
        if recipe_id not in text:
            differences.append(f"recipe id {recipe_id!r} missing as substring")
    for risk_id in QUALITY_RISK_IDS:
        if risk_id not in text:
            differences.append(f"quality risk {risk_id!r} missing as substring")
    for risk_id in DECODE_RISK_IDS:
        if risk_id not in text:
            differences.append(f"decode risk {risk_id!r} missing as substring")
    for dim_id in DESIGN_DIMENSIONS:
        if dim_id not in text:
            differences.append(f"design dimension {dim_id!r} missing as substring")

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
        raise QuantizationDesignSpaceMismatch(differences)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the quantization-design-space checker.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Instantiate Qwen3.8-27B language+MTP quantization-design-space "
            "summary from text_config and check "
            "docs/architecture/quantization-design-space.md."
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
        help="Write quantization-design-space summary JSON to stdout.",
    )
    parser.add_argument(
        "--quantization-design-space",
        type=Path,
        default=None,
        help="Markdown path whose first json fence must match live summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the quantization-design-space checker CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch, 2 missing
        config.
    """

    args = _parse_args(argv)
    try:
        config = load_config(args.config)
        summary = instantiate_quantization_design_space(config)
        if args.quantization_design_space is not None:
            if not args.quantization_design_space.is_file():
                print(
                    f"quantization-design-space file not found: "
                    f"{args.quantization_design_space}",
                    file=sys.stderr,
                )
                return 1
            check_quantization_design_space(summary, args.quantization_design_space)
    except MissingConfig as exc:
        print(str(exc.path), file=sys.stderr)
        return 2
    except QuantizationDesignSpaceMismatch as exc:
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

    if args.json or args.quantization_design_space is None:
        sys.stdout.write(dumps_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
