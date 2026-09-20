#!/usr/bin/env python3
"""Check Qwen3.8-27B quantization validation against sitting text_config.

Reads ``text_config`` from a Transformers ``config.json`` (no safetensor
payloads) and either prints the quantization-validation summary object or
checks that ``docs/architecture/quantization-validation.md`` matches it.
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

CANONICAL_SENTENCE_RECONSTRUCTION = (
    "Reconstruction diagnostics in this document are local screens, not "
    "model-level quality."
)

CANONICAL_SENTENCE_COMPARISONS = (
    "Teacher-forced and behavioral comparisons in this document are a "
    "methodology, not measurements."
)

CANONICAL_SENTENCE_Q4KM = (
    "Q4_K_M is a future black-box Pareto reference, not a requirement."
)

METHODOLOGY_QUESTION_SENTENCE = (
    "Reconstruction is a local screen with named limits; teacher-forced "
    "complete-map NLL versus keep_source is the primary quality comparison; "
    "Q4_K_M is a future black-box Pareto reference, not a requirement; final "
    "corpora, prompt suite, capability benchmarks, and acceptance frontier "
    "remain unselected."
)

BYTES_BF16 = 2
BYTES_F32 = 4
SCALE_STORAGE_ILLUSTRATION_BYTES = 2
N_LANGUAGE_MTP_TENSORS = 866
Q4KM_NAMED_PATH = "models/Qwen3.8-27B-Q4_K_M.gguf"

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

EXAMPLE_T_VALUES: tuple[int, ...] = (1, 4096)

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

SURVIVAL_RECONSTRUCTION_IDS: tuple[str, ...] = (
    "r_param_mse",
    "r_state_s",
    "r_state_s",
    "r_act_residual",
    "r_act_residual",
    "r_state_kv",
    "r_state_s",
    "r_param_mse",
)

SENSITIVE_HIGH_IDS: tuple[str, ...] = (
    "residual_stream",
    "rms_hidden",
    "softmax_over_T",
    "attn_av_over_T",
    "gdn_S_recurrent",
    "gdn_alpha_beta",
    "rope_phase",
    "state_kv_bf16",
    "s_below_f32",
)

EVAL_LAYER_IDS: tuple[str, ...] = (
    "reconstruction",
    "teacher_forced",
    "behavioral",
)

RECONSTRUCTION_DIAGNOSTIC_IDS: tuple[str, ...] = (
    "r_param_mse",
    "r_param_maxabs",
    "r_param_cosine",
    "r_clip_frac",
    "r_act_residual",
    "r_act_logits",
    "r_state_kv",
    "r_state_s",
)

RECONSTRUCTION_DIAGNOSTIC_ROLES: tuple[str, ...] = (
    "param",
    "param",
    "param",
    "param",
    "activation",
    "activation",
    "state",
    "state",
)

RECONSTRUCTION_LIMIT_IDS: tuple[str, ...] = (
    "lim_local_not_nll",
    "lim_residual_accum",
    "lim_gdn_horizon",
    "lim_identity_control",
    "lim_gguf_not_target",
    "lim_single_family",
)

TEACHER_FORCED_METRIC_IDS: tuple[str, ...] = (
    "tf_nll_language",
    "tf_nll_mtp",
    "tf_nll_complete",
    "tf_delta_vs_control",
    "tf_kl_vs_control",
)

BEHAVIORAL_CLASS_IDS: tuple[str, ...] = (
    "beh_greedy_prefix",
    "beh_prompt_suite",
    "beh_capability",
)

PARETO_AXIS_IDS: tuple[str, ...] = (
    "axis_quality",
    "axis_compression",
    "axis_reference_q4km",
)

CORPUS_CLASS_IDS: tuple[str, ...] = (
    "corpus_calibration",
    "corpus_eval_nll",
    "corpus_prompt",
    "corpus_capability",
)

METHODOLOGY_RISK_IDS: tuple[str, ...] = (
    "v_recon_as_quality",
    "v_gguf_as_requirement",
    "v_calib_eval_leak",
    "v_control_as_winner",
    "v_nll_without_mtp",
    "v_toks_as_quality",
)

METHODOLOGY_RISK_SEVERITIES: tuple[str, ...] = (
    "high",
    "high",
    "high",
    "medium",
    "medium",
    "low",
)

METHODOLOGY_HIGH_IDS: tuple[str, ...] = (
    "v_recon_as_quality",
    "v_gguf_as_requirement",
    "v_calib_eval_leak",
)

METHODOLOGY_MEDIUM_IDS: tuple[str, ...] = (
    "v_control_as_winner",
    "v_nll_without_mtp",
)

METHODOLOGY_LOW_IDS: tuple[str, ...] = ("v_toks_as_quality",)

DIAGRAM_REQUIRED_IDS: tuple[str, ...] = (
    "reconstruction",
    "teacher_forced",
    "behavioral",
    "q4km",
    "control",
    "pareto",
    "calibration",
    "eval",
)

LOCKED_DOCUMENT_TOKENS: tuple[str, ...] = (
    "5120",
    "17408",
    "248320",
    "866",
    "27320697856",
    "17112760320",
    "34225520640",
    "54641395712",
    "52098598912",
    "150994944",
    "13431670032",
    "0.2578125",
    "262144",
    "4096",
    "128",
    "130",
)

FORBIDDEN_WINNER_PHRASES: tuple[str, ...] = (
    "selected winner",
    "Pareto frontier is",
    "Q4_K_M is required",
    "must beat Q4_K_M",
    "must match Q4_K_M",
    "should use GGUF",
    "GGUF is the runtime format",
    "GGUF is the compiler input",
    "reconstruction is sufficient",
    "selected calibration corpus",
    "selected prompt suite",
    "selected capability benchmark",
    "acceptance frontier is",
    "should be 4-bit",
    "GPTQ is required",
    "quality winner",
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
    "T_max",
    "n_language_mtp_tensors",
    "n_language_mtp_parameters",
    "mlp_n",
    "embed_n",
    "mlp_bf16_bytes",
    "mlp_int4_g128_over_bf16",
    "weight_bytes_language_mtp_excl_vision",
    "weight_bytes_unique_non_embed",
    "unique_non_embed_int4_g128_total_bytes",
    "s_f32_bytes",
    "n_residual_adds_language",
    "n_residual_adds_complete",
    "example_T_values",
    "example_T_is_not_prompt_matrix",
    "eval_layer_ids",
    "n_eval_layers",
    "reconstruction_diagnostic_ids",
    "reconstruction_diagnostic_roles",
    "n_reconstruction_diagnostics",
    "reconstruction_limit_ids",
    "n_reconstruction_limits",
    "teacher_forced_metric_ids",
    "n_teacher_forced_metrics",
    "n_language_nll_events_offset",
    "n_mtp_nll_events_offset",
    "min_eval_tokens_language",
    "min_eval_tokens_mtp",
    "min_eval_tokens_complete",
    "mtp_in_primary_nll",
    "sampling_out_of_nll",
    "log_softmax_is_eval_readout",
    "behavioral_class_ids",
    "n_behavioral_classes",
    "pareto_axis_ids",
    "n_pareto_axes",
    "q4km_named_path",
    "corpus_class_ids",
    "n_corpus_classes",
    "calibration_eval_must_be_disjoint",
    "quality_high_ids",
    "n_quality_high",
    "survival_reconstruction_ids",
    "survival_requires_teacher_forced",
    "quality_risk_ids",
    "n_quality_risks",
    "sensitive_high_ids",
    "n_sensitive_high",
    "methodology_risk_ids",
    "methodology_risk_severities",
    "methodology_high_ids",
    "methodology_medium_ids",
    "methodology_low_ids",
    "n_methodology_risks",
    "n_methodology_high",
    "n_methodology_medium",
    "n_methodology_low",
    "policy_families",
    "candidate_recipe_ids",
    "n_policy_families",
    "n_candidate_recipes",
    "keep_source_packable_on_all_defined_families",
    "control_profile_id",
    "control_profile_is_keep_source",
    "reconstruction_is_not_quality",
    "teacher_forced_is_primary_quality",
    "behavioral_is_secondary_quality",
    "keep_source_is_identity_control",
    "keep_source_is_not_quality_winner",
    "gguf_is_not_a_recipe",
    "gguf_is_not_a_requirement",
    "gguf_is_pareto_reference",
    "gguf_is_not_the_runtime_format",
    "gguf_is_not_compiler_input",
    "gguf_payload_inspected",
    "q4km_must_be_beaten",
    "q4km_must_be_matched",
    "q4km_file_required_now",
    "q4km_nll_required",
    "q4km_nll_identity_matched_when_logits_available",
    "pareto_frontier_selected",
    "compiler_profile_selected",
    "calibration_corpus_selected",
    "eval_corpus_selected",
    "prompt_suite_selected",
    "capability_benchmark_selected",
    "acceptance_frontier_selected",
    "hypothesis_survival_selected",
    "ledger_open_question_corpora_closed",
    "nll_measured_here",
    "payloads_restreamed",
    "experiments_run",
    "methodology_defined",
    "toks_is_not_quality_axis",
    "gptq_is_not_a_scale_id",
    "gptq_required",
    "activation_aware_scale_id_added",
    "activation_quant_in_family_policies",
    "vision_eval_deferred",
    "safetensors_is_source_not_runtime",
    "diagram_ids",
    "n_diagrams",
    "canonical_sentence_logical",
    "canonical_sentence_reconstruction",
    "canonical_sentence_comparisons",
    "canonical_sentence_q4km",
    "methodology_question_sentence",
)

REQUIRED_HEADINGS: tuple[str, ...] = (
    "Authority",
    "Evaluation convention",
    "Reconstruction diagnostics and limits",
    "Teacher-forced comparisons",
    "Behavioral comparisons",
    "Black-box Pareto reference",
    "Corpora, splits, and unselected acceptance",
    "Hypothesis-survival protocol and methodology risks",
    "Deferred vision",
    "Machine-checkable summary JSON",
)

JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)
MERMAID_FENCE_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)
FORBIDDEN_RE = re.compile(r"TBD|TODO|\?\?\?")
UNKNOWN_RE = re.compile(r"UNKNOWN")

SUBSTRING_ID_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("eval layer", EVAL_LAYER_IDS),
    ("reconstruction diagnostic", RECONSTRUCTION_DIAGNOSTIC_IDS),
    ("reconstruction limit", RECONSTRUCTION_LIMIT_IDS),
    ("teacher-forced metric", TEACHER_FORCED_METRIC_IDS),
    ("behavioral class", BEHAVIORAL_CLASS_IDS),
    ("Pareto axis", PARETO_AXIS_IDS),
    ("corpus class", CORPUS_CLASS_IDS),
    ("methodology risk", METHODOLOGY_RISK_IDS),
    ("quality-high id", QUALITY_HIGH_IDS),
    ("policy family", POLICY_FAMILIES),
    ("recipe id", CANDIDATE_RECIPE_IDS),
)


class MissingConfig(Exception):
    """Raised when the config path is absent or unreadable."""

    def __init__(self, path: Path) -> None:
        super().__init__(str(path))
        self.path = path


class QuantizationValidationMismatch(Exception):
    """Raised when the quantization-validation markdown fails a content check."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def dumps_summary(summary: dict) -> str:
    """Pretty-print the quantization-validation summary object.

    Args:
        summary: Object produced by :func:`instantiate_quantization_validation`.

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
        number of bytes.

    Raises:
        AssertionError: If the payload is not an integer byte count.
    """

    payload = n_elements * code_bits / 8
    assert payload == int(payload), f"payload bytes not integral: {payload}"
    payload_bytes = int(payload)
    n_g = _n_groups(n_elements, group_size)
    meta_bytes = n_g * (scale_bytes + zero_point_bytes)
    return payload_bytes, n_g, meta_bytes, payload_bytes + meta_bytes


def instantiate_quantization_validation(config: dict) -> dict:
    """Compute the TASK-18 quantization-validation summary from ``config``.

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
    t_max = _require_int(text, "max_position_embeddings")
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
        f"layer_types full indices {full_from_types} != interval "
        f"{full_from_interval}"
    )
    n_full_layers = len(full_from_types)
    n_linear_layers = len(linear_from_types)

    linear_qkv_width = (
        2 * linear_num_key_heads + linear_num_value_heads
    ) * linear_key_head_dim
    linear_z_width = linear_num_value_heads * linear_value_head_dim
    q_proj_out = 2 * n_attn_heads * head_dim
    kv_proj_out = n_kv_heads * head_dim
    o_proj_in = n_attn_heads * head_dim

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

    scale_s = SCALE_STORAGE_ILLUSTRATION_BYTES
    (
        _mlp_g128_payload,
        _mlp_g128_n_g,
        _mlp_int4_g128_meta_bytes,
        mlp_int4_g128_total_bytes,
    ) = _integer_storage_bytes(mlp_n, 4, 128, scale_s, 0)
    mlp_int4_g128_over_bf16 = mlp_int4_g128_total_bytes / mlp_bf16_bytes
    (
        _une_payload,
        _une_n_g,
        _une_meta,
        unique_non_embed_int4_g128_total_bytes,
    ) = _integer_storage_bytes(unique_non_embed_n, 4, 128, scale_s, 0)

    s_elems_per_layer = (
        linear_num_value_heads * linear_key_head_dim * linear_value_head_dim
    )
    s_f32_bytes = n_linear_layers * s_elems_per_layer * BYTES_F32
    n_residual_adds_language = n_decoder_layers * 2
    n_residual_adds_complete = n_residual_adds_language + 2 * n_mtp_blocks

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
        "T_max": t_max,
        "n_language_mtp_tensors": N_LANGUAGE_MTP_TENSORS,
        "n_language_mtp_parameters": n_language_mtp_parameters,
        "mlp_n": mlp_n,
        "embed_n": embed_n,
        "mlp_bf16_bytes": mlp_bf16_bytes,
        "mlp_int4_g128_over_bf16": mlp_int4_g128_over_bf16,
        "weight_bytes_language_mtp_excl_vision": (
            weight_bytes_language_mtp_excl_vision
        ),
        "weight_bytes_unique_non_embed": weight_bytes_unique_non_embed,
        "unique_non_embed_int4_g128_total_bytes": (
            unique_non_embed_int4_g128_total_bytes
        ),
        "s_f32_bytes": s_f32_bytes,
        "n_residual_adds_language": n_residual_adds_language,
        "n_residual_adds_complete": n_residual_adds_complete,
        "example_T_values": list(EXAMPLE_T_VALUES),
        "example_T_is_not_prompt_matrix": True,
        "eval_layer_ids": list(EVAL_LAYER_IDS),
        "n_eval_layers": len(EVAL_LAYER_IDS),
        "reconstruction_diagnostic_ids": list(RECONSTRUCTION_DIAGNOSTIC_IDS),
        "reconstruction_diagnostic_roles": list(RECONSTRUCTION_DIAGNOSTIC_ROLES),
        "n_reconstruction_diagnostics": len(RECONSTRUCTION_DIAGNOSTIC_IDS),
        "reconstruction_limit_ids": list(RECONSTRUCTION_LIMIT_IDS),
        "n_reconstruction_limits": len(RECONSTRUCTION_LIMIT_IDS),
        "teacher_forced_metric_ids": list(TEACHER_FORCED_METRIC_IDS),
        "n_teacher_forced_metrics": len(TEACHER_FORCED_METRIC_IDS),
        "n_language_nll_events_offset": 1,
        "n_mtp_nll_events_offset": 2,
        "min_eval_tokens_language": 2,
        "min_eval_tokens_mtp": 3,
        "min_eval_tokens_complete": 3,
        "mtp_in_primary_nll": True,
        "sampling_out_of_nll": True,
        "log_softmax_is_eval_readout": True,
        "behavioral_class_ids": list(BEHAVIORAL_CLASS_IDS),
        "n_behavioral_classes": len(BEHAVIORAL_CLASS_IDS),
        "pareto_axis_ids": list(PARETO_AXIS_IDS),
        "n_pareto_axes": len(PARETO_AXIS_IDS),
        "q4km_named_path": Q4KM_NAMED_PATH,
        "corpus_class_ids": list(CORPUS_CLASS_IDS),
        "n_corpus_classes": len(CORPUS_CLASS_IDS),
        "calibration_eval_must_be_disjoint": True,
        "quality_high_ids": list(QUALITY_HIGH_IDS),
        "n_quality_high": len(QUALITY_HIGH_IDS),
        "survival_reconstruction_ids": list(SURVIVAL_RECONSTRUCTION_IDS),
        "survival_requires_teacher_forced": True,
        "quality_risk_ids": list(QUALITY_RISK_IDS),
        "n_quality_risks": len(QUALITY_RISK_IDS),
        "sensitive_high_ids": list(SENSITIVE_HIGH_IDS),
        "n_sensitive_high": len(SENSITIVE_HIGH_IDS),
        "methodology_risk_ids": list(METHODOLOGY_RISK_IDS),
        "methodology_risk_severities": list(METHODOLOGY_RISK_SEVERITIES),
        "methodology_high_ids": list(METHODOLOGY_HIGH_IDS),
        "methodology_medium_ids": list(METHODOLOGY_MEDIUM_IDS),
        "methodology_low_ids": list(METHODOLOGY_LOW_IDS),
        "n_methodology_risks": len(METHODOLOGY_RISK_IDS),
        "n_methodology_high": len(METHODOLOGY_HIGH_IDS),
        "n_methodology_medium": len(METHODOLOGY_MEDIUM_IDS),
        "n_methodology_low": len(METHODOLOGY_LOW_IDS),
        "policy_families": list(POLICY_FAMILIES),
        "candidate_recipe_ids": list(CANDIDATE_RECIPE_IDS),
        "n_policy_families": len(POLICY_FAMILIES),
        "n_candidate_recipes": len(CANDIDATE_RECIPE_IDS),
        "keep_source_packable_on_all_defined_families": True,
        "control_profile_id": "control",
        "control_profile_is_keep_source": True,
        "reconstruction_is_not_quality": True,
        "teacher_forced_is_primary_quality": True,
        "behavioral_is_secondary_quality": True,
        "keep_source_is_identity_control": True,
        "keep_source_is_not_quality_winner": True,
        "gguf_is_not_a_recipe": True,
        "gguf_is_not_a_requirement": True,
        "gguf_is_pareto_reference": True,
        "gguf_is_not_the_runtime_format": True,
        "gguf_is_not_compiler_input": True,
        "gguf_payload_inspected": False,
        "q4km_must_be_beaten": False,
        "q4km_must_be_matched": False,
        "q4km_file_required_now": False,
        "q4km_nll_required": False,
        "q4km_nll_identity_matched_when_logits_available": True,
        "pareto_frontier_selected": False,
        "compiler_profile_selected": False,
        "calibration_corpus_selected": False,
        "eval_corpus_selected": False,
        "prompt_suite_selected": False,
        "capability_benchmark_selected": False,
        "acceptance_frontier_selected": False,
        "hypothesis_survival_selected": False,
        "ledger_open_question_corpora_closed": False,
        "nll_measured_here": False,
        "payloads_restreamed": False,
        "experiments_run": False,
        "methodology_defined": True,
        "toks_is_not_quality_axis": True,
        "gptq_is_not_a_scale_id": True,
        "gptq_required": False,
        "activation_aware_scale_id_added": False,
        "activation_quant_in_family_policies": False,
        "vision_eval_deferred": True,
        "safetensors_is_source_not_runtime": True,
        "diagram_ids": list(DIAGRAM_REQUIRED_IDS),
        "n_diagrams": 1,
        "canonical_sentence_logical": CANONICAL_SENTENCE_LOGICAL,
        "canonical_sentence_reconstruction": CANONICAL_SENTENCE_RECONSTRUCTION,
        "canonical_sentence_comparisons": CANONICAL_SENTENCE_COMPARISONS,
        "canonical_sentence_q4km": CANONICAL_SENTENCE_Q4KM,
        "methodology_question_sentence": METHODOLOGY_QUESTION_SENTENCE,
    }
    _assert_identities(summary, text)
    return summary


def _assert_identities(summary: dict, text: dict) -> None:
    """Assert dossier identities on a live quantization-validation object.

    Args:
        summary: Object produced by :func:`instantiate_quantization_validation`.
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
    assert summary["T_max"] == 262144
    assert summary["T_max"] == text["max_position_embeddings"]
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
    assert summary["mlp_int4_g128_over_bf16"] == 0.2578125
    assert summary["unique_non_embed_int4_g128_total_bytes"] == 13431670032
    assert summary["weight_bytes_language_mtp_excl_vision"] == 54641395712
    assert summary["weight_bytes_unique_non_embed"] == 52098598912
    assert summary["s_f32_bytes"] == 150994944
    assert summary["n_residual_adds_language"] == 128
    assert summary["n_residual_adds_complete"] == 130
    assert summary["example_T_values"] == [1, 4096]
    assert summary["n_eval_layers"] == 3
    assert summary["n_reconstruction_diagnostics"] == 8
    assert summary["n_reconstruction_limits"] == 6
    assert summary["n_teacher_forced_metrics"] == 5
    assert summary["n_behavioral_classes"] == 3
    assert summary["n_pareto_axes"] == 3
    assert summary["n_corpus_classes"] == 4
    assert summary["n_methodology_risks"] == 6
    assert summary["n_methodology_high"] == 3
    assert summary["n_methodology_medium"] == 2
    assert summary["n_methodology_low"] == 1
    assert summary["n_quality_high"] == 8
    assert summary["n_quality_risks"] == 17
    assert summary["n_sensitive_high"] == 9
    assert summary["n_candidate_recipes"] == 22
    assert summary["n_policy_families"] == 16
    assert summary["n_diagrams"] == 1
    assert summary["candidate_recipe_ids"] == list(CANDIDATE_RECIPE_IDS)
    assert summary["policy_families"] == list(POLICY_FAMILIES)
    assert summary["quality_high_ids"] == list(QUALITY_HIGH_IDS)
    assert summary["survival_reconstruction_ids"] == list(
        SURVIVAL_RECONSTRUCTION_IDS
    )
    assert summary["min_eval_tokens_complete"] == 3
    assert summary["mtp_in_primary_nll"] is True
    assert summary["gguf_is_not_a_recipe"] is True
    assert summary["gguf_is_not_a_requirement"] is True
    assert summary["gguf_is_pareto_reference"] is True
    assert summary["gguf_payload_inspected"] is False
    assert summary["q4km_must_be_beaten"] is False
    assert summary["q4km_file_required_now"] is False
    assert summary["pareto_frontier_selected"] is False
    assert summary["compiler_profile_selected"] is False
    assert summary["calibration_corpus_selected"] is False
    assert summary["eval_corpus_selected"] is False
    assert summary["prompt_suite_selected"] is False
    assert summary["capability_benchmark_selected"] is False
    assert summary["acceptance_frontier_selected"] is False
    assert summary["hypothesis_survival_selected"] is False
    assert summary["ledger_open_question_corpora_closed"] is False
    assert summary["nll_measured_here"] is False
    assert summary["experiments_run"] is False
    assert summary["methodology_defined"] is True
    assert summary["reconstruction_is_not_quality"] is True
    assert summary["teacher_forced_is_primary_quality"] is True
    assert summary["keep_source_is_identity_control"] is True
    assert summary["keep_source_is_not_quality_winner"] is True
    assert summary["toks_is_not_quality_axis"] is True
    assert summary["gptq_is_not_a_scale_id"] is True
    assert summary["gptq_required"] is False
    assert summary["q4km_named_path"] == Q4KM_NAMED_PATH
    assert summary["methodology_risk_severities"] == list(
        METHODOLOGY_RISK_SEVERITIES
    )
    assert summary["eval_layer_ids"] == list(EVAL_LAYER_IDS)
    assert summary["reconstruction_diagnostic_ids"] == list(
        RECONSTRUCTION_DIAGNOSTIC_IDS
    )
    assert summary["reconstruction_diagnostic_roles"] == list(
        RECONSTRUCTION_DIAGNOSTIC_ROLES
    )
    assert summary["reconstruction_limit_ids"] == list(RECONSTRUCTION_LIMIT_IDS)
    assert summary["teacher_forced_metric_ids"] == list(TEACHER_FORCED_METRIC_IDS)
    assert summary["behavioral_class_ids"] == list(BEHAVIORAL_CLASS_IDS)
    assert summary["pareto_axis_ids"] == list(PARETO_AXIS_IDS)
    assert summary["corpus_class_ids"] == list(CORPUS_CLASS_IDS)
    assert summary["methodology_risk_ids"] == list(METHODOLOGY_RISK_IDS)
    assert summary["methodology_high_ids"] == list(METHODOLOGY_HIGH_IDS)
    assert summary["methodology_medium_ids"] == list(METHODOLOGY_MEDIUM_IDS)
    assert summary["methodology_low_ids"] == list(METHODOLOGY_LOW_IDS)
    assert summary["quality_risk_ids"] == list(QUALITY_RISK_IDS)
    assert summary["sensitive_high_ids"] == list(SENSITIVE_HIGH_IDS)
    assert summary["diagram_ids"] == list(DIAGRAM_REQUIRED_IDS)
    assert summary["n_language_nll_events_offset"] == 1
    assert summary["n_mtp_nll_events_offset"] == 2
    assert summary["min_eval_tokens_language"] == 2
    assert summary["min_eval_tokens_mtp"] == 3
    assert summary["sampling_out_of_nll"] is True
    assert summary["log_softmax_is_eval_readout"] is True
    assert summary["example_T_is_not_prompt_matrix"] is True
    assert summary["calibration_eval_must_be_disjoint"] is True
    assert summary["survival_requires_teacher_forced"] is True
    assert summary["keep_source_packable_on_all_defined_families"] is True
    assert summary["control_profile_id"] == "control"
    assert summary["control_profile_is_keep_source"] is True
    assert summary["behavioral_is_secondary_quality"] is True
    assert summary["gguf_is_not_the_runtime_format"] is True
    assert summary["gguf_is_not_compiler_input"] is True
    assert summary["q4km_must_be_matched"] is False
    assert summary["q4km_nll_required"] is False
    assert summary["q4km_nll_identity_matched_when_logits_available"] is True
    assert summary["payloads_restreamed"] is False
    assert summary["activation_aware_scale_id_added"] is False
    assert summary["activation_quant_in_family_policies"] is False
    assert summary["vision_eval_deferred"] is True
    assert summary["safetensors_is_source_not_runtime"] is True
    assert summary["bytes_bf16"] == 2
    assert summary["bytes_f32"] == 4
    assert summary["canonical_sentence_logical"] == CANONICAL_SENTENCE_LOGICAL
    assert summary["canonical_sentence_reconstruction"] == (
        CANONICAL_SENTENCE_RECONSTRUCTION
    )
    assert summary["canonical_sentence_comparisons"] == (
        CANONICAL_SENTENCE_COMPARISONS
    )
    assert summary["canonical_sentence_q4km"] == CANONICAL_SENTENCE_Q4KM
    assert summary["methodology_question_sentence"] == (
        METHODOLOGY_QUESTION_SENTENCE
    )
    high_from_ops = [
        risk_id
        for risk_id, severity in zip(
            summary["methodology_risk_ids"],
            summary["methodology_risk_severities"],
        )
        if severity == "high"
    ]
    medium_from_ops = [
        risk_id
        for risk_id, severity in zip(
            summary["methodology_risk_ids"],
            summary["methodology_risk_severities"],
        )
        if severity == "medium"
    ]
    low_from_ops = [
        risk_id
        for risk_id, severity in zip(
            summary["methodology_risk_ids"],
            summary["methodology_risk_severities"],
        )
        if severity == "low"
    ]
    assert high_from_ops == summary["methodology_high_ids"]
    assert medium_from_ops == summary["methodology_medium_ids"]
    assert low_from_ops == summary["methodology_low_ids"]
    assert len(summary["survival_reconstruction_ids"]) == len(
        summary["quality_high_ids"]
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


def diff_summary(live: dict, documented: dict) -> list[str]:
    """Compare documented JSON against a live quantization-validation object.

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
    """Return whether a forbidden-winner match is required allowed wording.

    Allowed wrappers are ``not a selected winner``, ``not selected winners``,
    ``not a quality winner``, ``not a requirement``, and the canonical Q4_K_M
    sentence.

    Args:
        text: Full markdown document.
        start: Match start index.
        phrase: Forbidden phrase that matched.

    Returns:
        True when this occurrence is allowed.
    """

    end = start + len(phrase)
    if phrase == "selected winner":
        example = text[max(0, start - 6) : end]
        canonical = text[max(0, start - 4) : end + 1]
        return example == "not a selected winner" or canonical == (
            "not selected winners"
        )
    if phrase == "quality winner":
        allowed = text[max(0, start - 6) : end]
        return allowed == "not a quality winner"
    window_start = max(0, start - len(CANONICAL_SENTENCE_Q4KM))
    window = text[window_start : end + len(CANONICAL_SENTENCE_Q4KM)]
    if CANONICAL_SENTENCE_Q4KM in window:
        q_start = text.find(CANONICAL_SENTENCE_Q4KM, window_start)
        while q_start >= 0 and q_start < end:
            q_end = q_start + len(CANONICAL_SENTENCE_Q4KM)
            if q_start <= start < q_end:
                return True
            q_start = text.find(CANONICAL_SENTENCE_Q4KM, q_start + 1)
            if q_start >= end:
                break
    if phrase == "requirement" or "requirement" in phrase:
        wrapper = text[max(0, start - 6) : end]
        if wrapper.endswith("not a requirement") or (
            text[max(0, start - 6) : start + len(phrase)]
            .endswith("not a requirement")
        ):
            return True
    return False


def check_quantization_validation(live: dict, design_path: Path) -> None:
    """Check a quantization-validation markdown file against a live summary.

    Args:
        live: Quantization-validation summary object from sitting config.
        design_path: Path to ``quantization-validation.md``.

    Raises:
        QuantizationValidationMismatch: On heading, JSON, diagram, or token
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
    if CANONICAL_SENTENCE_RECONSTRUCTION not in text:
        differences.append(
            "canonical_sentence_reconstruction not present verbatim"
        )
    if CANONICAL_SENTENCE_COMPARISONS not in text:
        differences.append("canonical_sentence_comparisons not present verbatim")
    if CANONICAL_SENTENCE_Q4KM not in text:
        differences.append("canonical_sentence_q4km not present verbatim")
    if METHODOLOGY_QUESTION_SENTENCE not in text:
        differences.append("methodology_question_sentence not present verbatim")

    for label, ids in SUBSTRING_ID_GROUPS:
        for item_id in ids:
            if item_id not in text:
                differences.append(f"{label} {item_id!r} missing as substring")

    if "HYPOTHESIS" not in text:
        differences.append("word HYPOTHESIS missing")
    if "not a selected winner" not in text:
        differences.append("phrase 'not a selected winner' missing")

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
        raise QuantizationValidationMismatch(differences)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the quantization-validation checker.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Instantiate Qwen3.8-27B language+MTP quantization-validation "
            "summary from text_config and check "
            "docs/architecture/quantization-validation.md."
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
        help="Write quantization-validation summary JSON to stdout.",
    )
    parser.add_argument(
        "--quantization-validation",
        type=Path,
        default=None,
        help="Markdown path whose first json fence must match live summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the quantization-validation checker CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch, 2 missing
        config.
    """

    args = _parse_args(argv)
    try:
        config = load_config(args.config)
        summary = instantiate_quantization_validation(config)
        if args.quantization_validation is not None:
            if not args.quantization_validation.is_file():
                print(
                    f"quantization-validation file not found: "
                    f"{args.quantization_validation}",
                    file=sys.stderr,
                )
                return 1
            check_quantization_validation(summary, args.quantization_validation)
    except MissingConfig as exc:
        print(str(exc.path), file=sys.stderr)
        return 2
    except QuantizationValidationMismatch as exc:
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

    if args.json or args.quantization_validation is None:
        sys.stdout.write(dumps_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
