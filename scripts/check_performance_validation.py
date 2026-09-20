#!/usr/bin/env python3
"""Check Qwen3.8-27B performance validation against sitting text_config.

Reads ``text_config`` from a Transformers ``config.json`` (no safetensor
payloads, no GPU query) and either prints the performance-validation
summary object or checks that
``docs/architecture/performance-validation.md`` matches it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

AUTHORITY = "docs/architecture/plan.md"
DELIVERABLE = "docs/architecture/performance-validation.md"

CANONICAL_SENTENCE_LOGICAL = (
    "Logical values in this document are graph nodes. They do not imply "
    "physical allocation, materialization, buffer reuse, or kernel fusion."
)
CANONICAL_SENTENCE_METHODOLOGY = (
    "Decode, prefill, kernel, memory, and end-to-end metrics in this "
    "document are a methodology, not measurements."
)
CANONICAL_SENTENCE_E2E = (
    "Microbenchmarks cannot pass a CUDA mapping; end-to-end measurement "
    "on a matching identity is required alongside them."
)
CANONICAL_SENTENCE_OPEN_QUESTION = (
    "Exact hardware, prompt matrix, and reproducibility protocol for the "
    "implementation phase remain unselected."
)
CANONICAL_SENTENCE_WINNER = (
    "This document fills no sitting SKU numbers and selects no mapping "
    "winner."
)
METHODOLOGY_QUESTION_SENTENCE = (
    "Decode, prefill, and kernel metrics are specified; end-to-end "
    "measurement is required alongside microbenchmarks; no implementation "
    "benchmarks are run in this study task; exact hardware, prompt matrix, "
    "and reproducibility protocol remain unselected."
)

BYTES_BF16 = 2
N_W = 32
N_BANK = 32
MAC_C_COMPLETE = 27433238528
MAC_A_COMPLETE = 208896
MAC_DECODE_COMPLETE_T1 = 27433447424
MAC_PREFILL_COMPLETE_T1 = 27433447424
MAC_DECODE_COMPLETE_T4096 = 28288876544
MAC_PREFILL_COMPLETE_T4096 = 114119319486464
N_LANGUAGE_MTP_PARAMETERS = 27320697856
WEIGHT_BYTES_LANGUAGE_MTP_EXCL_VISION = 54641395712
WEIGHT_BYTES_UNIQUE_NON_EMBED = 52098598912

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

MAPPING_IDS: tuple[str, ...] = (
    "map_embed_thread_element",
    "map_embed_warp_row",
    "map_embed_cta_vector",
    "map_attn_cta_head",
    "map_attn_warp_t",
    "map_attn_cta_splitk",
    "map_gdn_cta_head",
    "map_gdn_warp_recurrent",
    "map_gdn_cta_chunk",
    "map_mlp_cta_dout",
    "map_mlp_cta_splitk",
    "map_mlp_grid_T",
    "map_lm_cta_vocab",
    "map_lm_cta_splitk",
    "map_lm_warp_gemv",
    "map_mtp_cta_fc",
    "map_mtp_cta_fused",
    "map_mtp_split_norm_gemm",
)

EVALUATION_CRITERION_IDS: tuple[str, ...] = (
    "crit_occupancy",
    "crit_latency_hiding",
    "crit_wave_quant",
    "crit_intensity_roofline",
    "crit_sync_class",
    "crit_pipeline_mix",
    "crit_fusion_delta",
)

METRIC_FAMILY_IDS: tuple[str, ...] = (
    "decode",
    "prefill",
    "kernel",
    "memory",
    "end_to_end",
)

METRIC_CLASS_IDS: tuple[str, ...] = (
    "prefill",
    "decode_only",
    "complete_request",
    "component",
    "memory",
)

DECODE_METRIC_IDS: tuple[str, ...] = (
    "dec_toks",
    "dec_step_ms",
    "dec_p50_step_ms",
    "dec_p99_step_ms",
)

DECODE_METRIC_CLASSES: tuple[str, ...] = (
    "decode_only",
    "decode_only",
    "decode_only",
    "decode_only",
)

PREFILL_METRIC_IDS: tuple[str, ...] = (
    "pre_toks",
    "pre_ms",
    "pre_ttft_ms",
    "pre_mac_cite",
)

KERNEL_METRIC_IDS: tuple[str, ...] = (
    "k_node_ms",
    "k_mapping_ms",
    "k_leaf_union_ms",
    "k_graph_envelope_ms",
    "k_occupancy_achieved",
)

MEMORY_METRIC_IDS: tuple[str, ...] = (
    "mem_hbm_gbps",
    "mem_weight_bytes",
    "mem_state_bytes",
    "mem_act_bytes",
    "mem_working_set",
)

E2E_METRIC_IDS: tuple[str, ...] = (
    "e2e_latency_ms",
    "e2e_output_toks",
    "e2e_ttft_ms",
)

IDENTITY_FIELD_IDS: tuple[str, ...] = (
    "engine_commit",
    "loaded_binary_hash",
    "build_flags",
    "artifact_selectors",
    "input_token_hash",
    "prefix",
    "output_eval_counts",
    "first_token_convention",
    "allocated_capacity",
    "populated_length",
    "sampling_output_policy",
    "graph_mode",
    "clocks_residents",
    "warmups_samples",
    "numerator_start_end_events",
)

COVERAGE_FIELD_IDS: tuple[str, ...] = (
    "captured_tables",
    "units",
    "clock_domain",
    "capture_bounds",
    "streams",
    "expected_graph_node_counts",
    "observed_graph_node_counts",
    "family_mapping",
    "unclassified_work",
)

TIME_ACCOUNTING_RULE_IDS: tuple[str, ...] = (
    "ta_disjoint_union",
    "ta_no_parent_child",
    "ta_no_cpu_gpu_double",
    "ta_sync_is_wait",
    "ta_setup_outside_rate",
)

SKU_UNKNOWN_SYMBOLS: tuple[str, ...] = (
    "N_SM",
    "W_max",
    "S_reg",
    "C_smem",
    "T_max",
    "B_max",
    "N_bar",
    "N_sched",
    "G_reg",
    "G_smem",
    "Beta_HBM",
    "Pi_FMA",
    "Pi_TC",
    "L_issue",
    "async_copy_cap",
    "mma_shapes",
    "cluster_cap",
)

TRAFFIC_CHANNEL_IDS: tuple[str, ...] = (
    "weight",
    "state",
    "activation",
)

METHODOLOGY_RISK_IDS: tuple[str, ...] = (
    "v_mixed_identity",
    "v_envelope_as_leaf",
    "v_micro_as_winner",
    "v_missing_coverage_zero",
    "v_bound_as_measured",
    "v_datasheet_sku",
    "v_t1_prefill_as_decode",
    "v_toks_as_quality",
)

METHODOLOGY_RISK_SEVERITIES: tuple[str, ...] = (
    "high",
    "high",
    "high",
    "high",
    "medium",
    "medium",
    "medium",
    "low",
)

DIAGRAM_JSON_IDS: tuple[str, ...] = (
    "decode",
    "prefill",
    "kernel",
    "memory",
    "identity",
    "e2e",
    "sku",
    "openq",
)

DIAGRAM_MERMAID_IDS: tuple[str, ...] = (
    "DECODE",
    "PREFILL",
    "KERNEL",
    "MEMORY",
    "IDENTITY",
    "E2E",
    "SKU",
    "OPENQ",
)

REQUIRED_HEADINGS: tuple[str, ...] = (
    "Authority",
    "Measurement convention",
    "Decode metrics",
    "Prefill metrics",
    "Kernel metrics",
    "Memory metrics",
    "End-to-end metrics",
    "Identity, coverage, and time accounting",
    "SKU-fill protocol and unselected hardware",
    "Methodology risks",
    "Deferred vision",
    "Machine-checkable summary JSON",
)

LOCKED_DOCUMENT_INTEGERS: tuple[int, ...] = (
    5120,
    17408,
    248320,
    64,
    48,
    16,
    262144,
    27320697856,
    54641395712,
    52098598912,
    27433238528,
    208896,
    27433447424,
    28288876544,
    114119319486464,
    135,
    32,
)

FORBIDDEN_WINNER_PHRASES: tuple[str, ...] = (
    "selected mapping winner",
    "winning kernel",
    "winning mapping",
    "SKU table is filled",
    "selected hardware",
    "selected prompt matrix",
    "selected reproducibility protocol",
    "should use this GPU",
    "datasheet SKU is measured",
    "microbenchmarks pass the mapping",
    "reconstruction is sufficient",
    "tok/s is the quality axis",
)

ALLOWED_EXCEPTION_PHRASES: tuple[str, ...] = (
    "selects no mapping winner",
    "not measurements",
    "remain unselected",
    "not a selected winner",
    "unselected hardware",
    "unselected prompt matrix",
    "unselected reproducibility protocol",
)

SUBSTRING_ID_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("metric family", METRIC_FAMILY_IDS),
    ("metric class", METRIC_CLASS_IDS),
    ("decode metric", DECODE_METRIC_IDS),
    ("prefill metric", PREFILL_METRIC_IDS),
    ("kernel metric", KERNEL_METRIC_IDS),
    ("memory metric", MEMORY_METRIC_IDS),
    ("e2e metric", E2E_METRIC_IDS),
    ("identity field", IDENTITY_FIELD_IDS),
    ("coverage field", COVERAGE_FIELD_IDS),
    ("time-accounting rule", TIME_ACCOUNTING_RULE_IDS),
    ("methodology risk", METHODOLOGY_RISK_IDS),
    ("node type", NODE_TYPE_IDS),
    ("stage kind", STAGE_KIND_IDS),
    ("mapping", MAPPING_IDS),
    ("evaluation criterion", EVALUATION_CRITERION_IDS),
    ("sku unknown symbol", SKU_UNKNOWN_SYMBOLS),
    ("traffic channel", TRAFFIC_CHANNEL_IDS),
)

JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)
MERMAID_FENCE_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
HEADING_RE = re.compile(r"^## (?!#)(.+)$", re.MULTILINE)
FORBIDDEN_RE = re.compile(r"TBD|TODO|\?\?\?")
UNKNOWN_RE = re.compile(r"UNKNOWN")

SCHEMA_KEYS: tuple[str, ...] = (
    "authority",
    "deliverable",
    "hidden_size",
    "intermediate_size",
    "vocab_size",
    "n_decoder_layers",
    "n_linear_layers",
    "n_full_layers",
    "n_mtp_blocks",
    "full_attention_indices",
    "model_T_max",
    "dtype",
    "mamba_ssm_dtype",
    "n_language_mtp_parameters",
    "weight_bytes_language_mtp_excl_vision",
    "weight_bytes_unique_non_embed",
    "mac_C_complete",
    "mac_A_complete",
    "mac_decode_complete_T1",
    "mac_prefill_complete_T1",
    "mac_decode_complete_T4096",
    "mac_prefill_complete_T4096",
    "example_T_values",
    "example_T_is_not_prompt_matrix",
    "N_w",
    "N_bank",
    "n_node_types",
    "node_type_ids",
    "n_embed_instances",
    "n_gated_attn_instances",
    "n_gated_delta_net_instances",
    "n_mlp_instances",
    "n_lm_head_instances",
    "n_mtp_mix_instances",
    "n_node_instances_complete",
    "n_stage_kinds",
    "stage_kind_ids",
    "n_mappings",
    "mapping_ids",
    "n_mappings_selected",
    "n_evaluation_criteria",
    "evaluation_criterion_ids",
    "evaluation_measured",
    "n_metric_families",
    "metric_family_ids",
    "n_metric_classes",
    "metric_class_ids",
    "n_decode_metrics",
    "decode_metric_ids",
    "decode_metric_classes",
    "n_prefill_metrics",
    "prefill_metric_ids",
    "n_kernel_metrics",
    "kernel_metric_ids",
    "n_memory_metrics",
    "memory_metric_ids",
    "n_e2e_metrics",
    "e2e_metric_ids",
    "e2e_primary_id",
    "n_identity_fields",
    "identity_field_ids",
    "n_coverage_fields",
    "coverage_field_ids",
    "n_time_accounting_rules",
    "time_accounting_rule_ids",
    "n_sku_unknown_symbols",
    "sku_unknown_symbols",
    "n_traffic_channels",
    "traffic_channel_ids",
    "n_methodology_risks",
    "methodology_risk_ids",
    "methodology_risk_severities",
    "methodology_risk_usefulness_label",
    "n_methodology_high",
    "n_methodology_medium",
    "n_methodology_low",
    "n_diagrams",
    "diagram_ids",
    "n_headings",
    "headings",
    "decode_work_identity",
    "prefill_work_identity",
    "decode_T_new",
    "decode_incoming_state",
    "prefill_incoming_state",
    "first_token_in_prefill",
    "decode_only_excludes_first_token",
    "first_token_convention_id",
    "ratio_requires_matching_identity",
    "decode_is_not_prefill",
    "decode_is_not_complete_request",
    "t1_mac_equal_does_not_imply_equal_traffic",
    "graph_envelope_is_not_leaf",
    "missing_coverage_is_not_zero",
    "kernel_cannot_pass_mapping",
    "microbenchmark_cannot_pass_mapping",
    "e2e_required_alongside_microbenchmarks",
    "e2e_mixed_toks_is_not_decode_only",
    "complete_request_vs_decode_only_forbidden",
    "lower_bound_is_not_measured_bandwidth",
    "peak_memory_is_not_live_set",
    "public_sample_eval_output_is_quality",
    "toks_is_not_quality_axis",
    "identity_protocol_defined",
    "identity_values_selected",
    "coverage_protocol_defined",
    "coverage_values_selected",
    "time_accounting_protocol_defined",
    "sampling_policy_selected",
    "warmup_count_selected",
    "clock_policy_selected",
    "sku_table_filled",
    "sku_fill_protocol_defined",
    "sku_from_datasheet_is_not_measurement",
    "hardware_selected",
    "prompt_matrix_selected",
    "reproducibility_protocol_selected",
    "ledger_open_question_hardware_closed",
    "mapping_winner_selected",
    "winner_selected_without_measurements",
    "ledger_open_question_mapping_winner_closed",
    "n_mappings_per_node",
    "decode_prefill_share_graph",
    "decode_prefill_share_artifact",
    "decode_prefill_share_metric_identity",
    "activations_in_layout_scope",
    "benchmarks_run",
    "toks_measured_here",
    "experiments_run",
    "methodology_defined",
    "nll_measured_here",
    "payloads_restreamed",
    "gguf_payload_inspected",
    "gguf_is_not_design_authority",
    "quartz_inspected",
    "llama_inspected",
    "llama_is_not_design_authority",
    "device_query_run",
    "nsight_run",
    "achieved_occupancy_measured_here",
    "vision_eval_deferred",
    "safetensors_is_source_not_runtime",
    "canonical_sentence_logical",
    "canonical_sentence_methodology",
    "canonical_sentence_e2e",
    "canonical_sentence_open_question",
    "canonical_sentence_winner",
    "methodology_question_sentence",
)


class MissingConfig(Exception):
    """Raised when the config path is absent or unreadable."""

    def __init__(self, path: Path) -> None:
        super().__init__(str(path))
        self.path = path


class PerformanceValidationMismatch(Exception):
    """Raised when the performance-validation markdown fails a content check."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def dumps_summary(summary: dict) -> str:
    """Pretty-print the performance-validation summary object.

    Args:
        summary: Object produced by :func:`instantiate_performance_validation`.

    Returns:
        Pretty-printed JSON including a trailing newline.
    """

    return json.dumps(summary, indent=2) + "\n"


def first_json_fence(text: str) -> tuple[dict, re.Match[str]]:
    """Parse the first fenced ``json`` code block in ``text``.

    Args:
        text: Markdown document containing a `` ```json `` fence.

    Returns:
        Decoded JSON object and the regex match for the fence.

    Raises:
        AssertionError: If no fence exists or the payload is not an object.
    """

    match = JSON_FENCE_RE.search(text)
    assert match is not None, "no fenced json block found"
    payload = json.loads(match.group(1))
    assert isinstance(payload, dict), "fenced json block is not an object"
    return payload, match


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


def instantiate_performance_validation(config: dict) -> dict:
    """Compute the TASK-19 performance-validation summary from ``config``.

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
    model_t_max = _require_int(text, "max_position_embeddings")
    param_dtype = _require_str(text, "dtype")
    mamba_ssm_dtype = _require_str(text, "mamba_ssm_dtype")

    assert param_dtype == "bfloat16", f"dtype is not bfloat16: {param_dtype!r}"
    assert mamba_ssm_dtype == "float32", (
        f"mamba_ssm_dtype is not float32: {mamba_ssm_dtype!r}"
    )
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
    assert n_full_layers == math.ceil(n_decoder_layers / full_attention_interval)

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
    weight_bytes_language_mlp = BYTES_BF16 * mlp_n
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

    n_embed_instances = 2
    n_gated_attn_instances = n_full_layers + n_mtp_blocks
    n_gated_delta_net_instances = n_linear_layers
    n_mlp_instances = n_decoder_layers + n_mtp_blocks
    n_lm_head_instances = 2
    n_mtp_mix_instances = 1
    n_node_instances_complete = (
        n_embed_instances
        + n_gated_attn_instances
        + n_gated_delta_net_instances
        + n_mlp_instances
        + n_lm_head_instances
        + n_mtp_mix_instances
    )

    n_methodology_high = sum(
        1 for severity in METHODOLOGY_RISK_SEVERITIES if severity == "high"
    )
    n_methodology_medium = sum(
        1 for severity in METHODOLOGY_RISK_SEVERITIES if severity == "medium"
    )
    n_methodology_low = sum(
        1 for severity in METHODOLOGY_RISK_SEVERITIES if severity == "low"
    )

    summary = {
        "authority": AUTHORITY,
        "deliverable": DELIVERABLE,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "n_decoder_layers": n_decoder_layers,
        "n_linear_layers": n_linear_layers,
        "n_full_layers": n_full_layers,
        "n_mtp_blocks": n_mtp_blocks,
        "full_attention_indices": full_from_types,
        "model_T_max": model_t_max,
        "dtype": param_dtype,
        "mamba_ssm_dtype": mamba_ssm_dtype,
        "n_language_mtp_parameters": n_language_mtp_parameters,
        "weight_bytes_language_mtp_excl_vision": (
            weight_bytes_language_mtp_excl_vision
        ),
        "weight_bytes_unique_non_embed": weight_bytes_unique_non_embed,
        "mac_C_complete": MAC_C_COMPLETE,
        "mac_A_complete": MAC_A_COMPLETE,
        "mac_decode_complete_T1": MAC_DECODE_COMPLETE_T1,
        "mac_prefill_complete_T1": MAC_PREFILL_COMPLETE_T1,
        "mac_decode_complete_T4096": MAC_DECODE_COMPLETE_T4096,
        "mac_prefill_complete_T4096": MAC_PREFILL_COMPLETE_T4096,
        "example_T_values": list(EXAMPLE_T_VALUES),
        "example_T_is_not_prompt_matrix": True,
        "N_w": N_W,
        "N_bank": N_BANK,
        "n_node_types": len(NODE_TYPE_IDS),
        "node_type_ids": list(NODE_TYPE_IDS),
        "n_embed_instances": n_embed_instances,
        "n_gated_attn_instances": n_gated_attn_instances,
        "n_gated_delta_net_instances": n_gated_delta_net_instances,
        "n_mlp_instances": n_mlp_instances,
        "n_lm_head_instances": n_lm_head_instances,
        "n_mtp_mix_instances": n_mtp_mix_instances,
        "n_node_instances_complete": n_node_instances_complete,
        "n_stage_kinds": len(STAGE_KIND_IDS),
        "stage_kind_ids": list(STAGE_KIND_IDS),
        "n_mappings": len(MAPPING_IDS),
        "mapping_ids": list(MAPPING_IDS),
        "n_mappings_selected": 0,
        "n_evaluation_criteria": len(EVALUATION_CRITERION_IDS),
        "evaluation_criterion_ids": list(EVALUATION_CRITERION_IDS),
        "evaluation_measured": False,
        "n_metric_families": len(METRIC_FAMILY_IDS),
        "metric_family_ids": list(METRIC_FAMILY_IDS),
        "n_metric_classes": len(METRIC_CLASS_IDS),
        "metric_class_ids": list(METRIC_CLASS_IDS),
        "n_decode_metrics": len(DECODE_METRIC_IDS),
        "decode_metric_ids": list(DECODE_METRIC_IDS),
        "decode_metric_classes": list(DECODE_METRIC_CLASSES),
        "n_prefill_metrics": len(PREFILL_METRIC_IDS),
        "prefill_metric_ids": list(PREFILL_METRIC_IDS),
        "n_kernel_metrics": len(KERNEL_METRIC_IDS),
        "kernel_metric_ids": list(KERNEL_METRIC_IDS),
        "n_memory_metrics": len(MEMORY_METRIC_IDS),
        "memory_metric_ids": list(MEMORY_METRIC_IDS),
        "n_e2e_metrics": len(E2E_METRIC_IDS),
        "e2e_metric_ids": list(E2E_METRIC_IDS),
        "e2e_primary_id": "e2e_latency_ms",
        "n_identity_fields": len(IDENTITY_FIELD_IDS),
        "identity_field_ids": list(IDENTITY_FIELD_IDS),
        "n_coverage_fields": len(COVERAGE_FIELD_IDS),
        "coverage_field_ids": list(COVERAGE_FIELD_IDS),
        "n_time_accounting_rules": len(TIME_ACCOUNTING_RULE_IDS),
        "time_accounting_rule_ids": list(TIME_ACCOUNTING_RULE_IDS),
        "n_sku_unknown_symbols": len(SKU_UNKNOWN_SYMBOLS),
        "sku_unknown_symbols": list(SKU_UNKNOWN_SYMBOLS),
        "n_traffic_channels": len(TRAFFIC_CHANNEL_IDS),
        "traffic_channel_ids": list(TRAFFIC_CHANNEL_IDS),
        "n_methodology_risks": len(METHODOLOGY_RISK_IDS),
        "methodology_risk_ids": list(METHODOLOGY_RISK_IDS),
        "methodology_risk_severities": list(METHODOLOGY_RISK_SEVERITIES),
        "methodology_risk_usefulness_label": "HYPOTHESIS",
        "n_methodology_high": n_methodology_high,
        "n_methodology_medium": n_methodology_medium,
        "n_methodology_low": n_methodology_low,
        "n_diagrams": 1,
        "diagram_ids": list(DIAGRAM_JSON_IDS),
        "n_headings": len(REQUIRED_HEADINGS),
        "headings": list(REQUIRED_HEADINGS),
        "decode_work_identity": "C+AT",
        "prefill_work_identity": "TC+AT(T+1)/2",
        "decode_T_new": 1,
        "decode_incoming_state": "populated",
        "prefill_incoming_state": "zeros",
        "first_token_in_prefill": True,
        "decode_only_excludes_first_token": True,
        "first_token_convention_id": "ttft_in_prefill",
        "ratio_requires_matching_identity": True,
        "decode_is_not_prefill": True,
        "decode_is_not_complete_request": True,
        "t1_mac_equal_does_not_imply_equal_traffic": True,
        "graph_envelope_is_not_leaf": True,
        "missing_coverage_is_not_zero": True,
        "kernel_cannot_pass_mapping": True,
        "microbenchmark_cannot_pass_mapping": True,
        "e2e_required_alongside_microbenchmarks": True,
        "e2e_mixed_toks_is_not_decode_only": True,
        "complete_request_vs_decode_only_forbidden": True,
        "lower_bound_is_not_measured_bandwidth": True,
        "peak_memory_is_not_live_set": True,
        "public_sample_eval_output_is_quality": True,
        "toks_is_not_quality_axis": True,
        "identity_protocol_defined": True,
        "identity_values_selected": False,
        "coverage_protocol_defined": True,
        "coverage_values_selected": False,
        "time_accounting_protocol_defined": True,
        "sampling_policy_selected": False,
        "warmup_count_selected": False,
        "clock_policy_selected": False,
        "sku_table_filled": False,
        "sku_fill_protocol_defined": True,
        "sku_from_datasheet_is_not_measurement": True,
        "hardware_selected": False,
        "prompt_matrix_selected": False,
        "reproducibility_protocol_selected": False,
        "ledger_open_question_hardware_closed": False,
        "mapping_winner_selected": False,
        "winner_selected_without_measurements": False,
        "ledger_open_question_mapping_winner_closed": False,
        "n_mappings_per_node": 3,
        "decode_prefill_share_graph": True,
        "decode_prefill_share_artifact": True,
        "decode_prefill_share_metric_identity": False,
        "activations_in_layout_scope": False,
        "benchmarks_run": False,
        "toks_measured_here": False,
        "experiments_run": False,
        "methodology_defined": True,
        "nll_measured_here": False,
        "payloads_restreamed": False,
        "gguf_payload_inspected": False,
        "gguf_is_not_design_authority": True,
        "quartz_inspected": False,
        "llama_inspected": False,
        "llama_is_not_design_authority": True,
        "device_query_run": False,
        "nsight_run": False,
        "achieved_occupancy_measured_here": False,
        "vision_eval_deferred": True,
        "safetensors_is_source_not_runtime": True,
        "canonical_sentence_logical": CANONICAL_SENTENCE_LOGICAL,
        "canonical_sentence_methodology": CANONICAL_SENTENCE_METHODOLOGY,
        "canonical_sentence_e2e": CANONICAL_SENTENCE_E2E,
        "canonical_sentence_open_question": CANONICAL_SENTENCE_OPEN_QUESTION,
        "canonical_sentence_winner": CANONICAL_SENTENCE_WINNER,
        "methodology_question_sentence": METHODOLOGY_QUESTION_SENTENCE,
    }
    _assert_identities(summary, text)
    return summary


def _assert_identities(summary: dict, text: dict) -> None:
    """Assert dossier identities on a live performance-validation object.

    Args:
        summary: Object produced by :func:`instantiate_performance_validation`.
        text: The ``text_config`` mapping used to build ``summary``.
    """

    assert list(summary) == list(SCHEMA_KEYS), (
        f"schema key order mismatch: {list(summary)} vs {list(SCHEMA_KEYS)}"
    )
    assert summary["authority"] == AUTHORITY
    assert summary["deliverable"] == DELIVERABLE
    assert summary["n_linear_layers"] == 48
    assert summary["n_full_layers"] == 16
    assert summary["n_mtp_blocks"] == 1
    assert summary["full_attention_indices"] == list(LOCKED_FULL_ATTENTION_INDICES)
    assert text.get("dtype") == "bfloat16"
    assert text.get("mamba_ssm_dtype") == "float32"
    assert summary["dtype"] == "bfloat16"
    assert summary["mamba_ssm_dtype"] == "float32"
    assert summary["model_T_max"] == 262144
    assert summary["model_T_max"] == text["max_position_embeddings"]
    assert summary["hidden_size"] == 5120
    assert summary["intermediate_size"] == 17408
    assert summary["vocab_size"] == 248320
    assert summary["n_decoder_layers"] == 64
    assert summary["n_language_mtp_parameters"] == N_LANGUAGE_MTP_PARAMETERS
    assert (
        summary["weight_bytes_language_mtp_excl_vision"]
        == WEIGHT_BYTES_LANGUAGE_MTP_EXCL_VISION
    )
    assert summary["weight_bytes_unique_non_embed"] == WEIGHT_BYTES_UNIQUE_NON_EMBED
    assert summary["mac_C_complete"] == MAC_C_COMPLETE
    assert summary["mac_A_complete"] == MAC_A_COMPLETE
    assert summary["mac_decode_complete_T1"] == MAC_DECODE_COMPLETE_T1
    assert summary["mac_prefill_complete_T1"] == MAC_PREFILL_COMPLETE_T1
    assert summary["mac_decode_complete_T1"] == summary["mac_prefill_complete_T1"]
    assert summary["mac_decode_complete_T1"] == MAC_C_COMPLETE + MAC_A_COMPLETE
    assert summary["mac_decode_complete_T4096"] == MAC_DECODE_COMPLETE_T4096
    assert summary["mac_decode_complete_T4096"] == (
        MAC_C_COMPLETE + MAC_A_COMPLETE * 4096
    )
    assert summary["mac_prefill_complete_T4096"] == MAC_PREFILL_COMPLETE_T4096
    assert summary["example_T_values"] == [1, 4096]
    assert summary["N_w"] == 32
    assert summary["N_bank"] == 32
    assert summary["n_node_types"] == 6
    assert summary["n_node_instances_complete"] == 135
    assert summary["n_stage_kinds"] == 9
    assert summary["n_mappings"] == 18
    assert summary["n_mappings_per_node"] == 3
    assert summary["n_mappings_selected"] == 0
    assert summary["n_evaluation_criteria"] == 7
    assert summary["evaluation_measured"] is False
    assert summary["n_metric_families"] == 5
    assert summary["n_metric_classes"] == 5
    assert summary["n_decode_metrics"] == 4
    assert summary["n_prefill_metrics"] == 4
    assert summary["n_kernel_metrics"] == 5
    assert summary["n_memory_metrics"] == 5
    assert summary["n_e2e_metrics"] == 3
    assert summary["n_identity_fields"] == 15
    assert summary["n_coverage_fields"] == 9
    assert summary["n_time_accounting_rules"] == 5
    assert summary["n_sku_unknown_symbols"] == 17
    assert summary["n_traffic_channels"] == 3
    assert summary["n_methodology_risks"] == 8
    assert summary["n_methodology_high"] == 4
    assert summary["n_methodology_medium"] == 3
    assert summary["n_methodology_low"] == 1
    assert summary["n_diagrams"] == 1
    assert summary["n_headings"] == 12
    assert summary["node_type_ids"] == list(NODE_TYPE_IDS)
    assert summary["mapping_ids"] == list(MAPPING_IDS)
    assert summary["sku_unknown_symbols"] == list(SKU_UNKNOWN_SYMBOLS)
    assert summary["stage_kind_ids"] == list(STAGE_KIND_IDS)
    assert summary["evaluation_criterion_ids"] == list(EVALUATION_CRITERION_IDS)
    assert summary["first_token_in_prefill"] is True
    assert summary["decode_only_excludes_first_token"] is True
    assert summary["t1_mac_equal_does_not_imply_equal_traffic"] is True
    assert summary["graph_envelope_is_not_leaf"] is True
    assert summary["missing_coverage_is_not_zero"] is True
    assert summary["microbenchmark_cannot_pass_mapping"] is True
    assert summary["e2e_required_alongside_microbenchmarks"] is True
    assert summary["kernel_cannot_pass_mapping"] is True
    assert summary["e2e_primary_id"] == "e2e_latency_ms"
    assert summary["complete_request_vs_decode_only_forbidden"] is True
    assert summary["example_T_is_not_prompt_matrix"] is True
    assert summary["sku_table_filled"] is False
    assert summary["hardware_selected"] is False
    assert summary["prompt_matrix_selected"] is False
    assert summary["reproducibility_protocol_selected"] is False
    assert summary["ledger_open_question_hardware_closed"] is False
    assert summary["mapping_winner_selected"] is False
    assert summary["winner_selected_without_measurements"] is False
    assert summary["ledger_open_question_mapping_winner_closed"] is False
    assert summary["benchmarks_run"] is False
    assert summary["toks_measured_here"] is False
    assert summary["experiments_run"] is False
    assert summary["methodology_defined"] is True
    assert summary["toks_is_not_quality_axis"] is True
    assert summary["nll_measured_here"] is False
    assert summary["gguf_payload_inspected"] is False
    assert summary["quartz_inspected"] is False
    assert summary["llama_inspected"] is False
    assert summary["device_query_run"] is False
    assert summary["nsight_run"] is False
    assert summary["achieved_occupancy_measured_here"] is False
    assert summary["decode_prefill_share_metric_identity"] is False
    assert summary["vision_eval_deferred"] is True
    assert summary["n_embed_instances"] == 2
    assert summary["n_gated_attn_instances"] == 17
    assert summary["n_gated_delta_net_instances"] == 48
    assert summary["n_mlp_instances"] == 65
    assert summary["n_lm_head_instances"] == 2
    assert summary["n_mtp_mix_instances"] == 1
    assert summary["decode_work_identity"] == "C+AT"
    assert summary["prefill_work_identity"] == "TC+AT(T+1)/2"
    assert summary["decode_T_new"] == 1
    assert summary["decode_incoming_state"] == "populated"
    assert summary["prefill_incoming_state"] == "zeros"
    assert summary["first_token_convention_id"] == "ttft_in_prefill"
    assert summary["headings"] == list(REQUIRED_HEADINGS)
    assert summary["diagram_ids"] == list(DIAGRAM_JSON_IDS)
    assert summary["metric_family_ids"] == list(METRIC_FAMILY_IDS)
    assert summary["metric_class_ids"] == list(METRIC_CLASS_IDS)
    assert summary["decode_metric_ids"] == list(DECODE_METRIC_IDS)
    assert summary["decode_metric_classes"] == list(DECODE_METRIC_CLASSES)
    assert summary["prefill_metric_ids"] == list(PREFILL_METRIC_IDS)
    assert summary["kernel_metric_ids"] == list(KERNEL_METRIC_IDS)
    assert summary["memory_metric_ids"] == list(MEMORY_METRIC_IDS)
    assert summary["e2e_metric_ids"] == list(E2E_METRIC_IDS)
    assert summary["identity_field_ids"] == list(IDENTITY_FIELD_IDS)
    assert summary["coverage_field_ids"] == list(COVERAGE_FIELD_IDS)
    assert summary["time_accounting_rule_ids"] == list(TIME_ACCOUNTING_RULE_IDS)
    assert summary["methodology_risk_ids"] == list(METHODOLOGY_RISK_IDS)
    assert summary["methodology_risk_severities"] == list(
        METHODOLOGY_RISK_SEVERITIES
    )
    assert summary["methodology_risk_usefulness_label"] == "HYPOTHESIS"
    assert summary["traffic_channel_ids"] == list(TRAFFIC_CHANNEL_IDS)
    assert summary["canonical_sentence_logical"] == CANONICAL_SENTENCE_LOGICAL
    assert summary["canonical_sentence_methodology"] == (
        CANONICAL_SENTENCE_METHODOLOGY
    )
    assert summary["canonical_sentence_e2e"] == CANONICAL_SENTENCE_E2E
    assert summary["canonical_sentence_open_question"] == (
        CANONICAL_SENTENCE_OPEN_QUESTION
    )
    assert summary["canonical_sentence_winner"] == CANONICAL_SENTENCE_WINNER
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
    assert len(high_from_ops) == summary["n_methodology_high"]
    assert len(medium_from_ops) == summary["n_methodology_medium"]
    assert len(low_from_ops) == summary["n_methodology_low"]


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
    """Compare documented JSON against a live performance-validation object.

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


def _exception_spans(text: str) -> list[tuple[int, int]]:
    """Return character spans where forbidden-winner phrases are allowed.

    Args:
        text: Full markdown document.

    Returns:
        Inclusive-start exclusive-end spans covering canonical sentences and
        the dossier's allowed wrappers.
    """

    phrases = (
        CANONICAL_SENTENCE_LOGICAL,
        CANONICAL_SENTENCE_METHODOLOGY,
        CANONICAL_SENTENCE_E2E,
        CANONICAL_SENTENCE_OPEN_QUESTION,
        CANONICAL_SENTENCE_WINNER,
        METHODOLOGY_QUESTION_SENTENCE,
        *ALLOWED_EXCEPTION_PHRASES,
    )
    spans: list[tuple[int, int]] = []
    for phrase in phrases:
        start = 0
        while True:
            index = text.find(phrase, start)
            if index < 0:
                break
            spans.append((index, index + len(phrase)))
            start = index + 1
    return spans


def _inside_exception(
    start: int, end: int, spans: list[tuple[int, int]]
) -> bool:
    """Return whether ``[start, end)`` lies inside an allowed exception span.

    Args:
        start: Match start index.
        end: Match end index.
        spans: Allowed spans from :func:`_exception_spans`.

    Returns:
        True when the match is fully contained in an allowed span.
    """

    return any(span_start <= start and end <= span_end for span_start, span_end in spans)


def check_performance_validation(live: dict, design_path: Path) -> None:
    """Check a performance-validation markdown file against a live summary.

    Args:
        live: Performance-validation summary object from sitting config.
        design_path: Path to ``performance-validation.md``.

    Raises:
        PerformanceValidationMismatch: On heading, JSON, diagram, or token
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

    fence_match: re.Match[str] | None = None
    try:
        documented, fence_match = first_json_fence(text)
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
            stripped.startswith("flowchart TB")
            or stripped.startswith("flowchart LR")
        ):
            differences.append(
                "mermaid fence body does not start with flowchart TB or "
                "flowchart LR"
            )
        if "flowchart" not in body:
            differences.append("mermaid fence does not contain flowchart")
        missing_ids = [
            node_id for node_id in DIAGRAM_MERMAID_IDS if node_id not in body
        ]
        if missing_ids:
            differences.append(f"diagram 1: missing node ids {missing_ids}")
        sku_heading = re.search(
            r"^## SKU-fill protocol and unselected hardware\s*$",
            text,
            flags=re.MULTILINE,
        )
        risk_heading = re.search(
            r"^## Methodology risks\s*$", text, flags=re.MULTILINE
        )
        if sku_heading is None or risk_heading is None:
            differences.append(
                "missing SKU-fill protocol or Methodology risks heading"
            )
        elif not (sku_heading.end() <= match.start() < risk_heading.start()):
            differences.append(
                "mermaid fence is not under ## SKU-fill protocol and "
                "unselected hardware"
            )
        else:
            caption = text[sku_heading.end() : match.start()]
            if "OPENQ" not in caption or "not a selected winner" not in caption:
                differences.append(
                    "mermaid caption missing OPENQ or 'not a selected winner'"
                )

    if CANONICAL_SENTENCE_LOGICAL not in text:
        differences.append("canonical_sentence_logical not present verbatim")
    if CANONICAL_SENTENCE_METHODOLOGY not in text:
        differences.append(
            "canonical_sentence_methodology not present verbatim"
        )
    if CANONICAL_SENTENCE_E2E not in text:
        differences.append("canonical_sentence_e2e not present verbatim")
    if CANONICAL_SENTENCE_OPEN_QUESTION not in text:
        differences.append(
            "canonical_sentence_open_question not present verbatim"
        )
    if CANONICAL_SENTENCE_WINNER not in text:
        differences.append("canonical_sentence_winner not present verbatim")
    if METHODOLOGY_QUESTION_SENTENCE not in text:
        differences.append("methodology_question_sentence not present verbatim")

    convention = re.search(
        r"^## Measurement convention\s*$", text, flags=re.MULTILINE
    )
    decode_heading = re.search(
        r"^## Decode metrics\s*$", text, flags=re.MULTILINE
    )
    if convention is None or decode_heading is None:
        differences.append("missing Measurement convention or Decode metrics")
    else:
        section = text[convention.end() : decode_heading.start()]
        for label, sentence in (
            ("canonical_sentence_logical", CANONICAL_SENTENCE_LOGICAL),
            ("canonical_sentence_methodology", CANONICAL_SENTENCE_METHODOLOGY),
            ("canonical_sentence_e2e", CANONICAL_SENTENCE_E2E),
            (
                "canonical_sentence_open_question",
                CANONICAL_SENTENCE_OPEN_QUESTION,
            ),
            ("canonical_sentence_winner", CANONICAL_SENTENCE_WINNER),
            ("methodology_question_sentence", METHODOLOGY_QUESTION_SENTENCE),
        ):
            if sentence not in section:
                differences.append(f"{label} missing under Measurement convention")

    for label, ids in SUBSTRING_ID_GROUPS:
        for item_id in ids:
            if item_id not in text:
                differences.append(f"{label} {item_id!r} missing as substring")

    if "HYPOTHESIS" not in text:
        differences.append("word HYPOTHESIS missing")
    if "not measurements" not in text:
        differences.append("phrase 'not measurements' missing")

    for match in FORBIDDEN_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        differences.append(f"line {line}: forbidden token {match.group(0)!r}")

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

    try:
        vis_start, vis_end = _deferred_vision_span(text)
        if not UNKNOWN_RE.search(text, vis_start, vis_end):
            differences.append("word UNKNOWN missing in Deferred vision")
    except AssertionError as exc:
        differences.append(str(exc))

    for number in LOCKED_DOCUMENT_INTEGERS:
        token = str(number)
        if token not in text:
            differences.append(
                f"locked number {token} missing as decimal substring"
            )

    if differences:
        raise PerformanceValidationMismatch(differences)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the performance-validation checker.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Instantiate Qwen3.8-27B language+MTP performance-validation "
            "summary from text_config and check "
            "docs/architecture/performance-validation.md."
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
        help="Write performance-validation summary JSON to stdout.",
    )
    parser.add_argument(
        "--performance-validation",
        type=Path,
        default=None,
        help="Markdown path whose first json fence must match live summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the performance-validation checker CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch, 2 missing
        config.
    """

    args = _parse_args(argv)
    check_path = args.performance_validation
    print_json = args.json or check_path is None
    try:
        config = load_config(args.config)
        summary = instantiate_performance_validation(config)
        if check_path is not None:
            if not check_path.is_file():
                print(
                    f"performance-validation file not found: {check_path}",
                    file=sys.stderr,
                )
                return 1
            check_performance_validation(summary, check_path)
    except MissingConfig as exc:
        print(str(exc.path), file=sys.stderr)
        return 2
    except PerformanceValidationMismatch as exc:
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

    if print_json:
        sys.stdout.write(dumps_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
