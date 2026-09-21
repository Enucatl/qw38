#!/usr/bin/env python3
"""Check Qwen3.8-27B clean-sheet architecture synthesis against sitting text_config.

Reads ``text_config`` from a Transformers ``config.json`` (no safetensor
payloads, no GPU query, no GGUF) and either prints the clean-sheet
summary object or checks that
``docs/architecture/clean-sheet-architecture.md`` and
``docs/architecture/experiment-backlog.md`` match it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

AUTHORITY = "docs/architecture/plan.md"
DELIVERABLE_ARCHITECTURE = "docs/architecture/clean-sheet-architecture.md"
DELIVERABLE_BACKLOG = "docs/architecture/experiment-backlog.md"

CANONICAL_SENTENCE_LOGICAL = (
    "Logical values in this document are graph nodes. They do not imply "
    "physical allocation, materialization, buffer reuse, or kernel fusion."
)
CANONICAL_SENTENCE_SYNTHESIS = (
    "This document is a Phase 1 synthesis of TASK-01 through TASK-19, "
    "not a selected runtime and not measurements."
)
CANONICAL_SENTENCE_ALTERNATIVES = (
    "Unselected alternatives remain alternatives; listing them is not "
    "selecting a winner."
)
CANONICAL_SENTENCE_UNRUN = (
    "The experiment backlog is dependency-ordered and unrun; no "
    "implementation benchmarks or NLL are collected in this study task."
)
CANONICAL_SENTENCE_BACKLOG = "Experiment entries in this document are a backlog, not results."
CANONICAL_SENTENCE_DEPENDS = (
    "Dependency order is required; an entry may not run before every id "
    "in depends_on has a measured status."
)
CANONICAL_SENTENCE_EVAL_SPLIT = (
    "Microbenchmarks cannot pass a CUDA mapping; reconstruction is not "
    "model-level quality; Q4_K_M is a future black-box Pareto reference, "
    "not a requirement."
)
SYNTHESIS_QUESTION_SENTENCE = (
    "The required architecture chain and evidence classes are "
    "synthesized; an architecture diagram and alternatives are included; "
    "dependency-ordered experiment entries list all requested fields; no "
    "winners are selected and no experiments are run in this study task."
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

COMPILER_STAGE_IDS: tuple[str, ...] = (
    "validation",
    "analysis",
    "quantization",
    "packing",
    "metadata",
    "integrity",
)

COMPILER_PROFILE_IDS: tuple[str, ...] = ("quality", "balanced", "compression")

CONTROL_PROFILE_ID = "control"

ARTIFACT_APPROACH_IDS: tuple[str, ...] = (
    "portable_only",
    "backend_specialized_only",
    "portable_plus_specialized_views",
    "manifest_plus_backend_blobs",
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

INTEGRITY_ALGORITHM_CANDIDATES: tuple[str, ...] = ("none", "checksum")

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

HOIST_HYPOTHESIS_IDS: tuple[str, ...] = (
    "hoist_embed_next",
    "overlap_fanout_h64",
    "reuse_E",
    "reuse_W_lm",
    "reuse_h64",
)

REPRESENTATION_HYPOTHESIS_IDS: tuple[str, ...] = (
    "distinct_prefill_gemm_view",
    "distinct_decode_gemv_view",
    "shared_view_both_modes",
    "dual_view_binding",
)

MODE_HYPOTHESIS_IDS: tuple[str, ...] = (
    "token_serial_prefill",
    "inference_last_logits",
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

SENSITIVE_OPS: tuple[str, ...] = (
    "param_bf16",
    "residual_stream",
    "live_across_gates",
    "silu_sigmoid",
    "rms_hidden",
    "rms_head",
    "l2_gdn",
    "softmax_over_T",
    "attn_av_over_T",
    "gemm_k5120",
    "gemm_k6144",
    "gemm_k17408",
    "gemm_lm_head",
    "gdn_S_recurrent",
    "gdn_inner_d128",
    "gdn_alpha_beta",
    "rope_phase",
    "state_kv_bf16",
    "state_c_bf16",
    "s_below_f32",
    "conv_fir",
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

BOTTLENECK_LABELS: tuple[str, ...] = (
    "weight_memory",
    "vocab_memory",
    "state_memory",
    "kv_memory",
    "quadratic_attn",
    "compute",
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

TEACHER_FORCED_METRIC_IDS: tuple[str, ...] = (
    "tf_nll_language",
    "tf_nll_mtp",
    "tf_nll_complete",
    "tf_delta_vs_control",
    "tf_kl_vs_control",
)

DECODE_METRIC_IDS: tuple[str, ...] = (
    "dec_toks",
    "dec_step_ms",
    "dec_p50_step_ms",
    "dec_p99_step_ms",
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

COMPARISON_DIMENSION_IDS: tuple[str, ...] = (
    "compile_once",
    "load_convert",
    "byte_sequence",
    "tile_freedom",
    "store_amplification",
    "consumer_portability",
)

SYNTHESIZED_TASK_IDS: tuple[str, ...] = tuple(
    f"TASK-{index:02d}" for index in range(1, 20)
)

SATELLITE_TASK_IDS: tuple[str, ...] = (
    "TASK-05",
    "TASK-06",
    "TASK-09",
    "TASK-12",
    "TASK-16",
)

CHAIN_IDS: tuple[str, ...] = (
    "mathematics",
    "logical_dataflow",
    "lifetime_and_state",
    "numerical_sensitivity",
    "precision_strategy",
    "custom_quantization",
    "custom_physical_layouts",
    "offline_model_compiler",
    "semantic_graph",
    "decode_prefill_plans",
    "cuda_mappings",
    "experimental_validation",
)

CHAIN_SOURCE_TASK_GROUPS: dict[str, tuple[str, ...]] = {
    "mathematics": ("TASK-01", "TASK-02"),
    "logical_dataflow": ("TASK-03",),
    "lifetime_and_state": ("TASK-04",),
    "numerical_sensitivity": ("TASK-07",),
    "precision_strategy": ("TASK-07",),
    "custom_quantization": ("TASK-08",),
    "custom_physical_layouts": ("TASK-15",),
    "offline_model_compiler": ("TASK-09", "TASK-10"),
    "semantic_graph": ("TASK-11", "TASK-12"),
    "decode_prefill_plans": ("TASK-13", "TASK-14"),
    "cuda_mappings": ("TASK-16", "TASK-17"),
    "experimental_validation": ("TASK-18", "TASK-19"),
}

EVIDENCE_CLASS_IDS: tuple[str, ...] = (
    "DERIVED",
    "OBSERVED",
    "MEASURED",
    "HYPOTHESIS",
    "UNKNOWN",
)

TASK_EVIDENCE_CLASSES: dict[str, tuple[str, ...]] = {
    "TASK-01": ("OBSERVED", "DERIVED"),
    "TASK-02": ("DERIVED",),
    "TASK-03": ("DERIVED",),
    "TASK-04": ("DERIVED",),
    "TASK-05": ("MEASURED",),
    "TASK-06": ("DERIVED", "HYPOTHESIS"),
    "TASK-07": ("DERIVED", "HYPOTHESIS"),
    "TASK-08": ("OBSERVED", "MEASURED", "DERIVED", "HYPOTHESIS"),
    "TASK-09": ("DERIVED", "HYPOTHESIS"),
    "TASK-10": ("DERIVED", "HYPOTHESIS"),
    "TASK-11": ("DERIVED",),
    "TASK-12": ("DERIVED", "HYPOTHESIS"),
    "TASK-13": ("DERIVED", "HYPOTHESIS"),
    "TASK-14": ("DERIVED", "HYPOTHESIS"),
    "TASK-15": ("DERIVED", "HYPOTHESIS"),
    "TASK-16": ("DERIVED", "UNKNOWN"),
    "TASK-17": ("DERIVED", "HYPOTHESIS", "UNKNOWN"),
    "TASK-18": ("DERIVED", "HYPOTHESIS"),
    "TASK-19": ("DERIVED", "HYPOTHESIS", "UNKNOWN"),
}

TASKS_OPEN_QUESTION_CLOSED: tuple[str, ...] = (
    "TASK-01",
    "TASK-02",
    "TASK-03",
    "TASK-04",
    "TASK-05",
    "TASK-06",
    "TASK-11",
    "TASK-12",
    "TASK-13",
)

ALTERNATIVE_FAMILY_IDS: tuple[str, ...] = (
    "alt_recipes",
    "alt_profiles",
    "alt_artifact_approaches",
    "alt_consumer_sequences",
    "alt_integrity",
    "alt_representation",
    "alt_mode",
    "alt_fusion",
    "alt_hoist",
    "alt_orderings",
    "alt_tiles",
    "alt_decompositions",
    "alt_conversions",
    "alt_mappings",
)

ALTERNATIVE_FAMILY_COUNTS: tuple[int, ...] = (
    22,
    3,
    4,
    7,
    2,
    4,
    2,
    22,
    5,
    14,
    9,
    9,
    4,
    18,
)

REMAINING_OPEN_QUESTION_IDS: tuple[str, ...] = (
    "oq_risk_survival",
    "oq_pareto_frontier",
    "oq_artifact_boundary",
    "oq_ideal_byte_sequence",
    "oq_integrity_algorithm",
    "oq_fusion_winner",
    "oq_decode_prefill_views",
    "oq_parallel_decomposition",
    "oq_sku_limits",
    "oq_mapping_winner",
    "oq_eval_corpora",
    "oq_hardware_protocol",
)

EXPERIMENT_FIELD_IDS: tuple[str, ...] = (
    "id",
    "title",
    "depends_on",
    "source_tasks",
    "chain_step",
    "hypothesis",
    "alternatives",
    "eval_family",
    "metric_ids",
    "disconfirming_outcome",
    "blocking_unknowns",
    "status",
)

EXPERIMENT_EVAL_FAMILY_IDS: tuple[str, ...] = (
    "infrastructure",
    "quality",
    "format",
    "performance",
)

EXPERIMENT_STATUS_UNRUN = "unrun"

EMPTY_METRIC_EXPERIMENT_IDS: frozenset[str] = frozenset(
    {"exp_sku_fill", "exp_identity_harness", "exp_integrity_algorithm"}
)
EMPTY_ALTERNATIVE_EXPERIMENT_IDS: frozenset[str] = frozenset(
    {"exp_identity_harness"}
)
EMPTY_DEPENDS_EXPERIMENT_IDS: frozenset[str] = frozenset(
    {"exp_sku_fill", "exp_identity_harness"}
)

ARCHITECTURE_HEADINGS: tuple[str, ...] = (
    "Authority",
    "Synthesis convention",
    "Architecture chain and evidence classes",
    "Locked structure",
    "Architecture diagram and alternatives",
    "Remaining unknowns",
    "Deferred vision",
    "Machine-checkable summary JSON",
)

BACKLOG_HEADINGS: tuple[str, ...] = (
    "Authority",
    "Backlog convention",
    "Requested fields",
    "Dependency-ordered experiment entries",
    "Deferred vision",
    "Machine-checkable summary JSON",
)

DIAGRAM_IDS: tuple[str, ...] = (
    "math",
    "dataflow",
    "lifetime",
    "sens",
    "quant",
    "layout",
    "compiler",
    "graph",
    "decode",
    "prefill",
    "cuda",
    "backlog",
    "alts",
    "openq",
)

DIAGRAM_MERMAID_IDS: tuple[str, ...] = (
    "MATH",
    "DATAFLOW",
    "LIFETIME",
    "SENS",
    "QUANT",
    "LAYOUT",
    "COMPILER",
    "GRAPH",
    "DECODE",
    "PREFILL",
    "CUDA",
    "BACKLOG",
    "ALTS",
    "OPENQ",
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
    125,
    19,
    12,
    16,
)

FORBIDDEN_WINNER_PHRASES: tuple[str, ...] = (
    "selected mapping winner",
    "winning kernel",
    "winning mapping",
    "selected recipe",
    "selected fusion",
    "FROZEN_FOR_COMPARATIVE_REVIEW",
    "should use this GPU",
    "experiments were run",
    "NLL was measured in this study",
    "SKU table is filled",
    "tok/s is the quality axis",
)

ALLOWED_WRAPPER_PHRASES: tuple[str, ...] = (
    "selects no",
    "not a selected",
    "remain unselected",
    "not measurements",
    "unrun",
)

JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)
MERMAID_FENCE_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
HEADING_RE = re.compile(r"^## (?!#)(.+)$", re.MULTILINE)
FORBIDDEN_RE = re.compile(r"TBD|TODO|\?\?\?")
UNKNOWN_RE = re.compile(r"UNKNOWN")

SCHEMA_KEYS: tuple[str, ...] = (
    "authority",
    "deliverable_architecture",
    "deliverable_backlog",
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
    "n_candidate_recipes",
    "candidate_recipe_ids",
    "n_policy_families",
    "policy_families",
    "n_compiler_stages",
    "compiler_stage_ids",
    "n_compiler_profiles",
    "compiler_profile_ids",
    "control_profile_id",
    "n_artifact_approaches",
    "artifact_approach_ids",
    "n_consumer_sequences",
    "consumer_sequence_ids",
    "integrity_algorithm_candidates",
    "n_fusion_hypotheses",
    "fusion_hypothesis_ids",
    "n_hoist_hypotheses",
    "hoist_hypothesis_ids",
    "n_representation_hypotheses",
    "representation_hypothesis_ids",
    "n_mode_hypotheses",
    "mode_hypothesis_ids",
    "n_orderings",
    "ordering_ids",
    "n_tile_families",
    "tile_family_ids",
    "n_parallel_decompositions",
    "parallel_decomposition_ids",
    "n_conversion_hypotheses",
    "conversion_hypothesis_ids",
    "n_mappings",
    "mapping_ids",
    "n_sku_unknown_symbols",
    "sku_unknown_symbols",
    "n_sensitive_ops",
    "sensitive_ops",
    "quality_high_ids",
    "bottleneck_labels",
    "evaluation_criterion_ids",
    "pareto_axis_ids",
    "corpus_class_ids",
    "reconstruction_diagnostic_ids",
    "teacher_forced_metric_ids",
    "decode_metric_ids",
    "prefill_metric_ids",
    "kernel_metric_ids",
    "memory_metric_ids",
    "e2e_metric_ids",
    "comparison_dimension_ids",
    "n_synthesized_tasks",
    "synthesized_task_ids",
    "n_satellite_tasks",
    "satellite_task_ids",
    "n_chain_steps",
    "chain_ids",
    "chain_source_task_groups",
    "n_evidence_classes",
    "evidence_class_ids",
    "task_evidence_classes",
    "task_open_question_closed",
    "n_tasks_open_question_closed",
    "n_tasks_open_question_open",
    "n_alternative_families",
    "alternative_family_ids",
    "alternative_family_counts",
    "n_catalogued_alternatives",
    "n_alternatives_selected",
    "alternative_usefulness_label",
    "n_remaining_open_questions",
    "remaining_open_question_ids",
    "n_remaining_open_questions_closed",
    "n_experiment_fields",
    "experiment_field_ids",
    "n_experiment_eval_families",
    "experiment_eval_family_ids",
    "experiment_status_unrun",
    "n_experiments",
    "experiment_ids",
    "experiments",
    "experiment_selected",
    "n_experiments_run",
    "n_empty_metric_experiments",
    "n_empty_alternative_experiments",
    "n_diagrams",
    "diagram_ids",
    "n_architecture_headings",
    "architecture_headings",
    "n_backlog_headings",
    "backlog_headings",
    "n_headings",
    "chain_follows_plan_path",
    "chain_is_not_ledger_order",
    "experiment_order_is_topological",
    "ledger_open_question_alternatives_catalogued",
    "frozen_for_comparative_review",
    "pareto_frontier_selected",
    "compiler_profile_selected",
    "artifact_boundary_selected",
    "ideal_byte_sequence_selected",
    "integrity_algorithm_selected",
    "fusion_winner_selected",
    "decode_prefill_distinct_views_selected",
    "layout_winner_selected",
    "ordering_selected",
    "tile_size_selected",
    "mapping_winner_selected",
    "winner_selected_without_measurements",
    "hardware_selected",
    "prompt_matrix_selected",
    "reproducibility_protocol_selected",
    "calibration_corpus_selected",
    "eval_corpus_selected",
    "acceptance_frontier_selected",
    "sku_table_filled",
    "hypothesis_survival_selected",
    "n_mappings_selected",
    "n_fusion_hypotheses_selected",
    "keep_source_is_not_quality_winner",
    "control_profile_is_keep_source",
    "reconstruction_is_not_quality",
    "toks_is_not_quality_axis",
    "microbenchmark_cannot_pass_mapping",
    "e2e_required_alongside_microbenchmarks",
    "decode_prefill_share_graph",
    "decode_prefill_share_artifact",
    "decode_prefill_share_metric_identity",
    "experiments_run",
    "nll_measured_here",
    "toks_measured_here",
    "benchmarks_run",
    "payloads_restreamed",
    "gguf_payload_inspected",
    "gguf_is_not_design_authority",
    "gguf_is_pareto_reference",
    "gguf_is_not_a_requirement",
    "quartz_inspected",
    "llama_inspected",
    "llama_is_not_design_authority",
    "device_query_run",
    "nsight_run",
    "vision_eval_deferred",
    "safetensors_is_source_not_runtime",
    "canonical_sentence_logical",
    "canonical_sentence_synthesis",
    "canonical_sentence_alternatives",
    "canonical_sentence_unrun",
    "canonical_sentence_backlog",
    "canonical_sentence_depends",
    "canonical_sentence_eval_split",
    "synthesis_question_sentence",
)


class MissingConfig(Exception):
    """Raised when the config path is absent or unreadable."""

    def __init__(self, path: Path) -> None:
        super().__init__(str(path))
        self.path = path


class CleanSheetMismatch(Exception):
    """Raised when a synthesis markdown file fails a content check."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def dumps_summary(summary: dict) -> str:
    """Pretty-print the clean-sheet summary object.

    Args:
        summary: Object produced by :func:`instantiate_clean_sheet_architecture`.

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


def _experiment(
    experiment_id: str,
    title: str,
    depends_on: list[str],
    source_tasks: list[str],
    chain_step: str,
    hypothesis: str,
    alternatives: list[str],
    eval_family: str,
    metric_ids: list[str],
    disconfirming_outcome: str,
    blocking_unknowns: list[str],
) -> dict:
    """Build one experiment object with locked field order.

    Args:
        experiment_id: Stable experiment id.
        title: Short name.
        depends_on: Earlier experiment ids.
        source_tasks: TASK-NN citations.
        chain_step: One of ``CHAIN_IDS``.
        hypothesis: HYPOTHESIS under test.
        alternatives: Unselected candidate ids.
        eval_family: One of ``EXPERIMENT_EVAL_FAMILY_IDS``.
        metric_ids: Metric or diagnostic ids.
        disconfirming_outcome: Result that rejects the hypothesis.
        blocking_unknowns: Remaining open-question ids.

    Returns:
        Ordered experiment dict with the twelve requested fields.
    """

    return {
        "id": experiment_id,
        "title": title,
        "depends_on": list(depends_on),
        "source_tasks": list(source_tasks),
        "chain_step": chain_step,
        "hypothesis": hypothesis,
        "alternatives": list(alternatives),
        "eval_family": eval_family,
        "metric_ids": list(metric_ids),
        "disconfirming_outcome": disconfirming_outcome,
        "blocking_unknowns": list(blocking_unknowns),
        "status": EXPERIMENT_STATUS_UNRUN,
    }


def _locked_experiments() -> list[dict]:
    """Return the sixteen locked unrun experiment entries.

    Returns:
        Experiment objects in topological ``experiment_ids`` order.
    """

    return [
        _experiment(
            "exp_sku_fill",
            "Fill sitting SKU table",
            [],
            ["TASK-16", "TASK-19"],
            "cuda_mappings",
            (
                "Sitting-device capacity and peak symbols can be filled from "
                "a declared sitting identity; a datasheet is not a measurement."
            ),
            list(SKU_UNKNOWN_SYMBOLS),
            "infrastructure",
            [],
            (
                "Filling sku_unknown_symbols from an unnamed datasheet or "
                "treating datasheet peaks as mem_hbm_gbps."
            ),
            ["oq_hardware_protocol"],
        ),
        _experiment(
            "exp_identity_harness",
            "Identity coverage and NLL harness",
            [],
            ["TASK-18", "TASK-19"],
            "experimental_validation",
            (
                "Matching-identity capture is implementable without mixing "
                "decode-only, prefill, and complete-request windows."
            ),
            [],
            "infrastructure",
            [],
            (
                "Mixed (T+N)/t reported as decode-only, or missing coverage "
                "treated as zero excess."
            ),
            ["oq_hardware_protocol", "oq_eval_corpora"],
        ),
        _experiment(
            "exp_control_keep_source",
            "Control keep_source baseline",
            ["exp_identity_harness"],
            ["TASK-08", "TASK-10", "TASK-18", "TASK-19"],
            "experimental_validation",
            (
                "keep_source control compile reproduces the TASK-02 map at "
                "conceptual BF16 parameters and F32 S; it is an identity "
                "control, not a quality winner."
            ),
            [CONTROL_PROFILE_ID],
            "quality",
            [
                "tf_nll_complete",
                "tf_nll_language",
                "tf_nll_mtp",
                "e2e_latency_ms",
                "dec_toks",
                "pre_toks",
            ],
            "Undefined control NLL, or ranking keep_source as a quality winner.",
            ["oq_eval_corpora", "oq_hardware_protocol"],
        ),
        _experiment(
            "exp_reconstruction_screen",
            "Reconstruction screens over candidate recipes",
            ["exp_control_keep_source"],
            ["TASK-08", "TASK-18"],
            "custom_quantization",
            (
                "Local reconstruction can fail catastrophic recipes and "
                "cannot place a Pareto quality point."
            ),
            list(CANDIDATE_RECIPE_IDS),
            "quality",
            list(RECONSTRUCTION_DIAGNOSTIC_IDS),
            (
                "Placing a Pareto point from r_* diagnostics, or treating "
                "reconstruction as NLL."
            ),
            [],
        ),
        _experiment(
            "exp_family_nll",
            "Teacher-forced family NLL",
            ["exp_reconstruction_screen"],
            ["TASK-07", "TASK-08", "TASK-18"],
            "custom_quantization",
            (
                "Family-specific candidate sets, especially quality_high_ids "
                "families, change complete-map NLL versus keep_source."
            ),
            list(POLICY_FAMILIES),
            "quality",
            ["tf_nll_complete", "tf_delta_vs_control", "tf_kl_vs_control"],
            (
                "Language-only NLL substituting for the complete map, or "
                "sampling used as NLL."
            ),
            ["oq_eval_corpora"],
        ),
        _experiment(
            "exp_state_precision",
            "Persistent-state precision",
            ["exp_family_nll"],
            ["TASK-04", "TASK-07", "TASK-08", "TASK-18"],
            "precision_strategy",
            (
                "Narrowing conceptual F32 S or BF16 K,V,C changes NLL "
                "and/or state reconstruction."
            ),
            ["state_kv", "state_c", "state_s", "keep_source", "narrow_bf16"],
            "quality",
            ["r_state_kv", "r_state_s", "tf_nll_complete"],
            (
                "Treating chunkwise GDN as zero S traffic, or dropping MTP "
                "from the complete map."
            ),
            [],
        ),
        _experiment(
            "exp_risk_survival",
            "Numerical-risk hypothesis survival",
            ["exp_family_nll", "exp_state_precision"],
            ["TASK-07", "TASK-18"],
            "numerical_sensitivity",
            (
                "Some of the twenty sensitive-op HYPOTHESIS severities "
                "survive model-level validation."
            ),
            list(SENSITIVE_OPS),
            "quality",
            ["tf_nll_complete", "r_act_residual", "r_act_logits"],
            "Declaring hypothesis_survival_selected without complete-map NLL.",
            ["oq_eval_corpora"],
        ),
        _experiment(
            "exp_pareto_profiles",
            "Profile Pareto versus Q4_K_M reference",
            ["exp_risk_survival"],
            ["TASK-08", "TASK-10", "TASK-18"],
            "custom_quantization",
            (
                "Legal quality, balanced, and compression recipe maps form a "
                "Pareto surface on quality and compression with Q4_K_M as a "
                "reference point, not a requirement."
            ),
            ["quality", "balanced", "compression", "control"],
            "quality",
            [
                "axis_quality",
                "axis_compression",
                "axis_reference_q4km",
                "tf_nll_complete",
            ],
            "Requiring Q4_K_M to be beaten, or using tok/s as Pareto Y.",
            ["oq_eval_corpora"],
        ),
        _experiment(
            "exp_artifact_boundary",
            "Portable versus backend-specialized artifact",
            ["exp_reconstruction_screen"],
            ["TASK-09", "TASK-10"],
            "offline_model_compiler",
            (
                "Artifact-boundary choice changes load/convert cost and "
                "consumer portability without changing the TASK-02 map."
            ),
            list(ARTIFACT_APPROACH_IDS),
            "format",
            list(COMPARISON_DIMENSION_IDS),
            "Selecting artifact_boundary without a six-dimension comparison.",
            [],
        ),
        _experiment(
            "exp_consumer_sequence",
            "Ideal consumer byte sequences",
            ["exp_artifact_boundary"],
            ["TASK-09", "TASK-13", "TASK-14"],
            "custom_physical_layouts",
            (
                "One of seven consumer sequences is ideal per access class "
                "after a boundary is chosen."
            ),
            list(CONSUMER_SEQUENCE_IDS),
            "format",
            ["mem_weight_bytes", "k_node_ms"],
            (
                "Ranking sequences by wall time without coverage, or "
                "selecting seq_specialized_tile extents here."
            ),
            ["oq_hardware_protocol"],
        ),
        _experiment(
            "exp_integrity_algorithm",
            "Integrity algorithm none versus checksum",
            ["exp_artifact_boundary"],
            ["TASK-09", "TASK-10"],
            "offline_model_compiler",
            (
                "checksum versus none is an integrity and load-safety "
                "choice, not a quality or NLL decision."
            ),
            list(INTEGRITY_ALGORITHM_CANDIDATES),
            "format",
            [],
            "Treating checksum as a quality winner.",
            [],
        ),
        _experiment(
            "exp_decode_prefill_views",
            "Decode/prefill representation views",
            ["exp_consumer_sequence"],
            ["TASK-13", "TASK-14"],
            "decode_prefill_plans",
            (
                "Distinct GEMV/GEMM views, a shared view, or dual-view "
                "binding changes traffic versus a single sequence; mode "
                "hypotheses remain unselected until measured."
            ),
            list(REPRESENTATION_HYPOTHESIS_IDS) + list(MODE_HYPOTHESIS_IDS),
            "performance",
            ["pre_toks", "dec_toks", "mem_act_bytes", "e2e_latency_ms"],
            (
                "Reporting T=1 prefill as decode-only, or selecting views "
                "without end-to-end measurement."
            ),
            ["oq_hardware_protocol"],
        ),
        _experiment(
            "exp_fusion_hoist",
            "Fusion and hoist hypotheses",
            ["exp_control_keep_source"],
            ["TASK-12", "TASK-13", "TASK-14", "TASK-16", "TASK-17"],
            "semantic_graph",
            (
                "Some of twenty-two fusion and five hoist hypotheses improve "
                "end-to-end latency without violating TASK-16 occupancy "
                "algebra."
            ),
            list(FUSION_HYPOTHESIS_IDS) + list(HOIST_HYPOTHESIS_IDS),
            "performance",
            [
                "e2e_latency_ms",
                "k_node_ms",
                "k_occupancy_achieved",
                "mem_working_set",
            ],
            (
                "Selecting fusion from microbenchmarks only, or "
                "occupancy-unaware fusion."
            ),
            ["oq_hardware_protocol"],
        ),
        _experiment(
            "exp_layout_decomp",
            "Parallel decompositions justifying layouts",
            ["exp_decode_prefill_views", "exp_sku_fill"],
            ["TASK-15", "TASK-17"],
            "custom_physical_layouts",
            (
                "A planned parallel decomposition justifies a candidate "
                "ordering and tile family per layout object."
            ),
            (
                list(PARALLEL_DECOMPOSITION_IDS)
                + list(ORDERING_IDS)
                + list(TILE_FAMILY_IDS)
                + list(CONVERSION_HYPOTHESIS_IDS)
            ),
            "performance",
            ["k_mapping_ms", "mem_hbm_gbps", "k_occupancy_achieved"],
            (
                "Claiming layout optimality before CUDA analysis, or "
                "selecting tiles without a decomposition."
            ),
            ["oq_sku_limits"],
        ),
        _experiment(
            "exp_mapping_micro",
            "CUDA mapping microbenchmarks",
            ["exp_layout_decomp", "exp_fusion_hoist", "exp_sku_fill"],
            ["TASK-17", "TASK-19"],
            "cuda_mappings",
            (
                "Eighteen ownership and reduction mappings differ on kernel "
                "and memory component metrics; none may be declared winner "
                "from this experiment alone."
            ),
            list(MAPPING_IDS),
            "performance",
            list(KERNEL_METRIC_IDS) + list(MEMORY_METRIC_IDS),
            (
                "mapping_winner_selected from microbenchmarks, or ranking "
                "graph envelopes as leaf kernel time."
            ),
            ["oq_hardware_protocol"],
        ),
        _experiment(
            "exp_mapping_e2e",
            "CUDA mapping end-to-end",
            ["exp_mapping_micro", "exp_control_keep_source"],
            ["TASK-17", "TASK-19", "TASK-18"],
            "experimental_validation",
            (
                "Identity-matched end-to-end latency plus decode-only and "
                "prefill metrics, with complete-map NLL as a quality guard, "
                "can fail mappings; tok/s is not the quality axis."
            ),
            list(MAPPING_IDS),
            "performance",
            [
                "e2e_latency_ms",
                "e2e_output_toks",
                "dec_toks",
                "pre_toks",
                "tf_nll_complete",
            ],
            (
                "Mixed (T+N)/t as decode-only, skipping e2e, or using tok/s "
                "as Pareto Y."
            ),
            ["oq_hardware_protocol", "oq_eval_corpora"],
        ),
    ]


def instantiate_clean_sheet_architecture(config: dict) -> dict:
    """Compute the TASK-20 clean-sheet summary from ``config``.

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

    experiments = _locked_experiments()
    experiment_ids = [row["id"] for row in experiments]
    task_open_question_closed = {
        task_id: task_id in TASKS_OPEN_QUESTION_CLOSED
        for task_id in SYNTHESIZED_TASK_IDS
    }
    n_tasks_open_question_closed = sum(
        1 for closed in task_open_question_closed.values() if closed
    )
    n_catalogued_alternatives = sum(ALTERNATIVE_FAMILY_COUNTS)

    summary = {
        "authority": AUTHORITY,
        "deliverable_architecture": DELIVERABLE_ARCHITECTURE,
        "deliverable_backlog": DELIVERABLE_BACKLOG,
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
        "n_candidate_recipes": len(CANDIDATE_RECIPE_IDS),
        "candidate_recipe_ids": list(CANDIDATE_RECIPE_IDS),
        "n_policy_families": len(POLICY_FAMILIES),
        "policy_families": list(POLICY_FAMILIES),
        "n_compiler_stages": len(COMPILER_STAGE_IDS),
        "compiler_stage_ids": list(COMPILER_STAGE_IDS),
        "n_compiler_profiles": len(COMPILER_PROFILE_IDS),
        "compiler_profile_ids": list(COMPILER_PROFILE_IDS),
        "control_profile_id": CONTROL_PROFILE_ID,
        "n_artifact_approaches": len(ARTIFACT_APPROACH_IDS),
        "artifact_approach_ids": list(ARTIFACT_APPROACH_IDS),
        "n_consumer_sequences": len(CONSUMER_SEQUENCE_IDS),
        "consumer_sequence_ids": list(CONSUMER_SEQUENCE_IDS),
        "integrity_algorithm_candidates": list(INTEGRITY_ALGORITHM_CANDIDATES),
        "n_fusion_hypotheses": len(FUSION_HYPOTHESIS_IDS),
        "fusion_hypothesis_ids": list(FUSION_HYPOTHESIS_IDS),
        "n_hoist_hypotheses": len(HOIST_HYPOTHESIS_IDS),
        "hoist_hypothesis_ids": list(HOIST_HYPOTHESIS_IDS),
        "n_representation_hypotheses": len(REPRESENTATION_HYPOTHESIS_IDS),
        "representation_hypothesis_ids": list(REPRESENTATION_HYPOTHESIS_IDS),
        "n_mode_hypotheses": len(MODE_HYPOTHESIS_IDS),
        "mode_hypothesis_ids": list(MODE_HYPOTHESIS_IDS),
        "n_orderings": len(ORDERING_IDS),
        "ordering_ids": list(ORDERING_IDS),
        "n_tile_families": len(TILE_FAMILY_IDS),
        "tile_family_ids": list(TILE_FAMILY_IDS),
        "n_parallel_decompositions": len(PARALLEL_DECOMPOSITION_IDS),
        "parallel_decomposition_ids": list(PARALLEL_DECOMPOSITION_IDS),
        "n_conversion_hypotheses": len(CONVERSION_HYPOTHESIS_IDS),
        "conversion_hypothesis_ids": list(CONVERSION_HYPOTHESIS_IDS),
        "n_mappings": len(MAPPING_IDS),
        "mapping_ids": list(MAPPING_IDS),
        "n_sku_unknown_symbols": len(SKU_UNKNOWN_SYMBOLS),
        "sku_unknown_symbols": list(SKU_UNKNOWN_SYMBOLS),
        "n_sensitive_ops": len(SENSITIVE_OPS),
        "sensitive_ops": list(SENSITIVE_OPS),
        "quality_high_ids": list(QUALITY_HIGH_IDS),
        "bottleneck_labels": list(BOTTLENECK_LABELS),
        "evaluation_criterion_ids": list(EVALUATION_CRITERION_IDS),
        "pareto_axis_ids": list(PARETO_AXIS_IDS),
        "corpus_class_ids": list(CORPUS_CLASS_IDS),
        "reconstruction_diagnostic_ids": list(RECONSTRUCTION_DIAGNOSTIC_IDS),
        "teacher_forced_metric_ids": list(TEACHER_FORCED_METRIC_IDS),
        "decode_metric_ids": list(DECODE_METRIC_IDS),
        "prefill_metric_ids": list(PREFILL_METRIC_IDS),
        "kernel_metric_ids": list(KERNEL_METRIC_IDS),
        "memory_metric_ids": list(MEMORY_METRIC_IDS),
        "e2e_metric_ids": list(E2E_METRIC_IDS),
        "comparison_dimension_ids": list(COMPARISON_DIMENSION_IDS),
        "n_synthesized_tasks": len(SYNTHESIZED_TASK_IDS),
        "synthesized_task_ids": list(SYNTHESIZED_TASK_IDS),
        "n_satellite_tasks": len(SATELLITE_TASK_IDS),
        "satellite_task_ids": list(SATELLITE_TASK_IDS),
        "n_chain_steps": len(CHAIN_IDS),
        "chain_ids": list(CHAIN_IDS),
        "chain_source_task_groups": {
            chain_id: list(task_ids)
            for chain_id, task_ids in CHAIN_SOURCE_TASK_GROUPS.items()
        },
        "n_evidence_classes": len(EVIDENCE_CLASS_IDS),
        "evidence_class_ids": list(EVIDENCE_CLASS_IDS),
        "task_evidence_classes": {
            task_id: list(TASK_EVIDENCE_CLASSES[task_id])
            for task_id in SYNTHESIZED_TASK_IDS
        },
        "task_open_question_closed": task_open_question_closed,
        "n_tasks_open_question_closed": n_tasks_open_question_closed,
        "n_tasks_open_question_open": (
            len(SYNTHESIZED_TASK_IDS) - n_tasks_open_question_closed
        ),
        "n_alternative_families": len(ALTERNATIVE_FAMILY_IDS),
        "alternative_family_ids": list(ALTERNATIVE_FAMILY_IDS),
        "alternative_family_counts": list(ALTERNATIVE_FAMILY_COUNTS),
        "n_catalogued_alternatives": n_catalogued_alternatives,
        "n_alternatives_selected": 0,
        "alternative_usefulness_label": "HYPOTHESIS",
        "n_remaining_open_questions": len(REMAINING_OPEN_QUESTION_IDS),
        "remaining_open_question_ids": list(REMAINING_OPEN_QUESTION_IDS),
        "n_remaining_open_questions_closed": 0,
        "n_experiment_fields": len(EXPERIMENT_FIELD_IDS),
        "experiment_field_ids": list(EXPERIMENT_FIELD_IDS),
        "n_experiment_eval_families": len(EXPERIMENT_EVAL_FAMILY_IDS),
        "experiment_eval_family_ids": list(EXPERIMENT_EVAL_FAMILY_IDS),
        "experiment_status_unrun": EXPERIMENT_STATUS_UNRUN,
        "n_experiments": len(experiments),
        "experiment_ids": experiment_ids,
        "experiments": experiments,
        "experiment_selected": {row_id: False for row_id in experiment_ids},
        "n_experiments_run": 0,
        "n_empty_metric_experiments": len(EMPTY_METRIC_EXPERIMENT_IDS),
        "n_empty_alternative_experiments": len(EMPTY_ALTERNATIVE_EXPERIMENT_IDS),
        "n_diagrams": 1,
        "diagram_ids": list(DIAGRAM_IDS),
        "n_architecture_headings": len(ARCHITECTURE_HEADINGS),
        "architecture_headings": list(ARCHITECTURE_HEADINGS),
        "n_backlog_headings": len(BACKLOG_HEADINGS),
        "backlog_headings": list(BACKLOG_HEADINGS),
        "n_headings": len(ARCHITECTURE_HEADINGS) + len(BACKLOG_HEADINGS),
        "chain_follows_plan_path": True,
        "chain_is_not_ledger_order": True,
        "experiment_order_is_topological": True,
        "ledger_open_question_alternatives_catalogued": True,
        "frozen_for_comparative_review": False,
        "pareto_frontier_selected": False,
        "compiler_profile_selected": False,
        "artifact_boundary_selected": False,
        "ideal_byte_sequence_selected": False,
        "integrity_algorithm_selected": False,
        "fusion_winner_selected": False,
        "decode_prefill_distinct_views_selected": False,
        "layout_winner_selected": False,
        "ordering_selected": False,
        "tile_size_selected": False,
        "mapping_winner_selected": False,
        "winner_selected_without_measurements": False,
        "hardware_selected": False,
        "prompt_matrix_selected": False,
        "reproducibility_protocol_selected": False,
        "calibration_corpus_selected": False,
        "eval_corpus_selected": False,
        "acceptance_frontier_selected": False,
        "sku_table_filled": False,
        "hypothesis_survival_selected": False,
        "n_mappings_selected": 0,
        "n_fusion_hypotheses_selected": 0,
        "keep_source_is_not_quality_winner": True,
        "control_profile_is_keep_source": True,
        "reconstruction_is_not_quality": True,
        "toks_is_not_quality_axis": True,
        "microbenchmark_cannot_pass_mapping": True,
        "e2e_required_alongside_microbenchmarks": True,
        "decode_prefill_share_graph": True,
        "decode_prefill_share_artifact": True,
        "decode_prefill_share_metric_identity": False,
        "experiments_run": False,
        "nll_measured_here": False,
        "toks_measured_here": False,
        "benchmarks_run": False,
        "payloads_restreamed": False,
        "gguf_payload_inspected": False,
        "gguf_is_not_design_authority": True,
        "gguf_is_pareto_reference": True,
        "gguf_is_not_a_requirement": True,
        "quartz_inspected": False,
        "llama_inspected": False,
        "llama_is_not_design_authority": True,
        "device_query_run": False,
        "nsight_run": False,
        "vision_eval_deferred": True,
        "safetensors_is_source_not_runtime": True,
        "canonical_sentence_logical": CANONICAL_SENTENCE_LOGICAL,
        "canonical_sentence_synthesis": CANONICAL_SENTENCE_SYNTHESIS,
        "canonical_sentence_alternatives": CANONICAL_SENTENCE_ALTERNATIVES,
        "canonical_sentence_unrun": CANONICAL_SENTENCE_UNRUN,
        "canonical_sentence_backlog": CANONICAL_SENTENCE_BACKLOG,
        "canonical_sentence_depends": CANONICAL_SENTENCE_DEPENDS,
        "canonical_sentence_eval_split": CANONICAL_SENTENCE_EVAL_SPLIT,
        "synthesis_question_sentence": SYNTHESIS_QUESTION_SENTENCE,
    }
    assert list(summary) == list(SCHEMA_KEYS), "summary key order drifted"
    return summary


def assert_summary(summary: dict) -> None:
    """Run locked internal asserts on a live clean-sheet summary.

    Args:
        summary: Object from :func:`instantiate_clean_sheet_architecture`.

    Raises:
        AssertionError: If a locked identity fails.
    """

    assert summary["n_linear_layers"] == 48
    assert summary["n_full_layers"] == 16
    assert summary["n_mtp_blocks"] == 1
    assert summary["full_attention_indices"] == list(LOCKED_FULL_ATTENTION_INDICES)
    assert summary["dtype"] == "bfloat16"
    assert summary["mamba_ssm_dtype"] == "float32"
    assert summary["model_T_max"] == 262144
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
    assert summary["mac_decode_complete_T1"] == 27433447424
    assert summary["mac_decode_complete_T4096"] == MAC_DECODE_COMPLETE_T4096
    assert summary["mac_prefill_complete_T4096"] == MAC_PREFILL_COMPLETE_T4096
    assert summary["example_T_values"] == list(EXAMPLE_T_VALUES)
    assert summary["N_w"] == 32
    assert summary["N_bank"] == 32
    assert summary["n_node_types"] == 6
    assert summary["n_node_instances_complete"] == 135
    assert summary["n_stage_kinds"] == 9
    assert summary["n_candidate_recipes"] == 22
    assert summary["n_policy_families"] == 16
    assert summary["n_mappings"] == 18
    assert summary["n_fusion_hypotheses"] == 22
    assert summary["n_synthesized_tasks"] == 19
    assert summary["n_chain_steps"] == 12
    assert summary["n_evidence_classes"] == 5
    assert list(summary["chain_source_task_groups"]) == list(summary["chain_ids"])
    assert summary["chain_source_task_groups"]["experimental_validation"] == [
        "TASK-18",
        "TASK-19",
    ]
    for task_ids in summary["chain_source_task_groups"].values():
        assert "TASK-20" not in task_ids
        for task_id in task_ids:
            assert task_id in summary["synthesized_task_ids"]
    assert summary["n_alternative_families"] == 14
    assert summary["n_catalogued_alternatives"] == 125
    assert summary["n_alternatives_selected"] == 0
    assert sum(summary["alternative_family_counts"]) == 125
    assert summary["n_remaining_open_questions"] == 12
    assert summary["n_remaining_open_questions_closed"] == 0
    assert summary["n_experiment_fields"] == 12
    assert summary["n_experiments"] == 16
    assert summary["n_experiments_run"] == 0
    assert summary["n_tasks_open_question_closed"] == 9
    assert summary["n_tasks_open_question_open"] == 10
    assert summary["n_architecture_headings"] == 8
    assert summary["n_backlog_headings"] == 6
    assert summary["n_headings"] == 14
    assert summary["n_diagrams"] == 1
    assert summary["node_type_ids"] == list(NODE_TYPE_IDS)
    assert summary["mapping_ids"] == list(MAPPING_IDS)
    assert summary["candidate_recipe_ids"] == list(CANDIDATE_RECIPE_IDS)
    assert summary["fusion_hypothesis_ids"] == list(FUSION_HYPOTHESIS_IDS)
    assert summary["sku_unknown_symbols"] == list(SKU_UNKNOWN_SYMBOLS)
    assert summary["stage_kind_ids"] == list(STAGE_KIND_IDS)
    experiments = summary["experiments"]
    assert len(experiments) == 16
    experiment_ids = summary["experiment_ids"]
    assert len(experiment_ids) == len(set(experiment_ids))
    index_by_id = {row_id: index for index, row_id in enumerate(experiment_ids)}
    source_task_re = re.compile(r"^TASK-(0[1-9]|1[0-9])$")
    for row in experiments:
        assert list(row) == list(EXPERIMENT_FIELD_IDS)
        assert row["id"] in index_by_id
        assert row["status"] == EXPERIMENT_STATUS_UNRUN
        assert row["chain_step"] in summary["chain_ids"]
        assert row["eval_family"] in summary["experiment_eval_family_ids"]
        for dep in row["depends_on"]:
            assert dep in index_by_id, f"unknown depends_on {dep!r}"
            assert index_by_id[dep] < index_by_id[row["id"]], (
                f"{row['id']} depends on later {dep}"
            )
        for task_id in row["source_tasks"]:
            assert source_task_re.match(task_id), f"bad source_task {task_id!r}"
        if row["id"] in EMPTY_METRIC_EXPERIMENT_IDS:
            assert row["metric_ids"] == []
        else:
            assert row["metric_ids"]
        if row["id"] in EMPTY_ALTERNATIVE_EXPERIMENT_IDS:
            assert row["alternatives"] == []
        else:
            assert row["alternatives"]
        if row["id"] in EMPTY_DEPENDS_EXPERIMENT_IDS:
            assert row["depends_on"] == []
        for unknown_id in row["blocking_unknowns"]:
            assert unknown_id in summary["remaining_open_question_ids"]
    empty_metric = [
        row["id"] for row in experiments if row["metric_ids"] == []
    ]
    empty_alts = [
        row["id"] for row in experiments if row["alternatives"] == []
    ]
    assert set(empty_metric) == EMPTY_METRIC_EXPERIMENT_IDS
    assert set(empty_alts) == EMPTY_ALTERNATIVE_EXPERIMENT_IDS
    assert summary["ledger_open_question_alternatives_catalogued"] is True
    assert summary["frozen_for_comparative_review"] is False
    assert summary["mapping_winner_selected"] is False
    assert summary["pareto_frontier_selected"] is False
    assert summary["fusion_winner_selected"] is False
    assert summary["experiments_run"] is False
    assert summary["nll_measured_here"] is False
    assert summary["toks_measured_here"] is False
    assert summary["benchmarks_run"] is False
    assert summary["sku_table_filled"] is False
    assert summary["toks_is_not_quality_axis"] is True
    assert summary["reconstruction_is_not_quality"] is True
    assert summary["microbenchmark_cannot_pass_mapping"] is True
    assert summary["gguf_payload_inspected"] is False
    assert summary["quartz_inspected"] is False
    assert summary["llama_inspected"] is False
    assert summary["device_query_run"] is False
    assert summary["nsight_run"] is False
    assert summary["decode_prefill_share_metric_identity"] is False
    assert summary["vision_eval_deferred"] is True
    assert summary["n_embed_instances"] == 2
    assert summary["n_gated_attn_instances"] == 17
    assert summary["n_gated_delta_net_instances"] == 48
    assert summary["n_mlp_instances"] == 65
    assert summary["n_lm_head_instances"] == 2
    assert summary["n_mtp_mix_instances"] == 1


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
    """Compare documented JSON against a live clean-sheet object.

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


def _canonical_spans(text: str, sentences: tuple[str, ...]) -> list[tuple[int, int]]:
    """Return character spans covering canonical sentences in ``text``.

    Args:
        text: Full markdown document.
        sentences: Canonical strings that may host forbidden substrings.

    Returns:
        Inclusive-start exclusive-end spans.
    """

    spans: list[tuple[int, int]] = []
    for phrase in sentences:
        start = 0
        while True:
            index = text.find(phrase, start)
            if index < 0:
                break
            spans.append((index, index + len(phrase)))
            start = index + 1
    return spans


def _inside_span(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    """Return whether ``[start, end)`` lies inside any span.

    Args:
        start: Match start index.
        end: Match end index.
        spans: Allowed spans.

    Returns:
        True when the match is fully contained in an allowed span.
    """

    return any(
        span_start <= start and end <= span_end for span_start, span_end in spans
    )


def _line_contains_wrapper(text: str, index: int) -> bool:
    """Return whether the line containing ``index`` has an allowed wrapper.

    Args:
        text: Full markdown document.
        index: Character index of a forbidden-phrase match.

    Returns:
        True when the same line contains an allowed wrapper phrase.
    """

    line_start = text.rfind("\n", 0, index) + 1
    line_end = text.find("\n", index)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end]
    return any(wrapper in line for wrapper in ALLOWED_WRAPPER_PHRASES)


def _collect_forbidden_phrase_diffs(
    text: str,
    fence_match: re.Match[str] | None,
    canonical_sentences: tuple[str, ...],
) -> list[str]:
    """Collect forbidden-winner-phrase mismatches.

    Args:
        text: Full markdown document.
        fence_match: First JSON fence match, if parsed.
        canonical_sentences: Document canonical sentences.

    Returns:
        Difference strings for disallowed matches.
    """

    differences: list[str] = []
    spans = _canonical_spans(text, canonical_sentences)
    json_span: tuple[int, int] | None = None
    if fence_match is not None:
        json_span = (fence_match.start(), fence_match.end())
    for phrase in FORBIDDEN_WINNER_PHRASES:
        start = 0
        while True:
            index = text.find(phrase, start)
            if index < 0:
                break
            end = index + len(phrase)
            allowed = _inside_span(index, end, spans) or _line_contains_wrapper(
                text, index
            )
            if json_span is not None and json_span[0] <= index and end <= json_span[1]:
                allowed = True
            if not allowed:
                line = text.count("\n", 0, index) + 1
                differences.append(
                    f"line {line}: forbidden winner phrase {phrase!r}"
                )
            start = index + 1
    return differences


def _check_common_markdown(
    live: dict,
    design_path: Path,
    required_headings: tuple[str, ...],
    label: str,
) -> tuple[str, list[str], re.Match[str] | None]:
    """Run heading, JSON, placeholder, UNKNOWN, and integer checks.

    Args:
        live: Live summary object.
        design_path: Markdown path.
        required_headings: Locked ``##`` headings in order.
        label: Short file label for error strings.

    Returns:
        Tuple of document text, difference strings, and JSON fence match.

    Raises:
        OSError: If the file cannot be read.
    """

    text = design_path.read_text(encoding="utf-8")
    differences: list[str] = []

    headings = HEADING_RE.findall(text)
    if headings != list(required_headings):
        differences.append(
            f"{label} heading mismatch:\n"
            f"  documented: {headings}\n"
            f"  required:   {list(required_headings)}"
        )

    fence_match: re.Match[str] | None = None
    try:
        documented, fence_match = first_json_fence(text)
    except (AssertionError, json.JSONDecodeError) as exc:
        differences.append(f"{label} json fence: {exc}")
        documented = None

    if documented is not None:
        differences.extend(diff_summary(live, documented))

    for match in FORBIDDEN_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        differences.append(
            f"{label} line {line}: forbidden token {match.group(0)!r}"
        )

    try:
        vis_start, vis_end = _deferred_vision_span(text)
        if not UNKNOWN_RE.search(text, vis_start, vis_end):
            differences.append(
                f"{label}: word UNKNOWN missing in Deferred vision"
            )
    except AssertionError as exc:
        differences.append(f"{label}: {exc}")

    for number in LOCKED_DOCUMENT_INTEGERS:
        token = str(number)
        if token not in text:
            differences.append(
                f"{label}: locked number {token} missing as decimal substring"
            )

    return text, differences, fence_match


def check_architecture(live: dict, design_path: Path) -> None:
    """Check the architecture markdown file against a live summary.

    Args:
        live: Clean-sheet summary object from sitting config.
        design_path: Path to ``clean-sheet-architecture.md``.

    Raises:
        CleanSheetMismatch: On heading, JSON, diagram, or token mismatches.
        OSError: If the file cannot be read.
    """

    text, differences, fence_match = _check_common_markdown(
        live, design_path, ARCHITECTURE_HEADINGS, "architecture"
    )

    mermaid_fences = list(MERMAID_FENCE_RE.finditer(text))
    if len(mermaid_fences) != 1:
        differences.append(
            f"architecture mermaid fence count {len(mermaid_fences)} != 1"
        )
    else:
        match = mermaid_fences[0]
        body = match.group(1)
        stripped = body.lstrip()
        if not (
            stripped.startswith("flowchart TB")
            or stripped.startswith("flowchart LR")
        ):
            differences.append(
                "architecture mermaid fence body does not start with "
                "flowchart TB or flowchart LR"
            )
        if "flowchart" not in body:
            differences.append("architecture mermaid fence does not contain flowchart")
        missing_ids = [
            node_id for node_id in DIAGRAM_MERMAID_IDS if node_id not in body
        ]
        if missing_ids:
            differences.append(
                f"architecture diagram missing node ids {missing_ids}"
            )
        heading = re.search(
            r"^## Architecture diagram and alternatives\s*$",
            text,
            flags=re.MULTILINE,
        )
        next_heading = re.search(
            r"^## Remaining unknowns\s*$", text, flags=re.MULTILINE
        )
        if heading is None or next_heading is None:
            differences.append(
                "missing Architecture diagram and alternatives or Remaining unknowns"
            )
        elif not (heading.end() <= match.start() < next_heading.start()):
            differences.append(
                "mermaid fence is not under ## Architecture diagram and alternatives"
            )

    architecture_sentences = (
        CANONICAL_SENTENCE_LOGICAL,
        CANONICAL_SENTENCE_SYNTHESIS,
        CANONICAL_SENTENCE_ALTERNATIVES,
        CANONICAL_SENTENCE_UNRUN,
        SYNTHESIS_QUESTION_SENTENCE,
    )
    for label, sentence in (
        ("canonical_sentence_logical", CANONICAL_SENTENCE_LOGICAL),
        ("canonical_sentence_synthesis", CANONICAL_SENTENCE_SYNTHESIS),
        ("canonical_sentence_alternatives", CANONICAL_SENTENCE_ALTERNATIVES),
        ("canonical_sentence_unrun", CANONICAL_SENTENCE_UNRUN),
        ("synthesis_question_sentence", SYNTHESIS_QUESTION_SENTENCE),
    ):
        if sentence not in text:
            differences.append(f"architecture {label} not present verbatim")

    convention = re.search(
        r"^## Synthesis convention\s*$", text, flags=re.MULTILINE
    )
    chain_heading = re.search(
        r"^## Architecture chain and evidence classes\s*$",
        text,
        flags=re.MULTILINE,
    )
    if convention is None or chain_heading is None:
        differences.append(
            "missing Synthesis convention or Architecture chain and evidence classes"
        )
    else:
        section = text[convention.end() : chain_heading.start()]
        for label, sentence in (
            ("canonical_sentence_logical", CANONICAL_SENTENCE_LOGICAL),
            ("canonical_sentence_synthesis", CANONICAL_SENTENCE_SYNTHESIS),
            ("canonical_sentence_alternatives", CANONICAL_SENTENCE_ALTERNATIVES),
            ("canonical_sentence_unrun", CANONICAL_SENTENCE_UNRUN),
            ("synthesis_question_sentence", SYNTHESIS_QUESTION_SENTENCE),
        ):
            if sentence not in section:
                differences.append(
                    f"architecture {label} missing under Synthesis convention"
                )

    for label, ids in (
        ("chain", CHAIN_IDS),
        ("evidence class", EVIDENCE_CLASS_IDS),
        ("synthesized task", SYNTHESIZED_TASK_IDS),
        ("alternative family", ALTERNATIVE_FAMILY_IDS),
        ("remaining open question", REMAINING_OPEN_QUESTION_IDS),
        ("experiment", tuple(live["experiment_ids"])),
        ("node type", NODE_TYPE_IDS),
        ("stage kind", STAGE_KIND_IDS),
        ("candidate recipe", CANDIDATE_RECIPE_IDS),
        ("mapping", MAPPING_IDS),
        ("fusion hypothesis", FUSION_HYPOTHESIS_IDS),
        ("sku unknown symbol", SKU_UNKNOWN_SYMBOLS),
    ):
        for item_id in ids:
            if item_id not in text:
                differences.append(
                    f"architecture {label} {item_id!r} missing as substring"
                )

    if "HYPOTHESIS" not in text:
        differences.append("architecture word HYPOTHESIS missing")
    if "not measurements" not in text:
        differences.append("architecture phrase 'not measurements' missing")

    differences.extend(
        _collect_forbidden_phrase_diffs(text, fence_match, architecture_sentences)
    )

    if differences:
        raise CleanSheetMismatch(differences)


def check_experiment_backlog(live: dict, design_path: Path) -> None:
    """Check the experiment-backlog markdown file against a live summary.

    Args:
        live: Clean-sheet summary object from sitting config.
        design_path: Path to ``experiment-backlog.md``.

    Raises:
        CleanSheetMismatch: On heading, JSON, or token mismatches.
        OSError: If the file cannot be read.
    """

    text, differences, fence_match = _check_common_markdown(
        live, design_path, BACKLOG_HEADINGS, "backlog"
    )

    mermaid_fences = list(MERMAID_FENCE_RE.finditer(text))
    if mermaid_fences:
        differences.append(
            f"backlog mermaid fence count {len(mermaid_fences)} != 0"
        )

    backlog_sentences = (
        CANONICAL_SENTENCE_LOGICAL,
        CANONICAL_SENTENCE_BACKLOG,
        CANONICAL_SENTENCE_DEPENDS,
        CANONICAL_SENTENCE_EVAL_SPLIT,
        SYNTHESIS_QUESTION_SENTENCE,
    )
    for label, sentence in (
        ("canonical_sentence_logical", CANONICAL_SENTENCE_LOGICAL),
        ("canonical_sentence_backlog", CANONICAL_SENTENCE_BACKLOG),
        ("canonical_sentence_depends", CANONICAL_SENTENCE_DEPENDS),
        ("canonical_sentence_eval_split", CANONICAL_SENTENCE_EVAL_SPLIT),
        ("synthesis_question_sentence", SYNTHESIS_QUESTION_SENTENCE),
    ):
        if sentence not in text:
            differences.append(f"backlog {label} not present verbatim")

    convention = re.search(
        r"^## Backlog convention\s*$", text, flags=re.MULTILINE
    )
    fields_heading = re.search(
        r"^## Requested fields\s*$", text, flags=re.MULTILINE
    )
    if convention is None or fields_heading is None:
        differences.append("missing Backlog convention or Requested fields")
    else:
        section = text[convention.end() : fields_heading.start()]
        for label, sentence in (
            ("canonical_sentence_logical", CANONICAL_SENTENCE_LOGICAL),
            ("canonical_sentence_backlog", CANONICAL_SENTENCE_BACKLOG),
            ("canonical_sentence_depends", CANONICAL_SENTENCE_DEPENDS),
            ("canonical_sentence_eval_split", CANONICAL_SENTENCE_EVAL_SPLIT),
            ("synthesis_question_sentence", SYNTHESIS_QUESTION_SENTENCE),
        ):
            if sentence not in section:
                differences.append(
                    f"backlog {label} missing under Backlog convention"
                )

    for item_id in live["experiment_ids"]:
        if item_id not in text:
            differences.append(f"backlog experiment id {item_id!r} missing")
    for field_id in EXPERIMENT_FIELD_IDS:
        if field_id not in text:
            differences.append(f"backlog experiment field {field_id!r} missing")
    for row in live["experiments"]:
        title = row["title"]
        if title not in text:
            differences.append(f"backlog experiment title {title!r} missing")

    if "backlog" not in text:
        differences.append("backlog word 'backlog' missing")
    if "unrun" not in text:
        differences.append("backlog word 'unrun' missing")

    differences.extend(
        _collect_forbidden_phrase_diffs(text, fence_match, backlog_sentences)
    )

    if differences:
        raise CleanSheetMismatch(differences)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the clean-sheet-architecture checker.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Instantiate Qwen3.8-27B language+MTP clean-sheet architecture "
            "summary from text_config and check "
            "docs/architecture/clean-sheet-architecture.md and "
            "docs/architecture/experiment-backlog.md."
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
        help="Write clean-sheet summary JSON to stdout.",
    )
    parser.add_argument(
        "--architecture",
        type=Path,
        default=None,
        help="Architecture markdown whose first json fence must match live summary.",
    )
    parser.add_argument(
        "--experiment-backlog",
        type=Path,
        default=None,
        help="Backlog markdown whose first json fence must match live summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the clean-sheet-architecture checker CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch, 2 missing
        config.
    """

    args = _parse_args(argv)
    architecture_path = args.architecture
    backlog_path = args.experiment_backlog
    print_json = args.json
    if architecture_path is None and backlog_path is None and not args.json:
        architecture_path = Path(DELIVERABLE_ARCHITECTURE)
        backlog_path = Path(DELIVERABLE_BACKLOG)
    try:
        config = load_config(args.config)
        summary = instantiate_clean_sheet_architecture(config)
        assert_summary(summary)
        if architecture_path is not None:
            if not architecture_path.is_file():
                print(
                    f"architecture file not found: {architecture_path}",
                    file=sys.stderr,
                )
                return 1
            check_architecture(summary, architecture_path)
        if backlog_path is not None:
            if not backlog_path.is_file():
                print(
                    f"experiment-backlog file not found: {backlog_path}",
                    file=sys.stderr,
                )
                return 1
            check_experiment_backlog(summary, backlog_path)
    except MissingConfig as exc:
        print(str(exc.path), file=sys.stderr)
        return 2
    except CleanSheetMismatch as exc:
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
