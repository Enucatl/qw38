#!/usr/bin/env python3
"""Check Qwen3.8-27B CUDA design-space contracts against sitting text_config.

Reads ``text_config`` from a Transformers ``config.json`` (no safetensor
payloads, no GPU query) and either prints the CUDA-design-space summary
object or checks that ``docs/architecture/cuda-design-space.md`` matches
it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

AUTHORITY = ".cache/authorities/qwen3.8-27b-transformers"
DEFAULT_CUDA_DESIGN_SPACE = Path("docs/architecture/cuda-design-space.md")
SKU_POLICY = "parameterized_unknown_until_measured_table"

CANONICAL_SENTENCE_LOGICAL = (
    "Logical values in this document are graph nodes. They do not imply "
    "physical allocation, materialization, buffer reuse, or kernel fusion."
)
CANONICAL_SENTENCE_MAPPINGS = (
    "CUDA mappings in this document are ownership and reduction "
    "alternatives per semantic node, not selected kernels and not "
    "measured winners."
)
CANONICAL_SENTENCE_SYMBOLIC = (
    "Occupancy, intensity, and roofline figures in this document "
    "instantiate TASK-16 algebra with symbolic SKU limits; they are not "
    "sitting-device measurements."
)
CANONICAL_SENTENCE_OPEN_QUESTION = (
    "Which mappings win on target hardware and profiles remains open "
    "until TASK-19 measurements."
)
CANONICAL_SENTENCE_WINNER = "This document selects no CUDA mapping winner."
CANONICAL_SENTENCE_LAYOUT = (
    "Instantiating a TASK-15 parallel decomposition as a CUDA ownership "
    "axis does not select a layout and does not justify an ordering or tile."
)

NODE_TYPE_IDS: tuple[str, ...] = (
    "embed",
    "gated_attn",
    "gated_delta_net",
    "mlp",
    "lm_head",
    "mtp_mix",
)

OWNERSHIP_IDS: tuple[str, ...] = (
    "thread",
    "warp",
    "cta",
    "grid",
)

REDUCTION_IDS: tuple[str, ...] = (
    "red_none",
    "red_warp",
    "red_cta",
    "red_splitk_cta",
    "red_grid",
)

ESTIMATE_DIMENSION_IDS: tuple[str, ...] = (
    "work",
    "storage",
    "access",
    "synchronization",
    "occupancy",
    "mode_suitability",
)

SYNC_CLASS_IDS: tuple[str, ...] = (
    "sync_stream_event",
    "sync_warp",
    "sync_cta",
    "sync_grid",
    "sync_none",
)

PIPELINE_IDS: tuple[str, ...] = (
    "fma",
    "ffma",
    "load_store",
    "tensor_core_mma",
    "async_gmem_to_smem",
    "tma",
)

MODE_FIT_IDS: tuple[str, ...] = (
    "decode_primary",
    "prefill_primary",
    "both",
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

MAPPING_NODE_TYPES: tuple[str, ...] = (
    "embed",
    "embed",
    "embed",
    "gated_attn",
    "gated_attn",
    "gated_attn",
    "gated_delta_net",
    "gated_delta_net",
    "gated_delta_net",
    "mlp",
    "mlp",
    "mlp",
    "lm_head",
    "lm_head",
    "lm_head",
    "mtp_mix",
    "mtp_mix",
    "mtp_mix",
)

MAPPING_OWNERSHIP_IDS: tuple[str, ...] = (
    "thread",
    "warp",
    "cta",
    "cta",
    "warp",
    "cta",
    "cta",
    "warp",
    "cta",
    "cta",
    "cta",
    "grid",
    "cta",
    "cta",
    "warp",
    "cta",
    "cta",
    "grid",
)

MAPPING_REDUCTION_IDS: tuple[str, ...] = (
    "red_none",
    "red_none",
    "red_none",
    "red_none",
    "red_warp",
    "red_splitk_cta",
    "red_none",
    "red_none",
    "red_cta",
    "red_none",
    "red_splitk_cta",
    "red_none",
    "red_none",
    "red_splitk_cta",
    "red_warp",
    "red_none",
    "red_none",
    "red_none",
)

MAPPING_SYNC_CLASS_IDS: tuple[str, ...] = (
    "sync_none",
    "sync_warp",
    "sync_cta",
    "sync_cta",
    "sync_warp",
    "sync_cta",
    "sync_cta",
    "sync_warp",
    "sync_cta",
    "sync_cta",
    "sync_cta",
    "sync_none",
    "sync_cta",
    "sync_cta",
    "sync_warp",
    "sync_cta",
    "sync_cta",
    "sync_stream_event",
)

MAPPING_PRIMARY_PIPELINE_IDS: tuple[str, ...] = (
    "load_store",
    "load_store",
    "load_store",
    "fma",
    "fma",
    "fma",
    "fma",
    "fma",
    "fma",
    "fma",
    "fma",
    "tensor_core_mma",
    "fma",
    "fma",
    "fma",
    "fma",
    "fma",
    "load_store",
)

MAPPING_SECONDARY_PIPELINE_IDS: dict[str, str | None] = {
    "map_embed_thread_element": None,
    "map_embed_warp_row": None,
    "map_embed_cta_vector": "async_gmem_to_smem",
    "map_attn_cta_head": "tensor_core_mma",
    "map_attn_warp_t": None,
    "map_attn_cta_splitk": None,
    "map_gdn_cta_head": None,
    "map_gdn_warp_recurrent": None,
    "map_gdn_cta_chunk": None,
    "map_mlp_cta_dout": "tensor_core_mma",
    "map_mlp_cta_splitk": None,
    "map_mlp_grid_T": None,
    "map_lm_cta_vocab": "tensor_core_mma",
    "map_lm_cta_splitk": None,
    "map_lm_warp_gemv": None,
    "map_mtp_cta_fc": "tensor_core_mma",
    "map_mtp_cta_fused": None,
    "map_mtp_split_norm_gemm": None,
}

MAPPING_MODE_FIT_IDS: tuple[str, ...] = (
    "both",
    "both",
    "both",
    "both",
    "decode_primary",
    "both",
    "both",
    "decode_primary",
    "prefill_primary",
    "both",
    "both",
    "prefill_primary",
    "both",
    "both",
    "decode_primary",
    "both",
    "both",
    "both",
)

MAPPING_WORK_MAC_IDS: tuple[str, ...] = (
    "mac_embed",
    "mac_embed",
    "mac_embed",
    "mac_full_proj_per_layer",
    "mac_attn_coeff_per_full_layer",
    "mac_attn_coeff_per_full_layer",
    "mac_lin_token_per_layer",
    "mac_gdn_per_layer",
    "mac_gdn_per_layer",
    "mac_mlp_per_layer",
    "mac_mlp_per_layer",
    "mac_mlp_per_layer",
    "mac_lm_head",
    "mac_lm_head",
    "mac_lm_head",
    "mac_mtp_fc",
    "mac_mtp_fc",
    "mac_mtp_fc",
)

MAPPING_LAYOUT_OBJECT_IDS: dict[str, tuple[str, ...]] = {
    "map_embed_thread_element": ("gather_row",),
    "map_embed_warp_row": ("gather_row",),
    "map_embed_cta_vector": ("gather_row",),
    "map_attn_cta_head": ("dense_gemm", "state_kv"),
    "map_attn_warp_t": ("dense_gemm", "state_kv"),
    "map_attn_cta_splitk": ("dense_gemm", "state_kv"),
    "map_gdn_cta_head": (
        "dense_gemm",
        "depthwise_conv",
        "vector_param",
        "state_c",
        "state_s",
    ),
    "map_gdn_warp_recurrent": (
        "dense_gemm",
        "depthwise_conv",
        "vector_param",
        "state_c",
        "state_s",
    ),
    "map_gdn_cta_chunk": (
        "dense_gemm",
        "depthwise_conv",
        "vector_param",
        "state_c",
        "state_s",
    ),
    "map_mlp_cta_dout": ("dense_gemm",),
    "map_mlp_cta_splitk": ("dense_gemm",),
    "map_mlp_grid_T": ("dense_gemm",),
    "map_lm_cta_vocab": ("dense_gemm",),
    "map_lm_cta_splitk": ("dense_gemm",),
    "map_lm_warp_gemv": ("dense_gemm",),
    "map_mtp_cta_fc": ("dense_gemm",),
    "map_mtp_cta_fused": ("dense_gemm",),
    "map_mtp_split_norm_gemm": ("dense_gemm",),
}

MAPPING_DECOMPOSITION_IDS: dict[str, tuple[str, ...]] = {
    "map_embed_thread_element": ("par_embed_row",),
    "map_embed_warp_row": ("par_embed_row",),
    "map_embed_cta_vector": ("par_embed_row",),
    "map_attn_cta_head": ("par_attn_head", "par_kv_head"),
    "map_attn_warp_t": ("par_attn_T",),
    "map_attn_cta_splitk": ("par_attn_T", "par_gemm_d_in"),
    "map_gdn_cta_head": ("par_gdn_head", "par_conv_channel"),
    "map_gdn_warp_recurrent": ("par_gdn_head", "par_conv_channel"),
    "map_gdn_cta_chunk": ("par_gdn_head", "par_conv_channel"),
    "map_mlp_cta_dout": ("par_gemm_d_out",),
    "map_mlp_cta_splitk": ("par_gemm_d_in",),
    "map_mlp_grid_T": ("par_gemm_T",),
    "map_lm_cta_vocab": ("par_gemm_d_out",),
    "map_lm_cta_splitk": ("par_gemm_d_in",),
    "map_lm_warp_gemv": ("par_gemm_d_out",),
    "map_mtp_cta_fc": ("par_gemm_d_out",),
    "map_mtp_cta_fused": ("par_gemm_d_out",),
    "map_mtp_split_norm_gemm": ("par_gemm_d_out",),
}

EVALUATION_CRITERION_IDS: tuple[str, ...] = (
    "crit_occupancy",
    "crit_latency_hiding",
    "crit_wave_quant",
    "crit_intensity_roofline",
    "crit_sync_class",
    "crit_pipeline_mix",
    "crit_fusion_delta",
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

DECOMPOSITION_MAPPING_IDS: dict[str, tuple[str, ...]] = {
    "par_gemm_d_out": (
        "map_mlp_cta_dout",
        "map_lm_cta_vocab",
        "map_lm_warp_gemv",
        "map_mtp_cta_fc",
        "map_mtp_cta_fused",
        "map_mtp_split_norm_gemm",
    ),
    "par_gemm_d_in": (
        "map_mlp_cta_splitk",
        "map_lm_cta_splitk",
        "map_attn_cta_splitk",
    ),
    "par_gemm_T": ("map_mlp_grid_T",),
    "par_attn_head": ("map_attn_cta_head",),
    "par_attn_T": ("map_attn_warp_t", "map_attn_cta_splitk"),
    "par_gdn_head": (
        "map_gdn_cta_head",
        "map_gdn_warp_recurrent",
        "map_gdn_cta_chunk",
    ),
    "par_conv_channel": (
        "map_gdn_cta_head",
        "map_gdn_warp_recurrent",
        "map_gdn_cta_chunk",
    ),
    "par_kv_head": ("map_attn_cta_head",),
    "par_embed_row": (
        "map_embed_thread_element",
        "map_embed_warp_row",
        "map_embed_cta_vector",
    ),
}

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

MAPPING_RISK_IDS: tuple[str, ...] = (
    "m_occ_fusion",
    "m_splitk_sync",
    "m_mma_sku",
    "m_async_absent",
    "m_cluster_absent",
    "m_decode_prefill_same_map",
    "m_gdn_serial",
    "m_lm_vocab_wave",
)

MAPPING_RISK_SEVERITIES: tuple[str, ...] = (
    "high",
    "medium",
    "medium",
    "low",
    "low",
    "medium",
    "medium",
    "medium",
)

MAPPING_HIGH_IDS: tuple[str, ...] = ("m_occ_fusion",)
MAPPING_MEDIUM_IDS: tuple[str, ...] = (
    "m_splitk_sync",
    "m_mma_sku",
    "m_decode_prefill_same_map",
    "m_gdn_serial",
    "m_lm_vocab_wave",
)
MAPPING_LOW_IDS: tuple[str, ...] = (
    "m_async_absent",
    "m_cluster_absent",
)

BOTTLENECK_LABELS: tuple[str, ...] = (
    "weight_memory",
    "vocab_memory",
    "state_memory",
    "kv_memory",
    "quadratic_attn",
    "compute",
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

CTA_T_CANDIDATES: tuple[int, ...] = (32, 64, 128, 256)
EXAMPLE_T: tuple[int, ...] = (1, 4096)
N_W = 32
N_BANK = 32
BYTES_BF16 = 2
BYTES_F32 = 4

DIAGRAM_REQUIRED_IDS: tuple[str, ...] = (
    "embed",
    "gated_attn",
    "gated_delta_net",
    "mlp",
    "lm_head",
    "mtp_mix",
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

FORMULA_TAGS: tuple[str, ...] = (
    "(F1)",
    "(F7)",
    "(F8)",
    "(F9)",
    "(F10)",
    "(F11)",
    "(F12)",
    "(F13)",
    "(F14)",
)

FORMULA_SUBSTRINGS: tuple[str, ...] = (
    r"O = \min(O_\text{reg}, O_\text{smem}, O_\text{threads}, O_\text{cta})",
    r"W_\text{cta} = \lceil T_\text{cta} / N_w \rceil",
    r"O = W_\text{active} / W_\max",
    r"W_\text{need} = N_\text{sched} \cdot L_\text{issue}",
    r"N_\text{waves} = \lceil N_\text{grid} / (N_\text{SM} \cdot B_\text{SM}) \rceil",
    r"R_f \ge \max(R_1, R_2)",
    r"C_f \ge \max(C_1, C_2)",
    r"I = F / B",
    r"\Pi \le \min(\Pi_\text{peak}, I \cdot \Beta)",
)

LOCKED_DOCUMENT_NUMBERS: tuple[int | float, ...] = (
    5120,
    17408,
    248320,
    10240,
    256,
    128,
    48,
    24,
    4,
    16,
    17,
    64,
    135,
    104857600,
    12288,
    118235136,
    40960,
    2359296,
    267386880,
    1271398400,
    52428800,
    4096,
    69632,
    61440,
    3145728,
    150994944,
    2542796800,
    0.75,
)

REQUIRED_HEADINGS: tuple[str, ...] = (
    "Authority",
    "Mapping convention",
    "Mapping vocabulary",
    "Per-node mapping alternatives",
    "Work, storage, and access estimates",
    "Synchronization, occupancy, and mode suitability",
    "Layout instantiations",
    "Evaluation instantiation",
    "Fusion versus occupancy",
    "Non-decisions",
    "Deferred vision",
    "Machine-checkable summary JSON",
)

FORBIDDEN_WINNER_PHRASES: tuple[str, ...] = (
    "winning mapping",
    "selected kernel",
    "best occupancy",
    "should use tensor cores",
    "recommend this mapping",
    "deviceQuery",
    "Quartz kernel",
    "llama.cpp kernel",
    "GGUF kernel",
    "FlashAttention",
    "cuBLAS",
    "CUTLASS",
    "selected winner",
    "MMA shape is",
    "achieved occupancy",
    "this mapping is required",
    "selected distinct views",
    "prefill requires a distinct view",
)

ALLOWED_EXCEPTION_PHRASES: tuple[str, ...] = (
    "not selected kernels",
    "not measured winners",
    "selects no CUDA mapping winner",
    "does not select a layout",
    "does not justify",
)

SUBSTRING_ID_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("node type", NODE_TYPE_IDS),
    ("mapping", MAPPING_IDS),
    ("ownership", OWNERSHIP_IDS),
    ("reduction", REDUCTION_IDS),
    ("estimate dimension", ESTIMATE_DIMENSION_IDS),
    ("sync class", SYNC_CLASS_IDS),
    ("pipeline", PIPELINE_IDS),
    ("mode fit", MODE_FIT_IDS),
    ("evaluation criterion", EVALUATION_CRITERION_IDS),
    ("parallel decomposition", PARALLEL_DECOMPOSITION_IDS),
    ("justification hypothesis", JUSTIFICATION_HYPOTHESIS_IDS),
    ("fusion hypothesis", FUSION_HYPOTHESIS_IDS),
    ("mapping risk", MAPPING_RISK_IDS),
    ("stage kind", STAGE_KIND_IDS),
    ("tile family", TILE_FAMILY_IDS),
    ("bottleneck label", BOTTLENECK_LABELS),
    ("sku unknown symbol", SKU_UNKNOWN_SYMBOLS),
)

JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)
MERMAID_FENCE_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
HEADING_RE = re.compile(r"^## (?!#)(.+)$", re.MULTILINE)
FORBIDDEN_RE = re.compile(r"TBD|TODO|\?\?\?")
UNKNOWN_RE = re.compile(r"UNKNOWN")

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
    "d_qkv",
    "bytes_bf16",
    "bytes_f32",
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
    "s_f32_bytes",
    "weight_bytes_lm_head",
    "weight_bytes_mtp_fc",
    "i_mlp_weight_only",
    "i_lm_head_weight_only",
    "i_gdn_vs_s_rw",
    "i_attn_core_vs_kv",
    "bottleneck_labels",
    "n_bottleneck_labels",
    "N_w",
    "N_bank",
    "sku_policy",
    "sku_unknown_symbols",
    "cta_T_candidates",
    "n_cta_T_candidates",
    "n_cta_T_selected",
    "cta_T_all_multiples_of_warp",
    "node_type_ids",
    "n_node_types",
    "n_embed_instances",
    "n_gated_attn_instances",
    "n_gated_delta_net_instances",
    "n_mlp_instances",
    "n_lm_head_instances",
    "n_mtp_mix_instances",
    "n_node_instances_complete",
    "ownership_ids",
    "n_ownership_classes",
    "reduction_ids",
    "n_reduction_classes",
    "estimate_dimension_ids",
    "n_estimate_dimensions",
    "sync_class_ids",
    "n_sync_classes",
    "pipeline_ids",
    "n_pipeline_ids",
    "mode_fit_ids",
    "n_mode_fit_ids",
    "consumer_mode_ids",
    "n_consumer_modes",
    "n_consumer_modes_selected",
    "stage_kind_ids",
    "n_stage_kinds",
    "mapping_ids",
    "n_mappings",
    "n_mappings_per_node",
    "n_mappings_selected",
    "mapping_selected",
    "mapping_node_types",
    "mapping_ownership_ids",
    "mapping_reduction_ids",
    "mapping_sync_class_ids",
    "mapping_primary_pipeline_ids",
    "mapping_secondary_pipeline_ids",
    "mapping_mode_fit_ids",
    "mapping_work_mac_ids",
    "mapping_layout_object_ids",
    "mapping_decomposition_ids",
    "mapping_records",
    "mapping_usefulness_label",
    "n_decode_primary_mappings",
    "n_prefill_primary_mappings",
    "n_both_mode_mappings",
    "n_mappings_using_red_grid",
    "n_mappings_using_sync_grid",
    "n_secondary_pipeline_hypotheses",
    "n_secondary_pipelines_selected",
    "attn_prefill_mma_is_secondary_hypothesis",
    "gdn_chunk_is_algebraic_equivalent",
    "work_partition_identity_present",
    "storage_rt_numeric",
    "storage_hbm_cited",
    "access_usefulness_label",
    "coalescing_identity_observed",
    "n_access_winners_selected",
    "mode_suitability_label",
    "n_mode_winners_selected",
    "mode_fit_at_T1_grid_T_degenerates",
    "evaluation_criterion_ids",
    "n_evaluation_criteria",
    "evaluation_usefulness_label",
    "n_evaluation_winners_selected",
    "roofline_is_bound_not_measurement",
    "occupancy_sku_symbols_unknown",
    "achieved_occupancy_reported",
    "parallel_decomposition_ids",
    "n_parallel_decompositions",
    "n_parallel_decompositions_selected",
    "decomposition_mapping_ids",
    "justification_hypothesis_ids",
    "n_justification_hypotheses",
    "n_justification_hypotheses_selected",
    "justification_usefulness_label",
    "tile_family_ids",
    "n_tile_families",
    "mma_decomposition_id",
    "mma_pipeline_id",
    "mma_tile_family_id",
    "fusion_hypothesis_ids",
    "n_fusion_hypotheses",
    "n_fusion_hypotheses_selected",
    "fusion_usefulness_label",
    "var_splitk_grid_id",
    "n_reduction_variant_hypotheses",
    "n_reduction_variant_hypotheses_selected",
    "red_grid_requires_cooperative_or_atomics",
    "sync_grid_requires_cooperative",
    "async_copy_optional",
    "mapping_risk_ids",
    "mapping_risk_severities",
    "mapping_high_ids",
    "mapping_medium_ids",
    "mapping_low_ids",
    "n_mapping_risks",
    "n_mapping_high",
    "n_mapping_medium",
    "n_mapping_low",
    "example_T",
    "T_is_stored_length_after_append",
    "primary_includes_mtp",
    "decode_prefill_share_artifact",
    "decode_prefill_share_graph",
    "decode_prefill_distinct_views_selected",
    "activations_in_layout_scope",
    "activation_live_is_occupancy_symbol",
    "hardware_independent",
    "cuda_mapping_deferred",
    "thread_geometry_absent",
    "launch_config_selected",
    "mapping_winner_selected",
    "kernel_named",
    "layout_winner_selected",
    "ordering_selected",
    "tile_size_selected",
    "mma_tile_extents_selected",
    "mma_shapes_selected",
    "mma_decomposition_named",
    "alignment_grain_selected",
    "conversion_pipeline_selected",
    "parallel_decomposition_selected",
    "parallel_decomposition_justifies_layout_selected",
    "ideal_byte_sequence_selected",
    "artifact_boundary_selected",
    "fusion_winner_selected",
    "ledger_open_question_mapping_winner_closed",
    "ledger_open_question_parallel_decomposition_closed",
    "evaluation_measured",
    "sku_limits_unknown",
    "async_copy_cap_unknown",
    "mma_shapes_unknown",
    "cluster_cap_unknown",
    "occupancy_formulae_instantiated",
    "gdn_primary_is_recurrent_eq_17",
    "chunkwise_not_zero_s_traffic",
    "paper_s_transpose_same_map",
    "state_write_not_optional",
    "kv_rope_baked_into_k",
    "gqa_repeat_not_stored",
    "conv_z_does_not_enter_conv",
    "weight_unique_counted_once",
    "weight_second_w_lm_read_is_hypothesis",
    "gguf_is_not_the_runtime_format",
    "safetensors_is_source_not_runtime",
    "vision_interface_is_not_a_node",
    "payloads_restreamed",
    "analyzes_work",
    "analyzes_storage",
    "analyzes_access",
    "analyzes_synchronization",
    "analyzes_occupancy",
    "analyzes_mode_suitability",
    "multiple_alternatives_per_node",
    "winner_selected_without_measurements",
    "cluster_assumed_present",
    "tma_assumed_present",
    "red_grid_selected",
    "activation_dtype_decided",
    "diagram_ids",
    "n_diagrams",
    "canonical_sentence_logical",
    "canonical_sentence_mappings",
    "canonical_sentence_symbolic",
    "canonical_sentence_open_question",
    "canonical_sentence_winner",
    "canonical_sentence_layout",
)


class MissingConfig(Exception):
    """Raised when the config path is absent or unreadable."""

    def __init__(self, path: Path) -> None:
        super().__init__(str(path))
        self.path = path


class CudaDesignSpaceMismatch(Exception):
    """Raised when the CUDA-design-space markdown fails a content check."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def dumps_summary(summary: dict) -> str:
    """Pretty-print the CUDA-design-space summary object.

    Args:
        summary: Object produced by :func:`instantiate_cuda_design_space`.

    Returns:
        Pretty-printed JSON including a trailing newline.
    """

    return json.dumps(summary, indent=2) + "\n"


def first_json_fence(text: str) -> tuple[dict, re.Match[str]]:
    """Parse the first fenced ``json`` code block in ``text``.

    Args:
        text: Markdown document containing a `` ```json `` fence.

    Returns:
        Decoded JSON object and the regex match covering the fence.

    Raises:
        AssertionError: If no fence exists or the payload is not an object.
        json.JSONDecodeError: If the fence is not valid JSON.
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


def _ratio_number(numerator: int, denominator: int) -> int | float:
    """Return ``numerator / denominator`` as an int when exact.

    Args:
        numerator: Intensity numerator (FLOP-like product).
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


def _ordered_str_list_mapping(
    keys: tuple[str, ...], values: dict[str, tuple[str, ...]]
) -> dict[str, list[str]]:
    """Copy ``values`` in ``keys`` order, converting tuples to lists.

    Args:
        keys: Required key order.
        values: Mapping keyed by those ids.

    Returns:
        Ordered mapping with list values.
    """

    return {key: list(values[key]) for key in keys}


def _ordered_optional_mapping(
    keys: tuple[str, ...], values: dict[str, str | None]
) -> dict[str, str | None]:
    """Copy optional string values in ``keys`` order.

    Args:
        keys: Required key order.
        values: Mapping keyed by those ids.

    Returns:
        Ordered mapping with string or null values.
    """

    return {key: values[key] for key in keys}


def instantiate_cuda_design_space(config: dict) -> dict:
    """Compute the TASK-17 CUDA-design-space summary from ``config``.

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
    dtype = _require_str(text, "dtype")
    mamba_ssm_dtype = _require_str(text, "mamba_ssm_dtype")
    assert dtype == "bfloat16", f"text_config.dtype {dtype!r} != 'bfloat16'"
    assert mamba_ssm_dtype == "float32", (
        f"text_config.mamba_ssm_dtype {mamba_ssm_dtype!r} != 'float32'"
    )

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
    linear_qkv_width = d_qkv
    linear_z_width = linear_num_value_heads * linear_value_head_dim
    q_proj_out = 2 * num_attention_heads * head_dim
    kv_proj_out = num_key_value_heads * head_dim
    o_proj_in = num_attention_heads * head_dim

    n_embed_instances = 2
    n_gated_attn_instances = n_full_layers_with_kv
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

    mac_embed = 0
    mac_full_proj_per_layer = (
        q_proj_out * hidden_size
        + kv_proj_out * hidden_size
        + kv_proj_out * hidden_size
        + hidden_size * o_proj_in
    )
    mac_attn_coeff_per_full_layer = 2 * num_attention_heads * head_dim
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
    kv_bytes_per_full_layer_per_token = (
        2 * num_key_value_heads * head_dim * BYTES_BF16
    )
    kv_bytes_all_per_token = (
        kv_bytes_per_full_layer_per_token * n_full_layers_with_kv
    )
    c_bytes_per_layer = linear_conv_delay * d_qkv * BYTES_BF16
    s_bytes_per_layer = (
        linear_num_value_heads
        * linear_key_head_dim
        * linear_value_head_dim
        * BYTES_F32
    )
    s_f32_bytes = s_bytes_per_layer * n_linear_layers
    weight_bytes_lm_head = vocab_size * hidden_size * BYTES_BF16
    weight_bytes_mtp_fc = 2 * hidden_size * (2 * hidden_size)
    mlp_weight_bytes_one_layer = (
        3 * intermediate_size * hidden_size * BYTES_BF16
    )
    i_mlp_weight_only = _ratio_number(
        2 * mac_mlp_per_layer, mlp_weight_bytes_one_layer
    )
    i_lm_head_weight_only = _ratio_number(2 * mac_lm_head, weight_bytes_lm_head)
    i_gdn_vs_s_rw = _ratio_number(2 * mac_gdn_per_layer, 2 * s_bytes_per_layer)
    i_attn_core_vs_kv = _ratio_number(
        2 * mac_attn_coeff_per_full_layer, kv_bytes_per_full_layer_per_token
    )

    mapping_selected = {mapping_id: False for mapping_id in MAPPING_IDS}
    mapping_secondary = _ordered_optional_mapping(
        MAPPING_IDS, MAPPING_SECONDARY_PIPELINE_IDS
    )
    mapping_layout_objects = _ordered_str_list_mapping(
        MAPPING_IDS, MAPPING_LAYOUT_OBJECT_IDS
    )
    mapping_decompositions = _ordered_str_list_mapping(
        MAPPING_IDS, MAPPING_DECOMPOSITION_IDS
    )
    hbm_citations = {
        "embed": ("TASK-06 weight_gather_bytes_per_row", "weight_gather_bytes_per_row"),
        "gated_attn": ("TASK-06 kv_bytes_per_full_layer_per_token", "kv_bytes_per_full_layer_per_token * sequence_length"),
        "gated_delta_net": ("TASK-06 c_bytes_per_layer and s_bytes_per_layer", "c_bytes_per_layer + s_bytes_per_layer"),
        "mlp": ("TASK-06 weight-memory class", "3 * intermediate_size * hidden_size * bytes_bf16"),
        "lm_head": ("TASK-06 weight_bytes_lm_head", "weight_bytes_lm_head"),
        "mtp_mix": ("TASK-06 weight_bytes_mtp_fc", "weight_bytes_mtp_fc"),
    }
    mapping_records: dict[str, dict] = {}
    for index, mapping_id in enumerate(MAPPING_IDS):
        node_type = MAPPING_NODE_TYPES[index]
        sync_class = MAPPING_SYNC_CLASS_IDS[index]
        dependency: object = "none" if sync_class == "sync_none" else f"{sync_class} orders mapping-local dependences"
        if mapping_id == "map_mtp_split_norm_gemm":
            dependency = {
                "stages": ["embedding_norm", "hidden_norm", "concat_materialize", "fc_gemm"],
                "events": ["embedding_norm->concat_materialize", "hidden_norm->concat_materialize", "concat_materialize->fc_gemm"],
            }
        hbm_citation, hbm_expression = hbm_citations[node_type]
        mapping_records[mapping_id] = {
            "node_type": node_type,
            "work": {"citation": f"TASK-06 {MAPPING_WORK_MAC_IDS[index]}", "partition": "TASK-17 F_owner=(e/L)*F_node for red_none; otherwise cited reduction partials"},
            "hbm": {"citation": hbm_citation, "expression": hbm_expression},
            "access": {"layout_objects": mapping_layout_objects[mapping_id], "decompositions": mapping_decompositions[mapping_id], "owner_expression": f"{MAPPING_OWNERSHIP_IDS[index]} owns {','.join(mapping_decompositions[mapping_id])}"},
            "synchronization": {"class": sync_class, "scope": MAPPING_OWNERSHIP_IDS[index], "dependency": dependency},
            "resources": {"R_t": "symbolic", "C_cta": "symbolic", "T_cta_candidates": list(CTA_T_CANDIDATES)},
            "occupancy": {"B_warp": "floor(W_max/W_cta)", "B_SM": "min(B_reg,B_smem,B_threads,B_cta,B_warp)", "O": "B_SM*W_cta/W_max", "W_need": "ceil(L_issue*N_sched)"},
            "waves": {"N_grid": "symbolic mapping-dependent grid size", "N_waves": "ceil(N_grid/(N_SM*B_SM))", "eta_wave": "N_grid/(N_waves*N_SM*B_SM)"},
            "mode_fit": MAPPING_MODE_FIT_IDS[index],
        }
    decomposition_mappings = _ordered_str_list_mapping(
        PARALLEL_DECOMPOSITION_IDS, DECOMPOSITION_MAPPING_IDS
    )
    n_secondary_pipeline_hypotheses = sum(
        1 for value in mapping_secondary.values() if value is not None
    )
    n_decode_primary_mappings = MAPPING_MODE_FIT_IDS.count("decode_primary")
    n_prefill_primary_mappings = MAPPING_MODE_FIT_IDS.count("prefill_primary")
    n_both_mode_mappings = MAPPING_MODE_FIT_IDS.count("both")
    n_mappings_using_red_grid = MAPPING_REDUCTION_IDS.count("red_grid")
    n_mappings_using_sync_grid = MAPPING_SYNC_CLASS_IDS.count("sync_grid")
    cta_t_all_multiples = all(
        candidate % N_W == 0 and candidate > 0 for candidate in CTA_T_CANDIDATES
    )
    _ = math.ceil(CTA_T_CANDIDATES[0] / N_W)

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
        "d_qkv": d_qkv,
        "bytes_bf16": BYTES_BF16,
        "bytes_f32": BYTES_F32,
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
        "s_f32_bytes": s_f32_bytes,
        "weight_bytes_lm_head": weight_bytes_lm_head,
        "weight_bytes_mtp_fc": weight_bytes_mtp_fc,
        "i_mlp_weight_only": i_mlp_weight_only,
        "i_lm_head_weight_only": i_lm_head_weight_only,
        "i_gdn_vs_s_rw": i_gdn_vs_s_rw,
        "i_attn_core_vs_kv": i_attn_core_vs_kv,
        "bottleneck_labels": list(BOTTLENECK_LABELS),
        "n_bottleneck_labels": len(BOTTLENECK_LABELS),
        "N_w": N_W,
        "N_bank": N_BANK,
        "sku_policy": SKU_POLICY,
        "sku_unknown_symbols": list(SKU_UNKNOWN_SYMBOLS),
        "cta_T_candidates": list(CTA_T_CANDIDATES),
        "n_cta_T_candidates": len(CTA_T_CANDIDATES),
        "n_cta_T_selected": 0,
        "cta_T_all_multiples_of_warp": cta_t_all_multiples,
        "node_type_ids": list(NODE_TYPE_IDS),
        "n_node_types": len(NODE_TYPE_IDS),
        "n_embed_instances": n_embed_instances,
        "n_gated_attn_instances": n_gated_attn_instances,
        "n_gated_delta_net_instances": n_gated_delta_net_instances,
        "n_mlp_instances": n_mlp_instances,
        "n_lm_head_instances": n_lm_head_instances,
        "n_mtp_mix_instances": n_mtp_mix_instances,
        "n_node_instances_complete": n_node_instances_complete,
        "ownership_ids": list(OWNERSHIP_IDS),
        "n_ownership_classes": len(OWNERSHIP_IDS),
        "reduction_ids": list(REDUCTION_IDS),
        "n_reduction_classes": len(REDUCTION_IDS),
        "estimate_dimension_ids": list(ESTIMATE_DIMENSION_IDS),
        "n_estimate_dimensions": len(ESTIMATE_DIMENSION_IDS),
        "sync_class_ids": list(SYNC_CLASS_IDS),
        "n_sync_classes": len(SYNC_CLASS_IDS),
        "pipeline_ids": list(PIPELINE_IDS),
        "n_pipeline_ids": len(PIPELINE_IDS),
        "mode_fit_ids": list(MODE_FIT_IDS),
        "n_mode_fit_ids": len(MODE_FIT_IDS),
        "consumer_mode_ids": list(CONSUMER_MODE_IDS),
        "n_consumer_modes": len(CONSUMER_MODE_IDS),
        "n_consumer_modes_selected": 0,
        "stage_kind_ids": list(STAGE_KIND_IDS),
        "n_stage_kinds": len(STAGE_KIND_IDS),
        "mapping_ids": list(MAPPING_IDS),
        "n_mappings": len(MAPPING_IDS),
        "n_mappings_per_node": 3,
        "n_mappings_selected": 0,
        "mapping_selected": mapping_selected,
        "mapping_node_types": list(MAPPING_NODE_TYPES),
        "mapping_ownership_ids": list(MAPPING_OWNERSHIP_IDS),
        "mapping_reduction_ids": list(MAPPING_REDUCTION_IDS),
        "mapping_sync_class_ids": list(MAPPING_SYNC_CLASS_IDS),
        "mapping_primary_pipeline_ids": list(MAPPING_PRIMARY_PIPELINE_IDS),
        "mapping_secondary_pipeline_ids": mapping_secondary,
        "mapping_mode_fit_ids": list(MAPPING_MODE_FIT_IDS),
        "mapping_work_mac_ids": list(MAPPING_WORK_MAC_IDS),
        "mapping_layout_object_ids": mapping_layout_objects,
        "mapping_decomposition_ids": mapping_decompositions,
        "mapping_records": mapping_records,
        "mapping_usefulness_label": "HYPOTHESIS",
        "n_decode_primary_mappings": n_decode_primary_mappings,
        "n_prefill_primary_mappings": n_prefill_primary_mappings,
        "n_both_mode_mappings": n_both_mode_mappings,
        "n_mappings_using_red_grid": n_mappings_using_red_grid,
        "n_mappings_using_sync_grid": n_mappings_using_sync_grid,
        "n_secondary_pipeline_hypotheses": n_secondary_pipeline_hypotheses,
        "n_secondary_pipelines_selected": 0,
        "attn_prefill_mma_is_secondary_hypothesis": True,
        "gdn_chunk_is_algebraic_equivalent": True,
        "work_partition_identity_present": True,
        "storage_rt_numeric": False,
        "storage_hbm_cited": True,
        "access_usefulness_label": "HYPOTHESIS",
        "coalescing_identity_observed": True,
        "n_access_winners_selected": 0,
        "mode_suitability_label": "HYPOTHESIS",
        "n_mode_winners_selected": 0,
        "mode_fit_at_T1_grid_T_degenerates": True,
        "evaluation_criterion_ids": list(EVALUATION_CRITERION_IDS),
        "n_evaluation_criteria": len(EVALUATION_CRITERION_IDS),
        "evaluation_usefulness_label": "HYPOTHESIS",
        "n_evaluation_winners_selected": 0,
        "roofline_is_bound_not_measurement": True,
        "occupancy_sku_symbols_unknown": True,
        "achieved_occupancy_reported": False,
        "parallel_decomposition_ids": list(PARALLEL_DECOMPOSITION_IDS),
        "n_parallel_decompositions": len(PARALLEL_DECOMPOSITION_IDS),
        "n_parallel_decompositions_selected": 0,
        "decomposition_mapping_ids": decomposition_mappings,
        "justification_hypothesis_ids": list(JUSTIFICATION_HYPOTHESIS_IDS),
        "n_justification_hypotheses": len(JUSTIFICATION_HYPOTHESIS_IDS),
        "n_justification_hypotheses_selected": 0,
        "justification_usefulness_label": "HYPOTHESIS",
        "tile_family_ids": list(TILE_FAMILY_IDS),
        "n_tile_families": len(TILE_FAMILY_IDS),
        "mma_decomposition_id": "par_gemm_d_out",
        "mma_pipeline_id": "tensor_core_mma",
        "mma_tile_family_id": "tile_mma_shaped",
        "fusion_hypothesis_ids": list(FUSION_HYPOTHESIS_IDS),
        "n_fusion_hypotheses": len(FUSION_HYPOTHESIS_IDS),
        "n_fusion_hypotheses_selected": 0,
        "fusion_usefulness_label": "HYPOTHESIS",
        "var_splitk_grid_id": "var_splitk_grid",
        "n_reduction_variant_hypotheses": 1,
        "n_reduction_variant_hypotheses_selected": 0,
        "red_grid_requires_cooperative_or_atomics": True,
        "sync_grid_requires_cooperative": True,
        "async_copy_optional": True,
        "mapping_risk_ids": list(MAPPING_RISK_IDS),
        "mapping_risk_severities": list(MAPPING_RISK_SEVERITIES),
        "mapping_high_ids": list(MAPPING_HIGH_IDS),
        "mapping_medium_ids": list(MAPPING_MEDIUM_IDS),
        "mapping_low_ids": list(MAPPING_LOW_IDS),
        "n_mapping_risks": len(MAPPING_RISK_IDS),
        "n_mapping_high": len(MAPPING_HIGH_IDS),
        "n_mapping_medium": len(MAPPING_MEDIUM_IDS),
        "n_mapping_low": len(MAPPING_LOW_IDS),
        "example_T": list(EXAMPLE_T),
        "T_is_stored_length_after_append": True,
        "primary_includes_mtp": True,
        "decode_prefill_share_artifact": True,
        "decode_prefill_share_graph": True,
        "decode_prefill_distinct_views_selected": False,
        "activations_in_layout_scope": False,
        "activation_live_is_occupancy_symbol": True,
        "hardware_independent": False,
        "cuda_mapping_deferred": False,
        "thread_geometry_absent": False,
        "launch_config_selected": False,
        "mapping_winner_selected": False,
        "kernel_named": False,
        "layout_winner_selected": False,
        "ordering_selected": False,
        "tile_size_selected": False,
        "mma_tile_extents_selected": False,
        "mma_shapes_selected": False,
        "mma_decomposition_named": True,
        "alignment_grain_selected": False,
        "conversion_pipeline_selected": False,
        "parallel_decomposition_selected": False,
        "parallel_decomposition_justifies_layout_selected": False,
        "ideal_byte_sequence_selected": False,
        "artifact_boundary_selected": False,
        "fusion_winner_selected": False,
        "ledger_open_question_mapping_winner_closed": False,
        "ledger_open_question_parallel_decomposition_closed": False,
        "evaluation_measured": False,
        "sku_limits_unknown": True,
        "async_copy_cap_unknown": True,
        "mma_shapes_unknown": True,
        "cluster_cap_unknown": True,
        "occupancy_formulae_instantiated": True,
        "gdn_primary_is_recurrent_eq_17": True,
        "chunkwise_not_zero_s_traffic": True,
        "paper_s_transpose_same_map": True,
        "state_write_not_optional": True,
        "kv_rope_baked_into_k": True,
        "gqa_repeat_not_stored": True,
        "conv_z_does_not_enter_conv": True,
        "weight_unique_counted_once": True,
        "weight_second_w_lm_read_is_hypothesis": True,
        "gguf_is_not_the_runtime_format": True,
        "safetensors_is_source_not_runtime": True,
        "vision_interface_is_not_a_node": True,
        "payloads_restreamed": False,
        "analyzes_work": True,
        "analyzes_storage": True,
        "analyzes_access": True,
        "analyzes_synchronization": True,
        "analyzes_occupancy": True,
        "analyzes_mode_suitability": True,
        "multiple_alternatives_per_node": True,
        "winner_selected_without_measurements": False,
        "cluster_assumed_present": False,
        "tma_assumed_present": False,
        "red_grid_selected": False,
        "activation_dtype_decided": False,
        "diagram_ids": list(DIAGRAM_REQUIRED_IDS),
        "n_diagrams": 1,
        "canonical_sentence_logical": CANONICAL_SENTENCE_LOGICAL,
        "canonical_sentence_mappings": CANONICAL_SENTENCE_MAPPINGS,
        "canonical_sentence_symbolic": CANONICAL_SENTENCE_SYMBOLIC,
        "canonical_sentence_open_question": CANONICAL_SENTENCE_OPEN_QUESTION,
        "canonical_sentence_winner": CANONICAL_SENTENCE_WINNER,
        "canonical_sentence_layout": CANONICAL_SENTENCE_LAYOUT,
    }
    _assert_summary(summary)
    return summary


def _assert_summary(summary: dict) -> None:
    """Assert dossier identities on a live summary object.

    Args:
        summary: Object produced by :func:`instantiate_cuda_design_space`.

    Raises:
        AssertionError: If a locked identity does not hold.
    """

    assert list(summary) == list(SCHEMA_KEYS), (
        f"schema key order mismatch: {list(summary)} vs {list(SCHEMA_KEYS)}"
    )
    assert summary["n_linear_layers"] == 48
    assert summary["n_full_layers"] == 16
    assert summary["n_mtp_blocks"] == 1
    assert summary["n_full_layers_with_kv"] == 17
    assert summary["full_attention_indices"] == list(LOCKED_FULL_ATTENTION_INDICES)
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
    assert summary["d_qkv"] == 10240
    assert summary["n_node_types"] == 6
    assert summary["n_node_instances_complete"] == 135
    assert summary["n_embed_instances"] == 2
    assert summary["n_gated_attn_instances"] == 17
    assert summary["n_gated_delta_net_instances"] == 48
    assert summary["n_mlp_instances"] == 65
    assert summary["n_lm_head_instances"] == 2
    assert summary["n_mtp_mix_instances"] == 1
    assert summary["n_mappings"] == 18
    assert summary["n_mappings_per_node"] == 3
    assert summary["n_mappings_selected"] == 0
    for node_id in NODE_TYPE_IDS:
        assert summary["mapping_node_types"].count(node_id) == 3, (
            f"{node_id} does not appear exactly 3 times"
        )
    assert summary["n_ownership_classes"] == 4
    assert summary["n_reduction_classes"] == 5
    assert summary["n_estimate_dimensions"] == 6
    assert summary["n_sync_classes"] == 5
    assert summary["n_pipeline_ids"] == 6
    assert summary["n_mode_fit_ids"] == 3
    assert summary["n_evaluation_criteria"] == 7
    assert summary["n_parallel_decompositions"] == 9
    assert summary["n_justification_hypotheses"] == 10
    assert summary["n_fusion_hypotheses"] == 22
    assert summary["n_mapping_risks"] == 8
    assert summary["n_mapping_high"] == 1
    assert summary["n_mapping_medium"] == 5
    assert summary["n_mapping_low"] == 2
    assert summary["n_cta_T_candidates"] == 4
    assert summary["n_cta_T_selected"] == 0
    assert summary["cta_T_all_multiples_of_warp"] is True
    assert summary["N_w"] == 32
    assert summary["N_bank"] == 32
    assert summary["n_mappings_using_red_grid"] == 0
    assert summary["n_mappings_using_sync_grid"] == 0
    assert summary["n_decode_primary_mappings"] == 3
    assert summary["n_prefill_primary_mappings"] == 2
    assert summary["n_both_mode_mappings"] == 13
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
    assert summary["mac_full_proj_per_layer"] == (
        2 * 24 * 256 * hidden_size
        + 2 * 4 * 256 * hidden_size
        + hidden_size * (24 * 256)
    )
    assert summary["mac_mlp_per_layer"] == (
        3 * summary["intermediate_size"] * hidden_size
    )
    assert summary["mac_lm_head"] == summary["vocab_size"] * hidden_size
    assert summary["mac_gdn_per_layer"] == 3 * 48 * 128 * 128
    assert summary["mac_mtp_fc"] == hidden_size * (2 * hidden_size)
    assert summary["mac_lin_conv_per_layer"] == summary["d_qkv"] * 4
    assert summary["i_mlp_weight_only"] == 1
    assert summary["i_lm_head_weight_only"] == 1
    assert summary["i_gdn_vs_s_rw"] == 0.75
    assert summary["i_attn_core_vs_kv"] == 6
    assert summary["kv_bytes_all_per_token"] == 69632
    assert summary["s_f32_bytes"] == 150994944
    assert summary["weight_gather_bytes_per_row"] == 10240
    assert summary["weight_bytes_lm_head"] == 2542796800
    assert summary["weight_bytes_mtp_fc"] == 104857600
    assert summary["mapping_winner_selected"] is False
    assert summary["launch_config_selected"] is False
    assert summary["kernel_named"] is False
    assert summary["ledger_open_question_mapping_winner_closed"] is False
    assert summary["parallel_decomposition_justifies_layout_selected"] is False
    assert summary["mma_decomposition_named"] is True
    assert summary["mma_tile_extents_selected"] is False
    assert summary["mma_shapes_unknown"] is True
    assert summary["evaluation_measured"] is False
    assert summary["sku_limits_unknown"] is True
    assert summary["cluster_assumed_present"] is False
    assert summary["tma_assumed_present"] is False
    assert summary["fusion_winner_selected"] is False
    assert summary["decode_prefill_distinct_views_selected"] is False
    assert summary["analyzes_work"] is True
    assert summary["analyzes_storage"] is True
    assert summary["analyzes_access"] is True
    assert summary["analyzes_synchronization"] is True
    assert summary["analyzes_occupancy"] is True
    assert summary["analyzes_mode_suitability"] is True
    assert summary["multiple_alternatives_per_node"] is True
    assert summary["winner_selected_without_measurements"] is False
    assert summary["thread_geometry_absent"] is False
    assert summary["cuda_mapping_deferred"] is False
    assert summary["hardware_independent"] is False
    assert summary["n_diagrams"] == 1
    assert list(summary["mapping_records"]) == list(MAPPING_IDS)
    for record in summary["mapping_records"].values():
        assert set(record) == {"node_type", "work", "hbm", "access", "synchronization", "resources", "occupancy", "waves", "mode_fit"}
        assert set(record["work"]) == {"citation", "partition"}
        assert set(record["hbm"]) == {"citation", "expression"}
        assert set(record["access"]) == {"layout_objects", "decompositions", "owner_expression"}
        assert set(record["synchronization"]) == {"class", "scope", "dependency"}
        assert set(record["resources"]) == {"R_t", "C_cta", "T_cta_candidates"}
        assert set(record["occupancy"]) == {"B_warp", "B_SM", "O", "W_need"}
        assert set(record["waves"]) == {"N_grid", "N_waves", "eta_wave"}
        assert record["work"]["citation"].startswith("TASK-06 ")
        assert record["hbm"]["citation"].startswith("TASK-06 ")
    assert summary["mapping_records"]["map_mlp_grid_T"]["synchronization"] == {"class": "sync_none", "scope": "grid", "dependency": "none"}
    assert summary["mapping_records"]["map_mtp_split_norm_gemm"]["synchronization"]["dependency"]["stages"] == ["embedding_norm", "hidden_norm", "concat_materialize", "fc_gemm"]
    assert summary["n_secondary_pipelines_selected"] == 0
    assert summary["n_fusion_hypotheses_selected"] == 0
    used_decomps: set[str] = set()
    for attached in summary["mapping_decomposition_ids"].values():
        used_decomps.update(attached)
    assert set(PARALLEL_DECOMPOSITION_IDS) <= used_decomps
    assert summary["mma_decomposition_id"] == "par_gemm_d_out"
    assert summary["mma_pipeline_id"] == "tensor_core_mma"
    assert summary["mma_tile_family_id"] == "tile_mma_shaped"
    assert summary["sku_policy"] == SKU_POLICY
    assert len(summary["sku_unknown_symbols"]) == 17
    _assert_per_node_partition(summary)
    _assert_bidirectional_decompositions(summary)


def _assert_per_node_partition(summary: dict) -> None:
    """Assert three mappings per node with distinct ownership and sync/reduce.

    Args:
        summary: Live CUDA-design-space object.

    Raises:
        AssertionError: If a node type lacks the required alternative spread.
    """

    by_node: dict[str, list[int]] = {node_id: [] for node_id in NODE_TYPE_IDS}
    for index, node_id in enumerate(summary["mapping_node_types"]):
        by_node[node_id].append(index)
    for node_id, indices in by_node.items():
        assert len(indices) == 3, f"{node_id} mapping count {len(indices)} != 3"
        ownerships = {summary["mapping_ownership_ids"][index] for index in indices}
        reductions = {summary["mapping_reduction_ids"][index] for index in indices}
        syncs = {summary["mapping_sync_class_ids"][index] for index in indices}
        assert len(ownerships) >= 2, (
            f"{node_id} has fewer than two distinct ownership_ids"
        )
        assert len(reductions) >= 2 or len(syncs) >= 2, (
            f"{node_id} needs two distinct reduction_ids or sync_class_ids"
        )


def _assert_bidirectional_decompositions(summary: dict) -> None:
    """Assert mapping attachments and reverse decomposition tables agree.

    Args:
        summary: Live CUDA-design-space object.

    Raises:
        AssertionError: If a pairing is one-sided.
    """

    forward = summary["mapping_decomposition_ids"]
    reverse = summary["decomposition_mapping_ids"]
    for mapping_id, decomps in forward.items():
        for decomp_id in decomps:
            assert mapping_id in reverse[decomp_id], (
                f"{mapping_id} attaches {decomp_id} but reverse table omits it"
            )
    for decomp_id, mapping_ids in reverse.items():
        for mapping_id in mapping_ids:
            assert decomp_id in forward[mapping_id], (
                f"{decomp_id} lists {mapping_id} but forward table omits it"
            )


def diff_summary(live: dict, documented: dict) -> list[str]:
    """Compare documented JSON against a live summary object.

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
    if documented != live:
        for key in SCHEMA_KEYS:
            if key not in documented or key not in live:
                continue
            if documented[key] != live[key]:
                diffs.append(
                    f"{key}: documented {documented[key]!r} != live {live[key]!r}"
                )
        if not any(key in documented and key in live for key in SCHEMA_KEYS):
            diffs.append("documented JSON is not deep-equal to the live object")
    return diffs


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


def check_cuda_design_space(live: dict, design_space_path: Path) -> None:
    """Check a CUDA-design-space markdown file against a live summary.

    Args:
        live: CUDA-design-space summary object from sitting config.
        design_space_path: Path to ``cuda-design-space.md``.

    Raises:
        CudaDesignSpaceMismatch: On heading, JSON, diagram, or token
            mismatches.
        OSError: If the file cannot be read.
    """

    text = design_space_path.read_text(encoding="utf-8")
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

    if fence_match is not None:
        search_text = text[: fence_match.start()] + text[fence_match.end() :]
    else:
        search_text = text

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
            node_id for node_id in DIAGRAM_REQUIRED_IDS if node_id not in body
        ]
        if missing_ids:
            differences.append(f"diagram 1: missing node ids {missing_ids}")
        alt_match = re.search(
            r"^## Per-node mapping alternatives\s*$", text, flags=re.MULTILINE
        )
        work_match = re.search(
            r"^## Work, storage, and access estimates\s*$",
            text,
            flags=re.MULTILINE,
        )
        if alt_match is None or work_match is None:
            differences.append(
                "missing Per-node mapping alternatives or Work heading"
            )
        elif not (alt_match.end() <= match.start() < work_match.start()):
            differences.append(
                "mermaid fence is not under ## Per-node mapping alternatives"
            )
        caption = text[alt_match.end() : match.start()] if alt_match else ""
        if "HYPOTHESIS" not in caption or "winner" not in caption:
            differences.append(
                "mermaid caption missing HYPOTHESIS or winner"
            )

    if CANONICAL_SENTENCE_LOGICAL not in text:
        differences.append("canonical_sentence_logical not present verbatim")
    if CANONICAL_SENTENCE_MAPPINGS not in text:
        differences.append("canonical_sentence_mappings not present verbatim")
    if CANONICAL_SENTENCE_SYMBOLIC not in text:
        differences.append("canonical_sentence_symbolic not present verbatim")
    if CANONICAL_SENTENCE_OPEN_QUESTION not in text:
        differences.append(
            "canonical_sentence_open_question not present verbatim"
        )
    if CANONICAL_SENTENCE_WINNER not in text:
        differences.append("canonical_sentence_winner not present verbatim")
    if CANONICAL_SENTENCE_LAYOUT not in text:
        differences.append("canonical_sentence_layout not present verbatim")
    layout_heading = re.search(
        r"^## Layout instantiations\s*$", text, flags=re.MULTILINE
    )
    eval_heading = re.search(
        r"^## Evaluation instantiation\s*$", text, flags=re.MULTILINE
    )
    fusion_heading = re.search(
        r"^## Fusion versus occupancy\s*$", text, flags=re.MULTILINE
    )
    if layout_heading is None or eval_heading is None:
        differences.append(
            "missing Layout instantiations or Evaluation instantiation heading"
        )
    else:
        layout_section = text[layout_heading.end() : eval_heading.start()]
        if CANONICAL_SENTENCE_LAYOUT not in layout_section:
            differences.append(
                "canonical_sentence_layout missing under Layout instantiations"
            )
    if eval_heading is None or fusion_heading is None:
        differences.append(
            "missing Evaluation instantiation or Fusion versus occupancy heading"
        )
    elif CANONICAL_SENTENCE_WINNER not in text[
        eval_heading.end() : fusion_heading.start()
    ]:
        differences.append(
            "canonical_sentence_winner missing under Evaluation instantiation"
        )

    for label, ids in SUBSTRING_ID_GROUPS:
        for item_id in ids:
            if item_id not in text:
                differences.append(f"{label} {item_id!r} missing as substring")

    if "HYPOTHESIS" not in text:
        differences.append("word HYPOTHESIS missing")
    if "unresolved" not in text:
        differences.append("word unresolved missing")

    for match in FORBIDDEN_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        differences.append(f"line {line}: forbidden token {match.group(0)!r}")

    try:
        vis_start, vis_end = _deferred_vision_span(text)
        if not UNKNOWN_RE.search(text, vis_start, vis_end):
            differences.append("word UNKNOWN missing in Deferred vision")
    except AssertionError as exc:
        differences.append(str(exc))

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

    for tag in FORMULA_TAGS:
        if tag not in search_text:
            differences.append(f"formula tag {tag} missing outside JSON fence")
    for substring in FORMULA_SUBSTRINGS:
        if substring not in search_text:
            differences.append(
                f"formula substring missing outside JSON fence: {substring}"
            )

    if differences:
        raise CudaDesignSpaceMismatch(differences)


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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the CUDA-design-space checker.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Instantiate Qwen3.8-27B language+MTP CUDA-design-space summary "
            "from text_config and check docs/architecture/cuda-design-space.md."
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
        help="Write CUDA-design-space summary JSON to stdout.",
    )
    parser.add_argument(
        "--cuda-design-space",
        type=Path,
        default=None,
        help="Markdown path whose first json fence must match live summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the CUDA-design-space checker CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch, 2 missing
        config.
    """

    args = _parse_args(argv)
    check_path = args.cuda_design_space
    print_json = args.json or check_path is None
    try:
        config = load_config(args.config)
        summary = instantiate_cuda_design_space(config)
        if check_path is not None:
            if not check_path.is_file():
                print(
                    f"cuda-design-space file not found: {check_path}",
                    file=sys.stderr,
                )
                return 1
            check_cuda_design_space(summary, check_path)
    except MissingConfig as exc:
        print(str(exc.path), file=sys.stderr)
        return 2
    except CudaDesignSpaceMismatch as exc:
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
