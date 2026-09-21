#!/usr/bin/env python3
"""Check Qwen3.8-27B clean-sheet review against sitting text_config.

Reads ``text_config`` from a Transformers ``config.json`` (no safetensor
payloads, no GPU query, no GGUF), compares first JSON fences of the
twenty-one TASK-01–20 corpus documents, and either prints the review
summary object or checks that
``docs/architecture/clean-sheet-review.md`` matches it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

AUTHORITY = "docs/architecture/plan.md"
DELIVERABLE_REVIEW = "docs/architecture/clean-sheet-review.md"
SEMANTICS_PATH = "docs/architecture/model-semantics.md"
WORK_TRAFFIC_PATH = "docs/architecture/work-and-traffic.md"
NUMERICAL_PATH = "docs/architecture/numerical-sensitivity.md"
ARCHITECTURE_PATH = "docs/architecture/clean-sheet-architecture.md"
BACKLOG_PATH = "docs/architecture/experiment-backlog.md"
TENSOR_ANALYSIS_PATH = "docs/architecture/bf16-tensor-analysis.md"

CANONICAL_SENTENCE_LOGICAL = (
    "Logical values in this document are graph nodes. They do not imply "
    "physical allocation, materialization, buffer reuse, or kernel fusion."
)
CANONICAL_SENTENCE_REVIEW = (
    "This document is a Phase 1 consistency review of TASK-01 through "
    "TASK-20, not a selected runtime and not measurements."
)
CANONICAL_SENTENCE_UNKNOWNS = (
    "True unknowns remain unknowns; consistency review does not select "
    "winners or close experiment questions."
)
CANONICAL_SENTENCE_FREEZE = (
    "The reviewed design is marked FROZEN_FOR_COMPARATIVE_REVIEW; "
    "existing runtimes are not thereby design authority."
)
REVIEW_QUESTION_SENTENCE = (
    "Cross-document first-fence schemas, keys, counts, and flags are mechanically "
    "checked; this is not independent semantic proof; true unknowns are preserved; "
    "the reviewed design is marked FROZEN_FOR_COMPARATIVE_REVIEW."
)
FREEZE_NON_AUTHORITY_SENTENCE = (
    "Existing Quartz and llama.cpp/GGML Qwen implementations are not "
    "design authority by this freeze; they may be compared later as "
    "black-box references with matching identities."
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
N_TENSORS = 1199
N_SHARDS = 18
N_PARAMETERS = 27781427952
N_BYTES = 55562855904
N_LANGUAGE_MTP_PARAMETERS = 27320697856
N_VISION_PARAMETERS = 460730096
N_LANGUAGE_MTP_TENSORS = 866
N_VISION_TENSORS = 333
WEIGHT_BYTES_LANGUAGE_MTP_EXCL_VISION = 54641395712
WEIGHT_BYTES_UNIQUE_NON_EMBED = 52098598912
N_CATALOG_NODES = 52
N_FULL_LAYERS_WITH_KV = 17
KV_BYTES_ALL_PER_TOKEN = 69632
STORAGE_KV_BYTES_COEFF_T = 69632
STORAGE_FIXED_BYTES = 153944064
S_BYTES_ALL = 150994944
C_BYTES_ALL = 2949120
N_EQUATION_TAGS = 24
EQUATION_TAG_START = 1
EQUATION_TAG_END = 24
N_CANDIDATE_RECIPES = 22
N_FUSION_HYPOTHESES = 22
N_MAPPINGS = 18
N_SENSITIVE_OPS = 21
N_CATALOGUED_ALTERNATIVES = 125
POOLED_ALL_ABSMAX = 25.5
POOLED_LANGUAGE_MTP_ABSMAX = 19.25
POOLED_VISION_ABSMAX = 25.5
N_KV_HEADS = 4

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

CORPUS_PATHS: tuple[str, ...] = (
    "docs/architecture/model-inventory.md",
    "docs/architecture/model-semantics.md",
    "docs/architecture/dataflow.md",
    "docs/architecture/lifetime-and-state.md",
    "docs/architecture/bf16-tensor-analysis.md",
    "docs/architecture/work-and-traffic.md",
    "docs/architecture/numerical-sensitivity.md",
    "docs/architecture/quantization-design-space.md",
    "docs/architecture/runtime-format-design.md",
    "docs/architecture/model-compiler-plan.md",
    "docs/architecture/semantic-graph.md",
    "docs/architecture/materialization-and-fusion.md",
    "docs/architecture/decode-plan.md",
    "docs/architecture/prefill-plan.md",
    "docs/architecture/layout-strategy.md",
    "docs/architecture/cuda-hardware-model.md",
    "docs/architecture/cuda-design-space.md",
    "docs/architecture/quantization-validation.md",
    "docs/architecture/performance-validation.md",
    "docs/architecture/clean-sheet-architecture.md",
    "docs/architecture/experiment-backlog.md",
)

CORPUS_TASK_IDS: dict[str, str] = {
    "docs/architecture/model-inventory.md": "TASK-01",
    "docs/architecture/model-semantics.md": "TASK-02",
    "docs/architecture/dataflow.md": "TASK-03",
    "docs/architecture/lifetime-and-state.md": "TASK-04",
    "docs/architecture/bf16-tensor-analysis.md": "TASK-05",
    "docs/architecture/work-and-traffic.md": "TASK-06",
    "docs/architecture/numerical-sensitivity.md": "TASK-07",
    "docs/architecture/quantization-design-space.md": "TASK-08",
    "docs/architecture/runtime-format-design.md": "TASK-09",
    "docs/architecture/model-compiler-plan.md": "TASK-10",
    "docs/architecture/semantic-graph.md": "TASK-11",
    "docs/architecture/materialization-and-fusion.md": "TASK-12",
    "docs/architecture/decode-plan.md": "TASK-13",
    "docs/architecture/prefill-plan.md": "TASK-14",
    "docs/architecture/layout-strategy.md": "TASK-15",
    "docs/architecture/cuda-hardware-model.md": "TASK-16",
    "docs/architecture/cuda-design-space.md": "TASK-17",
    "docs/architecture/quantization-validation.md": "TASK-18",
    "docs/architecture/performance-validation.md": "TASK-19",
    "docs/architecture/clean-sheet-architecture.md": "TASK-20",
    "docs/architecture/experiment-backlog.md": "TASK-20",
}

CHECK_FAMILY_IDS: tuple[str, ...] = (
    "equations",
    "dimensions",
    "totals",
    "state",
    "contracts",
    "proposals",
)

NOTATION_ALIAS_IDS: tuple[str, ...] = (
    "alias_dtype_bf16_bfloat16",
    "alias_tmax_model_vs_sku",
    "alias_authority_checkpoint_vs_plan",
    "alias_n_decoder_layers_num_hidden_layers",
    "alias_n_mtp_blocks_mtp_num_hidden_layers",
    "alias_n_kv_heads_num_key_value_heads",
    "alias_warp_n_w_n_bank",
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

SIBLING_CHECKER_SCRIPTS: tuple[str, ...] = (
    "scripts/inventory_bf16_checkpoint.py",
    "scripts/check_model_semantics.py",
    "scripts/check_dataflow.py",
    "scripts/check_lifetime_and_state.py",
    "scripts/check_work_and_traffic.py",
    "scripts/check_numerical_sensitivity.py",
    "scripts/check_quantization_design_space.py",
    "scripts/check_runtime_format_design.py",
    "scripts/check_model_compiler_plan.py",
    "scripts/check_semantic_graph.py",
    "scripts/check_materialization_and_fusion.py",
    "scripts/check_decode_plan.py",
    "scripts/check_prefill_plan.py",
    "scripts/check_layout_strategy.py",
    "scripts/check_cuda_hardware_model.py",
    "scripts/check_cuda_design_space.py",
    "scripts/check_quantization_validation.py",
    "scripts/check_performance_validation.py",
    "scripts/check_clean_sheet_architecture.py",
)

REVIEW_HEADINGS: tuple[str, ...] = (
    "Authority",
    "Review convention",
    "Corpus and check families",
    "Cross-document findings",
    "Preserved unknowns",
    "Freeze",
    "Deferred vision",
    "Machine-checkable summary JSON",
)

DIAGRAM_IDS: tuple[str, ...] = (
    "corpus",
    "eq",
    "dim",
    "tot",
    "state",
    "contract",
    "prop",
    "alias",
    "openq",
    "freeze",
)

DIAGRAM_MERMAID_IDS: tuple[str, ...] = (
    "CORPUS",
    "EQ",
    "DIM",
    "TOT",
    "STATE",
    "CONTRACT",
    "PROP",
    "ALIAS",
    "OPENQ",
    "FREEZE",
)

KEY_ALIASES: dict[str, str] = {
    "num_hidden_layers": "n_decoder_layers",
    "mtp_num_hidden_layers": "n_mtp_blocks",
    "num_key_value_heads": "n_kv_heads",
    "warp_size": "N_w",
    "n_bank": "N_bank",
}

SELECTED_FLAGS_MUST_BE_FALSE: tuple[str, ...] = (
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
)

LOCKED_DOCUMENT_INTEGERS: tuple[int | float, ...] = (
    5120,
    17408,
    248320,
    64,
    48,
    16,
    262144,
    1199,
    27781427952,
    55562855904,
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
    52,
    69632,
    153944064,
    21,
    12,
    24,
    19.25,
    25.5,
)

FORBIDDEN_WINNER_PHRASES: tuple[str, ...] = (
    "selected mapping winner",
    "winning kernel",
    "winning mapping",
    "selected recipe",
    "selected fusion",
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
    "not thereby design authority",
)

JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)
MERMAID_FENCE_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
HEADING_RE = re.compile(r"^## (?!#)(.+)$", re.MULTILINE)
FORBIDDEN_RE = re.compile(r"TBD|TODO|\?\?\?")
UNKNOWN_RE = re.compile(r"UNKNOWN")

SCHEMA_KEYS: tuple[str, ...] = (
    "authority",
    "deliverable_review",
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
    "n_tensors",
    "n_shards",
    "n_parameters",
    "n_bytes",
    "n_language_mtp_parameters",
    "n_vision_parameters",
    "n_language_mtp_tensors",
    "n_vision_tensors",
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
    "n_catalog_nodes",
    "catalog_ids",
    "n_full_layers_with_kv",
    "kv_bytes_all_per_token",
    "storage_kv_bytes_coeff_T",
    "storage_fixed_bytes",
    "s_bytes_all",
    "c_bytes_all",
    "n_equation_tags",
    "equation_tag_start",
    "equation_tag_end",
    "n_node_types",
    "node_type_ids",
    "n_node_instances_complete",
    "n_stage_kinds",
    "stage_kind_ids",
    "n_candidate_recipes",
    "n_fusion_hypotheses",
    "n_mappings",
    "n_sku_unknown_symbols",
    "sku_unknown_symbols",
    "n_sensitive_ops",
    "n_catalogued_alternatives",
    "pooled_all_absmax",
    "pooled_language_mtp_absmax",
    "pooled_vision_absmax",
    "n_corpus_documents",
    "corpus_paths",
    "corpus_task_ids",
    "n_check_families",
    "check_family_ids",
    "n_notation_aliases",
    "notation_alias_ids",
    "n_inconsistencies_found",
    "n_inconsistencies_corrected",
    "inconsistency_ids",
    "n_remaining_open_questions",
    "remaining_open_question_ids",
    "n_remaining_open_questions_closed",
    "n_sibling_checkers",
    "sibling_checker_scripts",
    "n_review_headings",
    "review_headings",
    "n_headings",
    "n_diagrams",
    "diagram_ids",
    "earlier_task_wins",
    "ledger_open_question_unknowns_preserved",
    "frozen_for_comparative_review",
    "synthesis_frozen_flag",
    "freeze_recorded_in_review_only",
    "freeze_authorizes_later_comparative_review",
    "freeze_makes_existing_runtimes_design_authority",
    "comparative_review_executed_here",
    "draft_banners_left_in_place",
    "pareto_frontier_selected",
    "fusion_winner_selected",
    "mapping_winner_selected",
    "winner_selected_without_measurements",
    "n_alternatives_selected",
    "sku_table_filled",
    "keep_source_is_not_quality_winner",
    "reconstruction_is_not_quality",
    "toks_is_not_quality_axis",
    "microbenchmark_cannot_pass_mapping",
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
    "alternative_usefulness_label",
    "canonical_sentence_logical",
    "canonical_sentence_review",
    "canonical_sentence_unknowns",
    "canonical_sentence_freeze",
    "review_question_sentence",
    "freeze_non_authority_sentence",
)


class MissingConfig(Exception):
    """Raised when the config path is absent or unreadable."""

    def __init__(self, path: Path) -> None:
        super().__init__(str(path))
        self.path = path


class CleanSheetMismatch(Exception):
    """Raised when the review markdown or corpus fences fail a check."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def dumps_summary(summary: dict) -> str:
    """Pretty-print the clean-sheet-review summary object.

    Args:
        summary: Object produced by :func:`instantiate_clean_sheet_review`.

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


def _normalize_dtype(value: object) -> object:
    """Return the dtype alias used for cross-document comparison.

    Args:
        value: Fence ``dtype`` field.

    Returns:
        ``bfloat16`` when the value is the documented BF16 alias, else
        ``value`` unchanged.
    """

    if value in ("BF16", "bfloat16"):
        return "bfloat16"
    return value


def _comparison_locked(occupancy: dict) -> dict:
    """Return locked occupancy/totals/state keys used for fence comparison.

    Args:
        occupancy: Live occupancy dict from sitting ``text_config``.

    Returns:
        Mapping of comparable keys to locked values. Review-only keys
        such as ``frozen_for_comparative_review`` are omitted.
    """

    return {
        "hidden_size": occupancy["hidden_size"],
        "intermediate_size": occupancy["intermediate_size"],
        "vocab_size": occupancy["vocab_size"],
        "n_decoder_layers": occupancy["n_decoder_layers"],
        "n_linear_layers": occupancy["n_linear_layers"],
        "n_full_layers": occupancy["n_full_layers"],
        "n_mtp_blocks": occupancy["n_mtp_blocks"],
        "full_attention_indices": occupancy["full_attention_indices"],
        "dtype": occupancy["dtype"],
        "mamba_ssm_dtype": occupancy["mamba_ssm_dtype"],
        "model_T_max": occupancy["model_T_max"],
        "n_kv_heads": occupancy["n_kv_heads"],
        "n_tensors": N_TENSORS,
        "n_shards": N_SHARDS,
        "n_parameters": N_PARAMETERS,
        "n_bytes": N_BYTES,
        "n_language_mtp_parameters": occupancy["n_language_mtp_parameters"],
        "n_vision_parameters": N_VISION_PARAMETERS,
        "n_language_mtp_tensors": N_LANGUAGE_MTP_TENSORS,
        "n_vision_tensors": N_VISION_TENSORS,
        "weight_bytes_language_mtp_excl_vision": occupancy[
            "weight_bytes_language_mtp_excl_vision"
        ],
        "weight_bytes_unique_non_embed": occupancy[
            "weight_bytes_unique_non_embed"
        ],
        "mac_C_complete": MAC_C_COMPLETE,
        "mac_A_complete": MAC_A_COMPLETE,
        "mac_decode_complete_T1": MAC_DECODE_COMPLETE_T1,
        "mac_prefill_complete_T1": MAC_PREFILL_COMPLETE_T1,
        "mac_decode_complete_T4096": MAC_DECODE_COMPLETE_T4096,
        "mac_prefill_complete_T4096": MAC_PREFILL_COMPLETE_T4096,
        "example_T_values": list(EXAMPLE_T_VALUES),
        "N_w": N_W,
        "N_bank": N_BANK,
        "n_catalog_nodes": N_CATALOG_NODES,
        "catalog_ids": list(CATALOG_IDS),
        "n_full_layers_with_kv": occupancy["n_full_layers_with_kv"],
        "kv_bytes_all_per_token": KV_BYTES_ALL_PER_TOKEN,
        "storage_kv_bytes_coeff_T": STORAGE_KV_BYTES_COEFF_T,
        "storage_fixed_bytes": STORAGE_FIXED_BYTES,
        "s_bytes_all": S_BYTES_ALL,
        "c_bytes_all": C_BYTES_ALL,
        "n_node_types": len(NODE_TYPE_IDS),
        "node_type_ids": list(NODE_TYPE_IDS),
        "n_node_instances_complete": occupancy["n_node_instances_complete"],
        "n_stage_kinds": len(STAGE_KIND_IDS),
        "stage_kind_ids": list(STAGE_KIND_IDS),
        "n_candidate_recipes": N_CANDIDATE_RECIPES,
        "n_fusion_hypotheses": N_FUSION_HYPOTHESES,
        "n_mappings": N_MAPPINGS,
        "n_sku_unknown_symbols": len(SKU_UNKNOWN_SYMBOLS),
        "sku_unknown_symbols": list(SKU_UNKNOWN_SYMBOLS),
        "n_sensitive_ops": N_SENSITIVE_OPS,
        "n_catalogued_alternatives": N_CATALOGUED_ALTERNATIVES,
        "n_remaining_open_questions": len(REMAINING_OPEN_QUESTION_IDS),
        "remaining_open_question_ids": list(REMAINING_OPEN_QUESTION_IDS),
        "n_remaining_open_questions_closed": 0,
        "n_alternatives_selected": 0,
        "experiments_run": False,
        "nll_measured_here": False,
        "toks_measured_here": False,
        "benchmarks_run": False,
        "payloads_restreamed": False,
        "sku_table_filled": False,
        "pareto_frontier_selected": False,
        "fusion_winner_selected": False,
        "mapping_winner_selected": False,
        "winner_selected_without_measurements": False,
        "alternative_usefulness_label": "HYPOTHESIS",
    }


def _occupancy_from_text_config(text: dict) -> dict:
    """Derive layer counts, occupancy, and cited MAC/byte totals.

    Args:
        text: Sitting ``text_config`` object.

    Returns:
        Occupancy dict used by the summary and fence comparison.

    Raises:
        AssertionError: If required fields are missing or identities fail.
    """

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
    n_full_layers_with_kv = n_full_layers + n_mtp_blocks

    return {
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
        "n_kv_heads": n_kv_heads,
        "n_language_mtp_parameters": n_language_mtp_parameters,
        "weight_bytes_language_mtp_excl_vision": (
            weight_bytes_language_mtp_excl_vision
        ),
        "weight_bytes_unique_non_embed": weight_bytes_unique_non_embed,
        "n_node_instances_complete": n_node_instances_complete,
        "n_full_layers_with_kv": n_full_layers_with_kv,
    }


def _pooled_absmax(fence: dict, key: str) -> float:
    """Return ``fence[key]["absmax"]`` as a float.

    Args:
        fence: Parsed TASK-05 JSON fence.
        key: ``pooled_all``, ``pooled_language_mtp``, or ``pooled_vision``.

    Returns:
        MEASURED absmax cited from the published fence.

    Raises:
        AssertionError: If the nested field is missing or not numeric.
    """

    assert key in fence, f"TASK-05 fence missing {key}"
    pooled = fence[key]
    assert isinstance(pooled, dict), f"TASK-05 {key} is not an object"
    assert "absmax" in pooled, f"TASK-05 {key} missing absmax"
    value = pooled["absmax"]
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"TASK-05 {key}.absmax is not numeric: {value!r}"
    )
    return float(value)


def load_corpus_fences() -> dict[str, dict]:
    """Parse the first JSON fence of every locked corpus document.

    Returns:
        Mapping from corpus path to parsed fence object.

    Raises:
        CleanSheetMismatch: If a corpus file is missing or has no JSON.
        OSError: If a present file cannot be read.
    """

    fences: dict[str, dict] = {}
    differences: list[str] = []
    for path_str in CORPUS_PATHS:
        path = Path(path_str)
        if not path.is_file():
            differences.append(f"corpus file missing: {path_str}")
            continue
        text = path.read_text(encoding="utf-8")
        try:
            fence, _match = first_json_fence(text)
        except (AssertionError, json.JSONDecodeError) as exc:
            differences.append(f"{path_str} json fence: {exc}")
            continue
        fences[path_str] = fence
    if differences:
        raise CleanSheetMismatch(differences)
    return fences


def _compare_present_keys(
    path_str: str,
    fence: dict,
    locked: dict,
) -> list[str]:
    """Compare fence keys that exist in ``locked`` after alias normalization.

    Args:
        path_str: Corpus path for error strings.
        fence: Parsed first JSON fence.
        locked: Comparison locked constants.

    Returns:
        Difference strings; empty when present keys agree.
    """

    differences: list[str] = []
    for raw_key, value in fence.items():
        if raw_key == "authority":
            continue
        if raw_key == "frozen_for_comparative_review":
            continue
        if raw_key == "T_max":
            if isinstance(value, int) and not isinstance(value, bool):
                locked_key = "model_T_max"
            else:
                continue
        else:
            locked_key = KEY_ALIASES.get(raw_key, raw_key)
        if locked_key not in locked:
            continue
        expected = locked[locked_key]
        actual: object = value
        if locked_key == "dtype":
            actual = _normalize_dtype(value)
        if actual != expected:
            differences.append(
                f"{path_str} {raw_key}: {actual!r} != locked {locked_key} "
                f"{expected!r} (planning miss if not a locked alias)"
            )
    for flag in SELECTED_FLAGS_MUST_BE_FALSE:
        if flag in fence and fence[flag] is not False:
            differences.append(
                f"{path_str} {flag}: {fence[flag]!r} != False"
            )
    if "n_alternatives_selected" in fence:
        if fence["n_alternatives_selected"] != 0:
            differences.append(
                f"{path_str} n_alternatives_selected: "
                f"{fence['n_alternatives_selected']!r} != 0"
            )
    return differences


def compare_corpus_fences(fences: dict[str, dict], occupancy: dict) -> None:
    """Fail closed on a non-alias cross-document fence conflict.

    Args:
        fences: Parsed corpus fences keyed by path.
        occupancy: Live occupancy from sitting ``text_config``.

    Raises:
        CleanSheetMismatch: On missing equation tags, unequal TASK-20
            fences, TASK-20 freeze true, or a present-key mismatch.
    """

    differences: list[str] = []
    locked = _comparison_locked(occupancy)

    semantics = Path(SEMANTICS_PATH).read_text(encoding="utf-8")
    for index in range(EQUATION_TAG_START, EQUATION_TAG_END + 1):
        tag = f"\\tag{{{index}}}"
        if tag not in semantics:
            differences.append(f"{SEMANTICS_PATH} missing {tag}")

    for path_str in (WORK_TRAFFIC_PATH, NUMERICAL_PATH):
        text = Path(path_str).read_text(encoding="utf-8")
        if "TASK-02" not in text:
            differences.append(f"{path_str} does not cite TASK-02")
        if "(1)" not in text:
            differences.append(f"{path_str} missing equation citation (1)")

    for path_str, fence in fences.items():
        differences.extend(_compare_present_keys(path_str, fence, locked))

    architecture = fences[ARCHITECTURE_PATH]
    backlog = fences[BACKLOG_PATH]
    if architecture != backlog:
        differences.append(
            f"{ARCHITECTURE_PATH} and {BACKLOG_PATH} first JSON fences "
            "are not deep-equal"
        )
    for path_str in (ARCHITECTURE_PATH, BACKLOG_PATH):
        fence = fences[path_str]
        if fence.get("frozen_for_comparative_review") is not False:
            differences.append(
                f"{path_str} frozen_for_comparative_review is not JSON false"
            )
        if fence.get("n_remaining_open_questions_closed") != 0:
            differences.append(
                f"{path_str} n_remaining_open_questions_closed != 0"
            )
        for flag in SELECTED_FLAGS_MUST_BE_FALSE:
            if flag in fence and fence[flag] is not False:
                differences.append(f"{path_str} {flag} is not false")

    analysis = fences[TENSOR_ANALYSIS_PATH]
    pooled_all = _pooled_absmax(analysis, "pooled_all")
    pooled_language = _pooled_absmax(analysis, "pooled_language_mtp")
    pooled_vision = _pooled_absmax(analysis, "pooled_vision")
    if pooled_all != POOLED_ALL_ABSMAX:
        differences.append(
            f"{TENSOR_ANALYSIS_PATH} pooled_all.absmax {pooled_all!r} "
            f"!= {POOLED_ALL_ABSMAX!r}"
        )
    if pooled_language != POOLED_LANGUAGE_MTP_ABSMAX:
        differences.append(
            f"{TENSOR_ANALYSIS_PATH} pooled_language_mtp.absmax "
            f"{pooled_language!r} != {POOLED_LANGUAGE_MTP_ABSMAX!r}"
        )
    if pooled_vision != POOLED_VISION_ABSMAX:
        differences.append(
            f"{TENSOR_ANALYSIS_PATH} pooled_vision.absmax {pooled_vision!r} "
            f"!= {POOLED_VISION_ABSMAX!r}"
        )

    if differences:
        raise CleanSheetMismatch(differences)


def instantiate_clean_sheet_review(config: dict) -> dict:
    """Compute the TASK-21 clean-sheet-review summary from ``config``.

    Args:
        config: Parsed Transformers ``config.json`` object.

    Returns:
        Ordered summary dict matching the dossier JSON schema.

    Raises:
        AssertionError: If required fields are missing or identities fail.
        CleanSheetMismatch: If a corpus fence is missing or conflicts.
    """

    assert isinstance(config, dict), "config.json root is not an object"
    assert "text_config" in config, "config.json has no text_config"
    text = config["text_config"]
    assert isinstance(text, dict), "text_config is not an object"

    occupancy = _occupancy_from_text_config(text)
    fences = load_corpus_fences()
    compare_corpus_fences(fences, occupancy)
    analysis = fences[TENSOR_ANALYSIS_PATH]
    pooled_all = _pooled_absmax(analysis, "pooled_all")
    pooled_language = _pooled_absmax(analysis, "pooled_language_mtp")
    pooled_vision = _pooled_absmax(analysis, "pooled_vision")
    architecture = fences[ARCHITECTURE_PATH]
    synthesis_frozen_flag = bool(architecture.get("frozen_for_comparative_review"))

    summary = {
        "authority": AUTHORITY,
        "deliverable_review": DELIVERABLE_REVIEW,
        "hidden_size": occupancy["hidden_size"],
        "intermediate_size": occupancy["intermediate_size"],
        "vocab_size": occupancy["vocab_size"],
        "n_decoder_layers": occupancy["n_decoder_layers"],
        "n_linear_layers": occupancy["n_linear_layers"],
        "n_full_layers": occupancy["n_full_layers"],
        "n_mtp_blocks": occupancy["n_mtp_blocks"],
        "full_attention_indices": occupancy["full_attention_indices"],
        "model_T_max": occupancy["model_T_max"],
        "dtype": occupancy["dtype"],
        "mamba_ssm_dtype": occupancy["mamba_ssm_dtype"],
        "n_tensors": N_TENSORS,
        "n_shards": N_SHARDS,
        "n_parameters": N_PARAMETERS,
        "n_bytes": N_BYTES,
        "n_language_mtp_parameters": occupancy["n_language_mtp_parameters"],
        "n_vision_parameters": N_VISION_PARAMETERS,
        "n_language_mtp_tensors": N_LANGUAGE_MTP_TENSORS,
        "n_vision_tensors": N_VISION_TENSORS,
        "weight_bytes_language_mtp_excl_vision": occupancy[
            "weight_bytes_language_mtp_excl_vision"
        ],
        "weight_bytes_unique_non_embed": occupancy[
            "weight_bytes_unique_non_embed"
        ],
        "mac_C_complete": MAC_C_COMPLETE,
        "mac_A_complete": MAC_A_COMPLETE,
        "mac_decode_complete_T1": MAC_DECODE_COMPLETE_T1,
        "mac_prefill_complete_T1": MAC_PREFILL_COMPLETE_T1,
        "mac_decode_complete_T4096": MAC_DECODE_COMPLETE_T4096,
        "mac_prefill_complete_T4096": MAC_PREFILL_COMPLETE_T4096,
        "example_T_values": list(EXAMPLE_T_VALUES),
        "N_w": N_W,
        "N_bank": N_BANK,
        "n_catalog_nodes": N_CATALOG_NODES,
        "catalog_ids": list(CATALOG_IDS),
        "n_full_layers_with_kv": occupancy["n_full_layers_with_kv"],
        "kv_bytes_all_per_token": KV_BYTES_ALL_PER_TOKEN,
        "storage_kv_bytes_coeff_T": STORAGE_KV_BYTES_COEFF_T,
        "storage_fixed_bytes": STORAGE_FIXED_BYTES,
        "s_bytes_all": S_BYTES_ALL,
        "c_bytes_all": C_BYTES_ALL,
        "n_equation_tags": N_EQUATION_TAGS,
        "equation_tag_start": EQUATION_TAG_START,
        "equation_tag_end": EQUATION_TAG_END,
        "n_node_types": len(NODE_TYPE_IDS),
        "node_type_ids": list(NODE_TYPE_IDS),
        "n_node_instances_complete": occupancy["n_node_instances_complete"],
        "n_stage_kinds": len(STAGE_KIND_IDS),
        "stage_kind_ids": list(STAGE_KIND_IDS),
        "n_candidate_recipes": N_CANDIDATE_RECIPES,
        "n_fusion_hypotheses": N_FUSION_HYPOTHESES,
        "n_mappings": N_MAPPINGS,
        "n_sku_unknown_symbols": len(SKU_UNKNOWN_SYMBOLS),
        "sku_unknown_symbols": list(SKU_UNKNOWN_SYMBOLS),
        "n_sensitive_ops": N_SENSITIVE_OPS,
        "n_catalogued_alternatives": N_CATALOGUED_ALTERNATIVES,
        "pooled_all_absmax": pooled_all,
        "pooled_language_mtp_absmax": pooled_language,
        "pooled_vision_absmax": pooled_vision,
        "n_corpus_documents": len(CORPUS_PATHS),
        "corpus_paths": list(CORPUS_PATHS),
        "corpus_task_ids": dict(CORPUS_TASK_IDS),
        "n_check_families": len(CHECK_FAMILY_IDS),
        "check_family_ids": list(CHECK_FAMILY_IDS),
        "n_notation_aliases": len(NOTATION_ALIAS_IDS),
        "notation_alias_ids": list(NOTATION_ALIAS_IDS),
        "n_inconsistencies_found": 0,
        "n_inconsistencies_corrected": 0,
        "inconsistency_ids": [],
        "n_remaining_open_questions": len(REMAINING_OPEN_QUESTION_IDS),
        "remaining_open_question_ids": list(REMAINING_OPEN_QUESTION_IDS),
        "n_remaining_open_questions_closed": 0,
        "n_sibling_checkers": len(SIBLING_CHECKER_SCRIPTS),
        "sibling_checker_scripts": list(SIBLING_CHECKER_SCRIPTS),
        "n_review_headings": len(REVIEW_HEADINGS),
        "review_headings": list(REVIEW_HEADINGS),
        "n_headings": len(REVIEW_HEADINGS),
        "n_diagrams": 1,
        "diagram_ids": list(DIAGRAM_IDS),
        "earlier_task_wins": True,
        "ledger_open_question_unknowns_preserved": True,
        "frozen_for_comparative_review": True,
        "synthesis_frozen_flag": synthesis_frozen_flag,
        "freeze_recorded_in_review_only": True,
        "freeze_authorizes_later_comparative_review": True,
        "freeze_makes_existing_runtimes_design_authority": False,
        "comparative_review_executed_here": False,
        "draft_banners_left_in_place": True,
        "pareto_frontier_selected": False,
        "fusion_winner_selected": False,
        "mapping_winner_selected": False,
        "winner_selected_without_measurements": False,
        "n_alternatives_selected": 0,
        "sku_table_filled": False,
        "keep_source_is_not_quality_winner": True,
        "reconstruction_is_not_quality": True,
        "toks_is_not_quality_axis": True,
        "microbenchmark_cannot_pass_mapping": True,
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
        "alternative_usefulness_label": "HYPOTHESIS",
        "canonical_sentence_logical": CANONICAL_SENTENCE_LOGICAL,
        "canonical_sentence_review": CANONICAL_SENTENCE_REVIEW,
        "canonical_sentence_unknowns": CANONICAL_SENTENCE_UNKNOWNS,
        "canonical_sentence_freeze": CANONICAL_SENTENCE_FREEZE,
        "review_question_sentence": REVIEW_QUESTION_SENTENCE,
        "freeze_non_authority_sentence": FREEZE_NON_AUTHORITY_SENTENCE,
    }
    assert list(summary) == list(SCHEMA_KEYS), "summary key order drifted"
    return summary


def assert_summary(summary: dict) -> None:
    """Run locked internal asserts on a live review summary.

    Args:
        summary: Object from :func:`instantiate_clean_sheet_review`.

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
    assert summary["n_tensors"] == 1199
    assert summary["n_parameters"] == 27781427952
    assert summary["n_bytes"] == 55562855904
    assert summary["n_language_mtp_parameters"] == N_LANGUAGE_MTP_PARAMETERS
    assert summary["n_vision_parameters"] == N_VISION_PARAMETERS
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
    assert summary["n_catalog_nodes"] == 52
    assert len(summary["catalog_ids"]) == 52
    assert summary["catalog_ids"] == list(CATALOG_IDS)
    assert summary["kv_bytes_all_per_token"] == 69632
    assert summary["storage_fixed_bytes"] == 153944064
    assert summary["n_equation_tags"] == 24
    assert summary["equation_tag_start"] == 1
    assert summary["equation_tag_end"] == 24
    assert summary["n_node_types"] == 6
    assert summary["n_node_instances_complete"] == 135
    assert summary["n_stage_kinds"] == 9
    assert summary["n_candidate_recipes"] == 22
    assert summary["n_fusion_hypotheses"] == 22
    assert summary["n_mappings"] == 18
    assert summary["n_corpus_documents"] == 21
    assert summary["n_check_families"] == 6
    assert summary["n_notation_aliases"] == 7
    assert summary["n_inconsistencies_found"] == 0
    assert summary["n_inconsistencies_corrected"] == 0
    assert summary["inconsistency_ids"] == []
    assert summary["n_remaining_open_questions"] == 12
    assert summary["n_remaining_open_questions_closed"] == 0
    assert summary["n_sibling_checkers"] == 19
    assert summary["n_review_headings"] == 8
    assert summary["n_diagrams"] == 1
    assert summary["pooled_language_mtp_absmax"] == 19.25
    assert summary["pooled_all_absmax"] == 25.5
    assert summary["pooled_vision_absmax"] == 25.5
    assert summary["frozen_for_comparative_review"] is True
    assert summary["synthesis_frozen_flag"] is False
    assert summary["freeze_recorded_in_review_only"] is True
    assert summary["freeze_makes_existing_runtimes_design_authority"] is False
    assert summary["comparative_review_executed_here"] is False
    assert summary["ledger_open_question_unknowns_preserved"] is True
    assert summary["mapping_winner_selected"] is False
    assert summary["pareto_frontier_selected"] is False
    assert summary["fusion_winner_selected"] is False
    assert summary["experiments_run"] is False
    assert summary["nll_measured_here"] is False
    assert summary["toks_measured_here"] is False
    assert summary["benchmarks_run"] is False
    assert summary["sku_table_filled"] is False
    assert summary["payloads_restreamed"] is False
    assert summary["gguf_payload_inspected"] is False
    assert summary["quartz_inspected"] is False
    assert summary["llama_inspected"] is False
    assert summary["device_query_run"] is False
    assert summary["nsight_run"] is False
    assert summary["vision_eval_deferred"] is True
    assert summary["n_alternatives_selected"] == 0
    assert summary["n_catalogued_alternatives"] == 125
    assert summary["n_full_layers_with_kv"] == 17
    assert summary["node_type_ids"] == list(NODE_TYPE_IDS)
    assert summary["stage_kind_ids"] == list(STAGE_KIND_IDS)
    assert summary["sku_unknown_symbols"] == list(SKU_UNKNOWN_SYMBOLS)
    assert summary["remaining_open_question_ids"] == list(
        REMAINING_OPEN_QUESTION_IDS
    )
    assert summary["notation_alias_ids"] == list(NOTATION_ALIAS_IDS)
    assert summary["check_family_ids"] == list(CHECK_FAMILY_IDS)
    assert summary["corpus_paths"] == list(CORPUS_PATHS)
    assert summary["sibling_checker_scripts"] == list(SIBLING_CHECKER_SCRIPTS)
    assert summary["review_headings"] == list(REVIEW_HEADINGS)
    assert summary["diagram_ids"] == list(DIAGRAM_IDS)
    assert summary["n_headings"] == 8
    assert summary["earlier_task_wins"] is True
    assert summary["draft_banners_left_in_place"] is True
    assert summary["alternative_usefulness_label"] == "HYPOTHESIS"


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
    """Compare documented JSON against a live review object.

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


def check_review(live: dict, design_path: Path) -> None:
    """Check the review markdown file against a live summary.

    Args:
        live: Review summary object from sitting config plus corpus fences.
        design_path: Path to ``clean-sheet-review.md``.

    Raises:
        CleanSheetMismatch: On heading, JSON, diagram, or token mismatches.
        OSError: If the file cannot be read.
    """

    text = design_path.read_text(encoding="utf-8")
    differences: list[str] = []

    headings = HEADING_RE.findall(text)
    if headings != list(REVIEW_HEADINGS):
        differences.append(
            "review heading mismatch:\n"
            f"  documented: {headings}\n"
            f"  required:   {list(REVIEW_HEADINGS)}"
        )

    fence_match: re.Match[str] | None = None
    try:
        documented, fence_match = first_json_fence(text)
    except (AssertionError, json.JSONDecodeError) as exc:
        differences.append(f"review json fence: {exc}")
        documented = None

    if documented is not None:
        differences.extend(diff_summary(live, documented))

    for match in FORBIDDEN_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        differences.append(f"review line {line}: forbidden token {match.group(0)!r}")

    try:
        vis_start, vis_end = _deferred_vision_span(text)
        if not UNKNOWN_RE.search(text, vis_start, vis_end):
            differences.append("review: word UNKNOWN missing in Deferred vision")
    except AssertionError as exc:
        differences.append(f"review: {exc}")

    for number in LOCKED_DOCUMENT_INTEGERS:
        token = str(number)
        if token not in text:
            differences.append(
                f"review: locked number {token} missing as decimal substring"
            )

    mermaid_fences = list(MERMAID_FENCE_RE.finditer(text))
    if len(mermaid_fences) != 1:
        differences.append(f"review mermaid fence count {len(mermaid_fences)} != 1")
    else:
        match = mermaid_fences[0]
        body = match.group(1)
        stripped = body.lstrip()
        if not (
            stripped.startswith("flowchart TB")
            or stripped.startswith("flowchart LR")
        ):
            differences.append(
                "review mermaid fence body does not start with "
                "flowchart TB or flowchart LR"
            )
        if "flowchart" not in body:
            differences.append("review mermaid fence does not contain flowchart")
        missing_ids = [
            node_id for node_id in DIAGRAM_MERMAID_IDS if node_id not in body
        ]
        if missing_ids:
            differences.append(f"review diagram missing node ids {missing_ids}")
        heading = re.search(
            r"^## Cross-document findings\s*$",
            text,
            flags=re.MULTILINE,
        )
        next_heading = re.search(
            r"^## Preserved unknowns\s*$", text, flags=re.MULTILINE
        )
        if heading is None or next_heading is None:
            differences.append(
                "missing Cross-document findings or Preserved unknowns"
            )
        elif not (heading.end() <= match.start() < next_heading.start()):
            differences.append(
                "mermaid fence is not under ## Cross-document findings"
            )

    review_sentences = (
        CANONICAL_SENTENCE_LOGICAL,
        CANONICAL_SENTENCE_REVIEW,
        CANONICAL_SENTENCE_UNKNOWNS,
        CANONICAL_SENTENCE_FREEZE,
        REVIEW_QUESTION_SENTENCE,
        FREEZE_NON_AUTHORITY_SENTENCE,
    )
    for label, sentence in (
        ("canonical_sentence_logical", CANONICAL_SENTENCE_LOGICAL),
        ("canonical_sentence_review", CANONICAL_SENTENCE_REVIEW),
        ("canonical_sentence_unknowns", CANONICAL_SENTENCE_UNKNOWNS),
        ("canonical_sentence_freeze", CANONICAL_SENTENCE_FREEZE),
        ("review_question_sentence", REVIEW_QUESTION_SENTENCE),
        ("freeze_non_authority_sentence", FREEZE_NON_AUTHORITY_SENTENCE),
    ):
        if sentence not in text:
            differences.append(f"review {label} not present verbatim")

    convention = re.search(
        r"^## Review convention\s*$", text, flags=re.MULTILINE
    )
    corpus_heading = re.search(
        r"^## Corpus and check families\s*$",
        text,
        flags=re.MULTILINE,
    )
    if convention is None or corpus_heading is None:
        differences.append(
            "missing Review convention or Corpus and check families"
        )
    else:
        section = text[convention.end() : corpus_heading.start()]
        for label, sentence in (
            ("canonical_sentence_logical", CANONICAL_SENTENCE_LOGICAL),
            ("canonical_sentence_review", CANONICAL_SENTENCE_REVIEW),
            ("canonical_sentence_unknowns", CANONICAL_SENTENCE_UNKNOWNS),
            ("canonical_sentence_freeze", CANONICAL_SENTENCE_FREEZE),
            ("review_question_sentence", REVIEW_QUESTION_SENTENCE),
        ):
            if sentence not in section:
                differences.append(
                    f"review {label} missing under Review convention"
                )

    freeze_heading = re.search(r"^## Freeze\s*$", text, flags=re.MULTILINE)
    vision_heading = re.search(
        r"^## Deferred vision\s*$", text, flags=re.MULTILINE
    )
    if freeze_heading is None or vision_heading is None:
        differences.append("missing Freeze or Deferred vision")
    else:
        freeze_section = text[freeze_heading.end() : vision_heading.start()]
        if "FROZEN_FOR_COMPARATIVE_REVIEW" not in freeze_section:
            differences.append(
                "review Freeze section missing FROZEN_FOR_COMPARATIVE_REVIEW"
            )
        if FREEZE_NON_AUTHORITY_SENTENCE not in freeze_section:
            differences.append(
                "review freeze_non_authority_sentence missing under Freeze"
            )

    for label, ids in (
        ("corpus basename", tuple(Path(path).name for path in CORPUS_PATHS)),
        ("check family", CHECK_FAMILY_IDS),
        ("notation alias", NOTATION_ALIAS_IDS),
        ("remaining open question", REMAINING_OPEN_QUESTION_IDS),
        ("catalog", CATALOG_IDS),
        ("node type", NODE_TYPE_IDS),
        ("stage kind", STAGE_KIND_IDS),
        ("sku unknown symbol", SKU_UNKNOWN_SYMBOLS),
    ):
        for item_id in ids:
            if item_id not in text:
                differences.append(
                    f"review {label} {item_id!r} missing as substring"
                )

    if "HYPOTHESIS" not in text:
        differences.append("review word HYPOTHESIS missing")
    if "FROZEN_FOR_COMPARATIVE_REVIEW" not in text:
        differences.append("review token FROZEN_FOR_COMPARATIVE_REVIEW missing")

    differences.extend(
        _collect_forbidden_phrase_diffs(text, fence_match, review_sentences)
    )

    if differences:
        raise CleanSheetMismatch(differences)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the clean-sheet-review checker.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Instantiate Qwen3.8-27B language+MTP clean-sheet review "
            "summary from text_config and corpus JSON fences, and check "
            "docs/architecture/clean-sheet-review.md."
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
        help="Write clean-sheet-review summary JSON to stdout.",
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=None,
        help="Review markdown whose first json fence must match live summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the clean-sheet-review checker CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch, 2 missing
        config.
    """

    args = _parse_args(argv)
    review_path = args.review
    print_json = args.json
    if review_path is None and not args.json:
        review_path = Path(DELIVERABLE_REVIEW)
    try:
        config = load_config(args.config)
        text_config = config["text_config"]
        assert isinstance(text_config, dict), "text_config is not an object"
        assert _require_str(text_config, "dtype") == "bfloat16"
        assert _require_str(text_config, "mamba_ssm_dtype") == "float32"
        summary = instantiate_clean_sheet_review(config)
        assert_summary(summary)
        if review_path is not None:
            if not review_path.is_file():
                print(f"review file not found: {review_path}", file=sys.stderr)
                return 1
            check_review(summary, review_path)
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
