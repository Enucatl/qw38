#!/usr/bin/env python3
"""Check Qwen3.8-27B numerical-sensitivity analysis against sitting text_config.

Reads ``text_config`` from a Transformers ``config.json`` (no safetensor
payloads) and either prints the numerical-sensitivity summary object or checks
that ``docs/architecture/numerical-sensitivity.md`` matches it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

AUTHORITY = ".cache/authorities/qwen3.8-27b-transformers"

CANONICAL_SENTENCE_LOGICAL = (
    "Logical values in this document are graph nodes. They do not imply "
    "physical allocation, materialization, buffer reuse, or kernel fusion."
)

CANONICAL_SENTENCE_HYPOTHESIS = (
    "Risk classifications in this document are hypotheses, not measurements."
)

CANONICAL_SENTENCE_DTYPE = (
    "Config dtypes name conceptual element types for parameters and "
    "persistent state; they are not CUDA accumulation or kernel dtypes."
)

PRECISION_ROLES: tuple[str, ...] = ("param", "activation", "accum", "state")

RISK_MECHANISMS: tuple[str, ...] = (
    "long_sum",
    "normalize",
    "exp_range",
    "recurrent",
    "roundtrip",
    "residual_add",
    "narrow_element",
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

SENSITIVE_OP_ROLES: tuple[str, ...] = (
    "param",
    "activation",
    "activation",
    "activation",
    "accum",
    "accum",
    "accum",
    "accum",
    "accum",
    "accum",
    "accum",
    "accum",
    "accum",
    "accum",
    "accum",
    "activation",
    "state",
    "state",
    "state",
    "accum",
)

SENSITIVE_OP_MECHANISMS: tuple[str, ...] = (
    "narrow_element",
    "residual_add",
    "narrow_element",
    "exp_range",
    "normalize",
    "normalize",
    "normalize",
    "exp_range",
    "long_sum",
    "long_sum",
    "long_sum",
    "long_sum",
    "recurrent",
    "long_sum",
    "exp_range",
    "exp_range",
    "roundtrip",
    "roundtrip",
    "narrow_element",
    "long_sum",
)

SENSITIVE_OP_SEVERITIES: tuple[str, ...] = (
    "medium",
    "high",
    "medium",
    "low",
    "high",
    "medium",
    "medium",
    "high",
    "high",
    "medium",
    "medium",
    "medium",
    "high",
    "medium",
    "high",
    "high",
    "high",
    "low",
    "high",
    "low",
)

HIGH_RISK_IDS: tuple[str, ...] = (
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

MEDIUM_RISK_IDS: tuple[str, ...] = (
    "param_bf16",
    "live_across_gates",
    "rms_head",
    "l2_gdn",
    "gemm_k5120",
    "gemm_k17408",
    "gemm_lm_head",
    "gdn_inner_d128",
)

LOW_RISK_IDS: tuple[str, ...] = (
    "silu_sigmoid",
    "state_c_bf16",
    "conv_fir",
)

STATE_IDS: tuple[str, ...] = ("K_state", "V_state", "C_state", "S")

DIAGRAM_REQUIRED_IDS: tuple[str, ...] = (
    "param",
    "activation",
    "accum",
    "state",
    "S",
    "KV",
    "rms",
    "softmax",
    "gdn",
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
BF16_MANTISSA_BITS = 7
BF16_EXP_BITS = 8
F32_MANTISSA_BITS = 23
F32_EXP_BITS = 8
BF16_ULP_AT_1 = 0.0078125
SOFTMAX_SCALE = 0.0625

LOCKED_DOCUMENT_NUMBERS: tuple[int | float, ...] = (
    5120,
    17408,
    248320,
    256,
    128,
    262144,
    10000000,
    0.0078125,
    0.0625,
    129,
    134,
    130,
    69632,
    2949120,
    150994944,
    786432,
    10240,
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
    "linear_qkv_width",
    "linear_conv_kernel_dim",
    "linear_conv_delay",
    "bytes_bf16",
    "bytes_f32",
    "bf16_mantissa_bits",
    "bf16_exp_bits",
    "f32_mantissa_bits",
    "f32_exp_bits",
    "bf16_ulp_at_1",
    "eps",
    "eps_lt_bf16_ulp_at_1",
    "softmax_scale",
    "rope_theta",
    "T_max",
    "rope_phase_at_T_max_j0",
    "T_is_stored_length_after_append",
    "decode_T_new",
    "primary_includes_mtp",
    "example_T",
    "param_dtype",
    "kv_conceptual_dtype",
    "c_conceptual_dtype",
    "s_conceptual_dtype",
    "mamba_ssm_dtype_is_not_activation_dtype",
    "activation_dtype_decided",
    "gdn_primary_is_recurrent_eq_17",
    "chunkwise_fp_gap_is_hypothesis",
    "reduction_rms_hidden",
    "reduction_rms_qk",
    "reduction_rms_gdn",
    "reduction_l2_gdn",
    "reduction_gemm_hidden",
    "reduction_gemm_mlp_down",
    "reduction_lm_head_k",
    "reduction_lm_head_outputs",
    "reduction_conv",
    "reduction_gdn_inner",
    "reduction_attn_over_T_max",
    "reduction_attn_over_T_at_example_T",
    "n_residual_adds_language",
    "n_residual_adds_complete",
    "n_rms_hidden_language",
    "n_rms_hidden_complete",
    "n_rms_qk",
    "n_rms_gdn",
    "kv_bytes_all_per_token",
    "c_bytes_all",
    "s_bytes_all",
    "s_elems_per_layer",
    "precision_roles",
    "risk_mechanisms",
    "sensitive_ops",
    "sensitive_op_roles",
    "sensitive_op_mechanisms",
    "sensitive_op_severities",
    "high_risk_ids",
    "medium_risk_ids",
    "low_risk_ids",
    "n_sensitive_ops",
    "n_high_risk",
    "n_medium_risk",
    "n_low_risk",
    "state_ids",
    "diagram_ids",
    "n_diagrams",
    "canonical_sentence_logical",
    "canonical_sentence_hypothesis",
    "canonical_sentence_dtype",
)

REQUIRED_HEADINGS: tuple[str, ...] = (
    "Authority",
    "Precision roles",
    "IEEE widths and config dtypes",
    "Weights",
    "Activations",
    "Reductions and accumulation paths",
    "Persistent state",
    "Risk classification",
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


class NumericalSensitivityMismatch(Exception):
    """Raised when the numerical-sensitivity markdown fails a content check."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def dumps_summary(summary: dict) -> str:
    """Pretty-print the numerical-sensitivity summary object.

    Args:
        summary: Object produced by :func:`instantiate_numerical_sensitivity`.

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


def _require_float(mapping: dict, key: str) -> float:
    """Return ``mapping[key]`` as a ``float``.

    Args:
        mapping: Config object.
        key: Required field name.

    Returns:
        Floating-point field value.

    Raises:
        AssertionError: If the field is missing or not a real number.
    """

    assert key in mapping, f"missing text_config field: {key}"
    value = mapping[key]
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"text_config.{key} is not a number: {value!r}"
    )
    return float(value)


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


def instantiate_numerical_sensitivity(config: dict) -> dict:
    """Compute the TASK-07 numerical-sensitivity summary from ``config``.

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
    eps = _require_float(text, "rms_norm_eps")
    param_dtype = _require_str(text, "dtype")
    mamba_ssm_dtype = _require_str(text, "mamba_ssm_dtype")

    assert "rope_parameters" in text, "missing rope_parameters"
    rope = text["rope_parameters"]
    assert isinstance(rope, dict), "rope_parameters is not an object"
    rope_theta = _require_int(rope, "rope_theta")

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
    linear_conv_delay = linear_conv_kernel_dim - 1

    softmax_scale = 1.0 / (head_dim ** 0.5)
    rope_phase_at_t_max_j0 = t_max * 1
    eps_lt_bf16_ulp_at_1 = eps < BF16_ULP_AT_1
    example_t = list(EXAMPLE_T)

    n_residual_adds_language = n_decoder_layers * 2
    n_residual_adds_complete = n_residual_adds_language + 2 * n_mtp_blocks
    n_rms_hidden_language = n_decoder_layers * 2 + 1
    n_rms_hidden_complete = n_rms_hidden_language + 5 * n_mtp_blocks
    n_rms_qk = n_full_layers_with_kv * 2
    n_rms_gdn = n_linear_layers

    kv_elems_per_full_layer_per_token = 2 * n_kv_heads * head_dim
    kv_bytes_all_per_token = (
        n_full_layers_with_kv * kv_elems_per_full_layer_per_token * BYTES_BF16
    )
    c_bytes_all = (
        n_linear_layers * linear_conv_delay * linear_qkv_width * BYTES_BF16
    )
    s_elems_per_layer = (
        linear_num_value_heads * linear_key_head_dim * linear_value_head_dim
    )
    s_bytes_all = n_linear_layers * s_elems_per_layer * BYTES_F32

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
        "linear_qkv_width": linear_qkv_width,
        "linear_conv_kernel_dim": linear_conv_kernel_dim,
        "linear_conv_delay": linear_conv_delay,
        "bytes_bf16": BYTES_BF16,
        "bytes_f32": BYTES_F32,
        "bf16_mantissa_bits": BF16_MANTISSA_BITS,
        "bf16_exp_bits": BF16_EXP_BITS,
        "f32_mantissa_bits": F32_MANTISSA_BITS,
        "f32_exp_bits": F32_EXP_BITS,
        "bf16_ulp_at_1": BF16_ULP_AT_1,
        "eps": eps,
        "eps_lt_bf16_ulp_at_1": eps_lt_bf16_ulp_at_1,
        "softmax_scale": softmax_scale,
        "rope_theta": rope_theta,
        "T_max": t_max,
        "rope_phase_at_T_max_j0": rope_phase_at_t_max_j0,
        "T_is_stored_length_after_append": True,
        "decode_T_new": 1,
        "primary_includes_mtp": True,
        "example_T": example_t,
        "param_dtype": param_dtype,
        "kv_conceptual_dtype": "bfloat16",
        "c_conceptual_dtype": "bfloat16",
        "s_conceptual_dtype": mamba_ssm_dtype,
        "mamba_ssm_dtype_is_not_activation_dtype": True,
        "activation_dtype_decided": False,
        "gdn_primary_is_recurrent_eq_17": True,
        "chunkwise_fp_gap_is_hypothesis": True,
        "reduction_rms_hidden": hidden_size,
        "reduction_rms_qk": head_dim,
        "reduction_rms_gdn": linear_value_head_dim,
        "reduction_l2_gdn": linear_key_head_dim,
        "reduction_gemm_hidden": hidden_size,
        "reduction_gemm_mlp_down": intermediate_size,
        "reduction_lm_head_k": hidden_size,
        "reduction_lm_head_outputs": vocab_size,
        "reduction_conv": linear_conv_kernel_dim,
        "reduction_gdn_inner": linear_key_head_dim,
        "reduction_attn_over_T_max": t_max,
        "reduction_attn_over_T_at_example_T": list(example_t),
        "n_residual_adds_language": n_residual_adds_language,
        "n_residual_adds_complete": n_residual_adds_complete,
        "n_rms_hidden_language": n_rms_hidden_language,
        "n_rms_hidden_complete": n_rms_hidden_complete,
        "n_rms_qk": n_rms_qk,
        "n_rms_gdn": n_rms_gdn,
        "kv_bytes_all_per_token": kv_bytes_all_per_token,
        "c_bytes_all": c_bytes_all,
        "s_bytes_all": s_bytes_all,
        "s_elems_per_layer": s_elems_per_layer,
        "precision_roles": list(PRECISION_ROLES),
        "risk_mechanisms": list(RISK_MECHANISMS),
        "sensitive_ops": list(SENSITIVE_OPS),
        "sensitive_op_roles": list(SENSITIVE_OP_ROLES),
        "sensitive_op_mechanisms": list(SENSITIVE_OP_MECHANISMS),
        "sensitive_op_severities": list(SENSITIVE_OP_SEVERITIES),
        "high_risk_ids": list(HIGH_RISK_IDS),
        "medium_risk_ids": list(MEDIUM_RISK_IDS),
        "low_risk_ids": list(LOW_RISK_IDS),
        "n_sensitive_ops": len(SENSITIVE_OPS),
        "n_high_risk": len(HIGH_RISK_IDS),
        "n_medium_risk": len(MEDIUM_RISK_IDS),
        "n_low_risk": len(LOW_RISK_IDS),
        "state_ids": list(STATE_IDS),
        "diagram_ids": list(DIAGRAM_REQUIRED_IDS),
        "n_diagrams": 1,
        "canonical_sentence_logical": CANONICAL_SENTENCE_LOGICAL,
        "canonical_sentence_hypothesis": CANONICAL_SENTENCE_HYPOTHESIS,
        "canonical_sentence_dtype": CANONICAL_SENTENCE_DTYPE,
    }
    _assert_identities(summary, text)
    return summary


def _assert_identities(summary: dict, text: dict) -> None:
    """Assert dossier identities on a live numerical-sensitivity object.

    Args:
        summary: Object produced by :func:`instantiate_numerical_sensitivity`.
        text: The ``text_config`` mapping used to build ``summary``.
    """

    assert list(summary) == list(SCHEMA_KEYS), (
        f"schema key order mismatch: {list(summary)} vs {list(SCHEMA_KEYS)}"
    )
    assert summary["authority"] == AUTHORITY
    assert summary["n_linear_layers"] == 48
    assert summary["n_full_layers"] == 16
    assert summary["n_mtp_blocks"] == 1
    assert summary["n_full_layers_with_kv"] == (
        summary["n_full_layers"] + summary["n_mtp_blocks"]
    )
    assert summary["n_full_layers_with_kv"] == 17
    assert summary["full_attention_indices"] == list(LOCKED_FULL_ATTENTION_INDICES)
    assert text.get("dtype") == "bfloat16"
    assert text.get("mamba_ssm_dtype") == "float32"
    assert summary["param_dtype"] == "bfloat16"
    assert summary["s_conceptual_dtype"] == "float32"
    assert summary["kv_conceptual_dtype"] == "bfloat16"
    assert summary["c_conceptual_dtype"] == "bfloat16"
    assert summary["mamba_ssm_dtype_is_not_activation_dtype"] is True
    assert summary["activation_dtype_decided"] is False
    assert summary["bf16_mantissa_bits"] == 7
    assert summary["f32_mantissa_bits"] == 23
    assert summary["bf16_exp_bits"] == 8
    assert summary["f32_exp_bits"] == 8
    assert summary["bf16_ulp_at_1"] == 0.0078125
    assert summary["bytes_bf16"] == 2
    assert summary["bytes_f32"] == 4
    assert summary["eps_lt_bf16_ulp_at_1"] is True
    assert summary["eps"] < summary["bf16_ulp_at_1"]
    assert summary["head_dim"] == 256
    assert 1.0 / (summary["head_dim"] ** 0.5) == 0.0625
    assert summary["softmax_scale"] == 0.0625
    assert summary["reduction_rms_hidden"] == summary["hidden_size"]
    assert summary["hidden_size"] == 5120
    assert summary["reduction_rms_hidden"] == 5120
    assert summary["reduction_gemm_mlp_down"] == 17408
    assert summary["reduction_gemm_mlp_down"] == summary["intermediate_size"]
    assert summary["reduction_lm_head_outputs"] == 248320
    assert summary["reduction_lm_head_outputs"] == summary["vocab_size"]
    assert summary["reduction_attn_over_T_max"] == 262144
    assert summary["reduction_attn_over_T_max"] == summary["T_max"]
    assert summary["T_max"] == text["max_position_embeddings"]
    assert summary["rope_phase_at_T_max_j0"] == 262144
    assert summary["rope_phase_at_T_max_j0"] == summary["T_max"]
    assert summary["n_residual_adds_language"] == 128
    assert summary["n_residual_adds_complete"] == 130
    assert summary["n_rms_hidden_language"] == 129
    assert summary["n_rms_hidden_complete"] == 134
    assert summary["n_rms_qk"] == 34
    assert summary["n_rms_gdn"] == 48
    assert summary["n_sensitive_ops"] == 20
    assert summary["n_high_risk"] == 9
    assert summary["n_medium_risk"] == 8
    assert summary["n_low_risk"] == 3
    assert summary["sensitive_ops"] == list(SENSITIVE_OPS)
    assert summary["sensitive_op_roles"] == list(SENSITIVE_OP_ROLES)
    assert summary["sensitive_op_mechanisms"] == list(SENSITIVE_OP_MECHANISMS)
    assert summary["sensitive_op_severities"] == list(SENSITIVE_OP_SEVERITIES)
    assert summary["high_risk_ids"] == list(HIGH_RISK_IDS)
    assert summary["medium_risk_ids"] == list(MEDIUM_RISK_IDS)
    assert summary["low_risk_ids"] == list(LOW_RISK_IDS)
    assert len(summary["sensitive_ops"]) == summary["n_sensitive_ops"]
    assert len(summary["sensitive_op_roles"]) == summary["n_sensitive_ops"]
    assert len(summary["sensitive_op_mechanisms"]) == summary["n_sensitive_ops"]
    assert len(summary["sensitive_op_severities"]) == summary["n_sensitive_ops"]
    assert summary["gdn_primary_is_recurrent_eq_17"] is True
    assert summary["chunkwise_fp_gap_is_hypothesis"] is True
    assert summary["primary_includes_mtp"] is True
    assert summary["kv_bytes_all_per_token"] == 69632
    assert summary["kv_bytes_all_per_token"] == (
        summary["n_full_layers_with_kv"]
        * 2
        * summary["n_kv_heads"]
        * summary["head_dim"]
        * summary["bytes_bf16"]
    )
    assert summary["c_bytes_all"] == 2949120
    assert summary["c_bytes_all"] == (
        summary["n_linear_layers"]
        * summary["linear_conv_delay"]
        * summary["linear_qkv_width"]
        * summary["bytes_bf16"]
    )
    assert summary["s_elems_per_layer"] == 786432
    assert summary["s_elems_per_layer"] == (
        summary["linear_num_value_heads"]
        * summary["linear_key_head_dim"]
        * summary["linear_value_head_dim"]
    )
    assert summary["s_bytes_all"] == 150994944
    assert summary["s_bytes_all"] == (
        summary["n_linear_layers"]
        * summary["s_elems_per_layer"]
        * summary["bytes_f32"]
    )
    assert summary["n_diagrams"] == 1
    assert summary["diagram_ids"] == list(DIAGRAM_REQUIRED_IDS)
    assert summary["state_ids"] == list(STATE_IDS)
    assert summary["precision_roles"] == list(PRECISION_ROLES)
    assert summary["risk_mechanisms"] == list(RISK_MECHANISMS)
    assert summary["example_T"] == list(EXAMPLE_T)
    assert summary["reduction_attn_over_T_at_example_T"] == list(EXAMPLE_T)
    assert summary["T_is_stored_length_after_append"] is True
    assert summary["decode_T_new"] == 1
    assert summary["canonical_sentence_logical"] == CANONICAL_SENTENCE_LOGICAL
    assert summary["canonical_sentence_hypothesis"] == CANONICAL_SENTENCE_HYPOTHESIS
    assert summary["canonical_sentence_dtype"] == CANONICAL_SENTENCE_DTYPE
    assert summary["reduction_rms_qk"] == summary["head_dim"]
    assert summary["reduction_rms_gdn"] == summary["linear_value_head_dim"]
    assert summary["reduction_l2_gdn"] == summary["linear_key_head_dim"]
    assert summary["reduction_gemm_hidden"] == summary["hidden_size"]
    assert summary["reduction_lm_head_k"] == summary["hidden_size"]
    assert summary["reduction_conv"] == summary["linear_conv_kernel_dim"]
    assert summary["reduction_gdn_inner"] == summary["linear_key_head_dim"]
    high_from_ops = [
        op_id
        for op_id, severity in zip(
            summary["sensitive_ops"], summary["sensitive_op_severities"]
        )
        if severity == "high"
    ]
    medium_from_ops = [
        op_id
        for op_id, severity in zip(
            summary["sensitive_ops"], summary["sensitive_op_severities"]
        )
        if severity == "medium"
    ]
    low_from_ops = [
        op_id
        for op_id, severity in zip(
            summary["sensitive_ops"], summary["sensitive_op_severities"]
        )
        if severity == "low"
    ]
    assert high_from_ops == summary["high_risk_ids"]
    assert medium_from_ops == summary["medium_risk_ids"]
    assert low_from_ops == summary["low_risk_ids"]


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
        text = config_path.read_text(encoding="utf-8")
        payload = json.loads(text)
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
    """Compare documented JSON against a live numerical-sensitivity object.

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


def check_numerical_sensitivity(live: dict, numerical_path: Path) -> None:
    """Check a numerical-sensitivity markdown file against a live summary.

    Args:
        live: Numerical-sensitivity summary object from sitting config.
        numerical_path: Path to ``numerical-sensitivity.md``.

    Raises:
        NumericalSensitivityMismatch: On heading, JSON, diagram, or token
            mismatches.
        OSError: If the file cannot be read.
    """

    text = numerical_path.read_text(encoding="utf-8")
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
    if CANONICAL_SENTENCE_HYPOTHESIS not in text:
        differences.append("canonical_sentence_hypothesis not present verbatim")
    if CANONICAL_SENTENCE_DTYPE not in text:
        differences.append("canonical_sentence_dtype not present verbatim")

    for op_id in SENSITIVE_OPS:
        if op_id not in text:
            differences.append(f"sensitive op {op_id!r} missing as substring")

    for role in PRECISION_ROLES:
        if role not in text:
            differences.append(f"precision role {role!r} missing as substring")

    for mechanism in RISK_MECHANISMS:
        if mechanism not in text:
            differences.append(
                f"risk mechanism {mechanism!r} missing as substring"
            )

    if "HYPOTHESIS" not in text:
        differences.append("word HYPOTHESIS missing")

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

    for number in LOCKED_DOCUMENT_NUMBERS:
        token = str(number)
        if token not in text:
            differences.append(
                f"locked number {token} missing as decimal substring"
            )

    if differences:
        raise NumericalSensitivityMismatch(differences)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the numerical-sensitivity checker."""

    parser = argparse.ArgumentParser(
        description=(
            "Instantiate Qwen3.8-27B language+MTP numerical-sensitivity summary "
            "from text_config and check "
            "docs/architecture/numerical-sensitivity.md."
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
        help="Write numerical-sensitivity summary JSON to stdout.",
    )
    parser.add_argument(
        "--numerical-sensitivity",
        type=Path,
        default=None,
        help="Markdown path whose first json fence must match live summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the numerical-sensitivity checker CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch, 2 missing
        config.
    """

    args = _parse_args(argv)
    try:
        config = load_config(args.config)
        summary = instantiate_numerical_sensitivity(config)
        if args.numerical_sensitivity is not None:
            if not args.numerical_sensitivity.is_file():
                print(
                    f"numerical-sensitivity file not found: "
                    f"{args.numerical_sensitivity}",
                    file=sys.stderr,
                )
                return 1
            check_numerical_sensitivity(summary, args.numerical_sensitivity)
    except MissingConfig as exc:
        print(str(exc.path), file=sys.stderr)
        return 2
    except NumericalSensitivityMismatch as exc:
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

    if args.json or args.numerical_sensitivity is None:
        sys.stdout.write(dumps_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
