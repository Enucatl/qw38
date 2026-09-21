#!/usr/bin/env python3
"""Check Qwen3.8-27B work and traffic bounds against sitting text_config.

Reads ``text_config`` from a Transformers ``config.json`` (no safetensor
payloads) and either prints the work-and-traffic summary object or checks that
``docs/architecture/work-and-traffic.md`` matches it.
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
    "Bottleneck classifications in this document are hypotheses, not measurements."
)

CANONICAL_SENTENCE_ACTIVATION = (
    "Activation traffic is a DERIVED minimum materialization from catalog ranks, "
    "not a CUDA fusion or buffer-reuse claim."
)

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

BOTTLENECK_LABELS: tuple[str, ...] = (
    "weight_memory",
    "vocab_memory",
    "state_memory",
    "kv_memory",
    "quadratic_attn",
    "compute",
)

REGION_IDS: tuple[str, ...] = (
    "embed",
    "linear_attn",
    "full_attn",
    "mlp",
    "lm_head",
    "mtp",
)

DIAGRAM_REQUIRED_IDS: tuple[str, ...] = (
    "embed",
    "linear_attn",
    "full_attn",
    "mlp",
    "lm_head",
    "mtp",
    "weight",
    "state",
    "activation",
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

BYTES_F32 = 4

LOCKED_DOCUMENT_INTEGERS: tuple[int, ...] = (
    104857600,
    118235136,
    267386880,
    1271398400,
    52428800,
    2359296,
    40960,
    5675286528,
    1677721600,
    17112760320,
    1696071680,
    25737166848,
    27433238528,
    196608,
    208896,
    25737363456,
    27433447424,
    26542473216,
    28288876544,
    107069105504256,
    114119319486464,
    101862729056256,
    103706566590464,
    11124102144,
    3355459584,
    34225520640,
    54641395712,
    52098598912,
    52098619392,
    2542796800,
    10240,
    69632,
    152047616,
    153944064,
    439087104,
    154013696,
    439156736,
    583972945920,
    285212672,
    2593792,
    3123200,
    5245952,
    5847040,
    21487419392,
    23949475840,
    15097856,
    15852544,
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
    "linear_z_width",
    "linear_conv_kernel_dim",
    "bytes_bf16",
    "flop_per_mac",
    "T_is_stored_length_after_append",
    "decode_T_new",
    "primary_includes_mtp",
    "gdn_uses_rank1_eq_17_18",
    "elementwise_not_in_primary",
    "elementwise_upper_bound_decode_T4096",
    "example_T",
    "mac_embed",
    "mac_full_proj_per_layer",
    "mac_attn_coeff_per_full_layer",
    "mac_lin_proj_per_layer",
    "mac_lin_conv_per_layer",
    "mac_gdn_per_layer",
    "mac_lin_out_per_layer",
    "mac_lin_token_per_layer",
    "mac_mlp_per_layer",
    "mac_lm_head",
    "mac_mtp_fc",
    "mac_C_linear_attn",
    "mac_C_full_attn",
    "mac_C_mlp",
    "mac_C_lm_head",
    "mac_C_mtp",
    "mac_C_language",
    "mac_C_complete",
    "mac_A_language",
    "mac_A_complete",
    "mac_decode_language_at_example_T",
    "mac_decode_complete_at_example_T",
    "mac_prefill_language_at_example_T",
    "mac_prefill_complete_at_example_T",
    "mac_prefill_inference_last_logits_language_at_example_T",
    "mac_prefill_inference_last_logits_complete_at_example_T",
    "weight_bytes_language_linear_attn",
    "weight_bytes_language_self_attn",
    "weight_bytes_language_mlp",
    "weight_bytes_language_layer_norms",
    "weight_bytes_language_final_norm",
    "weight_bytes_lm_head",
    "weight_bytes_mtp",
    "weight_bytes_language_mtp_excl_vision",
    "weight_bytes_embed_table",
    "weight_bytes_unique_non_embed",
    "weight_gather_bytes_per_row",
    "weight_gather_bytes_decode_language",
    "weight_gather_bytes_decode_complete",
    "weight_unique_plus_gather_decode_complete",
    "weight_gather_bytes_prefill_language_at_example_T",
    "weight_gather_bytes_prefill_complete_at_example_T",
    "kv_bytes_all_per_token",
    "c_bytes_all",
    "c_write_bytes_per_token_all",
    "s_bytes_all",
    "storage_kv_bytes_coeff_T",
    "storage_fixed_bytes",
    "decode_write_bytes",
    "decode_read_kv_bytes_coeff_Tm1",
    "decode_read_fixed_bytes",
    "storage_bytes_at_example_T",
    "decode_read_bytes_at_example_T",
    "prefill_kv_read_bytes_at_example_T",
    "prefill_kv_write_bytes_at_example_T",
    "residual_vector_bytes",
    "g_bytes_per_full_layer",
    "z_bytes_per_linear_layer",
    "logits_bytes",
    "act_forced_decode_language_bytes",
    "act_forced_decode_complete_bytes",
    "act_region_cut_decode_language_bytes",
    "act_region_cut_decode_complete_bytes",
    "act_region_cut_prefill_language_at_example_T",
    "act_region_cut_prefill_complete_at_example_T",
    "act_gemm_io_lin_layer_bytes",
    "act_gemm_io_full_proj_bytes",
    "act_gemm_io_mlp_layer_bytes",
    "act_gemm_io_lm_head_bytes",
    "act_gemm_io_mtp_fc_bytes",
    "act_gemm_io_decode_language_bytes",
    "act_gemm_io_decode_complete_bytes",
    "i_mlp_weight_only",
    "i_lm_head_weight_only",
    "i_gdn_vs_s_rw",
    "i_attn_core_vs_kv",
    "bottleneck_labels",
    "region_ids",
    "region_accounting_status",
    "n_catalog_nodes",
    "catalog_ids",
    "n_diagrams",
    "canonical_sentence_logical",
    "canonical_sentence_hypothesis",
    "canonical_sentence_activation",
)

REQUIRED_HEADINGS: tuple[str, ...] = (
    "Authority",
    "Work convention",
    "Symbolic work",
    "Instantiated work",
    "Weight traffic",
    "State traffic",
    "Activation traffic",
    "Intensity and bottleneck hypotheses",
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


class WorkTrafficMismatch(Exception):
    """Raised when the work-and-traffic markdown fails a content check."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def dumps_summary(summary: dict) -> str:
    """Pretty-print the work-and-traffic summary object.

    Args:
        summary: Object produced by :func:`instantiate_work_traffic_summary`.

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


def _gemm_io_bytes(d_in: int, d_out: int, bytes_bf16: int) -> int:
    """Return unfused GEMM operand+result bytes for one y=Wx map.

    Args:
        d_in: Input width.
        d_out: Output width.
        bytes_bf16: Element size in bytes.

    Returns:
        ``(d_in + d_out) * bytes_bf16``.
    """

    return (d_in + d_out) * bytes_bf16


def instantiate_work_traffic_summary(config: dict) -> dict:
    """Compute the TASK-06 work-and-traffic summary from ``config``.

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
    bytes_bf16 = 2
    flop_per_mac = 2
    linear_conv_delay = linear_conv_kernel_dim - 1

    mac_embed = 0
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
        3
        * linear_num_value_heads
        * linear_key_head_dim
        * linear_value_head_dim
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

    def _decode_mac(c_mac: int, a_mac: int, t_value: int) -> int:
        return c_mac + a_mac * t_value

    def _prefill_mac(c_mac: int, a_mac: int, t_value: int) -> int:
        return t_value * c_mac + a_mac * t_value * (t_value + 1) // 2

    mac_decode_language = [
        _decode_mac(mac_c_language, mac_a_language, t_value) for t_value in example_t
    ]
    mac_decode_complete = [
        _decode_mac(mac_c_complete, mac_a_complete, t_value) for t_value in example_t
    ]
    mac_prefill_language = [
        _prefill_mac(mac_c_language, mac_a_language, t_value) for t_value in example_t
    ]
    mac_prefill_complete = [
        _prefill_mac(mac_c_complete, mac_a_complete, t_value) for t_value in example_t
    ]
    mac_prefill_last_language = [
        mac_prefill_language[index] - (t_value - 1) * mac_c_lm_head
        for index, t_value in enumerate(example_t)
    ]
    mac_prefill_last_complete = [
        mac_prefill_complete[index] - 2 * (t_value - 1) * mac_c_lm_head
        for index, t_value in enumerate(example_t)
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
        n_linear_layers * lin_params_per_layer * bytes_bf16
    )
    weight_bytes_language_self_attn = (
        n_full_layers * full_params_per_layer * bytes_bf16
    )
    weight_bytes_language_mlp = (
        n_decoder_layers * 3 * intermediate_size * hidden_size * bytes_bf16
    )
    weight_bytes_language_layer_norms = (
        2 * n_decoder_layers * hidden_size * bytes_bf16
    )
    weight_bytes_language_final_norm = hidden_size * bytes_bf16
    weight_bytes_lm_head = vocab_size * hidden_size * bytes_bf16
    weight_bytes_mtp = mtp_params * bytes_bf16
    weight_bytes_embed_table = vocab_size * hidden_size * bytes_bf16
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
    weight_gather_bytes_per_row = hidden_size * bytes_bf16
    weight_gather_bytes_decode_language = weight_gather_bytes_per_row
    weight_gather_bytes_decode_complete = 2 * weight_gather_bytes_per_row
    weight_unique_plus_gather_decode_complete = (
        weight_bytes_unique_non_embed + weight_gather_bytes_decode_complete
    )
    weight_gather_bytes_prefill_language = [
        t_value * weight_gather_bytes_per_row for t_value in example_t
    ]
    weight_gather_bytes_prefill_complete = [
        2 * t_value * weight_gather_bytes_per_row for t_value in example_t
    ]

    kv_bytes_per_full_layer_per_token = 2 * n_kv_heads * head_dim * bytes_bf16
    kv_bytes_all_per_token = n_full_layers_with_kv * kv_bytes_per_full_layer_per_token
    c_bytes_all = (
        n_linear_layers * linear_conv_delay * linear_qkv_width * bytes_bf16
    )
    c_write_bytes_per_token_all = n_linear_layers * linear_qkv_width * bytes_bf16
    s_bytes_per_layer = (
        linear_num_value_heads
        * linear_key_head_dim
        * linear_value_head_dim
        * BYTES_F32
    )
    s_bytes_all = n_linear_layers * s_bytes_per_layer
    storage_kv_bytes_coeff_t = kv_bytes_all_per_token
    storage_fixed_bytes = c_bytes_all + s_bytes_all
    decode_write_bytes = (
        kv_bytes_all_per_token + c_write_bytes_per_token_all + s_bytes_all
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
    prefill_kv_read_bytes_at_example_t = [
        kv_bytes_all_per_token * t_value * (t_value - 1) // 2
        for t_value in example_t
    ]
    prefill_kv_write_bytes_at_example_t = [
        kv_bytes_all_per_token * t_value for t_value in example_t
    ]

    residual_vector_bytes = hidden_size * bytes_bf16
    g_bytes_per_full_layer = n_attn_heads * head_dim * bytes_bf16
    z_bytes_per_linear_layer = linear_z_width * bytes_bf16
    logits_bytes = vocab_size * bytes_bf16
    act_forced_decode_language_bytes = (
        n_decoder_layers * residual_vector_bytes * 2
        + n_full_layers * g_bytes_per_full_layer
        + n_linear_layers * z_bytes_per_linear_layer
        + logits_bytes
    )
    act_forced_decode_complete_bytes = (
        act_forced_decode_language_bytes
        + 2 * residual_vector_bytes
        + g_bytes_per_full_layer
        + logits_bytes
    )
    act_region_cut_decode_language_bytes = (
        act_forced_decode_language_bytes
        + residual_vector_bytes
        + n_decoder_layers * residual_vector_bytes
        + n_decoder_layers * residual_vector_bytes
        + n_decoder_layers * residual_vector_bytes
        + n_decoder_layers * residual_vector_bytes
        + residual_vector_bytes
        + residual_vector_bytes
    )
    act_region_cut_decode_complete_bytes = (
        act_region_cut_decode_language_bytes
        + 2 * residual_vector_bytes
        + g_bytes_per_full_layer
        + logits_bytes
        + 4 * residual_vector_bytes
        + 3 * residual_vector_bytes
    )
    act_region_cut_prefill_language = [
        t_value * act_region_cut_decode_language_bytes for t_value in example_t
    ]
    act_region_cut_prefill_complete = [
        t_value * act_region_cut_decode_complete_bytes for t_value in example_t
    ]

    act_gemm_io_lin_layer_bytes = (
        _gemm_io_bytes(hidden_size, linear_qkv_width, bytes_bf16)
        + _gemm_io_bytes(hidden_size, linear_z_width, bytes_bf16)
        + _gemm_io_bytes(hidden_size, linear_num_value_heads, bytes_bf16)
        + _gemm_io_bytes(hidden_size, linear_num_value_heads, bytes_bf16)
        + _gemm_io_bytes(linear_z_width, hidden_size, bytes_bf16)
    )
    act_gemm_io_full_proj_bytes = (
        _gemm_io_bytes(hidden_size, q_proj_out, bytes_bf16)
        + _gemm_io_bytes(hidden_size, kv_proj_out, bytes_bf16)
        + _gemm_io_bytes(hidden_size, kv_proj_out, bytes_bf16)
        + _gemm_io_bytes(o_proj_in, hidden_size, bytes_bf16)
    )
    act_gemm_io_mlp_layer_bytes = (
        _gemm_io_bytes(hidden_size, intermediate_size, bytes_bf16)
        + _gemm_io_bytes(hidden_size, intermediate_size, bytes_bf16)
        + _gemm_io_bytes(intermediate_size, hidden_size, bytes_bf16)
    )
    act_gemm_io_lm_head_bytes = _gemm_io_bytes(
        hidden_size, vocab_size, bytes_bf16
    )
    act_gemm_io_mtp_fc_bytes = _gemm_io_bytes(
        2 * hidden_size, hidden_size, bytes_bf16
    )
    act_gemm_io_decode_language_bytes = (
        n_linear_layers * act_gemm_io_lin_layer_bytes
        + n_full_layers * act_gemm_io_full_proj_bytes
        + n_decoder_layers * act_gemm_io_mlp_layer_bytes
        + act_gemm_io_lm_head_bytes
    )
    act_gemm_io_decode_complete_bytes = (
        act_gemm_io_decode_language_bytes
        + act_gemm_io_mtp_fc_bytes
        + act_gemm_io_full_proj_bytes
        + act_gemm_io_mlp_layer_bytes
        + act_gemm_io_lm_head_bytes
    )

    mlp_weight_bytes_one_layer = (
        3 * intermediate_size * hidden_size * bytes_bf16
    )
    i_mlp_weight_only = _ratio_number(2 * mac_mlp_per_layer, mlp_weight_bytes_one_layer)
    i_lm_head_weight_only = _ratio_number(2 * mac_lm_head, weight_bytes_lm_head)
    i_gdn_vs_s_rw = _ratio_number(2 * mac_gdn_per_layer, 2 * s_bytes_per_layer)
    i_attn_core_vs_kv = _ratio_number(
        2 * mac_attn_coeff_per_full_layer, kv_bytes_per_full_layer_per_token
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
        "n_attn_heads": n_attn_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "linear_num_value_heads": linear_num_value_heads,
        "linear_key_head_dim": linear_key_head_dim,
        "linear_value_head_dim": linear_value_head_dim,
        "linear_qkv_width": linear_qkv_width,
        "linear_z_width": linear_z_width,
        "linear_conv_kernel_dim": linear_conv_kernel_dim,
        "bytes_bf16": bytes_bf16,
        "flop_per_mac": flop_per_mac,
        "T_is_stored_length_after_append": True,
        "decode_T_new": 1,
        "primary_includes_mtp": True,
        "gdn_uses_rank1_eq_17_18": True,
        "elementwise_not_in_primary": True,
        "elementwise_upper_bound_decode_T4096": 100000000,
        "example_T": example_t,
        "mac_embed": mac_embed,
        "mac_full_proj_per_layer": mac_full_proj_per_layer,
        "mac_attn_coeff_per_full_layer": mac_attn_coeff_per_full_layer,
        "mac_lin_proj_per_layer": mac_lin_proj_per_layer,
        "mac_lin_conv_per_layer": mac_lin_conv_per_layer,
        "mac_gdn_per_layer": mac_gdn_per_layer,
        "mac_lin_out_per_layer": mac_lin_out_per_layer,
        "mac_lin_token_per_layer": mac_lin_token_per_layer,
        "mac_mlp_per_layer": mac_mlp_per_layer,
        "mac_lm_head": mac_lm_head,
        "mac_mtp_fc": mac_mtp_fc,
        "mac_C_linear_attn": mac_c_linear_attn,
        "mac_C_full_attn": mac_c_full_attn,
        "mac_C_mlp": mac_c_mlp,
        "mac_C_lm_head": mac_c_lm_head,
        "mac_C_mtp": mac_c_mtp,
        "mac_C_language": mac_c_language,
        "mac_C_complete": mac_c_complete,
        "mac_A_language": mac_a_language,
        "mac_A_complete": mac_a_complete,
        "mac_decode_language_at_example_T": mac_decode_language,
        "mac_decode_complete_at_example_T": mac_decode_complete,
        "mac_prefill_language_at_example_T": mac_prefill_language,
        "mac_prefill_complete_at_example_T": mac_prefill_complete,
        "mac_prefill_inference_last_logits_language_at_example_T": (
            mac_prefill_last_language
        ),
        "mac_prefill_inference_last_logits_complete_at_example_T": (
            mac_prefill_last_complete
        ),
        "weight_bytes_language_linear_attn": weight_bytes_language_linear_attn,
        "weight_bytes_language_self_attn": weight_bytes_language_self_attn,
        "weight_bytes_language_mlp": weight_bytes_language_mlp,
        "weight_bytes_language_layer_norms": weight_bytes_language_layer_norms,
        "weight_bytes_language_final_norm": weight_bytes_language_final_norm,
        "weight_bytes_lm_head": weight_bytes_lm_head,
        "weight_bytes_mtp": weight_bytes_mtp,
        "weight_bytes_language_mtp_excl_vision": (
            weight_bytes_language_mtp_excl_vision
        ),
        "weight_bytes_embed_table": weight_bytes_embed_table,
        "weight_bytes_unique_non_embed": weight_bytes_unique_non_embed,
        "weight_gather_bytes_per_row": weight_gather_bytes_per_row,
        "weight_gather_bytes_decode_language": weight_gather_bytes_decode_language,
        "weight_gather_bytes_decode_complete": weight_gather_bytes_decode_complete,
        "weight_unique_plus_gather_decode_complete": (
            weight_unique_plus_gather_decode_complete
        ),
        "weight_gather_bytes_prefill_language_at_example_T": (
            weight_gather_bytes_prefill_language
        ),
        "weight_gather_bytes_prefill_complete_at_example_T": (
            weight_gather_bytes_prefill_complete
        ),
        "kv_bytes_all_per_token": kv_bytes_all_per_token,
        "c_bytes_all": c_bytes_all,
        "c_write_bytes_per_token_all": c_write_bytes_per_token_all,
        "s_bytes_all": s_bytes_all,
        "storage_kv_bytes_coeff_T": storage_kv_bytes_coeff_t,
        "storage_fixed_bytes": storage_fixed_bytes,
        "decode_write_bytes": decode_write_bytes,
        "decode_read_kv_bytes_coeff_Tm1": decode_read_kv_bytes_coeff_tm1,
        "decode_read_fixed_bytes": decode_read_fixed_bytes,
        "storage_bytes_at_example_T": storage_bytes_at_example_t,
        "decode_read_bytes_at_example_T": decode_read_bytes_at_example_t,
        "prefill_kv_read_bytes_at_example_T": prefill_kv_read_bytes_at_example_t,
        "prefill_kv_write_bytes_at_example_T": prefill_kv_write_bytes_at_example_t,
        "residual_vector_bytes": residual_vector_bytes,
        "g_bytes_per_full_layer": g_bytes_per_full_layer,
        "z_bytes_per_linear_layer": z_bytes_per_linear_layer,
        "logits_bytes": logits_bytes,
        "act_forced_decode_language_bytes": act_forced_decode_language_bytes,
        "act_forced_decode_complete_bytes": act_forced_decode_complete_bytes,
        "act_region_cut_decode_language_bytes": (
            act_region_cut_decode_language_bytes
        ),
        "act_region_cut_decode_complete_bytes": (
            act_region_cut_decode_complete_bytes
        ),
        "act_region_cut_prefill_language_at_example_T": (
            act_region_cut_prefill_language
        ),
        "act_region_cut_prefill_complete_at_example_T": (
            act_region_cut_prefill_complete
        ),
        "act_gemm_io_lin_layer_bytes": act_gemm_io_lin_layer_bytes,
        "act_gemm_io_full_proj_bytes": act_gemm_io_full_proj_bytes,
        "act_gemm_io_mlp_layer_bytes": act_gemm_io_mlp_layer_bytes,
        "act_gemm_io_lm_head_bytes": act_gemm_io_lm_head_bytes,
        "act_gemm_io_mtp_fc_bytes": act_gemm_io_mtp_fc_bytes,
        "act_gemm_io_decode_language_bytes": act_gemm_io_decode_language_bytes,
        "act_gemm_io_decode_complete_bytes": act_gemm_io_decode_complete_bytes,
        "i_mlp_weight_only": i_mlp_weight_only,
        "i_lm_head_weight_only": i_lm_head_weight_only,
        "i_gdn_vs_s_rw": i_gdn_vs_s_rw,
        "i_attn_core_vs_kv": i_attn_core_vs_kv,
        "bottleneck_labels": list(BOTTLENECK_LABELS),
        "region_ids": list(REGION_IDS),
        "region_accounting_status": "forced_is_semantic_minimum; region_cut_is_assumed_region_interface_accounting",
        "n_catalog_nodes": 52,
        "catalog_ids": list(CATALOG_IDS),
        "n_diagrams": 1,
        "canonical_sentence_logical": CANONICAL_SENTENCE_LOGICAL,
        "canonical_sentence_hypothesis": CANONICAL_SENTENCE_HYPOTHESIS,
        "canonical_sentence_activation": CANONICAL_SENTENCE_ACTIVATION,
    }
    _assert_identities(summary, text)
    return summary


def _assert_identities(summary: dict, text: dict) -> None:
    """Assert dossier identities on a live work-and-traffic summary.

    Args:
        summary: Object produced by :func:`instantiate_work_traffic_summary`.
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
    assert summary["n_full_layers_with_kv"] == (
        summary["n_full_layers"] + summary["n_mtp_blocks"]
    )
    assert summary["full_attention_indices"] == list(LOCKED_FULL_ATTENTION_INDICES)
    assert summary["flop_per_mac"] == 2
    assert summary["mac_embed"] == 0
    assert summary["gdn_uses_rank1_eq_17_18"] is True
    assert summary["mac_full_proj_per_layer"] == 104857600
    assert summary["mac_lin_token_per_layer"] == 118235136
    assert summary["mac_mlp_per_layer"] == 267386880
    assert summary["mac_lm_head"] == 1271398400
    assert summary["mac_C_language"] == 25737166848
    assert summary["mac_C_complete"] == 27433238528
    assert summary["mac_A_language"] == 196608
    assert summary["mac_A_complete"] == 208896
    assert summary["mac_C_complete"] == (
        summary["mac_C_language"] + summary["mac_C_mtp"]
    )
    assert summary["weight_bytes_language_linear_attn"] == 11124102144
    assert summary["weight_bytes_language_self_attn"] == 3355459584
    assert summary["weight_bytes_language_mlp"] == 34225520640
    assert summary["weight_bytes_language_layer_norms"] == 1310720
    assert summary["weight_bytes_language_final_norm"] == 10240
    assert summary["weight_bytes_lm_head"] == 2542796800
    assert summary["weight_bytes_mtp"] == 849398784
    assert summary["weight_bytes_language_mtp_excl_vision"] == 54641395712
    assert summary["weight_bytes_embed_table"] == 2542796800
    assert summary["weight_bytes_unique_non_embed"] == 52098598912
    assert summary["weight_unique_plus_gather_decode_complete"] == 52098619392
    assert summary["i_mlp_weight_only"] == 1
    assert summary["i_lm_head_weight_only"] == 1
    assert summary["i_gdn_vs_s_rw"] == 0.75
    assert summary["i_attn_core_vs_kv"] == 6
    assert summary["elementwise_upper_bound_decode_T4096"] < (
        0.01 * (2 * summary["mac_decode_complete_at_example_T"][1])
    )
    assert summary["primary_includes_mtp"] is True
    assert summary["n_diagrams"] == 1
    assert summary["bottleneck_labels"] == list(BOTTLENECK_LABELS)
    assert summary["region_ids"] == list(REGION_IDS)
    assert summary["n_catalog_nodes"] == 52
    assert summary["catalog_ids"] == list(CATALOG_IDS)
    assert summary["example_T"] == list(EXAMPLE_T)
    assert summary["T_is_stored_length_after_append"] is True
    assert summary["decode_T_new"] == 1
    assert summary["elementwise_not_in_primary"] is True
    assert text.get("dtype") == "bfloat16"
    assert summary["canonical_sentence_logical"] == CANONICAL_SENTENCE_LOGICAL
    assert summary["canonical_sentence_hypothesis"] == CANONICAL_SENTENCE_HYPOTHESIS
    assert summary["canonical_sentence_activation"] == CANONICAL_SENTENCE_ACTIVATION
    assert summary["mac_lin_token_per_layer"] == (
        summary["mac_lin_proj_per_layer"]
        + summary["mac_lin_conv_per_layer"]
        + summary["mac_gdn_per_layer"]
        + summary["mac_lin_out_per_layer"]
    )
    assert summary["mac_C_mtp"] == (
        summary["mac_mtp_fc"]
        + summary["mac_full_proj_per_layer"]
        + summary["mac_mlp_per_layer"]
        + summary["mac_lm_head"]
    )
    assert summary["mac_A_complete"] == (
        summary["mac_A_language"] + summary["mac_attn_coeff_per_full_layer"]
    )
    for index, t_value in enumerate(summary["example_T"]):
        expected_decode_language = (
            summary["mac_C_language"] + summary["mac_A_language"] * t_value
        )
        expected_decode_complete = (
            summary["mac_C_complete"] + summary["mac_A_complete"] * t_value
        )
        expected_prefill_language = (
            t_value * summary["mac_C_language"]
            + summary["mac_A_language"] * t_value * (t_value + 1) // 2
        )
        expected_prefill_complete = (
            t_value * summary["mac_C_complete"]
            + summary["mac_A_complete"] * t_value * (t_value + 1) // 2
        )
        expected_last_language = expected_prefill_language - (
            t_value - 1
        ) * summary["mac_C_lm_head"]
        expected_last_complete = expected_prefill_complete - 2 * (
            t_value - 1
        ) * summary["mac_C_lm_head"]
        expected_store = (
            summary["storage_kv_bytes_coeff_T"] * t_value
            + summary["storage_fixed_bytes"]
        )
        expected_decode_read = (
            summary["decode_read_kv_bytes_coeff_Tm1"] * (t_value - 1)
            + summary["decode_read_fixed_bytes"]
        )
        expected_kv_read = (
            summary["kv_bytes_all_per_token"] * t_value * (t_value - 1) // 2
        )
        expected_kv_write = summary["kv_bytes_all_per_token"] * t_value
        assert summary["mac_decode_language_at_example_T"][index] == (
            expected_decode_language
        ), f"decode language T={t_value} mismatch"
        assert summary["mac_decode_complete_at_example_T"][index] == (
            expected_decode_complete
        ), f"decode complete T={t_value} mismatch"
        assert summary["mac_prefill_language_at_example_T"][index] == (
            expected_prefill_language
        ), f"prefill language T={t_value} mismatch"
        assert summary["mac_prefill_complete_at_example_T"][index] == (
            expected_prefill_complete
        ), f"prefill complete T={t_value} mismatch"
        assert summary["mac_prefill_inference_last_logits_language_at_example_T"][
            index
        ] == expected_last_language, f"last-logits language T={t_value} mismatch"
        assert summary["mac_prefill_inference_last_logits_complete_at_example_T"][
            index
        ] == expected_last_complete, f"last-logits complete T={t_value} mismatch"
        assert summary["storage_bytes_at_example_T"][index] == expected_store, (
            f"storage T={t_value} mismatch"
        )
        assert summary["decode_read_bytes_at_example_T"][index] == (
            expected_decode_read
        ), f"decode-read T={t_value} mismatch"
        assert summary["prefill_kv_read_bytes_at_example_T"][index] == (
            expected_kv_read
        ), f"prefill KV read T={t_value} mismatch"
        assert summary["prefill_kv_write_bytes_at_example_T"][index] == (
            expected_kv_write
        ), f"prefill KV write T={t_value} mismatch"
    assert summary["storage_bytes_at_example_T"] == [154013696, 439156736]
    assert summary["decode_read_bytes_at_example_T"] == [153944064, 439087104]
    assert summary["prefill_kv_read_bytes_at_example_T"] == [0, 583972945920]
    assert summary["mac_decode_language_at_example_T"] == [
        25737363456,
        26542473216,
    ]
    assert summary["mac_decode_complete_at_example_T"] == [
        27433447424,
        28288876544,
    ]
    assert summary["mac_prefill_language_at_example_T"] == [
        25737363456,
        107069105504256,
    ]
    assert summary["mac_prefill_complete_at_example_T"] == [
        27433447424,
        114119319486464,
    ]
    assert summary["mac_prefill_inference_last_logits_language_at_example_T"] == [
        25737363456,
        101862729056256,
    ]
    assert summary["mac_prefill_inference_last_logits_complete_at_example_T"] == [
        27433447424,
        103706566590464,
    ]
    assert summary["act_forced_decode_language_bytes"] == 2593792
    assert summary["act_forced_decode_complete_bytes"] == 3123200
    assert summary["act_region_cut_decode_language_bytes"] == 5245952
    assert summary["act_region_cut_decode_complete_bytes"] == 5847040
    assert summary["act_region_cut_prefill_language_at_example_T"][1] == 21487419392
    assert summary["act_region_cut_prefill_complete_at_example_T"][1] == 23949475840
    assert summary["act_gemm_io_decode_language_bytes"] == 15097856
    assert summary["act_gemm_io_decode_complete_bytes"] == 15852544
    assert summary["decode_write_bytes"] == 152047616
    assert summary["kv_bytes_all_per_token"] == 69632
    assert summary["decode_read_fixed_bytes"] == 153944064


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
    """Compare documented JSON against a live work-and-traffic object.

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


def check_work_traffic(live: dict, work_traffic_path: Path) -> None:
    """Check a work-and-traffic markdown file against a live summary object.

    Args:
        live: Work-and-traffic summary object from sitting config.
        work_traffic_path: Path to ``work-and-traffic.md``.

    Raises:
        WorkTrafficMismatch: On heading, JSON, diagram, or token mismatches.
        OSError: If the file cannot be read.
    """

    text = work_traffic_path.read_text(encoding="utf-8")
    differences: list[str] = []
    if "All MTP-only work and traffic totals are conditional on TASK-02's unverified analysis model." not in text:
        differences.append("missing conditional MTP work qualification")

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
    if CANONICAL_SENTENCE_ACTIVATION not in text:
        differences.append("canonical_sentence_activation not present verbatim")

    for node_id in CATALOG_IDS:
        if node_id not in text:
            differences.append(f"catalog id {node_id!r} missing as substring")

    for label in BOTTLENECK_LABELS:
        if label not in text:
            differences.append(f"bottleneck label {label!r} missing as substring")

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

    for integer in LOCKED_DOCUMENT_INTEGERS:
        if str(integer) not in text:
            differences.append(
                f"locked integer {integer} missing as decimal substring"
            )

    if differences:
        raise WorkTrafficMismatch(differences)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the work-and-traffic checker."""

    parser = argparse.ArgumentParser(
        description=(
            "Instantiate Qwen3.8-27B language+MTP work and traffic summary from "
            "text_config and check docs/architecture/work-and-traffic.md."
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
        help="Write work-and-traffic summary JSON to stdout.",
    )
    parser.add_argument(
        "--work-traffic",
        type=Path,
        default=None,
        help="Markdown path whose first json fence must match live summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the work-and-traffic checker CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch, 2 missing config.
    """

    args = _parse_args(argv)
    try:
        config = load_config(args.config)
        summary = instantiate_work_traffic_summary(config)
        if args.work_traffic is not None:
            if not args.work_traffic.is_file():
                print(
                    f"work-traffic file not found: {args.work_traffic}",
                    file=sys.stderr,
                )
                return 1
            check_work_traffic(summary, args.work_traffic)
    except MissingConfig as exc:
        print(str(exc.path), file=sys.stderr)
        return 2
    except WorkTrafficMismatch as exc:
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

    if args.json or args.work_traffic is None:
        sys.stdout.write(dumps_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
