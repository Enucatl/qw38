#!/usr/bin/env python3
"""Check Qwen3.8-27B lifetime and persistent-state analysis against sitting text_config.

Reads ``text_config`` from a Transformers ``config.json`` (no safetensor
payloads) and either prints the lifetime-summary object or checks that
``docs/architecture/lifetime-and-state.md`` matches it.
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

CANONICAL_SENTENCE = (
    "A value that must survive a named boundary is mathematical state for "
    "that boundary, not a cache layout or CUDA allocation."
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

LIFETIME_CLASSES: tuple[str, ...] = (
    "ephemeral",
    "live-across",
    "layer-residual",
    "token-persistent",
    "output-sink",
)

EPHEMERAL_IDS: tuple[str, ...] = (
    "token_id",
    "e",
    "h_tilde",
    "h_post",
    "h_64",
    "h_final",
    "u_q",
    "q_prime",
    "k_raw",
    "v_full",
    "q_n",
    "k_n",
    "q_rope",
    "k_rope",
    "attn",
    "y_gate",
    "mix_full",
    "qkv",
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
    "o",
    "u_gdn",
    "mix_lin",
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
)

LIVE_ACROSS_IDS: tuple[str, ...] = ("g", "z")
LAYER_RESIDUAL_IDS: tuple[str, ...] = ("h", "h_mid")
TOKEN_PERSISTENT_IDS: tuple[str, ...] = ("K_state", "V_state", "C_state", "S")
OUTPUT_SINK_IDS: tuple[str, ...] = ("logits_0", "logits_1")
STATE_IDS: tuple[str, ...] = TOKEN_PERSISTENT_IDS
REQUIRES_PRIOR_STATE_IDS: tuple[str, ...] = TOKEN_PERSISTENT_IDS
BOUNDARY_TOKEN_IDS: tuple[str, ...] = TOKEN_PERSISTENT_IDS
BOUNDARY_RESIDUAL_ADD_IDS: tuple[str, ...] = LAYER_RESIDUAL_IDS
BOUNDARY_LIVE_ACROSS_IDS: tuple[str, ...] = LIVE_ACROSS_IDS
BOUNDARY_OUTPUT_IDS: tuple[str, ...] = OUTPUT_SINK_IDS

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

DIAGRAM_REQUIRED_IDS: tuple[str, ...] = (
    "h",
    "h_mid",
    "g",
    "z",
    "K_state",
    "V_state",
    "C_state",
    "S",
    "logits_0",
    "logits_1",
)

LOCKED_DOCUMENT_INTEGERS: tuple[int, ...] = (
    2048,
    34816,
    32768,
    1474560,
    37748736,
    4096,
    69632,
    2949120,
    150994944,
    39223296,
    153944064,
    38275072,
    152047616,
    491520,
    983040,
    30720,
    61440,
    786432,
    3145728,
    39258112,
    154013696,
    142606336,
    285212672,
    181829632,
    439156736,
    181794816,
    439087104,
)

SCHEMA_KEYS: tuple[str, ...] = (
    "authority",
    "hidden_size",
    "n_decoder_layers",
    "n_linear_layers",
    "n_full_layers",
    "n_mtp_blocks",
    "n_full_layers_with_kv",
    "full_attention_indices",
    "n_kv_heads",
    "head_dim",
    "linear_qkv_width",
    "linear_conv_delay",
    "linear_state_heads",
    "linear_state_dk",
    "linear_state_dv",
    "bytes_bf16",
    "bytes_f32",
    "n_catalog_nodes",
    "catalog_ids",
    "lifetime_classes",
    "ephemeral_ids",
    "live_across_ids",
    "layer_residual_ids",
    "token_persistent_ids",
    "output_sink_ids",
    "state_ids",
    "requires_prior_state_ids",
    "boundary_token_ids",
    "boundary_residual_add_ids",
    "boundary_live_across_ids",
    "boundary_output_ids",
    "primary_includes_mtp_kv",
    "T_is_stored_length_after_append",
    "decode_T_new",
    "kv_elems_per_full_layer_per_token",
    "kv_bytes_per_full_layer_per_token",
    "kv_elems_all_per_token",
    "kv_bytes_all_per_token",
    "kv_elems_language_per_token",
    "kv_bytes_language_per_token",
    "c_elems_per_layer",
    "c_bytes_per_layer",
    "c_elems_all",
    "c_bytes_all",
    "c_write_elems_per_token_all",
    "c_write_bytes_per_token_all",
    "s_elems_per_layer",
    "s_bytes_per_layer",
    "s_elems_all",
    "s_bytes_all",
    "storage_kv_elems_coeff_T",
    "storage_kv_bytes_coeff_T",
    "storage_fixed_elems",
    "storage_fixed_bytes",
    "decode_write_elems",
    "decode_write_bytes",
    "decode_read_kv_elems_coeff_Tm1",
    "decode_read_kv_bytes_coeff_Tm1",
    "decode_read_fixed_elems",
    "decode_read_fixed_bytes",
    "example_T",
    "storage_elems_at_example_T",
    "storage_bytes_at_example_T",
    "decode_read_elems_at_example_T",
    "decode_read_bytes_at_example_T",
    "n_diagrams",
    "canonical_sentence",
    "canonical_sentence_logical",
)

REQUIRED_HEADINGS: tuple[str, ...] = (
    "Authority",
    "Logical versus physical",
    "Lifetime taxonomy",
    "Catalog lifetime table",
    "Persistent state storage",
    "Per-token read/write volumes",
    "Semantic storage candidates",
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


class LifetimeMismatch(Exception):
    """Raised when the lifetime markdown fails a content check."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def dumps_summary(summary: dict) -> str:
    """Pretty-print the lifetime-summary object.

    Args:
        summary: Object produced by :func:`instantiate_lifetime_summary`.

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


def instantiate_lifetime_summary(config: dict) -> dict:
    """Compute the TASK-04 lifetime-summary object from ``config``.

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
    n_decoder_layers = _require_int(text, "num_hidden_layers")
    full_attention_interval = _require_int(text, "full_attention_interval")
    n_mtp_blocks = _require_int(text, "mtp_num_hidden_layers")
    n_kv_heads = _require_int(text, "num_key_value_heads")
    head_dim = _require_int(text, "head_dim")
    linear_num_key_heads = _require_int(text, "linear_num_key_heads")
    linear_num_value_heads = _require_int(text, "linear_num_value_heads")
    linear_key_head_dim = _require_int(text, "linear_key_head_dim")
    linear_value_head_dim = _require_int(text, "linear_value_head_dim")
    linear_conv_kernel_dim = _require_int(text, "linear_conv_kernel_dim")

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
    linear_state_heads = linear_num_value_heads
    linear_state_dk = linear_key_head_dim
    linear_state_dv = linear_value_head_dim

    bytes_bf16 = 2
    bytes_f32 = 4

    kv_elems_per_full_layer_per_token = 2 * n_kv_heads * head_dim
    kv_bytes_per_full_layer_per_token = (
        kv_elems_per_full_layer_per_token * bytes_bf16
    )
    kv_elems_all_per_token = n_full_layers_with_kv * kv_elems_per_full_layer_per_token
    kv_bytes_all_per_token = kv_elems_all_per_token * bytes_bf16
    kv_elems_language_per_token = n_full_layers * kv_elems_per_full_layer_per_token
    kv_bytes_language_per_token = kv_elems_language_per_token * bytes_bf16

    c_elems_per_layer = linear_conv_delay * linear_qkv_width
    c_bytes_per_layer = c_elems_per_layer * bytes_bf16
    c_elems_all = n_linear_layers * c_elems_per_layer
    c_bytes_all = c_elems_all * bytes_bf16
    c_write_elems_per_token_all = n_linear_layers * linear_qkv_width
    c_write_bytes_per_token_all = c_write_elems_per_token_all * bytes_bf16

    s_elems_per_layer = (
        linear_state_heads * linear_state_dk * linear_state_dv
    )
    s_bytes_per_layer = s_elems_per_layer * bytes_f32
    s_elems_all = n_linear_layers * s_elems_per_layer
    s_bytes_all = s_elems_all * bytes_f32

    storage_kv_elems_coeff_T = kv_elems_all_per_token
    storage_kv_bytes_coeff_T = kv_bytes_all_per_token
    storage_fixed_elems = c_elems_all + s_elems_all
    storage_fixed_bytes = c_bytes_all + s_bytes_all
    decode_write_elems = (
        kv_elems_all_per_token + c_write_elems_per_token_all + s_elems_all
    )
    decode_write_bytes = (
        kv_bytes_all_per_token + c_write_bytes_per_token_all + s_bytes_all
    )
    decode_read_kv_elems_coeff_Tm1 = kv_elems_all_per_token
    decode_read_kv_bytes_coeff_Tm1 = kv_bytes_all_per_token
    decode_read_fixed_elems = storage_fixed_elems
    decode_read_fixed_bytes = storage_fixed_bytes

    example_t = list(EXAMPLE_T)
    storage_elems_at_example_t = [
        storage_kv_elems_coeff_T * t_value + storage_fixed_elems
        for t_value in example_t
    ]
    storage_bytes_at_example_t = [
        storage_kv_bytes_coeff_T * t_value + storage_fixed_bytes
        for t_value in example_t
    ]
    decode_read_elems_at_example_t = [
        decode_read_kv_elems_coeff_Tm1 * (t_value - 1) + decode_read_fixed_elems
        for t_value in example_t
    ]
    decode_read_bytes_at_example_t = [
        decode_read_kv_bytes_coeff_Tm1 * (t_value - 1) + decode_read_fixed_bytes
        for t_value in example_t
    ]

    summary = {
        "authority": AUTHORITY,
        "hidden_size": hidden_size,
        "n_decoder_layers": n_decoder_layers,
        "n_linear_layers": n_linear_layers,
        "n_full_layers": n_full_layers,
        "n_mtp_blocks": n_mtp_blocks,
        "n_full_layers_with_kv": n_full_layers_with_kv,
        "full_attention_indices": full_from_types,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "linear_qkv_width": linear_qkv_width,
        "linear_conv_delay": linear_conv_delay,
        "linear_state_heads": linear_state_heads,
        "linear_state_dk": linear_state_dk,
        "linear_state_dv": linear_state_dv,
        "bytes_bf16": bytes_bf16,
        "bytes_f32": bytes_f32,
        "n_catalog_nodes": 52,
        "catalog_ids": list(CATALOG_IDS),
        "lifetime_classes": list(LIFETIME_CLASSES),
        "ephemeral_ids": list(EPHEMERAL_IDS),
        "live_across_ids": list(LIVE_ACROSS_IDS),
        "layer_residual_ids": list(LAYER_RESIDUAL_IDS),
        "token_persistent_ids": list(TOKEN_PERSISTENT_IDS),
        "output_sink_ids": list(OUTPUT_SINK_IDS),
        "state_ids": list(STATE_IDS),
        "requires_prior_state_ids": list(REQUIRES_PRIOR_STATE_IDS),
        "boundary_token_ids": list(BOUNDARY_TOKEN_IDS),
        "boundary_residual_add_ids": list(BOUNDARY_RESIDUAL_ADD_IDS),
        "boundary_live_across_ids": list(BOUNDARY_LIVE_ACROSS_IDS),
        "boundary_output_ids": list(BOUNDARY_OUTPUT_IDS),
        "primary_includes_mtp_kv": True,
        "T_is_stored_length_after_append": True,
        "decode_T_new": 1,
        "kv_elems_per_full_layer_per_token": kv_elems_per_full_layer_per_token,
        "kv_bytes_per_full_layer_per_token": kv_bytes_per_full_layer_per_token,
        "kv_elems_all_per_token": kv_elems_all_per_token,
        "kv_bytes_all_per_token": kv_bytes_all_per_token,
        "kv_elems_language_per_token": kv_elems_language_per_token,
        "kv_bytes_language_per_token": kv_bytes_language_per_token,
        "c_elems_per_layer": c_elems_per_layer,
        "c_bytes_per_layer": c_bytes_per_layer,
        "c_elems_all": c_elems_all,
        "c_bytes_all": c_bytes_all,
        "c_write_elems_per_token_all": c_write_elems_per_token_all,
        "c_write_bytes_per_token_all": c_write_bytes_per_token_all,
        "s_elems_per_layer": s_elems_per_layer,
        "s_bytes_per_layer": s_bytes_per_layer,
        "s_elems_all": s_elems_all,
        "s_bytes_all": s_bytes_all,
        "storage_kv_elems_coeff_T": storage_kv_elems_coeff_T,
        "storage_kv_bytes_coeff_T": storage_kv_bytes_coeff_T,
        "storage_fixed_elems": storage_fixed_elems,
        "storage_fixed_bytes": storage_fixed_bytes,
        "decode_write_elems": decode_write_elems,
        "decode_write_bytes": decode_write_bytes,
        "decode_read_kv_elems_coeff_Tm1": decode_read_kv_elems_coeff_Tm1,
        "decode_read_kv_bytes_coeff_Tm1": decode_read_kv_bytes_coeff_Tm1,
        "decode_read_fixed_elems": decode_read_fixed_elems,
        "decode_read_fixed_bytes": decode_read_fixed_bytes,
        "example_T": example_t,
        "storage_elems_at_example_T": storage_elems_at_example_t,
        "storage_bytes_at_example_T": storage_bytes_at_example_t,
        "decode_read_elems_at_example_T": decode_read_elems_at_example_t,
        "decode_read_bytes_at_example_T": decode_read_bytes_at_example_t,
        "n_diagrams": 1,
        "canonical_sentence": CANONICAL_SENTENCE,
        "canonical_sentence_logical": CANONICAL_SENTENCE_LOGICAL,
    }
    _assert_identities(summary, text)
    return summary


def _assert_identities(summary: dict, text: dict) -> None:
    """Assert dossier identities on a live lifetime-summary object.

    Args:
        summary: Object produced by :func:`instantiate_lifetime_summary`.
        text: The ``text_config`` mapping used to build ``summary``.
    """

    assert list(summary) == list(SCHEMA_KEYS), (
        f"schema key order mismatch: {list(summary)} vs {list(SCHEMA_KEYS)}"
    )
    assert summary["authority"] == AUTHORITY
    assert summary["n_catalog_nodes"] == 52
    assert len(summary["catalog_ids"]) == 52
    assert summary["catalog_ids"] == list(CATALOG_IDS)
    assert summary["n_full_layers_with_kv"] == (
        summary["n_full_layers"] + summary["n_mtp_blocks"]
    )
    assert summary["n_full_layers_with_kv"] == 17
    assert summary["full_attention_indices"] == list(LOCKED_FULL_ATTENTION_INDICES)
    assert text.get("dtype") == "bfloat16"
    assert text.get("mamba_ssm_dtype") == "float32"
    assert summary["bytes_bf16"] == 2
    assert summary["bytes_f32"] == 4
    assert summary["kv_elems_per_full_layer_per_token"] == (
        2 * summary["n_kv_heads"] * summary["head_dim"]
    )
    assert summary["kv_elems_per_full_layer_per_token"] == 2048
    assert summary["kv_elems_all_per_token"] == (
        summary["n_full_layers_with_kv"] * 2048
    )
    assert summary["kv_elems_all_per_token"] == 34816
    assert summary["c_elems_all"] == (
        summary["n_linear_layers"]
        * summary["linear_conv_delay"]
        * summary["linear_qkv_width"]
    )
    assert summary["c_elems_all"] == 1474560
    assert summary["s_elems_all"] == (
        summary["n_linear_layers"]
        * summary["linear_state_heads"]
        * summary["linear_state_dk"]
        * summary["linear_state_dv"]
    )
    assert summary["s_elems_all"] == 37748736
    assert summary["kv_bytes_per_full_layer_per_token"] == (
        summary["kv_elems_per_full_layer_per_token"] * 2
    )
    assert summary["kv_bytes_all_per_token"] == summary["kv_elems_all_per_token"] * 2
    assert summary["kv_bytes_language_per_token"] == (
        summary["kv_elems_language_per_token"] * 2
    )
    assert summary["c_bytes_per_layer"] == summary["c_elems_per_layer"] * 2
    assert summary["c_bytes_all"] == summary["c_elems_all"] * 2
    assert summary["c_write_bytes_per_token_all"] == (
        summary["c_write_elems_per_token_all"] * 2
    )
    assert summary["s_bytes_per_layer"] == summary["s_elems_per_layer"] * 4
    assert summary["s_bytes_all"] == summary["s_elems_all"] * 4
    assert summary["storage_kv_bytes_coeff_T"] == (
        summary["storage_kv_elems_coeff_T"] * 2
    )
    assert summary["decode_read_kv_bytes_coeff_Tm1"] == (
        summary["decode_read_kv_elems_coeff_Tm1"] * 2
    )
    assert summary["storage_fixed_elems"] == (
        summary["c_elems_all"] + summary["s_elems_all"]
    )
    assert summary["storage_fixed_bytes"] == (
        summary["c_bytes_all"] + summary["s_bytes_all"]
    )
    assert summary["decode_write_elems"] == (
        summary["kv_elems_all_per_token"]
        + summary["c_write_elems_per_token_all"]
        + summary["s_elems_all"]
    )
    assert summary["decode_write_bytes"] == (
        summary["kv_bytes_all_per_token"]
        + summary["c_write_bytes_per_token_all"]
        + summary["s_bytes_all"]
    )
    assert summary["primary_includes_mtp_kv"] is True
    class_arrays = [
        summary["ephemeral_ids"],
        summary["live_across_ids"],
        summary["layer_residual_ids"],
        summary["token_persistent_ids"],
        summary["output_sink_ids"],
    ]
    seen: list[str] = []
    for array in class_arrays:
        for node_id in array:
            assert node_id not in seen, f"lifetime class overlap on {node_id!r}"
            seen.append(node_id)
    assert set(seen) == set(summary["catalog_ids"]), (
        "lifetime class union is not catalog_ids"
    )
    assert len(seen) == 52
    assert summary["token_persistent_ids"] == summary["state_ids"]
    assert summary["state_ids"] == summary["requires_prior_state_ids"]
    assert summary["token_persistent_ids"] == list(TOKEN_PERSISTENT_IDS)
    assert summary["n_diagrams"] == 1
    assert summary["example_T"] == list(EXAMPLE_T)
    assert summary["canonical_sentence"] == CANONICAL_SENTENCE
    assert summary["canonical_sentence_logical"] == CANONICAL_SENTENCE_LOGICAL
    for index, t_value in enumerate(summary["example_T"]):
        expected_store_elems = (
            summary["storage_kv_elems_coeff_T"] * t_value
            + summary["storage_fixed_elems"]
        )
        expected_store_bytes = (
            summary["storage_kv_bytes_coeff_T"] * t_value
            + summary["storage_fixed_bytes"]
        )
        expected_read_elems = (
            summary["decode_read_kv_elems_coeff_Tm1"] * (t_value - 1)
            + summary["decode_read_fixed_elems"]
        )
        expected_read_bytes = (
            summary["decode_read_kv_bytes_coeff_Tm1"] * (t_value - 1)
            + summary["decode_read_fixed_bytes"]
        )
        assert summary["storage_elems_at_example_T"][index] == expected_store_elems, (
            f"storage_elems_at_example_T[{index}] mismatch"
        )
        assert summary["storage_bytes_at_example_T"][index] == expected_store_bytes, (
            f"storage_bytes_at_example_T[{index}] mismatch"
        )
        assert (
            summary["decode_read_elems_at_example_T"][index] == expected_read_elems
        ), f"decode_read_elems_at_example_T[{index}] mismatch"
        assert (
            summary["decode_read_bytes_at_example_T"][index] == expected_read_bytes
        ), f"decode_read_bytes_at_example_T[{index}] mismatch"


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
    """Compare documented JSON against a live lifetime-summary object.

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


def check_lifetime(live: dict, lifetime_path: Path) -> None:
    """Check a lifetime markdown file against a live summary object.

    Args:
        live: Lifetime-summary object from sitting config.
        lifetime_path: Path to ``lifetime-and-state.md``.

    Raises:
        LifetimeMismatch: On heading, JSON, diagram, or token mismatches.
        OSError: If the file cannot be read.
    """

    text = lifetime_path.read_text(encoding="utf-8")
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
        differences.append(
            f"mermaid fence count {len(mermaid_fences)} != 1"
        )
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
    if CANONICAL_SENTENCE not in text:
        differences.append("canonical_sentence not present verbatim")

    for node_id in CATALOG_IDS:
        if node_id not in text:
            differences.append(f"catalog id {node_id!r} missing as substring")

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
        raise LifetimeMismatch(differences)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the lifetime-and-state checker."""

    parser = argparse.ArgumentParser(
        description=(
            "Instantiate Qwen3.8-27B language+MTP lifetime and persistent-state "
            "summary from text_config and check "
            "docs/architecture/lifetime-and-state.md."
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
        help="Write lifetime-summary JSON to stdout.",
    )
    parser.add_argument(
        "--lifetime",
        type=Path,
        default=None,
        help="Markdown path whose first json fence must match live summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the lifetime-and-state checker CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch, 2 missing config.
    """

    args = _parse_args(argv)
    try:
        config = load_config(args.config)
        summary = instantiate_lifetime_summary(config)
        if args.lifetime is not None:
            if not args.lifetime.is_file():
                print(f"lifetime file not found: {args.lifetime}", file=sys.stderr)
                return 1
            check_lifetime(summary, args.lifetime)
    except MissingConfig as exc:
        print(str(exc.path), file=sys.stderr)
        return 2
    except LifetimeMismatch as exc:
        print("\n".join(exc.differences), file=sys.stderr)
        return 1
    except AssertionError as exc:
        print(f"assert failed: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"assert failed: invalid config JSON: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        missing = Path(getattr(exc, "filename", args.config))
        print(str(missing), file=sys.stderr)
        return 2

    if args.json or args.lifetime is None:
        sys.stdout.write(dumps_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
