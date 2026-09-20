# Qwen3.8-27B model inventory (TASK-01)

> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.

## Authority

| Item | Value | Label |
| --- | --- | --- |
| Checkpoint path | `.cache/authorities/qwen3.8-27b-transformers` | OBSERVED |
| `config.json` | architecture, `text_config`, `vision_config`, token ids | OBSERVED |
| `model.safetensors.index.json` | `weight_map` (1199 keys), `metadata.total_size` | OBSERVED |
| Safetensor shard headers | 18 files `model-*-of-00018.safetensors` (payloads unread) | OBSERVED |
| Evidence policy | `OBSERVED` = read from files; `DERIVED` = arithmetic from OBSERVED (`docs/architecture/plan.md`) | OBSERVED |
| In scope (detailed) | Language stack (`model.language_model.*`), MTP (`mtp.*`), untied `lm_head` | — |
| Deferred (coarse totals only) | Vision encoder (`model.visual.*`), preprocessor/video configs | — |
| Out of scope | Quartz/CUDA, llama.cpp/GGML, GGUF, safetensor payloads | — |

## Configuration

### Top-level (non-vision)

| Key | Value | Label |
| --- | --- | --- |
| `architectures` | `["Qwen3_5ForConditionalGeneration"]` | OBSERVED |
| `model_type` | `"qwen3_5"` | OBSERVED |
| `language_model_only` | `false` | OBSERVED |
| `tie_word_embeddings` | `false` | OBSERVED |
| `transformers_version` | `"5.8.0.dev0"` | OBSERVED |
| `image_token_id` | `248056` | OBSERVED |
| `video_token_id` | `248057` | OBSERVED |
| `vision_start_token_id` | `248053` | OBSERVED |
| `vision_end_token_id` | `248054` | OBSERVED |

### `text_config`

| Key | Value | Label |
| --- | --- | --- |
| `attention_bias` | `false` | OBSERVED |
| `attention_dropout` | `0.0` | OBSERVED |
| `attn_output_gate` | `true` | OBSERVED |
| `bos_token_id` | `248044` | OBSERVED |
| `dtype` | `"bfloat16"` | OBSERVED |
| `eos_token_id` | `248044` | OBSERVED |
| `full_attention_interval` | `4` | OBSERVED |
| `head_dim` | `256` | OBSERVED |
| `hidden_act` | `"silu"` | OBSERVED |
| `hidden_size` | `5120` | OBSERVED |
| `initializer_range` | `0.02` | OBSERVED |
| `intermediate_size` | `17408` | OBSERVED |
| `layer_types` | `["linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention"]` | OBSERVED |
| `linear_conv_kernel_dim` | `4` | OBSERVED |
| `linear_key_head_dim` | `128` | OBSERVED |
| `linear_num_key_heads` | `16` | OBSERVED |
| `linear_num_value_heads` | `48` | OBSERVED |
| `linear_value_head_dim` | `128` | OBSERVED |
| `mamba_ssm_dtype` | `"float32"` | OBSERVED |
| `max_position_embeddings` | `262144` | OBSERVED |
| `model_type` | `"qwen3_5_text"` | OBSERVED |
| `mtp_num_hidden_layers` | `1` | OBSERVED |
| `mtp_use_dedicated_embeddings` | `false` | OBSERVED |
| `num_attention_heads` | `24` | OBSERVED |
| `num_hidden_layers` | `64` | OBSERVED |
| `num_key_value_heads` | `4` | OBSERVED |
| `output_gate_type` | `"swish"` | OBSERVED |
| `pad_token_id` | `null` | OBSERVED |
| `partial_rotary_factor` | `0.25` | OBSERVED |
| `rms_norm_eps` | `1e-06` | OBSERVED |
| `rope_parameters` | `{"mrope_interleaved": true, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"}` | OBSERVED |
| `tie_word_embeddings` | `false` | OBSERVED |
| `use_cache` | `true` | OBSERVED |
| `vocab_size` | `248320` | OBSERVED |

### `vision_config` (deferred)

| Key | Value | Label |
| --- | --- | --- |
| `deepstack_visual_indexes` | `[]` | OBSERVED |
| `depth` | `27` | OBSERVED |
| `hidden_act` | `"gelu_pytorch_tanh"` | OBSERVED |
| `hidden_size` | `1152` | OBSERVED |
| `in_channels` | `3` | OBSERVED |
| `initializer_range` | `0.02` | OBSERVED |
| `intermediate_size` | `4304` | OBSERVED |
| `model_type` | `"qwen3_5"` | OBSERVED |
| `num_heads` | `16` | OBSERVED |
| `num_position_embeddings` | `2304` | OBSERVED |
| `out_hidden_size` | `5120` | OBSERVED |
| `patch_size` | `16` | OBSERVED |
| `spatial_merge_size` | `2` | OBSERVED |
| `temporal_patch_size` | `2` | OBSERVED |

Vision encoder mathematics, preprocessor algorithm, and chat-template internals are deferred; this table records checkpoint metadata only.

### Generation metadata (OBSERVED only)

| Source | Field | Value |
| --- | --- | --- |
| `generation_config.json` | `eos_token_id` | `[248046, 248044]` |
| `text_config` | `eos_token_id` | `248044` |

## Layer ordering

| Property | Value | Label |
| --- | --- | --- |
| `num_hidden_layers` | 64 | OBSERVED |
| `layer_types` length | 64 | OBSERVED |
| `linear_attention` count | 48 | DERIVED |
| `full_attention` count | 16 | DERIVED |
| Repeating pattern | three `linear_attention` then one `full_attention` (`i % 4 == 3` → full) | DERIVED |
| Full-attention layer indices | [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63] | DERIVED |
| `full_attention_interval` | 4 | OBSERVED |
| `mtp_num_hidden_layers` | 1 | OBSERVED |
| MTP placement | One extra full-attention + MLP block under `mtp.layers.0.*`, outside `layer_types`, after the 64-layer language stack | OBSERVED |

## State implications

Structural naming only; forward equations and recurrence details are TASK-02.

### Full attention (16 language layers + MTP self-attention)

- **KV candidates:** `k_proj` / `v_proj` shapes `(1024, 5120)` = `4×256 × H` per full-attention layer and in MTP self-attention. **DERIVED**
- **`use_cache`:** `true` — token-persistent KV cache is config-enabled. **OBSERVED**
- **RoPE:** from `rope_parameters` and `partial_rotary_factor` 0.25; rotary dim = 0.25 × 256 = **64**. **DERIVED**; no `inv_freq` tensor in checkpoint. **OBSERVED**
- **`mrope_section`:** `[11, 11, 10]` (sum 32). Relationship to rotary dim is **UNKNOWN** (TASK-02). **OBSERVED**
- Language full-attention layers have no bias tensors. **OBSERVED**

### Linear attention (48 layers)

- **`in_proj_qkv`:** `(10240, 5120)` = `(16+16+48)×128 × H`. **DERIVED**
- **`conv1d.weight`:** `(10240, 1, 4)` — length-`3` convolution delay along the 10240-wide qkv stream. **DERIVED**
- **`A_log` / `dt_bias`:** shape `(48,)` = `linear_num_value_heads`. Recurrent parameters. **DERIVED**
- **`mamba_ssm_dtype`:** `float32` — config-stated runtime SSM state dtype; all checkpoint weights are BF16. **OBSERVED**
- Exact SSM / delta recurrence is **UNKNOWN** (TASK-02).

### Output gate

- **`attn_output_gate`:** `true`; **`output_gate_type`:** `swish`. **OBSERVED**
- Doubled `q_proj` `(12288, 5120)` = `2×24×256 × H` and matching MTP `q_proj`; no dedicated gate tensor. **DERIVED** / **OBSERVED**
- Application rule is **UNKNOWN** (TASK-02).

### Embeddings

- Untied: `embed_tokens` and `lm_head` both `(248320, 5120)`. **OBSERVED**
- **`mtp_use_dedicated_embeddings`:** `false` — MTP shares language embeddings. **OBSERVED**

### MTP

- One `mtp.layers.0` block: input/post norms, gated full self-attention (`q_proj` doubled), MLP, plus `mtp.fc` `(5120, 10240)`, `mtp.norm`, and pre-FC norms. **OBSERVED**
- Inventoried in detail (not deferred).

### Vision (deferred)

- No language-decode state inventory. Visual tokens enter the language residual as 5120-d vectors via `merger` (`out_hidden_size` 5120). **OBSERVED**
- Family totals below include vision for checkpoint completeness.

## Semantic-family inventory

Pattern `{i}` is the language layer index `0…63`. Grand totals sum all level-2 families including deferred vision.

### Level-1 rollup

| Level-1 id | Tensors | Parameters | BF16 bytes | Label |
| --- | ---: | ---: | ---: | --- |
| `language_embed` | 1 | 1271398400 | 2542796800 | OBSERVED |
| `language_final_norm` | 1 | 5120 | 10240 | OBSERVED |
| `language_lm_head` | 1 | 1271398400 | 2542796800 | OBSERVED |
| `language_layer_norms` | 128 | 655360 | 1310720 | OBSERVED |
| `language_linear_attn` | 432 | 5562051072 | 11124102144 | OBSERVED |
| `language_self_attn` | 96 | 1677729792 | 3355459584 | OBSERVED |
| `language_mlp` | 192 | 17112760320 | 34225520640 | OBSERVED |
| `mtp` | 15 | 424699392 | 849398784 | OBSERVED |
| `vision` | 333 | 460730096 | 921460192 | OBSERVED |
| **Grand total** | **1199** | **27781427952** | **55562855904** | OBSERVED |
| Language + MTP (excl. vision) | 866 | 27320697856 | 54641395712 | DERIVED |

### Level-2 families

| Level-2 id | Level-1 | Pattern | Count | Unique shapes | Parameters | BF16 bytes | dtype |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| `embed` | `language_embed` | `model.language_model.embed_tokens.weight` | 1 | [248320, 5120] | 1271398400 | 2542796800 | BF16 |
| `final_norm` | `language_final_norm` | `model.language_model.norm.weight` | 1 | [5120] | 5120 | 10240 | BF16 |
| `lm_head` | `language_lm_head` | `lm_head.weight` | 1 | [248320, 5120] | 1271398400 | 2542796800 | BF16 |
| `input_layernorm` | `language_layer_norms` | `model.language_model.layers.{i}.input_layernorm.weight` | 64 | [5120] | 327680 | 655360 | BF16 |
| `post_attention_layernorm` | `language_layer_norms` | `model.language_model.layers.{i}.post_attention_layernorm.weight` | 64 | [5120] | 327680 | 655360 | BF16 |
| `linear_attn.A_log` | `language_linear_attn` | `model.language_model.layers.{i}.linear_attn.A_log` | 48 | [48] | 2304 | 4608 | BF16 |
| `linear_attn.conv1d` | `language_linear_attn` | `model.language_model.layers.{i}.linear_attn.conv1d.weight` | 48 | [10240, 1, 4] | 1966080 | 3932160 | BF16 |
| `linear_attn.dt_bias` | `language_linear_attn` | `model.language_model.layers.{i}.linear_attn.dt_bias` | 48 | [48] | 2304 | 4608 | BF16 |
| `linear_attn.in_proj_a` | `language_linear_attn` | `model.language_model.layers.{i}.linear_attn.in_proj_a.weight` | 48 | [48, 5120] | 11796480 | 23592960 | BF16 |
| `linear_attn.in_proj_b` | `language_linear_attn` | `model.language_model.layers.{i}.linear_attn.in_proj_b.weight` | 48 | [48, 5120] | 11796480 | 23592960 | BF16 |
| `linear_attn.in_proj_qkv` | `language_linear_attn` | `model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight` | 48 | [10240, 5120] | 2516582400 | 5033164800 | BF16 |
| `linear_attn.in_proj_z` | `language_linear_attn` | `model.language_model.layers.{i}.linear_attn.in_proj_z.weight` | 48 | [6144, 5120] | 1509949440 | 3019898880 | BF16 |
| `linear_attn.norm` | `language_linear_attn` | `model.language_model.layers.{i}.linear_attn.norm.weight` | 48 | [128] | 6144 | 12288 | BF16 |
| `linear_attn.out_proj` | `language_linear_attn` | `model.language_model.layers.{i}.linear_attn.out_proj.weight` | 48 | [5120, 6144] | 1509949440 | 3019898880 | BF16 |
| `self_attn.q_proj` | `language_self_attn` | `model.language_model.layers.{i}.self_attn.q_proj.weight` | 16 | [12288, 5120] | 1006632960 | 2013265920 | BF16 |
| `self_attn.k_proj` | `language_self_attn` | `model.language_model.layers.{i}.self_attn.k_proj.weight` | 16 | [1024, 5120] | 83886080 | 167772160 | BF16 |
| `self_attn.v_proj` | `language_self_attn` | `model.language_model.layers.{i}.self_attn.v_proj.weight` | 16 | [1024, 5120] | 83886080 | 167772160 | BF16 |
| `self_attn.o_proj` | `language_self_attn` | `model.language_model.layers.{i}.self_attn.o_proj.weight` | 16 | [5120, 6144] | 503316480 | 1006632960 | BF16 |
| `self_attn.q_norm` | `language_self_attn` | `model.language_model.layers.{i}.self_attn.q_norm.weight` | 16 | [256] | 4096 | 8192 | BF16 |
| `self_attn.k_norm` | `language_self_attn` | `model.language_model.layers.{i}.self_attn.k_norm.weight` | 16 | [256] | 4096 | 8192 | BF16 |
| `mlp.gate_proj` | `language_mlp` | `model.language_model.layers.{i}.mlp.gate_proj.weight` | 64 | [17408, 5120] | 5704253440 | 11408506880 | BF16 |
| `mlp.up_proj` | `language_mlp` | `model.language_model.layers.{i}.mlp.up_proj.weight` | 64 | [17408, 5120] | 5704253440 | 11408506880 | BF16 |
| `mlp.down_proj` | `language_mlp` | `model.language_model.layers.{i}.mlp.down_proj.weight` | 64 | [5120, 17408] | 5704253440 | 11408506880 | BF16 |
| `mtp.fc` | `mtp` | `mtp.fc.weight` | 1 | [5120, 10240] | 52428800 | 104857600 | BF16 |
| `mtp.norm` | `mtp` | `mtp.norm.weight` | 1 | [5120] | 5120 | 10240 | BF16 |
| `mtp.pre_fc_norm_embedding` | `mtp` | `mtp.pre_fc_norm_embedding.weight` | 1 | [5120] | 5120 | 10240 | BF16 |
| `mtp.pre_fc_norm_hidden` | `mtp` | `mtp.pre_fc_norm_hidden.weight` | 1 | [5120] | 5120 | 10240 | BF16 |
| `mtp.input_layernorm` | `mtp` | `mtp.layers.0.input_layernorm.weight` | 1 | [5120] | 5120 | 10240 | BF16 |
| `mtp.post_attention_layernorm` | `mtp` | `mtp.layers.0.post_attention_layernorm.weight` | 1 | [5120] | 5120 | 10240 | BF16 |
| `mtp.self_attn.q_proj` | `mtp` | `mtp.layers.0.self_attn.q_proj.weight` | 1 | [12288, 5120] | 62914560 | 125829120 | BF16 |
| `mtp.self_attn.k_proj` | `mtp` | `mtp.layers.0.self_attn.k_proj.weight` | 1 | [1024, 5120] | 5242880 | 10485760 | BF16 |
| `mtp.self_attn.v_proj` | `mtp` | `mtp.layers.0.self_attn.v_proj.weight` | 1 | [1024, 5120] | 5242880 | 10485760 | BF16 |
| `mtp.self_attn.o_proj` | `mtp` | `mtp.layers.0.self_attn.o_proj.weight` | 1 | [5120, 6144] | 31457280 | 62914560 | BF16 |
| `mtp.self_attn.q_norm` | `mtp` | `mtp.layers.0.self_attn.q_norm.weight` | 1 | [256] | 256 | 512 | BF16 |
| `mtp.self_attn.k_norm` | `mtp` | `mtp.layers.0.self_attn.k_norm.weight` | 1 | [256] | 256 | 512 | BF16 |
| `mtp.mlp.gate_proj` | `mtp` | `mtp.layers.0.mlp.gate_proj.weight` | 1 | [17408, 5120] | 89128960 | 178257920 | BF16 |
| `mtp.mlp.up_proj` | `mtp` | `mtp.layers.0.mlp.up_proj.weight` | 1 | [17408, 5120] | 89128960 | 178257920 | BF16 |
| `mtp.mlp.down_proj` | `mtp` | `mtp.layers.0.mlp.down_proj.weight` | 1 | [5120, 17408] | 89128960 | 178257920 | BF16 |
| `vision.patch_embed` | `vision` | `model.visual.patch_embed.*` | 2 | [1152]; [1152, 3, 2, 16, 16] | 1770624 | 3541248 | BF16 |
| `vision.pos_embed` | `vision` | `model.visual.pos_embed.*` | 1 | [2304, 1152] | 2654208 | 5308416 | BF16 |
| `vision.blocks` | `vision` | `model.visual.blocks.*` | 324 | [1152]; [1152, 1152]; [1152, 4304]; [3456]; [3456, 1152]; [4304]; [4304, 1152] | 411466608 | 822933216 | BF16 |
| `vision.merger` | `vision` | `model.visual.merger.*` | 6 | [1152]; [4608]; [4608, 4608]; [5120]; [5120, 4608] | 44838656 | 89677312 | BF16 |

## Checkpoint verification

Targets below are **unverified** draft conclusions pending independent verification.

| Check | Expected | Label |
| --- | --- | --- |
| Tensor count | 1199 | OBSERVED |
| Shard count | 18 unique filenames in `weight_map` | OBSERVED |
| Dtype | all tensors `BF16` | OBSERVED |
| Parameter total | 27781427952 | OBSERVED |
| Payload byte total | 55562855904 (= parameters × 2) | OBSERVED |
| `int(metadata.total_size)` | 55562855904 (equals payload bytes) | OBSERVED |
| Header name set | equals `weight_map` key set | OBSERVED |
| `layer_types` vs modules | each layer has `linear_attn.*` xor `self_attn.*` matching `layer_types[i]` | OBSERVED |
| Shard file size | may exceed payload bytes (JSON header overhead); verification uses payload bytes only | — |

## Method

- **Script:** `scripts/inventory_bf16_checkpoint.py` (Python 3.11+, stdlib only).
- **Parse:** per shard, uint64-le header length → UTF-8 JSON header; `__metadata__` dropped; `dtype`, `shape`, `data_offsets` recorded; payloads unread.
- **Byte size:** `data_offsets[1] - data_offsets[0]`; asserted equal to `numel(shape) × 2` for BF16.
- **Family mapping:** exhaustive level-2 taxonomy in TASK-01 dossier; every tensor maps to exactly one family.
- **Reproduce:**

```sh
python3 scripts/inventory_bf16_checkpoint.py \
  --checkpoint .cache/authorities/qwen3.8-27b-transformers \
  --json
```

## Machine-checkable totals

Embedded from implementation-stage `scripts/inventory_bf16_checkpoint.py --json` stdout (pretty-printed, script key order).

```json
{
  "authority": ".cache/authorities/qwen3.8-27b-transformers",
  "n_tensors": 1199,
  "n_shards": 18,
  "dtype": "BF16",
  "n_parameters": 27781427952,
  "n_bytes": 55562855904,
  "index_metadata_total_size": 55562855904,
  "families": {
    "embed": {
      "level1": "language_embed",
      "n_tensors": 1,
      "n_parameters": 1271398400,
      "n_bytes": 2542796800,
      "dtype": "BF16",
      "shapes": [
        [
          248320,
          5120
        ]
      ]
    },
    "final_norm": {
      "level1": "language_final_norm",
      "n_tensors": 1,
      "n_parameters": 5120,
      "n_bytes": 10240,
      "dtype": "BF16",
      "shapes": [
        [
          5120
        ]
      ]
    },
    "lm_head": {
      "level1": "language_lm_head",
      "n_tensors": 1,
      "n_parameters": 1271398400,
      "n_bytes": 2542796800,
      "dtype": "BF16",
      "shapes": [
        [
          248320,
          5120
        ]
      ]
    },
    "input_layernorm": {
      "level1": "language_layer_norms",
      "n_tensors": 64,
      "n_parameters": 327680,
      "n_bytes": 655360,
      "dtype": "BF16",
      "shapes": [
        [
          5120
        ]
      ]
    },
    "post_attention_layernorm": {
      "level1": "language_layer_norms",
      "n_tensors": 64,
      "n_parameters": 327680,
      "n_bytes": 655360,
      "dtype": "BF16",
      "shapes": [
        [
          5120
        ]
      ]
    },
    "linear_attn.A_log": {
      "level1": "language_linear_attn",
      "n_tensors": 48,
      "n_parameters": 2304,
      "n_bytes": 4608,
      "dtype": "BF16",
      "shapes": [
        [
          48
        ]
      ]
    },
    "linear_attn.conv1d": {
      "level1": "language_linear_attn",
      "n_tensors": 48,
      "n_parameters": 1966080,
      "n_bytes": 3932160,
      "dtype": "BF16",
      "shapes": [
        [
          10240,
          1,
          4
        ]
      ]
    },
    "linear_attn.dt_bias": {
      "level1": "language_linear_attn",
      "n_tensors": 48,
      "n_parameters": 2304,
      "n_bytes": 4608,
      "dtype": "BF16",
      "shapes": [
        [
          48
        ]
      ]
    },
    "linear_attn.in_proj_a": {
      "level1": "language_linear_attn",
      "n_tensors": 48,
      "n_parameters": 11796480,
      "n_bytes": 23592960,
      "dtype": "BF16",
      "shapes": [
        [
          48,
          5120
        ]
      ]
    },
    "linear_attn.in_proj_b": {
      "level1": "language_linear_attn",
      "n_tensors": 48,
      "n_parameters": 11796480,
      "n_bytes": 23592960,
      "dtype": "BF16",
      "shapes": [
        [
          48,
          5120
        ]
      ]
    },
    "linear_attn.in_proj_qkv": {
      "level1": "language_linear_attn",
      "n_tensors": 48,
      "n_parameters": 2516582400,
      "n_bytes": 5033164800,
      "dtype": "BF16",
      "shapes": [
        [
          10240,
          5120
        ]
      ]
    },
    "linear_attn.in_proj_z": {
      "level1": "language_linear_attn",
      "n_tensors": 48,
      "n_parameters": 1509949440,
      "n_bytes": 3019898880,
      "dtype": "BF16",
      "shapes": [
        [
          6144,
          5120
        ]
      ]
    },
    "linear_attn.norm": {
      "level1": "language_linear_attn",
      "n_tensors": 48,
      "n_parameters": 6144,
      "n_bytes": 12288,
      "dtype": "BF16",
      "shapes": [
        [
          128
        ]
      ]
    },
    "linear_attn.out_proj": {
      "level1": "language_linear_attn",
      "n_tensors": 48,
      "n_parameters": 1509949440,
      "n_bytes": 3019898880,
      "dtype": "BF16",
      "shapes": [
        [
          5120,
          6144
        ]
      ]
    },
    "self_attn.q_proj": {
      "level1": "language_self_attn",
      "n_tensors": 16,
      "n_parameters": 1006632960,
      "n_bytes": 2013265920,
      "dtype": "BF16",
      "shapes": [
        [
          12288,
          5120
        ]
      ]
    },
    "self_attn.k_proj": {
      "level1": "language_self_attn",
      "n_tensors": 16,
      "n_parameters": 83886080,
      "n_bytes": 167772160,
      "dtype": "BF16",
      "shapes": [
        [
          1024,
          5120
        ]
      ]
    },
    "self_attn.v_proj": {
      "level1": "language_self_attn",
      "n_tensors": 16,
      "n_parameters": 83886080,
      "n_bytes": 167772160,
      "dtype": "BF16",
      "shapes": [
        [
          1024,
          5120
        ]
      ]
    },
    "self_attn.o_proj": {
      "level1": "language_self_attn",
      "n_tensors": 16,
      "n_parameters": 503316480,
      "n_bytes": 1006632960,
      "dtype": "BF16",
      "shapes": [
        [
          5120,
          6144
        ]
      ]
    },
    "self_attn.q_norm": {
      "level1": "language_self_attn",
      "n_tensors": 16,
      "n_parameters": 4096,
      "n_bytes": 8192,
      "dtype": "BF16",
      "shapes": [
        [
          256
        ]
      ]
    },
    "self_attn.k_norm": {
      "level1": "language_self_attn",
      "n_tensors": 16,
      "n_parameters": 4096,
      "n_bytes": 8192,
      "dtype": "BF16",
      "shapes": [
        [
          256
        ]
      ]
    },
    "mlp.gate_proj": {
      "level1": "language_mlp",
      "n_tensors": 64,
      "n_parameters": 5704253440,
      "n_bytes": 11408506880,
      "dtype": "BF16",
      "shapes": [
        [
          17408,
          5120
        ]
      ]
    },
    "mlp.up_proj": {
      "level1": "language_mlp",
      "n_tensors": 64,
      "n_parameters": 5704253440,
      "n_bytes": 11408506880,
      "dtype": "BF16",
      "shapes": [
        [
          17408,
          5120
        ]
      ]
    },
    "mlp.down_proj": {
      "level1": "language_mlp",
      "n_tensors": 64,
      "n_parameters": 5704253440,
      "n_bytes": 11408506880,
      "dtype": "BF16",
      "shapes": [
        [
          5120,
          17408
        ]
      ]
    },
    "mtp.fc": {
      "level1": "mtp",
      "n_tensors": 1,
      "n_parameters": 52428800,
      "n_bytes": 104857600,
      "dtype": "BF16",
      "shapes": [
        [
          5120,
          10240
        ]
      ]
    },
    "mtp.norm": {
      "level1": "mtp",
      "n_tensors": 1,
      "n_parameters": 5120,
      "n_bytes": 10240,
      "dtype": "BF16",
      "shapes": [
        [
          5120
        ]
      ]
    },
    "mtp.pre_fc_norm_embedding": {
      "level1": "mtp",
      "n_tensors": 1,
      "n_parameters": 5120,
      "n_bytes": 10240,
      "dtype": "BF16",
      "shapes": [
        [
          5120
        ]
      ]
    },
    "mtp.pre_fc_norm_hidden": {
      "level1": "mtp",
      "n_tensors": 1,
      "n_parameters": 5120,
      "n_bytes": 10240,
      "dtype": "BF16",
      "shapes": [
        [
          5120
        ]
      ]
    },
    "mtp.input_layernorm": {
      "level1": "mtp",
      "n_tensors": 1,
      "n_parameters": 5120,
      "n_bytes": 10240,
      "dtype": "BF16",
      "shapes": [
        [
          5120
        ]
      ]
    },
    "mtp.post_attention_layernorm": {
      "level1": "mtp",
      "n_tensors": 1,
      "n_parameters": 5120,
      "n_bytes": 10240,
      "dtype": "BF16",
      "shapes": [
        [
          5120
        ]
      ]
    },
    "mtp.self_attn.q_proj": {
      "level1": "mtp",
      "n_tensors": 1,
      "n_parameters": 62914560,
      "n_bytes": 125829120,
      "dtype": "BF16",
      "shapes": [
        [
          12288,
          5120
        ]
      ]
    },
    "mtp.self_attn.k_proj": {
      "level1": "mtp",
      "n_tensors": 1,
      "n_parameters": 5242880,
      "n_bytes": 10485760,
      "dtype": "BF16",
      "shapes": [
        [
          1024,
          5120
        ]
      ]
    },
    "mtp.self_attn.v_proj": {
      "level1": "mtp",
      "n_tensors": 1,
      "n_parameters": 5242880,
      "n_bytes": 10485760,
      "dtype": "BF16",
      "shapes": [
        [
          1024,
          5120
        ]
      ]
    },
    "mtp.self_attn.o_proj": {
      "level1": "mtp",
      "n_tensors": 1,
      "n_parameters": 31457280,
      "n_bytes": 62914560,
      "dtype": "BF16",
      "shapes": [
        [
          5120,
          6144
        ]
      ]
    },
    "mtp.self_attn.q_norm": {
      "level1": "mtp",
      "n_tensors": 1,
      "n_parameters": 256,
      "n_bytes": 512,
      "dtype": "BF16",
      "shapes": [
        [
          256
        ]
      ]
    },
    "mtp.self_attn.k_norm": {
      "level1": "mtp",
      "n_tensors": 1,
      "n_parameters": 256,
      "n_bytes": 512,
      "dtype": "BF16",
      "shapes": [
        [
          256
        ]
      ]
    },
    "mtp.mlp.gate_proj": {
      "level1": "mtp",
      "n_tensors": 1,
      "n_parameters": 89128960,
      "n_bytes": 178257920,
      "dtype": "BF16",
      "shapes": [
        [
          17408,
          5120
        ]
      ]
    },
    "mtp.mlp.up_proj": {
      "level1": "mtp",
      "n_tensors": 1,
      "n_parameters": 89128960,
      "n_bytes": 178257920,
      "dtype": "BF16",
      "shapes": [
        [
          17408,
          5120
        ]
      ]
    },
    "mtp.mlp.down_proj": {
      "level1": "mtp",
      "n_tensors": 1,
      "n_parameters": 89128960,
      "n_bytes": 178257920,
      "dtype": "BF16",
      "shapes": [
        [
          5120,
          17408
        ]
      ]
    },
    "vision.patch_embed": {
      "level1": "vision",
      "n_tensors": 2,
      "n_parameters": 1770624,
      "n_bytes": 3541248,
      "dtype": "BF16",
      "shapes": [
        [
          1152
        ],
        [
          1152,
          3,
          2,
          16,
          16
        ]
      ]
    },
    "vision.pos_embed": {
      "level1": "vision",
      "n_tensors": 1,
      "n_parameters": 2654208,
      "n_bytes": 5308416,
      "dtype": "BF16",
      "shapes": [
        [
          2304,
          1152
        ]
      ]
    },
    "vision.blocks": {
      "level1": "vision",
      "n_tensors": 324,
      "n_parameters": 411466608,
      "n_bytes": 822933216,
      "dtype": "BF16",
      "shapes": [
        [
          1152
        ],
        [
          1152,
          1152
        ],
        [
          1152,
          4304
        ],
        [
          3456
        ],
        [
          3456,
          1152
        ],
        [
          4304
        ],
        [
          4304,
          1152
        ]
      ]
    },
    "vision.merger": {
      "level1": "vision",
      "n_tensors": 6,
      "n_parameters": 44838656,
      "n_bytes": 89677312,
      "dtype": "BF16",
      "shapes": [
        [
          1152
        ],
        [
          4608
        ],
        [
          4608,
          4608
        ],
        [
          5120
        ],
        [
          5120,
          4608
        ]
      ]
    }
  }
}
```
