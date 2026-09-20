# TASK-01 — Establish authoritative model facts

## Control

- Primary ID: `TASK-01`
- Coupled IDs: `none`
- Dependencies: `TASK-00` (DONE at admission)
- Status: `DONE`
- Ledger acceptance: Document observed structure and state implications; inventory tensors by semantic family with parameter and BF16 byte totals; verify totals against the checkpoint.

## Goal and boundaries

Produce `docs/architecture/model-inventory.md` as the Phase 1 source of truth for Qwen3.8-27B configuration, layer ordering, implied state structures, and a complete BF16 tensor inventory from the Transformers checkpoint at `.cache/authorities/qwen3.8-27b-transformers`.

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - The BF16 checkpoint (`config.json`, `model.safetensors.index.json`, 18 `model-*-of-00018.safetensors` shards) is the only weight authority.
  - Label inventory claims `OBSERVED` or `DERIVED` per the plan evidence policy. Do not inspect Quartz execution/CUDA code, llama.cpp/GGML Qwen construction or kernels, or `models/Qwen3.8-27B-Q4_K_M.gguf`.
  - Do not load safetensor payloads; headers plus the index are sufficient and required.
  - Vision is present in the checkpoint (`language_model_only: false`) but this study’s central path is Qwen3.8 language mathematics: inventory vision as present-but-deferred (coarse totals only).
  - MTP (multi-token prediction) is language-side and is inventoried in detail, not deferred.
- Non-goals:
  - No forward equations, algebraic alternatives, or kernel/graph/layout decisions (TASK-02+).
  - No lifetime/traffic quantification (TASK-04) beyond naming implied state ranks from config and tensor shapes.
  - No weight-distribution statistics (TASK-05).
  - No vision encoder mathematics, preprocessor algorithm, chat template, or tokenizer-merge internals.
  - No GGUF, quantization, or runtime format work.
  - No `pyproject.toml` / uv package / Ruff / pytest suite in this increment (stdlib script only).
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib inventory script used only as evidence tooling)

## Repository evidence

Planning inspected checkpoint metadata and all 18 safetensor JSON headers (payloads unread). Implementation must recompute; if a fresh run disagrees, stop.

- `docs/architecture/plan.md:35-52` — BF16 Transformers checkpoint is the high-precision authority; Quartz/llama.cpp inspection forbidden; tokenizer assets allowed.
- `docs/architecture/plan.md:54-64` — OBSERVED / DERIVED evidence labels.
- `docs/architecture/plan.md:78-79` — TASK-01 inventories the checkpoint and derives dimensions and parameter/byte totals.
- `docs/architecture/task_ledger.md` TASK-01 row — produces `docs/architecture/model-inventory.md`; completion is structure, family inventory with param/byte totals, and checkpoint verification.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — architecture `Qwen3_5ForConditionalGeneration`, `model_type` `qwen3_5`, `language_model_only: false`; `text_config` 64 layers, `layer_types` 48× `linear_attention` + 16× `full_attention` with `full_attention_interval: 4`; hidden 5120, intermediate 17408, vocab 248320, head_dim 256, 24/4 GQA, linear heads 16/48 at dim 128, conv kernel 4, `attn_output_gate: true`, `partial_rotary_factor: 0.25`, `mamba_ssm_dtype: float32`, `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false`, `tie_word_embeddings: false`; `vision_config` depth 27, hidden 1152, out 5120.
- `.cache/authorities/qwen3.8-27b-transformers/model.safetensors.index.json` — `weight_map` has 1199 keys; `metadata.total_size` is JSON float `55562855904.0`; 18 unique shard filenames.
- Safetensor headers of those 18 shards — 1199 tensors, every dtype `BF16`, header names equal `weight_map` keys, parameter count 27,781,427,952, payload bytes 55,562,855,904 matching `int(total_size)`. Language modules match `layer_types` (no mixed `linear_attn`/`self_attn` on one layer). Full-attention layer indices: 3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63.
- Tensor name prefixes — language under `model.language_model.*`, vision under `model.visual.*` (333 tensors), MTP under `mtp.*` (15 tensors), plus untied `lm_head.weight`.
- Shape vs config (planning DERIVED, re-assert in script): `q_proj` is `(12288, 5120)` = `2 * 24 * 256 × H` while `o_proj` is `(5120, 6144)` = `H × 24 * 256`, consistent with `attn_output_gate`; linear `in_proj_qkv` is `(10240, 5120)` = `(16+16+48)*128`; no RoPE `inv_freq` tensor; language attention has no bias tensors; vision blocks have biases.
- `.gitignore` — `.cache/` is untracked; the checkpoint is local authority, not a git artifact. Missing checkpoint is `incomplete`/`blocked`, not an inventory-content failure.

## Performance evidence

N/A — documentation and checkpoint-metadata inventory; no prefill/decode/component timing, no keep/reject, no sink ranking.

## Implementation decisions

### Deliverable structure (`docs/architecture/model-inventory.md`)

Use these sections in this order. Compact tables over prose. Every numeric claim is `OBSERVED` (read from files) or `DERIVED` (arithmetic from OBSERVED values). Record `UNKNOWN` only for mathematical meaning that TASK-02 owns (for example how the output gate is applied, exact linear-attention recurrence).

1. **Authority** — checkpoint relative path; files used (`config.json`, index, 18 shard headers); evidence labels; in-scope (language + MTP) vs deferred (vision encoder and its preprocessor/video configs).
2. **Configuration** — tables of every `text_config` key; top-level non-vision keys (`architectures`, `model_type`, `language_model_only`, token ids, `tie_word_embeddings`, `transformers_version`); `vision_config` as a single deferred table (do not expand into a vision spec). Note `generation_config.json` eos-id list `[248046, 248044]` vs `text_config.eos_token_id` `248044` as OBSERVED metadata only.
3. **Layer ordering** — `layer_types` length 64 equals `num_hidden_layers`; repeating pattern three `linear_attention` then one `full_attention` (`i % 4 == 3`); list of 16 full-attention indices; MTP is one extra full-attention+MLP block outside `layer_types`, after the language stack.
4. **State implications** (structural only):
   - Full attention: token-persistent KV candidates from `k_proj`/`v_proj` shapes `(1024, 5120)` = 4×256, 16 such layers plus MTP self-attn; `use_cache: true`; RoPE from config only (`rope_parameters`, `partial_rotary_factor` 0.25 ⇒ DERIVED rotary dim 64); `mrope_section` `[11, 11, 10]` OBSERVED (sum 32); relationship to rotary dim is `UNKNOWN` until TASK-02. No RoPE weights in the checkpoint.
   - Linear attention: 48 layers; `conv1d.weight` `(10240, 1, 4)` implies a length-`kernel-1` convolution delay along the 10240-wide qkv stream; `A_log` and `dt_bias` `(48,)` = `linear_num_value_heads` are recurrent parameters; `mamba_ssm_dtype: float32` is a config-stated runtime state dtype, not a checkpoint dtype (all weights BF16). Exact SSM/delta recurrence is TASK-02.
   - Output gate: config `attn_output_gate: true` / `output_gate_type: swish` plus doubled `q_proj` / MTP `q_proj`; no dedicated gate tensor. Application rule is TASK-02.
   - Embeddings untied: distinct `embed_tokens` and `lm_head`, both `(248320, 5120)`; MTP `mtp_use_dedicated_embeddings: false` so MTP shares those.
   - Vision: no language-decode state inventory; note only that visual tokens would enter the language residual as 5120-d vectors via `merger` (`out_hidden_size` 5120). Deferred.
5. **Semantic-family inventory** — level-1 rollup table and level-2 table (count, unique shapes, parameters, BF16 bytes, dtype). Name patterns use `{i}` for layer/block index. Do not dump all 1199 rows.
6. **Checkpoint verification** — tensor count, shard count, all-BF16, parameter total, byte total, `int(index.metadata.total_size)` equality, header-name set equality with `weight_map`, `layer_types` vs module names. Shard file sizes may exceed payload bytes by JSON-header overhead; verification target is payload bytes, not file size.
7. **Method** — script path, header-only parse, family mapping reference.
8. **Machine-checkable totals** — exactly one fenced `json` code block in the file, containing the object defined below, copied from a fresh script run (pretty-printed, key order as emitted by the script).

### Tooling

Create `scripts/inventory_bf16_checkpoint.py` (Python 3.11+, stdlib only: `argparse`, `json`, `struct`, `pathlib`, `re`, `sys`, `typed` annotations, Google docstrings). Do not add `safetensors`, PyTorch, uv, `pyproject.toml`, Ruff, or pytest.

Parse each shard as: uint64-le header length, then UTF-8 JSON; drop `__metadata__`; record `dtype`, `shape`, `data_offsets`. Byte size = `end - start` and must equal `numel * 2` for `BF16`.

CLI (cwd = repository root):

```text
python3 scripts/inventory_bf16_checkpoint.py \
  --checkpoint .cache/authorities/qwen3.8-27b-transformers \
  [--json] \
  [--check-inventory docs/architecture/model-inventory.md]
```

- Default / `--json`: write the totals JSON object to stdout and run internal asserts (below). Exit 0 on success.
- `--check-inventory PATH`: parse the first fenced `json` block in PATH (` ```json ` … ` ``` `), compare to a live inventory of `--checkpoint` (all keys and nested family numeric fields and `shapes`). Exit 1 with a readable diff on mismatch.
- Missing checkpoint directory or shard: exit 2 and print the missing path (verifier treats this as blocked, not a content fail).

### Family mapping (exhaustive)

Every tensor maps to exactly one **level-2** id. Unmapped or double-mapped names fail the script.

**Language** (`model.language_model.layers.{i}.…` unless noted):

| level-2 id | name pattern | level-1 |
|---|---|---|
| `embed` | `model.language_model.embed_tokens.weight` | `language_embed` |
| `final_norm` | `model.language_model.norm.weight` | `language_final_norm` |
| `lm_head` | `lm_head.weight` | `language_lm_head` |
| `input_layernorm` | `…layers.{i}.input_layernorm.weight` | `language_layer_norms` |
| `post_attention_layernorm` | `…layers.{i}.post_attention_layernorm.weight` | `language_layer_norms` |
| `linear_attn.A_log` | `…linear_attn.A_log` | `language_linear_attn` |
| `linear_attn.conv1d` | `…linear_attn.conv1d.weight` | `language_linear_attn` |
| `linear_attn.dt_bias` | `…linear_attn.dt_bias` | `language_linear_attn` |
| `linear_attn.in_proj_a` | `…linear_attn.in_proj_a.weight` | `language_linear_attn` |
| `linear_attn.in_proj_b` | `…linear_attn.in_proj_b.weight` | `language_linear_attn` |
| `linear_attn.in_proj_qkv` | `…linear_attn.in_proj_qkv.weight` | `language_linear_attn` |
| `linear_attn.in_proj_z` | `…linear_attn.in_proj_z.weight` | `language_linear_attn` |
| `linear_attn.norm` | `…linear_attn.norm.weight` | `language_linear_attn` |
| `linear_attn.out_proj` | `…linear_attn.out_proj.weight` | `language_linear_attn` |
| `self_attn.q_proj` | `…self_attn.q_proj.weight` | `language_self_attn` |
| `self_attn.k_proj` | `…self_attn.k_proj.weight` | `language_self_attn` |
| `self_attn.v_proj` | `…self_attn.v_proj.weight` | `language_self_attn` |
| `self_attn.o_proj` | `…self_attn.o_proj.weight` | `language_self_attn` |
| `self_attn.q_norm` | `…self_attn.q_norm.weight` | `language_self_attn` |
| `self_attn.k_norm` | `…self_attn.k_norm.weight` | `language_self_attn` |
| `mlp.gate_proj` | `…mlp.gate_proj.weight` | `language_mlp` |
| `mlp.up_proj` | `…mlp.up_proj.weight` | `language_mlp` |
| `mlp.down_proj` | `…mlp.down_proj.weight` | `language_mlp` |

**MTP** (strip `mtp.layers.0.` to the `mtp.` + remainder without `.weight`):

| level-2 id | name pattern | level-1 |
|---|---|---|
| `mtp.fc` | `mtp.fc.weight` | `mtp` |
| `mtp.norm` | `mtp.norm.weight` | `mtp` |
| `mtp.pre_fc_norm_embedding` | `mtp.pre_fc_norm_embedding.weight` | `mtp` |
| `mtp.pre_fc_norm_hidden` | `mtp.pre_fc_norm_hidden.weight` | `mtp` |
| `mtp.input_layernorm` | `mtp.layers.0.input_layernorm.weight` | `mtp` |
| `mtp.post_attention_layernorm` | `mtp.layers.0.post_attention_layernorm.weight` | `mtp` |
| `mtp.self_attn.q_proj` | `mtp.layers.0.self_attn.q_proj.weight` | `mtp` |
| `mtp.self_attn.k_proj` | `mtp.layers.0.self_attn.k_proj.weight` | `mtp` |
| `mtp.self_attn.v_proj` | `mtp.layers.0.self_attn.v_proj.weight` | `mtp` |
| `mtp.self_attn.o_proj` | `mtp.layers.0.self_attn.o_proj.weight` | `mtp` |
| `mtp.self_attn.q_norm` | `mtp.layers.0.self_attn.q_norm.weight` | `mtp` |
| `mtp.self_attn.k_norm` | `mtp.layers.0.self_attn.k_norm.weight` | `mtp` |
| `mtp.mlp.gate_proj` | `mtp.layers.0.mlp.gate_proj.weight` | `mtp` |
| `mtp.mlp.up_proj` | `mtp.layers.0.mlp.up_proj.weight` | `mtp` |
| `mtp.mlp.down_proj` | `mtp.layers.0.mlp.down_proj.weight` | `mtp` |

**Vision (deferred, still counted):**

| level-2 id | name prefix | level-1 |
|---|---|---|
| `vision.patch_embed` | `model.visual.patch_embed.` | `vision` |
| `vision.pos_embed` | `model.visual.pos_embed.` | `vision` |
| `vision.blocks` | `model.visual.blocks.` | `vision` |
| `vision.merger` | `model.visual.merger.` | `vision` |

Level-1 ids are exactly: `language_embed`, `language_final_norm`, `language_lm_head`, `language_layer_norms`, `language_linear_attn`, `language_self_attn`, `language_mlp`, `mtp`, `vision`. Level-1 parameter/byte totals in the markdown are sums of their level-2 rows. Grand totals are sums of all level-2 rows.

### Totals JSON schema

Top-level keys (all required): `authority` (string, exactly `.cache/authorities/qwen3.8-27b-transformers`), `n_tensors` (int), `n_shards` (int), `dtype` (string, exactly `BF16`), `n_parameters` (int), `n_bytes` (int), `index_metadata_total_size` (int, `int(metadata.total_size)`), `families` (object).

Each `families` entry is keyed by a level-2 id and has: `level1` (string), `n_tensors` (int), `n_parameters` (int), `n_bytes` (int), `dtype` (exactly `BF16`), `shapes` (sorted unique shapes as arrays of ints).

`families` must contain every level-2 id above and no extras. All counts and sizes are live checkpoint values, not schema placeholders. JSON numbers must be integers, not floats.

### Script asserts (fail closed)

Using `text_config` as `H, I, V, …`:

- `len(layer_types) == num_hidden_layers == 64`; counts 48 / 16; every index `i % 4 == 3` is `full_attention`, others `linear_attention`.
- Layer `i` has `linear_attn.*` xor `self_attn.*` matching `layer_types[i]`.
- All dtypes `BF16`; `n_bytes == n_parameters * 2 == index_metadata_total_size`; header names == `weight_map` keys; 18 shards.
- Shape equalities: embed/lm_head `(V,H)`; final/input/post norms `(H,)`; mlp gate/up `(I,H)`, down `(H,I)`; self-attn (and MTP self-attn) `q_proj` `(2*24*256, H)`, `k_proj`/`v_proj` `(4*256, H)`, `o_proj` `(H, 24*256)`, `q_norm`/`k_norm` `(256,)`; linear `in_proj_qkv` `(10240,H)`, `in_proj_z`/`out_proj` complementary `(6144,H)` / `(H,6144)`, `conv1d` `(10240,1,4)`, `A_log`/`dt_bias` `(48,)`, `in_proj_a`/`in_proj_b` `(48,H)`, `norm` `(128,)`; `mtp.fc` `(H, 2*H)`; MTP mlp/norms match one language layer.
- Family occupancy: linear_* families 48 tensors; self_attn_* 16; mlp_* and language layernorms 64; each mtp.* family 1; `vision.blocks` 324 tensors (27×12); vision total 333.

### Stage split

- **Implementation** writes the script, runs `--json`, records stdout in this dossier. Does not write `model-inventory.md`.
- **Documentation** writes `docs/architecture/model-inventory.md` from `config.json` plus that JSON (and re-runnable script). Embeds the JSON fence verbatim. Draft conclusions stay `unverified`.
- **Verification** independently re-runs the focused commands, checks the diff against this dossier and the ledger row, and traces every markdown total to script output or inspected checkpoint fields.

- Invariants:
  - Family partition is exhaustive and disjoint.
  - Byte totals reconcile to the index.
  - Vision remains deferred in prose while still included in grand totals.
  - No Quartz/llama.cpp/GGUF evidence.
- Rejected alternatives:
  - Index-only inventory: rejected; the index lacks shapes/dtypes.
  - `safetensors`/`torch` load: rejected; headers suffice and payloads are ~52 GiB.
  - uv/`src/` package + Ruff/pytest: rejected for this docs-only increment; stdlib script is enough.
  - Deferring MTP with vision: rejected; MTP is a language residual block with full-attention tensors.
  - Omitting vision from grand totals: rejected; checkpoint verification must cover all 1199 tensors.
  - Per-tensor 1199-row markdown dump: rejected; uniform shapes per pattern make family tables complete.
- Discovered ledger work: `none`
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/model-inventory.md` exists and follows the section list above.
  - Structure, `layer_types` ordering, and state implications are documented as specified (no forward math).
  - Level-1 and level-2 family tables report parameter and BF16 byte totals; grand totals equal the checkpoint.
  - The JSON fence matches a live script inventory of the BF16 checkpoint.
  - Coupled IDs: none.
- Tests/fixtures to add or change: `scripts/inventory_bf16_checkpoint.py` only (no pytest fixtures).
- Focused commands (repository root; checkpoint must exist):

```sh
python3 -m py_compile scripts/inventory_bf16_checkpoint.py
python3 scripts/inventory_bf16_checkpoint.py \
  --checkpoint .cache/authorities/qwen3.8-27b-transformers \
  --json
python3 scripts/inventory_bf16_checkpoint.py \
  --checkpoint .cache/authorities/qwen3.8-27b-transformers \
  --check-inventory docs/architecture/model-inventory.md
```

- Candidate quality: not required — no model execution or NLL.
- Repository-wide commands:

```sh
test -f docs/architecture/model-inventory.md
python3 -m py_compile scripts/inventory_bf16_checkpoint.py
python3 scripts/inventory_bf16_checkpoint.py \
  --checkpoint .cache/authorities/qwen3.8-27b-transformers \
  --check-inventory docs/architecture/model-inventory.md
```

Do not run Ruff, pytest, CMake, or CUDA; this increment does not introduce those gates.

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Checkpoint presence is a local-file requirement, not a GPU gate.
- Documentation/evidence updates:
  - `docs/architecture/model-inventory.md` (create)
  - `scripts/inventory_bf16_checkpoint.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged)
- Definition of done: inventory document published, JSON fence verifies against the sitting BF16 checkpoint, ledger TASK-01 completion checkboxes can be marked at delivery.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high`
- UTC/time/tokens/cost: `2026-09-20T09:06:27Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-01.md`. Coupled IDs none. Inventory structure, family taxonomy, stdlib header-parse tooling, vision-deferred vs MTP-in-scope, and acceptance commands are closed.
- Performance evidence applied: N/A (documentation / metadata inventory; no timing)

### Implementation

- Agent/model: `cursor-grok-4.6-high`
- Changes:
  - Created `scripts/inventory_bf16_checkpoint.py` (stdlib header-only inventory; family mapping; `--json` and `--check-inventory`; fail-closed asserts). Did not write `docs/architecture/model-inventory.md`.
- Commands:
  - `python3 -m py_compile scripts/inventory_bf16_checkpoint.py` — **pass** (exit 0, empty stdout/stderr)
  - `python3 scripts/inventory_bf16_checkpoint.py --checkpoint .cache/authorities/qwen3.8-27b-transformers --json` — **pass** (exit 0). Salient totals: `n_tensors=1199`, `n_shards=18`, `dtype=BF16`, `n_parameters=27781427952`, `n_bytes=55562855904`, `index_metadata_total_size=55562855904` (agrees with planning). `--check-inventory` not run: documentation stage owns `model-inventory.md`.
- Captured `--json` stdout (verbatim; documentation must embed this object as the markdown JSON fence):

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

- UTC/time/tokens/cost: `2026-09-20T09:13:15Z`; `telemetry_unavailable`

### Documentation

- Agent/model: `cursor-composer` (documentation subagent)
- Changes:
  - Created `docs/architecture/model-inventory.md` — eight sections in dossier order (Authority, Configuration, Layer ordering, State implications, Semantic-family inventory, Checkpoint verification, Method, Machine-checkable totals JSON fence).
  - Evidence: `config.json`, `generation_config.json`, implementation-stage `--json` stdout (embedded verbatim), level-1/level-2 tables traced to script families.
  - Vision deferred in prose; MTP detailed; grand totals include vision (27,781,427,952 parameters / 55,562,855,904 BF16 bytes across 1199 tensors).
- Commands:
  - `python3 scripts/inventory_bf16_checkpoint.py --checkpoint .cache/authorities/qwen3.8-27b-transformers --json` — **pass** (exit 0; confirms implementation JSON still matches checkpoint).
  - `python3 scripts/inventory_bf16_checkpoint.py --checkpoint .cache/authorities/qwen3.8-27b-transformers --check-inventory docs/architecture/model-inventory.md` — **pass** (exit 0; JSON fence matches live inventory).
- Draft conclusions: `unverified` (verification stage has not run; no verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T09:16:00Z`; `telemetry_unavailable`

### Verification

- Attempt: 1
- Agent/model: `cursor-composer` (integration verifier subagent)
- Diff review:
  - `scripts/inventory_bf16_checkpoint.py` — stdlib header-only inventory; exhaustive level-2 family mapping; `--json` and `--check-inventory`; fail-closed asserts match dossier.
  - `docs/architecture/model-inventory.md` — eight sections in dossier order; OBSERVED/DERIVED labels; level-1/level-2 family tables; JSON fence; vision deferred in prose, MTP detailed.
  - `docs/architecture/tasks/TASK-01.md` — implementation and documentation run records present; verification record appended.
  - `docs/architecture/task_ledger.md` — status `TODO` → `IN PROGRESS` only (expected).
  - `docs/architecture/plan.md` — unchanged.
  - No Quartz/llama.cpp/GGUF inspection artifacts in deliverables.
- Independent raw-record checks:
  - Level-1 rollup from live `--json` matches markdown grand totals (1199 tensors, 27781427952 parameters, 55562855904 BF16 bytes).
  - Spot-check: `language_linear_attn`, `language_mlp`, `mtp`, `vision` level-1 rows match live family sums.
- Commands:
  - `python3 -m py_compile scripts/inventory_bf16_checkpoint.py` — **pass** (exit 0)
  - `python3 scripts/inventory_bf16_checkpoint.py --checkpoint .cache/authorities/qwen3.8-27b-transformers --json` — **pass** (exit 0; totals match dossier implementation capture)
  - `python3 scripts/inventory_bf16_checkpoint.py --checkpoint .cache/authorities/qwen3.8-27b-transformers --check-inventory docs/architecture/model-inventory.md` — **pass** (exit 0; JSON fence matches live inventory)
  - `test -f docs/architecture/model-inventory.md` — **pass** (exit 0)
- Formatting changed files: none
- Verdict: **pass** — deliverables satisfy dossier acceptance and ledger completion criteria; JSON fence verifies against sitting BF16 checkpoint.
- UTC/time/tokens/cost: `2026-09-20T09:17:00Z`; `telemetry_unavailable`

### Retries and escalation

none

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification pass — inventory document, family totals, and JSON fence reconcile to checkpoint
- Candidate measured delta: N/A (no throughput work)
- Shipping delta: N/A (diagnostics/documentation)
- Quality result: not required
- Evidence completeness: N/A (no performance-evidence checks)
- Throughput delta: N/A — TASK-01 does not execute or time the model
- Commit: `e522a7a` — Establish authoritative Qwen3.8 model facts
- Push: `origin/clean-sheet` (`5b8dee5..e522a7a`, success)
- First-pass acceptance: **verified**
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: none identified; local `.cache/` checkpoint must remain present for focused commands
