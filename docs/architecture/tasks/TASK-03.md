# TASK-03 — Build the logical dataflow graph

## Control

- Primary ID: `TASK-03`
- Coupled IDs: `none`
- Dependencies: `TASK-02` (DONE at admission)
- Status: `DONE`
- Ledger acceptance: Diagram all required model regions and token-to-token state; tabulate significant logical intermediates and their consumers; state that logical values do not imply physical allocation.

## Goal and boundaries

Produce `docs/architecture/dataflow.md` as the Phase 1 logical DAG for the Qwen3.8-27B language + MTP forward map: producers, consumers, fan-out, and token-to-token state transitions derived from `docs/architecture/model-semantics.md`. Close the ledger open question by naming which logical results have the most consequential sharing and reuse, as **DERIVED** graph structure, not as a materialization or performance claim.

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - Equations, ranks, and state kinds come from `docs/architecture/model-semantics.md` (TASK-02). Dimensions instantiate `docs/architecture/model-inventory.md` / sitting `text_config`. Do not invent operators, re-inventory tensors, or rewrite forward math.
  - Label claims `OBSERVED` (config/inventory already established), `DERIVED` (producer/consumer/fan-out from TASK-02 equations), or `UNKNOWN` only for vision-encoder internals deferred here.
  - GitHub Markdown. Diagrams are Mermaid `flowchart` fences in the markdown deliverable. Cite TASK-02 equation tags `(1)`–`(24)` rather than repeating derivations.
  - Allowed evidence: TASK-02 semantics, TASK-01 inventory, sitting `config.json`, and the plan evidence vocabulary. Hugging Face / paper sources are already resolved in TASK-02; do not reopen gate/RoPE/GDN identities.
  - Do not inspect Quartz execution/CUDA code, llama.cpp/GGML Qwen construction or kernels, or `models/Qwen3.8-27B-Q4_K_M.gguf`.
- Non-goals:
  - No lifetime class, recomputability verdict, or byte traffic (TASK-04). Token-crossing edges may be named; bytes and “must materialize” may not.
  - No FLOP / arithmetic-intensity bounds (TASK-06).
  - No semantic-graph execution contracts (TASK-11). Regions here are **model regions**, not CUDA or compiler nodes.
  - No fusion, buffer reuse, or physical-allocation decisions (TASK-12).
  - No layouts, schedules, kernel names, or decode/prefill **execution** plans (TASK-13+). Prefill vs decode is the same DAG with different $T$ and incoming state.
  - No new equations, algebraic alternatives, or vision-encoder internals.
  - No 64-layer unrolling, Graphviz/DOT files, Archify HTML, or sequence diagrams.
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only, matching TASK-01/02).
  - Do not edit `docs/architecture/model-semantics.md`, `docs/architecture/model-inventory.md`, or `docs/architecture/plan.md`.
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib graph-summary checker used only as evidence tooling)

## Repository evidence

- `docs/architecture/plan.md:9-13` — central path: mathematics → **logical dataflow** → lifetime/state.
- `docs/architecture/plan.md:81-83` — TASK-03 translates the TASK-02 specification into a logical DAG.
- `docs/architecture/task_ledger.md` TASK-03 row — produces `docs/architecture/dataflow.md`; completion is region+state diagrams, intermediate/consumer table, and the logical-≠-physical statement. Open question to close: which logical results have the most consequential sharing and reuse.
- `docs/architecture/task_ledger.md` TASK-04/11/12 rows — downstream consumers of this DAG (lifetimes, semantic nodes, materialization). Do not perform those analyses here.
- `docs/architecture/tasks/TASK-02.md` — locked equations, state ranks $K,V,C,S$, algebraic equivalents (same map, not extra nodes), and explicit non-goal “no logical DAG drawing.”
- `docs/architecture/model-semantics.md` — mathematical authority: residual wrapper (4)–(5), Gated Attention (6)–(10), mRoPE (11)–(12), GDN (13)–(20), MLP (21), state table, primary logits (22), MTP (23)–(24). Prefill is the map from zero state; decode is the same map with $T=1$ plus incoming state.
- `docs/architecture/model-inventory.md` — 64 layers, 48 linear + 16 full at indices 3,7,…,63; MTP one full-attention block; untied $E$ / $W_\text{lm}$; vision merger interface 5120-d.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — live `text_config` for checker arithmetic (`num_hidden_layers` 64, `full_attention_interval` 4, linear/full head counts). Do not read safetensor payloads.
- `scripts/check_model_semantics.py` — precedent for stdlib heading/JSON-fence/forbidden-token checks. TASK-03’s checker is a sibling script, not an import of that file.

## Performance evidence

N/A — logical DAG documentation; no prefill/decode/component timing, no keep/reject, no sink ranking.

## Implementation decisions

### Authority for the DAG

The graph is the TASK-02 inference-time language+MTP map drawn as named values and edges. If a drawing would disagree with an equation, the equation wins and the drawing is wrong.

- Node set = locked catalog IDs below (activations + persistent state). Weights are drawn where they clarify a producer but are **not** catalog intermediates except as the separate shared-parameter pair $E$, $W_\text{lm}$.
- Algebraic equivalents in TASK-02 (chunkwise GDN, SDPA, GQA-as-repeat, RoPE complex form, $S$ vs $S^\top$, omitting MTP when only $\ell^{(0)}$ is required) are **not** extra nodes.
- Prefill and decode share one DAG. Incoming $(K,V,C,S)$ is zeros vs populated; $T>1$ vs $T=1$. Do not duplicate region diagrams per mode.
- Disjoint splits (last-axis $q\|g$, QKV channel split) are partitions, not reuse. Consequential sharing is defined below.

### Deliverable structure (`docs/architecture/dataflow.md`)

Use these **level-2 headings in this order**. Compact tables + Mermaid fences + short captions. Every numeric rank is TASK-02 / config `OBSERVED` or `DERIVED`. Do not leave `TBD`. The only `UNKNOWN` allowed is vision-encoder internals, isolated in Deferred vision.

1. **Authority** — this dossier, `model-semantics.md`, inventory, config, checker; evidence labels; in-scope (language + MTP logical DAG) vs deferred (vision encoder). State that the document specifies producers/consumers, not kernels.
2. **Logical versus physical** — the canonical sentence (exact text below) plus three bullets: fan-out ≠ must-store; token-crossing names mathematical state, not cache layout; the DAG is not an execution schedule.
3. **Graph vocabulary** — node kinds, edge kinds, fan-out / live-across / split rules, region IDs (table below).
4. **Top-level token map** — one Mermaid flowchart (diagram 1).
5. **Residual decoder layer** — one Mermaid flowchart (diagram 2); mixer is GDN xor Gated Attention by `layer_types`.
6. **Full attention** — one Mermaid flowchart (diagram 3) including KV dashed state edges.
7. **Linear attention** — one Mermaid flowchart (diagram 4) including $C$ and $S$ dashed state edges.
8. **MLP** — one Mermaid flowchart (diagram 5); show `h_post` fan-out to gate and up.
9. **Persistent state transitions** — one Mermaid flowchart (diagram 6) of token $t-1 \to t$ for $K,V,C,S$.
10. **Primary logits and MTP** — one Mermaid flowchart (diagram 7); shared $E$ / $W_\text{lm}$ as stadium nodes.
11. **Logical intermediates** — the locked catalog table (all 52 IDs).
12. **Sharing and reuse** — one Mermaid flowchart (diagram 8) of high-fan-out / live-across / shared weights; ranked DERIVED list that closes the ledger open question.
13. **Deferred vision** — residual-stream interface node only.
14. **Machine-checkable graph summary** — one fenced `json` object copied from a fresh checker `--json` run.

Immediately under **Logical versus physical**, include this **canonical sentence verbatim** (checker substring match):

> Logical values in this document are graph nodes. They do not imply physical allocation, materialization, buffer reuse, or kernel fusion.

### Graph vocabulary (lock)

| Kind | Mermaid shape | Meaning |
| --- | --- | --- |
| value | `id[label]` rectangle | Named logical tensor (catalog ID) |
| weight | `id([label])` stadium | Checkpoint parameter; not a catalog intermediate |
| state | `id[(label)]` cylinder | Token-persistent catalog node $K,V,C,S$ |
| region | `subgraph` | Model region boundary, not a tensor |

Edge kinds: solid `-->` data; dashed `-.->` state read/write or token-crossing. Put the producing operator on the edge label (for example `\|W_q\|`, `\|RMSNorm\|`, `\|Eq 17\|`). Do not add a separate “fan-out” edge type.

**Fan-out** of a value = number of distinct catalog consumer IDs that **read the same value** (including a state-write consumer). A disjoint partition counts as producing child nodes, not as reuse of the parent; do not place split parents in `high_fanout_ids`.

**Live-across** = fan-out 1 but the unique consumer is not the immediate successor operator (other catalog nodes are produced in between). Locked live-across IDs: `g` (consumed after attention), `z` (consumed after recurrence).

**Internal uses** = times one equation reads a value to produce a single consumer. Locked: `k_hat` is used twice in Eq. (17) to produce `S`.

**Token-crossing** = an edge into the next token’s map (`K_state`, `V_state`, `C_state`, `S`).

Do not use Mermaid `sequenceDiagram` or `stateDiagram-v2`. Do not set `%%{init:...}%%` themes. Do not unroll 64 layers. Node IDs in fences **must equal catalog IDs** (and `vision_if`, `E`, `W_lm` where drawn).

### Region IDs (lock)

| Region ID | What it contains | Drawn in |
| --- | --- | --- |
| `embed` | `token_id` → `e` | diagram 1 |
| `decoder_stack` | 64 residual layers, 3:1 GDN:GA | diagram 1 (collapsed) |
| `residual_layer` | Eqs. (4)–(5) template | diagram 2 |
| `full_attn` | Eqs. (6)–(10) + KV | diagram 3 |
| `linear_attn` | Eqs. (13)–(20) + $C,S$ | diagram 4 |
| `mlp` | Eq. (21) | diagram 5 |
| `persistent_state` | $K,V,C,S$ token update | diagram 6 |
| `primary_logits` | Eq. (22) | diagrams 1 and 7 |
| `mtp` | Eqs. (23)–(24) | diagram 7 |
| `vision_interface` | optional replace into $h^{(0)}$ | diagram 1 as `vision_if` |

### Diagram format and required node IDs

Exactly **eight** fenced `mermaid` blocks, in heading order 4,5,6,7,8,9,10,12. Each fence body starts with `flowchart TB` or `flowchart LR` (either direction is allowed per diagram). Captions sit in markdown above the fence, not inside Mermaid.

| # | Heading | Required IDs in that fence |
| --- | --- | --- |
| 1 | Top-level token map | `token_id`, `e`, `vision_if`, `h`, `h_64`, `h_final`, `logits_0`, `e_next`, `logits_1`, `K_state`, `V_state`, `C_state`, `S` |
| 2 | Residual decoder layer | `h`, `h_tilde`, `mix_lin`, `mix_full`, `h_mid`, `h_post`, `mlp_out` |
| 3 | Full attention | `h_tilde`, `u_q`, `q_prime`, `g`, `k_raw`, `v_full`, `q_n`, `k_n`, `q_rope`, `k_rope`, `attn`, `y_gate`, `mix_full`, `K_state`, `V_state` |
| 4 | Linear attention | `h_tilde`, `qkv`, `z`, `a`, `b`, `c_tilde`, `c`, `q_lin`, `k_lin`, `v_lin`, `q_hat`, `k_hat`, `alpha`, `beta`, `S`, `o`, `u_gdn`, `mix_lin`, `C_state` |
| 5 | MLP | `h_post`, `g_mlp`, `up`, `swiglu`, `mlp_out` |
| 6 | Persistent state transitions | `K_state`, `V_state`, `C_state`, `S`, `k_rope`, `v_full`, `qkv` |
| 7 | Primary logits and MTP | `h_64`, `h_final`, `logits_0`, `e_next`, `e_next_n`, `h64_n`, `mtp_cat`, `mtp_u`, `h_mtp`, `logits_1`, `E`, `W_lm` |
| 8 | Sharing and reuse | `h`, `h_tilde`, `h_mid`, `h_post`, `h_64`, `k_rope`, `v_full`, `qkv`, `S`, `g`, `z`, `E`, `W_lm` |

Diagram 2 must show both mixers as alternatives (one layer uses exactly one): `h_tilde` → `mix_lin` xor `mix_full` → residual add into `h_mid`. Annotate the 3:1 pattern in prose under diagram 1 ($\ell \bmod 4 = 3$ full).

Diagram 1 may collapse the 64-layer stack as a `subgraph` that still contains `h` (layer I/O) and dashed cylinders for the four state kinds.

Diagram 6 must show: past `K_state`/`V_state` concatenated with current `k_rope`/`v_full`; past `C_state` (3-tap delay) concatenated with current `qkv` into the FIR; past `S` and current recurrence producing next `S`. Ranks: $K,V\in\mathbb{R}^{4\times T\times 256}$ (grows with $T$); $C$ last 3 vectors in $\mathbb{R}^{10240}$; $S\in\mathbb{R}^{48\times 128\times 128}$. RoPE is baked into stored $K$ (TASK-02).

### Logical intermediates catalog (lock all 52 IDs)

Table columns, in this order:

`ID` | `Symbol` | `Rank (no batch)` | `Producer` | `Consumer IDs` | `Fan-out` | `Internal uses` | `Token-crossing` | `Multiplicity` | `Eq`

Ranks are per token unless $T$ appears. Multiplicity is how many instances exist in one forward map (not a byte count). Consumer IDs must be catalog IDs or `output` (primary/MTP logits sink). Implementation copies these rows; do not add/remove IDs.

**Residual stream / logits (9)**

| ID | Symbol | Rank | Producer | Consumer IDs | Fan-out | Internal uses | Token-crossing | Multiplicity | Eq |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `token_id` | $\mathrm{id}_t$ | scalar | input | `e` | 1 | 1 | no | $T$ | (1) |
| `e` | $e_t$ | $H=5120$ | $E_{:,\mathrm{id}_t}$ | `h` (as $h^{(0)}$) | 1 | 1 | no | $T$ | (1) |
| `h` | $h^{(\ell)}$ | $H$ | `e` or previous layer residual | `h_tilde`, `h_mid` | 2 | 1 | no | 64 layer inputs | (4) |
| `h_tilde` | $\tilde h^{(\ell)}$ | $H$ | $\operatorname{RMSNorm}_{1+\gamma_\text{in}}(h)$ | full: `u_q`,`k_raw`,`v_full`; linear: `qkv`,`z`,`a`,`b` | 3 or 4 | 1 | no | 64 | (4) |
| `h_mid` | $h^{(\ell+1/2)}$ | $H$ | $h+\operatorname{Mix}$ | `h_post`, next `h` (with `mlp_out`) | 2 | 1 | no | 64 | (4)–(5) |
| `h_post` | post-attn RMSNorm | $H$ | $\operatorname{RMSNorm}_{1+\gamma_\text{post}}(h_\text{mid})$ | `g_mlp`, `up` | 2 | 1 | no | 64 | (5),(21) |
| `h_64` | $h^{(64)}$ | $H$ | layer 63 output | `h_final`, `h64_n` | 2 | 1 | no | 1 per token | (5),(22),(23) |
| `h_final` | $h^{\text{final}}$ | $H$ | $\operatorname{RMSNorm}_{1+\gamma_\text{final}}(h_{64})$ | `logits_0` | 1 | 1 | no | 1 | (22) |
| `logits_0` | $\ell^{(0)}_t$ | $V=248320$ | $W_\text{lm}\,h^{\text{final}}$ | `output` | 1 | 1 | no | 1 | (22) |

`e_next` is the embedding of token $t+1$, not a second consumer of `e` at time $t$. Sampling from `logits_0` is outside the forward DAG (TASK-02 alignment prose); do not list `e_next` as a consumer of `logits_0`.

**Full attention (14)** — template for each $\ell\in\mathcal{L}_\text{full}$ and MTP self-attn (MTP instances counted under MTP rows for mix I/O; this template is the 16 language layers).

| ID | Symbol | Rank | Producer | Consumer IDs | Fan-out | Internal uses | Token-crossing | Multiplicity | Eq |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| `u_q` | $u=W_q x$ | $12288$ | $W_q\,\tilde h$ | `q_prime`, `g` (disjoint split) | 2 (split, not reuse) | 1 | no | 16 | (6)–(7) |
| `q_prime` | $q'_h$ | $(24,256)$ | split last axis of `u_q` | `q_n` | 1 | 1 | no | 16 | (7) |
| `g` | $g_h$ | $(24,256)$ | split last axis of `u_q` | `y_gate` | 1 | 1 | no | 16 | (7),(10) |
| `k_raw` | $k^{\text{raw}}$ | $(4,256)$ | $W_k\,\tilde h$ | `k_n` | 1 | 1 | no | 16 | (6) |
| `v_full` | $v$ | $(4,256)$ | $W_v\,\tilde h$ | `attn`, `V_state` | 2 | 1 | write | 16 | (6),(9) |
| `q_n` | Q after RMSNorm | $(24,256)$ | $\operatorname{RMSNorm}_{1+\gamma_q}(q')$ | `q_rope` | 1 | 1 | no | 16 | (8) |
| `k_n` | K after RMSNorm | $(4,256)$ | $\operatorname{RMSNorm}_{1+\gamma_k}(k^{\text{raw}})$ | `k_rope` | 1 | 1 | no | 16 | (8) |
| `q_rope` | Q after partial mRoPE | $(24,256)$ | RoPE on first 64 dims | `attn` | 1 | 1 | no | 16 | (9),(12) |
| `k_rope` | K after partial mRoPE | $(4,256)$ | RoPE on first 64 dims | `attn`, `K_state` | 2 | 1 | write | 16 | (9),(12) |
| `attn` | $\operatorname{Attn}(Q,K,V)$ | $(24,256)$ | causal GQA softmax | `y_gate` | 1 | 1 | reads `K_state`,`V_state` | 16 | (9) |
| `y_gate` | $\operatorname{Attn}\odot\sigma(g)$ | $6144$ | sigmoid channel gate | `mix_full` | 1 | 1 | no | 16 | (10) |
| `mix_full` | $\operatorname{Mix}_\text{full}$ | $H$ | $W_o y$ | `h_mid` | 1 | 1 | no | 16 | (10) |
| `K_state` | $K^{(\ell)}$ | $(4,T,256)$ | append `k_rope` (RoPE baked in) | future `attn` | 1 | 1 | yes | 16 language + 1 MTP | state |
| `V_state` | $V^{(\ell)}$ | $(4,T,256)$ | append `v_full` | future `attn` | 1 | 1 | yes | 16 language + 1 MTP | state |

GQA grouping is an edge annotation on `attn`, not extra `k_repeated` / `v_repeated` nodes (TASK-02 algebraic equivalent).

**Linear attention (18)** — template for each $\ell\in\mathcal{L}_\text{lin}$.

| ID | Symbol | Rank | Producer | Consumer IDs | Fan-out | Internal uses | Token-crossing | Multiplicity | Eq |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| `qkv` | $\mathrm{qkv}$ | $10240$ | $W_\text{qkv}\,\tilde h$ | `c_tilde`, `C_state` | 2 | 1 | write | 48 | (13)–(14) |
| `z` | $z$ | $(48,128)$ | $W_z\,\tilde h$ | `u_gdn` | 1 | 1 | no | 48 | (13),(20) |
| `a` | $a$ | $48$ | $W_a\,\tilde h$ | `alpha` | 1 | 1 | no | 48 | (13),(15) |
| `b` | $b$ | $48$ | $W_b\,\tilde h$ | `beta` | 1 | 1 | no | 48 | (13),(15) |
| `c_tilde` | $\tilde c_t$ | $10240$ | depthwise causal conv (reads `C_state`) | `c` | 1 | 1 | reads `C_state` | 48 | (14) |
| `c` | $c_t=\operatorname{SiLU}(\tilde c)$ | $10240$ | SiLU | `q_lin`, `k_lin`, `v_lin` (disjoint split) | 3 (split, not reuse) | 1 | no | 48 | (14) |
| `q_lin` | $q$ | $(48,128)$ | split + repeat_interleave $r^\ell=3$ | `q_hat` | 1 | 1 | no | 48 | (14),(16) |
| `k_lin` | $k$ | $(48,128)$ | split + repeat_interleave | `k_hat` | 1 | 1 | no | 48 | (14),(16) |
| `v_lin` | $v$ | $(48,128)$ | split (no L2) | `S` | 1 | 1 | no | 48 | (14),(17) |
| `q_hat` | $\tilde q$ | $(48,128)$ | L2-normalize `q_lin` | `o` | 1 | 1 | no | 48 | (16),(18) |
| `k_hat` | $\tilde k$ | $(48,128)$ | L2-normalize `k_lin` | `S` | 1 | 2 | no | 48 | (16),(17) |
| `alpha` | $\alpha_t$ | $48$ | Eq. (15) from `a`, `A_log`, `dt_bias` | `S` | 1 | 1 | no | 48 | (15) |
| `beta` | $\beta_t$ | $48$ | $\sigma(b)$ | `S` | 1 | 1 | no | 48 | (15) |
| `S` | $S_t$ | $(48,128,128)$ | Eq. (17) from `S`$_{t-1}$, `k_hat`, `v_lin`, `alpha`, `beta` | `o`, next `S` | 2 | 1 | yes | 48 | (17)–(19) |
| `o` | $o_t$ | $(48,128)$ | Eq. (18) | `u_gdn` | 1 | 1 | no | 48 | (18) |
| `u_gdn` | GatedRMSNorm$(o,z)$ | $(48,128)$ | Eq. (3)/(20) | `mix_lin` | 1 | 1 | no | 48 | (20) |
| `mix_lin` | $\operatorname{Mix}_\text{lin}$ | $H$ | $W_\text{out}\operatorname{vec}(u)$ | `h_mid` | 1 | 1 | no | 48 | (20) |
| `C_state` | $C^{(\ell)}$ | $3\times 10240$ | delay of `qkv` | next `c_tilde` | 1 | 1 | yes | 48 | (14) |

**MLP (4)** — 64 language layers; MTP MLP is the same template inside the MTP block (do not add extra IDs).

| ID | Symbol | Rank | Producer | Consumer IDs | Fan-out | Internal uses | Token-crossing | Multiplicity | Eq |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| `g_mlp` | $W_\text{gate}x$ | $17408$ | $W_\text{gate}\,h_\text{post}$ | `swiglu` | 1 | 1 | no | 64 (+1 MTP) | (21) |
| `up` | $W_\text{up}x$ | $17408$ | $W_\text{up}\,h_\text{post}$ | `swiglu` | 1 | 1 | no | 64 (+1 MTP) | (21) |
| `swiglu` | $\operatorname{SiLU}(g)\odot\text{up}$ | $17408$ | elementwise | `mlp_out` | 1 | 1 | no | 64 (+1 MTP) | (21) |
| `mlp_out` | $\operatorname{MLP}$ | $H$ | $W_\text{down}$ | next `h` (with `h_mid`) | 1 | 1 | no | 64 (+1 MTP) | (5),(21) |

**MTP (7)**

| ID | Symbol | Rank | Producer | Consumer IDs | Fan-out | Internal uses | Token-crossing | Multiplicity | Eq |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| `e_next` | $e_{t+1}$ | $H$ | shared $E$ | `e_next_n` | 1 | 1 | no | 1 | (1),(23) |
| `e_next_n` | $\operatorname{RMSNorm}(e_{t+1})$ | $H$ | `mtp.pre_fc_norm_embedding` | `mtp_cat` | 1 | 1 | no | 1 | (23) |
| `h64_n` | $\operatorname{RMSNorm}(h^{(64)})$ | $H$ | `mtp.pre_fc_norm_hidden` | `mtp_cat` | 1 | 1 | no | 1 | (23) |
| `mtp_cat` | concat | $10240$ | concat embed-then-hidden | `mtp_u` | 1 | 1 | no | 1 | (23) |
| `mtp_u` | $u_t$ | $H$ | $W_\text{fc}$ | MTP residual `h` (block input) | 1 | 1 | no | 1 | (23) |
| `h_mtp` | $h^{\text{mtp}}$ | $H$ | MTP full layer (4)–(10),(21) | `logits_1` via `mtp.norm` | 1 | 1 | MTP KV yes | 1 | (24) |
| `logits_1` | $\ell^{(1)}_t$ | $V$ | shared $W_\text{lm}$ after `mtp.norm` | `output` | 1 | 1 | no | 1 | (24) |

MTP self-attention reuses the full-attention template (including doubled `q_proj` / `g` / sigmoid gate) with its own `K_state`/`V_state` instance; do not duplicate those 14 IDs.

**Shared parameters** (not catalog intermediates; still drawn and listed in JSON `shared_weight_ids`):

| ID | Rank | Consumers | Fan-out |
| --- | --- | --- | ---: |
| `E` | $(H,V)$ stored $(V,H)$ | `e`, `e_next` | 2 |
| `W_lm` | $(V,H)$ | `logits_0`, `logits_1` | 2 |

### Explicit catalog exclusions

Do not add nodes for: RMS scalars; softmax scores $P_{h,s}$; $\omega_j$ / cos / sin tables (label the RoPE edge); `A_log` / `dt_bias` as activations (they are weights into `alpha`); paper $S^\top$ as a second state; GQA-repeated KV; vision encoder activations; batch dimension.

### Sharing and reuse (closes ledger open question)

This section is **DERIVED** from the catalog. It is not a HYPOTHESIS about runtime, fusion, or bytes.

**High-fan-out IDs** (reuse of the same value, `high_fanout_ids`; exclude disjoint splits `u_q`, `c`):

`h`, `h_tilde`, `h_mid`, `h_post`, `h_64`, `k_rope`, `v_full`, `qkv`, `S`

**Live-across IDs:** `g`, `z`

**Intra-equation reuse:** `k_hat` (Eq. 17)

**Shared weights:** `E`, `W_lm`

Ranked consequential sharing (this order is required in the prose list; criterion = residual survival, projection fan-out, then token-crossing, then live-across, then shared weights):

1. `h` / `h_mid` — residual stream: each mixer/MLP both reads the pre-op vector and adds to it, so the pre-op value has a second consumer besides the mixer/MLP body.
2. `h_tilde` — fans out to 3 (full: q/k/v) or 4 (linear: qkv/z/a/b) independent projections.
3. `h_post` — fans out to `g_mlp` and `up`.
4. `k_rope` and `v_full` — current `attn` plus persistent KV write.
5. `qkv` — FIR conv plus conv-delay `C_state`.
6. `S` — produces `o` and is the next-token GDN state.
7. `h_64` — primary logits path and MTP mix.
8. `E` and `W_lm` — embed/MTP token lookup and both logit heads.
9. `g` — produced at the `q_proj` split, consumed only in Eq. (10) after attention (live-across).
10. `z` — produced at `in_proj_z`, consumed only in GatedRMSNorm after the recurrence (live-across).
11. `k_hat` — two reads inside Eq. (17) (intra-equation; not a second catalog consumer).

Do not rank by parameter count, FLOPs, or predicted speedup. Do not say which values “should” be stored.

### Tooling

Create `scripts/check_dataflow.py` (Python 3.11+, stdlib only: `argparse`, `json`, `re`, `sys`, `pathlib`, Google docstrings). No torch, safetensors, mermaid parser package, uv, Ruff, or pytest. Do not import `scripts/check_model_semantics.py`; duplicate the small `text_config` arithmetic needed for layer counts and `full_attention_indices`.

CLI (cwd = repository root):

```text
python3 scripts/check_dataflow.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --dataflow docs/architecture/dataflow.md \
  [--json]
```

Behavior:

- Read `text_config` from `--config`. Build the graph-summary object (schema below). Live fields: `hidden_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `full_attention_indices`. Constant fields: catalog IDs, diagram count, canonical sentence, id lists.
- `--json`: print that object to stdout (pretty-printed, script key order); run internal asserts (`n_catalog_nodes == 52`, `n_diagrams == 8`, index list length 16, `mrope` not required here); exit 0.
- Default / `--dataflow PATH`: also require PATH to contain (1) every required `##` heading listed above **in order**, (2) the first fenced `json` block equal to the live object, (3) exactly eight ` ```mermaid ` fences, each containing `flowchart`, (4) the canonical sentence verbatim, (5) every `catalog_ids` entry present as a substring, (6) each diagram’s required IDs present **inside that mermaid fence** (split the eight fences in document order), (7) none of `TBD`, `TODO`, `???`, (8) no `UNKNOWN` except inside the Deferred vision section. Exit 1 with a readable list on mismatch.
- Missing config: exit 2 (blocked, not a content fail).

Do not read safetensor payloads. Do not require `model-semantics.md` JSON equality (that remains TASK-02’s checker).

### Instantiated graph-summary JSON schema

Top-level keys (all required; script key order locked as this list):

`authority` (exactly `.cache/authorities/qwen3.8-27b-transformers`), `hidden_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `full_attention_indices` (int array from config, must equal `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`), `n_diagrams` (8), `n_catalog_nodes` (52), `catalog_ids` (array, **this exact order**):

`token_id`, `e`, `h`, `h_tilde`, `h_mid`, `h_post`, `h_64`, `h_final`, `logits_0`, `u_q`, `q_prime`, `g`, `k_raw`, `v_full`, `q_n`, `k_n`, `q_rope`, `k_rope`, `attn`, `y_gate`, `mix_full`, `K_state`, `V_state`, `qkv`, `z`, `a`, `b`, `c_tilde`, `c`, `q_lin`, `k_lin`, `v_lin`, `q_hat`, `k_hat`, `alpha`, `beta`, `S`, `o`, `u_gdn`, `mix_lin`, `C_state`, `g_mlp`, `up`, `swiglu`, `mlp_out`, `e_next`, `e_next_n`, `h64_n`, `mtp_cat`, `mtp_u`, `h_mtp`, `logits_1`

then `high_fanout_ids` `["h","h_tilde","h_mid","h_post","h_64","k_rope","v_full","qkv","S"]`, `live_across_ids` `["g","z"]`, `intra_equation_reuse_ids` `["k_hat"]`, `shared_weight_ids` `["E","W_lm"]`, `state_ids` `["K_state","V_state","C_state","S"]`, `regions` `["embed","decoder_stack","residual_layer","full_attn","linear_attn","mlp","persistent_state","primary_logits","mtp","vision_interface"]`, `canonical_sentence` (the verbatim two-sentence string above).

`n_mtp_blocks` is 1 from `mtp_num_hidden_layers`. `n_linear_layers` / `n_full_layers` from `layer_types` or equivalently 48 / 16 from interval 4.

### Stage split

- **Implementation** writes `scripts/check_dataflow.py` **and** `docs/architecture/dataflow.md` (diagrams and catalog are the increment). Runs `--json` and `--dataflow` after the document exists. Records command outcomes in this dossier. Does not commit.
- **Documentation** performs a mechanical pass only: draft-status `unverified` banner, Authority table links to this dossier / semantics / inventory / plan evidence policy, heading/JSON fence consistency. Must not change locked IDs, fan-out numbers, sharing rank order, or Mermaid node IDs. Does not invent regions. Does not edit TASK-02 artifacts.
- **Verification** independently re-runs focused commands, reads the document against this dossier (every region diagram, all 52 catalog rows, canonical sentence, sharing list), and confirms no Quartz/llama.cpp/GGUF evidence and no `plan.md` edit.

- Invariants:
  - Eight Mermaid `flowchart` fences; catalog has exactly the 52 IDs above; every catalog ID appears in at least one fence as specified.
  - Canonical sentence present verbatim.
  - Prefill and decode are one DAG; state kinds are only $K,V,C,S$.
  - Logical values do not imply allocation; sharing rank is DERIVED, not a fusion plan.
  - No new operators beyond TASK-02; no kernel/layout/schedule text except as things the DAG does **not** decide.
  - Vision encoder remains unexpanded (`vision_if` interface only).
- Rejected alternatives:
  - Archify / standalone HTML / PNG export: extra artifact not in the ledger; Node toolchain.
  - Graphviz DOT or ASCII-only diagrams: not GitHub-native; worse reviewability than Mermaid in `dataflow.md`.
  - One unrolled 64-layer chart: unreadable and not checkable.
  - `sequenceDiagram` / `stateDiagram-v2`: wrong abstraction (value DAG, not calls or UI states).
  - Treating algebraic equivalents as extra nodes: rejected; same map.
  - Cataloguing RMS scalars, softmax scores, or GQA-repeated KV: not significant sharing; TASK-02 already names the equivalents.
  - Byte sizes, lifetime classes, or “must materialize” columns: TASK-04/12.
  - Ranking sharing by parameter count or predicted kernels: not this task.
  - Importing `check_model_semantics.py` as a library: keep evidence scripts standalone.
  - uv / Ruff / pytest for this increment: rejected; stdlib checker matches TASK-01/02.
  - Editing frozen TASK-02 semantics/inventory: one-way citations from `dataflow.md` suffice.
  - Inspecting Quartz or llama.cpp to “confirm” dataflow: forbidden by plan.md.
- Discovered ledger work: `none`
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/dataflow.md` exists and follows the heading list above.
  - All required model regions and token-to-token $K,V,C,S$ are diagrammed (eight Mermaid flowcharts with locked node IDs).
  - The 52-row catalog tabulates significant logical intermediates and their consumers, with fan-out and token-crossing.
  - Canonical sentence states that logical values do not imply physical allocation (verbatim).
  - Sharing-and-reuse section lists the ranked DERIVED items 1–11 and the locked id arrays; ledger open question is closed (not left UNKNOWN).
  - JSON fence matches a live `--json` object from config arithmetic plus locked constants.
  - No kernel/layout/fusion/allocation **decisions**; no Quartz/llama.cpp/GGUF evidence.
  - Coupled IDs: none.
- Tests/fixtures to add or change: `scripts/check_dataflow.py` only (no pytest fixtures).
- Focused commands (repository root; config must exist):

```sh
python3 -m py_compile scripts/check_dataflow.py
python3 scripts/check_dataflow.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
python3 scripts/check_dataflow.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --dataflow docs/architecture/dataflow.md
```

- Candidate quality: not required — no model execution or NLL.
- Repository-wide commands:

```sh
test -f docs/architecture/dataflow.md
python3 -m py_compile scripts/check_dataflow.py
python3 scripts/check_dataflow.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --dataflow docs/architecture/dataflow.md
```

Do not run Ruff, pytest, CMake, or CUDA; this increment does not introduce those gates.

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Config presence is a local-file requirement, not a GPU gate.
- Documentation/evidence updates:
  - `docs/architecture/dataflow.md` (create)
  - `scripts/check_dataflow.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged; `model-semantics.md` unchanged; `model-inventory.md` unchanged)
- Definition of done: dataflow document published with locked region diagrams, 52-row catalog, canonical logical-≠-physical sentence, and sharing ranking; JSON fence verifies against sitting `config.json` plus locked graph constants; ledger TASK-03 completion checkboxes can be marked at delivery.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high`
- UTC/time/tokens/cost: `2026-09-20T09:31:00Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-03.md`. Coupled IDs none. Document structure (14 headings), Mermaid flowchart format (8 fences), 52-ID intermediate catalog, sharing ranking, stdlib checker, JSON schema, and acceptance commands are closed. `dataflow.md` not written in this stage.
- Performance evidence applied: N/A (logical DAG documentation; no timing)

### Implementation

- Agent/model: `cursor-grok-4.6-high`
- Changes:
  - created `scripts/check_dataflow.py` (stdlib graph-summary checker; no import of `check_model_semantics.py`)
  - created `docs/architecture/dataflow.md` (14 locked headings, 8 Mermaid `flowchart` fences, 52-row catalog, canonical logical-≠-physical sentence, ranked sharing 1–11, JSON fence from live `--json`)
  - did not edit `docs/architecture/plan.md`, `docs/architecture/model-semantics.md`, or `docs/architecture/model-inventory.md`
- Commands (repository root; all exit 0):
  - `python3 -m py_compile scripts/check_dataflow.py`
  - `python3 scripts/check_dataflow.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — printed graph-summary with `n_catalog_nodes` 52, `n_diagrams` 8, `full_attention_indices` length 16 matching `[3,7,…,63]`, `n_decoder_layers` 64 / `n_linear_layers` 48 / `n_full_layers` 16 / `n_mtp_blocks` 1 / `hidden_size` 5120
  - `python3 scripts/check_dataflow.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --dataflow docs/architecture/dataflow.md` — headings, JSON fence equality, eight flowcharts, canonical sentence, catalog IDs, diagram node IDs, no `TBD`/`TODO`/`???`, `UNKNOWN` only in Deferred vision
- UTC/time/tokens/cost: `2026-09-20T09:41:47Z`; `telemetry_unavailable`
- Commit: not created (implementation stage)

### Documentation

- Agent/model: `cursor-composer` (documentation subagent)
- Changes and evidence:
  - `docs/architecture/dataflow.md` — added draft-status banner (`unverified`); Authority table already cross-links dossier, [`model-semantics.md`](model-semantics.md), [`model-inventory.md`](model-inventory.md), sitting `config.json`, [`check_dataflow.py`](../../scripts/check_dataflow.py), and plan evidence policy; 14 required `##` headings and JSON fence unchanged aside from banner.
- Commands:
  - `python3 scripts/check_dataflow.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — **pass** (exit 0; JSON fence source unchanged).
  - `python3 scripts/check_dataflow.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --dataflow docs/architecture/dataflow.md` — **pass** (exit 0; headings, JSON fence, eight flowcharts, canonical sentence, catalog IDs, diagram node IDs).
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T09:43:00Z`; `telemetry_unavailable`

### Verification

- Attempt: 1
- Agent/model: `cursor-composer` (integration verifier subagent)
- Diff review:
  - `docs/architecture/dataflow.md` — 14 required `##` headings in dossier order; canonical logical-≠-physical sentence verbatim; eight Mermaid `flowchart` fences with locked node IDs per diagram; 52-row catalog with fan-out, token-crossing, and consumer columns; sharing section closes ledger open question with ranked DERIVED list 1–11 and locked `high_fanout_ids` / `live_across_ids` / `intra_equation_reuse_ids` / `shared_weight_ids`; deferred vision interface only (`vision_if`).
  - `scripts/check_dataflow.py` — stdlib-only (`argparse`, `json`, `re`, `sys`, `pathlib`); no import of `check_model_semantics.py`; `--json` arithmetic and `--dataflow` heading/JSON-fence/Mermaid/forbidden-token checks match dossier.
  - Prefill and decode share one DAG; state kinds are only $K,V,C,S$; algebraic equivalents are not extra nodes.
  - No kernel/layout/fusion/allocation decisions; no Quartz/llama.cpp/GGUF inspection artifacts.
  - `docs/architecture/plan.md` — unchanged.
  - `docs/architecture/model-semantics.md` and `docs/architecture/model-inventory.md` — unchanged.
- Independent raw-record checks:
  - First fenced `json` block in `dataflow.md` matches live `--json` output (`hidden_size` 5120, `n_catalog_nodes` 52, `n_diagrams` 8, `full_attention_indices` `[3,7,…,63]`, `n_decoder_layers` 64 / `n_linear_layers` 48 / `n_full_layers` 16 / `n_mtp_blocks` 1).
  - `UNKNOWN` appears only in Deferred vision section; no `TBD`, `TODO`, or `???` in `dataflow.md`.
- Commands:
  - `python3 -m py_compile scripts/check_dataflow.py` — **pass** (exit 0)
  - `python3 scripts/check_dataflow.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — **pass** (exit 0)
  - `python3 scripts/check_dataflow.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --dataflow docs/architecture/dataflow.md` — **pass** (exit 0)
  - `test -f docs/architecture/dataflow.md` — **pass** (exit 0)
  - `python3 -m py_compile scripts/check_dataflow.py` (repository-wide duplicate) — **pass** (exit 0)
  - `python3 scripts/check_dataflow.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --dataflow docs/architecture/dataflow.md` (repository-wide duplicate) — **pass** (exit 0)
- Formatting changed files: none
- Verdict: **pass** — deliverables satisfy dossier acceptance and ledger completion criteria; eight region diagrams, 52-row catalog, canonical sentence, and sharing ranking close the open question; JSON fence verifies against sitting `config.json`.
- UTC/time/tokens/cost: `2026-09-20T09:44:00Z`; `telemetry_unavailable`

### Retries and escalation

none

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification pass — `dataflow.md` and JSON fence reconcile to sitting `config.json`; checker passes focused commands; sharing/reuse open question closed
- Candidate measured delta: N/A (no throughput work)
- Shipping delta: N/A (diagnostics/documentation)
- Quality result: not required
- Evidence completeness: N/A (no performance-evidence checks)
- Throughput delta: N/A — TASK-03 does not execute or time the model
- Commit: Publish Qwen3.8 logical dataflow graph
- Push: `origin/clean-sheet` (success)
- First-pass acceptance: **verified**
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: local `.cache/` config must remain present for focused commands
