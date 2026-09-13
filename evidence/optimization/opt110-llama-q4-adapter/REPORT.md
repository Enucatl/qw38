# OPT-110 — Execute pinned-llama Q4_K MMVQ through a Quartz adapter

Status: **kept**. Authority llama.cpp `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`.
Control is production `integer_q8_late` / Q8Block / `raw_gguf`.
Candidate `llama_q4k_mmvq` is the source-faithful GENERIC nwarps=4
Q4_K × Q8_1 MMVQ adapter plus native `block_q8_1` staging.
OPT-093 factored association and OPT-102 aligned metadata are not used.

## Primitive screen (must win before engine integration)

primitive_win=True.
Go/no-go is staging + dot on identical BF16 and raw Q4_K at
5120×17408 (down) and fused 17408×5120 (gate/up).

- down: control=0.0389056008 ms candidate=0.0192831999 ms saving=0.01962240096 stage=0.00326719999/0.00372160001 dot=0.0356384009/0.0155616 ci95=[0.017607321284164486, 0.021637480635835517] positive=True
- gate_up: control=0.0640639998 ms candidate=0.0333311999 ms saving=0.030732799890000002 stage=0.00704960003/0.00593599996 dot=0.0570144005/0.0273952 ci95=[0.025754307794180878, 0.03571129198581913] positive=True

combined_control_ms=0.10296960059999999
combined_candidate_ms=0.052614399799999996
combined_saving_ms=0.05035520079999999
reason=matched_primitive_staging_plus_dot_won

## Production pin

shipping_q4_decode=`llama_q4k_mmvq`; production_kept=True; claims_throughput=True.

tok/s delta vs OPT-114 D2048 baseline (48.3830): **+8.2829**.
D128 delta (53.4602 baseline): **+4.8093**; P4096 delta (2981.0894 baseline): **+35.4539**.

## Complete rotating FFN

skipped=False reason=None
control_mean_ms=9.8138144
candidate_mean_ms=9.0967041
saving_ms=0.7171102999999995 ci95=[0.6779576387425308, 0.7562629612574683] positive=True ffn_keep=True

## d128
skipped=False reason=None control_ms=581.6889526 candidate_ms=549.1734985999999 tok_s=55.01393056553077/58.26948240532975 improved=True ratio=None

## d2048
skipped=False reason=None control_ms=600.2489257999999 candidate_ms=564.713257 tok_s=53.31197692410444/56.66596483211246 improved=True ratio=None

## prefill-guard
skipped=False reason=None control_ms=1357.63989 candidate_ms=1357.84558 tok_s=3017.0003328349467/3016.54330973335 improved=None ratio=0.9998485173844288

## Quality

model_quality_pass=True skipped=False reason=opt110_quality_pass
candidate held_out NLL=1.7875990840085783 control=1.7872306470098762 ppl_ratio=1.0003685048799495
candidate wikitext NLL=1.5249350773895778 control=1.5264653580189336 ppl_ratio=0.9984708896530169
recurrence_incremental_nll=-0.006112820460778767 vs control (max |delta| 0.02)

## Verdict

verdict=keep keep=True reasons=[]
Keep bar requires ≥0.50 ms/token complete FFN, positive 95% CI, D128/D2048 throughput, P4096 guard, and candidate NLL.
Performance and candidate quality gates passed; shipping pin may flip to `llama_q4k_mmvq`.

## Adapter notes

Live attrs: MMVQ 63 regs / occ 8 / 0 spill; fused SWIGLU 48 regs / occ 10 / 0 spill; quant 16 regs / occ 6. SASS dp4a/idp=520.
Local mods vs pin: BF16→Q8_1 (llama quantize is float); no ggml PDL/fastdiv/ids; fused SWIGLU stores BF16; compile-time production shapes; diagnostic engine hooks only.
Primitive Q4_K used 144B layout-identical synthetic blocks, not a live GGUF tensor copy. Engine phases used production `raw_gguf`.

