# OPT-104 rejection — retain raw GGUF Q6_K

Production pins remain `kSelectedQ6DeviceLayout[] = "raw_gguf"` and
`kSelectedQ6DecodePath[] = "integer_q8_1"`.
The aligned SoA candidate is not a replacement layout.

Q6 control 1.559 ms vs candidate
1.566 ms; mean_diff
-0.0072 ms; CI
-0.0156 .. 0.0012.
Keep required ≥0.10 ms/token with a positive paired 95% interval.
The candidate is slower on combined Q6 attention-output plus logits, so
raw GGUF Q6_K is retained. D128/D2048/P4096 gates were not run.
Kernel parity 12/12 bitwise and OPT-089 same-math quality reuse both passed.
No public GGUF rewrite. No extra resident copy.
status=measured_reject.
tok/s delta vs OPT-098 P4096 3046.23: 0
