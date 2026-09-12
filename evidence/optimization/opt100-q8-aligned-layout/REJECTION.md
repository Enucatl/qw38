# OPT-100 rejection — retain raw GGUF Q8_0

Production pin remains `kSelectedQ8DeviceLayout[] = "raw_gguf"`.
The aligned SoA candidate is not a replacement layout.

Complete rotating Q8 mixer (48 GDN + 16 attention, grouped_r1_w4, 64
input-projection launches) on this sitting:

- feedback 1+3: control 6.916 ms vs candidate 6.906 ms; mean_diff 0.010 ms;
  CI95 [−0.160, 0.179]; `positive=false`
- acceptance 3+10: control **6.902 ms** vs candidate **7.085 ms**;
  mean_diff **−0.183 ms**; CI95 **[−0.275, −0.091]**; `positive=false`

Keep required ≥0.10 ms/token with a positive paired 95% interval. The
candidate is slower on the complete mixer, so raw GGUF Q8_0 is retained.
Kernel parity 12/12 bitwise and OPT-089 same-math quality reuse both passed.
No Q8 D2R revival. No public GGUF rewrite. No extra resident copy.

status=measured_reject.
