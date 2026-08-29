# 15. Tokenizer authority and fixtures

[Previous](14-artifact-validation.md) · [Index](README.md)

TOK-001 uses tokenizer assets from the same immutable official Qwen revision as
the source model. Their byte sizes and SHA-256 values live in
[`pins/artifacts.lock.json`](../pins/artifacts.lock.json); runtime copies under
`models/tokenizer/` are ignored by git.

The observed tokenizer is **External primary** behavior until the native encoder
matches it. It uses an NFC normalizer, a Unicode-aware split followed by GPT-2
byte-level transformation, and BPE. The base tokenizer JSON has 248,044 entries
and 247,587 merges; the GGUF carries 248,320 token entries, including model
special/reserved entries, and the same merge count.

[`tools/generate_tokenizer_fixtures.py`](../tools/generate_tokenizer_fixtures.py)
fails unless every downloaded authority asset matches its pin. It then uses
`tokenizers==0.22.1` to write
[`fixtures/tokenizer_authority.json`](../fixtures/tokenizer_authority.json).
Fixtures retain UTF-8 input bytes, token IDs, token strings, and decoded UTF-8
bytes for empty, ASCII, contraction, whitespace, newline, punctuation, NFC/NFD,
multilingual, emoji, code, and special-token cases.

This is reproducible oracle evidence, not completion of TOK-001. The production
C++ tokenizer must load the GGUF vocabulary/merges and match every ID exactly.
In particular it must not replace the Unicode split with an ASCII approximation
or skip NFC normalization. See the TOK-001 entry and chronological evidence in
[`implementation_ledger.md`](../implementation_ledger.md).
