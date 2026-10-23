# Immediate-index acquisition: historical receipt

> The acquisition-only status below is retained from9b81e9d. The later native semantic profile passes all64 G17 cases; see [M5-INDEX-SEMANTICS.md](M5-INDEX-SEMANTICS.md). Target105 remains refused.

Continuation of M5-OPERAND-RESEARCH.md, after source/evidence commit f5c6118e60ade1e09469ba78bbb003e02e0fc81f.

The distinct immediate0..63 sweep completed with acquisition exit0: **128 paired captures**, exactly one G16 and one G17 object for every immediate. `m5-index-corpus.tar.gz` contains original source, own-shader compiled outputs, symbol tables, acquisition log, exact input/output hash inventory and terminal sentinel. The producer and native extractor executable fingerprints were verified unchanged at completion. No Apple executable is redistributed.

The source keeps three operands live, reads `a[gid+K]`, computes `gid+K`, and separately stores both the loaded value and computed index. This distinguishes address-index arithmetic from memory selection and avoids relying on a single multiply result.

All64 G17 entries have the recognizable four-byte get_sr prefix at the symbol-established main64: one `1c a0 10 06`, five `ac a0 10 06`, and58 `bc a0 10 06`. The unknown eight-byte entry from pressure18/25 did not occur in this acquisition. These counts were checked against exact filename membership, not inferred from the loop's intended range.

**NOT YET QUALIFIED:** No native semantic replay of this new index family has been recorded. Acquisition success is not an emitter, decoder, dependency or hardware verdict. Target105 remains refused; full M1–M5 acceptance and independent review remain open. L742 is the separate read-only evidence audit, not a second implementation owner.
