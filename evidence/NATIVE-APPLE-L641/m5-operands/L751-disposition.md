# L751 parent disposition: bounded review accepted, M5 remains HOLD

Reviewed source: `da94831331abd3c66d05541fc53017ce4ed30371`.
Existing full replay evidence: `ead7fc45fad3aa3f0786fa60f5d0698eb2adc88b`.
Original independent report SHA256: `e93c92a704f6fa75e2b9ec1cf35750a5888460cba6522c12799c8702ba02f612`.

## Acceptance and verification

The original report is preserved unchanged as `L751-review-original.md`. It found no correctness defect in the frozen pressure/index research contracts, but did not qualify a production backend, arbitrary G17g programs, dependency scheduling, caches or hardware execution.

Parent inspected the review's raw terminal-result excerpt, including the initial wrong-driver-path compilation failure and later wrong-corpus-path failure. Neither is counted as a passing runtime. The successful focused replay reports K=0,1,31,32,63; index mutation/restoration K=1,32,63; pressure1,18,25; and entry mutation/restoration18.

Parent read back all four reviewer executable hashes. They match the published frozen-source replay identities, are nonempty static AArch64 ELF files, and were then exercised directly: index32, index-mutants32, pressure18 and entry-mutants18 all returned0. See `m5-L751-parent-verification.txt`. No full-suite duplication or GPU launch was performed.

The original K1/K2 byte correlation at offset0x5c is verified. Both captures contain232 bytes and the12-byte add differs in its immediate byte. This is NOT a whole-text one-byte minimal pair: other register-allocation/address bytes also change. The parent receipt lists every differing offset. The cleaner whole-text axis minimal-pair result remains separately qualified.

## F4 resolved at the reference-document level

L751 correctly identified apparently conflicting summaries: EXP-M5-07 calls load byte+5/+6/+7 inert, whereas EXP-M5-11 calls byte+5 an index register. The detailed EXP-M5-11 patch at the SAME upstream pin explicitly scopes the observations to different address modes:

- `amode=0x22`: gid-direct form; its byte+5 is described as inert.
- `amode=0x02`: computed-index form; byte+5 is described as the index register. The cited `ao_idx` splice changes it from0x20 to0x00.

Source: [isadb.py.EXP-M5-11.patch, lines94–117](https://github.com/ADevWithAnIdea/agx-re/blob/c677ff970ca9d5ab98770100e790f8a62c60a2a0/experiments/EXP-M5-11-isa-integration/isadb.py.EXP-M5-11.patch#L94-L117), SHA256 `e3ee2a8d2d26584fa200f2fe7bd8d1d62fa07b5716270d6b13877d76cdab13d9`.

Thus the later patch already corrects/scopes the older blanket negative. No upstream rewrite or speculative interpreter change is needed to reconcile that prose. This does NOT independently reproduce the claimed hardware splice or establish that the reference's instruction boundaries and physical register naming generalize to our captures. The reference's implicit/positional data-register model and our effective register model still need discriminating evidence. Third-party hardware claims remain attributed claims, not our own hardware results.

The original report's evidence table must be read with this disposition: reference field descriptions are not new production contracts. The G17P-labelled scoreboard note still does not establish G17g scheduling.

## Next concrete dependency

Acquire a separate compile-only direct/computed/gather address-family corpus, retaining the address value, loaded value and independent live values. Vary address source independently of register pressure. Compare actual G16/G17 captures and record the whole changed-byte context rather than transplanting the reference's field map. Use Omega for the first-party source producer and behavioral golden/refusal assertions; reuse the existing Apple compiler and Omega extractors. Do not launch these kernels or modify the reviewed checker while gathering that evidence.

L751 can close as a reviewed bounded-source/evidence deliverable after publication. L641, the production M5 emitter/interpreter and positive M1–M5 coverage remain OPEN. No merge, deployment or hardware approval is conferred.
