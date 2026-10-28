# Native kernel cache policy

This directory implements issue #158. It does not open a GPU, allocate device memory, create a session, or replace command lists. The existing FP8 GEMV, GEMM and tensor-core caches remain the owners of code allocations. Their existing session lock must cover every policy operation, including find/revision/pin sequences. A native unit cannot supply thread synchronization at the pinned OM-1 compiler.

## Immutable artifact identity

`digest.om` implements bounded SHA-256 using the existing Omega SHA-256 algorithm and filed decimal constants. `kc_sha` accepts `uchar[4096]`, an exact used length, and `uchar[32]` output. `kc_codesha` accepts `uchar[16384]`. Full blocks are read directly; only the final partial block is copied. Inputs exceeding capacity refuse without writing the output.

`artifact.om` consumes, rather than redefines, the QuantSchema and Composer serializers. Concatenate `quant.om`, `realize.om`, `digest.om`, then `artifact.om`. `kc_key` validates the canonical record with `compose_check`, then hashes exactly its used bytes. `kc_request` puts the digest into request words 3 through 6 as little-endian u64 limbs. It does not change caller-supplied generation, profile, code size or owner ticket.

The realization key is not the full artifact-store manifest ID. Existing `omega_native_artifact_export` and `omega_native_artifact_admit` remain authoritative for immutable portable artifact files, framed templates, relocations, target and driver admission. The cache never treats a filename, a hash match, or metadata validity as hardware qualification. `kc_validate` invokes the Composer-owned `compose_launchread` to check canonical launch declaration consistency, then independently binds code and launch bytes to their record digests and checks the stored realization key. The admitting runtime must still enforce existing full artifact admission and actual template-content identity before upload. Canonical bytes and code remain immutable for their entire registered lifetime; device handles and virtual addresses never enter the realization key.

## Retained admitted-template binding

`templates.om` adds private `native_templates(launch: uchar[4096], cubin: uchar[32768], observed: uchar[2048], sizes: ulong[8]) -> long`. Concatenate the same QuantSchema/Composer dependencies and `digest.om` before it. It is a **consistency checker after successful existing artifact admission**, never a second artifact-admission authority.

The runtime must supply launch bytes and expected manifest ID from the selected immutable canonical bundle, not derive expectations from the candidate snapshot. All observed bytes, actual successful admission ID and function name must come from that single retained admission snapshot. Do not reopen files between admission and this call. Keep the snapshot alive through checking/upload; the integration owner retains ownership of this custody and the existing session lock.

The private handoff has this exact layout; it is not a new realization schema:

| `observed` bytes | Contents |
| --- | --- |
| 0 through 767 | Exact retained QMD0 followed by QMD1 |
| 768 through 895 | Retained cbank argument bytes, zero unused tail |
| 896 through 959 | Retained admitted function name, no embedded NUL, zero unused tail |
| 960 through 991 | Expected manifest ID as raw SHA-256 bytes from selected immutable bundle |
| 992 through 1023 | Zero |
| 1024 through 1087 | Actual successful admission ID, exactly 64 lowercase hex bytes |
| 1088 through 2047 | Zero |

`sizes[0..2]` are exact cubin bytes, cbank bytes and function-name bytes; words 3 through 7 are zero. Cubin capacity is 32768, cbank capacity 128, and name length is 1 through 63. Cubin unused tail and canonical launch reserved/tail bytes must be zero.

The checker bounds every ELF64LE range before reading it, requires a unique exact `.nv.info.<admitted-name>` section, and checks its links to the matching text section and symbol table. It parses the ordered existing ABI prefix from the section start: 0x37, at most 32 ordinal-ordered KPARAM records, 0x19, then PARAM_CBANK. It checks parameter bounds, pointer alignment/extent and the referenced constant bank. The SHA prefix ends immediately before 0x50. The remaining attribute stream must be bounded, contain exactly one matching GPR declaration, and contain no later ABI override. The scalar KPARAM encoding does not state its byte width; this checker does not invent one.

It hashes the exact ABI prefix, retained cbank bytes and QMD0/QMD1 pair with SHA-256 and compares them to the Composer-owned serialized launch identities. QMD grid/block, register allocation and shared-memory declarations are cross-checked against launch metadata, including when a changed QMD is rehashed. UGPR/barrier fields have no separate observed declaration in the existing exporter; their authority remains the selected canonical compiler/encoder lineage and code identity, not a newly claimed QMD measurement. Existing full admission still owns zero-VA/template policy, target/driver checks and the complete manifest/file identity.

Success is zero. Refusals: -20 shape/padding, -21 ELF framing/ranges, -22 unique section/name/links, -23 ABI stream/parameter/cbank/GPR mismatch, -24 admission-ID syntax/mismatch, -25 QMD scalar mismatch, -26 ABI-prefix SHA, -27 cbank SHA, -28 QMD-pair SHA.

`templates_fixture.om` preserves the exact first independently admitted C exporter fixture from `tests/native_artifact_manifest_test.c`, including its cubin and templates as immutable hex bytes. It uses `compose_launchwrite`, not a parallel canonical serializer. Its standard SHA oracle prefix was independently located by `readelf`: section offset 0x560, length 40 ending before 0x50. This synthetic zero-text fixture has `device-load=NOT_RUN`; successful admission is not successful execution or numerical proof.

First compile the common dependencies plus `templates.om` **without a main function**, exactly as the production bridge does. This function image must compile on the production-pinned compiler. Function mode forbids top-level text globals; the checker uses allocation-free packed prefix words already filed by the cubin packer instead. A program-mode test alone cannot establish bridge compatibility.

Then compile two program units using the same compiler: the common dependencies plus `templates.om` plus `templates_fixture.om`, and the common dependencies plus `templates.om` plus `templates_test.om`. Pipe the fixture executable into the test executable. The unchanged native test must exit zero. It checks positive admission-fixture binding, changed IDs/name/reserved bytes, malformed ELF/link/duplicate sections, ordered ABI/parameter/cbank/GPR checks, all three digest bindings, later ABI overrides, and QMD semantic drift even after rehashing. `templates-evidence.json` records the original independently preserved snapshot comparison and program-mode runs. `templates-function-evidence.json` records the subsequent production function-mode fix, same-source fixture regressions, caught guard-removal mutants and a text-global negative control. Neither receipt replaces the integration owner's actual existing-runtime lifecycle smoke.

## Session-resident metadata

`residency.om` is a standalone native unit. All storage is caller-owned: a zero-initialized `ulong[528]` contains a header and up to 32 fixed rows, matching the combined existing GEMV/GEMM/TC cache slot bound. `kc_init(pool,generation,profile,byte_budget,slot_budget)` accepts a nonzero profile/budget and 1 through 32 slots. Generation zero is valid: the existing `omega_gpu_session_recoveries()` and command-list generation start there. Reinitializing an active pool refuses; session changes use `kc_invalidate` with the strictly newer existing recovery generation. Profile is an opaque runtime admission identity covering target and driver compatibility, not a replacement for checking them.

The caller supplies `ulong[16]` requests:

| Words | Meaning |
| --- | --- |
| 0, 1 | Current session generation and admitted profile |
| 2 | Actual charged resident code allocation bytes, including upload padding |
| 3 through 6 | SHA-256 realization digest, four little-endian u64 limbs |
| 7 | Nonzero, strictly increasing registration owner ticket within the session |
| 8 through 15 | Reserved; caller initializes to zero |

The pool contains no device handles or addresses. Callers must not mutate pool cells directly. `kc_revision(pool,slot,generation)` returns zero for invalid/stale/busy entries and otherwise the nonzero resident revision. `kc_livebytes` reports committed resident code bytes for the current generation. The header also records lookup hits, misses and committed evictions; these are bookkeeping, not timing or serving-throughput evidence.

The existing allocator is authoritative for word 2: use `omega_gpu_code_alloc_bytes(code_size)`, exactly as `omega_gpu_session_alloc_code` does, without duplicating its padding arithmetic. Raw payload length is separately bound by `kc_validate`'s `sizes[1]` and the code digest. `kc_request` deliberately preserves the caller's charged size. `kc_find` refuses an inconsistent charged size for an otherwise identical resident key; a hash hit cannot detach physical budget accounting from the real allocation. Selector row 16 carries the same charged allocation size.

### Lifecycle

1. Validate the immutable artifact and exact target/profile. Call `kc_prepare(pool,request)` under the existing session lock. Positive results are a zero-based slot plus one. A hit needs no compile/upload. A miss reserves a slot transactionally.
2. Keep the selected old handle alive while allocating and uploading the replacement through the existing runtime. Budget admission includes both the old committed allocation and every pending upload. `kc_commit(pool,slot,owner,generation)` publishes the new identity and revision only after successful upload. The caller can then free the old, unreferenced allocation. `kc_abort` clears only the matching pending reservation and preserves the old identity and revision.
3. Known-shape preparation uses `kc_hold(...,+1)` after commit/hit, paired with `kc_hold(...,-1)` on reservation release. `kc_pin` holds each recorded command-list reference. `kc_acquire` adds an in-flight reference before submit, and requires a recorded reference. `kc_unpin` releases a recorded reference only when its list drops the recording. Prepared, recorded and in-flight references are separate checked counters.
4. `kc_retire` releases an in-flight reference only after the existing command-list completion mechanism has observed successful completion. Timeout, interruption and failed submission recovery are not successful completion. They must not retire uncertain in-flight references. Code remains pinned until completion or authoritative session recovery invalidates dead handles.
5. `kc_seal(pool,generation,1)` refuses while registration is pending. Sealed misses refuse deterministically: no compilation, allocation, upload or eviction. Existing hits remain usable. Unsealing does not bypass reference counts.
6. Existing session close/recovery destroys old device handles, then `kc_invalidate(pool,new_generation,new_profile)` clears only resident bookkeeping. It requires a strictly newer generation. Immutable portable artifacts remain available for fresh admission. Every old list/token refuses reuse.

Empty slots are chosen first. Otherwise deterministic round-robin replacement skips pending, prepared, recorded and in-flight rows. Owner tickets and revision tokens prevent interrupted or concurrent registration from committing somebody else's slot. All mutation must be serialized by the existing session lock; no atomics, unsafe wrappers, second allocator or second serving runtime are introduced.

### Status codes

`kc_prepare` and `kc_find` return `slot+1` on a hit/reservation. `kc_find` alone returns zero for a miss. Other successful operations return zero. Refusals: -1 invalid input, -2 stale generation/profile, -3 sealed miss, -4 simultaneous byte budget, -5 pinned/full slot budget, -6 pending/busy, -7 owner/revision token mismatch, -8 reference-state violation, -9 counter/revision overflow. Artifact validation uses -10 malformed canonical record, -11 code identity mismatch, -12 launch identity mismatch and -13 stored realization-key mismatch.

## Host checks and limits

Concatenate `residency.om` and `lifecycle_test.om`, then compile and execute with the production-pinned compiler. The native test exercises duplicate pending registration, full/pinned refusal, wrong owner, abort preservation, prepared/record/in-flight lifetimes, interrupted completion, sealed miss, coexistence byte budget, owner-ticket reuse, revision reuse and stale-session refusal. It must exit zero. Removing the in-flight victim guard must make the unchanged test exit 12.

Concatenate `digest.om` and `digest_test.om`, compile, then feed exact stdin bytes. It emits the binary SHA-256 digest for comparison with an independent existing SHA-256 checker. The caller cross-checks both native entry points for lengths within the canonical-record capacity. Wrong-round-constant mutation must disagree with the independent oracle. Over-capacity input must refuse.

Concatenate `quant.om`, `realize.om`, `digest.om`, `residency.om`, `artifact.om`, Composer's `fixture.om`, then `artifact_test.om`. This host-only fixture invokes the sole-owner QuantSchema/Composer serializers, validates payload hashes and launch declarations, constructs a cache request, commits residency, and proves a changed valid scale declaration misses. It also refuses changed scale, tile, spec, target and numerical identities; corrupt code; malformed launch framing; canonical padding; wrong stored keys; and invalid payload bounds. Synthetic fixture payloads and lineage digests are not production artifacts. Omitting QuantSchema bytes from the hash makes the unchanged test exit 12; skipping launch declaration consistency makes it exit 23.

`host-evidence.json` records the initial exercised source at commit `f0909113243674eeda1cf7539710ac425978e2f1`, dependency hashes, compiler and executable identities, digest oracle vectors, native caller exits and caught mutants. `generation-evidence.json` preserves that receipt and adds the final source hashes, immutable QuantSchema/Composer dependency commits, generation-zero compatibility caller, and the caught zero-generation rejection mutant. Storage capacity bookkeeping is supported by the mounted Wolfram query `16+32*16 -> 528` in `/home/drakestapleton/handoffs/wolfram-lane-query-results-20261028.json`; it proves no lifetime property.

`digest-varied-evidence.json` additionally checks exact varied source and native-executable byte prefixes against the independent SHA-256 oracle. These exercise direct full-block byte order, not only repeated-byte padding boundaries.

Host policy checks are not upload, launch, whole-model or hardware qualification. Hardware upload timing, paired cold/warm microbenchmarks and full-model acceptance remain NOT_RUN until an explicitly authorized window. Production bridge registration and existing runtime caller cutover are coordinated by the integration owner, not duplicated in this directory.
