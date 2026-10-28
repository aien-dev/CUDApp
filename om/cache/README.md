# Native kernel cache policy

This directory implements issue #158. It does not open a GPU, allocate device memory, create a session, or replace command lists. The existing FP8 GEMV, GEMM and tensor-core caches remain the owners of code allocations. Their existing session lock must cover every policy operation, including find/revision/pin sequences. A native unit cannot supply thread synchronization at the pinned OM-1 compiler.

## Immutable artifact identity

`digest.om` implements bounded SHA-256 using the existing Omega SHA-256 algorithm and filed decimal constants. `kc_sha` accepts `uchar[4096]`, an exact used length, and `uchar[32]` output. `kc_codesha` accepts `uchar[16384]`. Full blocks are read directly; only the final partial block is copied. Inputs exceeding capacity refuse without writing the output.

`artifact.om` consumes, rather than redefines, the QuantSchema and Composer serializers. Concatenate `quant.om`, `realize.om`, `digest.om`, then `artifact.om`. `kc_key` validates the canonical record with `compose_check`, then hashes exactly its used bytes. `kc_request` puts the digest into request words 3 through 6 as little-endian u64 limbs. It does not change caller-supplied generation, profile, code size or owner ticket.

The realization key is not the full artifact-store manifest ID. Existing `omega_native_artifact_export` and `omega_native_artifact_admit` remain authoritative for immutable portable artifact files, framed templates, relocations, target and driver admission. The cache never treats a filename, a hash match, or metadata validity as hardware qualification. `kc_validate` invokes the Composer-owned `compose_launchread` to check canonical launch declaration consistency, then independently binds code and launch bytes to their record digests and checks the stored realization key. The admitting runtime must still enforce existing full artifact admission and actual template-content identity before upload. Canonical bytes and code remain immutable for their entire registered lifetime; device handles and virtual addresses never enter the realization key.

## Session-resident metadata

`residency.om` is a standalone native unit. All storage is caller-owned: a zero-initialized `ulong[528]` contains a header and up to 32 fixed rows, matching the combined existing GEMV/GEMM/TC cache slot bound. `kc_init(pool,generation,profile,byte_budget,slot_budget)` accepts a nonzero profile/budget and 1 through 32 slots. The metadata ABI permits generation zero for historical host controls. Integrated callers use the actual monotonic session epoch, not a recovery count that can alias a normal close/reopen. Reinitializing an active pool refuses; changes use `kc_invalidate` with a strictly newer epoch after device handles are dead. Profile is an opaque runtime admission identity, not a replacement for target/driver checking.

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

## Integrated native entries

`runtime.om` retains operations0..15 and supplies16 (validate current token),
17 (initialize/current-context/revoke metadata), and18 (derive the next owner
ticket and prepare a private request copy). Operation17 returns1 after a new
context and0 for an unchanged valid context. It cannot prove device teardown:
the existing session owner must destroy handles before advancing the epoch.
Operation18 refuses owner overflow and never mutates its borrowed request.

`lease.om` is a separate final native entry that checks retained session epoch
and nonzero revision before dispatch. `slotrev.om` returns the full unsigned
revision without a signed-host conversion. Each production frame must end in
its declared entry; concatenating several public roots does not export them
from the seeded compiler. `om/units.sh` registers these separate frames.

Existing FP8/GEMM/TC owners invoke generated policy under the session lock.
Actual selected artifacts retain canonical bytes, code, launch, manifest ID
and the existing immutable artifact-admission snapshot through preparation.
Runtime allocation/upload/recording consumes those exact bytes and leases.
No caller-supplied qualification flag, realization hash, or cache hit replaces
the artifact manifest or independently accepted qualification authority.

`runtime-separated-evidence.json` records the exact separated-entry source,
64 exercised controls and9 caught compiled mutants. Earlier operation/lease
receipts remain source-bound historical observations, including their explicit
entry-layout limitation. `runtime_ops_test.om` is the native host conductor.
The existing fake-platform oracle additionally exercises actual runtime
preparation/recording, close/reopen revocation, timeout retention and dead-handle
recovery. Its handles and completion model are synthetic, never device evidence.

## Host checks and limits

Concatenate `residency.om` and `lifecycle_test.om`, then compile and execute with the production-pinned compiler. The native test exercises duplicate pending registration, full/pinned refusal, wrong owner, abort preservation, prepared/record/in-flight lifetimes, interrupted completion, sealed miss, coexistence byte budget, owner-ticket reuse, revision reuse and stale-session refusal. It must exit zero. Removing the in-flight victim guard must make the unchanged test exit 12.

Concatenate `digest.om` and `digest_test.om`, compile, then feed exact stdin bytes. It emits the binary SHA-256 digest for comparison with an independent existing SHA-256 checker. The caller cross-checks both native entry points for lengths within the canonical-record capacity. Wrong-round-constant mutation must disagree with the independent oracle. Over-capacity input must refuse.

Concatenate `quant.om`, `realize.om`, `digest.om`, `residency.om`, `artifact.om`, Composer's `fixture.om`, then `artifact_test.om`. This host-only fixture invokes the sole-owner QuantSchema/Composer serializers, validates payload hashes and launch declarations, constructs a cache request, commits residency, and proves a changed valid scale declaration misses. It also refuses changed scale, tile, spec, target and numerical identities; corrupt code; malformed launch framing; canonical padding; wrong stored keys; and invalid payload bounds. Synthetic fixture payloads and lineage digests are not production artifacts. Omitting QuantSchema bytes from the hash makes the unchanged test exit 12; skipping launch declaration consistency makes it exit 23.

`host-evidence.json` records the initial exercised source at commit `f0909113243674eeda1cf7539710ac425978e2f1`, dependency hashes, compiler and executable identities, digest oracle vectors, native caller exits and caught mutants. `generation-evidence.json` preserves that receipt and adds the final source hashes, immutable QuantSchema/Composer dependency commits, generation-zero compatibility caller, and the caught zero-generation rejection mutant. Storage capacity bookkeeping is supported by the mounted Wolfram query `16+32*16 -> 528` in `$HOME/handoffs/wolfram-lane-query-results-20261028.json`; it proves no lifetime property.

`digest-varied-evidence.json` additionally checks exact varied source and native-executable byte prefixes against the independent SHA-256 oracle. These exercise direct full-block byte order, not only repeated-byte padding boundaries.

Host checks do not qualify upload timing, numerical execution, hardware, or a full model. Paired cold/warm chip benchmarks and full-model acceptance remain NOT_RUN until explicitly authorized. Shared production registration and caller binding are implemented by the integration owner; the production graph resolver remains NULL/count0 because no new exact artifact has accepted independent hardware/full-model qualification.
