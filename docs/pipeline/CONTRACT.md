# Deterministic dual-pipeline policy (#57)

This is a pure native Omega predicate, not another tuner, qualification service,
GPU runtime, or activation gate. It scans at most sixteen owner-resolved records.
No M cutoff exists. Direct GEMV (family 1), register-fed warp MMA (2), and staged
GEMM (3) compete only on a frozen complete-workload cost. A family number does
not establish hardware support. The incumbent is preserved on ties and failed
alternative admission; an inapplicable incumbent causes refusal, not an unsafe
fallback. PR #139 remains unqualified: its filed full-cost gate fails small
cells. Nothing in this policy imports that branch or changes current defaults.

## Trust boundary

`pl_select(request[64], records[1024], output[16])` consumes trusted inputs from
the existing #35 realization/#54 candidate, #53 evaluator, and #17 measurement
owners. Matching digests and gate bits do **not** authenticate evidence. The
owner must resolve a single immutable qualification bundle whose subjects bind
the canonical realization, exact emitted code and launch, quant descriptor,
compiler/encoder/spec pins, frozen workload, graph dependencies, numerical mode,
physical execution, full-path performance and model quality. Candidate writers
must not supply or modify these records. No new signer or trust store is added.

The predicate cannot turn arbitrary caller assertions into physical evidence.
Synthetic tests are policy controls only and are not production-consumable
qualification bundles. No production registry is populated by this change.

## Request cells (unsigned 64-bit)

| Cells | Meaning |
| --- | --- |
| 0 | Version 1 |
| 1 | Active record count, 1..16 |
| 2 | Incumbent zero-based record ordinal |
| 3..6 | Session generation, exact target/driver profile, buffer ownership generation, lifecycle context (cold/warm/prepared identity) |
| 7..10 | Available workspace bytes, register ceiling, shared bytes, thread ceiling |
| 11 | Exact admitted capability bits; never inferred from SM number |
| 12 | Reviewed minimum absolute complete-path benefit, ns |
| 13 | Reviewed maximum complete-path cost, ns |
| 14..17 | Frozen workload-equivalence SHA256 |
| 18..21 | Current graph/dependency closure SHA256 |
| 22..25 | Reviewed policy SHA256 (includes per-workload constraints and aggregation protocol) |
| 26..29 | Frozen measurement/comparison protocol SHA256 |
| 30..33 | Incumbent artifact SHA256; invalidates dependent evidence on replacement |
| 34..37 | Exact applicability SHA256 from canonical dependency adapter |
| 38..63 | Reserved zero |

All digest groups contain four canonical little-endian unsigned 64-bit limbs and
must be nonzero. Applicability binds exact semantic/quant/shape/routing/layout/
strides/alignment/tails/alias/numerical contract and compiler/encoder/spec pins.
Alternative tile/warp/stage/pipeline/code identities are excluded **only** from
workload equivalence, never from individual qualification subjects. Resource and
lifecycle fields are checked separately, not discarded from receipt scope.

## Record cells (sixteen rows, stride 64)

| Offset | Meaning |
| --- | --- |
| 0 | Family 1 direct GEMV, 2 warp MMA, 3 staged GEMM; unknown refuses |
| 1 | Independent semantic, byte/launch, physical, full-workload, quality gates: exact mask 31 |
| 2 | Known nonoverlapping phase mask, exactly 63; zero durations mean measured zero, not unknown |
| 3 | Complete-path cost, ns, positive |
| 4..9 | Packing/conversion, upload/transfer, preparation (compile/cache invalidation), dispatch, synchronization/barriers, complete execution, ns |
| 10 | Microkernel-only cost, diagnostic; never used to rank |
| 11..14 | Workspace bytes, registers, shared bytes, threads |
| 15 | Required hardware feature bits, subset of admitted exact profile |
| 16 | Charged resident allocation bytes, including actual upload padding (exact cache request size) |
| 17..20 | Same lifecycle/generation fields as request 3..6 |
| 21..24 | Artifact SHA256 |
| 25..28 | Launch SHA256 |
| 29..32 | Immutable resolved qualification-bundle SHA256 |
| 33..36 | Workload-equivalence SHA256 |
| 37..40 | Graph/dependency closure SHA256 |
| 41..44 | Reviewed policy SHA256 |
| 45..48 | Measurement protocol SHA256 |
| 49..52 | Incumbent dependency artifact SHA256 |
| 53..56 | Canonical applicability SHA256 |
| 57..60 | Final code SHA256, separate from canonical realization artifact digest |
| 61..63 | Reserved zero |

Unsigned phase addition is checked before every operation; overflow or a phase
sum exceeding measured complete-path cost rejects the row. Components must be
nonoverlapping under the measurement protocol; amortization is supplied by its
frozen lifecycle record, never calculated from a guessed reuse count. Packing,
upload and preparation may not be silently omitted for a cold candidate.

Duplicate artifact identities refuse the set, avoiding ambiguous slot reuse or
bundle choice. Strictly lower complete cost wins; incumbent wins equal cost.
Other ties use ascending canonical digest limb order. Array order cannot change
the chosen identity. Unknown/unqualified alternatives cannot replace incumbent.
No TMA or tcgen05/TMEM support is inferred on SM121: the target owner and physical
receipt must explicitly qualify each required feature and exact artifact.

## Result and consumption

Return 0 means a policy result, 65 malformed/ambiguous set, 69 no qualified exact
incumbent. Output is cleared on every refusal. Success output: 0 valid marker 1;
1 ordinal; 2 family; 3 complete cost; 4..7 artifact digest; 8..11 qualification
bundle digest; 12..15 session/profile/buffer/lifecycle snapshot.

This result is advisory, **not** a kernel handle or launch capability. Before
recording, Main's existing graph/cache owner must revalidate the exact bundle,
request snapshot and generations and atomically acquire/pin the artifact and
all tensor/resource dependencies through existing completion. Cache slot reuse,
eviction, recovery, tensor changes or budget changes invalidate the result.
Do not retry the incumbent after any partial candidate execution: state rollback
belongs to the existing KV/recurrent state owner, not this selector.

`pl_realcheck(request_real[4096], candidate_real[4096], row[64])` consumes the
Composer's exact `compose_match`, `compose_codeid`, and `compose_launchid` and
Cache's `kc_sha` over precisely 1152 canonical bytes. Quant validation is the
Composer validator's exact `quant_check` dependency. No parallel descriptor
serializer exists here. Return 69 rejects changed applicability/code/launch or
artifact identity. `pl_pin(request, records, cache[528], output, lease[16])`
then takes a recorded-code reference on an already prepared hit. It never
compiles, uploads, evicts or prewarms during graph recording. Miss/stale/busy
returns 75 and clears result/lease. Lease contains valid marker, slot, revision,
generation, artifact/bundle digests, profile/buffer/lifecycle context, reserved.
The existing runtime must use `kc_acquire` before submission, `kc_retire` only
after observed completion, and `kc_unpin` when the recorded list is discarded.
All calls require the existing session lock; this is not cross-process isolation.
Resident charge in row16 is the existing allocator's actual budget charge,
including its allocation padding, not the raw instruction payload length.
The latter remains independently checked by `kc_validate` sizes/code SHA.
`kc_find` must retain exact size matching: never hide an allocation difference
behind a key hit or silently detach resident accounting from the real allocation.

`pl_graph(q, rows, pool, request_real, selected_real, decision[32])` is the
actual native graph-admission caller: select, validate canonical realization and
declared family/features/GPR/shared/threads, then pin. Decision cells0..15 are
the policy output and16..31 the lease. Refusal clears all32 cells. The graph
owner resolves `selected_real` from the exact chosen artifact, not a shape
lookup. It must also check `compose_launch` and actual code/templates through the
existing artifact loader; matching stored digests alone does not check blobs.

The frozen Composer schema `2e93f998b70852abcd425f8865d842a05dad76ea`
used by these host receipts admits native direct1 and row-pipelined2 only.
These receipts exercise family3 policy with synthetic records and explicit
canonical staged refusal. Newer Composer #166 adds structural family3 support;
it does not confer candidate qualification or device/full-model evidence.
Unknown families/features remain refused.

The shared integration binds the selected immutable artifact to the actual
FP8/GEMM cache upload and recorded code handle under the existing session lock.
The private `src/omega_gpu_q36_native.h` resolves exact graph-owned contexts,
prepared artifacts, canonical requests and immutable owner observations.
The call binding additionally checks actual tensor allocation epoch against
current session epoch and its independently assigned owner generation.
Failed canonical, artifact, session, ownership or qualification checks refuse;
they do not retry an incumbent after partial candidate work.

The actual `rec_fp8`, `rec_gemm` and paired expert recording paths consume that
selection instead of discarding it. A generic standalone selected expert
record refuses when a paired GROUP/window contract is required. No API lets
candidate writers populate accepted records. Production graph state remains
NULL/count0 until the existing independent evaluator/receipt owners accept
exact artifacts; current defaults and the incumbent path are unchanged.

## Predeclared measurement families

Use actual `omega_gpu_q36_graph.h` contracts: decode M=1; supported conversation
strokes through MAX_HEADS; batch v2 dense and nine-group experts; prefill through
PREFILL_MAX_ROWS including boundary and ragged expert segments. Dense shapes are
GDN mix N=12288,K=2048; attention mix N=9216,K=2048; output N=2048,K=4096;
gate/up N=1024,K=2048; down N=2048,K=512. Freeze real checkpoint/target/config,
per-workload bounds, warm/cold policy, aggregation weights, numerical thresholds
and minimum benefit through #17 before final measurements. No invented timing
threshold or paper-derived performance ratio is adopted here.

Predeclare decode head counts1..8 and prefill row counts
`{1,2,4,8,16,32,64,128}`, plus ragged/empty expert segments at
`{0,1,7,8,9,127,128}` where each existing API's capacity permits. These are
measurement cells, never selection thresholds. Include existing #139 failing
small cells unchanged; a new family does not inherit that PR's correctness
receipt or erase its failed performance evidence. Compare both cold and
already-prepared cache lifecycles, one-shot and frozen repeated-use requests.

Measure preprocessing, repack, compile, upload, dispatch, synchronization and
complete execution using paired/interleaved complete paths, then full-model
quality/latency against the filed #604 baseline. Preserve every failure and
inconclusive result. Host policy controls are not device or serving evidence.

## Host controls and reproduction

`tests/pipeline/host-evidence.json` records exact ordered source hashes, pinned
compiler, all three native caller exits0, and five compiled guard mutants
rejected by those same controls. `fixture.om` and Composer's `fixture.om` are
test-only synthetic declarations. They never enter a trusted production registry.
The current source/file-content dependency pins are in `dependencies.lock`;
Main integrates the exact peer commits before bridge registration.
`tests/pipeline/pin-invalidation-evidence.json` adds the exact compiler-digest
mutation at canonical byte544, recording the final graph-test source/binary
and successful control plus caught bypass mutant. The earlier receipt preserves
the preceding semantic-digest mutation rather than rewriting that evidence.
`tests/pipeline/final-source-evidence.json` is the final-source run of all three
controls and five mutants after aligning with existing session recovery
generation0. Cache initial generation0, pin/inflight/completion and later
invalidation are exercised. Earlier source receipts remain historical;
dependency lock and this final receipt identify the integrated source exactly.

Compile each test by concatenating Quant's `quant.om`, Composer's `realize.om`,
Cache's `digest.om` and `residency.om`, this lane's `select.om`, `graph.om`,
`consume.om`, and `tests/pipeline/fixture.om`, followed by the test source.
For `graph-caller.om` insert Composer's shared `fixture.om` before the test.
Feed that source to the exact pinned `omc`; execute the host AArch64 output.
The policy, cache-caller and graph-caller expected exit is0. Mutation definitions
and observed nonzero exits are retained in the JSON receipt; a mutant compile
failure would not count as detection.

Independent symbolic synthetic cost oracle, through the mounted parent proxy:
`WolframLanguageEvaluator:
{Simplify[packing+upload+prepare+dispatch+barriers+execution,Assumptions->And@@Thread[{packing,upload,prepare,dispatch,barriers,execution}>=0]],Total[{2,3,5,7,11,13}],Total[{1,1,1,1,1,100}],Total[{20,20,20,20,20,1}],100-41,100-105,100-101}`
`-> {barriers + dispatch + execution + packing + prepare + upload,41,105,101,59,-5,-1}`.
These costs are explicit synthetic policy controls, not observed nanoseconds,
speedups, GB10 measurements or serving evidence.

## Private graph context lookup

`om/pipeline/context.om` exports
`native_context(keys view ulong128, count ulong, actual view ulong4) -> long`
for the existing ABI1 bridge. Up to thirty-two active private graph-owned tuples
have stride4: `{operation kind, layer index, window start, window rows}`.
Return is the unique 1-based ordinal,0 absent,-1 for count above32 or multiple
matches. Count0 is a normal absent binding. Unused cells are ignored; no portable
serialization, padding convention or artifact schema is introduced.

Each selected ordinal identifies its own immutable existing query, rows, wanted
canonical record, candidates and prepared artifacts under the existing session
lock. This lookup must never derive a new wanted/query from the selected code.
The graph actual execution tuple supplies the lookup key; candidate/tensor
lifetimes remain owned by the graph. Unknown operation kinds may match raw
registered tuples here; support and exact MNK/launch are independently refused
by the existing binding validator. Matching a tuple never authenticates a
qualification, joint MoE closure, router/permutation/combine semantics, or a
production registration. With no accepted receipt-authority construction path,
production context count remains0.

`tests/pipeline/context.om` exercises count32/33, every ordinal including31/32,
empty/invalid counts, every tuple field, late excluded entries, a duplicate at
the final active slot and unrelated duplicate isolation. Current focused
controls and compiled old-cap/tuple/duplicate/count-guard mutants are recorded
in `tests/pipeline/context32-evidence.json`. The original sixteen-context
`context-evidence.json` remains historical, unchanged. Compile the test after
only the exact context unit; it has no Quant/Composer/Cache dependency. This
bounded private arena does not promise a model-wide context organization.

## Actual recording and mode-final closure binding

`omega_gpu_cmdlist_owner` and `omega_gpu_cmdlist_entry` project real borrowed
owner descriptors, successful-add ordinals, code VA, argument count, generation,
bank and QMD slots. Observers execute once under the session lock after every
slot has its final consumed PCAS/INLINE mode, before any synchronization or
queue commit. INLINE emits those same mode-final bytes, including list reuse
after PCAS. An observer refusal queues nothing. Uncertain completion retains
in-flight and recorded references; reset cannot revoke pending code.

For selected native additions, generated `native_mcseal` freezes the complete
active bank prefix and arguments plus both active384-byte QMDs before publishing
the successful add. Only the two mode-dependent QMD1 fields are normalized.
The private GU/DN `native_mcintent` guard binds roles6/7, actual ordinal and
generation, code VA, final mode and actual typed QMD-slot VA to that seal.
Legacy recording gains no extra seal computation. Seals are owner-intent
bindings, not signatures, manifest replacements, or qualification statements.

The graph projects actual router/directory/tensor/code owners. Six nested
canonical hashes are recomputed from those observations, and native coverage
checks the actual same-list role sequence and every GU/DN occurrence against
the graph-owned window plan. Neither expected wire bytes nor candidate flags
manufacture provenance. Missing owner provenance or an unsupported full-chunk
closure refuses. The canonical MC family3 refusal remains unchanged; structural
staged emission is not a qualified MoE closure or full-model capability.

`om/pipeline/test_mccover.om` exercises positive coverage and normalized mode
reuse, wrong identity/generation/count/order, missing GU/DN, active bank/QMD
mutation, and malformed mode/slot bindings. The independent native observer
fixture consumes actual existing host-runtime owner wires against intent hashes
frozen by the legacy C oracle before recording. Synthetic role0 labels and
fake-device completion prove host caller/byte binding only, not GROUP semantics,
kernel execution, physical correctness, performance or qualification.

