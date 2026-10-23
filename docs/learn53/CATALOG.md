# Frozen public CUL1 catalog and Omega653 adapter

This is a **native-host domain component**, not a generated-candidate experiment,
protected evaluator, authenticated verdict service or GPU qualification. All
implementation and behavioral tests are seeded `.om`. Existing Linux tools are
development orchestration only; #15 still owns the native build boundary.

## Frozen public profile

`kload(id:ulong)->ulong` writes the fixed task into `kt`, returning its length.
IDs outside1..8 return0 without modifying either `kt` or `ko`. Valid loads clear
all4096 task bytes before writing; they never modify candidate observations `ko`.
The contents, arithmetic mode, strides, tolerances and task membership cannot be
selected through observed-output bytes. Existing generic `kvalid`/`keval` remain
useful trusted-manifest APIs; this catalog does not replace or rename them.

|ID|Task|Task bytes|Guarded output bytes|
|--|--|--:|--:|
|1|add3, stride2, wraparound|136|28|
|2|add8, dense,7+9|160|40|
|3|add17, dense,UINT32_MAX+3|232|76|
|4|reduce3, stride2,UINT32_MAX+2+3|116|12|
|5|reduce8, dense, allUINT32_MAX|128|12|
|6|reduce17, dense, allUINT32_MAX|164|12|
|7|matmul M2,N3,K2, row strides3/4/4|144|36|
|8|matmul M3,N2,K3, row strides3/2/2|156|32|

CUL1 pool1 means public development. It maps to Omega653 public development
split2. All these cases are public, deliberately share algorithm/template
families, and are **not** disjoint training/adaptive/final evaluation pools.
Memorizing their outputs can pass them; no generalization claim is possible.
Exact emitted bytes/digests are published in the continuation receipt.

`catalog-export.om` is a development fixture emitter: exactly one stdin byte
selects ID1..8; stdout is only the frozen CUL1 task. Invalid frame length65,
unsupported ID77, I/O failure74; failure emits no task bytes. Its test checker
compares complete streams natively. Neither program executes candidate code.

## Canonical dependency, not a second framework

`native/learn53/omega653.lock` pins Omega source84d03c55 and the exact SHA256 of
`src/learning/benchmark/core.om`. Export that committed Git object explicitly,
check its digest and concatenate it with task.om, catalog.om and omega-adapter.om.
There is no vendored implementation, mutable sibling binary, implicit download
or changed upstream code. The inspected exact upstream LICENSE is GNU AGPLv3.
Its185-check component ELF is independently reconstructed byte-for-byte before
this consumer is exercised. Full upstream653 review/acceptance remains separate.

## Adapter API and authority

`aobserve(id,status,code,n)->ulong` consumes candidate bytes from `LGot[256]`.
It reloads the fixed catalog, validates CUL1, constructs independent expected
bytes from the existing `kexpect` reference (including both guards and padding)
in `LWant`, then calls the unchanged `lobserve`. It does not accept a claimed
PASS, tolerance, task payload or expected-output field from a candidate.
Result codes are canonical:0 correct,1..6 executor failures,7 wrong bytes,
8 exit/trap mismatch,9 empty output,10 malformed/unknown task.

`abegin(budget)->ulong` requires eight caller-owned LT records in ID order1..8:
public development split2, visibility1, numeric family=operation1/2/3, source
profile/schema1, oracle53, template1, input/output bounds from the table, declared
step bound1000000 and reserved0. The five digest groups retain their upstream
meaning and must be supplied by the qualified independent metadata owner; this
adapter does not invent or authenticate them. `lbegin(8,1,budget)` validates the
canonical manifest and starts a one-attempt-per-task run. If scalar catalog
fields disagree, `lbegin(0,1,budget)` intentionally invalidates the fresh run
(code1), clearing any prior completed-valid scores. An already active run is
refused with20 and preserved, matching the canonical lifecycle.

`aattempt(index,ordinal,status,code,n)->ulong` observes catalog ID=index+1 and
passes the derived outcome to `lattempt`. It never rewrites LC, LPin or the
outcome to fabricate success. Caller finishes using canonical `lfinish()`;
omitted tasks30, repeated ordinals28, bad identities26, missing costs23 and
exceeded time budget29 prevent a valid completed run. A wrong candidate may
produce a valid experiment record with a failure in the fixed denominator;
that is not candidate acceptance. Only completed-valid runs can report scores.
The step bound is declared metadata, not a process time limit enforced here.

All arrays are single-threaded, nonconcurrent scratch. LT, LPin, LC, status and
exit code belong to a **trusted executor**, not the candidate. Matching claimed
hashes do not authenticate an executable or establish custody. A same-process
hostile caller can change globals; this API is not a sandbox. Source hashing,
process supervision, measured costs, result authenticity and protected task
custody remain Omega653/L408/L605 and existing canonical owners' obligations.

## Evidence boundary

The tests use **explicitly synthetic** LT digest words, compiler/ELF identities
and cost vectors to exercise the canonical admission/accounting behavior. They
are not real candidate records, training feedback, measured baseline costs,
complete-workload performance, corpus qualification or release approval. Literal
observed-output fixtures are independently derived public arithmetic answers;
they do not claim to be output from a newly generated candidate program.

No runtime, encoder, disassembler, operation lowering, numerical contract,
benchmark collector, admission/session system, model graph, reuse store, trainer
or RSI judge is replaced. No legacy dependency is retired because no production
consumer has cut over. Larger generic CUL1 layouts remain valid in task.om but
are not silently narrowed into the256-byte Omega public observation profile.
Full #53 and #54–60 acceptance gates remain open.
