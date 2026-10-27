# Host-reference receipt, not native/GPU qualification

`receipt.json` names the immutable executable/test source commit and all source,
compiler and ELF digests. `streams.tsv` preserves observed export/checker results.
All new executable product and behavioral test source is `.om`. The development
referee below is existing external Omega code, not a new implementation or
fallback in CUDApp. No generated AArch64 ELF was executed on this x86_64 machine.

## External referee and compiler source

In a separate scratch directory, export `tools/omega_ref/omega_ref.c` from
Omega commit `190130b71d0a59b93f01c515218a5509b149a145`. Check its source hash in
the receipt. Compile it using the measured GCC version and flags:

```sh
cc -O2 -g -std=gnu11 -Wall -Wextra -Werror -o omega_ref omega_ref.c -lpthread
```

The local debug build's binary hash is recorded, but debug paths may make a
different build directory produce a different referee binary. Source identity
and recorded build options must not be confused with binary reproducibility.

Export `seed/omc/omc.om` at compiler revision
`7505b3a3e3cf5ee46933e3a89a021b4289037983`. Its independently recorded seeded
binary SHA256 is `499236398aabc652e2dae27fe4bdc4c5db3d3726314356e0dbef23e3345803ec`.
Do not use a sidecar rewritten by a new birth as the expected digest.

```sh
./omega_ref check omc.om
./omega_ref run omc.om < omc.om > compiler-self.elf 2> compiler-self.stderr
sha256sum omc.om compiler-self.elf
```

Both commands exited 0, stderr was empty, and the output matched the recorded
seeded binary digest. This is compiler-source execution by a referee, not native
compiler execution, diverse compilation, or a complete seed-chain qualification.

## Compose and compile

Use the exact source commit in the receipt, preferably a fresh `git archive`.
Compose the common source and three entry points as in
`docs/shader-pilot/README.md`. For each `UNIT` (`core`, `export`, `stream-check`):

```sh
./omega_ref check "$UNIT.om"
./omega_ref run omc.om < "$UNIT.om" > "$UNIT.elf" 2> "$UNIT.compile.stderr"
```

All commands exited 0 with empty stderr. An independent immutable Git export
reproduced the three ELF files byte-for-byte. Do not execute them on x86_64.
Semantic runs in this receipt instead use:

```sh
./omega_ref run core.om
printf '\002\001' | ./omega_ref run export.om > task2.cul1
printf '\003\001' | ./omega_ref run export.om > task3.cul1
printf '\002\002' | ./omega_ref run export.om > vector2.bin
printf '\003\002' | ./omega_ref run export.om > vector3.bin
cmp vector2.bin vector3.bin
```

The core prints `SHADER_PILOT_CORE_PASS`, exit 0. Every export is repeated with
identical bytes and exit 0. Both code streams match the previously recorded
unchanged vector hash, not an independently decoded or executed GPU result.

Prefix a valid export with its original two-byte request and feed it to
`./omega_ref run stream-check.om`: expected exit 0. For each selection, remove
the last payload byte or append one byte: expected 65. Flip bit0 of the first
or last payload byte: expected 1. Substitute ID3 for the ID2 task: expected65
for the original short task, and mismatch1 after extending it to task3 length.
The core also rejects corruption at every byte offset of both candidates.

Exporter refusal matrix, all with empty stdout:

- Empty, one-byte, three-byte, and a valid two-byte prefix plus a long suffix:65.
- Every one-byte ID except2/3, with selector1:77.
- Every one-byte selector except1/2, with ID2:77.
- Closed stdin:74. Redirect valid request stdout to `/dev/full`:74.

These are process status observations, not a claim of a native sandbox or wall
timeout. Callers must bound wall time independently; a pipe without EOF can wait.

## Red tests and wrong implementations

The initial core test failed with `SHADER_PILOT_CORE_FAIL`, exit1, using explicit
unimplemented `spload`/`spmatch` stubs. The added profile tests similarly rejected
an always-true `spprofile` stub. An exporter stub returning77 with no payload was
rejected on the valid request by status and by stream-check exit65. The final
complete core test was also compiled and rerun against all-unimplemented stubs;
it still fails rather than silently becoming a passing test.

`receipt.json` contains exact replacement strings for every measured mutation.
Apply each once in a disposable source copy, compose it with the unchanged
entry point, require successful compiler/reference checks, then run it.

Eight core mutations each print `SHADER_PILOT_CORE_FAIL` to stderr and exit1:
omitted task scan, omitted last code byte, short task/code accepted, stale
catalog selection, omitted stride/shape guard, and omitted canonical reload.
Two exporter mutations instead produce the deliberately wrong externally
observable result: a three-byte request succeeds with payload, or a task request
emits instruction bytes. The unchanged refusal/output expectations detect both.
Compiler acceptance alone did not validate any mutant.

The original unchanged #53 evaluator/core and catalog tests also ran under the
external referee, with their original PASS lines preserved. This does not rerun
their historical native compiler or establish native regression qualification.

## Timings and other failed checks

Raw timing rows in `receipt.json` are **MEASURED** reference-process startup,
parsing, compiler interpretation and ELF output. They are not native compile
latency, a cache-hit speedup, GPU time or a graphics frame result. The timed
series followed earlier validation compiles; no first-sample cold-cache claim
is made. The original GNU-time attempt failed because `/usr/bin/time` was absent.
The measured fallback was explicitly Bash `time`, `TIMEFORMAT='%R,%U,%S'`;
peak RSS was not collected. No summary statistic or performance comparison is
claimed. Every timed process emitted the same successful ELF.

Both before and after this change, `make test` exited2 on existing AArch64
`ds` assembly in `omega_blackwell_engine.c:393,558` under the x86 assembler.
The figure check exited1 for the known stale `stack.svg`. `crumb verify .`
exited1 only for existing trace documentation/tests; newly added pilot crumbs
were current. These failures were not hidden, weakened or repaired outside scope.
The standalone `.om` tests still need the existing #15 native-build owner to
integrate them. Native AArch64, GPU, independent instruction/launch validation,
cache/pipeline/first-use/graphics work remain NOT_RUN.
