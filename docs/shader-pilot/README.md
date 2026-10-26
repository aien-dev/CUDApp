# Offline compute specimen for #86

This first increment connects existing components, not a graphics shader
compiler. It exports the unchanged `native/blackwell/vector.om` instructions
and two public #53 CUL1 tasks. It never opens a device, constructs a submission,
executes candidate code or loads a cache entry. The real community shader and
pipeline reproducer required by #86 is still missing.

## Supported semantics and ownership

| CUL1 ID | Input | Expected active output | Existing launch contract |
| --- | --- | --- | --- |
| 2 | Dense unsigned32 add, 8 elements, 7 plus 9 | Eight values of 16 | One 64-thread CTA, 56 inactive threads |
| 3 | Dense unsigned32 add, 17 elements, UINT32_MAX plus 3 | Seventeen values of 2, modulo 2^32 | One 64-thread CTA, 47 inactive threads |

Expected arithmetic and inactive-thread counts are **MODELED**, not GPU results.
Wolfram queries: `{Mod[7+9,2^32],Mod[(2^32-1)+3,2^32]}` gives `{16,2}`;
`{Ceiling[8/64],Ceiling[17/64],64-8,64-17}` gives `{1,1,56,47}`.

The existing emitter computes a linear index and exits out-of-range threads
before accessing memory. `native/blackwell/launch.om` sets the grid to
`(n + 63) / 64` and the vector constant argument to `n`. The pilot neither
calls that launcher nor proves that its machine instructions have these effects.
These tasks cover a fixed arithmetic/branch-boundary specimen at the source
contract level only. No new arithmetic lowering or control-flow kernel is added.

The profile requires disjoint dense input/output buffers, unsigned32 wraparound,
no floating point or tolerance, no subgroup communication, and no graphics stage.
CUL1 stores its existing two output canaries around the active result.
Its target/evidence-class word remains 1 (native-host reference), not 121.
Pairing that reference with raw code does not turn it into a GPU admission record.
Stride2, reduction, matrix multiply, arbitrary shapes, specialization and new
targets are refused rather than silently translated. The CUL1 evaluator and its
public task semantics remain owned by #53. Scheduling, transformations, reuse
and native build integration remain #22, #54, #58 and #15/#79 respectively.

## Product boundary

Compose `native/learn53/task.om`, `native/learn53/catalog.om`,
`native/blackwell/vector.om`, `native/shader-pilot/core.om`, then one entry point.

- `spprofile(id,n)->bool` checks canonical scratch `kt` is a supported dense
  unsigned32 add profile. It does not prove that an arbitrary task is catalog data.
- `spload(id)->ulong` reloads the frozen catalog task and emits `gbvec`, returning
  task length, or zero for unsupported ID/profile. An unsupported ID does not
  alter the previous scratch. A failed profile check may have reloaded `kt`;
  callers must honor the zero return and must not reuse previous results.
- `spmatch(id,tn,cn)->int` compares caller-owned `sptask[232]` and `spcode[512]`
  with freshly generated task/code bytes. It returns 0 for exact content,
  65 for bad length, 77 for unsupported selection, and 1 for a byte difference.
  Candidate buffers are never overwritten. All scratch is single-threaded.

This is a deterministic equality check, **not** cryptographic authentication,
source/driver/launch admission, an independent decoder or a new cache framework.
An adversarial same-process caller can modify globals. The matcher and exporter
share the existing emitter, so agreement between them cannot prove the emitter.
Keep the existing native-artifact admission and future seeded replacement as
the execution boundary; these raw files are not its admitted manifest format.

### Export request

`export.om` reads exactly two unsigned bytes, `[catalog_id, selector]`, then EOF.
Selector 1 writes the unchanged CUL1 task; selector 2 writes the unchanged
512-byte native instruction stream. Nothing else is written to stdout.
Bad framing exits 65; unsupported ID/selector exits 77; I/O failure exits 74.
Validation refusals emit no bytes. A write failure may leave a partial stream:
discard all output after **any** nonzero status. A closed pipe can also terminate
the process via ordinary Linux SIGPIPE. Only a zero exit validates export.

## Native-host replay, no GPU

Use a pinned, independently hash-checked `omc` on Linux AArch64. `OMC` is the
compiler, not the historical console or a reference interpreter. The checked
receipt identifies the compiler used for this increment. These shell commands
are development orchestration, not a new product build system.

```sh
mkdir -p build/shader-pilot
cat native/learn53/task.om native/learn53/catalog.om \
  native/blackwell/vector.om native/shader-pilot/core.om \
  > build/shader-pilot/common.om
cat build/shader-pilot/common.om tests/shader-pilot/core-test.om \
  > build/shader-pilot/core.om
cat build/shader-pilot/common.om native/shader-pilot/export.om \
  > build/shader-pilot/export.om
cat build/shader-pilot/common.om tests/shader-pilot/stream-check.om \
  > build/shader-pilot/stream-check.om
for unit in core export stream-check; do
  "$OMC" < "build/shader-pilot/$unit.om" \
    > "build/shader-pilot/$unit" 2> "build/shader-pilot/$unit.stderr" || exit 1
  chmod +x "build/shader-pilot/$unit"
done
build/shader-pilot/core || exit 1
printf '\002\001' > build/shader-pilot/request
build/shader-pilot/export < build/shader-pilot/request \
  > build/shader-pilot/task2.cul1 || exit 1
cat build/shader-pilot/request build/shader-pilot/task2.cul1 \
  | build/shader-pilot/stream-check || exit 1
printf '\003\001' | build/shader-pilot/export > build/shader-pilot/task3.cul1 || exit 1
printf '\002\002' | build/shader-pilot/export > build/shader-pilot/vector.bin || exit 1
printf '\003\002' | build/shader-pilot/export > build/shader-pilot/vector-repeat.bin || exit 1
cmp build/shader-pilot/vector.bin build/shader-pilot/vector-repeat.bin || exit 1
sha256sum "$OMC" build/shader-pilot/*.om build/shader-pilot/*.cul1 \
  build/shader-pilot/*.bin
```

Expected core stdout is `SHADER_PILOT_CORE_PASS`, exit 0. Task lengths are
160 and 232 bytes; code length is 512 bytes. On x86_64, native-host execution
is **NOT_RUN**. The existing frozen Omega referee can be used as a separately
labeled development oracle, not a substitute native compiler/runtime or a new
interpreter implementation. The receipt reports original `.om` source execution
by that external referee and compiler-source execution separately.

## Identity, stale/corrupted artifacts and receipt invalidation

Record exact repository revision and hashes for every composed input, original
compiler source and compiler binary, compiled entry-point ELF and each exported
file. Record compiler options (stdin program mode, no additional options here),
catalog ID, semantic contract, target sm_121, existing VECTOR_1D launch source
hash, and the required 580.173.02 driver/runtime contract as **unexercised**.
There is no live driver or GPU identity in this host-only receipt.

Always store the task identity with code: IDs 2 and 3 deliberately have identical
code but different task bytes and launch `n`. Code hash alone is not a compatible
task, launch or pipeline key. A changed source/compiler/options/semantics/launch/
target/runtime input invalidates prior evidence even when code bytes are equal.
This rule is a reproduction protocol, not an implemented persistent cache key or
authentication system. Do not feed these files to a driver as a shortcut.

The core test rejects every single-byte task/code mutation, wrong lengths,
task2 paired with ID3, unsupported valid CUL1 profiles and poisoned canonical
scratch. Its literal expected outputs exercise the existing evaluator and
deliberately wrong/skipped-result and canary controls. Stream checks cover
empty/one/extra-byte requests, unsupported IDs/modes, truncated/extended payloads,
corruption and stale task selection. A stream checker failure must be retained,
not retried until it happens to pass. The receipt records successfully compiled
wrong implementations rejected by these tests.

## Compilation and first-use measurement protocol

No persistent compiler or artifact cache is implemented by this increment.
Consequently there is no measured cache hit/miss and no warm-cache speedup.
Do not describe repeated compilation as a cache hit.

1. Pin/export sources before timing. Record compiler/host identity, exact command,
   process isolation, source size, governor/load conditions if observed and what
   is not controlled. Do not clear shared OS caches or change host configuration.
2. Start each compiler in a fresh process. Record the **first observed compile**
   separately from repeated processes. Unless cache state was actually controlled,
   call the first sample first-observed, not a proven cold-cache sample.
3. Capture every status/stderr and output digest before comparing times. Save raw
   wall/user/system seconds and peak RSS when available, for example when GNU
   `/usr/bin/time` is installed:

   ```sh
   /usr/bin/time -f '%e,%U,%S,%M' -o build/shader-pilot/sample.csv \
     "$OMC" < build/shader-pilot/export.om \
     > build/shader-pilot/timed-export 2> build/shader-pilot/timed-export.stderr
   ```

   If that tool is absent, Bash's `TIMEFORMAT='%R,%U,%S'` and `time` keyword
   provide wall/user/system seconds, but not peak RSS. Record the missing field
   as NOT_RUN. Do not substitute a different timer without recording it.

4. Repeat the identical command under distinct sample filenames; retain failures
   in the denominator. Report all raw samples. Any percentiles/distribution or
   tail claim needs a stated sample count and enough independent observations;
   this initial host-only protocol exercise makes no tail-latency claim.
5. If using the external referee to execute compiler source, label timing
   **reference-process startup + source parsing + compiler interpretation +
   ELF output**. It is not native compiler timing. Even native compiler process
   wall time includes startup and I/O unless measured separately.
6. Record export process time separately from compilation. GPU artifact loading,
   cache lookup, pipeline creation, first use, GPU execution, and application
   frame-time spikes are all **NOT_RUN**, not zero. Background preparation cost
   and graphics fallback integration remain future work.

The standalone seeded tests are not part of `make test` yet; #15 owns the native
build/test migration. Still run and report `make test` on the exact tree. A
passing offline test or hosted CodeQL run does not replace native or device gates.
No Omega Harness language, compiler, Bridge or qualification contract is changed,
so no upstream guide change is needed for this bounded consumer addition.
