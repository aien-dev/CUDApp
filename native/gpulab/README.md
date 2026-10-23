# Seed-native AMD and Apple GPU instruction lab

This is the first AMD-then-Apple implementation slice, written personally in
seeded Omega. **Every first-party generator, interpreter, fixture writer,
numerical reference, byte comparator and test program is `.om`.**

The Makefile and command examples are host build/process glue. The seed compiler
produces static AArch64 Linux executables using Linux read/write/exit syscalls.
No C, Rust, Python, Metal, Vulkan, HIP, CUDA or third-party inference runtime is
built or linked into this lab. LLVM is used only as an independent AMD assembler
oracle. Wolfram supplies independent calculations and fixture values; it is not
a runtime dependency.

## What executes

| Target | Code generated | Virtual execution |
|---|---|---|
| AMD `gfx1100`, RDNA3 | Real instruction words for global 32-bit loads/stores, wait, move, integer multiply/add and program end | Omega decodes the emitted bytes and executes the supported instructions against bounded virtual registers/memory |
| Apple G13/G14 AGX | Real-format load/store, wait, immediate, integer multiply/add and stop bytes | Separate Omega instruction decoder and executor using the same bounded virtual-memory contract |

The workload is `y[row] = sum(W[row,k] * x[k])`, with depth 1..64 and rows 1..32.
Signed-byte values are **sign-extended into 32-bit storage**; accumulation is
modulo 2^32. This is not packed-int8/tensor-core acceleration. The signed-int8
bounded workload fits signed 32-bit accumulation, independently checked in
Wolfram. Tests also compare every possible pair of signed-byte operands.

The generator specializes depth into a straight-line program. The simulator
executes each active lane through the emitted bytes, including memory operations
and dependency checks. It does not call the scalar reference to produce outputs.

## Exact lab ABI

Virtual memory is 4096 32-bit slots:
- weights: slots `0 .. rows*depth-1`, row-major;
- input: slots `2048 .. 2048+depth-1`;
- output: slots `3072 .. 3072+rows-1`;
- all other slots are outside the kernel's permitted memory regions.

AMD register entry values for lane `r`: `v[0:1]` holds the weight-row byte address,
`v[2:3]` holds byte address 8192, and `v[4:5]` holds the output byte address.
`v6` is the accumulator, `v7` a product, `v8/v9` loaded operands. The simulator
sets the high halves to zero. Only nonnegative immediate offsets in the supported
range are admitted.

AGX entry values: `r0 = r*depth`, `r1 = 0`, `r2 = r`, with uniform buffer bases
`du0/du2/du4` mapped to weights/input/output. Register names here are 32-bit word
registers; the AGX encoding itself numbers 16-bit halves. `r6` is the accumulator,
`r7` a product, `r8/r9` loaded operands. The emitter increments `r0/r1` after each
term. Only the documented plain-register ALU and uniform-base memory forms are
executed by this simulator; other legitimate AGX forms are explicitly unsupported.

**This entry-state setup is a lab ABI, not an implemented hardware kernel-launch
ABI.** No GPU queue packets, driver, firmware, hardware dispatch, cache hierarchy,
SIMT divergence, atomics, full ISA, cycle timing, or device reset is modeled.
Rows model active lanes only, not a complete wave scheduler. The stricter wait
rules are safety rules for this subset, not a claim that every valid hardware
program requires identical waits.

## Build and run

Supported host for the native executable: AArch64 Linux. The Mac is used for
editing/transport, not claimed to execute these Linux ELF files natively.

Use the already source-built, qualified compiler:

- compiler source origin: `omega-mv@29b12760f4c962606411d6bf25032aedef063301`;
- source bootstrap package: `omega-bridge@d2010ab927b608931800c20d73a4c2b25f00a9fa`;
- required compiler SHA256:
  `191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`.

The compiler is an explicit prerequisite, not silently downloaded or replaced.
This task reused the qualified source-built compiler; it did not rerun the seed's
whole bootstrap campaign. The pinned source, fixed-point history and compiler
fingerprint are separate from this lab's clean-source replay.

From the repository root:

```sh
make -f native/gpulab/Makefile test OMEGA_NATIVE_COMPILER=/absolute/path/to/omc
```

Host tools: make, Bash, cp/tee/chmod/mv, sha256sum/cut/stat, LLVM 18 `llvm-mc-18`
and `llvm-objcopy-18`. These assemble source files, start processes and supply the
external oracle. No first-party algorithm is implemented in those tools.

This is an optional standalone native suite. It does not alter the repository's
legacy C suite, production build, installed `omega` command or serving process.
The legacy C suite was **not run**, consistent with the user's native-only scope.
Do not concurrently run this Makefile in the same build directory. The single-user
build guards reject symlinks and multiply-linked binary/source outputs, but do not
claim hostile-race filesystem isolation. Run fresh replays in separate exports.

## Standalone code generation

`build/native-gpulab/emit-main` consumes exactly eight bytes on stdin followed by
EOF: little-endian u32 target (`1=gfx1100`, `2=G13/G14`) and little-endian u32 depth.
It produces raw instruction bytes on stdout. Unsupported target/depth or malformed
length returns 65 with no stdout. Transport errors return 74 or an OS signal;
a failed output write may leave partial bytes, so consumers must stage output and
publish only on exit 0. Internal emitter helpers are not public arbitrary-field
APIs: only the validated target/depth entry point is exposed to untrusted input.

Example using the **Omega-written request fixture** to request depth four:

```sh
printf 1 | build/native-gpulab/request-fixture | build/native-gpulab/emit-main > amd.isa
printf 2 | build/native-gpulab/request-fixture | build/native-gpulab/emit-main > apple.isa
build/native-gpulab/test-wire < amd.isa
build/native-gpulab/test-wire < apple.isa
```

Use `set -o pipefail` when consuming these pipelines. `test-wire` is a deliberately
fixed depth-four integration fixture, not a general inference command. It receives
the standalone emitter's actual stdout, runs it in the virtual ISA and checks the
Wolfram reference outputs 20, 30 and 40.

**Raw `.isa` files are lab data, not installable binaries or runnable GPU jobs.**
They have no hardware launcher or production attachment.

## Tests and receipts

See `../../evidence/NATIVE-GPULAB-L617/`:
- compilable refusal baselines and raw RED logs;
- AMD 8,224 checks and AGX 8,218 checks;
- cross-target 262,928 checks, including every signed-byte operand pair, Wolfram
  dot/extreme fixtures, every truncated prefix of depth-four shaders, extra bytes,
  invalid sizes/targets and no-publication-on-fault checks;
- complete AMD depth-64 emitted shader byte equality with independent LLVM output;
- AGX fixed byte fixtures from the prior Apple-produced corpus, including separate
  immediate, add, multiply, load and store forms;
- actual stdin/stdout tests and same-length corrupted machine-code controls;
- clean committed-source replay, build fingerprints and observed exit statuses.

For numerical gates, the final two log lines are total checks then failed checks;
preceding numeric lines identify failed assertions. A nonzero exit is failure.
Mutation tests edit actual emitted instructions, not a mock verdict. Wrong-but-legal
arithmetic is caught by output comparison; unknown/reserved forms, missing waits,
bad addresses, truncations and missing completion are refused. Results are staged
and only copied to public virtual output memory after every lane succeeds.

Compile/setup failures are retained separately; they are not counted as behavioral
RED or as caught mutants. No independent reviewer was dispatched, per the user's
personal-work instruction. Review here is author review, not independent acceptance.

## What this proves, and what it does not

This proves a bounded native code-generation and virtual-execution path with the
listed independent checks. AMD instruction bytes match LLVM for the full emitted
maximum-depth program. Apple field encoding has pinned prior-art and recorded
Apple-byte checks, but the whole newly generated shader has not executed on an
Apple GPU. A simulator and emitter can share a mistaken interpretation; oracle
coverage narrows that risk, not eliminates it.

**HARDWARE NOT_RUN. SPEED NOT_MEASURED. FULL-MODEL INFERENCE NOT_IMPLEMENTED.**
No claims about other AMD generations, CDNA matrix instructions, Apple M3 and
later generations, Linux/Windows GPU driver portability, or beating another
engine follow from these tests. Follow-on work is real launch-ABI integration,
additional inference operations and formats, then exact-model hardware
qualification and end-to-end measurement. Those are not hidden behind a simulator
PASS. Existing L442 covers non-Spark hardware qualification; L444's OS bridges
are separate, not part of this native-only implementation.
