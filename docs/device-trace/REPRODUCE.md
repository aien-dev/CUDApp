# Reproduce the bounded host reference

This is preparation for CUDApp #82, not its completion. All stamps in these
fixtures are synthetic. No command below opens a GPU. Do not run a chip test
without the maintainer's separate GPU admission.

## Pins

- CUDApp base: `a520cd30e54c3b0ddd553ee3ccb3f5e673345cd6`.
- Compiler source: `omega-mv@55f9b0af47de70a3d06357985cf99ae3e45c016f`,
  path `seed/omc/omc.om`, SHA256
  `2e35db7d6c2066fb4a20d19b8a9579c44caca9c9937b0594a89422010b673637`.
- Expected seeded compiler: `seed/out/omc`, SHA256
  `734f0a39319fbfa930bcc62cb30dbf06d47d78e126372b74f5baf478e6126b61`.
- External oracle: `omega-mv@190130b71d0a59b93f01c515218a5509b149a145`,
  path `tools/omega_ref/omega_ref.c`, SHA256
  `52e747096c1d978103432460a2978b21a604cc5d3e961bbe4b253fcc7636290e`.

These are explicit historical source pins. They do not silently track later
OM-2 releases or change any compiler/Bridge ABI. The oracle is a pre-existing
C test referee, outside the delivered native implementation. It is not a
new production dependency, a native executor or a device qualification path.

## Compose the three units

From the CUDApp checkout, with an empty disposable `out` directory:

```sh
cat native/blackwell/trace.om tests/device-trace/contract-test.om > out/contract.om
cat native/blackwell/trace.om tests/device-trace/golden-test.om > out/golden.om
cat native/blackwell/vector.om tests/device-trace/disabled-test.om > out/disabled.om
```

The contract test prints `DEVICE_TRACE_CONTRACT_PASS` and exits 0 on success.
The golden unit emits exactly 192 bytes, equal to the hex in `golden.hex`.
The disabled unit emits the existing 512-byte vector kernel, SHA256
`39f3dfdfc529a75a8274a27900acf4e7382b7f280a211e0412c51609f44b62a1`.
This checks the existing emitter only. No enabled GPU trace emitter exists.

## Externally interpreted compiler and semantic oracle (x86_64 host)

Obtain both pinned source files in a separate oracle workspace. Verify the
hashes above before compiling. This command matches the existing referee's
build flags, without the machine-specific `quiet-build` wrapper:

```sh
cc -O2 -g -std=gnu11 -Wall -Wextra -Werror -o omega_ref omega_ref.c -lpthread
./omega_ref check omc.om
./omega_ref run omc.om < omc.om > omc-from-ref.elf 2> self.err
```

Require exit 0 and empty stderr for both calls. The measured self-output
SHA256 was the exact expected seeded compiler digest `734f0a39...6b61` above.
That byte identity does not turn the referee into a trusted native runtime.

For each composed unit:

```sh
./omega_ref check out/contract.om
./omega_ref run out/contract.om > out/contract.stdout 2> out/contract.stderr
./omega_ref run omc.om < out/contract.om > out/contract.elf 2> out/contract.compile.stderr
```

Record each exit separately. Repeat for `golden` and `disabled`. Semantic
execution uses the external interpreter; compilation executes the pinned
Omega compiler source through that interpreter. The emitted executables are
AArch64, not run by these commands. A nonzero compile must never be followed
by execution of its partial output.

Compare `golden.stdout` byte-for-byte with the decoded `golden.hex`, and check
`disabled.stdout` against the full digest above. Do not count file creation or
a matching byte count alone as a pass.

## Mutants (private copies only)

For each one-line replacement below, compose the mutated module with
`contract-test.om`. Require oracle `check` exit 0, oracle `run` exit 1 with
`DEVICE_TRACE_CONTRACT_FAIL` on stderr, and interpreted compiler exit 0.
Never mutate the committed implementation for a test run.

| Original | Replacement |
| --- | --- |
| `if or or gt cap 64 ge cta` | `if or or gt cap 65 ge cta` |
| `if ge n get gtstate 3` | `if gt n get gtstate 3` |
| `mul get gtstate 5 32` | `mul get gtstate 5 31` |
| `if ne gen get gtstate 0` | `if eq gen get gtstate 0` |
| `gtput32 60 1` | `gtput32 60 0` |
| `gtput32 add off 8 uint n` | `gtput32 add off 8 uint add n 1` |
| `for j ulong 16` | `for j ulong 0` |
| `for i ulong sub bytes off` | `for i ulong 0` |
| First `put gtstate 6 0` in `gtinit` | `put gtstate 6 1` |
| `if and ne gtget32 20 0 ne n cap` | `if and ne gtget32 20 0 gt n cap` |
| `if and eq gtget32 60 1 ne gtget32 20 4294967295` | `if and eq gtget32 60 2 ne gtget32 20 4294967295` |
| `if ne gtget64 add off 16 gtget64 32` | `if ne gtget64 add off 16 gtget64 add off 16` |

Without the module, `contract-test.om` is refused by both the oracle checker
and externally interpreted compiler with error 3 at the first `gtinit` call.
This absence check was run after implementation, not a pre-implementation
native red/green run. Native TDD execution remains unestablished.

## Native execution, still NOT_RUN

On an authorized Linux AArch64 host, verify the exact compiler digest first.
Compile each composed unit with that compiler, then execute only successful
outputs. Repeat the exact-byte and mutant comparisons above. Record unit and
ELF hashes, status, stdout and stderr. This is a host test, still not GB10.
The full CUDApp `make test` suite must also run in a complete checkout.

The local connector-materialized partial snapshot could not run `make test`:
make exited 2 because `tests/gpu_matmul_api_test.c` was absent. No target or
threshold was skipped or weakened. The native seed/compiler gates, native
ELF execution, full repository suite and all GPU gates are NOT_RUN.

## Independent arithmetic check

The golden was independently encoded with Wolfram Language using
`Reverse[IntegerDigits[value,256,width]]` for each field in contract order.
An independent `struct.pack` check matched those 192 bytes. The resulting
byte SHA256 is `88a3de3f88a0d59852704a97d2c9f859f87b57ee9efd0f2509aa95ace1adcab6`.
Wolfram also checked the finite store-extent property:

```text
And@@Flatten[Table[64+64*n+63 < 64+64*c,{c,1,64},{n,0,c-1}]] -> True
```

This is MODELED arithmetic, not a measured device memory-safety result.

## Current compiler compatibility, separate receipt

A later check used `omega-mv@7505b3a3e3cf5ee46933e3a89a021b4289037983`,
without changing the historical pins or receipt above. Its
`seed/omc/omc.om` SHA256 is
`c792b401193d8477d38965d6ddb0fd0cb8fd5ac24fcb92fae9af52037a765f1c`.
Executed through the same unchanged external oracle, its self-output matches
`499236398aabc652e2dae27fe4bdc4c5db3d3726314356e0dbef23e3345803ec`,
the pinned commit's `seed/out/omc.sha256`.

All three unmodified units and all twelve semantic mutants compile with
exit 0 and empty compiler stderr under this newer compiler source. The three
positive ELF files are byte-identical to the earlier compiler's outputs.
This is compatibility of these units only, not a full compiler qualification
or execution of those AArch64 files. See the additive
`evidence/device-trace/current-compiler-7505b3a/receipt.json` receipt.
