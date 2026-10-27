# Local host-only repair evidence

Read `receipt.json` for exact source, compiler, oracle, output hashes, statuses,
negative controls, and unverified boundaries. Its source commit is a local
isolated commit; file hashes identify the tested bytes independently of later
publication metadata. No native AArch64 or GPU output was executed.

The native commands for an appropriate AArch64 host are in
`tests/device-trace/README-safety.md`. The existing disassembler runner is not
green: its inherited golden describes the old pointer bug. No gate was bypassed
and no NVIDIA-tool listing was committed to conceal that mismatch.

## External oracle setup

Obtain the existing `tools/omega_ref/omega_ref.c` from Omega commit
`190130b71d0a59b93f01c515218a5509b149a145` in a separate test workspace. Check
its SHA256 against the receipt before compiling it:

```sh
cc -O2 -g -std=gnu11 -Wall -Wextra -Werror -o omega_ref omega_ref.c -lpthread
```

Obtain the compiler source `seed/omc/omc.om` at the separate compiler revision
in the receipt. Confirm both source and independently recorded seeded binary
digests. The measured self-output command was:

```sh
./omega_ref check omc.om
./omega_ref run omc.om < omc.om > compiler-self.elf 2> compiler-self.stderr
sha256sum compiler-self.elf
```

It exited zero with empty stderr and matched the independently recorded
`499236398aabc652e2dae27fe4bdc4c5db3d3726314356e0dbef23e3345803ec` digest.
This is compiler-source execution by an external referee, not native compiler
execution or seed-chain qualification.

## Original versus projected units

Compose the four original units as documented in the test README. The frozen
oracle predates `const` and refuses the unmodified emitter with error 16.
For semantic tests only, a separate copy was projected mechanically:

1. Collect `const NAME ulong VALUE` lines. The emitter contains only decimal
   unsigned-long literal constants and has no shadowed constant names.
2. Remove those declaration lines.
3. In non-`note` lines, replace whole `NAME` tokens with `ulong VALUE`. Preserve
   all other text. Do not change the original units.

For each original `unit.om` and its `unit.projected.om`:

```sh
./omega_ref check unit.projected.om
./omega_ref run unit.projected.om > unit.stdout 2> unit.stderr
./omega_ref run omc.om < unit.om > unit.elf 2> unit.compile.stderr
./omega_ref run omc.om < unit.projected.om > projected.elf 2> projected.compile.stderr
cmp unit.elf projected.elf
```

All final checks/runs/compiles exited zero with empty stderr, and every `cmp`
matched. The four original and projected ELF hashes are recorded separately.
They are AArch64 outputs and were **not executed**. Semantic execution claims
refer only to the explicitly projected unit under the external oracle.

## Red tests and mutation controls

Before production changes, the original emitter test passed while the new
address, guard, and whole-kernel tests failed with statuses 3, 1, and 1. After
test refinements, repeat the same composition against the exact original PR84
source from `git show 60d6c57fc6821accce0027d65dd239355beeb543:native/blackwell/trace-emit.om`.
The compiler accepted those final red units, and they still failed 3, 1, and 1.

Apply each mutation in a disposable source copy, never in the working tree.
The receipt lists the expected failing test and its measured status:

- Restore multiply-base-low plus high-copy address instructions.
- Restore the original `on == 1 && checks == 0` gate.
- Accept any nonzero `checks` instead of requiring exactly 1.
- Skip the body refusal entirely.
- Apply the body refusal only to `on == 1`.
- Compare predicate bits as part of EXIT instead of masking them.
- Set bit 16 or the sign bit of a store's signed 24-bit byte offset.
- Set the branch high displacement word, or its upper encoded bits.

All ten mutated units were accepted by the compiler and reference checker,
but failed their semantic tests. Independent review repeated the four final
unit runs and all ten mutation runs. Two initial test gaps and compiler
register-pressure corrections are preserved in the receipt, not hidden.

## External decode and full-suite boundary

A dump unit calls `gtewrap 1 2` with an empty body and writes `gbout` to stdout.
Its original and projected forms compiled to identical ELF bytes. The projected
oracle run produced the GPU words, decoded offline with:

```sh
nvdisasm --binary SM121 words.bin > decoded.sass
diff -u tests/device-trace/expected.sass decoded.sass
```

The disassembler exited zero. The diff exited one: only the two address
instructions per marker changed, as intended. The downloadable official tool
archive and all local output hashes are in the receipt. Those binaries and
listings stay outside the repository.

`make test` was attempted on both the base and repaired source. Both failed
building `build/omega_blackwell_engine.o`, because the x86 assembler rejects
the existing AArch64 `ds` instructions. No host suite success is claimed.
The instruction scheduler, GB10 memory effects, launch geometry, visibility,
overhead, and runtime overflow still require their separate authorized native
and hardware work. No hardware authorization was requested or inferred.
