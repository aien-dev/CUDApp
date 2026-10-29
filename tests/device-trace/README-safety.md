# Emitter safety regressions

These are host tests for the low-level emitter, not hardware qualification.
They do not establish that a complete kernel can be instrumented safely.

## Supported input

Enabled input is a prepared, straight-line body fragment. The caller must
initialise and preserve `R0 = TID.X`, launch exactly one CTA, and account for
the markers' `P0`, `R2`, `R4:R13`, and `UR4:UR5` clobbers. Both insertion points
must be reached by fallthrough. Branches, calls, returns, and early exits are
unsupported. The wrapper rejects the repository's `EXIT` encoding, regardless
of predicate, with error 5 before writing output. That scan is not a general
control-flow verifier. In particular, do not pass the complete `gbvec` kernel.

All nonzero enable values require `gtchecks == 1`. That is an explicit caller
precondition, not proof of launch geometry, memory visibility, scheduling, or
GPU safety. Disabled input remains byte-identical after the existing capacity
and instruction-alignment checks. Disabled capacity zero is still refused.

## Independent semantic tests

- `address-test.om` decodes the address pair and models each store's effective
  address inside a guarded capture extent. It checks ordinary and low-half-carry
  cases, untouched header/reserved bytes/canaries, signed 24-bit store offsets,
  the branch predicate and complete relative displacement, the GLOBALTIMERLO
  source and R2 destination, and each record field's store-source register.
  Its in-test timer and register mutations must be refused after each field is
  changed, and the original bytes are restored. It does not use
  the nvdisasm decode or the emitter's address helpers as its oracle. This limited
  arithmetic model does not validate GPU instruction scheduling or scoreboards.
- `guard-test.om` exercises enable/check values 0, 1, 2 and maximum unsigned
  64-bit value. Refusal must preserve every output byte and its prior length.
- `body-test.om` uses the actual `gbvec` kernel as a negative control, checks
  disabled byte identity, and refuses every predicate variant of `EXIT` for
  enable values 1, 2 and maximum unsigned 64-bit value.

On Linux AArch64, using an independently verified seeded compiler:

```sh
test "$(uname -m)" = aarch64 || exit 69
: "${OMEGA_COMPILER:?set the verified seeded compiler path}"
sha256sum "$OMEGA_COMPILER"
mkdir -p build/emitter-safety-native
for name in emitter address guard body; do
  unit=build/emitter-safety-native/$name
  if [ "$name" = body ]; then
    cat native/blackwell/vector.om native/blackwell/trace-emit.om \
      tests/device-trace/body-test.om > "$unit.om"
  else
    cat native/blackwell/trace-emit.om \
      "tests/device-trace/$name-test.om" > "$unit.om"
  fi
  "$OMEGA_COMPILER" < "$unit.om" > "$unit.elf" 2> "$unit.compile.stderr" || exit
  test ! -s "$unit.compile.stderr" || exit 1
  chmod +x "$unit.elf"
  "$unit.elf" || exit
done
```

All four units must exit zero. The original emitter test prints its individual
checks. The three new tests are silent on success and return a named assertion
status in their source on failure. `run-emitter-checks.sh` now compiles and runs
all four units before its external decode check. A failed compile, nonempty
compiler stderr or failing unit stops the runner before any later stage.
These units are still not integrated into `make test`.

The runner's shell control flow can be checked without a compiler or GPU:

```sh
sh tests/device-trace/run-runner-controls.sh
```

That check uses synthetic compiler programs and decode rows. It checks unit
composition and failure propagation only; it is not native Omega execution,
an instruction decoder or evidence that the emitter's instruction words work.

## Disassembler check (replaces the historical golden)

The historical `expected.sass` described the original PR84 head
`60d6c57fc6821accce0027d65dd239355beeb543`, including the erroneous
multiply-based address formation, and was itself an NVIDIA-tool listing.
PR #93 removed it. `run-emitter-checks.sh`
now keeps the nvdisasm listing in its temporary directory and checks it against
hand-written expectations: the opcode counts of the two 20-instruction marker
blocks and the address-formation lines (`MOV R6, 0x40` / `0x80`, then
`IMAD.WIDE.U32 R8, R6, 0x1, R4`). The emitter on main before this repair fails
that check (`IMAD.SHL R8, R4, 0x40, RZ` at offset 0x70). The historical receipt
`evidence/device-trace/emitter-host-20261026/receipt.json` stays unchanged as
a record of what ran on PR84's head.

## Remaining limits

Native AArch64 execution and the full native suite, GB10 execution, QMD/launch
wiring, paired off/on overhead, clock qualification, and chip repeatability
remain unverified by these host models. Runtime overflow/drop accounting,
capacity zero/one capture, and begin/end ranges are not implemented by this
fixed two-mark slice. No default-on mode, driver changes, or hardware authority
are introduced.
