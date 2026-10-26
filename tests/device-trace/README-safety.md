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
  and the branch predicate and complete relative displacement. It does not use
  `expected.sass` or the emitter's address helpers as its oracle. This limited
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
status in their source on failure. These tests are not yet integrated into
`make test` or the historical disassembler runner.

## Historical golden is not current

`expected.sass` and `evidence/device-trace/emitter-host-20261026/receipt.json`
describe the original PR84 head `60d6c57fc6821accce0027d65dd239355beeb543`.
They contain the erroneous multiply-based address formation. They are not
passing evidence for this repair, and the old `run-emitter-checks.sh` golden
comparison does not pass for the repaired output.

The repair was decoded separately with an external host-only disassembler.
The tool, output listing, and generated GPU words are not added to the repository.
`AGENTS.md` forbids NVIDIA-tool listings in the repository, whereas the inherited
runner offers to overwrite one. This conflict remains for the owner to resolve;
do not bypass the comparison or regenerate the listing here to claim a pass.

## Remaining limits

Native AArch64 execution and the full native suite, GB10 execution, QMD/launch
wiring, paired off/on overhead, clock qualification, and chip repeatability
remain unverified by these host models. Runtime overflow/drop accounting,
capacity zero/one capture, and begin/end ranges are not implemented by this
fixed two-mark slice. No default-on mode, driver changes, or hardware authority
are introduced.
