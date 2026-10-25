# Native reconstruction

Use Linux AArch64 and a fresh output directory. These are disclosed development
commands using existing host tools, not a new first-party runtime or installer.
No Python is needed. The native programs use only stdin/stdout and process exit.

Reconstruct the compiler from immutable `aien-dev/omega-bridge` revision
`0fc7d8662cb799e7c206b17c18ea4c3ec2b8fea9`, following its existing
`tools/bootstrap.sh --offline`. Its seed input is Omega revision
`29b12760f4c962606411d6bf25032aedef063301`. Require compiler SHA-256:

```
191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9
```

Set `OMC` to that reconstructed executable and `OUT` to an existing private
output directory. Export the exact source revision from the receipt with
`git archive`, then assemble these ordered translation units in the export:

| Program | Ordered `.om` inputs |
| --- | --- |
| policy-test | native/speculation/policy; tests/speculation/support, policy-test |
| cost-test | native/speculation/policy, costs; tests/speculation/support, cost-test |
| report-test | native/speculation/policy, report; tests/speculation/support, report-test |
| oracle-test | native/speculation/policy; tests/speculation/support, oracle-test |
| replay | native/speculation/policy, costs, report, replay |
| packet-fixture | native/speculation/policy; tests/speculation/support, packet-fixture |
| reply-test | native/speculation/policy; tests/speculation/support, reply-test |

For example:

```sh
cat native/speculation/policy.om tests/speculation/support.om tests/speculation/policy-test.om > "$OUT/policy-test.om"
quiet-build timeout 30 "$OMC" < "$OUT/policy-test.om" > "$OUT/policy-test.elf" 2> "$OUT/policy-test.compile.log"
# Require compiler exit 0 before execution.
test -s "$OUT/policy-test.elf"
readelf -h "$OUT/policy-test.elf"
chmod 700 "$OUT/policy-test.elf"
quiet-build timeout 10 "$OUT/policy-test.elf"
sha256sum "$OUT/policy-test.elf"
```

Apply the same compile/check/run sequence to each unit. The native fixture
emitter reads one selector byte: 1 policy, 2 cost collector, 3 paired report.
It emits explicitly synthetic records. For the policy replay:

```sh
printf '\001' | "$OUT/packet-fixture.elf" > "$OUT/policy.bin"
"$OUT/replay.elf" < "$OUT/policy.bin" > "$OUT/policy.reply"
"$OUT/reply-test.elf" < "$OUT/policy.reply"
```

Selectors `\002` and `\003` exercise the other modes. Preserve input, output,
stderr, exit status and executable hashes. An empty or truncated message must
return 65 with no stdout. Synthetic input must retain evidence class 1 in the
reply. It cannot satisfy a request requiring device evidence.

The receipt records compiled behavioral controls. Each applies one change to a
private copy of product source, compiles successfully, and runs the unchanged
tests, which must return 1. A compile error, crash or timeout does not count as
a killed behavioral control.

Existing regression command, separately from new native tests:

```sh
quiet-build make -j2 OUT_DIR=build/regression test
```

These 33 legacy host targets are regression evidence. They are not new seeded
tests, full-model correctness or GPU qualification.
