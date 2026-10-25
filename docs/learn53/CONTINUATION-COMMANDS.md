# Continuation reconstruction

Run on Linux AArch64. Source revision and resulting hashes are named in
`evidence/LEARN53-L777-CONTINUATION/RECEIPT.md`. Use a fresh private output
directory; commands are disclosed existing development orchestration, not a
shipped native runner, installer, performance harness or complete build cutover.

Reuse the previously reconstructed source-only toolchain at bridge
`0fc7d8662cb799e7c206b17c18ea4c3ec2b8fea9`. Check `sha256sum` of
`build/bootstrap/omc` equals
`191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`.

Export exact committed CUDApp sources with `git archive "$REV"`; do not use a
mutable worktree. Export upstream source with:

```sh
git -C "$OMEGA" show 84d03c5523f39ca1045d1c313c5f62597779462b:src/learning/benchmark/core.om > "$OUT/omega-core.om"
sha256sum "$OUT/omega-core.om"
```

Require `a5a2e74c815f9ca6a90dbdfbfc7ecc343f0362c824064f2058e9d1d7c119c40c`.
The corresponding upstream `tests/learning/benchmark/core-test.om` must build
and run with this source first; expected `BENCHMARK_CORE_PASS` and185.

Ordered concatenation lists (no hidden source inputs):

- catalog: task.om, catalog.om, tests/learn53/fixtures.om, catalog-fixtures.om, catalog-unit.om.
- omega: upstream omega-core.om, task.om, catalog.om, omega-adapter.om, tests/learn53/fixtures.om, catalog-fixtures.om, omega-unit.om.
- catalog-export: task.om, catalog.om, tests/learn53/catalog-export.om.
- catalog-check: task.om, catalog.om, tests/learn53/catalog-check.om.
- original-core/frame/stream: unchanged commands in evidence/LEARN53-L777/COMMANDS.md.

Product basenames above resolve under native/learn53; test basenames under
tests/learn53. Assemble each ordered list into a private translation unit, then:

```sh
quiet-build timeout 30 "$OMC" < "$OUT/unit.om" > "$OUT/unit.elf" 2> "$OUT/compile.stderr"
# Require compile exit0 before any execution; empty output is not an ELF.
test -s "$OUT/unit.elf"
readelf -h "$OUT/unit.elf"
chmod 700 "$OUT/unit.elf"
timeout 10 "$OUT/unit.elf"
sha256sum "$OUT/unit.elf"
```

Catalog expected83 PASS lines; Omega adapter160 PASS lines; original core77
and frame17. Every test main returns fixed1 for any failed assertion.
For each byte ID1..8, route it to catalog-export, save stdout as task-N.cul;
prepend that same selector byte and route the result to catalog-check (exit0).
Empty input or two selector bytes must yield exporter65; byte0 or9 yields77.
No failure may emit a partial task. Save all stdout/stderr/status.

The three compiled negative-control copies change only omega-adapter.om:
ignore-outcome passes0 to lattempt; omit-task calls lbegin7; ignore-length
passes expected length as observed length. Unchanged160-check suite must fail
with nonzero status after successful compilation. Original sources remain
unchanged. Raw source copies and outcomes are preserved in evidence.

Host command timings/RSS, where recorded, are coarse process observations only;
LC unit fixture values are NOT those measurements. Full lifecycle cost and
energy remain UNKNOWN. No GPU launch, candidate sandbox or model is invoked.
