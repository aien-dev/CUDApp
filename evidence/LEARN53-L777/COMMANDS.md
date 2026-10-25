# Reproducible commands (source11ad6a0)

Actual qualification host: Spark, Linux AArch64. Development commands below use existing host tools; they are documentation, not a newly shipped build/qualification wrapper. Do not run on main/shared output. No GPU invocation.

Source-only toolchain export/reconstruction actually run:

```sh
git -C /home/drakestapleton/workspace/omega-bridge archive 0fc7d8662cb799e7c206b17c18ea4c3ec2b8fea9 | tar -x -C /home/drakestapleton/handoffs/cuda-learn-L777/toolchain
cd /home/drakestapleton/handoffs/cuda-learn-L777/toolchain
quiet-build timeout 120 tools/bootstrap.sh --offline
sha256sum build/bootstrap/omc
```

Bootstrap exit0 and compiler191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9. No executable compiler fallback. Set `OMC` to the resulting compiler only after that comparison. Caller creates a fresh private `$OUT` directory; this is not a safety-qualified installer.

Source export actually run:

```sh
git -C /home/drakestapleton/workspace/hermes-desk/cudapp-learn-L777 archive 11ad6a0a7783f5d221de79111d6abba4ee1ae52f native/learn53 tests/learn53 docs/learn53 | tar -x -C /home/drakestapleton/handoffs/cuda-learn-L777/replay-11ad6a0
```

In that fresh source export, construct the following units:

```sh
cat native/learn53/task.om tests/learn53/fixtures.om tests/learn53/unit.om > "$OUT/unit.om"
cat native/learn53/task.om native/learn53/wire.om tests/learn53/fixtures.om tests/learn53/stream-fixtures.om tests/learn53/stream-unit.om > "$OUT/stream-unit.om"
cat native/learn53/task.om native/learn53/wire.om native/learn53/stream.om > "$OUT/stream.om"
cat native/learn53/task.om native/learn53/wire.om tests/learn53/fixtures.om tests/learn53/stream-fixtures.om tests/learn53/stream-input.om > "$OUT/stream-input.om"
cat tests/learn53/stream-check.om > "$OUT/stream-check.om"
```

For each named unit, actual command pattern (`name` was expanded individually, not a delivered script):

```sh
quiet-build timeout 30 "$OMC" < "$OUT/name.om" > "$OUT/name.elf" 2> "$OUT/name-compile.log"
# Require exit0. An empty or failed compile is NOT an executable or a killed mutant.
test -s "$OUT/name.elf"
readelf -h "$OUT/name.elf"
chmod 700 "$OUT/name.elf"
sha256sum "$OUT/name.elf"
```

All5 compiled and matched the worktree-built binaries byte-for-byte (`cmp`). Execute `timeout 10 "$OUT/unit.elf"` and `timeout 10 "$OUT/stream-unit.elf"`; actual rc0,77 and17 PASS lines respectively. Both test mains return fixed1 on any assertion failure, not a potentially truncated failure index.

For real CLI data tests, emitter input modes are single bytes0..5. Commands actually run for valid0 and wrong1 (analogous2..5 retained in cli status files):

```sh
printf '\000' | "$OUT/stream-input.elf" > "$OUT/frame0.bin"
timeout 10 "$OUT/stream.elf" < "$OUT/frame0.bin" > "$OUT/cli-0.out"
"$OUT/stream-check.elf" < "$OUT/cli-0.out"
printf '\001' | "$OUT/stream-input.elf" > "$OUT/frame1.bin"
timeout 10 "$OUT/stream.elf" < "$OUT/frame1.bin" > "$OUT/cli-1.out"
timeout 10 "$OUT/stream.elf" < /dev/null
```

Statuses0/1/65/65/65/77 for modes0..5; empty stdin65. Success diagnostic exactly12 bytes; failure stdout empty. `stream-check` rc0 on real valid output and rc1 on the old empty-output stub given the same valid frame.

Each `mutant-*.diff` was applied to a private copy of task.om, concatenated with the **unchanged** final fixture/unit files and compiled through the same compiler/ELF checks. Then `timeout 10 mutant-N.elf`; actual rc1 and nonzero FAIL assertions, not compiler failure or timeout. Original binaries remain unchanged. Compiler logs, ELF headers, patches, binary identities and complete77-line outputs are retained.

Runtime dependency observation actually run from fresh export:

```sh
strace -f -e trace=execve,openat,read,write,ioctl -o out/stream-syscalls.log timeout 10 out/stream.elf < out/public-frame.bin > out/stream.out
out/stream-check.elf < out/stream.out
readelf -l -d out/stream.elf
```

Both execution/check exit0. The timeout parent uses host libraries; after exec of the evaluated native ELF, observed calls are stdin reads and the fixed stdout write. No dynamic section. These trace/log files are new observations, not copied historical receipts from the toolchain export.

Unchanged legacy regression command, worktree rooted at public main80f3a24:

```sh
quiet-build timeout 600 make -j2 OUT_DIR=build/baseline test
```

Actual exit0 and33 host targets. No legacy source, Makefile, gate or dependency was modified. This is regression evidence only; native migration, seeded host automation and integrated consuming-graph gates remain open.
