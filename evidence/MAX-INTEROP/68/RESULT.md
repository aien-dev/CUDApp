# CUDApp #68/#69 — native artifact and cubin export, host result

Source: `src/omega_cubin.{c,h}`, `src/omega_native_artifact.{c,h}`,
`tools/artifact_export.c`, `tests/gpu_artifact_export_test.c`,
`tools/artifact_smoke.sh`, `tools/interop/cubin_probe.c`.

## Commands and results
```
make -j8 test-native-artifact
# test_gpu_artifact_export: 42 checks, 0 failed

./tools/artifact_smoke.sh evidence/MAX-INTEROP/68
# ARTIFACT SMOKE PASS   (nvdisasm + cuobjdump accept both cubins)

./build/artifact-export matmul --m 16 --k 16 --n 8 \
  --contract-id 647116fc... --out evidence/MAX-INTEROP/68/matmul-16x16x8
# artifact_id=8dcc18e06f2306a785dd413c73f7289e12df0df995066ccb8cd5ff985a709db6

./build/artifact-export attention --q-heads 8 --kv-heads 1 \
  --contract-id 003696e4... --out evidence/MAX-INTEROP/68/attention-hd128-gqa8
# artifact_id=b7c2c772ca3401160fa18c73d7c653114db20f928aaa993f217657e81df95cb8
```

Both cubins are sm_121 ELF (`e_machine=190`, `e_flags=0x6007902`), carry the
generated text byte-for-byte, bind text/QMD/cbank/SHA-256, contract id,
semantic spec id, shape, bounds, numeric tolerance, driver minimum and
provenance, and contain no device address (zero-VA QMD templates).

`tools/interop/cubin_probe` is built and **NOT_RUN**: the chip is reserved, so
no `SPARK_WINDOW` exists. `cuModuleLoadData`/`cuLaunchKernel` acceptance and
numeric parity on GB10 are unproven.

## Refusals exercised
Deterministic re-export, tampered cubin/text/QMD, wrong ISA, PTX format,
live-VA QMD, stale/too-new driver, malformed contract/artifact ids, bad
ownership/alias/receipt (adapter), overlapping/unaligned parameters, unaligned
text, gpr > 255, too-small cbank, invalid function name.
