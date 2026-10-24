# CUDApp #68/#69 — native artifact and cubin export, host result

Source: `src/omega_cubin.{c,h}`, `src/omega_native_artifact.{c,h}`,
`tools/artifact_export.c`, `tests/gpu_artifact_export_test.c`,
`tools/artifact_smoke.sh`, `tools/interop/cubin_probe.c`.

## Commands and results
```
make -j8 test-native-artifact
# test_gpu_artifact_export: 47 checks, 0 failed

./tools/artifact_smoke.sh evidence/MAX-INTEROP/68
# ARTIFACT SMOKE PASS   (nvdisasm + cuobjdump accept both cubins)

./build/artifact-export matmul --m 16 --k 16 --n 8 \
  --contract-id 647116fc... --out evidence/MAX-INTEROP/68/matmul-16x16x8
# artifact_id=7c5ab77748c1cd29a50b9cb91efcce6d8200eb30646ce88142f31ba4d7d5544c

./build/artifact-export attention --q-heads 8 --kv-heads 1 \
  --contract-id 003696e4... --out evidence/MAX-INTEROP/68/attention-hd128-gqa8
# artifact_id=48eac870570fd978aa011973bfc3407fedc6de2f1f1fbe1fb943cc96c9f95c22
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
