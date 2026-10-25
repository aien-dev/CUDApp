# GPU interop probes (not part of the library)

`cubin_probe` loads an exported artifact's cubin through the vendor driver and
runs it. It links `libcuda` on purpose: this is the interop boundary, and a
successful load is evidence that the cubin container and parameter ABI are
accepted. The library itself still opens no CUDA library.

## Run only in an authorized window

```sh
spark-window open maxinterop-cubin --ledger L679 --gb10 shared --quiet none --lease 10
spark-window run ./build/cubin_probe evidence/MAX-INTEROP/68/matmul-16x16x8.cubin omega_matmul_bf16 16 16 8
spark-window close
```

Without `SPARK_WINDOW` the probe exits 3. It never resets the GPU, changes the
driver, or touches the serving process. Expected PASS output:

```
cubin=... fn=omega_matmul_bf16 M=16 K=16 N=8 worst_abs=... bad=0 window=...
```

Status on 2026-10-23: **NOT_RUN**. The chip is reserved by the coordinator;
`spark-window open` refuses. No result is claimed.
