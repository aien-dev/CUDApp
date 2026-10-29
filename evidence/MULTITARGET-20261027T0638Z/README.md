# MULTITARGET-20261027T0638Z: portable backend host gate

Status: host EXERCISED on the Spark's CPU (Linux 7.0.0-1019-nvidia aarch64). No GPU ran any of this.
Every GPU target other than the GB10 is hardware NOT_RUN. The GB10 was not used (no gb10 lock, no
`/dev/nvidia*` opened; `no-gpu-open.txt`).

Source: CUDApp `0ec44766b54e09a4fa3f53943e52a50ec714f5df`, fresh clone of `backend/portable-interface`.
Tools and hashes: `tools.txt`. Unit, input and output digests: `outputs.sha256`.

## Commands

```
R=<dir with extracted spirv-tools, lld-18, libllvmspirvlib18.1, opencl-c-headers, libvulkan-dev>
LD_LIBRARY_PATH=$R/usr/lib/aarch64-linux-gnu:$R/usr/lib/llvm-18/lib \
SPIRV_VAL=$R/usr/bin/spirv-val LD_LLD=$R/usr/lib/llvm-18/bin/ld.lld HELLO_GPU=<clone>/build/hello_gpu \
CUDAPP_MIGRATION_PENDING_SPIRV_HOST=1 POCL_LIBRARY=<abs>/libpocl.so.2.16.0 MT_CFLAGS=-I$R/usr/include \
MT_OUT=<dir> nice -n 19 taskset -c 10-14 sh tests/multitarget/run.sh <abs>/omc      -> run.log, exit 0
make doctor OMC=<abs>/omc                                                         -> make-doctor.txt, exit 0
```

## Result (run.log)

`MULTITARGET PASS pass=63 fail=0 not_run=0`.

| Kernel | Host reference (conform.om) sha256 | PoCL 7.2 CPU (SPIRV_CL) | lavapipe, Mesa 25.2.8 CPU (SPIRV_VK) | hello_gpu --sim |
|---|---|---|---|---|
| vadd, 1048576 FP32 | `72d0c1b2928779e83518f0b4c7d101d933cf89f210cd4627e138c33c9ce9b6a6` | bit-exact | bit-exact | same digest |
| fsum16, 65536 rows × 16 FP32 | `6a6de93787e333d506d50996589af3e025c16b9bfc7c9493e1af34578e076290` | bit-exact | bit-exact | n/a |

Negative controls that went red as required (all in run.log):

- planted-wrong kernels (`emit … mutant`, valid SPIR-V): compare exits 1 on both CPU drivers
  (vadd 1048576 wrong; fsum16 44123 wrong); one flipped output byte: exits 1, first=1024;
- corrupted modules (truncated, bad magic, word-count overrun): `spirv-val` rejects, loader refuses 21/20/21;
- wrong environment (OpenCL module as SPIRV_VK and the reverse) 22; missing or prefix entry name 23;
  MSL_SOURCE and AARCH64 11; CUBIN 12; unknown tag 10; non-instruction data as SM121_RAW 40;
- HSACO copies with a wrong gfx request, patched e_flags (33), OSABI or ABI version 6 (31), note type
  (34), note target gfx1150 (36, which `llvm-readelf` also shows), truncation (34); SPIR-V as HSACO 30;
- the launcher's isolation scan exits 9 when built to forbid the CPU driver it loads (libpocl, vulkan_lvp);
- the target-table checker rejects a table where one row claims PASS instead of NOT_RUN.

`modules/` holds every emitted module, the ORACLE gfx1151 code object (`llvm-mc` + `ld.lld` from
`backend/amd/vadd-gfx1151.s`) and the refused copies. `hsaco.readelf.txt` and `hsaco.objdump.txt` are the
LLVM oracle's view of it.

## Limits

- PoCL and lavapipe are CPU implementations; bit-exact there says nothing about any GPU's FP32 rounding,
  memory model or compiler. Intel, AMD and every other vendor row stays hardware NOT_RUN.
- The gfx1151 object is an LLVM oracle fixture for the loader. Nothing first-party emits AMD machine
  code yet, and no AMD queue was created (that needs E1 KFD + AQL or the HSA runtime via E2, and the GPU).
- The launcher is migration-pending C until E2 (aien-dev/omega-mv#888).
