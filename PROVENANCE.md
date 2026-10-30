# Provenance: where CUDA++'s hardware knowledge comes from

This file is the rule for adding hardware facts (instruction encodings, container layouts, launch
descriptors, latencies) to CUDA++, and the record of how the existing ones were obtained. It is an
engineering record, not legal advice. Questions that need a lawyer are listed at the end and are open.

`THIRD_PARTY.md` lists third-party material and its licences. `docs/HOW_I_DID_IT.md` describes the method.
This file says which inputs are allowed, which are not, and what every new fact must record.

## 1. The licence terms this follows

Checked 2026-10-27 against the primary texts (copies and SHA-256 values are kept with the review that
produced this file):

- **CUDA Toolkit EULA** (https://docs.nvidia.com/cuda/eula/index.html; online copy "Last updated: January
  7, 2025", `/usr/local/cuda/EULA.txt` of CUDA 13.0 "Last updated: January 12, 2025"). It covers nvcc,
  ptxas, cuobjdump, nvdisasm and the CUDA libraries.
  - §1.2(1): "You may not reverse engineer, decompile or disassemble, or remove copyright or other
    proprietary notices from any portion of the SDK or copies of the SDK."
  - §1.2(8): "You may not reverse engineer, decompile or disassemble any portion of the output generated
    using SDK elements for the purpose of translating such output artifacts to target a non-NVIDIA
    platform."
  - §1.1.2(4): "Unless a developer tool is identified in this Agreement as distributable, it is delivered
    for your internal use only." nvcc, ptxas, cuobjdump and nvdisasm are not in the distributable list
    (Attachment A).
  - §2.1 (CUDA supplement): "The SDK is licensed for you to develop applications only for use in systems
    with NVIDIA GPUs."
- **NVIDIA Driver License Agreement** (driver 580, `/usr/share/doc/libnvidia-compute-580/copyright`):
  §2.1 firmware only on NVIDIA platforms and never translated; §2.2 "You may not reverse engineer,
  decompile, or disassemble the SOFTWARE provided in binary form".
- **NVIDIA open GPU kernel modules**: MIT per file (`third_party/nvidia-open-580.173.02/COPYING`).
- **Mesa NAK/NVK**: MIT. **LLVM**: Apache-2.0 WITH LLVM-exception. **CUTLASS**: BSD-3-Clause.

## 2. Allowed sources

| Class | Source | How it may be used |
|---|---|---|
| A. Public documentation | PTX ISA, CUDA Binary Utilities (including its SASS instruction lists), CUDA Driver API reference, NVIDIA `open-gpu-doc` | Read and cite. |
| A. Open source under a permissive licence | NVIDIA open kernel module headers (MIT), Mesa NAK/NVK (MIT), LLVM `BinaryFormat/ELF.h` (Apache-2.0 WITH LLVM-exception), CUTLASS (BSD-3-Clause) | Cite the file and commit for each fact. Code is copied only with its notice, and only into `third_party/`. |
| A. The chip | Running our own bytes on our own NVIDIA GPU | Any fact; the receipt is the evidence. |
| B. NVIDIA SDK tools, oracle only | `nvcc`, `ptxas`, `cuobjdump`, `nvdisasm` installed on our own machine under the CUDA EULA | Only on inputs we wrote: our own probe kernels (CUDA C or PTX we authored) and bytes our own encoder emitted. Only for NVIDIA-GPU targets. Record the tool version. The tools are never shipped, wrapped or run by CUDA++ itself. |

## 3. Not allowed

1. Running any tool on, extracting from, or reading the machine code or container bytes of an NVIDIA
   binary: `libcuda`, cuBLAS, cuDNN, cuFFT, TensorRT, NCCL, the compiler binaries, the firmware or any
   other driver or SDK file (CUDA EULA §1.2(1); driver licence §2.2). This includes the cubins embedded in
   NVIDIA libraries.
2. Using anything learned from NVIDIA tool output (class B) for a non-NVIDIA target (CUDA EULA §1.2(8)).
   Non-NVIDIA backends take their facts from class A sources for that vendor only.
3. Committing SDK files, NVIDIA headers other than the MIT open-module headers, NVIDIA binaries, or
   firmware.
4. Giving MAX (Modular) source or binaries to an AI system in order to produce software that reimplements
   or substitutes for MAX (MAX License §1.4, "Last Modified: October 21, 2026"). Reading MAX to
   interoperate with it is allowed by the same clause.
5. Describing the work as a reverse-engineered copy of NVIDIA's software. It is not: see section 5.

## 4. What every new hardware fact records

In the source comment next to the fact, or in the test that pins it:

- the class (A or B) and the exact source: document and section, file and commit, or tool and version;
- for class B, the probe input (path of the `.cu`/`.ptx` we wrote) and the command line;
- the chip receipt that confirmed it, or NOT_RUN;
- for golden words: whether they were written independently (from class A plus the chip) or copied from
  class B output. Copied words are a regression pin, not independent evidence
  (see the `independent-sass-opcode-goldens` practice in the omega-harness skills).

Stored NVIDIA tool listings are kept only when a test needs them, under `tests/` or `evidence/`, and each
is listed in section 6.

## 5. How the existing knowledge was obtained (inventory, 2026-10-27)

| Area | Files | Class | Notes |
|---|---|---|---|
| Kernel-module interface (ioctls, classes) | `nvrm/`, `m16/`, `third_party/nvidia-open-580.173.02/` | A | 67 MIT headers copied unchanged. |
| SASS encodings | `src/omega_blackwell_encoder.c`, `src/omega_blackwell_codegen*.c/.h` | A + B | Mesa NAK facts cited; words matched against `nvcc 13.0.88 -arch=sm_121 -cubin` + `cuobjdump -sass` of probe kernels I wrote (`docs/HOW_I_DID_IT.md` step 2). Chip receipts in `evidence/`. |
| Instruction latencies, QMD, dispatch order | `src/omega_gpu_elementwise_api.c`, `src/omega_gpu_cmdlist.c` | A | Mesa `sm120_instr_latencies.rs`, `qmd.rs`, `nvk_cmd_dispatch.c` (MIT). |
| Tensor-core arithmetic model | `tests/gpu_tc_probe_test.c`, `tests/bw_hmma_model.h` | A (chip) | Fitted on chip outputs. |
| cubin container writer | `src/omega_cubin.c`, `src/omega_cubin.h` | B | Layout read from a `ptxas 13.0.88 -arch=sm_121` reference output with `readelf`/`cuobjdump -elf` (the source does not name the probe input; the next change to the file records it). Header constants are also public in LLVM `ELF.h` (class A). No bytes copied. |
| Decode cross-checks | `tests/gpu_reconv_test.c`, `tests/device-trace/run-emitter-checks.sh`, `tools/artifact_smoke.sh` | B | `nvdisasm`/`cuobjdump` read our own bytes. `gpu_reconv_test.c` and `artifact_smoke.sh` skip unavailable tools; `run-emitter-checks.sh` requires `nvdisasm` on `PATH`. |

No source comment or record in this repository says a fact was obtained by disassembling an NVIDIA
library, driver or firmware. A fact found to have such an origin is removed and re-derived from class A
or B, and the removal is recorded here.

## 6. NVIDIA tool output stored in this repository

None. #84 stored `tests/device-trace/expected.sass`, an `nvdisasm 13.0 --binary SM121` listing of the
trace emitter's own words, as a regression pin. #93 (2026-10-27) removed it:
`tests/device-trace/run-emitter-checks.sh` now keeps the listing in a temporary directory and checks
hand-written expectations (opcode counts and the address-path instructions), so nothing `nvdisasm`
prints is committed.

No listing of nvcc-generated code and no NVIDIA binary is stored.

## 7. Open questions for a lawyer

1. Is compiling our own probe kernels with nvcc and reading the result with cuobjdump, to learn the
   encoding of NVIDIA instructions for an NVIDIA-GPU toolchain, within the CUDA EULA? §1.2(8) is limited to
   translation to non-NVIDIA platforms; does any other clause reach it?
2. Is a toolchain that replaces nvcc on NVIDIA GPUs an "application" developed "only for use in systems
   with NVIDIA GPUs" (§2.1)?
3. Can tool listings of our own code (section 6) be published, given §1.1.2(4)?
