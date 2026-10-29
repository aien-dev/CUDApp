# C0 frozen CUDA compatibility profile (CUDApp#184)

Status: IMPLEMENTED and host-EXERCISED (static admission only). Device execution is NOT_RUN. This directory
pins one CUDA Driver API application, its PTX module, and the exact surface either may use. Later gates
(Bridge#42 C1 ABI, CUDApp#185 C2 translate, CUDApp#186 runner) check against these files and do not
redefine them.

| File | Role |
|---|---|
| `fixture/vadd_drv.c` | Driver API application written for this gate: one context, one PTX module, three buffers, one launch, host-side verification. |
| `fixture/vadd_u32.ptx` | The module it loads, written by hand from the PTX ISA (class A, PROVENANCE.md section 2). |
| `build.sh OUT` | Rebuilds the aarch64 and x86_64 executables. A link-only stub `libcuda.so.1` (each cu* symbol defined as 0, no code) satisfies the link; it is never a runtime library. |
| `profile-c0-aarch64-v1.tsv`, `profile-c0-x86_64-v1.tsv` | The frozen profiles. |
| `c0_check.om` | Seeded Omega admission checker, composed after `om/cache/digest.om` (SHA-256). |
| `test.sh [WORKDIR]` | Rebuilds twice (hashes must match), admits both profiles, then requires 15 negative controls to refuse. `make test-cuda-c0-profile`. |

## Checker

```
c0_check check PROFILE EXE PTX RUNISA   # RUNISA: aarch64 | x86_64 | static
c0_check emit  PROFILE EXE PTX          # print the records read from the artifacts (authoring aid, never admission)
```

Exit 0 prints `CUDA-C0-PROFILE PASS records=N imports=I ptxops=P`. Exit 1 prints one
`CUDA-C0-PROFILE REFUSED <reason> [kind] <detail>` line for each refusal. Exit 2 means a usage error and
exit 3 an unreadable or oversized input (profile and PTX at most 64 KiB, executable at most 1 MiB). The
checker reads the three files and writes only to stdout. It never runs the artifact.

Admission requires all of the following:

- The profile is canonical. Lines end in LF, fields are separated by TAB, there are at most 8 non-empty
  printable-ASCII fields, the record kind is known, its arity is exact, and the first record is
  `schema AIEN_CUDA_COMPAT_PROFILE 1`. `schema`, `profile`, `host` and `ptx` each appear once, and there
  are exactly two `artifact` rows. Duplicate keys refuse.
- The SHA-256 of the executable and of the PTX equal the `artifact` rows.
- The ELF64 little-endian header is valid, and `e_machine` equals the `host` ISA (AArch64 183,
  x86-64 62). `RUNISA` must equal the host ISA unless it is `static`. An ARM64 profile is therefore never
  admitted on an x86-64 runner, and the reverse.
- Each of the following must equal its profile rows exactly, in both directions, so that a missing or
  extra item refuses:
  - DT_NEEDED;
  - undefined `.dynsym` symbols, with a weak binding if and only if the status is `weak`;
  - GNU verneed (version, file) pairs;
  - the PTX entry and its parameter count;
  - PTX directives, opcodes and special registers.
- A `cuda` import must be `supported`. Any import marked `unsupported` refuses.
- No import starts with a `lookup absent` prefix (`dlopen`, `dlmopen`, `dlsym`, `dlvsym`,
  `cuGetProcAddress`, `cuGetExportTable`). Every entry point the executable can reach is therefore in the
  static import list. Runtime-discovered calls are statically excluded; a runtime trace is the C3 arm
  below.
- The `host` libc floor equals the highest `GLIBC_` version the executable requires.
- The PTX `.version`, `.target` and `.address_size` match the `ptx` row, and each appears once.
- Every PTX instruction line holds exactly one `;`. No opcode root matches a `deny` row.

## Profile record kinds

`schema`, `profile <id> <ver> C0`, `host <isa> ELF64 <abi> <os> <GLIBC floor>`,
`toolchain <name> <version>`, `artifact <host-exe|ptx> <sha256> <name>`,
`needed <soname> <replacement|system>`, `verneed <version> <file>`,
`import <symbol> <cuda|libc|crt> <supported|system|weak> <source api>`, `lookup absent <prefix>`,
`ptx <version> <target> <address size>`, `entry <name> <params>`, `regprefix <p>`,
`ptxdir|ptxop|sreg <token> supported`, `deny <opcode root> <class>`,
`feature <name> <supported|unsupported> <detail>`, `numeric <op> <rule>`, `launch <k> <v>`,
`expect <k> <v>`, `runner <id> <gate> <AVAILABLE_NOT_RUN|BLOCKED_HARDWARE|BLOCKED_GPU_WINDOW|BLOCKED_DECISION> <evidence>`.

The C0 surface consists of these Driver API calls: cuInit, cuDeviceGet, cuCtxCreate (`_v4`),
cuModuleLoadData, cuModuleGetFunction, cuMemAlloc (`_v2`), cuMemcpyHtoD/DtoH (`_v2`), cuLaunchKernel,
cuCtxSynchronize, cuMemFree (`_v2`), cuModuleUnload and cuCtxDestroy (`_v2`). The PTX side is integer
global-memory arithmetic only.

The following are unsupported in C0, and each is either refused or absent:

- the runtime API;
- cubin and fatbin loading;
- streams and events;
- shared memory;
- warp collectives, barriers, atomics and fences;
- textures and surfaces;
- floating point, conversions and integer division;
- unified memory;
- peer access;
- graphs.

`numeric` rows fix the integer semantics (wrap modulo 2^32).

## Launch values (Wolfram)

`WolframLanguageEvaluator: With[{n=4099,blk=256}, ... a=Mod[i*2654435761,2^32], b=Mod[(2^32-1)-Mod[7i,2^32],2^32], c=Mod[a+b,2^32] ...]`
gives grid 17, 4352 threads, 253 idle tail threads, checksum 8806203976955 (fits in u64), and 4098 wrapped
sums.

## Provenance

- `fixture/*` are class A, written for this gate.
- `cuda.h` (CUDA 13.0, `CUDA_VERSION 13000`) is read from the local install at build time and is not
  committed.
- No NVIDIA binary is opened, linked or run. The stub `libcuda.so.1` is generated from our own object.
- The optional syntax check `ptxas -arch=sm_121 fixture/vadd_u32.ptx -o /dev/null` (ptxas V13.0.88) runs
  class B on our own input. Nothing learned from it feeds a non-NVIDIA target, and its output is never
  stored.
- The x86_64 executable is a cross-built candidate (x86_64-linux-gnu-gcc 13.3.0 on the Spark). It has been
  checked statically and never run.

## Runner inventory (observed 2026-10-29)

| Runner | Gate | State | Evidence |
|---|---|---|---|
| Spark host (aarch64, glibc 2.39) | C0 static | EXERCISED by `test.sh` | `evidence/CUDA-C0-PROFILE-20261029/` |
| Spark GB10 (10de:2e12 rev a1, NVRM 580.173.02 open, VBIOS 9a.0b.2d.00.00) | C3 device | BLOCKED_GPU_WINDOW | `/proc/driver/nvidia`, lspci; needs a spark-window and Drake's authorization |
| arm64 AMD / Intel GPU | C4 | BLOCKED_HARDWARE | none attached to the Spark |
| x86_64 NVIDIA / AMD / Intel host | C4 | BLOCKED_HARDWARE | no x86 GPU host in inventory |
| Stock NVIDIA cuda-samples (v13.4) | C3 | BLOCKED_DECISION | building them needs nvcc on NVIDIA's inputs, outside PROVENANCE class B |

## GPU arms (NOT_RUN; each needs a spark-window and Drake's authorization)

```
# C3: run the C0 fixture on GB10 against the CUDApp replacement libcuda, with a binding trace
spark-window open --gb10 shared --quiet none <ledger>        # coordinator-approved ledger only
compat/cuda/build.sh /tmp/c0 && cd /tmp/c0/aarch64
LD_LIBRARY_PATH=<cudapp replacement libcuda dir> LD_DEBUG=bindings ./vadd_drv ../vadd_u32.ptx 2> bindings.log
# expect: exit 0, stdout "VADD_U32 PASS n=4099 grid=17 block=256 checksum=8806203976955",
# every cu* binding resolved in the replacement library, and none in an NVIDIA libcuda
spark-window close <window-id>
```

- Bridge#42 (C1): check the replacement library's SONAME and exports against the `import ... cuda` rows,
  refuse a wrong-ISA library (`e_machine` against `host`), and propagate CUresult. Host part: ELF
  inspection. The device part follows the C3 arm.
- CUDApp#185 (C2): translate `fixture/vadd_u32.ptx` with exactly the `ptxop` and `sreg` rows. Every
  `deny` root must be refused.
- CUDApp#186: `build/cuda-compat-check --manifest compat/cuda/profile-c0-<isa>-v1.tsv --gate C0` wraps
  `c0_check`. The manifest is this TSV, not JSON. NOT_RUN rows stay NOT_RUN.
