# CUDApp architecture: today's layers and the portable-top / thin-backend split

**Status:** PROPOSED design (2026-10-27). Nothing in this file is implemented by it. No code moves until
the gate below is met. The decision record is ADR 0043 in aien-dev/aien-architecture-mv
(`docs/adr/0043-gpu-stack-portable-top-thin-backend.md`, draft). Read that first; this file is the
file-level map for CUDApp at main `780a0af1`.

**Companions, not repeated here:** `docs/runtime/PORTABILITY.md` and `portability_table.tsv` (per-file
chip and OS binding, checked by `make test-runtime-portability`; this file agrees with its finding that
the session, command list and kernel APIs hide the driver but not the GPU). PORTABILITY.md cites
`docs/numeric/OMEGA_GPU_NATIVE_BOUNDARY.md` and `OMEGA_GPU_ENGINE.md`; those live in omega-mv
(present on omega-mv main `e5b32567`), not in this repo.

**Gate.** Work under this design starts after P0 publishes a same-box head-to-head and bars C2–C5 are
QUALIFIED on the GB10 (`~/handoffs/GpuStack/PLAN.md`, phases P2–P4).

**Language.** Existing C stays where it is and is not reworked. New host-runtime code, every new
backend and the backend ABI are written in Omega (`.om`). Any new C needs a missing-feature reason with
an Omega extension proposal filed, behind a `migration-pending` build switch.

## 1. Target layers

```
 Omegese kernels ──► kernel IR + checker2 (omega-mv, pinned)
                          │ per-target lowering
 ┌──────────────── portable top (.om, CUDApp) ───────────────────┐
 │ model graphs, decode scheduling, batching, serving, memory      │
 │ planner (weights, KV pool, scratch), streams/events, sharing,   │
 │ watchdog policy, public runtime API                             │
 └───────────────────────┬────────────────────────────────────────┘
              OmegaBackend ABI 2.0: 30 calls (backend/abi.om)
 ┌────────────┬────────────┬─────────────┬───────────┬───────────┐
 │ gb10       │ amd        │ vulkan      │ metal     │ cpu-ref   │
 │ sm_121     │ gfx942 /   │ SPIR-V      │ MSL text  │ AArch64   │
 │ nvrm + UVM │ gfx1151    │ (loader:    │ (needs    │ CI and    │
 │ QMD/PB     │ KFD + AQL  │ needs E2)   │ E2)       │ negative  │
 │            │ (needs E1) │             │           │ controls  │
 └────────────┴────────────┴─────────────┴───────────┴───────────┘
 Integrations (omega-bridge connected repos): ggml-omega, vLLM platform plugin, ORT plugin EP
```

E1, E2 and E3 are the Omega extension proposals named in ADR 0043: E1 a KFD `devctl` allowlist, E2
checked shared-library calls and Mach-O output, E3 IR v7 (subgroup width, shared budget, nesting cap and
accuracy class as target parameters; address spaces, atomics, f16/bf16/fp8, one tiled MMA op), filed as
aien-dev/omega-mv#883.

## 2. Where today's files sit

Classes: **TOP** portable; **ENC** encoder, QMD, pushbuffer words; **KER** tuned sm_121 kernel
generators; **DRV** driver client; **MIXED** has to be split logically. Line numbers are at `780a0af1`.

| Class | Files |
|---|---|
| TOP | `omega_core`, `omega_canonical`, `omega_types.h`, `omega_vector`, `sha256`, `omega_runtime_api.h`, `omega_gpu_platform.{c,h}` (interface), `omega_gpu_engine` (interface), `omega_gpu_wait`, `omega_gpu_mem_cached`, `omega_gpu_serving`, `omega_gpu_chip`, `omega_gpu_hostmath`, `omega_gpu_fmt` |
| ENC | `omega_blackwell_codegen.{c,h}` core (BlackwellIR, regalloc, encode), `omega_blackwell_encoder`, `omega_blackwell_qmd`, `omega_bw_em.h`, `omega_bw_em_q.h`, `omega_bw_reconv.h`, `omega_gpu_code_alloc.h`, `omega_cubin`, `omega_blackwell_realize`, `omega_blackwell_matmul` |
| KER | `omega_blackwell_codegen_{attn,bf16,dec,gemm,moe,moe2,pattn,pf,pfhost,qg,qmma,router,tc,tc2,tern}`, the matmul generators inside `omega_blackwell_codegen.c`, `omega_gpu_fp8_pack.c`, `omega_gpu_pfgemm.c` |
| DRV | `nvrm/` (`nvrm.c` 908 lines, RM client against the 580.173.02 headers), `m16/`, `omega_gpu_platform_linux.c`, `omega_blackwell_engine.c`, `omega_blackwell_submit.c` |
| MIXED | `omega_gpu_session.c` (QMD and pushbuffer construction in `omega_gpu_session_launch` :445; includes qmd/submit :18-19), `omega_gpu_cmdlist.c` (`emit_tail` :199, `emit_stamp` :219, `emit_inline_qmd` :226, PCAS method words :308-309), `omega_native_artifact.c` (packaging is TOP, QMD template bytes are ENC), every `omega_gpu_*_api.c` (tensor lifetime is TOP; inline emitters in `attention_api.c` :69-143 and `elementwise_api.c` :61-145 are KER) |
| Already `.om` | `native/nvrm/context.om`, `native/blackwell/{trace-emit,trace,vector,launch}.om` |

Three structural defects the split removes (in the `.om` stack; the C stack is frozen, not reworked):

1. **Two backend tables over one driver client.** `OmegaGpuPlatform` 1.2 (`omega_gpu_platform.h`, 22
   function pointers) under session and cmdlist; `OmegaGpuBackend` (`omega_gpu_engine.h`, 15 ops,
   VECTOR_1D only) under the job engine, installed at `omega_blackwell_submit.c:102`.
2. **Raw NVIDIA method words cross the seam.** `queue_submit(words)` makes QMD and pushbuffer encoding
   happen above the backend, in session and cmdlist.
3. **Two bypasses.** `omega_blackwell_engine.c` (:125-516) and `omega_blackwell_submit.c` (:185-328) call
   `nvrm_*` and `m16_native_*` directly.

## 3. The backend ABI (30 calls)

The full contract table, with today's GB10 function for each call, is in ADR 0043 Decision 2.

| Group | Calls |
|---|---|
| Negotiate and device (7) | `ob_negotiate`, `ob_device_count`, `ob_device_open`, `ob_device_close`, `ob_device_caps`, `ob_device_status`, `ob_diag_text` |
| Memory (7) | `ob_mem_alloc`, `ob_mem_import_host` (zero-copy), `ob_mem_free`, `ob_mem_flush`, `ob_mem_invalidate`, `ob_mem_advise`, `ob_mem_info` |
| Code (3) | `ob_code_load`, `ob_code_info`, `ob_code_free` |
| Queues and dispatch (4) | `ob_queue_open`, `ob_queue_close`, `ob_dispatch`, `ob_submit` |
| Graphs (4) | `ob_graph_begin`, `ob_graph_end`, `ob_graph_launch` (one doorbell per token, patch list), `ob_graph_free` |
| Sync and recovery (5) | `ob_fence_wait`, `ob_event_record`, `ob_queue_wait_event`, `ob_queue_reset`, `ob_clock_ns` |

Leaving the backend compared with `OmegaGpuPlatform` 1.2: `time_now_ms/ns`, `wait_ms`, `lock`, `unlock`,
`host_alloc`, `host_free` (the top gets them from Omega host words), `mmio_*` and `irq_wait` (NULL on
Linux today), `queue_submit(words)` (replaced by `ob_dispatch` + `ob_submit`).

Code-object formats accepted by `ob_code_load`: `SM121_RAW`, `AMDGPU_HSACO`, `SPIRV_VK`, `MSL_SOURCE`,
`AARCH64`. `CUBIN` is reserved and unused while omega-mv #864 is on HOLD.

## 4. Per-token path, today and target (GB10)

| | Today (`omega_gpu_decode_graph`, Qwen3-30B-A3B) | Target |
|---|---|---|
| Launches per token | 436 (9 per layer × 48 + 4) | same graph |
| Submissions / doorbells | 1 / 1 (PCAS mode, cmdlist.c:325-333) | 1 / 1 via `ob_graph_launch` |
| CPU writes per token | 1 KB constants + 1 KB QMD per launch, ≈872 KiB [INFERENCE] | patch list only; QMDs pre-encoded at `ob_graph_end` |
| Weights | copied: `omega_gpu_fp8_api.c:269-275` `memcpy` + sync | mmapped file mapped by `ob_mem_import_host` |

Every row in the Target column is PROPOSED and needs a paired A/B receipt before it is described as
faster.

## 5. Public API consequences

- `docs/runtime/omega_runtime_api.manifest` (ABI `omega_runtime 0.1`, 91 functions, 21 STABLE) stays
  as is until a cutover. Moving STABLE `omega_gpu_session_launch` / `OmegaGpuLaunch` to a code object
  is a MAJOR bump.
- The decode-path APIs are not in the manifest today: fp8, fp8gemm, bf16_gemv, decode_attn, router,
  tern, tcgemm, prefill*, moe, decode_graph. They enter it as EXPERIMENTAL before any of them changes
  (CUDApp#124).
- `aien-serve-http` is not in CUDApp (aien-dev/aien-sovereign-core-mv `crates/aien-serve-http`, pin
  `754c07b1`); it consumes this library through that repo's FFI crate [INFERENCE].

## 6. Provenance rules for any backend

- Encodings come from Mesa NAK (MIT) plus chip tests (`THIRD_PARTY.md` §2). `docs/HOW_I_DID_IT.md:35-39`
  records a preference for nvcc output where it and NAK disagree; every encoder field gets a NAK
  citation or a chip test before the GB10 encoder is declared the backend.
- Nothing derived from omega-mv #864 (`cubin.om`) enters a backend while #864 is on HOLD.
- AMD encodings come from AMD's published ISA reference guides; LLVM (`llvm-mc`, `llvm-readelf`) is a
  verification oracle only. The terms of the ISA guides are an open item for the IP-lawyer review.
- Apple: MSL text through Apple's public compiler; no AGX reverse engineering.

## 7. Review

See `docs/REVIEWING.md` and the proposed owners file `docs/review/CODEOWNERS.proposed` (inactive until
human reviewers exist).
