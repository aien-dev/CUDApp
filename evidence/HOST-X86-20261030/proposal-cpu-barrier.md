# Title

Seed a host CPU publication barrier: `dsb sy` on AArch64, a real fence elsewhere

# Body

ADR 0043 missing feature. Seeded Omega has no host CPU publication barrier.

CUDA++ needs one. On the GB10 host the production instruction is `dsb sy`. On any other host the same driver sources have to compile, and the publication barrier has to be a real sequentially consistent fence. A no-op is not that barrier.

`nvrm/cpu_barrier.h` is the C stand-in. The AArch64 body is `dsb sy`. The other body is `__atomic_thread_fence(__ATOMIC_SEQ_CST)`. `nvrm_cpu_relax` is `yield` on AArch64 and a compiler barrier elsewhere. Callers in `nvrm/`, `m16/` and the Blackwell engine use `nvrm_cpu_barrier()`.

The non-AArch64 fence is migration-pending C. The portable library compiles those callers with the existing switches `CUDAPP_MIGRATION_PENDING_C_ENCODER` and `CUDAPP_MIGRATION_PENDING_C_CUBIN`.

An AArch64 cross build of the shared objects that include this header is byte-identical to CUDA++ main. That comparison is not a GB10 run. GB10 device behavior stays NOT RUN.

This draft is not yet filed on omega-mv. The header comment points here until a filed link replaces it.
