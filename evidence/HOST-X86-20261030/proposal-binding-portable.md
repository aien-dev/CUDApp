# Title

Emit a host-assembleable native binding, or keep a migration-pending portable stand-in

# Body

ADR 0043 missing feature. `src/cudapp_om_units.c` and `src/cudapp_freest_units.c` are generated AArch64 machine code. Seeded Omega does not yet emit a binding the build host can assemble. An x86-64 checkout cannot compile those two files.

`src/cudapp_binding_portable.c` is the migration-pending C stand-in. The AArch64 library does not link it. A non-AArch64 host links it only when the build passes `CUDAPP_MIGRATION_PENDING_C_ENCODER` and `CUDAPP_MIGRATION_PENDING_C_CUBIN`, the switches that select the C encoder and the C cubin writer. The host vector test defines `BINDING_PORTABLE`. The file refuses to compile if none of those three macros is set.

`omega_native_revision` and `omega_native_epoch` follow the committed freestanding units. Every other entry returns status 34 and does not write its result. That refusal is not a result from the seeded unit. The C encoder and the C cubin writer do the portable codegen. This is not a run of `bw_enc_sm121.om` or `cubin_elf.om`, and it is not a pass of the `.om` versus C differential.

This draft is not yet filed on omega-mv. The source comment points here until a filed link replaces it.
