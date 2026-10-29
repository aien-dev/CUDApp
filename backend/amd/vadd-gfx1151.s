// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// vadd-gfx1151.s: ORACLE fixture, not a first-party emitter. FP32 c[gid] = a[gid] + b[gid], gid < n,
// assembled by LLVM 18.1.3 llvm-mc and linked by ld.lld 18.1.3 (tests/multitarget/run.sh) to give
// backend/targets.om's AMDGPU_HSACO loader a real code object to validate and refuse. The first-party
// path is an Omega gfx11 encoder from the AMD RDNA3.5 ISA guide (docs/ARCHITECTURE.md: LLVM is a
// verification oracle only). Hardware NOT_RUN: no AMD GPU on this host.
// Kernarg layout: a @0, b @8, c @16 (global float pointers), count @24 (uint; "n" alone would parse as YAML false); 64 work-items per group.
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.globl vadd
.p2align 8
.type vadd,@function
vadd:
  s_load_b128 s[4:7], s[0:1], 0x0
  s_load_b64 s[8:9], s[0:1], 0x10
  s_load_b32 s10, s[0:1], 0x18
  v_and_b32 v0, 0x3ff, v0
  s_lshl_b32 s11, s2, 6
  v_add_nc_u32 v1, s11, v0
  s_waitcnt lgkmcnt(0)
  v_cmp_gt_u32 vcc_lo, s10, v1
  s_and_saveexec_b32 s12, vcc_lo
  s_cbranch_execz .Lend
  v_lshlrev_b32 v2, 2, v1
  global_load_b32 v3, v2, s[4:5]
  global_load_b32 v4, v2, s[6:7]
  s_waitcnt vmcnt(0)
  v_add_f32 v3, v3, v4
  global_store_b32 v2, v3, s[8:9]
.Lend:
  s_endpgm
.Lvadd_end:
.size vadd, .Lvadd_end-vadd

.rodata
.p2align 6
.amdhsa_kernel vadd
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_kernarg_size 28
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 13
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.target: amdgcn-amd-amdhsa--gfx1151
amdhsa.kernels:
  - .name: vadd
    .symbol: vadd.kd
    .kernarg_segment_size: 28
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 13
    .vgpr_count: 5
    .max_flat_workgroup_size: 64
    .args:
      - { .name: a, .offset: 0, .size: 8, .value_kind: global_buffer, .address_space: global }
      - { .name: b, .offset: 8, .size: 8, .value_kind: global_buffer, .address_space: global }
      - { .name: c, .offset: 16, .size: 8, .value_kind: global_buffer, .address_space: global }
      - { .name: count, .offset: 24, .size: 4, .value_kind: by_value }
...
.end_amdgpu_metadata
