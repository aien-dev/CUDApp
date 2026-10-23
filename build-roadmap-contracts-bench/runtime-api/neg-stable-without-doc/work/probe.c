#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "/home/drakestapleton/workspace/cudapp-roadmap-numerics/build-roadmap-contracts-bench/runtime-api/neg-stable-without-doc/src/omega_runtime_api.h"
#ifndef OMEGA_RUNTIME_API_VERSION_MAJOR
#error "version macro missing: OMEGA_RUNTIME_API_VERSION_MAJOR"
#endif
#ifndef OMEGA_RUNTIME_API_VERSION_MINOR
#error "version macro missing: OMEGA_RUNTIME_API_VERSION_MINOR"
#endif
#include "/home/drakestapleton/workspace/cudapp-roadmap-numerics/build-roadmap-contracts-bench/runtime-api/neg-stable-without-doc/src/omega_runtime_api.h"
#include "/home/drakestapleton/workspace/cudapp-roadmap-numerics/build-roadmap-contracts-bench/runtime-api/neg-stable-without-doc/src/omega_gpu_platform.h"
#include "/home/drakestapleton/workspace/cudapp-roadmap-numerics/build-roadmap-contracts-bench/runtime-api/neg-stable-without-doc/src/omega_gpu_session.h"
#include "/home/drakestapleton/workspace/cudapp-roadmap-numerics/build-roadmap-contracts-bench/runtime-api/neg-stable-without-doc/src/omega_gpu_mem_cached.h"
#include "/home/drakestapleton/workspace/cudapp-roadmap-numerics/build-roadmap-contracts-bench/runtime-api/neg-stable-without-doc/src/omega_gpu_cmdlist.h"
#include "/home/drakestapleton/workspace/cudapp-roadmap-numerics/build-roadmap-contracts-bench/runtime-api/neg-stable-without-doc/src/omega_gpu_engine.h"
#include "/home/drakestapleton/workspace/cudapp-roadmap-numerics/build-roadmap-contracts-bench/runtime-api/neg-stable-without-doc/src/omega_gpu_matmul_api.h"
#include "/home/drakestapleton/workspace/cudapp-roadmap-numerics/build-roadmap-contracts-bench/runtime-api/neg-stable-without-doc/src/omega_gpu_serving.h"
#include "/home/drakestapleton/workspace/cudapp-roadmap-numerics/build-roadmap-contracts-bench/runtime-api/neg-stable-without-doc/src/omega_gpu_elementwise_api.h"
#include "/home/drakestapleton/workspace/cudapp-roadmap-numerics/build-roadmap-contracts-bench/runtime-api/neg-stable-without-doc/src/omega_gpu_attention_api.h"
typedef uint64_t (*FP1)(void);
#if !defined(ONLY) || ONLY == 67
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_platform_refusal_name), const char * (int32_t)), "signature of omega_gpu_platform_refusal_name");
#endif
#if !defined(ONLY) || ONLY == 68
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_platform_negotiate), int32_t (const OmegaGpuPlatformCaps *, const OmegaGpuPlatformCaps *)), "signature of omega_gpu_platform_negotiate");
#endif
#if !defined(ONLY) || ONLY == 69
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_platform_check), int32_t (const OmegaGpuPlatform *, const OmegaGpuPlatformCaps *)), "signature of omega_gpu_platform_check");
#endif
#if !defined(ONLY) || ONLY == 70
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_platform_check_for), int32_t (const OmegaGpuPlatform *, const OmegaGpuPlatformCaps *, uint32_t)), "signature of omega_gpu_platform_check_for");
#endif
#if !defined(ONLY) || ONLY == 71
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_platform_check_buffer), int32_t (const OmegaGpuBufferDesc *, const OmegaGpuPlatformCaps *)), "signature of omega_gpu_platform_check_buffer");
#endif
#if !defined(ONLY) || ONLY == 72
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_handles_init), void (OmegaGpuHandleTable *)), "signature of omega_gpu_handles_init");
#endif
#if !defined(ONLY) || ONLY == 73
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_handles_open), int32_t (OmegaGpuHandleTable *, uint32_t, uint64_t, uint32_t, uint64_t, uint32_t *, uint32_t *)), "signature of omega_gpu_handles_open");
#endif
#if !defined(ONLY) || ONLY == 74
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_handles_check), int32_t (const OmegaGpuHandleTable *, uint32_t, uint32_t, uint32_t)), "signature of omega_gpu_handles_check");
#endif
#if !defined(ONLY) || ONLY == 75
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_handles_close), int32_t (OmegaGpuHandleTable *, uint32_t, uint32_t, uint32_t)), "signature of omega_gpu_handles_close");
#endif
#if !defined(ONLY) || ONLY == 113
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_lock), void (void)), "signature of omega_gpu_session_lock");
#endif
#if !defined(ONLY) || ONLY == 114
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_unlock), void (void)), "signature of omega_gpu_session_unlock");
#endif
#if !defined(ONLY) || ONLY == 115
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_open), OmegaGpuSession * (void)), "signature of omega_gpu_session_open");
#endif
#if !defined(ONLY) || ONLY == 116
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_is_open), bool (void)), "signature of omega_gpu_session_is_open");
#endif
#if !defined(ONLY) || ONLY == 117
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_is_blocked), int32_t (void)), "signature of omega_gpu_session_is_blocked");
#endif
#if !defined(ONLY) || ONLY == 118
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_close), void (void)), "signature of omega_gpu_session_close");
#endif
#if !defined(ONLY) || ONLY == 119
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_open_count), uint32_t (void)), "signature of omega_gpu_session_open_count");
#endif
#if !defined(ONLY) || ONLY == 120
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_alloc), int32_t (size_t, OmegaGpuMem *)), "signature of omega_gpu_session_alloc");
#endif
#if !defined(ONLY) || ONLY == 121
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_alloc_code), int32_t (size_t, OmegaGpuMem *)), "signature of omega_gpu_session_alloc_code");
#endif
#if !defined(ONLY) || ONLY == 122
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_free), void (OmegaGpuMem *)), "signature of omega_gpu_session_free");
#endif
#if !defined(ONLY) || ONLY == 123
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_sync_to_gpu), int32_t (const OmegaGpuMem *)), "signature of omega_gpu_session_sync_to_gpu");
#endif
#if !defined(ONLY) || ONLY == 124
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_sync_to_cpu), int32_t (const OmegaGpuMem *)), "signature of omega_gpu_session_sync_to_cpu");
#endif
#if !defined(ONLY) || ONLY == 125
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_scratch), int32_t (OmegaGpuScratch *, size_t)), "signature of omega_gpu_session_scratch");
#endif
#if !defined(ONLY) || ONLY == 126
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_scratch_free), void (OmegaGpuScratch *)), "signature of omega_gpu_session_scratch_free");
#endif
#if !defined(ONLY) || ONLY == 127
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_launch), int32_t (const OmegaGpuLaunch *, uint64_t *, uint32_t *)), "signature of omega_gpu_session_launch");
#endif
#if !defined(ONLY) || ONLY == 128
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_last_error), const char * (void)), "signature of omega_gpu_session_last_error");
#endif
#if !defined(ONLY) || ONLY == 129
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_parked), void (OmegaGpuParkStats *)), "signature of omega_gpu_session_parked");
#endif
#if !defined(ONLY) || ONLY == 130
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_is_parked), int32_t (OmegaGpuBufferHandle)), "signature of omega_gpu_session_is_parked");
#endif
#if !defined(ONLY) || ONLY == 131
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_alloc_cached), int32_t (size_t, OmegaGpuMem *)), "signature of omega_gpu_session_alloc_cached");
#endif
#if !defined(ONLY) || ONLY == 132
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_scratch_reserve), int32_t (OmegaGpuScratch *, size_t)), "signature of omega_gpu_session_scratch_reserve");
#endif
#if !defined(ONLY) || ONLY == 133
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_scratch_unreserve), void (OmegaGpuScratch *)), "signature of omega_gpu_session_scratch_unreserve");
#endif
#if !defined(ONLY) || ONLY == 134
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_cancel), void (void)), "signature of omega_gpu_session_cancel");
#endif
#if !defined(ONLY) || ONLY == 135
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_set_platform), int32_t (const OmegaGpuPlatform *, const OmegaGpuPlatformCaps *)), "signature of omega_gpu_session_set_platform");
#endif
#if !defined(ONLY) || ONLY == 136
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_platform), const OmegaGpuPlatform * (void)), "signature of omega_gpu_session_platform");
#endif
#if !defined(ONLY) || ONLY == 137
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_submit_prefix_words), uint32_t (void)), "signature of omega_gpu_session_submit_prefix_words");
#endif
#if !defined(ONLY) || ONLY == 138
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_completion_mode), OmegaGpuCompletionMode (void)), "signature of omega_gpu_session_completion_mode");
#endif
#if !defined(ONLY) || ONLY == 139
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_now_ns), uint64_t (void)), "signature of omega_gpu_session_now_ns");
#endif
#if !defined(ONLY) || ONLY == 140
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_host_alloc), void * (size_t)), "signature of omega_gpu_host_alloc");
#endif
#if !defined(ONLY) || ONLY == 141
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_host_calloc), void * (size_t, size_t)), "signature of omega_gpu_host_calloc");
#endif
#if !defined(ONLY) || ONLY == 142
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_host_free), void (void *)), "signature of omega_gpu_host_free");
#endif
#if !defined(ONLY) || ONLY == 158
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_cmdlist_barrier_name), const char * (OmegaGpuCmdListBarrier)), "signature of omega_gpu_cmdlist_barrier_name");
#endif
#if !defined(ONLY) || ONLY == 159
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_cmdlist_create), int32_t (uint32_t, OmegaGpuCmdList * *)), "signature of omega_gpu_cmdlist_create");
#endif
#if !defined(ONLY) || ONLY == 160
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_cmdlist_free), void (OmegaGpuCmdList *)), "signature of omega_gpu_cmdlist_free");
#endif
#if !defined(ONLY) || ONLY == 161
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_cmdlist_reset), void (OmegaGpuCmdList *)), "signature of omega_gpu_cmdlist_reset");
#endif
#if !defined(ONLY) || ONLY == 162
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_cmdlist_count), uint32_t (const OmegaGpuCmdList *)), "signature of omega_gpu_cmdlist_count");
#endif
#if !defined(ONLY) || ONLY == 163
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_cmdlist_mark_independent), int32_t (OmegaGpuCmdList *)), "signature of omega_gpu_cmdlist_mark_independent");
#endif
#if !defined(ONLY) || ONLY == 164
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_cmdlist_add), int32_t (OmegaGpuCmdList *, const OmegaGpuLaunch *)), "signature of omega_gpu_cmdlist_add");
#endif
#if !defined(ONLY) || ONLY == 165
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_cmdlist_submit), int32_t (OmegaGpuCmdList *, const OmegaGpuCmdListOpts *, OmegaGpuCmdListStats *)), "signature of omega_gpu_cmdlist_submit");
#endif
#if !defined(ONLY) || ONLY == 166
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_cmdlist_is_latched), int32_t (void)), "signature of omega_gpu_cmdlist_is_latched");
#endif
#if !defined(ONLY) || ONLY == 167
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_cmdlist_stamps), int32_t (const OmegaGpuCmdList *, uint64_t *, uint32_t)), "signature of omega_gpu_cmdlist_stamps");
#endif
#if !defined(ONLY) || ONLY == 230
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_execute), int32_t (const OmegaGpuJob *, OmegaGpuResult *)), "signature of omega_gpu_execute");
#endif
#if !defined(ONLY) || ONLY == 231
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_engine_set_backend), void (const OmegaGpuBackend *)), "signature of omega_gpu_engine_set_backend");
#endif
#if !defined(ONLY) || ONLY == 232
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_engine_is_blocked), int32_t (void)), "signature of omega_gpu_engine_is_blocked");
#endif
#if !defined(ONLY) || ONLY == 233
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_engine_set_clock), void (FP1)), "signature of omega_gpu_engine_set_clock");
#endif
#if !defined(ONLY) || ONLY == 234
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_engine_failure_name), const char * (int32_t)), "signature of omega_gpu_engine_failure_name");
#endif
#if !defined(ONLY) || ONLY == 235
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_engine_state_name), const char * (int32_t)), "signature of omega_gpu_engine_state_name");
#endif
#if !defined(ONLY) || ONLY == 236
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_engine_wait_name), const char * (int32_t)), "signature of omega_gpu_engine_wait_name");
#endif
#if !defined(ONLY) || ONLY == 237
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_engine_step_name), const char * (int32_t)), "signature of omega_gpu_engine_step_name");
#endif
#if !defined(ONLY) || ONLY == 347
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_matmul_bf16), int32_t (uint32_t, uint32_t, uint32_t, const uint16_t *, const uint16_t *, float *, OmegaGpuMatmulInfo *)), "signature of omega_gpu_matmul_bf16");
#endif
#if !defined(ONLY) || ONLY == 348
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_matmul_f32), int32_t (uint32_t, uint32_t, uint32_t, const float *, const float *, float *, OmegaGpuMatmulInfo *)), "signature of omega_gpu_matmul_f32");
#endif
#if !defined(ONLY) || ONLY == 349
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_matmul_rc_name), const char * (int32_t)), "signature of omega_gpu_matmul_rc_name");
#endif
#if !defined(ONLY) || ONLY == 350
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_matmul_cache_clear), void (void)), "signature of omega_gpu_matmul_cache_clear");
#endif
#if !defined(ONLY) || ONLY == 351
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_tensor_upload_bf16), int32_t (uint32_t, uint32_t, const uint16_t *, OmegaGpuTensor * *)), "signature of omega_gpu_tensor_upload_bf16");
#endif
#if !defined(ONLY) || ONLY == 352
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_tensor_upload_f32), int32_t (uint32_t, uint32_t, const float *, OmegaGpuTensor * *)), "signature of omega_gpu_tensor_upload_f32");
#endif
#if !defined(ONLY) || ONLY == 353
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_tensor_free), void (OmegaGpuTensor *)), "signature of omega_gpu_tensor_free");
#endif
#if !defined(ONLY) || ONLY == 354
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_tensor_shape), void (const OmegaGpuTensor *, uint32_t *, uint32_t *)), "signature of omega_gpu_tensor_shape");
#endif
#if !defined(ONLY) || ONLY == 355
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_matmul_resident_bf16), int32_t (uint32_t, const uint16_t *, const OmegaGpuTensor *, float *, OmegaGpuMatmulInfo *)), "signature of omega_gpu_matmul_resident_bf16");
#endif
#if !defined(ONLY) || ONLY == 356
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_matmul_resident_f32), int32_t (uint32_t, const float *, const OmegaGpuTensor *, float *, OmegaGpuMatmulInfo *)), "signature of omega_gpu_matmul_resident_f32");
#endif
#if !defined(ONLY) || ONLY == 357
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_matmul_last_error), const char * (void)), "signature of omega_gpu_matmul_last_error");
#endif
#if !defined(ONLY) || ONLY == 358
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_matmul_is_blocked), int32_t (void)), "signature of omega_gpu_matmul_is_blocked");
#endif
#if !defined(ONLY) || ONLY == 359
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_device_close), void (void)), "signature of omega_gpu_device_close");
#endif
#if !defined(ONLY) || ONLY == 360
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_matmul_set_cta_budget), void (uint32_t)), "signature of omega_gpu_matmul_set_cta_budget");
#endif
#if !defined(ONLY) || ONLY == 361
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_matmul_cta_budget), uint32_t (void)), "signature of omega_gpu_matmul_cta_budget");
#endif
#if !defined(ONLY) || ONLY == 362
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_set_spin_us), int32_t (uint32_t)), "signature of omega_gpu_session_set_spin_us");
#endif
#if !defined(ONLY) || ONLY == 363
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_spin_us), uint32_t (void)), "signature of omega_gpu_session_spin_us");
#endif
#if !defined(ONLY) || ONLY == 364
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_reserve_serving), int32_t (const OmegaGpuServingBounds *)), "signature of omega_gpu_reserve_serving");
#endif
#if !defined(ONLY) || ONLY == 365
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_serving_release), void (void)), "signature of omega_gpu_serving_release");
#endif
#if !defined(ONLY) || ONLY == 366
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_matmul_prepare), int32_t (uint32_t, uint32_t, uint32_t)), "signature of omega_gpu_matmul_prepare");
#endif
#if !defined(ONLY) || ONLY == 367
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_serving_seal), void (void)), "signature of omega_gpu_serving_seal");
#endif
#if !defined(ONLY) || ONLY == 368
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_session_alloc_stats), void (OmegaGpuAllocStats *)), "signature of omega_gpu_session_alloc_stats");
#endif
#if !defined(ONLY) || ONLY == 369
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_rmsnorm_f32), int32_t (uint32_t, uint32_t, const float *, const float *, float, float *, OmegaGpuEwInfo *)), "signature of omega_gpu_rmsnorm_f32");
#endif
#if !defined(ONLY) || ONLY == 370
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_rope_f32), int32_t (uint32_t, uint32_t, const float *, const float *, const float *, float *, OmegaGpuEwInfo *)), "signature of omega_gpu_rope_f32");
#endif
#if !defined(ONLY) || ONLY == 371
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_swiglu_f32), int32_t (uint32_t, const float *, const float *, float *, OmegaGpuEwInfo *)), "signature of omega_gpu_swiglu_f32");
#endif
#if !defined(ONLY) || ONLY == 372
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_elementwise_rc_name), const char * (int32_t)), "signature of omega_gpu_elementwise_rc_name");
#endif
#if !defined(ONLY) || ONLY == 373
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_gqa_attention_f32), int32_t (const float *, const float *, const float *, uint32_t, uint32_t, uint32_t, uint32_t, float *, OmegaGpuAttnInfo *)), "signature of omega_gpu_gqa_attention_f32");
#endif
#if !defined(ONLY) || ONLY == 374
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_gqa_attention_rows_f32), int32_t (const float *, const float *, const float *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float *, OmegaGpuAttnInfo *)), "signature of omega_gpu_gqa_attention_rows_f32");
#endif
#if !defined(ONLY) || ONLY == 375
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_attention_pool_alloc), int32_t (size_t, void * *)), "signature of omega_gpu_attention_pool_alloc");
#endif
#if !defined(ONLY) || ONLY == 376
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_attention_pool_free), void (void)), "signature of omega_gpu_attention_pool_free");
#endif
#if !defined(ONLY) || ONLY == 377
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_paged_attention_bf16), int32_t (const float *, const void *, const OmegaGpuKvLayout *, const uint32_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float *, OmegaGpuAttnInfo *)), "signature of omega_gpu_paged_attention_bf16");
#endif
#if !defined(ONLY) || ONLY == 378
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_paged_attention_batch_bf16), int32_t (const float *, const void *, const OmegaGpuKvLayout *, const int32_t *, const int32_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float *, OmegaGpuAttnInfo *)), "signature of omega_gpu_paged_attention_batch_bf16");
#endif
#if !defined(ONLY) || ONLY == 379
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_attention_rc_name), const char * (int32_t)), "signature of omega_gpu_attention_rc_name");
#endif
#if !defined(ONLY) || ONLY == 380
_Static_assert(__builtin_types_compatible_p(__typeof__(omega_gpu_attention_last_error), const char * (void)), "signature of omega_gpu_attention_last_error");
#endif
int main(void) {
  printf("const OMEGA_RUNTIME_API_VERSION_MAJOR %lld\n", (long long)(OMEGA_RUNTIME_API_VERSION_MAJOR));
  printf("const OMEGA_RUNTIME_API_VERSION_MINOR %lld\n", (long long)(OMEGA_RUNTIME_API_VERSION_MINOR));
#if !defined(ONLY) || ONLY == 1
  printf("const OMEGA_RUNTIME_API_VERSION_MAJOR %lld\n", (long long)(OMEGA_RUNTIME_API_VERSION_MAJOR));
#endif
#if !defined(ONLY) || ONLY == 2
  printf("const OMEGA_RUNTIME_API_VERSION_MINOR %lld\n", (long long)(OMEGA_RUNTIME_API_VERSION_MINOR));
#endif
#if !defined(ONLY) || ONLY == 3
  printf("struct OmegaGpuDeviceHandle %zu %zu\n", sizeof(OmegaGpuDeviceHandle), _Alignof(OmegaGpuDeviceHandle));
#endif
#if !defined(ONLY) || ONLY == 4
  printf("field OmegaGpuDeviceHandle index %zu %zu\n", offsetof(OmegaGpuDeviceHandle, index), sizeof(((OmegaGpuDeviceHandle *)0)->index));
#endif
#if !defined(ONLY) || ONLY == 5
  printf("field OmegaGpuDeviceHandle generation %zu %zu\n", offsetof(OmegaGpuDeviceHandle, generation), sizeof(((OmegaGpuDeviceHandle *)0)->generation));
#endif
#if !defined(ONLY) || ONLY == 6
  printf("struct OmegaGpuBufferHandle %zu %zu\n", sizeof(OmegaGpuBufferHandle), _Alignof(OmegaGpuBufferHandle));
#endif
#if !defined(ONLY) || ONLY == 7
  printf("field OmegaGpuBufferHandle index %zu %zu\n", offsetof(OmegaGpuBufferHandle, index), sizeof(((OmegaGpuBufferHandle *)0)->index));
#endif
#if !defined(ONLY) || ONLY == 8
  printf("field OmegaGpuBufferHandle generation %zu %zu\n", offsetof(OmegaGpuBufferHandle, generation), sizeof(((OmegaGpuBufferHandle *)0)->generation));
#endif
#if !defined(ONLY) || ONLY == 9
  printf("struct OmegaGpuQueueHandle %zu %zu\n", sizeof(OmegaGpuQueueHandle), _Alignof(OmegaGpuQueueHandle));
#endif
#if !defined(ONLY) || ONLY == 10
  printf("field OmegaGpuQueueHandle index %zu %zu\n", offsetof(OmegaGpuQueueHandle, index), sizeof(((OmegaGpuQueueHandle *)0)->index));
#endif
#if !defined(ONLY) || ONLY == 11
  printf("field OmegaGpuQueueHandle generation %zu %zu\n", offsetof(OmegaGpuQueueHandle, generation), sizeof(((OmegaGpuQueueHandle *)0)->generation));
#endif
#if !defined(ONLY) || ONLY == 12
  printf("struct OmegaCpuView %zu %zu\n", sizeof(OmegaCpuView), _Alignof(OmegaCpuView));
#endif
#if !defined(ONLY) || ONLY == 13
  printf("field OmegaCpuView ptr %zu %zu\n", offsetof(OmegaCpuView, ptr), sizeof(((OmegaCpuView *)0)->ptr));
#endif
#if !defined(ONLY) || ONLY == 14
  printf("field OmegaCpuView len %zu %zu\n", offsetof(OmegaCpuView, len), sizeof(((OmegaCpuView *)0)->len));
#endif
#if !defined(ONLY) || ONLY == 15
  printf("struct OmegaGpuVa %zu %zu\n", sizeof(OmegaGpuVa), _Alignof(OmegaGpuVa));
#endif
#if !defined(ONLY) || ONLY == 16
  printf("field OmegaGpuVa va %zu %zu\n", offsetof(OmegaGpuVa, va), sizeof(((OmegaGpuVa *)0)->va));
#endif
#if !defined(ONLY) || ONLY == 17
  printf("struct OmegaGpuBusAddr %zu %zu\n", sizeof(OmegaGpuBusAddr), _Alignof(OmegaGpuBusAddr));
#endif
#if !defined(ONLY) || ONLY == 18
  printf("field OmegaGpuBusAddr iova %zu %zu\n", offsetof(OmegaGpuBusAddr, iova), sizeof(((OmegaGpuBusAddr *)0)->iova));
#endif
#if !defined(ONLY) || ONLY == 19
  printf("struct OmegaGpuBufferDesc %zu %zu\n", sizeof(OmegaGpuBufferDesc), _Alignof(OmegaGpuBufferDesc));
#endif
#if !defined(ONLY) || ONLY == 20
  printf("field OmegaGpuBufferDesc kind %zu %zu\n", offsetof(OmegaGpuBufferDesc, kind), sizeof(((OmegaGpuBufferDesc *)0)->kind));
#endif
#if !defined(ONLY) || ONLY == 21
  printf("field OmegaGpuBufferDesc rights %zu %zu\n", offsetof(OmegaGpuBufferDesc, rights), sizeof(((OmegaGpuBufferDesc *)0)->rights));
#endif
#if !defined(ONLY) || ONLY == 22
  printf("field OmegaGpuBufferDesc gpu_cache %zu %zu\n", offsetof(OmegaGpuBufferDesc, gpu_cache), sizeof(((OmegaGpuBufferDesc *)0)->gpu_cache));
#endif
#if !defined(ONLY) || ONLY == 23
  printf("field OmegaGpuBufferDesc align %zu %zu\n", offsetof(OmegaGpuBufferDesc, align), sizeof(((OmegaGpuBufferDesc *)0)->align));
#endif
#if !defined(ONLY) || ONLY == 24
  printf("field OmegaGpuBufferDesc bytes %zu %zu\n", offsetof(OmegaGpuBufferDesc, bytes), sizeof(((OmegaGpuBufferDesc *)0)->bytes));
#endif
#if !defined(ONLY) || ONLY == 25
  printf("struct OmegaGpuPlatformCaps %zu %zu\n", sizeof(OmegaGpuPlatformCaps), _Alignof(OmegaGpuPlatformCaps));
#endif
#if !defined(ONLY) || ONLY == 26
  printf("field OmegaGpuPlatformCaps abi_major %zu %zu\n", offsetof(OmegaGpuPlatformCaps, abi_major), sizeof(((OmegaGpuPlatformCaps *)0)->abi_major));
#endif
#if !defined(ONLY) || ONLY == 27
  printf("field OmegaGpuPlatformCaps abi_minor %zu %zu\n", offsetof(OmegaGpuPlatformCaps, abi_minor), sizeof(((OmegaGpuPlatformCaps *)0)->abi_minor));
#endif
#if !defined(ONLY) || ONLY == 28
  printf("field OmegaGpuPlatformCaps services %zu %zu\n", offsetof(OmegaGpuPlatformCaps, services), sizeof(((OmegaGpuPlatformCaps *)0)->services));
#endif
#if !defined(ONLY) || ONLY == 29
  printf("field OmegaGpuPlatformCaps max_devices %zu %zu\n", offsetof(OmegaGpuPlatformCaps, max_devices), sizeof(((OmegaGpuPlatformCaps *)0)->max_devices));
#endif
#if !defined(ONLY) || ONLY == 30
  printf("field OmegaGpuPlatformCaps max_buffers %zu %zu\n", offsetof(OmegaGpuPlatformCaps, max_buffers), sizeof(((OmegaGpuPlatformCaps *)0)->max_buffers));
#endif
#if !defined(ONLY) || ONLY == 31
  printf("field OmegaGpuPlatformCaps max_queues %zu %zu\n", offsetof(OmegaGpuPlatformCaps, max_queues), sizeof(((OmegaGpuPlatformCaps *)0)->max_queues));
#endif
#if !defined(ONLY) || ONLY == 32
  printf("field OmegaGpuPlatformCaps max_inflight %zu %zu\n", offsetof(OmegaGpuPlatformCaps, max_inflight), sizeof(((OmegaGpuPlatformCaps *)0)->max_inflight));
#endif
#if !defined(ONLY) || ONLY == 33
  printf("field OmegaGpuPlatformCaps max_buffer_bytes %zu %zu\n", offsetof(OmegaGpuPlatformCaps, max_buffer_bytes), sizeof(((OmegaGpuPlatformCaps *)0)->max_buffer_bytes));
#endif
#if !defined(ONLY) || ONLY == 34
  printf("field OmegaGpuPlatformCaps max_total_bytes %zu %zu\n", offsetof(OmegaGpuPlatformCaps, max_total_bytes), sizeof(((OmegaGpuPlatformCaps *)0)->max_total_bytes));
#endif
#if !defined(ONLY) || ONLY == 35
  printf("field OmegaGpuPlatformCaps submit_prefix_words %zu %zu\n", offsetof(OmegaGpuPlatformCaps, submit_prefix_words), sizeof(((OmegaGpuPlatformCaps *)0)->submit_prefix_words));
#endif
#if !defined(ONLY) || ONLY == 36
  printf("struct OmegaGpuPlatform %zu %zu\n", sizeof(OmegaGpuPlatform), _Alignof(OmegaGpuPlatform));
#endif
#if !defined(ONLY) || ONLY == 37
  printf("field OmegaGpuPlatform ctx %zu %zu\n", offsetof(OmegaGpuPlatform, ctx), sizeof(((OmegaGpuPlatform *)0)->ctx));
#endif
#if !defined(ONLY) || ONLY == 38
  printf("field OmegaGpuPlatform time_now_ms %zu %zu\n", offsetof(OmegaGpuPlatform, time_now_ms), sizeof(((OmegaGpuPlatform *)0)->time_now_ms));
#endif
#if !defined(ONLY) || ONLY == 39
  printf("field OmegaGpuPlatform wait_ms %zu %zu\n", offsetof(OmegaGpuPlatform, wait_ms), sizeof(((OmegaGpuPlatform *)0)->wait_ms));
#endif
#if !defined(ONLY) || ONLY == 40
  printf("field OmegaGpuPlatform lock %zu %zu\n", offsetof(OmegaGpuPlatform, lock), sizeof(((OmegaGpuPlatform *)0)->lock));
#endif
#if !defined(ONLY) || ONLY == 41
  printf("field OmegaGpuPlatform unlock %zu %zu\n", offsetof(OmegaGpuPlatform, unlock), sizeof(((OmegaGpuPlatform *)0)->unlock));
#endif
#if !defined(ONLY) || ONLY == 42
  printf("field OmegaGpuPlatform mmio_read32 %zu %zu\n", offsetof(OmegaGpuPlatform, mmio_read32), sizeof(((OmegaGpuPlatform *)0)->mmio_read32));
#endif
#if !defined(ONLY) || ONLY == 43
  printf("field OmegaGpuPlatform mmio_write32 %zu %zu\n", offsetof(OmegaGpuPlatform, mmio_write32), sizeof(((OmegaGpuPlatform *)0)->mmio_write32));
#endif
#if !defined(ONLY) || ONLY == 44
  printf("field OmegaGpuPlatform dma_alloc %zu %zu\n", offsetof(OmegaGpuPlatform, dma_alloc), sizeof(((OmegaGpuPlatform *)0)->dma_alloc));
#endif
#if !defined(ONLY) || ONLY == 45
  printf("field OmegaGpuPlatform dma_free %zu %zu\n", offsetof(OmegaGpuPlatform, dma_free), sizeof(((OmegaGpuPlatform *)0)->dma_free));
#endif
#if !defined(ONLY) || ONLY == 46
  printf("field OmegaGpuPlatform sync_cpu_to_gpu %zu %zu\n", offsetof(OmegaGpuPlatform, sync_cpu_to_gpu), sizeof(((OmegaGpuPlatform *)0)->sync_cpu_to_gpu));
#endif
#if !defined(ONLY) || ONLY == 47
  printf("field OmegaGpuPlatform sync_gpu_to_cpu %zu %zu\n", offsetof(OmegaGpuPlatform, sync_gpu_to_cpu), sizeof(((OmegaGpuPlatform *)0)->sync_gpu_to_cpu));
#endif
#if !defined(ONLY) || ONLY == 48
  printf("field OmegaGpuPlatform poll_word %zu %zu\n", offsetof(OmegaGpuPlatform, poll_word), sizeof(((OmegaGpuPlatform *)0)->poll_word));
#endif
#if !defined(ONLY) || ONLY == 49
  printf("field OmegaGpuPlatform irq_wait %zu %zu\n", offsetof(OmegaGpuPlatform, irq_wait), sizeof(((OmegaGpuPlatform *)0)->irq_wait));
#endif
#if !defined(ONLY) || ONLY == 50
  printf("field OmegaGpuPlatform device_open %zu %zu\n", offsetof(OmegaGpuPlatform, device_open), sizeof(((OmegaGpuPlatform *)0)->device_open));
#endif
#if !defined(ONLY) || ONLY == 51
  printf("field OmegaGpuPlatform device_close %zu %zu\n", offsetof(OmegaGpuPlatform, device_close), sizeof(((OmegaGpuPlatform *)0)->device_close));
#endif
#if !defined(ONLY) || ONLY == 52
  printf("field OmegaGpuPlatform diag_text %zu %zu\n", offsetof(OmegaGpuPlatform, diag_text), sizeof(((OmegaGpuPlatform *)0)->diag_text));
#endif
#if !defined(ONLY) || ONLY == 53
  printf("field OmegaGpuPlatform queue_open %zu %zu\n", offsetof(OmegaGpuPlatform, queue_open), sizeof(((OmegaGpuPlatform *)0)->queue_open));
#endif
#if !defined(ONLY) || ONLY == 54
  printf("field OmegaGpuPlatform queue_close %zu %zu\n", offsetof(OmegaGpuPlatform, queue_close), sizeof(((OmegaGpuPlatform *)0)->queue_close));
#endif
#if !defined(ONLY) || ONLY == 55
  printf("field OmegaGpuPlatform queue_submit %zu %zu\n", offsetof(OmegaGpuPlatform, queue_submit), sizeof(((OmegaGpuPlatform *)0)->queue_submit));
#endif
#if !defined(ONLY) || ONLY == 56
  printf("field OmegaGpuPlatform queue_retire %zu %zu\n", offsetof(OmegaGpuPlatform, queue_retire), sizeof(((OmegaGpuPlatform *)0)->queue_retire));
#endif
#if !defined(ONLY) || ONLY == 57
  printf("field OmegaGpuPlatform time_now_ns %zu %zu\n", offsetof(OmegaGpuPlatform, time_now_ns), sizeof(((OmegaGpuPlatform *)0)->time_now_ns));
#endif
#if !defined(ONLY) || ONLY == 58
  printf("field OmegaGpuPlatform host_alloc %zu %zu\n", offsetof(OmegaGpuPlatform, host_alloc), sizeof(((OmegaGpuPlatform *)0)->host_alloc));
#endif
#if !defined(ONLY) || ONLY == 59
  printf("field OmegaGpuPlatform host_free %zu %zu\n", offsetof(OmegaGpuPlatform, host_free), sizeof(((OmegaGpuPlatform *)0)->host_free));
#endif
#if !defined(ONLY) || ONLY == 60
  printf("struct OmegaGpuHandleTable %zu %zu\n", sizeof(OmegaGpuHandleTable), _Alignof(OmegaGpuHandleTable));
#endif
#if !defined(ONLY) || ONLY == 61
  printf("field OmegaGpuHandleTable generation %zu %zu\n", offsetof(OmegaGpuHandleTable, generation), sizeof(((OmegaGpuHandleTable *)0)->generation));
#endif
#if !defined(ONLY) || ONLY == 62
  printf("field OmegaGpuHandleTable kind %zu %zu\n", offsetof(OmegaGpuHandleTable, kind), sizeof(((OmegaGpuHandleTable *)0)->kind));
#endif
#if !defined(ONLY) || ONLY == 63
  printf("field OmegaGpuHandleTable bytes %zu %zu\n", offsetof(OmegaGpuHandleTable, bytes), sizeof(((OmegaGpuHandleTable *)0)->bytes));
#endif
#if !defined(ONLY) || ONLY == 64
  printf("field OmegaGpuHandleTable live %zu %zu\n", offsetof(OmegaGpuHandleTable, live), sizeof(((OmegaGpuHandleTable *)0)->live));
#endif
#if !defined(ONLY) || ONLY == 65
  printf("field OmegaGpuHandleTable nlive %zu %zu\n", offsetof(OmegaGpuHandleTable, nlive), sizeof(((OmegaGpuHandleTable *)0)->nlive));
#endif
#if !defined(ONLY) || ONLY == 66
  printf("field OmegaGpuHandleTable total_bytes %zu %zu\n", offsetof(OmegaGpuHandleTable, total_bytes), sizeof(((OmegaGpuHandleTable *)0)->total_bytes));
#endif
#if !defined(ONLY) || ONLY == 76
  printf("struct OmegaGpuMem %zu %zu\n", sizeof(OmegaGpuMem), _Alignof(OmegaGpuMem));
#endif
#if !defined(ONLY) || ONLY == 77
  printf("field OmegaGpuMem handle %zu %zu\n", offsetof(OmegaGpuMem, handle), sizeof(((OmegaGpuMem *)0)->handle));
#endif
#if !defined(ONLY) || ONLY == 78
  printf("field OmegaGpuMem kind %zu %zu\n", offsetof(OmegaGpuMem, kind), sizeof(((OmegaGpuMem *)0)->kind));
#endif
#if !defined(ONLY) || ONLY == 79
  printf("field OmegaGpuMem cpu %zu %zu\n", offsetof(OmegaGpuMem, cpu), sizeof(((OmegaGpuMem *)0)->cpu));
#endif
#if !defined(ONLY) || ONLY == 80
  printf("field OmegaGpuMem va %zu %zu\n", offsetof(OmegaGpuMem, va), sizeof(((OmegaGpuMem *)0)->va));
#endif
#if !defined(ONLY) || ONLY == 81
  printf("field OmegaGpuMem size %zu %zu\n", offsetof(OmegaGpuMem, size), sizeof(((OmegaGpuMem *)0)->size));
#endif
#if !defined(ONLY) || ONLY == 82
  printf("struct OmegaGpuSession %zu %zu\n", sizeof(OmegaGpuSession), _Alignof(OmegaGpuSession));
#endif
#if !defined(ONLY) || ONLY == 83
  printf("field OmegaGpuSession dev %zu %zu\n", offsetof(OmegaGpuSession, dev), sizeof(((OmegaGpuSession *)0)->dev));
#endif
#if !defined(ONLY) || ONLY == 84
  printf("field OmegaGpuSession queue %zu %zu\n", offsetof(OmegaGpuSession, queue), sizeof(((OmegaGpuSession *)0)->queue));
#endif
#if !defined(ONLY) || ONLY == 85
  printf("field OmegaGpuSession pb %zu %zu\n", offsetof(OmegaGpuSession, pb), sizeof(((OmegaGpuSession *)0)->pb));
#endif
#if !defined(ONLY) || ONLY == 86
  printf("field OmegaGpuSession cbank %zu %zu\n", offsetof(OmegaGpuSession, cbank), sizeof(((OmegaGpuSession *)0)->cbank));
#endif
#if !defined(ONLY) || ONLY == 87
  printf("field OmegaGpuSession marker %zu %zu\n", offsetof(OmegaGpuSession, marker), sizeof(((OmegaGpuSession *)0)->marker));
#endif
#if !defined(ONLY) || ONLY == 88
  printf("field OmegaGpuSession qmd %zu %zu\n", offsetof(OmegaGpuSession, qmd), sizeof(((OmegaGpuSession *)0)->qmd));
#endif
#if !defined(ONLY) || ONLY == 89
  printf("field OmegaGpuSession opens %zu %zu\n", offsetof(OmegaGpuSession, opens), sizeof(((OmegaGpuSession *)0)->opens));
#endif
#if !defined(ONLY) || ONLY == 90
  printf("field OmegaGpuSession last_open_ns %zu %zu\n", offsetof(OmegaGpuSession, last_open_ns), sizeof(((OmegaGpuSession *)0)->last_open_ns));
#endif
#if !defined(ONLY) || ONLY == 91
  printf("struct OmegaGpuScratch %zu %zu\n", sizeof(OmegaGpuScratch), _Alignof(OmegaGpuScratch));
#endif
#if !defined(ONLY) || ONLY == 92
  printf("field OmegaGpuScratch mem %zu %zu\n", offsetof(OmegaGpuScratch, mem), sizeof(((OmegaGpuScratch *)0)->mem));
#endif
#if !defined(ONLY) || ONLY == 93
  printf("field OmegaGpuScratch cap %zu %zu\n", offsetof(OmegaGpuScratch, cap), sizeof(((OmegaGpuScratch *)0)->cap));
#endif
#if !defined(ONLY) || ONLY == 94
  printf("field OmegaGpuScratch high_water %zu %zu\n", offsetof(OmegaGpuScratch, high_water), sizeof(((OmegaGpuScratch *)0)->high_water));
#endif
#if !defined(ONLY) || ONLY == 95
  printf("field OmegaGpuScratch reserved %zu %zu\n", offsetof(OmegaGpuScratch, reserved), sizeof(((OmegaGpuScratch *)0)->reserved));
#endif
#if !defined(ONLY) || ONLY == 96
  printf("struct OmegaGpuLaunch %zu %zu\n", sizeof(OmegaGpuLaunch), _Alignof(OmegaGpuLaunch));
#endif
#if !defined(ONLY) || ONLY == 97
  printf("field OmegaGpuLaunch code_va %zu %zu\n", offsetof(OmegaGpuLaunch, code_va), sizeof(((OmegaGpuLaunch *)0)->code_va));
#endif
#if !defined(ONLY) || ONLY == 98
  printf("field OmegaGpuLaunch gpr_count %zu %zu\n", offsetof(OmegaGpuLaunch, gpr_count), sizeof(((OmegaGpuLaunch *)0)->gpr_count));
#endif
#if !defined(ONLY) || ONLY == 99
  printf("field OmegaGpuLaunch threads_x %zu %zu\n", offsetof(OmegaGpuLaunch, threads_x), sizeof(((OmegaGpuLaunch *)0)->threads_x));
#endif
#if !defined(ONLY) || ONLY == 100
  printf("field OmegaGpuLaunch threads_y %zu %zu\n", offsetof(OmegaGpuLaunch, threads_y), sizeof(((OmegaGpuLaunch *)0)->threads_y));
#endif
#if !defined(ONLY) || ONLY == 101
  printf("field OmegaGpuLaunch grid_x %zu %zu\n", offsetof(OmegaGpuLaunch, grid_x), sizeof(((OmegaGpuLaunch *)0)->grid_x));
#endif
#if !defined(ONLY) || ONLY == 102
  printf("field OmegaGpuLaunch grid_y %zu %zu\n", offsetof(OmegaGpuLaunch, grid_y), sizeof(((OmegaGpuLaunch *)0)->grid_y));
#endif
#if !defined(ONLY) || ONLY == 103
  printf("field OmegaGpuLaunch num_elements %zu %zu\n", offsetof(OmegaGpuLaunch, num_elements), sizeof(((OmegaGpuLaunch *)0)->num_elements));
#endif
#if !defined(ONLY) || ONLY == 104
  printf("field OmegaGpuLaunch args %zu %zu\n", offsetof(OmegaGpuLaunch, args), sizeof(((OmegaGpuLaunch *)0)->args));
#endif
#if !defined(ONLY) || ONLY == 105
  printf("field OmegaGpuLaunch n_args %zu %zu\n", offsetof(OmegaGpuLaunch, n_args), sizeof(((OmegaGpuLaunch *)0)->n_args));
#endif
#if !defined(ONLY) || ONLY == 106
  printf("field OmegaGpuLaunch timeout_ms %zu %zu\n", offsetof(OmegaGpuLaunch, timeout_ms), sizeof(((OmegaGpuLaunch *)0)->timeout_ms));
#endif
#if !defined(ONLY) || ONLY == 107
  printf("field OmegaGpuLaunch shared_bytes %zu %zu\n", offsetof(OmegaGpuLaunch, shared_bytes), sizeof(((OmegaGpuLaunch *)0)->shared_bytes));
#endif
#if !defined(ONLY) || ONLY == 108
  printf("field OmegaGpuLaunch spin_us %zu %zu\n", offsetof(OmegaGpuLaunch, spin_us), sizeof(((OmegaGpuLaunch *)0)->spin_us));
#endif
#if !defined(ONLY) || ONLY == 109
  printf("struct OmegaGpuParkStats %zu %zu\n", sizeof(OmegaGpuParkStats), _Alignof(OmegaGpuParkStats));
#endif
#if !defined(ONLY) || ONLY == 110
  printf("field OmegaGpuParkStats buffers %zu %zu\n", offsetof(OmegaGpuParkStats, buffers), sizeof(((OmegaGpuParkStats *)0)->buffers));
#endif
#if !defined(ONLY) || ONLY == 111
  printf("field OmegaGpuParkStats untracked %zu %zu\n", offsetof(OmegaGpuParkStats, untracked), sizeof(((OmegaGpuParkStats *)0)->untracked));
#endif
#if !defined(ONLY) || ONLY == 112
  printf("field OmegaGpuParkStats bytes %zu %zu\n", offsetof(OmegaGpuParkStats, bytes), sizeof(((OmegaGpuParkStats *)0)->bytes));
#endif
#if !defined(ONLY) || ONLY == 143
  printf("struct OmegaGpuCmdListOpts %zu %zu\n", sizeof(OmegaGpuCmdListOpts), _Alignof(OmegaGpuCmdListOpts));
#endif
#if !defined(ONLY) || ONLY == 144
  printf("field OmegaGpuCmdListOpts mode %zu %zu\n", offsetof(OmegaGpuCmdListOpts, mode), sizeof(((OmegaGpuCmdListOpts *)0)->mode));
#endif
#if !defined(ONLY) || ONLY == 145
  printf("field OmegaGpuCmdListOpts barrier %zu %zu\n", offsetof(OmegaGpuCmdListOpts, barrier), sizeof(((OmegaGpuCmdListOpts *)0)->barrier));
#endif
#if !defined(ONLY) || ONLY == 146
  printf("field OmegaGpuCmdListOpts timeout_ms %zu %zu\n", offsetof(OmegaGpuCmdListOpts, timeout_ms), sizeof(((OmegaGpuCmdListOpts *)0)->timeout_ms));
#endif
#if !defined(ONLY) || ONLY == 147
  printf("field OmegaGpuCmdListOpts spin_us %zu %zu\n", offsetof(OmegaGpuCmdListOpts, spin_us), sizeof(((OmegaGpuCmdListOpts *)0)->spin_us));
#endif
#if !defined(ONLY) || ONLY == 148
  printf("field OmegaGpuCmdListOpts stamps %zu %zu\n", offsetof(OmegaGpuCmdListOpts, stamps), sizeof(((OmegaGpuCmdListOpts *)0)->stamps));
#endif
#if !defined(ONLY) || ONLY == 149
  printf("field OmegaGpuCmdListOpts skip_launch %zu %zu\n", offsetof(OmegaGpuCmdListOpts, skip_launch), sizeof(((OmegaGpuCmdListOpts *)0)->skip_launch));
#endif
#if !defined(ONLY) || ONLY == 150
  printf("struct OmegaGpuCmdListStats %zu %zu\n", sizeof(OmegaGpuCmdListStats), _Alignof(OmegaGpuCmdListStats));
#endif
#if !defined(ONLY) || ONLY == 151
  printf("field OmegaGpuCmdListStats launches %zu %zu\n", offsetof(OmegaGpuCmdListStats, launches), sizeof(((OmegaGpuCmdListStats *)0)->launches));
#endif
#if !defined(ONLY) || ONLY == 152
  printf("field OmegaGpuCmdListStats submits %zu %zu\n", offsetof(OmegaGpuCmdListStats, submits), sizeof(((OmegaGpuCmdListStats *)0)->submits));
#endif
#if !defined(ONLY) || ONLY == 153
  printf("field OmegaGpuCmdListStats words %zu %zu\n", offsetof(OmegaGpuCmdListStats, words), sizeof(((OmegaGpuCmdListStats *)0)->words));
#endif
#if !defined(ONLY) || ONLY == 154
  printf("field OmegaGpuCmdListStats elapsed_ns %zu %zu\n", offsetof(OmegaGpuCmdListStats, elapsed_ns), sizeof(((OmegaGpuCmdListStats *)0)->elapsed_ns));
#endif
#if !defined(ONLY) || ONLY == 155
  printf("field OmegaGpuCmdListStats host_ns %zu %zu\n", offsetof(OmegaGpuCmdListStats, host_ns), sizeof(((OmegaGpuCmdListStats *)0)->host_ns));
#endif
#if !defined(ONLY) || ONLY == 156
  printf("field OmegaGpuCmdListStats sems_done %zu %zu\n", offsetof(OmegaGpuCmdListStats, sems_done), sizeof(((OmegaGpuCmdListStats *)0)->sems_done));
#endif
#if !defined(ONLY) || ONLY == 157
  printf("field OmegaGpuCmdListStats first_missing %zu %zu\n", offsetof(OmegaGpuCmdListStats, first_missing), sizeof(((OmegaGpuCmdListStats *)0)->first_missing));
#endif
#if !defined(ONLY) || ONLY == 168
  printf("struct OmegaGpuTimeouts %zu %zu\n", sizeof(OmegaGpuTimeouts), _Alignof(OmegaGpuTimeouts));
#endif
#if !defined(ONLY) || ONLY == 169
  printf("field OmegaGpuTimeouts marker_ms %zu %zu\n", offsetof(OmegaGpuTimeouts, marker_ms), sizeof(((OmegaGpuTimeouts *)0)->marker_ms));
#endif
#if !defined(ONLY) || ONLY == 170
  printf("field OmegaGpuTimeouts release_semaphore_ms %zu %zu\n", offsetof(OmegaGpuTimeouts, release_semaphore_ms), sizeof(((OmegaGpuTimeouts *)0)->release_semaphore_ms));
#endif
#if !defined(ONLY) || ONLY == 171
  printf("field OmegaGpuTimeouts marker2_ms %zu %zu\n", offsetof(OmegaGpuTimeouts, marker2_ms), sizeof(((OmegaGpuTimeouts *)0)->marker2_ms));
#endif
#if !defined(ONLY) || ONLY == 172
  printf("field OmegaGpuTimeouts visibility_ms %zu %zu\n", offsetof(OmegaGpuTimeouts, visibility_ms), sizeof(((OmegaGpuTimeouts *)0)->visibility_ms));
#endif
#if !defined(ONLY) || ONLY == 173
  printf("struct OmegaGpuJob %zu %zu\n", sizeof(OmegaGpuJob), _Alignof(OmegaGpuJob));
#endif
#if !defined(ONLY) || ONLY == 174
  printf("field OmegaGpuJob program %zu %zu\n", offsetof(OmegaGpuJob, program), sizeof(((OmegaGpuJob *)0)->program));
#endif
#if !defined(ONLY) || ONLY == 175
  printf("field OmegaGpuJob program_len %zu %zu\n", offsetof(OmegaGpuJob, program_len), sizeof(((OmegaGpuJob *)0)->program_len));
#endif
#if !defined(ONLY) || ONLY == 176
  printf("field OmegaGpuJob layout %zu %zu\n", offsetof(OmegaGpuJob, layout), sizeof(((OmegaGpuJob *)0)->layout));
#endif
#if !defined(ONLY) || ONLY == 177
  printf("field OmegaGpuJob element_count %zu %zu\n", offsetof(OmegaGpuJob, element_count), sizeof(((OmegaGpuJob *)0)->element_count));
#endif
#if !defined(ONLY) || ONLY == 178
  printf("field OmegaGpuJob input_a %zu %zu\n", offsetof(OmegaGpuJob, input_a), sizeof(((OmegaGpuJob *)0)->input_a));
#endif
#if !defined(ONLY) || ONLY == 179
  printf("field OmegaGpuJob input_a_len %zu %zu\n", offsetof(OmegaGpuJob, input_a_len), sizeof(((OmegaGpuJob *)0)->input_a_len));
#endif
#if !defined(ONLY) || ONLY == 180
  printf("field OmegaGpuJob input_b %zu %zu\n", offsetof(OmegaGpuJob, input_b), sizeof(((OmegaGpuJob *)0)->input_b));
#endif
#if !defined(ONLY) || ONLY == 181
  printf("field OmegaGpuJob input_b_len %zu %zu\n", offsetof(OmegaGpuJob, input_b_len), sizeof(((OmegaGpuJob *)0)->input_b_len));
#endif
#if !defined(ONLY) || ONLY == 182
  printf("field OmegaGpuJob output %zu %zu\n", offsetof(OmegaGpuJob, output), sizeof(((OmegaGpuJob *)0)->output));
#endif
#if !defined(ONLY) || ONLY == 183
  printf("field OmegaGpuJob output_len %zu %zu\n", offsetof(OmegaGpuJob, output_len), sizeof(((OmegaGpuJob *)0)->output_len));
#endif
#if !defined(ONLY) || ONLY == 184
  printf("field OmegaGpuJob poison %zu %zu\n", offsetof(OmegaGpuJob, poison), sizeof(((OmegaGpuJob *)0)->poison));
#endif
#if !defined(ONLY) || ONLY == 185
  printf("field OmegaGpuJob poison_count %zu %zu\n", offsetof(OmegaGpuJob, poison_count), sizeof(((OmegaGpuJob *)0)->poison_count));
#endif
#if !defined(ONLY) || ONLY == 186
  printf("field OmegaGpuJob timeouts %zu %zu\n", offsetof(OmegaGpuJob, timeouts), sizeof(((OmegaGpuJob *)0)->timeouts));
#endif
#if !defined(ONLY) || ONLY == 187
  printf("field OmegaGpuJob flags %zu %zu\n", offsetof(OmegaGpuJob, flags), sizeof(((OmegaGpuJob *)0)->flags));
#endif
#if !defined(ONLY) || ONLY == 188
  printf("struct OmegaGpuResult %zu %zu\n", sizeof(OmegaGpuResult), _Alignof(OmegaGpuResult));
#endif
#if !defined(ONLY) || ONLY == 189
  printf("field OmegaGpuResult failure %zu %zu\n", offsetof(OmegaGpuResult, failure), sizeof(((OmegaGpuResult *)0)->failure));
#endif
#if !defined(ONLY) || ONLY == 190
  printf("field OmegaGpuResult last_state %zu %zu\n", offsetof(OmegaGpuResult, last_state), sizeof(((OmegaGpuResult *)0)->last_state));
#endif
#if !defined(ONLY) || ONLY == 191
  printf("field OmegaGpuResult failed_step %zu %zu\n", offsetof(OmegaGpuResult, failed_step), sizeof(((OmegaGpuResult *)0)->failed_step));
#endif
#if !defined(ONLY) || ONLY == 192
  printf("field OmegaGpuResult wait %zu %zu\n", offsetof(OmegaGpuResult, wait), sizeof(((OmegaGpuResult *)0)->wait));
#endif
#if !defined(ONLY) || ONLY == 193
  printf("field OmegaGpuResult wait_timeout_ms %zu %zu\n", offsetof(OmegaGpuResult, wait_timeout_ms), sizeof(((OmegaGpuResult *)0)->wait_timeout_ms));
#endif
#if !defined(ONLY) || ONLY == 194
  printf("field OmegaGpuResult waited_ms %zu %zu\n", offsetof(OmegaGpuResult, waited_ms), sizeof(((OmegaGpuResult *)0)->waited_ms));
#endif
#if !defined(ONLY) || ONLY == 195
  printf("field OmegaGpuResult drv_rc %zu %zu\n", offsetof(OmegaGpuResult, drv_rc), sizeof(((OmegaGpuResult *)0)->drv_rc));
#endif
#if !defined(ONLY) || ONLY == 196
  printf("field OmegaGpuResult err_no %zu %zu\n", offsetof(OmegaGpuResult, err_no), sizeof(((OmegaGpuResult *)0)->err_no));
#endif
#if !defined(ONLY) || ONLY == 197
  printf("field OmegaGpuResult drv_text %zu %zu\n", offsetof(OmegaGpuResult, drv_text), sizeof(((OmegaGpuResult *)0)->drv_text));
#endif
#if !defined(ONLY) || ONLY == 198
  printf("field OmegaGpuResult sync_valid %zu %zu\n", offsetof(OmegaGpuResult, sync_valid), sizeof(((OmegaGpuResult *)0)->sync_valid));
#endif
#if !defined(ONLY) || ONLY == 199
  printf("field OmegaGpuResult marker %zu %zu\n", offsetof(OmegaGpuResult, marker), sizeof(((OmegaGpuResult *)0)->marker));
#endif
#if !defined(ONLY) || ONLY == 200
  printf("field OmegaGpuResult marker2 %zu %zu\n", offsetof(OmegaGpuResult, marker2), sizeof(((OmegaGpuResult *)0)->marker2));
#endif
#if !defined(ONLY) || ONLY == 201
  printf("field OmegaGpuResult semaphore %zu %zu\n", offsetof(OmegaGpuResult, semaphore), sizeof(((OmegaGpuResult *)0)->semaphore));
#endif
#if !defined(ONLY) || ONLY == 202
  printf("field OmegaGpuResult retained %zu %zu\n", offsetof(OmegaGpuResult, retained), sizeof(((OmegaGpuResult *)0)->retained));
#endif
#if !defined(ONLY) || ONLY == 203
  printf("field OmegaGpuResult cleanup_failed %zu %zu\n", offsetof(OmegaGpuResult, cleanup_failed), sizeof(((OmegaGpuResult *)0)->cleanup_failed));
#endif
#if !defined(ONLY) || ONLY == 204
  printf("field OmegaGpuResult n_outputs %zu %zu\n", offsetof(OmegaGpuResult, n_outputs), sizeof(((OmegaGpuResult *)0)->n_outputs));
#endif
#if !defined(ONLY) || ONLY == 205
  printf("field OmegaGpuResult output_unchanged_words %zu %zu\n", offsetof(OmegaGpuResult, output_unchanged_words), sizeof(((OmegaGpuResult *)0)->output_unchanged_words));
#endif
#if !defined(ONLY) || ONLY == 206
  printf("struct OmegaGpuBackendDiag %zu %zu\n", sizeof(OmegaGpuBackendDiag), _Alignof(OmegaGpuBackendDiag));
#endif
#if !defined(ONLY) || ONLY == 207
  printf("field OmegaGpuBackendDiag drv_rc %zu %zu\n", offsetof(OmegaGpuBackendDiag, drv_rc), sizeof(((OmegaGpuBackendDiag *)0)->drv_rc));
#endif
#if !defined(ONLY) || ONLY == 208
  printf("field OmegaGpuBackendDiag drv_text %zu %zu\n", offsetof(OmegaGpuBackendDiag, drv_text), sizeof(((OmegaGpuBackendDiag *)0)->drv_text));
#endif
#if !defined(ONLY) || ONLY == 209
  printf("field OmegaGpuBackendDiag sync_valid %zu %zu\n", offsetof(OmegaGpuBackendDiag, sync_valid), sizeof(((OmegaGpuBackendDiag *)0)->sync_valid));
#endif
#if !defined(ONLY) || ONLY == 210
  printf("field OmegaGpuBackendDiag marker %zu %zu\n", offsetof(OmegaGpuBackendDiag, marker), sizeof(((OmegaGpuBackendDiag *)0)->marker));
#endif
#if !defined(ONLY) || ONLY == 211
  printf("field OmegaGpuBackendDiag marker2 %zu %zu\n", offsetof(OmegaGpuBackendDiag, marker2), sizeof(((OmegaGpuBackendDiag *)0)->marker2));
#endif
#if !defined(ONLY) || ONLY == 212
  printf("field OmegaGpuBackendDiag semaphore %zu %zu\n", offsetof(OmegaGpuBackendDiag, semaphore), sizeof(((OmegaGpuBackendDiag *)0)->semaphore));
#endif
#if !defined(ONLY) || ONLY == 213
  printf("struct OmegaGpuBackend %zu %zu\n", sizeof(OmegaGpuBackend), _Alignof(OmegaGpuBackend));
#endif
#if !defined(ONLY) || ONLY == 214
  printf("field OmegaGpuBackend ctx %zu %zu\n", offsetof(OmegaGpuBackend, ctx), sizeof(((OmegaGpuBackend *)0)->ctx));
#endif
#if !defined(ONLY) || ONLY == 215
  printf("field OmegaGpuBackend open_device %zu %zu\n", offsetof(OmegaGpuBackend, open_device), sizeof(((OmegaGpuBackend *)0)->open_device));
#endif
#if !defined(ONLY) || ONLY == 216
  printf("field OmegaGpuBackend create_channel %zu %zu\n", offsetof(OmegaGpuBackend, create_channel), sizeof(((OmegaGpuBackend *)0)->create_channel));
#endif
#if !defined(ONLY) || ONLY == 217
  printf("field OmegaGpuBackend alloc %zu %zu\n", offsetof(OmegaGpuBackend, alloc), sizeof(((OmegaGpuBackend *)0)->alloc));
#endif
#if !defined(ONLY) || ONLY == 218
  printf("field OmegaGpuBackend copy_in %zu %zu\n", offsetof(OmegaGpuBackend, copy_in), sizeof(((OmegaGpuBackend *)0)->copy_in));
#endif
#if !defined(ONLY) || ONLY == 219
  printf("field OmegaGpuBackend fill_poison %zu %zu\n", offsetof(OmegaGpuBackend, fill_poison), sizeof(((OmegaGpuBackend *)0)->fill_poison));
#endif
#if !defined(ONLY) || ONLY == 220
  printf("field OmegaGpuBackend build %zu %zu\n", offsetof(OmegaGpuBackend, build), sizeof(((OmegaGpuBackend *)0)->build));
#endif
#if !defined(ONLY) || ONLY == 221
  printf("field OmegaGpuBackend submit %zu %zu\n", offsetof(OmegaGpuBackend, submit), sizeof(((OmegaGpuBackend *)0)->submit));
#endif
#if !defined(ONLY) || ONLY == 222
  printf("field OmegaGpuBackend wait_marker %zu %zu\n", offsetof(OmegaGpuBackend, wait_marker), sizeof(((OmegaGpuBackend *)0)->wait_marker));
#endif
#if !defined(ONLY) || ONLY == 223
  printf("field OmegaGpuBackend wait_marker2 %zu %zu\n", offsetof(OmegaGpuBackend, wait_marker2), sizeof(((OmegaGpuBackend *)0)->wait_marker2));
#endif
#if !defined(ONLY) || ONLY == 224
  printf("field OmegaGpuBackend wait_release_semaphore %zu %zu\n", offsetof(OmegaGpuBackend, wait_release_semaphore), sizeof(((OmegaGpuBackend *)0)->wait_release_semaphore));
#endif
#if !defined(ONLY) || ONLY == 225
  printf("field OmegaGpuBackend barrier %zu %zu\n", offsetof(OmegaGpuBackend, barrier), sizeof(((OmegaGpuBackend *)0)->barrier));
#endif
#if !defined(ONLY) || ONLY == 226
  printf("field OmegaGpuBackend copy_out %zu %zu\n", offsetof(OmegaGpuBackend, copy_out), sizeof(((OmegaGpuBackend *)0)->copy_out));
#endif
#if !defined(ONLY) || ONLY == 227
  printf("field OmegaGpuBackend diagnostics %zu %zu\n", offsetof(OmegaGpuBackend, diagnostics), sizeof(((OmegaGpuBackend *)0)->diagnostics));
#endif
#if !defined(ONLY) || ONLY == 228
  printf("field OmegaGpuBackend free_buf %zu %zu\n", offsetof(OmegaGpuBackend, free_buf), sizeof(((OmegaGpuBackend *)0)->free_buf));
#endif
#if !defined(ONLY) || ONLY == 229
  printf("field OmegaGpuBackend close %zu %zu\n", offsetof(OmegaGpuBackend, close), sizeof(((OmegaGpuBackend *)0)->close));
#endif
#if !defined(ONLY) || ONLY == 238
  printf("struct OmegaGpuMatmulInfo %zu %zu\n", sizeof(OmegaGpuMatmulInfo), _Alignof(OmegaGpuMatmulInfo));
#endif
#if !defined(ONLY) || ONLY == 239
  printf("field OmegaGpuMatmulInfo m %zu %zu\n", offsetof(OmegaGpuMatmulInfo, m), sizeof(((OmegaGpuMatmulInfo *)0)->m));
#endif
#if !defined(ONLY) || ONLY == 240
  printf("field OmegaGpuMatmulInfo k %zu %zu\n", offsetof(OmegaGpuMatmulInfo, k), sizeof(((OmegaGpuMatmulInfo *)0)->k));
#endif
#if !defined(ONLY) || ONLY == 241
  printf("field OmegaGpuMatmulInfo n %zu %zu\n", offsetof(OmegaGpuMatmulInfo, n), sizeof(((OmegaGpuMatmulInfo *)0)->n));
#endif
#if !defined(ONLY) || ONLY == 242
  printf("field OmegaGpuMatmulInfo padded_m %zu %zu\n", offsetof(OmegaGpuMatmulInfo, padded_m), sizeof(((OmegaGpuMatmulInfo *)0)->padded_m));
#endif
#if !defined(ONLY) || ONLY == 243
  printf("field OmegaGpuMatmulInfo padded_n %zu %zu\n", offsetof(OmegaGpuMatmulInfo, padded_n), sizeof(((OmegaGpuMatmulInfo *)0)->padded_n));
#endif
#if !defined(ONLY) || ONLY == 244
  printf("field OmegaGpuMatmulInfo k_slices %zu %zu\n", offsetof(OmegaGpuMatmulInfo, k_slices), sizeof(((OmegaGpuMatmulInfo *)0)->k_slices));
#endif
#if !defined(ONLY) || ONLY == 245
  printf("field OmegaGpuMatmulInfo chip_calls %zu %zu\n", offsetof(OmegaGpuMatmulInfo, chip_calls), sizeof(((OmegaGpuMatmulInfo *)0)->chip_calls));
#endif
#if !defined(ONLY) || ONLY == 246
  printf("field OmegaGpuMatmulInfo rows_per_call %zu %zu\n", offsetof(OmegaGpuMatmulInfo, rows_per_call), sizeof(((OmegaGpuMatmulInfo *)0)->rows_per_call));
#endif
#if !defined(ONLY) || ONLY == 247
  printf("field OmegaGpuMatmulInfo elapsed_ns %zu %zu\n", offsetof(OmegaGpuMatmulInfo, elapsed_ns), sizeof(((OmegaGpuMatmulInfo *)0)->elapsed_ns));
#endif
#if !defined(ONLY) || ONLY == 248
  printf("field OmegaGpuMatmulInfo max_abs_err %zu %zu\n", offsetof(OmegaGpuMatmulInfo, max_abs_err), sizeof(((OmegaGpuMatmulInfo *)0)->max_abs_err));
#endif
#if !defined(ONLY) || ONLY == 249
  printf("field OmegaGpuMatmulInfo max_rel_err %zu %zu\n", offsetof(OmegaGpuMatmulInfo, max_rel_err), sizeof(((OmegaGpuMatmulInfo *)0)->max_rel_err));
#endif
#if !defined(ONLY) || ONLY == 250
  printf("field OmegaGpuMatmulInfo mismatch_count %zu %zu\n", offsetof(OmegaGpuMatmulInfo, mismatch_count), sizeof(((OmegaGpuMatmulInfo *)0)->mismatch_count));
#endif
#if !defined(ONLY) || ONLY == 251
  printf("field OmegaGpuMatmulInfo parity_verified %zu %zu\n", offsetof(OmegaGpuMatmulInfo, parity_verified), sizeof(((OmegaGpuMatmulInfo *)0)->parity_verified));
#endif
#if !defined(ONLY) || ONLY == 252
  printf("field OmegaGpuMatmulInfo completion_marker %zu %zu\n", offsetof(OmegaGpuMatmulInfo, completion_marker), sizeof(((OmegaGpuMatmulInfo *)0)->completion_marker));
#endif
#if !defined(ONLY) || ONLY == 253
  printf("field OmegaGpuMatmulInfo kernel_cache_hit %zu %zu\n", offsetof(OmegaGpuMatmulInfo, kernel_cache_hit), sizeof(((OmegaGpuMatmulInfo *)0)->kernel_cache_hit));
#endif
#if !defined(ONLY) || ONLY == 254
  printf("field OmegaGpuMatmulInfo target_chip %zu %zu\n", offsetof(OmegaGpuMatmulInfo, target_chip), sizeof(((OmegaGpuMatmulInfo *)0)->target_chip));
#endif
#if !defined(ONLY) || ONLY == 255
  printf("field OmegaGpuMatmulInfo sm_architecture %zu %zu\n", offsetof(OmegaGpuMatmulInfo, sm_architecture), sizeof(((OmegaGpuMatmulInfo *)0)->sm_architecture));
#endif
#if !defined(ONLY) || ONLY == 256
  printf("field OmegaGpuMatmulInfo grid_x %zu %zu\n", offsetof(OmegaGpuMatmulInfo, grid_x), sizeof(((OmegaGpuMatmulInfo *)0)->grid_x));
#endif
#if !defined(ONLY) || ONLY == 257
  printf("field OmegaGpuMatmulInfo grid_y %zu %zu\n", offsetof(OmegaGpuMatmulInfo, grid_y), sizeof(((OmegaGpuMatmulInfo *)0)->grid_y));
#endif
#if !defined(ONLY) || ONLY == 258
  printf("field OmegaGpuMatmulInfo padded_k %zu %zu\n", offsetof(OmegaGpuMatmulInfo, padded_k), sizeof(((OmegaGpuMatmulInfo *)0)->padded_k));
#endif
#if !defined(ONLY) || ONLY == 259
  printf("field OmegaGpuMatmulInfo resident %zu %zu\n", offsetof(OmegaGpuMatmulInfo, resident), sizeof(((OmegaGpuMatmulInfo *)0)->resident));
#endif
#if !defined(ONLY) || ONLY == 260
  printf("field OmegaGpuMatmulInfo oracle_ran %zu %zu\n", offsetof(OmegaGpuMatmulInfo, oracle_ran), sizeof(((OmegaGpuMatmulInfo *)0)->oracle_ran));
#endif
#if !defined(ONLY) || ONLY == 261
  printf("field OmegaGpuMatmulInfo call_ns %zu %zu\n", offsetof(OmegaGpuMatmulInfo, call_ns), sizeof(((OmegaGpuMatmulInfo *)0)->call_ns));
#endif
#if !defined(ONLY) || ONLY == 262
  printf("struct OmegaGpuAllocStats %zu %zu\n", sizeof(OmegaGpuAllocStats), _Alignof(OmegaGpuAllocStats));
#endif
#if !defined(ONLY) || ONLY == 263
  printf("field OmegaGpuAllocStats allocs %zu %zu\n", offsetof(OmegaGpuAllocStats, allocs), sizeof(((OmegaGpuAllocStats *)0)->allocs));
#endif
#if !defined(ONLY) || ONLY == 264
  printf("field OmegaGpuAllocStats alloc_bytes %zu %zu\n", offsetof(OmegaGpuAllocStats, alloc_bytes), sizeof(((OmegaGpuAllocStats *)0)->alloc_bytes));
#endif
#if !defined(ONLY) || ONLY == 265
  printf("field OmegaGpuAllocStats alloc_failures %zu %zu\n", offsetof(OmegaGpuAllocStats, alloc_failures), sizeof(((OmegaGpuAllocStats *)0)->alloc_failures));
#endif
#if !defined(ONLY) || ONLY == 266
  printf("field OmegaGpuAllocStats frees %zu %zu\n", offsetof(OmegaGpuAllocStats, frees), sizeof(((OmegaGpuAllocStats *)0)->frees));
#endif
#if !defined(ONLY) || ONLY == 267
  printf("field OmegaGpuAllocStats stale_refusals %zu %zu\n", offsetof(OmegaGpuAllocStats, stale_refusals), sizeof(((OmegaGpuAllocStats *)0)->stale_refusals));
#endif
#if !defined(ONLY) || ONLY == 268
  printf("struct OmegaGpuServingBounds %zu %zu\n", sizeof(OmegaGpuServingBounds), _Alignof(OmegaGpuServingBounds));
#endif
#if !defined(ONLY) || ONLY == 269
  printf("field OmegaGpuServingBounds max_context %zu %zu\n", offsetof(OmegaGpuServingBounds, max_context), sizeof(((OmegaGpuServingBounds *)0)->max_context));
#endif
#if !defined(ONLY) || ONLY == 270
  printf("field OmegaGpuServingBounds max_seqs %zu %zu\n", offsetof(OmegaGpuServingBounds, max_seqs), sizeof(((OmegaGpuServingBounds *)0)->max_seqs));
#endif
#if !defined(ONLY) || ONLY == 271
  printf("field OmegaGpuServingBounds num_q_heads %zu %zu\n", offsetof(OmegaGpuServingBounds, num_q_heads), sizeof(((OmegaGpuServingBounds *)0)->num_q_heads));
#endif
#if !defined(ONLY) || ONLY == 272
  printf("field OmegaGpuServingBounds num_kv_heads %zu %zu\n", offsetof(OmegaGpuServingBounds, num_kv_heads), sizeof(((OmegaGpuServingBounds *)0)->num_kv_heads));
#endif
#if !defined(ONLY) || ONLY == 273
  printf("field OmegaGpuServingBounds head_dim %zu %zu\n", offsetof(OmegaGpuServingBounds, head_dim), sizeof(((OmegaGpuServingBounds *)0)->head_dim));
#endif
#if !defined(ONLY) || ONLY == 274
  printf("field OmegaGpuServingBounds kv_block_size %zu %zu\n", offsetof(OmegaGpuServingBounds, kv_block_size), sizeof(((OmegaGpuServingBounds *)0)->kv_block_size));
#endif
#if !defined(ONLY) || ONLY == 275
  printf("field OmegaGpuServingBounds max_rows %zu %zu\n", offsetof(OmegaGpuServingBounds, max_rows), sizeof(((OmegaGpuServingBounds *)0)->max_rows));
#endif
#if !defined(ONLY) || ONLY == 276
  printf("field OmegaGpuServingBounds max_k %zu %zu\n", offsetof(OmegaGpuServingBounds, max_k), sizeof(((OmegaGpuServingBounds *)0)->max_k));
#endif
#if !defined(ONLY) || ONLY == 277
  printf("field OmegaGpuServingBounds max_n %zu %zu\n", offsetof(OmegaGpuServingBounds, max_n), sizeof(((OmegaGpuServingBounds *)0)->max_n));
#endif
#if !defined(ONLY) || ONLY == 278
  printf("field OmegaGpuServingBounds max_n_one_row %zu %zu\n", offsetof(OmegaGpuServingBounds, max_n_one_row), sizeof(((OmegaGpuServingBounds *)0)->max_n_one_row));
#endif
#if !defined(ONLY) || ONLY == 279
  printf("field OmegaGpuServingBounds kernel_slots %zu %zu\n", offsetof(OmegaGpuServingBounds, kernel_slots), sizeof(((OmegaGpuServingBounds *)0)->kernel_slots));
#endif
#if !defined(ONLY) || ONLY == 280
  printf("struct OmegaGpuEwInfo %zu %zu\n", sizeof(OmegaGpuEwInfo), _Alignof(OmegaGpuEwInfo));
#endif
#if !defined(ONLY) || ONLY == 281
  printf("field OmegaGpuEwInfo chip_calls %zu %zu\n", offsetof(OmegaGpuEwInfo, chip_calls), sizeof(((OmegaGpuEwInfo *)0)->chip_calls));
#endif
#if !defined(ONLY) || ONLY == 282
  printf("field OmegaGpuEwInfo elapsed_ns %zu %zu\n", offsetof(OmegaGpuEwInfo, elapsed_ns), sizeof(((OmegaGpuEwInfo *)0)->elapsed_ns));
#endif
#if !defined(ONLY) || ONLY == 283
  printf("field OmegaGpuEwInfo completion_marker %zu %zu\n", offsetof(OmegaGpuEwInfo, completion_marker), sizeof(((OmegaGpuEwInfo *)0)->completion_marker));
#endif
#if !defined(ONLY) || ONLY == 284
  printf("field OmegaGpuEwInfo unwritten_words %zu %zu\n", offsetof(OmegaGpuEwInfo, unwritten_words), sizeof(((OmegaGpuEwInfo *)0)->unwritten_words));
#endif
#if !defined(ONLY) || ONLY == 285
  printf("field OmegaGpuEwInfo kernel_cache_hit %zu %zu\n", offsetof(OmegaGpuEwInfo, kernel_cache_hit), sizeof(((OmegaGpuEwInfo *)0)->kernel_cache_hit));
#endif
#if !defined(ONLY) || ONLY == 286
  printf("field OmegaGpuEwInfo gpr_count %zu %zu\n", offsetof(OmegaGpuEwInfo, gpr_count), sizeof(((OmegaGpuEwInfo *)0)->gpr_count));
#endif
#if !defined(ONLY) || ONLY == 287
  printf("field OmegaGpuEwInfo insn_count %zu %zu\n", offsetof(OmegaGpuEwInfo, insn_count), sizeof(((OmegaGpuEwInfo *)0)->insn_count));
#endif
#if !defined(ONLY) || ONLY == 288
  printf("field OmegaGpuEwInfo threads_per_cta %zu %zu\n", offsetof(OmegaGpuEwInfo, threads_per_cta), sizeof(((OmegaGpuEwInfo *)0)->threads_per_cta));
#endif
#if !defined(ONLY) || ONLY == 289
  printf("field OmegaGpuEwInfo ctas_last_launch %zu %zu\n", offsetof(OmegaGpuEwInfo, ctas_last_launch), sizeof(((OmegaGpuEwInfo *)0)->ctas_last_launch));
#endif
#if !defined(ONLY) || ONLY == 290
  printf("field OmegaGpuEwInfo target_chip %zu %zu\n", offsetof(OmegaGpuEwInfo, target_chip), sizeof(((OmegaGpuEwInfo *)0)->target_chip));
#endif
#if !defined(ONLY) || ONLY == 291
  printf("field OmegaGpuEwInfo sm_architecture %zu %zu\n", offsetof(OmegaGpuEwInfo, sm_architecture), sizeof(((OmegaGpuEwInfo *)0)->sm_architecture));
#endif
#if !defined(ONLY) || ONLY == 292
  printf("field OmegaGpuEwInfo call_ns %zu %zu\n", offsetof(OmegaGpuEwInfo, call_ns), sizeof(((OmegaGpuEwInfo *)0)->call_ns));
#endif
#if !defined(ONLY) || ONLY == 293
  printf("struct OmegaGpuAttnInfo %zu %zu\n", sizeof(OmegaGpuAttnInfo), _Alignof(OmegaGpuAttnInfo));
#endif
#if !defined(ONLY) || ONLY == 294
  printf("field OmegaGpuAttnInfo chip_calls %zu %zu\n", offsetof(OmegaGpuAttnInfo, chip_calls), sizeof(((OmegaGpuAttnInfo *)0)->chip_calls));
#endif
#if !defined(ONLY) || ONLY == 295
  printf("field OmegaGpuAttnInfo elapsed_ns %zu %zu\n", offsetof(OmegaGpuAttnInfo, elapsed_ns), sizeof(((OmegaGpuAttnInfo *)0)->elapsed_ns));
#endif
#if !defined(ONLY) || ONLY == 296
  printf("field OmegaGpuAttnInfo call_ns %zu %zu\n", offsetof(OmegaGpuAttnInfo, call_ns), sizeof(((OmegaGpuAttnInfo *)0)->call_ns));
#endif
#if !defined(ONLY) || ONLY == 297
  printf("field OmegaGpuAttnInfo completion_marker %zu %zu\n", offsetof(OmegaGpuAttnInfo, completion_marker), sizeof(((OmegaGpuAttnInfo *)0)->completion_marker));
#endif
#if !defined(ONLY) || ONLY == 298
  printf("field OmegaGpuAttnInfo unwritten_words %zu %zu\n", offsetof(OmegaGpuAttnInfo, unwritten_words), sizeof(((OmegaGpuAttnInfo *)0)->unwritten_words));
#endif
#if !defined(ONLY) || ONLY == 299
  printf("field OmegaGpuAttnInfo gpr_count %zu %zu\n", offsetof(OmegaGpuAttnInfo, gpr_count), sizeof(((OmegaGpuAttnInfo *)0)->gpr_count));
#endif
#if !defined(ONLY) || ONLY == 300
  printf("field OmegaGpuAttnInfo insn_count %zu %zu\n", offsetof(OmegaGpuAttnInfo, insn_count), sizeof(((OmegaGpuAttnInfo *)0)->insn_count));
#endif
#if !defined(ONLY) || ONLY == 301
  printf("field OmegaGpuAttnInfo kernel_cache_hit %zu %zu\n", offsetof(OmegaGpuAttnInfo, kernel_cache_hit), sizeof(((OmegaGpuAttnInfo *)0)->kernel_cache_hit));
#endif
#if !defined(ONLY) || ONLY == 302
  printf("field OmegaGpuAttnInfo ctas_last_launch %zu %zu\n", offsetof(OmegaGpuAttnInfo, ctas_last_launch), sizeof(((OmegaGpuAttnInfo *)0)->ctas_last_launch));
#endif
#if !defined(ONLY) || ONLY == 303
  printf("field OmegaGpuAttnInfo threads_per_cta %zu %zu\n", offsetof(OmegaGpuAttnInfo, threads_per_cta), sizeof(((OmegaGpuAttnInfo *)0)->threads_per_cta));
#endif
#if !defined(ONLY) || ONLY == 304
  printf("field OmegaGpuAttnInfo target_chip %zu %zu\n", offsetof(OmegaGpuAttnInfo, target_chip), sizeof(((OmegaGpuAttnInfo *)0)->target_chip));
#endif
#if !defined(ONLY) || ONLY == 305
  printf("field OmegaGpuAttnInfo sm_architecture %zu %zu\n", offsetof(OmegaGpuAttnInfo, sm_architecture), sizeof(((OmegaGpuAttnInfo *)0)->sm_architecture));
#endif
#if !defined(ONLY) || ONLY == 306
  printf("field OmegaGpuAttnInfo kv_bytes_staged %zu %zu\n", offsetof(OmegaGpuAttnInfo, kv_bytes_staged), sizeof(((OmegaGpuAttnInfo *)0)->kv_bytes_staged));
#endif
#if !defined(ONLY) || ONLY == 307
  printf("field OmegaGpuAttnInfo kv_bytes_naive %zu %zu\n", offsetof(OmegaGpuAttnInfo, kv_bytes_naive), sizeof(((OmegaGpuAttnInfo *)0)->kv_bytes_naive));
#endif
#if !defined(ONLY) || ONLY == 308
  printf("field OmegaGpuAttnInfo kv_blocks_logical %zu %zu\n", offsetof(OmegaGpuAttnInfo, kv_blocks_logical), sizeof(((OmegaGpuAttnInfo *)0)->kv_blocks_logical));
#endif
#if !defined(ONLY) || ONLY == 309
  printf("field OmegaGpuAttnInfo kv_blocks_unique %zu %zu\n", offsetof(OmegaGpuAttnInfo, kv_blocks_unique), sizeof(((OmegaGpuAttnInfo *)0)->kv_blocks_unique));
#endif
#if !defined(ONLY) || ONLY == 310
  printf("field OmegaGpuAttnInfo q_bytes %zu %zu\n", offsetof(OmegaGpuAttnInfo, q_bytes), sizeof(((OmegaGpuAttnInfo *)0)->q_bytes));
#endif
#if !defined(ONLY) || ONLY == 311
  printf("field OmegaGpuAttnInfo tab_bytes %zu %zu\n", offsetof(OmegaGpuAttnInfo, tab_bytes), sizeof(((OmegaGpuAttnInfo *)0)->tab_bytes));
#endif
#if !defined(ONLY) || ONLY == 312
  printf("field OmegaGpuAttnInfo out_bytes %zu %zu\n", offsetof(OmegaGpuAttnInfo, out_bytes), sizeof(((OmegaGpuAttnInfo *)0)->out_bytes));
#endif
#if !defined(ONLY) || ONLY == 313
  printf("field OmegaGpuAttnInfo kv_source %zu %zu\n", offsetof(OmegaGpuAttnInfo, kv_source), sizeof(((OmegaGpuAttnInfo *)0)->kv_source));
#endif
#if !defined(ONLY) || ONLY == 314
  printf("struct OmegaGpuKvLayout %zu %zu\n", sizeof(OmegaGpuKvLayout), _Alignof(OmegaGpuKvLayout));
#endif
#if !defined(ONLY) || ONLY == 315
  printf("field OmegaGpuKvLayout block_stride_bytes %zu %zu\n", offsetof(OmegaGpuKvLayout, block_stride_bytes), sizeof(((OmegaGpuKvLayout *)0)->block_stride_bytes));
#endif
#if !defined(ONLY) || ONLY == 316
  printf("field OmegaGpuKvLayout layer_stride_bytes %zu %zu\n", offsetof(OmegaGpuKvLayout, layer_stride_bytes), sizeof(((OmegaGpuKvLayout *)0)->layer_stride_bytes));
#endif
#if !defined(ONLY) || ONLY == 317
  printf("field OmegaGpuKvLayout kv_plane_stride_bytes %zu %zu\n", offsetof(OmegaGpuKvLayout, kv_plane_stride_bytes), sizeof(((OmegaGpuKvLayout *)0)->kv_plane_stride_bytes));
#endif
#if !defined(ONLY) || ONLY == 318
  printf("field OmegaGpuKvLayout token_stride_bytes %zu %zu\n", offsetof(OmegaGpuKvLayout, token_stride_bytes), sizeof(((OmegaGpuKvLayout *)0)->token_stride_bytes));
#endif
#if !defined(ONLY) || ONLY == 319
  printf("field OmegaGpuKvLayout head_stride_bytes %zu %zu\n", offsetof(OmegaGpuKvLayout, head_stride_bytes), sizeof(((OmegaGpuKvLayout *)0)->head_stride_bytes));
#endif
#if !defined(ONLY) || ONLY == 320
  printf("field OmegaGpuKvLayout pool_bytes %zu %zu\n", offsetof(OmegaGpuKvLayout, pool_bytes), sizeof(((OmegaGpuKvLayout *)0)->pool_bytes));
#endif
#if !defined(ONLY) || ONLY == 321
  printf("field OmegaGpuKvLayout num_blocks %zu %zu\n", offsetof(OmegaGpuKvLayout, num_blocks), sizeof(((OmegaGpuKvLayout *)0)->num_blocks));
#endif
#if !defined(ONLY) || ONLY == 322
  printf("field OmegaGpuKvLayout num_layers %zu %zu\n", offsetof(OmegaGpuKvLayout, num_layers), sizeof(((OmegaGpuKvLayout *)0)->num_layers));
#endif
#if !defined(ONLY) || ONLY == 323
  printf("field OmegaGpuKvLayout block_size %zu %zu\n", offsetof(OmegaGpuKvLayout, block_size), sizeof(((OmegaGpuKvLayout *)0)->block_size));
#endif
#if !defined(ONLY) || ONLY == 324
  printf("const OMEGA_GPU_MATMUL_OK %lld\n", (long long)(OMEGA_GPU_MATMUL_OK));
#endif
#if !defined(ONLY) || ONLY == 325
  printf("const OMEGA_GPU_MATMUL_BAD_ARGS %lld\n", (long long)(OMEGA_GPU_MATMUL_BAD_ARGS));
#endif
#if !defined(ONLY) || ONLY == 326
  printf("const OMEGA_GPU_MATMUL_TOO_LARGE %lld\n", (long long)(OMEGA_GPU_MATMUL_TOO_LARGE));
#endif
#if !defined(ONLY) || ONLY == 327
  printf("const OMEGA_GPU_MATMUL_CODEGEN_FAIL %lld\n", (long long)(OMEGA_GPU_MATMUL_CODEGEN_FAIL));
#endif
#if !defined(ONLY) || ONLY == 328
  printf("const OMEGA_GPU_MATMUL_CHIP_FAIL %lld\n", (long long)(OMEGA_GPU_MATMUL_CHIP_FAIL));
#endif
#if !defined(ONLY) || ONLY == 329
  printf("const OMEGA_GPU_MATMUL_PARITY_FAIL %lld\n", (long long)(OMEGA_GPU_MATMUL_PARITY_FAIL));
#endif
#if !defined(ONLY) || ONLY == 330
  printf("const OMEGA_GPU_MATMUL_TILE_M %lld\n", (long long)(OMEGA_GPU_MATMUL_TILE_M));
#endif
#if !defined(ONLY) || ONLY == 331
  printf("const OMEGA_GPU_MATMUL_TILE_N %lld\n", (long long)(OMEGA_GPU_MATMUL_TILE_N));
#endif
#if !defined(ONLY) || ONLY == 332
  printf("const OMEGA_GPU_MATMUL_TILE_K %lld\n", (long long)(OMEGA_GPU_MATMUL_TILE_K));
#endif
#if !defined(ONLY) || ONLY == 333
  printf("const OMEGA_GPU_MATMUL_MAX_CTAS %lld\n", (long long)(OMEGA_GPU_MATMUL_MAX_CTAS));
#endif
#if !defined(ONLY) || ONLY == 334
  printf("const OMEGA_GPU_EW_OK %lld\n", (long long)(OMEGA_GPU_EW_OK));
#endif
#if !defined(ONLY) || ONLY == 335
  printf("const OMEGA_GPU_EW_BAD_ARGS %lld\n", (long long)(OMEGA_GPU_EW_BAD_ARGS));
#endif
#if !defined(ONLY) || ONLY == 336
  printf("const OMEGA_GPU_EW_TOO_LARGE %lld\n", (long long)(OMEGA_GPU_EW_TOO_LARGE));
#endif
#if !defined(ONLY) || ONLY == 337
  printf("const OMEGA_GPU_EW_CODEGEN_FAIL %lld\n", (long long)(OMEGA_GPU_EW_CODEGEN_FAIL));
#endif
#if !defined(ONLY) || ONLY == 338
  printf("const OMEGA_GPU_EW_CHIP_FAIL %lld\n", (long long)(OMEGA_GPU_EW_CHIP_FAIL));
#endif
#if !defined(ONLY) || ONLY == 339
  printf("const OMEGA_GPU_EW_UNWRITTEN %lld\n", (long long)(OMEGA_GPU_EW_UNWRITTEN));
#endif
#if !defined(ONLY) || ONLY == 340
  printf("const OMEGA_GPU_ATTN_OK %lld\n", (long long)(OMEGA_GPU_ATTN_OK));
#endif
#if !defined(ONLY) || ONLY == 341
  printf("const OMEGA_GPU_ATTN_BAD_ARGS %lld\n", (long long)(OMEGA_GPU_ATTN_BAD_ARGS));
#endif
#if !defined(ONLY) || ONLY == 342
  printf("const OMEGA_GPU_ATTN_TOO_LARGE %lld\n", (long long)(OMEGA_GPU_ATTN_TOO_LARGE));
#endif
#if !defined(ONLY) || ONLY == 343
  printf("const OMEGA_GPU_ATTN_CODEGEN_FAIL %lld\n", (long long)(OMEGA_GPU_ATTN_CODEGEN_FAIL));
#endif
#if !defined(ONLY) || ONLY == 344
  printf("const OMEGA_GPU_ATTN_CHIP_FAIL %lld\n", (long long)(OMEGA_GPU_ATTN_CHIP_FAIL));
#endif
#if !defined(ONLY) || ONLY == 345
  printf("const OMEGA_GPU_ATTN_UNWRITTEN %lld\n", (long long)(OMEGA_GPU_ATTN_UNWRITTEN));
#endif
#if !defined(ONLY) || ONLY == 346
  printf("const OMEGA_GPU_ATTN_KV_STAGED %lld\n", (long long)(OMEGA_GPU_ATTN_KV_STAGED));
#endif
#if !defined(ONLY) || ONLY == 381
  printf("const OMEGA_GPU_PLATFORM_ABI_MAJOR %lld\n", (long long)(OMEGA_GPU_PLATFORM_ABI_MAJOR));
#endif
#if !defined(ONLY) || ONLY == 382
  printf("const OMEGA_GPU_PLATFORM_ABI_MINOR %lld\n", (long long)(OMEGA_GPU_PLATFORM_ABI_MINOR));
#endif
#if !defined(ONLY) || ONLY == 383
  printf("const OMEGA_GPU_POLL_TIMED_OUT %lld\n", (long long)(OMEGA_GPU_POLL_TIMED_OUT));
#endif
#if !defined(ONLY) || ONLY == 384
  printf("const OMEGA_GPU_HANDLE_SLOTS %lld\n", (long long)(OMEGA_GPU_HANDLE_SLOTS));
#endif
#if !defined(ONLY) || ONLY == 385
  printf("const OMEGA_GPU_HANDLE_DEVICE %lld\n", (long long)(OMEGA_GPU_HANDLE_DEVICE));
#endif
#if !defined(ONLY) || ONLY == 386
  printf("const OMEGA_GPU_HANDLE_BUF_DATA %lld\n", (long long)(OMEGA_GPU_HANDLE_BUF_DATA));
#endif
#if !defined(ONLY) || ONLY == 387
  printf("const OMEGA_GPU_HANDLE_BUF_CODE %lld\n", (long long)(OMEGA_GPU_HANDLE_BUF_CODE));
#endif
#if !defined(ONLY) || ONLY == 388
  printf("const OMEGA_GPU_HANDLE_BUF_QUEUE %lld\n", (long long)(OMEGA_GPU_HANDLE_BUF_QUEUE));
#endif
#if !defined(ONLY) || ONLY == 389
  printf("const OMEGA_GPU_HANDLE_QUEUE %lld\n", (long long)(OMEGA_GPU_HANDLE_QUEUE));
#endif
#if !defined(ONLY) || ONLY == 390
  printf("const OMEGA_GPU_HANDLE_KIND_COUNT %lld\n", (long long)(OMEGA_GPU_HANDLE_KIND_COUNT));
#endif
#if !defined(ONLY) || ONLY == 391
  printf("const OMEGA_GPU_RIGHT_CPU_READ %lld\n", (long long)(OMEGA_GPU_RIGHT_CPU_READ));
#endif
#if !defined(ONLY) || ONLY == 392
  printf("const OMEGA_GPU_RIGHT_CPU_WRITE %lld\n", (long long)(OMEGA_GPU_RIGHT_CPU_WRITE));
#endif
#if !defined(ONLY) || ONLY == 393
  printf("const OMEGA_GPU_RIGHT_GPU_READ %lld\n", (long long)(OMEGA_GPU_RIGHT_GPU_READ));
#endif
#if !defined(ONLY) || ONLY == 394
  printf("const OMEGA_GPU_RIGHT_GPU_WRITE %lld\n", (long long)(OMEGA_GPU_RIGHT_GPU_WRITE));
#endif
#if !defined(ONLY) || ONLY == 395
  printf("const OMEGA_GPU_RIGHT_GPU_EXEC %lld\n", (long long)(OMEGA_GPU_RIGHT_GPU_EXEC));
#endif
#if !defined(ONLY) || ONLY == 396
  printf("const OMEGA_GPU_RIGHT_HOST_POLLED %lld\n", (long long)(OMEGA_GPU_RIGHT_HOST_POLLED));
#endif
#if !defined(ONLY) || ONLY == 397
  printf("const OMEGA_GPU_RIGHT_ALL %lld\n", (long long)(OMEGA_GPU_RIGHT_ALL));
#endif
#if !defined(ONLY) || ONLY == 398
  printf("const OMEGA_GPU_CACHE_GPU_UNCACHED %lld\n", (long long)(OMEGA_GPU_CACHE_GPU_UNCACHED));
#endif
#if !defined(ONLY) || ONLY == 399
  printf("const OMEGA_GPU_CACHE_GPU_CACHED %lld\n", (long long)(OMEGA_GPU_CACHE_GPU_CACHED));
#endif
#if !defined(ONLY) || ONLY == 400
  printf("const OMEGA_GPU_SVC_TIME %lld\n", (long long)(OMEGA_GPU_SVC_TIME));
#endif
#if !defined(ONLY) || ONLY == 401
  printf("const OMEGA_GPU_SVC_WAIT %lld\n", (long long)(OMEGA_GPU_SVC_WAIT));
#endif
#if !defined(ONLY) || ONLY == 402
  printf("const OMEGA_GPU_SVC_LOCK %lld\n", (long long)(OMEGA_GPU_SVC_LOCK));
#endif
#if !defined(ONLY) || ONLY == 403
  printf("const OMEGA_GPU_SVC_MMIO %lld\n", (long long)(OMEGA_GPU_SVC_MMIO));
#endif
#if !defined(ONLY) || ONLY == 404
  printf("const OMEGA_GPU_SVC_DMA %lld\n", (long long)(OMEGA_GPU_SVC_DMA));
#endif
#if !defined(ONLY) || ONLY == 405
  printf("const OMEGA_GPU_SVC_SYNC %lld\n", (long long)(OMEGA_GPU_SVC_SYNC));
#endif
#if !defined(ONLY) || ONLY == 406
  printf("const OMEGA_GPU_SVC_COMPLETION %lld\n", (long long)(OMEGA_GPU_SVC_COMPLETION));
#endif
#if !defined(ONLY) || ONLY == 407
  printf("const OMEGA_GPU_SVC_IRQ %lld\n", (long long)(OMEGA_GPU_SVC_IRQ));
#endif
#if !defined(ONLY) || ONLY == 408
  printf("const OMEGA_GPU_SVC_REQUIRED_V1 %lld\n", (long long)(OMEGA_GPU_SVC_REQUIRED_V1));
#endif
#if !defined(ONLY) || ONLY == 409
  printf("const OMEGA_GPU_SVC_DEVICE %lld\n", (long long)(OMEGA_GPU_SVC_DEVICE));
#endif
#if !defined(ONLY) || ONLY == 410
  printf("const OMEGA_GPU_SVC_QUEUE %lld\n", (long long)(OMEGA_GPU_SVC_QUEUE));
#endif
#if !defined(ONLY) || ONLY == 411
  printf("const OMEGA_GPU_SVC_CLOCK_NS %lld\n", (long long)(OMEGA_GPU_SVC_CLOCK_NS));
#endif
#if !defined(ONLY) || ONLY == 412
  printf("const OMEGA_GPU_SVC_HOST_MEM %lld\n", (long long)(OMEGA_GPU_SVC_HOST_MEM));
#endif
#if !defined(ONLY) || ONLY == 413
  printf("const OMEGA_GPU_SVC_REQUIRED_SESSION %lld\n", (long long)(OMEGA_GPU_SVC_REQUIRED_SESSION));
#endif
#if !defined(ONLY) || ONLY == 414
  printf("const OMEGA_GPU_SVC_ALL %lld\n", (long long)(OMEGA_GPU_SVC_ALL));
#endif
#if !defined(ONLY) || ONLY == 415
  printf("const OMEGA_GPU_PLAT_OK %lld\n", (long long)(OMEGA_GPU_PLAT_OK));
#endif
#if !defined(ONLY) || ONLY == 416
  printf("const OMEGA_GPU_PLAT_REFUSED_ARGS %lld\n", (long long)(OMEGA_GPU_PLAT_REFUSED_ARGS));
#endif
#if !defined(ONLY) || ONLY == 417
  printf("const OMEGA_GPU_PLAT_REFUSED_ABI %lld\n", (long long)(OMEGA_GPU_PLAT_REFUSED_ABI));
#endif
#if !defined(ONLY) || ONLY == 418
  printf("const OMEGA_GPU_PLAT_REFUSED_MISSING_SERVICE %lld\n", (long long)(OMEGA_GPU_PLAT_REFUSED_MISSING_SERVICE));
#endif
#if !defined(ONLY) || ONLY == 419
  printf("const OMEGA_GPU_PLAT_REFUSED_NULL_SERVICE %lld\n", (long long)(OMEGA_GPU_PLAT_REFUSED_NULL_SERVICE));
#endif
#if !defined(ONLY) || ONLY == 420
  printf("const OMEGA_GPU_PLAT_REFUSED_BAD_HANDLE %lld\n", (long long)(OMEGA_GPU_PLAT_REFUSED_BAD_HANDLE));
#endif
#if !defined(ONLY) || ONLY == 421
  printf("const OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE %lld\n", (long long)(OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE));
#endif
#if !defined(ONLY) || ONLY == 422
  printf("const OMEGA_GPU_PLAT_REFUSED_BAD_LENGTH %lld\n", (long long)(OMEGA_GPU_PLAT_REFUSED_BAD_LENGTH));
#endif
#if !defined(ONLY) || ONLY == 423
  printf("const OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS %lld\n", (long long)(OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS));
#endif
#if !defined(ONLY) || ONLY == 424
  printf("const OMEGA_GPU_PLAT_REFUSED_BAD_KIND %lld\n", (long long)(OMEGA_GPU_PLAT_REFUSED_BAD_KIND));
#endif
#if !defined(ONLY) || ONLY == 425
  printf("const OMEGA_GPU_PLAT_REFUSED_CACHE_POLICY %lld\n", (long long)(OMEGA_GPU_PLAT_REFUSED_CACHE_POLICY));
#endif
#if !defined(ONLY) || ONLY == 426
  printf("const OMEGA_GPU_PLAT_REFUSED_LIMIT %lld\n", (long long)(OMEGA_GPU_PLAT_REFUSED_LIMIT));
#endif
#if !defined(ONLY) || ONLY == 427
  printf("const OMEGA_GPU_PLAT_REFUSED_BUSY %lld\n", (long long)(OMEGA_GPU_PLAT_REFUSED_BUSY));
#endif
#if !defined(ONLY) || ONLY == 428
  printf("const OMEGA_GPU_PLAT_REFUSED_DEVICE %lld\n", (long long)(OMEGA_GPU_PLAT_REFUSED_DEVICE));
#endif
#if !defined(ONLY) || ONLY == 429
  printf("const OMEGA_GPU_PLAT_REFUSAL_COUNT %lld\n", (long long)(OMEGA_GPU_PLAT_REFUSAL_COUNT));
#endif
#if !defined(ONLY) || ONLY == 430
  printf("const OMEGA_GPU_SESSION_MAX_SPIN_US %lld\n", (long long)(OMEGA_GPU_SESSION_MAX_SPIN_US));
#endif
#if !defined(ONLY) || ONLY == 431
  printf("const OMEGA_GPU_SESSION_MAX_ARGS %lld\n", (long long)(OMEGA_GPU_SESSION_MAX_ARGS));
#endif
#if !defined(ONLY) || ONLY == 432
  printf("const OMEGA_GPU_SESSION_POLL_SLICE_MS %lld\n", (long long)(OMEGA_GPU_SESSION_POLL_SLICE_MS));
#endif
#if !defined(ONLY) || ONLY == 433
  printf("const OMEGA_GPU_SESSION_LOCK_ID %lld\n", (long long)(OMEGA_GPU_SESSION_LOCK_ID));
#endif
#if !defined(ONLY) || ONLY == 434
  printf("const OMEGA_GPU_SESSION_SCRATCH_RESERVED %lld\n", (long long)(OMEGA_GPU_SESSION_SCRATCH_RESERVED));
#endif
#if !defined(ONLY) || ONLY == 435
  printf("const OMEGA_GPU_SESSION_CANCELLED %lld\n", (long long)(OMEGA_GPU_SESSION_CANCELLED));
#endif
#if !defined(ONLY) || ONLY == 436
  printf("const OMEGA_GPU_SESSION_PARK_MAX %lld\n", (long long)(OMEGA_GPU_SESSION_PARK_MAX));
#endif
#if !defined(ONLY) || ONLY == 437
  printf("const OMEGA_GPU_COMPLETION_POLLING %lld\n", (long long)(OMEGA_GPU_COMPLETION_POLLING));
#endif
#if !defined(ONLY) || ONLY == 438
  printf("const OMEGA_GPU_CMDLIST_MAX_LAUNCHES %lld\n", (long long)(OMEGA_GPU_CMDLIST_MAX_LAUNCHES));
#endif
#if !defined(ONLY) || ONLY == 439
  printf("const OMEGA_GPU_CMDLIST_BANK_BYTES %lld\n", (long long)(OMEGA_GPU_CMDLIST_BANK_BYTES));
#endif
#if !defined(ONLY) || ONLY == 440
  printf("const OMEGA_GPU_CMDLIST_QMD_BYTES %lld\n", (long long)(OMEGA_GPU_CMDLIST_QMD_BYTES));
#endif
#if !defined(ONLY) || ONLY == 441
  printf("const OMEGA_GPU_CMDLIST_PCAS %lld\n", (long long)(OMEGA_GPU_CMDLIST_PCAS));
#endif
#if !defined(ONLY) || ONLY == 442
  printf("const OMEGA_GPU_CMDLIST_INLINE %lld\n", (long long)(OMEGA_GPU_CMDLIST_INLINE));
#endif
#if !defined(ONLY) || ONLY == 443
  printf("const OMEGA_GPU_CMDLIST_BARRIER_WFI %lld\n", (long long)(OMEGA_GPU_CMDLIST_BARRIER_WFI));
#endif
#if !defined(ONLY) || ONLY == 444
  printf("const OMEGA_GPU_CMDLIST_BARRIER_NONE %lld\n", (long long)(OMEGA_GPU_CMDLIST_BARRIER_NONE));
#endif
#if !defined(ONLY) || ONLY == 445
  printf("const OMEGA_GPU_CMDLIST_BARRIER_WFI_ONLY %lld\n", (long long)(OMEGA_GPU_CMDLIST_BARRIER_WFI_ONLY));
#endif
#if !defined(ONLY) || ONLY == 446
  printf("const OMEGA_GPU_CMDLIST_BARRIER_WFI_DATA %lld\n", (long long)(OMEGA_GPU_CMDLIST_BARRIER_WFI_DATA));
#endif
#if !defined(ONLY) || ONLY == 447
  printf("const OMEGA_GPU_CMDLIST_BARRIER_FLUSH_DATA %lld\n", (long long)(OMEGA_GPU_CMDLIST_BARRIER_FLUSH_DATA));
#endif
#if !defined(ONLY) || ONLY == 448
  printf("const OMEGA_GPU_ENGINE_FLAG_NO_C3 %lld\n", (long long)(OMEGA_GPU_ENGINE_FLAG_NO_C3));
#endif
#if !defined(ONLY) || ONLY == 449
  printf("const OMEGA_GPU_ENGINE_JOB_OUTPUTS %lld\n", (long long)(OMEGA_GPU_ENGINE_JOB_OUTPUTS));
#endif
#if !defined(ONLY) || ONLY == 450
  printf("const OMEGA_GPU_ENGINE_SEMAPHORE_WORDS %lld\n", (long long)(OMEGA_GPU_ENGINE_SEMAPHORE_WORDS));
#endif
#if !defined(ONLY) || ONLY == 451
  printf("const OMEGA_GPU_ENGINE_DRV_TEXT_BYTES %lld\n", (long long)(OMEGA_GPU_ENGINE_DRV_TEXT_BYTES));
#endif
#if !defined(ONLY) || ONLY == 452
  printf("const OMEGA_GPU_SYNC_MARKER %lld\n", (long long)(OMEGA_GPU_SYNC_MARKER));
#endif
#if !defined(ONLY) || ONLY == 453
  printf("const OMEGA_GPU_SYNC_MARKER2 %lld\n", (long long)(OMEGA_GPU_SYNC_MARKER2));
#endif
#if !defined(ONLY) || ONLY == 454
  printf("const OMEGA_GPU_SYNC_SEMAPHORE %lld\n", (long long)(OMEGA_GPU_SYNC_SEMAPHORE));
#endif
#if !defined(ONLY) || ONLY == 455
  printf("const OMEGA_GPU_BACKEND_TIMEOUT %lld\n", (long long)(OMEGA_GPU_BACKEND_TIMEOUT));
#endif
#if !defined(ONLY) || ONLY == 456
  printf("const OMEGA_GPU_BACKEND_UNKNOWN_STATE %lld\n", (long long)(OMEGA_GPU_BACKEND_UNKNOWN_STATE));
#endif
#if !defined(ONLY) || ONLY == 457
  printf("const OMEGA_GPU_ENGINE_OK %lld\n", (long long)(OMEGA_GPU_ENGINE_OK));
#endif
#if !defined(ONLY) || ONLY == 458
  printf("const OMEGA_GPU_ENGINE_DEVICE_OPEN %lld\n", (long long)(OMEGA_GPU_ENGINE_DEVICE_OPEN));
#endif
#if !defined(ONLY) || ONLY == 459
  printf("const OMEGA_GPU_ENGINE_CHANNEL_CREATE %lld\n", (long long)(OMEGA_GPU_ENGINE_CHANNEL_CREATE));
#endif
#if !defined(ONLY) || ONLY == 460
  printf("const OMEGA_GPU_ENGINE_ALLOC %lld\n", (long long)(OMEGA_GPU_ENGINE_ALLOC));
#endif
#if !defined(ONLY) || ONLY == 461
  printf("const OMEGA_GPU_ENGINE_PREPARE %lld\n", (long long)(OMEGA_GPU_ENGINE_PREPARE));
#endif
#if !defined(ONLY) || ONLY == 462
  printf("const OMEGA_GPU_ENGINE_SUBMIT %lld\n", (long long)(OMEGA_GPU_ENGINE_SUBMIT));
#endif
#if !defined(ONLY) || ONLY == 463
  printf("const OMEGA_GPU_ENGINE_COMPLETION_WAIT %lld\n", (long long)(OMEGA_GPU_ENGINE_COMPLETION_WAIT));
#endif
#if !defined(ONLY) || ONLY == 464
  printf("const OMEGA_GPU_ENGINE_RELEASE_WAIT %lld\n", (long long)(OMEGA_GPU_ENGINE_RELEASE_WAIT));
#endif
#if !defined(ONLY) || ONLY == 465
  printf("const OMEGA_GPU_ENGINE_VISIBILITY_WAIT %lld\n", (long long)(OMEGA_GPU_ENGINE_VISIBILITY_WAIT));
#endif
#if !defined(ONLY) || ONLY == 466
  printf("const OMEGA_GPU_ENGINE_OUTPUT_UNCHANGED %lld\n", (long long)(OMEGA_GPU_ENGINE_OUTPUT_UNCHANGED));
#endif
#if !defined(ONLY) || ONLY == 467
  printf("const OMEGA_GPU_ENGINE_INVALID_ARGS %lld\n", (long long)(OMEGA_GPU_ENGINE_INVALID_ARGS));
#endif
#if !defined(ONLY) || ONLY == 468
  printf("const OMEGA_GPU_ENGINE_INTERNAL_INVARIANT %lld\n", (long long)(OMEGA_GPU_ENGINE_INTERNAL_INVARIANT));
#endif
#if !defined(ONLY) || ONLY == 469
  printf("const OMEGA_GPU_ENGINE_CLEANUP %lld\n", (long long)(OMEGA_GPU_ENGINE_CLEANUP));
#endif
#if !defined(ONLY) || ONLY == 470
  printf("const OMEGA_GPU_ENGINE_UNCERTAIN_COMPLETION_BLOCKED %lld\n", (long long)(OMEGA_GPU_ENGINE_UNCERTAIN_COMPLETION_BLOCKED));
#endif
#if !defined(ONLY) || ONLY == 471
  printf("const OMEGA_GPU_ENGINE_FAILURE_COUNT %lld\n", (long long)(OMEGA_GPU_ENGINE_FAILURE_COUNT));
#endif
#if !defined(ONLY) || ONLY == 472
  printf("const OMEGA_GPU_ENGINE_STATE_INITIAL %lld\n", (long long)(OMEGA_GPU_ENGINE_STATE_INITIAL));
#endif
#if !defined(ONLY) || ONLY == 473
  printf("const OMEGA_GPU_ENGINE_STATE_PREPARED %lld\n", (long long)(OMEGA_GPU_ENGINE_STATE_PREPARED));
#endif
#if !defined(ONLY) || ONLY == 474
  printf("const OMEGA_GPU_ENGINE_STATE_SUBMITTED %lld\n", (long long)(OMEGA_GPU_ENGINE_STATE_SUBMITTED));
#endif
#if !defined(ONLY) || ONLY == 475
  printf("const OMEGA_GPU_ENGINE_STATE_GPU_COMPLETE %lld\n", (long long)(OMEGA_GPU_ENGINE_STATE_GPU_COMPLETE));
#endif
#if !defined(ONLY) || ONLY == 476
  printf("const OMEGA_GPU_ENGINE_STATE_OUTPUT_VISIBLE %lld\n", (long long)(OMEGA_GPU_ENGINE_STATE_OUTPUT_VISIBLE));
#endif
#if !defined(ONLY) || ONLY == 477
  printf("const OMEGA_GPU_ENGINE_STATE_OUTPUT_PRODUCED %lld\n", (long long)(OMEGA_GPU_ENGINE_STATE_OUTPUT_PRODUCED));
#endif
#if !defined(ONLY) || ONLY == 478
  printf("const OMEGA_GPU_ENGINE_STATE_SUCCESS %lld\n", (long long)(OMEGA_GPU_ENGINE_STATE_SUCCESS));
#endif
#if !defined(ONLY) || ONLY == 479
  printf("const OMEGA_GPU_ENGINE_STATE_COUNT %lld\n", (long long)(OMEGA_GPU_ENGINE_STATE_COUNT));
#endif
#if !defined(ONLY) || ONLY == 480
  printf("const OMEGA_GPU_ENGINE_WAIT_NONE %lld\n", (long long)(OMEGA_GPU_ENGINE_WAIT_NONE));
#endif
#if !defined(ONLY) || ONLY == 481
  printf("const OMEGA_GPU_ENGINE_WAIT_MARKER %lld\n", (long long)(OMEGA_GPU_ENGINE_WAIT_MARKER));
#endif
#if !defined(ONLY) || ONLY == 482
  printf("const OMEGA_GPU_ENGINE_WAIT_RELEASE_SEMAPHORE %lld\n", (long long)(OMEGA_GPU_ENGINE_WAIT_RELEASE_SEMAPHORE));
#endif
#if !defined(ONLY) || ONLY == 483
  printf("const OMEGA_GPU_ENGINE_WAIT_MARKER2 %lld\n", (long long)(OMEGA_GPU_ENGINE_WAIT_MARKER2));
#endif
#if !defined(ONLY) || ONLY == 484
  printf("const OMEGA_GPU_ENGINE_WAIT_COUNT %lld\n", (long long)(OMEGA_GPU_ENGINE_WAIT_COUNT));
#endif
#if !defined(ONLY) || ONLY == 485
  printf("const OMEGA_GPU_ENGINE_STEP_NONE %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_NONE));
#endif
#if !defined(ONLY) || ONLY == 486
  printf("const OMEGA_GPU_ENGINE_STEP_VALIDATE %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_VALIDATE));
#endif
#if !defined(ONLY) || ONLY == 487
  printf("const OMEGA_GPU_ENGINE_STEP_BACKEND_CHECK %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_BACKEND_CHECK));
#endif
#if !defined(ONLY) || ONLY == 488
  printf("const OMEGA_GPU_ENGINE_STEP_BLOCKED %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_BLOCKED));
#endif
#if !defined(ONLY) || ONLY == 489
  printf("const OMEGA_GPU_ENGINE_STEP_NOT_IMPLEMENTED %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_NOT_IMPLEMENTED));
#endif
#if !defined(ONLY) || ONLY == 490
  printf("const OMEGA_GPU_ENGINE_STEP_OPEN_DEVICE %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_OPEN_DEVICE));
#endif
#if !defined(ONLY) || ONLY == 491
  printf("const OMEGA_GPU_ENGINE_STEP_CREATE_CHANNEL %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_CREATE_CHANNEL));
#endif
#if !defined(ONLY) || ONLY == 492
  printf("const OMEGA_GPU_ENGINE_STEP_ALLOC %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_ALLOC));
#endif
#if !defined(ONLY) || ONLY == 493
  printf("const OMEGA_GPU_ENGINE_STEP_COPY_IN %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_COPY_IN));
#endif
#if !defined(ONLY) || ONLY == 494
  printf("const OMEGA_GPU_ENGINE_STEP_FILL_POISON %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_FILL_POISON));
#endif
#if !defined(ONLY) || ONLY == 495
  printf("const OMEGA_GPU_ENGINE_STEP_BUILD %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_BUILD));
#endif
#if !defined(ONLY) || ONLY == 496
  printf("const OMEGA_GPU_ENGINE_STEP_SUBMIT %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_SUBMIT));
#endif
#if !defined(ONLY) || ONLY == 497
  printf("const OMEGA_GPU_ENGINE_STEP_WAIT_MARKER %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_WAIT_MARKER));
#endif
#if !defined(ONLY) || ONLY == 498
  printf("const OMEGA_GPU_ENGINE_STEP_WAIT_MARKER2 %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_WAIT_MARKER2));
#endif
#if !defined(ONLY) || ONLY == 499
  printf("const OMEGA_GPU_ENGINE_STEP_WAIT_RELEASE_SEMAPHORE %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_WAIT_RELEASE_SEMAPHORE));
#endif
#if !defined(ONLY) || ONLY == 500
  printf("const OMEGA_GPU_ENGINE_STEP_BARRIER %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_BARRIER));
#endif
#if !defined(ONLY) || ONLY == 501
  printf("const OMEGA_GPU_ENGINE_STEP_COPY_OUT %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_COPY_OUT));
#endif
#if !defined(ONLY) || ONLY == 502
  printf("const OMEGA_GPU_ENGINE_STEP_SCAN %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_SCAN));
#endif
#if !defined(ONLY) || ONLY == 503
  printf("const OMEGA_GPU_ENGINE_STEP_FREE %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_FREE));
#endif
#if !defined(ONLY) || ONLY == 504
  printf("const OMEGA_GPU_ENGINE_STEP_CLOSE %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_CLOSE));
#endif
#if !defined(ONLY) || ONLY == 505
  printf("const OMEGA_GPU_ENGINE_STEP_CLOCK_CHECK %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_CLOCK_CHECK));
#endif
#if !defined(ONLY) || ONLY == 506
  printf("const OMEGA_GPU_ENGINE_STEP_COUNT %lld\n", (long long)(OMEGA_GPU_ENGINE_STEP_COUNT));
#endif
#if !defined(ONLY) || ONLY == 507
  printf("const OMEGA_GPU_LAYOUT_VECTOR_1D %lld\n", (long long)(OMEGA_GPU_LAYOUT_VECTOR_1D));
#endif
#if !defined(ONLY) || ONLY == 508
  printf("const OMEGA_GPU_LAYOUT_COUNT %lld\n", (long long)(OMEGA_GPU_LAYOUT_COUNT));
#endif
#if !defined(ONLY) || ONLY == 509
  printf("const OMEGA_GPU_BUF_SCRATCH %lld\n", (long long)(OMEGA_GPU_BUF_SCRATCH));
#endif
#if !defined(ONLY) || ONLY == 510
  printf("const OMEGA_GPU_BUF_PROGRAM %lld\n", (long long)(OMEGA_GPU_BUF_PROGRAM));
#endif
#if !defined(ONLY) || ONLY == 511
  printf("const OMEGA_GPU_BUF_INPUT_A %lld\n", (long long)(OMEGA_GPU_BUF_INPUT_A));
#endif
#if !defined(ONLY) || ONLY == 512
  printf("const OMEGA_GPU_BUF_INPUT_B %lld\n", (long long)(OMEGA_GPU_BUF_INPUT_B));
#endif
#if !defined(ONLY) || ONLY == 513
  printf("const OMEGA_GPU_BUF_OUTPUT %lld\n", (long long)(OMEGA_GPU_BUF_OUTPUT));
#endif
#if !defined(ONLY) || ONLY == 514
  printf("const OMEGA_GPU_BUF_ROLE_COUNT %lld\n", (long long)(OMEGA_GPU_BUF_ROLE_COUNT));
#endif
  return 0;
}
