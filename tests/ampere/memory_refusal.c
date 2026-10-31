/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Safe device proof: set reserve ABOVE observed free VRAM, then request one small page.
 * Never consume VRAM to induce pressure. Link --wrap=ioctl to count real RM allocation
 * calls independently of the allocation result; heap-info control ioctls are allowed. */
#define _POSIX_C_SOURCE 200809L
#include "omega_gpu_session.h"
#include "omega_gpu_platform_linux.h"
#include "nv-ioctl-numbers.h"
#include "nv_escape.h"
#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>

static uint64_t allocation_ioctls;
int __real_ioctl(int fd,unsigned long request,...);
int __wrap_ioctl(int fd,unsigned long request,...) {
    va_list ap; va_start(ap,request);void *argument=va_arg(ap,void *);va_end(ap);
    if(_IOC_TYPE(request)==NV_IOCTL_MAGIC &&
       (_IOC_NR(request)==NV_ESC_RM_ALLOC||_IOC_NR(request)==NV_ESC_RM_ALLOC_MEMORY||
        _IOC_NR(request)==NV_ESC_RM_ALLOC_OBJECT||_IOC_NR(request)==NV_ESC_RM_ALLOC_CONTEXT_DMA2))
        allocation_ioctls++;
    return __real_ioctl(fd,request,argument);
}
int main(int argc,char **argv) {
    if(argc!=2||strcmp(argv[1],"--device")) { puts("AMPERE MEMORY REFUSAL NOT_RUN (explicit --device required)");return 2; }
    int rc=1;uint64_t before=0,total=0,after=0,total_after=0;
    OmegaGpuMem small={0};
    omega_gpu_session_lock();
    if(!omega_gpu_session_open()||omega_gpu_session_target_sm()!=86) {
        fprintf(stderr,"AMPERE MEMORY REFUSAL NOT_RUN: open/target: %s\n",omega_gpu_session_last_error());rc=2;goto done;
    }
    if(omega_gpu_platform_linux_vram_info(&before,&total)||before>total||total==UINT64_MAX)goto done;
    /* Above total is also above free, even if another process frees VRAM
     * between these queries. This can never become a real pressure allocation. */
    char reserve[32];snprintf(reserve,sizeof reserve,"%"PRIu64,total+1);
    if(setenv("CUDAPP_GPU_MIN_FREE_BYTES",reserve,1))goto done;
    uint64_t calls_before=allocation_ioctls;
    int allocated=omega_gpu_session_alloc(4096,&small);
    uint64_t calls_after=allocation_ioctls;
    if(omega_gpu_platform_linux_vram_info(&after,&total_after))goto done;
    printf("reserve_bytes=%s requested_bytes=4096 free_before_bytes=%"PRIu64" free_after_bytes=%"PRIu64" total_bytes=%"PRIu64" rm_alloc_calls_before=%"PRIu64" rm_alloc_calls_after=%"PRIu64"\n",
           reserve,before,after,total,calls_before,calls_after);
    if(!allocated||small.cpu||small.va||calls_before!=calls_after||before!=after||total!=total_after) {
        fprintf(stderr,"AMPERE MEMORY REFUSAL FAIL (allocation, driver call, or VRAM delta observed)\n");goto done;
    }
    if(!strstr(omega_gpu_session_last_error(),"reserve")) {
        fprintf(stderr,"AMPERE MEMORY REFUSAL FAIL: refusal was not the reserve gate: %s\n",omega_gpu_session_last_error());goto done;
    }
    puts("AMPERE MEMORY REFUSAL PASS (small allocation refused before RM allocation; free VRAM unchanged)");rc=0;
done:
    if(small.cpu)omega_gpu_session_free(&small);
    if(rc==1)fprintf(stderr,"AMPERE MEMORY REFUSAL FAIL stage=%s\n",omega_gpu_session_last_error());
    omega_gpu_session_unlock();omega_gpu_session_close();return rc;
}
