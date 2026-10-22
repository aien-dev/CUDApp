/* SPDX-License-Identifier: MIT */
/* cudapp_host_run.h: the hello program (c[i] = a[i] + b[i]) in CUDA++'s own CPU code.
 * Same inputs as examples/hello_gpu. This is not a GPU launch and not a vendor math library. */
#ifndef CUDAPP_HOST_RUN_H
#define CUDAPP_HOST_RUN_H

#include <stddef.h>
#include <stdint.h>

#define CUDAPP_HOST_N (1u << 20) /* 1048576, the hello_gpu length */

/* 0: every element matches the CPU add. 1: at least one does not. -1: allocation failed.
 * mutate_index >= 0 leaves that element unwritten, so a correct checker must return 1. */
int cudapp_host_vecadd(uint32_t n, int mutate_index, double *seconds, uint8_t out_sha[32]);

/* Best of `reps` copies of `bytes` bytes. Returns seconds. *gib_s is bytes/seconds/2^30.
 * A probe of this run, not a published chip ceiling. */
double cudapp_host_copy_probe(size_t bytes, int reps, double *gib_s);

#endif
