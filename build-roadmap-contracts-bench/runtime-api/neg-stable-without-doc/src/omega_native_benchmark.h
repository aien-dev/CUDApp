/* SPDX-License-Identifier: MIT */
#ifndef OMEGA_NATIVE_BENCHMARK_H
#define OMEGA_NATIVE_BENCHMARK_H
#include "omega_gpu_decode_graph.h"
#include "omega_gpu_session.h"
#include <stdio.h>
/* Opt-in collection adapter for an already loaded native graph workload.
 * Does not load a model, open a server, or change arithmetic. Batch is one.
 * Caller serializes graph use and provides the matching correctness trace. */
typedef struct {
    const char *session; /* ASCII letters/digits/._- only, 1..96 bytes */
    uint32_t ordinal, input_tokens, seed;
    int cold, stamps; /* each 0 or 1 */
} OmegaBenchmarkSample;
/* Pure formatter for actual observations; separate for host fixture testing.
 * An enabled timeline needs exactly stats.launches+1 strictly increasing stamps.
 * Unknown counters are omitted, never reported as zero measurements. */
int omega_benchmark_write_graph(FILE *out, const OmegaBenchmarkSample *sample, uint32_t pos, uint32_t token,
    const OmegaGpuDecodeGraphStats *stats, const OmegaGpuAllocStats *before, const OmegaGpuAllocStats *after,
    const uint64_t *timestamps, const uint8_t *kinds, uint32_t launches);
/* Calls real graph_token, records session allocation deltas and optional graph
 * timestamps, then writes one JSONL event. rc<0 on capture/workload failure.
 * Output I/O and allocation-counter reads are outside graph stats.total_ns. */
int omega_benchmark_graph_token(FILE *out, OmegaGpuDecodeGraph *graph, const float *embedding, uint32_t pos,
    const OmegaBenchmarkSample *sample, uint32_t *token);
/* Optional bridge for existing callers of cmdlist_submit, without resubmitting.
 * Graph API does not expose this stats struct; do not fabricate its counters. */
#include "omega_gpu_cmdlist.h"
int omega_benchmark_write_cmdlist(FILE *out, const OmegaGpuCmdListStats *stats);

/* Serving-request bridge: caller supplies monotonic host clock readings around
 * its actual prefill and decode work. No request is sent by this function.
 * context is the final observed token's context; token is its greedy output.
 * A failed/incomplete request must not be submitted as a successful sample. */
typedef struct {
    uint64_t start_ns, prefill_end_ns, first_token_ns, end_ns;
    uint32_t context, output_tokens, token;
} OmegaBenchmarkRequest;
int omega_benchmark_write_request(FILE *out, const OmegaBenchmarkSample *sample, const OmegaBenchmarkRequest *request);
#endif
