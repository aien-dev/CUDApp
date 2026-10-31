#!/bin/bash
# After anchored 'TrainCarry: MERGED #112': fresh clones for arm A (main) and arm B (perf/gemm-tiles), builds, host+sim gates.
set -eu
W=/home/drakestapleton/work/GemmTune; H=/home/drakestapleton/handoffs/GemmTune
F="-std=gnu11 -Wall -Wextra -Werror -D_GNU_SOURCE -O2 -Isrc -I./m16 -I./nvrm -I./third_party/nvidia-open-580.173.02/src/common/sdk/nvidia/inc -I./third_party/nvidia-open-580.173.02/kernel-open/common/inc -I./third_party/nvidia-open-580.173.02/kernel-open/nvidia-uvm -I./third_party/nvidia-open-580.173.02/src/nvidia/arch/nvalloc/unix/include -ffp-contract=off -Itests"
for arm in A B; do
  d=$W/arm$arm; rm -rf $d; git clone -q https://github.com/aien-dev/CUDApp.git $d
done
cd $W/armB && git checkout -q -b perf/gemm-tiles origin/main && git apply --index $H/perf-gemm-tiles.patch
git -c user.name="GemmTune (Drake Stapleton's agent)" -c user.email=drake@aienos.com commit -q -m "gemm: cached packed scratch, banded MAIN grid, rt 4 register opt-in; default rt2 ct4 wm2 wn4 band 8

The packed operands live in a persistent GPU-cached scratch (AIEN_OMEGA_GEMM_PACK_CACHED=0: uncached, the old
behaviour); MAIN walks its tiles in column bands so a band of packed B stays in L2; rt 4 configs opt MAIN into the
hardware register limit. Defaults picked by the GemmTune exploration holds (EXPLORATORY), A/B in evidence/GEMM-TUNE-*."
for arm in A B; do
  cd $W/arm$arm; echo "== arm $arm $(git rev-parse HEAD)"
  nice -n 19 taskset -c 10-14 make -j5 build/libomega_gpu.a build/gpu_gemm_test build/gpu_tc_peak_test build/gpu_qmma_probe_test
  nice -n 19 taskset -c 10-14 cc $F -o build/gemm_ab $H/gemm_ab.c build/libomega_gpu.a -lpthread -lm
done
cd $W/armB && nice -n 19 taskset -c 10-14 make test-gpu-gemm
