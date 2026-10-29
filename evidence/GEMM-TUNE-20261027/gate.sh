#!/bin/bash
# Arm B correctness gate + ceilings at the branch head (one hold).
cd /home/drakestapleton/work/GemmTune/armB
echo "# head $(git rev-parse HEAD) dirty(code)=$(git status --porcelain -- src tests mk Makefile | wc -l)"
echo "## gpu_tc_peak_test"; ./build/gpu_tc_peak_test; echo "## tc_peak exit $?"
echo "## gpu_qmma_probe_test"; ./build/gpu_qmma_probe_test; echo "## qmma_probe exit $?"
echo "## gpu_gemm_test (full chip tier, defaults)"; ./build/gpu_gemm_test --ceiling ${GEMM_CEIL:-120.58,238.65}; echo "## gemm full exit $?"
