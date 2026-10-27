# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Developer bar D3, D4, D7, D10, D12 (docs/PAIN_POINTS.md): hello_gpu --sim runs the same kernel program
# on the host warp simulator; tests/dx/hello_sim_test.sh checks the answers, the fault report, the work
# line and that two runs give the same bits. Host only: --sim never opens a device.
.PHONY: test-hello-sim
test-hello-sim: $(OUT_DIR)/hello_gpu $(OUT_DIR)/hello_gpu_control
	sh tests/dx/hello_sim_test.sh $(OUT_DIR)/hello_gpu $(OUT_DIR)/hello_gpu_control
