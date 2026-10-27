# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# R2 (omega-mv#595): "Hello GPU" from a fresh checkout.  `make hello-gpu` builds examples/hello_gpu and runs it.
# Picked up by the `-include mk/*.mk` line at the top of the Makefile. The driver client is in this
# repository (nvrm/, m16/), so nothing is fetched.
HELLO_GPU = $(OUT_DIR)/hello_gpu

.PHONY: hello-gpu hello-gpu-build hello-gpu-control
$(HELLO_GPU): examples/hello_gpu/hello_gpu.c $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -o $@ examples/hello_gpu/hello_gpu.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

hello-gpu-build: $(HELLO_GPU)

hello-gpu: hello-gpu-build $(DOCTOR)
	$(DOCTOR)
	./$(HELLO_GPU)

# The negative control for hello-gpu on the chip: `hello_gpu --bug=short` tells the last launch one element too
# few (nothing is written outside a buffer). The check must catch the one unwritten answer: exit 1 and FAIL.
# This target passes only when it does; a PASS from the broken run would mean the check checks nothing.
hello-gpu-control: hello-gpu-build
	@./$(HELLO_GPU) --bug=short > $(OUT_DIR)/hello_gpu_control.log 2>&1; rc=$$?; cat $(OUT_DIR)/hello_gpu_control.log; \
	if [ $$rc -eq 1 ] && grep -q '^checked 1048576 of 1048576 elements: 1 wrong, 1 never written$$' $(OUT_DIR)/hello_gpu_control.log \
	   && grep -qx FAIL $(OUT_DIR)/hello_gpu_control.log; then echo "NEGATIVE CONTROL hello-gpu --bug=short: RED as it must (exit 1, the one unwritten answer caught)"; \
	else echo "NEGATIVE CONTROL hello-gpu --bug=short: NOT CAUGHT (exit $$rc)"; exit 1; fi
