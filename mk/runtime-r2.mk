# SPDX-License-Identifier: AGPL-3.0-only
# R2 (omega-mv#595): "Hello GPU" from a fresh checkout.  `make hello-gpu` builds examples/hello_gpu and runs it.
# Picked up by the `-include mk/*.mk` line at the top of the Makefile. The driver client is in this
# repository (nvrm/, m16/), so nothing is fetched.
HELLO_GPU = $(OUT_DIR)/hello_gpu

.PHONY: hello-gpu hello-gpu-build
$(HELLO_GPU): examples/hello_gpu/hello_gpu.c $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -o $@ examples/hello_gpu/hello_gpu.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

hello-gpu-build: $(HELLO_GPU)

hello-gpu: hello-gpu-build $(DOCTOR)
	$(DOCTOR)
	./$(HELLO_GPU)
