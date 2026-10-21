# R2 (omega-mv#595): "Hello GPU" from a fresh checkout.  `make hello-gpu` builds examples/hello_gpu and runs it.
# Picked up by the `-include mk/*.mk` line at the top of the Makefile, so no further include is needed.
#
# Physics: when PHYSICS_DIR is not set by the caller (environment or command line), the pinned copy is fetched
# into $(OUT_DIR)/physics by tools/runtime/fetch-physics.sh and used; the usual check-physics-lock then
# confirms it is the commit in physics.lock. Set PHYSICS_DIR to use your own checkout instead.
HELLO_GPU = $(OUT_DIR)/hello_gpu

.PHONY: hello-gpu hello-gpu-build
$(HELLO_GPU): examples/hello_gpu/hello_gpu.c $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -o $@ examples/hello_gpu/hello_gpu.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

hello-gpu-build: check-physics-lock $(HELLO_GPU)

ifeq ($(origin PHYSICS_DIR),file)
hello-gpu:
	@pd=$$(tools/runtime/fetch-physics.sh "$(OUT_DIR)/physics") && $(MAKE) --no-print-directory PHYSICS_DIR="$$pd" hello-gpu-build
	./$(HELLO_GPU)
else
hello-gpu: hello-gpu-build
	./$(HELLO_GPU)
endif
