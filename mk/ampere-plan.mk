# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Host checks never open a driver. Device targets build only; run.py requires --device.
AMPERE_MATH_DEVICE = $(OUT_DIR)/ampere_math_device
AMPERE_MEMORY_REFUSAL = $(OUT_DIR)/ampere_memory_refusal

.PHONY: ampere-device-build test-ampere-plan
$(AMPERE_MATH_DEVICE): tests/ampere/math_device.c src/omega_ampere_codegen.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -o $@ $< $(OUT_DIR)/libomega_gpu.a -lpthread -lm

$(AMPERE_MEMORY_REFUSAL): tests/ampere/memory_refusal.c src/omega_gpu_platform_linux.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -Wl,--wrap=ioctl -o $@ $< $(OUT_DIR)/libomega_gpu.a -lpthread -lm

ampere-device-build: $(AMPERE_MATH_DEVICE) $(AMPERE_MEMORY_REFUSAL) $(OUT_DIR)/hello_gpu $(OUT_DIR)/hello_gpu_control

test-ampere-plan: ampere-device-build
	python3 tests/ampere/test_doctor.py
	$(AMPERE_MATH_DEVICE) --self-test
	$(AMPERE_MATH_DEVICE) --emit-only
	$(OUT_DIR)/hello_gpu --target=sm_86 --sim
