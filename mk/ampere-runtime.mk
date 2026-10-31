# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# HOST_ONLY simulated RM/UVM and session paths; no /dev/nvidia opens.
AMPERE_RUNTIME_SRCS = src/omega_ampere_qmd.c $(CUDAPP_BINDING_SRCS)
AMPERE_SESSION_SIM_SRCS = src/omega_gpu_session.c src/omega_gpu_platform.c src/omega_gpu_fmt.c src/omega_blackwell_qmd.c

build/test_ampere_runtime: tests/test_ampere_runtime.c nvrm/nvrm.c nvrm/nvrm.h src/omega_ampere_qmd.h $(AMPERE_RUNTIME_SRCS)
	@mkdir -p build
	$(CC) -std=gnu11 -O2 -Wall -Wextra -Werror -D_GNU_SOURCE -Isrc $(GPU_ENGINE_PHYS_INC) -o $@ tests/test_ampere_runtime.c $(AMPERE_RUNTIME_SRCS)

build/test_ampere_session: tests/test_ampere_session.c $(AMPERE_SESSION_SIM_SRCS) src/omega_gpu_session.h src/omega_ampere_qmd.h $(AMPERE_RUNTIME_SRCS)
	@mkdir -p build
	$(CC) -std=gnu11 -O2 -Wall -Wextra -Werror -pthread -Isrc -o $@ tests/test_ampere_session.c $(AMPERE_SESSION_SIM_SRCS) $(AMPERE_RUNTIME_SRCS)

.PHONY: test-ampere-runtime
test-ampere-runtime: build/test_ampere_runtime build/test_ampere_session build/test_omega_gpu_platform_linux
	./build/test_ampere_runtime
	./build/test_ampere_session
	./build/test_omega_gpu_platform_linux
