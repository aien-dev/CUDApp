# SPDX-License-Identifier: MIT
# Host execution on the CPU you have. No NVIDIA headers, no driver, no vendor GPU stack.
#   make hello-host          run the hello program and expect PASS
#   make hello-host-mutant   the same program with one wrong answer; must exit 1
#   make test-targets        the table checks, then both of the above
HOST_TARGET_SRCS = src/cudapp_target.c src/cudapp_host_run.c src/sha256.c
HOST_TARGET_CFLAGS = -std=c11 -Wall -Wextra -Werror -O2 -Isrc -ffp-contract=off

.PHONY: hello-host hello-host-mutant test-targets

$(OUT_DIR)/hello_host: examples/hello_host/hello_host.c $(HOST_TARGET_SRCS) | $(OUT_DIR)
	$(CC) $(HOST_TARGET_CFLAGS) -o $@ examples/hello_host/hello_host.c $(HOST_TARGET_SRCS) -lm

$(OUT_DIR)/test_cudapp_targets: tests/test_cudapp_targets.c $(HOST_TARGET_SRCS) | $(OUT_DIR)
	$(CC) $(HOST_TARGET_CFLAGS) -o $@ tests/test_cudapp_targets.c $(HOST_TARGET_SRCS) -lm

hello-host: $(OUT_DIR)/hello_host
	./$(OUT_DIR)/hello_host

hello-host-mutant: $(OUT_DIR)/hello_host
	./$(OUT_DIR)/hello_host --mutate; test $$? -eq 1

test-targets: $(OUT_DIR)/test_cudapp_targets $(OUT_DIR)/hello_host
	./$(OUT_DIR)/test_cudapp_targets
	./$(OUT_DIR)/hello_host
	./$(OUT_DIR)/hello_host --mutate; test $$? -eq 1
