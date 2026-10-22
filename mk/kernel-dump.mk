# SPDX-License-Identifier: MIT
.PHONY: test-kernel-dump test-isolation-dump
$(OUT_DIR)/kernel_dump_test: tests/kernel_dump_test.c src/omega_blackwell_codegen.c src/sha256.c | $(OUT_DIR)
	$(CC) $(CFLAGS) -ffunction-sections -fdata-sections -Wl,--gc-sections -o $@ $^
test-kernel-dump: $(OUT_DIR)/kernel_dump_test
	./$(OUT_DIR)/kernel_dump_test
test-isolation-dump: $(OUT_DIR)/gpu_isolation_test
	sh tests/test_isolation_dump.sh ./$(OUT_DIR)/gpu_isolation_test
