# SPDX-License-Identifier: MIT
NATIVE_BENCH_TEST = $(OUT_DIR)/native_benchmark_adapter_test
$(NATIVE_BENCH_TEST): tests/native_benchmark_adapter_test.c src/omega_native_benchmark.c src/omega_native_benchmark.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -o $@ tests/native_benchmark_adapter_test.c src/omega_native_benchmark.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
.PHONY: test-native-benchmark
test-native-benchmark: $(NATIVE_BENCH_TEST) $(OUT_DIR)/native_benchmark_collect_test
	./$(NATIVE_BENCH_TEST)
	./$(OUT_DIR)/native_benchmark_collect_test
	CUDAPP_BENCH_ADAPTER=$(abspath $(NATIVE_BENCH_TEST)) node --test tests/native_benchmark_test.mjs
test: test-native-benchmark
NATIVE_COLLECT_TEST = $(OUT_DIR)/native_benchmark_collect_test
$(NATIVE_COLLECT_TEST): tests/native_benchmark_collect_test.c src/omega_native_benchmark.c src/omega_native_benchmark.h src/omega_gpu_numerical_contract.c | $(OUT_DIR)
	$(CC) $(CFLAGS) -ffp-contract=off -o $@ tests/native_benchmark_collect_test.c src/omega_native_benchmark.c src/omega_gpu_numerical_contract.c -lm
$(OUT_DIR)/libomega_native_benchmark.a: $(OUT_DIR)/omega_native_benchmark.o
	ar rcs $@ $^
.PHONY: native-benchmark-adapter
native-benchmark-adapter: $(OUT_DIR)/libomega_native_benchmark.a
