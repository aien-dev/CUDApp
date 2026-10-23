# Host build glue only. All first-party generators, simulators and gates are Omega.
# Run from the repository root. Nothing here loads a GPU or compiles C/Rust/Python.
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
OUT := build/native-gpulab
OMEGA_NATIVE_COMPILER ?=
PIN := 191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9
ROOT := native/gpulab
PARTS := $(ROOT)/common.om $(ROOT)/amd.om $(ROOT)/agx.om $(ROOT)/request.om
PROGRAMS := test-amd test-agx test-cross test-request emit-main test-wire test-amd-oracle
BINS := $(addprefix $(OUT)/,$(PROGRAMS))
.PHONY: all check-compiler test
all: $(BINS) $(OUT)/request-fixture $(OUT)/amd-oracle-input $(OUT)/mutate-bytes

check-compiler:
	@test -n "$(OMEGA_NATIVE_COMPILER)" || { printf 'Set OMEGA_NATIVE_COMPILER to the qualified seed-native compiler\n' >&2; exit 64; }
	@test -f "$(OMEGA_NATIVE_COMPILER)" && test -x "$(OMEGA_NATIVE_COMPILER)"
	@test "$$(sha256sum "$(OMEGA_NATIVE_COMPILER)" | cut -d ' ' -f1)" = "$(PIN)" || { printf 'Compiler identity refused\n' >&2; exit 65; }
	@test ! -L build && test ! -L "$(OUT)"
	@mkdir -p "$(OUT)"

$(BINS): $(OUT)/%: $(PARTS) $(ROOT)/%.om | check-compiler
	@for p in "$@" "$@.source.om" "$@.new"; do test ! -L "$$p"; if test -e "$$p"; then test -f "$$p" && test "$$(stat -c %h "$$p")" = 1; fi; done
	@cp "$(ROOT)/common.om" "$@.source.om"
	@for p in amd agx request "$*"; do tee -a "$@.source.om" < "$(ROOT)/$$p.om" > /dev/null; done
	@"$(OMEGA_NATIVE_COMPILER)" < "$@.source.om" > "$@.new"
	@test -s "$@.new" && chmod +x "$@.new" && mv "$@.new" "$@"

$(OUT)/request-fixture $(OUT)/amd-oracle-input $(OUT)/mutate-bytes: $(OUT)/%: $(ROOT)/%.om | check-compiler
	@for p in "$@" "$@.new"; do test ! -L "$$p"; if test -e "$$p"; then test -f "$$p" && test "$$(stat -c %h "$$p")" = 1; fi; done
	@"$(OMEGA_NATIVE_COMPILER)" < "$<" > "$@.new"
	@test -s "$@.new" && chmod +x "$@.new" && mv "$@.new" "$@"

# No timing claims. Native gates print their own counts/verdicts and nonzero failures.
test: all
	@printf 'AMD native gate\n'; "$(OUT)/test-amd"
	@printf 'Apple native gate\n'; "$(OUT)/test-agx"
	@printf 'Cross-target exhaustive signed-byte gate\n'; "$(OUT)/test-cross"
	@"$(OUT)/test-request"
	@for target in 1 2; do printf '%s' "$$target" | "$(OUT)/request-fixture" | "$(OUT)/emit-main" | "$(OUT)/test-wire"; done
	@"$(OUT)/amd-oracle-input" | llvm-mc-18 -triple=amdgcn -mcpu=gfx1100 -filetype=obj -o "$(OUT)/amd-oracle.o"
	@llvm-objcopy-18 --dump-section .text="$(OUT)/amd-oracle.bin" "$(OUT)/amd-oracle.o"
	@"$(OUT)/test-amd-oracle" < "$(OUT)/amd-oracle.bin"
	@printf 'NATIVE GPULAB SUITE PASS; HARDWARE NOT_RUN; SPEED NOT_MEASURED\n'
