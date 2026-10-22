#!/bin/sh
# Run the production architecture refusal on a non-AArch64 Linux host.
set -eu
R=$(cd "$(dirname "$0")/../.." && pwd)
M=${TARGET_ARCH:-$(uname -m)}
case "$M" in
  aarch64|arm64)
    echo "x86-64 Linux runtime guard: NOT_RUN (this host is $M; cross compile is not runtime proof)"
    exit 0
    ;;
esac
OUT=${OUT_DIR:-build}
CC_BIN=${CC:-gcc}
TP=$R/third_party/nvidia-open-580.173.02
mkdir -p "$OUT"
"$CC_BIN" -std=gnu11 -O2 -Wall -Wextra -Werror -D_GNU_SOURCE -pthread \
  -ffunction-sections -fdata-sections -I"$R/src" -I"$R/tests" \
  -I"$R/m16" -I"$R/nvrm" \
  -I"$TP/src/common/sdk/nvidia/inc" -I"$TP/kernel-open/common/inc" \
  -I"$TP/kernel-open/nvidia-uvm" -I"$TP/src/nvidia/arch/nvalloc/unix/include" \
  -o "$OUT/omega_blackwell_arch_guard_test" \
  "$R/tests/test_omega_blackwell_arch_guard.c" "$R/tests/fake_m16_native.c" \
  "$R/src/omega_blackwell_engine.c" "$R/src/omega_blackwell_submit.c" \
  "$R/src/omega_gpu_engine.c" "$R/src/omega_blackwell_matmul.c" \
  "$R/src/omega_blackwell_codegen.c" "$R/src/omega_blackwell_encoder.c" \
  "$R/src/omega_blackwell_qmd.c" "$R/src/omega_blackwell_realize.c" \
  "$R/src/omega_vector.c" "$R/src/omega_core.c" "$R/src/omega_canonical.c" \
  "$R/src/sha256.c" -Wl,--gc-sections -lm
if [ "${CROSS_COMPILE_ONLY:-0}" = 1 ]; then
  echo "x86-64 Linux runtime guard: NOT_RUN (cross compile only)"
  exit 0
fi
"$OUT/omega_blackwell_arch_guard_test"
