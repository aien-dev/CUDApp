#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Run the production architecture refusal. On AArch64 this row is NOT_RUN:
# the binary is not executed, so a device is not opened. A cross compile is
# not a runtime result.
set -eu
host=${TARGET_ARCH:-$(uname -m)}
case "$host" in
  aarch64|arm64)
    echo "x86-64 production refusal: NOT_RUN (this host is $host; the check runs on x86-64 Linux and does not open a device)"
    exit 0
    ;;
esac
bin=${1:?architecture-guard binary}
if [ "${CROSS_COMPILE_ONLY:-0}" = 1 ]; then
  echo "x86-64 production refusal: NOT_RUN (cross compile only, binary $bin)"
  exit 0
fi
[ -x "$bin" ] || {
  echo "x86-64 production refusal: binary is missing or not executable: $bin" >&2
  exit 1
}
"$bin"
