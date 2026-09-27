#!/bin/bash
# tests/run_nvrm_lifecycle_gates.sh -- M19R nvrm_free()/wait_marker_ge() lifecycle gates.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
LOCK_FILE=/tmp/aien-gb10.lock

cd "$REPO_ROOT"

echo "=== M19R NVRM Lifecycle Gates ==="
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Host: $(uname -n) ($(uname -m))"

echo -e "\n[*] Building lifecycle_gates..."
gcc -O2 -Wall -Wextra -Werror \
  -I./nvrm -I./m16 \
  -I./third_party/nvidia-open-580.173.02/src/common/sdk/nvidia/inc \
  -I./third_party/nvidia-open-580.173.02/kernel-open/common/inc \
  -I./third_party/nvidia-open-580.173.02/kernel-open/nvidia-uvm \
  -I./third_party/nvidia-open-580.173.02/src/nvidia/arch/nvalloc/unix/include \
  nvrm/nvrm.c m16/m16_native.c nvrm/lifecycle_gates.c -o nvrm/lifecycle_gates

echo "    lifecycle_gates built cleanly."

echo -e "\n[*] Running lifecycle_gates under GPU lock ($LOCK_FILE)..."
flock "$LOCK_FILE" ./nvrm/lifecycle_gates

echo -e "\n=== M19R NVRM Lifecycle Gates: done ==="
