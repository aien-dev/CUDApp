#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Drake Stapleton (AIEN / aien-dev)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# tests/run_nvrm_lifecycle_gates.sh -- M19R nvrm_free()/wait_marker_ge() lifecycle gates.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
. "$SCRIPT_DIR/gb10_lock.sh"

cd "$REPO_ROOT"

echo "=== M19R NVRM Lifecycle Gates ==="
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Host: $(uname -n) ($(uname -m))"

mkdir -p build
echo -e "\n[*] Building lifecycle_gates..."
gcc -O2 -Wall -Wextra -Werror \
  -I./nvrm -I./m16 \
  -I./third_party/nvidia-open-580.173.02/src/common/sdk/nvidia/inc \
  -I./third_party/nvidia-open-580.173.02/kernel-open/common/inc \
  -I./third_party/nvidia-open-580.173.02/kernel-open/nvidia-uvm \
  -I./third_party/nvidia-open-580.173.02/src/nvidia/arch/nvalloc/unix/include \
  nvrm/nvrm.c m16/m16_native.c nvrm/lifecycle_gates.c -o build/lifecycle_gates

echo "    lifecycle_gates built cleanly."

echo -e "\n[*] Running lifecycle_gates under GPU lock ($GB10_LOCK)..."
gb10_lock_acquire
./build/lifecycle_gates
gb10_lock_release

echo -e "\n=== M19R NVRM Lifecycle Gates: done ==="
