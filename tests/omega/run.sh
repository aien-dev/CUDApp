#!/bin/sh
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

# Run the omega compiler harness against this repo's omega/slices.
# Usage: tests/omega/run.sh [FUZZ]  (default 120)
set -eu
W=$HOME/workspace/omega-osc-cubin       # the compiler harness build
R=$(cd "$(dirname "$0")/../.." && pwd)  # this repo checkout
FUZZ=${1:-120}
# Every slice must be present by name: a deleted file is a loud failure here,
# not a silently smaller keyed count in the harness summary.
SLICES="doorbell.osc sequencer.osc nvrm.osc va.osc open_plan.osc kmod.osc mesa.osc launch.osc"
for f in $SLICES; do
  [ -f "$R/omega/slices/$f" ] || { echo "missing slice: omega/slices/$f" >&2; exit 1; }
done
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/slices"
ln -s "$W/tests/compiler/progs" "$T/progs"
ln -s "$W/tests/compiler/neg" "$T/neg"
for f in $SLICES; do cp "$R/omega/slices/$f" "$T/slices/"; done
cd "$W"
./build/compiler/test_osc_compiler "$FUZZ" "$T"
