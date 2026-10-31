#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# L136 correctness gates on the chip. $1 = existing evidence dir (absolute).
# Takes the GB10 lock itself for the set; do not wrap in another hold of the lock.
# Record the first failed child and stop with its exit status.
set -eu
E=${1:?usage: tools/l136_channel_gates.sh EVIDENCE_DIR}
N=$HOME/workspace/omega-gpu-channels/build
B=$HOME/workspace/omega-gpu-channels-base/build
LOCK=${LOCK:-/tmp/aien-gb10.lock}
cd "$HOME/workspace/omega-gpu-channels" || exit 1
if [ -e "$LOCK" ]; then exec 9<"$LOCK"; else exec 9>>"$LOCK"; fi
flock -w 900 9 || { echo 'NOT_RUN: GB10 lock busy for 900 s' >&2; exit 2; }
{ date -u; git rev-parse HEAD; uname -r; grep MemAvailable /proc/meminfo; cat /proc/loadavg; } > "$E/device.txt"
run() {
    name=$1; shift
    rc=0
    "$@" > "$E/$name.log" 2>&1 || rc=$?
    printf '%s rc=%s %s\n' "$name" "$rc" "$(tail -1 "$E/$name.log")" >> "$E/summary.txt"
    return "$rc"
}
unset AIEN_OMEGA_GPU_CHANNELS
# G1: one channel, the new build against the base build (0010f7d3 on physics 6d7cf0d4), same arguments
run g1-base-q38-l4 "$B/gpu_q38_graph_test" --layers 4 --mask 0x8 --reps 3
run g1-new-q38-l4  "$N/gpu_q38_graph_test" --layers 4 --mask 0x8 --reps 3
run g1-base-q38-l2 "$B/gpu_q38_graph_test" --reps 3
run g1-new-q38-l2  "$N/gpu_q38_graph_test" --reps 3
run g1-base-cmdlist "$B/gpu_cmdlist_test"
run g1-new-cmdlist  "$N/gpu_cmdlist_test"
run g1-base-dgraph "$B/gpu_decode_graph_test"
run g1-new-dgraph  "$N/gpu_decode_graph_test"
# refusal: no AIEN_OMEGA_GPU_CHANNELS, the pair is refused by name, the serial path decodes
run refusal "$N/gpu_q38_channels_test" --layers 4 --steps 2 --block 0 --refusal
# G2 small: two channels, 4 layers, 16 steps + 16-step blocks
AIEN_OMEGA_GPU_CHANNELS=2 run g2-l4 "$N/gpu_q38_channels_test" --layers 4 --steps 16 --block 16
# two-channel device: the one-channel tests still pass on queue 0
AIEN_OMEGA_GPU_CHANNELS=2 run g1-ch2-q38-l4 "$N/gpu_q38_graph_test" --layers 4 --mask 0x8 --reps 3
echo done >> "$E/summary.txt"
