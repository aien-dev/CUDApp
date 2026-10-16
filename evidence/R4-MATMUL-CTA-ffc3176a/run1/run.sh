#!/bin/bash
# R4 sealed receipt run (omega #473). Run from the omega worktree under quietlock.
set -u
EV="$(cd "$(dirname "$0")" && pwd)"
OM=$HOME/workspace/omega-mm-e2
{ echo "omega $(git -C $OM rev-parse HEAD) dirty=$(git -C $OM status --porcelain --untracked-files=no | wc -l)";
  echo "physics $(git -C $HOME/workspace/physics-e2-pin rev-parse HEAD) lock $(cat $OM/physics.lock)";
  echo "binary $(sha256sum $OM/build/gpu_matmul_r4_receipt | cut -c1-64)";
  echo "started $(date -u +%FT%TZ) host $(hostname) kernel $(uname -r)"; } > "$EV/ground.txt"
ps aux --sort=-%cpu | head -12 | cut -c1-160 > "$EV/ps-before.txt"; free -g > "$EV/free-before.txt"
timeout 1100 "$OM/build/gpu_matmul_r4_receipt" --out "$EV/receipt.json" "$@" > "$EV/receipt.log" 2>&1
rc=$?
echo "exit $rc finished $(date -u +%FT%TZ)" >> "$EV/ground.txt"
ps aux --sort=-%cpu | head -12 | cut -c1-160 > "$EV/ps-after.txt"; free -g > "$EV/free-after.txt"
exit $rc
