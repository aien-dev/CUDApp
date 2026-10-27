#!/bin/bash
# Frozen AB-PLAN amendment 3 supplementary run S: rounds given as arguments, arm order from the seed rule, one process per arm per round.
H=/home/drakestapleton/handoffs/GemmTune; W=/home/drakestapleton/work/GemmTune
sha256sum $H/AB-PLAN.md $H/gemm_ab.c
for arm in A B; do echo "# arm $arm $(git -C $W/arm$arm rev-parse HEAD) dirty=$(git -C $W/arm$arm status --porcelain | wc -l)"; done
for r in "$@"; do
  if (( (r * 2654435761 + 20261027) % 2 == 0 )); then order="A B"; else order="B A"; fi
  echo "== round $r order $order"
  for arm in $order; do
    echo "== arm $arm round $r load $(cat /proc/loadavg)"
    (cd $W/arm$arm && taskset -c 10-14 build/gemm_ab --spin-us 300000 --set ab --arm $arm --round $r --seed 20261027) || echo "== arm $arm round $r exit $?"
  done
done
