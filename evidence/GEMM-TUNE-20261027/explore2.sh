#!/bin/bash
# EXPLORATORY (AB-PLAN "Exploration", amendment 1): banded MAIN grid sweep, packed scratch cached, 2 passes per configuration.
cd /home/drakestapleton/work/GemmTune/dev
git rev-parse HEAD; git status --porcelain; sha256sum /home/drakestapleton/handoffs/GemmTune/gemm_ab.c
X="nice -n 19 taskset -c 10-14 build/gemm_ab --set explore --arm X"
run() { local name=$1; shift; for p in 1 2; do echo "== cfg $name cached 1 pass $p"; AIEN_OMEGA_GEMM_PACK_CACHED=1 $X --round $p "$@"; done; }
for b in 0 4 8 16; do
  run rt2ct4-b$b --cfg bf16 2,4,2,2,4,$b --cfg e4m3 2,4,2,2,2,$b
  run rt4ct4-b$b --cfg bf16 4,4,2,2,4,$b --cfg e4m3 4,4,2,2,2,$b
done
run rt2ct4w2x4-b4 --cfg bf16 2,4,2,4,4,4 --cfg e4m3 2,4,2,4,2,4
run rt2ct4w2x4-b8 --cfg bf16 2,4,2,4,4,8 --cfg e4m3 2,4,2,4,2,8
