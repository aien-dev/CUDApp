#!/bin/bash
# EXPLORATORY (AB-PLAN "Exploration", amendment 1): warp-grid and stall sweep around rt2 ct4 wm2 wn4 band 8, cached scratch, 2 passes.
cd /home/drakestapleton/work/GemmTune/dev
git rev-parse HEAD; git status --porcelain; sha256sum /home/drakestapleton/handoffs/GemmTune/gemm_ab.c
X="nice -n 19 taskset -c 10-14 build/gemm_ab --set explore --arm X"
run() { local name=$1; shift; for p in 1 2; do echo "== cfg $name cached 1 pass $p"; AIEN_OMEGA_GEMM_PACK_CACHED=1 $X --round $p "$@"; done; }
run w2x4-b8-s4s2 --cfg bf16 2,4,2,4,4,8 --cfg e4m3 2,4,2,4,2,8
run w2x4-b8-s2s1 --cfg bf16 2,4,2,4,2,8 --cfg e4m3 2,4,2,4,1,8
run w2x4-b8-s8s4 --cfg bf16 2,4,2,4,8,8 --cfg e4m3 2,4,2,4,4,8
run w4x2-b8 --cfg bf16 2,4,4,2,4,8 --cfg e4m3 2,4,4,2,2,8
run w4x4-b8 --cfg bf16 2,4,4,4,4,8 --cfg e4m3 2,4,4,4,2,8
run rt4ct4-w2x4-b8 --cfg bf16 4,4,2,4,4,8 --cfg e4m3 4,4,2,4,2,8
run rt4ct2-w2x4-b8 --cfg bf16 4,2,2,4,4,8 --cfg e4m3 4,2,2,4,2,8
run w2x4-b16 --cfg bf16 2,4,2,4,4,16 --cfg e4m3 2,4,2,4,2,16
run w2x4-b2 --cfg bf16 2,4,2,4,4,2 --cfg e4m3 2,4,2,4,2,2
