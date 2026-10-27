#!/bin/bash
# EXPLORATORY (AB-PLAN "Exploration"): dev clone build, explore set, 2 passes per configuration.
cd /home/drakestapleton/work/GemmTune/dev
git rev-parse HEAD; git status --porcelain
X="nice -n 19 taskset -c 10-14 build/gemm_ab --set explore --arm X"
run() { local name=$1 cached=$2; shift 2; for p in 1 2; do echo "== cfg $name cached $cached pass $p"; AIEN_OMEGA_GEMM_PACK_CACHED=$cached $X --round $p "$@"; done; }
run default-uncached 0
run default-cached 1
run rt4ct4-s4s2 1 --cfg bf16 4,4,2,2,4 --cfg e4m3 4,4,2,2,2
run rt4ct2-s4s2 1 --cfg bf16 4,2,2,2,4 --cfg e4m3 4,2,2,2,2
run rt2ct4-w2x4 1 --cfg bf16 2,4,2,4,4 --cfg e4m3 2,4,2,4,2
run rt4ct4-s2s1 1 --cfg bf16 4,4,2,2,2 --cfg e4m3 4,4,2,2,1
run rt4ct4-uncached 0 --cfg bf16 4,4,2,2,4 --cfg e4m3 4,4,2,2,2
