#!/bin/bash
# AttnGeneral chip hold (diagnostic driver, not product code). One flock hold, <= 14 min, beside the live 35B.
set -u
B=/home/drakestapleton/handoffs/BOARD.md
R=/home/drakestapleton/work/AttnGeneral/cudapp
M=/home/drakestapleton/work/AttnGeneral/main-c30c0be
ts() { date -u +%FT%TZ; }
post() { echo "$(ts) AttnGeneral: $*" >> "$B"; }
health() { apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' '); h=$(curl -s -m 2 127.0.0.1:18091/health); echo "apps=[$apps] health=$h"; }
window_open() { # 1 if the last Main WINDOW line is START
  l=$(grep -E '^[0-9TZ:.-]+ Main: WINDOW-(START|END|ABORTED)' "$B" | tail -n 1); [[ "$l" == *WINDOW-START* ]]; }
exec 9>/tmp/aien-gb10.lock
for try in 1 2 3 4 5 6; do
  if window_open; then echo "window open, waiting"; sleep 300; continue; fi
  flock -w 900 9 && break
done
flock -n 9 || { echo "lock not acquired"; exit 3; }
if window_open; then echo "window opened while waiting; releasing"; exit 3; fi
pre=$(health); echo "pre: $pre"
case "$pre" in *"apps=[79557 ]"*'"status":"ok"'*) ;; *) post "NOT_RUN chip hold: pre-health not clean ($pre)"; exit 4;; esac
HEAD=$(git -C "$R" rev-parse --short=8 HEAD)
E=$R/evidence/ATTN-GENERAL-20261027-$HEAD; mkdir -p "$E"
post "GPU-START attn-general chip (flock /tmp/aien-gb10.lock, <=14 min, beside live 35B; evidence $E)"
START=$(date +%s)
cd "$R"
echo "$pre" > "$E/health_pre.txt"
./build/gpu_attention_test --general --out "$E/general.json" > "$E/general.log" 2>&1; echo "general rc=$?" | tee -a "$E/rc.txt"
./build/gpu_attention_test --out "$E/hd64.json" > "$E/hd64.log" 2>&1; echo "hd64 rc=$?" | tee -a "$E/rc.txt"
./build/gpu_attention_test --hd 128 --out "$E/hd128.json" > "$E/hd128.log" 2>&1; echo "hd128 rc=$?" | tee -a "$E/rc.txt"
./build/gpu_attention_test --sweep --out "$E/sweep.json" > "$E/sweep.log" 2>&1; echo "sweep rc=$?" | tee -a "$E/rc.txt"
# A/B per AB-PLAN.md
D=$E/ab; mkdir -p "$D"
( while :; do t=$(date +%s%N); h=$(curl -s -m 0.5 127.0.0.1:18091/health); f=$(printf '%s' "$h" | grep -o '"in_flight":[0-9]*' | cut -d: -f2); echo "$t ${f:--1}"; sleep 0.03; done ) > "$D/inflight.log" &
P=$!; sleep 0.3
for pair in 1 2 3 4 5 6 7 8 9 10; do
  if [ $(( $(date +%s) - START )) -gt 780 ]; then echo "time cap reached at pair $pair" >> "$D/runs.tsv"; break; fi
  if (( pair % 2 )); then order="A B"; else order="B A"; fi
  for arm in $order; do
    bin=$R/build/gpu_attention_test; [ $arm = A ] && bin=$M/build/gpu_attention_test
    t0=$(date +%s%N); "$bin" --timing --out "$D/$arm$pair.json" > "$D/$arm$pair.log" 2>&1; rc=$?; t1=$(date +%s%N)
    echo -e "$arm\t$pair\t$t0\t$t1\t$rc" >> "$D/runs.tsv"
  done
done
sleep 0.3; kill $P; wait $P 2>/dev/null
post_h=$(health); echo "$post_h" > "$E/health_post.txt"; echo "post: $post_h"
post "GPU-END attn-general chip; $(tr '\n' ' ' < "$E/rc.txt"); post: $post_h"
