#!/bin/bash
# AttnGeneral A/B v2 (diagnostic driver). One flock hold, beside the live 35B; defers to a Main WINDOW.
set -u
B=/home/drakestapleton/handoffs/BOARD.md
R=/home/drakestapleton/work/AttnGeneral/cudapp
M=/home/drakestapleton/work/AttnGeneral/main-c30c0be
ts() { date -u +%FT%TZ; }
post() { echo "$(ts) AttnGeneral: $*" >> "$B"; }
health() { apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' '); h=$(curl -s -m 2 127.0.0.1:18091/health); echo "apps=[$apps] health=$h"; }
window_open() { l=$(grep -E '^[0-9TZ:.-]+ Main: WINDOW-(START|END|ABORTED)' "$B" | tail -n 1); [[ "$l" == *WINDOW-START* ]]; }
exec 9>/tmp/aien-gb10.lock
window_open && { echo "window open"; exit 3; }
flock -w 900 9 || { echo "lock not acquired"; exit 3; }
window_open && { echo "window opened"; exit 3; }
pre=$(health); echo "pre: $pre"
case "$pre" in *"apps=[79557 ]"*'"status":"ok"'*) ;; *) post "NOT_RUN ab2: pre-health not clean ($pre)"; exit 4;; esac
D=$(ls -d $R/evidence/ATTN-GENERAL-20261027-*)/ab2; mkdir -p "$D"
post "GPU-START attn-general ab2 (flock, <=5 min, beside live 35B; $D)"
( while :; do t=$(date +%s%N); h=$(curl -s -m 0.5 127.0.0.1:18091/health); f=$(printf '%s' "$h" | grep -o '"in_flight":[0-9]*' | cut -d: -f2); echo "$t ${f:--1}"; sleep 0.03; done ) > "$D/inflight.log" &
P=$!; sleep 0.3
for pair in $(seq 1 40); do
  if (( pair % 2 )); then order="A B"; else order="B A"; fi
  for arm in $order; do
    bin=$R/build/gpu_attention_test; [ $arm = A ] && bin=$M/build/gpu_attention_test
    t0=$(date +%s%N); "$bin" --timing --out "$D/$arm$pair.json" > "$D/$arm$pair.log" 2>&1; rc=$?; t1=$(date +%s%N)
    echo -e "$arm\t$pair\t$t0\t$t1\t$rc" >> "$D/runs.tsv"
  done
done
sleep 0.3; kill $P; wait $P 2>/dev/null
post_h=$(health); echo "$post_h" > "$D/health_post.txt"; echo "$pre" > "$D/health_pre.txt"; echo "post: $post_h"
post "GPU-END attn-general ab2; post: $post_h"
