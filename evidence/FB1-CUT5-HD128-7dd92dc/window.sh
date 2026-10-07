#!/bin/bash
# omega#323 final-head GB10 receipts (d6d82f), head 7dd92dc. hd64 regression runner, then hd128 battery. Never kill.
set -u
R=$HOME/workspace/hive/WATTN/receipts/final-7dd92dc; O=$HOME/workspace/hive/WATTN/omega
T0=$(date -u +%FT%TZ); echo "$T0" > $R/start.txt
{ echo "omega $(git -C $O rev-parse HEAD) dirty=$(git -C $O status --porcelain | wc -l)"; echo "physics $(git -C $O/../physics rev-parse HEAD)";
  nvidia-smi --query-gpu=driver_version --format=csv,noheader; uname -r;
  journalctl -k --no-pager -o short-iso | grep -c 'Xid' | sed 's/^/xid_lines_before=/'; } > $R/ground.txt 2>&1
(cd $O && PHYSICS_DIR=$O/../physics tools/run_gpu_attention_chip.sh $R > $R/hd64-runner.log 2>&1); echo "hd64 rc=$?" >> $R/status.txt
(cd $O && ./build/gpu_attention_test --hd 128 --out $R/receipt-hd128.json > $R/run-hd128.log 2>&1); echo "hd128 rc=$?" >> $R/status.txt
journalctl -k --no-pager -o short-iso --since "$(date -d "$T0" '+%F %T')" > $R/kernel-log.txt 2>&1
date -u +%FT%TZ > $R/end.txt
cd $R && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
