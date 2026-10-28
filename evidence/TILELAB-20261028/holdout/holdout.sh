#!/bin/sh
# Held-out validation of the calibrated tilelab planner: shapes the fit never saw.
# Arms: every rt2 ct4 config with wm,wn in {1,2,4} and band in {8,16,32}, plus band 0 for the 4x4/4x2/2x4 tiles,
# split into runs of at most 15 arms (the default arm is added by the tool to each run).
T=/tmp/tl/gpu_tilelab_test; O=/tmp/tl/ev/holdout; mkdir -p $O
S="3072,3072,3072;4096,1024,4096;1024,8192,8192"
all=""
for wm in 1 2 4; do for wn in 1 2 4; do for band in 8 16 32; do
  [ $wm = 2 ] && [ $wn = 4 ] && [ $band = 8 ] && continue
  all="$all w${wm}${wn}b${band}:2,4,$wm,$wn,$band"
done; done; done
all="$all w44b0:2,4,4,4,0 w42b0:2,4,4,2,0 w24b0:2,4,2,4,0"
i=0; grp=1; args=""
for a in $all; do
  args="$args --cfg $a"; i=$((i+1))
  if [ $i -eq 15 ]; then
    $T --shapes "$S" $args --rounds 9 --out $O/h$grp.json > $O/h$grp.txt 2>&1; echo "h$grp rc=$?"
    grp=$((grp+1)); i=0; args=""
  fi
done
[ -n "$args" ] && { $T --shapes "$S" $args --rounds 9 --out $O/h$grp.json > $O/h$grp.txt 2>&1; echo "h$grp rc=$?"; }
