#!/bin/sh
T=/tmp/tl/gpu_tilelab_test; O=/tmp/tl/ev/sweep; S="1024,1024,1024;4096,4096,4096;2048,8192,2048;8192,8192,8192"
args=""; n=0
for wm in 1 2 4; do for wn in 1 2 4; do for band in 0 8 32; do
  n=$((n+1)); [ $n -le 14 ] && continue
  args="$args --cfg w${wm}${wn}b${band}:2,4,$wm,$wn,$band"
done; done; done
$T --shapes "$S" $args --rounds 9 --out $O/g2.json > $O/g2.txt 2>&1; echo "g2 rc=$?"
$T --shapes "$S" --cfg r1c4:1,4,2,4,8 --cfg r2c2:2,2,2,4,8 --cfg r4c1:4,1,2,4,8 --cfg r4c2:4,2,2,4,8 --cfg r1c2:1,2,2,4,8 --rounds 9 --out $O/g3.json > $O/g3.txt 2>&1; echo "g3 rc=$?"
