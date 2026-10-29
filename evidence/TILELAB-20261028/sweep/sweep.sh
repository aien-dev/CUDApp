#!/bin/sh
# group of up to 15 arms per invocation; default is arm 0 in every group (anchor)
T=/tmp/tl/gpu_tilelab_test; O=/tmp/tl/ev/sweep; mkdir -p $O
S="1024,1024,1024;4096,4096,4096;2048,8192,2048;8192,8192,8192"
g=0; args=""; n=0
for wm in 1 2 4; do for wn in 1 2 4; do for band in 0 8 32; do
  args="$args --cfg w${wm}${wn}b${band}:2,4,$wm,$wn,$band"; n=$((n+1))
  if [ $n -eq 14 ]; then g=$((g+1)); $T --shapes "$S" $args --rounds 9 --out $O/g$g.json > $O/g$g.txt 2>&1; echo "g$g rc=$?"; args=""; n=0; fi
done; done; done
args="$args --cfg r1c4:1,4,2,4,8 --cfg r2c2:2,2,2,4,8 --cfg r4c1:4,1,2,4,8 --cfg r4c2:4,2,2,4,8 --cfg r1c2:1,2,2,4,8"
g=$((g+1)); $T --shapes "$S" $args --rounds 9 --out $O/g$g.json > $O/g$g.txt 2>&1; echo "g$g rc=$?"
