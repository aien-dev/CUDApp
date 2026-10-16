#!/bin/bash
OM=$HOME/workspace/omega-mm-e2; EV=/home/drakestapleton/workspace/omega-evidence-e2/SMOKE-1661fc72
echo "omega $(git -C $OM rev-parse HEAD) dirty=$(git -C $OM status --porcelain --untracked-files=no | wc -l) binary $(sha256sum $OM/build/gpu_matmul_api_test | cut -c1-64) started $(date -u +%FT%TZ)" > $EV/ground.txt
timeout 600 $OM/build/gpu_matmul_api_test --out $EV/sweep.json > $EV/sweep.log 2>&1; a=$?
timeout 300 $OM/build/gpu_matmul_api_test --mutant --out $EV/mutant.json > $EV/mutant.log 2>&1; b=$?
echo "sweep $a mutant $b finished $(date -u +%FT%TZ)" >> $EV/ground.txt; [ $a -eq 0 ] && [ $b -eq 0 ]
