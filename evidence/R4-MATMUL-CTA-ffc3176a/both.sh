#!/bin/bash
/home/drakestapleton/workspace/omega-evidence-e2/R4-ffc3176a/run1/run.sh; r1=$?
/home/drakestapleton/workspace/omega-evidence-e2/R4-ffc3176a/run2/run.sh; r2=$?
echo "run1 $r1 run2 $r2"; [ $r1 -eq 0 ] && [ $r2 -eq 0 ]
