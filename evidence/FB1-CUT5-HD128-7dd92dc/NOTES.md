# omega#323 GB10 receipts on the rebased head (session d6d82f, 2026-10-07)
Code under test: omega 7dd92dc (omega#323 rebased onto omega main 917e8b9, which carries omega#324's 2 KB code-buffer tail). Physics 6d7cf0d. Driver 580.173.02. Tree clean (ground.txt).
Hold 03:14:25Z-03:14:38Z via quietlock (window.sh in this folder). Nothing rerun or edited after the run.
- hd64 regression (tools/run_gpu_attention_chip.sh, FB1-CUT5-7dd92dc/): run 122/0, sweep 73/0, timing 2/0, sim 122/0.
- hd128 battery (gpu_attention_test --hd 128): PASS 126/0, receipt-hd128.json; every ctx256 case passes (Qwen3-4B shape 32q/8kv ctx256 worst_scaled_err 0.0022).
- kernel-log.txt: no Xid during the hold.
- The earlier FAIL receipts in evidence/FB1-CUT5-HD128-5a9f759 (ctx256 Xid 31, pre-#324) are kept unchanged. Cause and red/green: omega#324 evidence/CODEPAD-PREFETCH-f1d57c5.
- LIMITS: one pass of each battery, not a rate. Commits after 7dd92dc add only this folder and the crumb recompile.
