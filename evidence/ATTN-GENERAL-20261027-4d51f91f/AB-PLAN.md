AttnGeneral A/B plan (predeclared before any timing; 2026-10-27)

Question: does branch feat/attn-general (4d51f91) make the tuned attention path slower than main c30c0be?

Static part (done before timing): the 252 tuned kernels (hd 64/128 x f32/bf16 x log2bs 0..8 x log2gqa 0..6) from
omega_gpu_attention_codegen_hd are byte-identical on both builds (sha256 list bb54fd5c...fcd on both). So any difference
can only come from host-side routing code.

Timed part:
- Binaries: A = /home/drakestapleton/work/AttnGeneral/main-c30c0be/build/gpu_attention_test,
  B = /home/drakestapleton/work/AttnGeneral/cudapp/build/gpu_attention_test (4d51f91).
- Workload: `--timing --out <json>` (tuned gqa_f32 32q/4kv hd64 ctx2048, 100 timed calls per run, warm-up untimed).
- Order: 10 pairs, alternating ABAB... (A first in odd pairs, B first in even pairs), one hold of /tmp/aien-gb10.lock,
  beside the live 35B.
- Per-run value: median_ms from the run JSON.
- Discard rule: a run is dropped if any 18091 /health poll in [run start, run end] shows in_flight > 0, a poll gap > 200 ms,
  or no bracketing poll (DecodeKernels tag.py rule). Poller: abrun.sh-style, 30 ms.
- Statistic: Wolfram Median of clean A and clean B run medians and ratio B/A. Verdict "no slower" if ratio <= 1.05
  (5% noise band, predeclared) AND the static digest identity holds; otherwise report the number as measured.
- If fewer than 3 clean runs per arm: report NOT_RUN for the timed part (static identity still stands).
