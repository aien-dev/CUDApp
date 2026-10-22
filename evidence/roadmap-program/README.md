# Roadmap input and passive baseline identity

This is not a performance baseline. It records the actual GitHub issue/PR state, current source/model/binary/driver identity and exact read-only commands before implementation. No inference request, model load, GPU kernel, server restart or driver change was executed.

`tracker-snapshot.json` records issue26, issue5, issues15-25 and Omega PR614-618 with exact heads and current bodies/comments. All five Omega PRs were OPEN at capture. The program base is the existing canonical integration candidate349913f1, which carries these implementations; it is not main23233242 and is not production-qualified.

`passive-live-identity.json` records `ps -eo pid,ppid,stat,etime,args`, `/proc/PID/exe` SHA256, existing BUILD.txt, exact config bytes SHA256 and architectural fields, `/proc/PID/maps` CUDA-userspace matches, `curl -fsS --max-time10 http://127.0.0.1:18091/v1/models`, `uname -a`, and `nvidia-smi --query-gpu=name,uuid,pci.device_id,driver_version --format=csv,noheader` with exit codes/raw outputs. The live binary is not assumed to consume CUDApp just because its userspace maps are CUDA-free.

Target binding matters: the served35B config has hidden2048, MoE intermediate512,256 experts/top8,40 layers, full-attention head256 and separate GDN shapes. The old30B decode graph and q38-27B graph have different shapes and semantics. Their host stage tests may be reused as explicitly named kernel tests, never as complete35B behavior. Benchmark adapters must identify the actual native q36 path/consumer, not label a legacy graph result35B. No arithmetic equivalence or model-quality PASS is implied by this identity capture.

Performance baseline, instrumentation overhead, repeated paired GB10 kernel timing, full-model TTFT/TPOT/throughput, concurrent==alone, turn2 and tool-call correctness remain NOT_RUN pending canonical5 review fixes and a shared train admitted by its existing owner. No made-up measurements stand in for them.

Private x86 Linux userspace QEMU was downloaded/extracted and its package checksum verified without install/binfmt/service changes. A statically cross-compiled x86 guest executed an integer check successfully. This proves an available functional executor, not CUDApp host-suite PASS and not native-speed data. Package metadata, package/tool/guest hashes and raw smoke output stay in the handoff x86-runtime directory until full host-suite evidence is attached to issue15's exact source.
