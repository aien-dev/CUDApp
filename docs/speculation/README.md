# 27B adaptive sizing experiment

Refs #63, #64, #65 and #66. This is a native Omega component and diagnostic
replay, with synthetic tests. It is not a completed model experiment.

The implementation chooses a chain length using expected committed output
divided by the whole round cost. It keeps ordinary and fixed modes available,
falls back when observed speculation loses, and makes a bounded recovery probe
at most once per eight completed rounds. A separate collector validates and keeps
raw timing observations. The report compares adaptive against base, ordinary and
fixed, checks exact greedy output identity first, and retains failed attempts.

All new implementation and tests are handwritten `.om`, compiled to native
AArch64 through the pinned seed-built compiler. Development reconstruction uses
the existing shell tools disclosed in [REPRODUCE.md](REPRODUCE.md). Those commands
are not an Omega-native build system or a shipped model runner.

Read [CONTRACT.md](CONTRACT.md) for the component and stream interfaces,
[READINESS.md](READINESS.md) for actual dependencies, and
[the receipt](../../evidence/speculation/README.md) for executed checks.

## What remains

The existing model, drafter, accepted-state owner and native benchmark owner
must call these components through a qualified seeded adapter. No new loader,
cache subsystem, GPU runtime or compiler is introduced here. Device completion,
identity discovery, monotonic timing, full-state rollback and actual output
tokens must come from that integration. Caller-supplied metadata alone cannot
prove any of them.

Real 1K/32K trials, performance results, first-divergence analysis and independent
final-source review remain required. No issue is closed by this component.
