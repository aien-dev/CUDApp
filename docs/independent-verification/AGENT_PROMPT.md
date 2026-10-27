# Agent prompt: finish independent native GPU replication

Work in public repository https://github.com/aien-dev/CUDApp. Extend existing
issue #91 and its independent physical replication addendum, under release
review #88. Do not create a duplicate issue or take over #79's .om migration,
#15/#16/#17's qualification owners, trace work or shader pilot #86.

The goal is an independently reproducible, narrowly scoped physical GB10
vector result with inspectable evidence. The documentation and collection
schema in `docs/independent-verification/` prepare that work; they are not a
hardware receipt or a completed checker. Read the README and PLAN there first.

Reconcile current main, open PRs, owners and leases before editing. Read the
repository AGENTS.md, relevant .crumb files and local `.agents/skills` if a
needed skill is absent from your catalog. Follow the current Omega Harness,
encoding, communication and crumb guidance. Work in an isolated branch. Keep
unrelated changes and immutable evidence intact. Open or update a draft PR;
do not merge, release, deploy or change repository settings.

Known historical evidence: `evidence/speculation/release-candidate-teardown/receipt.json`
records CUDApp `2e2421694f6110bd84885ae0b2d3caf249809d34`, compiler source
`omega-mv@d7390d88d07dbf48cfbb0583070599c350f814d3`, and compiler SHA256
`9749f38841a55917d41924a68bce31463ec2339910bddd5e6cbf6ea4420551db`.
It reports an owner-operated GB10 pass, privately retained raw syscall trace,
and independent hardware replication NOT_RUN. Verify the receipt and current
source rather than treating this historical pin as qualification of main.
NVIDIA open kernel modules 580.173.02 and GSP firmware remain dependencies.
The recorded machine had the toolkit installed; not using it is not absence.

Implement the missing test capabilities in seeded .om using the existing
`native/nvrm/context.om`, `native/blackwell/vector.om`,
`native/blackwell/launch.om` and `tests/speculation/` harness. Do not build a
second encoder/runtime or add a permanent C/Python wrapper implementation.
Independent external test oracles remain clearly separate from the product.
Do not compose experimental trace emitters into this qualification path.

Before hardware use, write and exercise host tests for externally supplied
input/output transport, malformed or truncated inputs, supported size bounds,
stale output, wrong digest, missing writes and independent-oracle tampering.
The current harness uses fixed salts and an in-process CPU checker, checks
only available trailing words, and lacks an explicit no-submission arm.
Repair those gaps without disabling compiler loop limits, admission, bounds,
cleanup or existing test gates. If a required language feature is missing,
report the smallest reproducible need to its existing owner instead of adding
a hidden foreign fallback. Obtain independent review of the final code.

Arrange a real operator independent of candidate generation. Freeze the exact
candidate, harness, compiler, reviewed kernel variants and launch-builder
hashes before that operator chooses fresh unpredictable challenges. Runtime
launch descriptors/arguments depend on sizes and allocation addresses: hash
the finalized bytes after challenge selection/allocation and before each
submission, binding them to the frozen builder and exact inputs. Keep expected results in a separate CPU
oracle process/workspace and verify every output byte. Record custody,
timestamps and a witnessed challenge commitment, not just a hard-coded seed.
A source-reviewing agent is not itself an independent physical operator.

No device work is authorized until the owner explicitly grants the existing
reservation and safe admission for the selected machine and bounded workload.
If that authority or supported hardware is missing, complete host work and
report the exact pending action; leave all hardware rows NOT_RUN. Do not stop
a resident model, touch other jobs, uninstall CUDA, change drivers/firmware,
unload modules or reset/reboot the machine. A toolkit-absent test may run only
in a separately approved environment, never by altering the user's Spark.

In an admitted run, execute the correct kernel, a reviewed safe wrong-math
kernel, a fresh restored correct kernel, and a no-submission control that
leaves poisoned output unchanged and never waits for an unsubmitted launch.
Choose poison independently and check it against the oracle; do not leak
expected values by publishing their reversible complement as poison.
Check inputs and guards in every arm. Do not discard failed attempts or
repeat runs selectively to report only successes; diagnose failures and retain
all attempts. Use the existing cleanup path and owner recovery process if state is
uncertain. Capture exact commands, UTC times, exits, full output, source and
artifact digests, device/driver identity, bounded-memory admission and health.

Keep static ELF/build dependency evidence separate from dynamic process-tree,
load/device-call/submission/completion observations. Capture full raw logs and
correlate them to the exact kernel and readback. Static linking, an ioctl,
matching hashes, signatures or a printed PASS alone cannot prove GPU work.
Disclose the observer's coverage and the kernel/firmware floor. Distinguish
installed tools, build/oracle tools and runtime dependencies.

Create a new immutable public evidence bundle. Sanitize and manually review
all new text, logs, commands and artifacts before publication. Retain raw
originals in authorized custody, record raw and sanitized hashes, document
redactions and preserve event ordering and consistent aliases. Do not expose
private paths, addresses, service details, prompts, credentials or vendor
binaries/listings. Record exact-source license/provenance obligations and
coordinate unresolved findings with #89; do not assume the whole current
tree has one license or relicense source. Missing critical public evidence
stays incomplete; a hash
of a private trace is not public inspectability.

Fill an operator report, not the NOT_RUN template. The manifest is only a
collection envelope: COLLECTED is not PASS and claim_status stays
NOT_EVALUATED. Independently verify artifact existence, digests, references,
chronology, admission and semantic control outcomes. The operator describes
what they personally observed, their relationship to the author and remaining
trust assumptions; a separate reviewer accepts only the supported scope.
Record a second operator on the same device separately from an optional
second-machine replication. Neither can be invented from synthetic fixtures.

Run the repository's complete applicable host checks on the final commit;
report every failure by name. Requalify after executable inputs change. The
known x86_64 `ds` assembler failure in `src/omega_blackwell_engine.c` is not
permission to replace barriers with no-ops. Keep host/simulator/native host,
owner hardware, independent hardware and second-machine results separate.

Return the draft PR and issue links, exact final commits and artifact hashes,
commands/exits, independent review, a public evidence index, remaining
NOT_RUN/BLOCKED rows and the next required owner action. Do not auto-close
#91 or claim full-model correctness, performance superiority, all-GPU support,
C retirement or independence from NVIDIA kernel modules/GSP firmware.
