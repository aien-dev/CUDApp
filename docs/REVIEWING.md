# Reviewing CUDApp changes

**Status:** PROPOSED (2026-10-27). The rules in section 1 restate what already binds agent lanes. Section 2
takes effect only when Drake activates `docs/review/CODEOWNERS.proposed` and the matching GitHub ruleset.

## 1. Every pull request

1. **Draft first.** Open as a draft. No `Signed-off-by:` line is required (Drake declined DCO sign-off).
2. **Evidence on the exact head.** Fill in `.github/PULL_REQUEST_TEMPLATE.md`. Each gate is PASS, FAIL or
   NOT RUN with a reason. A gate that was not run is not a pass.
3. **Fresh-clone re-run.** Before review is requested, the gates are re-run from a fresh clone at the
   exact head commit and the receipt path is posted on the PR.
4. **Non-author approval.** Someone other than the author approves. Nobody merges their own PR.
5. **Merge on the reviewed head only:** `gh pr merge --match-head-commit <sha>`.
6. **Words.** Evidence levels are PROPOSED, IMPLEMENTED, EXERCISED, QUALIFIED, MERGED, DEPLOYED and are
   not interchangeable. Host or emulator results are not hardware qualification. Inferences are marked
   `[INFERENCE]`. Public wording comes from the verified claims ledger with its limit clause.

## 2. Core paths (after activation)

A PR touching a path listed in `docs/review/CODEOWNERS.proposed` also needs **one approving review from
a human reviewer** in the owning group, in addition to section 1. The groups are:

| Group | Paths | Reviewer checks |
|---|---|---|
| GPU core | encoders, QMD and code objects, driver client, submission, session and command list, the future `backend/` tree, the public ABI manifest | see 2.1 |
| IP review | `THIRD_PARTY.md`, `LICENSE`, `NOTICE`, licence and provenance docs | see 2.2 |

### 2.1 GPU core checklist

- **Encoding provenance.** Every new or changed instruction field, QMD field or method value cites
  either a Mesa NAK source line (MIT, `THIRD_PARTY.md` §2) or a chip test in this repo that measures it.
  NVIDIA compiler output (nvcc, ptxas, cubins) is not a source for fields changed after activation. The
  existing encoder was built with nvcc and cuobjdump as offline oracles and follows nvcc where it and NAK
  disagree (`docs/HOW_I_DID_IT.md`, steps 1-2); those fields keep that record until each gets a NAK
  citation or a chip test. `nvdisasm` may be used only as a
  read-only checker of words we produced.
- **No HOLD material.** Nothing derived from omega-mv #864 (`cubin.om`) while it is on HOLD.
- **Negative control.** A deliberately broken kernel or input is refused or detected by the change's own
  test.
- **Cache and coherence.** Anything the CPU polls for a GPU write is GPU-uncached; flush and invalidate
  calls match the direction of each transfer.
- **Fault behaviour.** Timeouts and faults are returned as results, not hangs; the queue is usable or
  reported lost afterwards.
- **ABI.** A change to a STABLE manifest entry is a MAJOR bump; EXPERIMENTAL entries are versioned
  before they change.
- **Language rule.** New host-runtime code and new backends are `.om`. New C carries a named
  missing Omega feature, a filed extension proposal, and a `migration-pending` build switch.
- **Hardware claims.** A GB10 result cites its receipt (commit, machine, driver 580.173.02, command,
  exit code). Without one the result is NOT RUN.

### 2.2 IP review checklist

- Third-party material is listed with licence and exact source; nothing is copied where only facts are
  cited.
- No vendor binaries, firmware or headers whose licence forbids redistribution.
- Licence changes are Drake's decision and are never bundled with code changes.

## 3. Enforcement

GitHub enforces section 2 only through a ruleset that requires code-owner review on the default branch.
Changing rulesets is Drake's decision. Until then section 2 is followed by convention.
