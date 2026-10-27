# Security policy

CUDA++ is a research-stage GPU toolchain with direct access to low-level device interfaces. No
general production-security certification, bug-bounty program, or fixed security-response time is
claimed.

## Reporting a vulnerability

Report privately. Do not open a public issue, pull request or discussion about it. Do not include
exploit details, credentials, host addresses, confidential logs, or proof-of-concept payloads in a
public issue or pull request.

1. GitHub: open this repository's Security tab and choose "Report a vulnerability". This
   creates a private advisory that only the maintainers can see. If that option is not shown,
   private vulnerability reporting has not been enabled for this repository yet; use email.
2. Email: drake@aienos.com with the subject `[CUDApp] security report`.

Include the commit or release you tested, the component (path), steps to reproduce, and what an
attacker gains. If the report involves a secret, say where it was exposed and do not send the
secret itself. Encrypted mail is not required; do not post proof-of-concept payloads publicly.

For ordinary bugs that do not expose a security vulnerability, use the regular issue templates.
Include the affected commit and environment, a minimal non-sensitive reproduction, expected and
actual behavior, and any verified mitigations.

Never run proof-of-concept tests on a shared GPU service without authorization.

## Response targets

This project has one maintainer; the targets below are best effort and stated honestly.

- Acknowledgement: within 3 working days.
- Triage and severity: within 7 days of acknowledgement.
- Fix or mitigation for confirmed issues: aim for 30 days for high and critical, 90 days
  otherwise, with a status update at least every 14 days until closed.
- Disclosure: coordinated. An advisory is published with credit (unless you ask otherwise)
  once the fix is on the default branch. Default embargo 90 days, negotiable.

Security fixes must preserve existing negative controls, must not silently relax device or memory
protections, and should be qualified through the relevant host and approved hardware gates before
being called resolved.

## Scope

In scope:

- The driver client in `nvrm/` and `m16/`: wrong ioctl arguments or sizes, a buffer mapped with
  more rights than the platform contract allows, a completion wait that can hang without its
  deadline, or one channel able to affect another.
- The platform contract and runtime (`src/omega_gpu_platform*`, `src/omega_gpu_session.*`,
  `src/omega_gpu_cmdlist.*`, `src/omega_gpu_engine.*`): a silent fallback, a skipped memory
  reserve check, or GPU work allowed after an unknown completion state.
- The encoder, kernel builder and QMD writer (`src/omega_blackwell_*`): generated code or launch
  descriptors that read or write outside the buffers the caller passed.
- `make doctor` (`tools/doctor/`) and `tools/zero-cuda-gate.sh` reporting READY or PASS on a
  machine or binary that should fail.
- A secret, private path or host address committed to the repository.

Out of scope:

- NVIDIA's open kernel module headers in `third_party/nvidia-open-580.173.02/` (copied
  unchanged), NVIDIA's kernel module, and the GSP firmware; report those to NVIDIA.
- `make cubin-probe` (`tools/interop/cubin_probe.c`), an interop check that links NVIDIA's
  `libcuda` and is not part of CUDA++.
- The host operating system.

## Supported versions

Only the default branch `main`; there is no release. Reports against older commits are welcome but
fixes land on the default branch only.

## Status

Research stage, not a drop-in replacement for CUDA (`docs/START_HERE.md`). The only qualified chip
target is the NVIDIA GB10 (DGX Spark, sm_121) with open kernel module 580.173.02; current native
qualification of this checkout is NOT_RUN (README). Direct execution of the binaries and
applications linking the library have no driver-version or PCI-ID admission guard; only the
`make doctor` wrapper refuses another module version. See AUDIT_SCOPE.md for what to review and how
to build and test it, and also [CONTRIBUTING.md](CONTRIBUTING.md), [the provenance
audit](docs/AUDIT.md), and [the runtime contract](docs/runtime/API.md).
