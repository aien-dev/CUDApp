# Security policy

CUDA++ is a research-stage GPU toolchain with direct access to low-level device interfaces. No general production-security certification, bug-bounty program, or fixed security-response time is claimed.

## Reporting a vulnerability

**Do not include exploit details, credentials, host addresses, confidential logs, or proof-of-concept payloads in a public issue or pull request.**

Use GitHub's private **Report a vulnerability** option on this repository's Security tab **if it is enabled**. If private reporting is not enabled, open a public issue asking maintainers to provide a private reporting channel **without including any vulnerability details**. A maintainer should agree on a private channel before evidence is shared.

For ordinary bugs that do not expose a security vulnerability, use the regular issue templates. Please include the affected commit and environment, a minimal non-sensitive reproduction, expected and actual behavior, and any verified mitigations.

Security fixes must preserve existing negative controls, do not silently relax device or memory protections, and should be qualified through the relevant host and approved hardware gates before being called resolved. Never run proof-of-concept tests on a shared GPU service without authorization.

See [CONTRIBUTING.md](CONTRIBUTING.md), [the provenance audit](docs/AUDIT.md), and [the runtime contract](docs/runtime/API.md).
