# Tool inventory: everything becomes Omega, nothing gets renamed

The key point: developers who are using these tools today keep using the same
names, the same paths, the same flags and the same output. A tool is replaced
when the Omega build sits at the same path and behaves the same. Nobody has to
learn anything. What changes under the hood is that the logic is written in our
language, compiled from the seed, and fast.

Rule per tool: the Omega source lives in this repo under `tools/<name>/`, the
gate is `tests/omega/run.sh`, and the shipped binary installs over
`~/.local/bin/<name>`. A replacement that changes an output format or a flag is
a bug, not a rewrite.

## What is on the box today

`~/.local/bin` holds 163 entries: about 120 real tools and launchers (the rest
are `.bak-*` copies). By interpreter: 37 `/bin/sh`, 19 `/bin/bash`, 38
`/usr/bin/env` (python or node), 47 compiled binaries or data. Three shell
libraries behind them (`~/.local/lib/aien-local-ci/`, `~/.local/lib/aien-watch/`,
`~/.local/lib/gh-guard/`). About 25 cron jobs and roughly 20 active user
services call into these.

## Tier 1 - pure logic, replace now (OSC, already possible)

| Tool | Lines | Language | Does |
|---|---|---|---|
| `ledger` | 54 | python | TSV ledger: add/note/done/list/stale |
| `spark-window` | 378 | sh | chip window allocator and lock ledger |
| `quiet-build` | 58 | sh | quiet-lock build runner |
| `quiet-measure` | 44 | sh | chip-lock measurement runner |
| `lock-long-holds` | 76 | sh | lock janitor |
| `q36-extra` | 68 | bash | extra model load gate |
| `org-check` | 60 | sh | repo dashboard checks |
| `org-mirror` | 19 | sh | org mirror |
| `ghmerge` | 18 | bash | merge queue helper |
| `watch-merge` | 16 | bash | merge watcher |
| `rebase-crumb` | 34 | bash | crumb on rebase |
| `spark-sensors` | 15 | sh | repo dashboard sensor |
| `spark-receipt` | 102 | sh | receipt builder |
| `spark-calibrate` | 97 | sh | calibration receipt |
| `spark-baseline` | 149 | sh | baseline receipt |
| `spark-build-once` | 121 | sh | keyed build cache |
| `after-window-18191` | 14 | bash | window follow-up |
| `sweep-build-scratch` | 14 | python | build scratch sweeper |
| `wip-snapshot` | 29 | python | WIP snapshots |
| `vault-add-secret` | 9 | bash | vault add |
| `wall-power-log.sh` | 7 | sh | power log |
| `claude-chat-status` | 13 | python | chat status line |
| `qwen-logs` / `qwen-restart` / `qwen-stop` / `qwen-runtime` / `qwen-gate` | 49 | python | model server control |

Tier-1 cores are parse-and-decide: TSV and lock files, budget arithmetic,
allowlists. Their Omega ports are the pattern already proven by `nvrm.osc` and
`kmod.osc`: the decision logic in `.osc` with a run line per rule, a thin
same-name launcher in front.

## Tier 2 - file and process inspection, replace now (syscall landed)

`osc/syscall@01f933a8` unblocked these: open, read, write, close,
clock_gettime, getpid are real words.

| Tool | Lines | Language | Does |
|---|---|---|---|
| `chip-watchdog` | 30 | bash | stops hung lock holders (reads /proc, nvidia-smi) |
| `agent-revive` | 21 | bash | dead agent reaper |
| `coordinator-tick` | 11 | bash | restart-request reader |
| `aien-local-ci` | 54 | bash | PR judge entry (`lib/pipeline.sh`, `cron-tick.sh` behind it) |
| `aien-watch` | 124 | python | watch daemon pass (`lib/aien-watch.sh`) |
| `now-publish` | 70 | python | aienos.com /now/ publisher |
| `spark-cleanup` | 366 | python | worktree/target cleaner |
| `investigate.sh` | 49 | sh | incident probe |
| `hw-grep` / `hw-page` / `docs-read` / `env` | 44 | sh | small readers |
| `claude-chat-feed` / `claude-inbox-hook` | 34 | python | chat archival |
| `renda` | 34 | python | PR desk entry |
| `renda-local` | 278 | python | local-model PR desk |
| `forgejo-backup` | 212 | sh | forge backup |
| `agent-mail` | 19 | python | mail entry |
| `cortex-tool` | 135 | python | cortex client |
| `crumb` / `crumb-doctor` | 300 | python | crumb notes |
| `gh` | 314 | python | guarded gh wrapper (the guard logic ports; the gh binary stays) |

## Tier 3 - needs fork/exec/argv (next compiler rungs)

These launch other programs, so they wait on `execve` with argv arrays and
`wait4`. Named now so the rung is planned, not discovered later:
`renda-harness` (577 lines, drives review lanes), `renda-loop`,
`spark-connector-supervise`, `qwen-code`, `spark-window`'s launch path,
`aien-watch-hook`, `approve-cleared`, `ask-muse`, `rad-push`.

## Tier 4 - compiled services (stay until the runtime grows)

Rust/Go binaries running as services: `cortex-rs`, `cortex-encoder-rs`,
`spark-cockpit-rs`, `aien` (53k lines), `aien-mail`, `conduit`, `rad`,
`radicle-node`, `git-remote-rad`, `forgejo-runner`, `mobhub`, `tectonic`,
`ripwire`, `scrub`, `openclaw`, `agy`, `grok`, `spark-*` compiled daemons
(dream, harness, supervisor, debugger, distill, discord-hub, muse-bridge,
mail, inquisitor, connector, rsi, crumbs), `muse`, `agent`, `agent-archive`.
Their service contracts stay identical; each becomes a porting lane with the
same name when the runtime can host it.

## Keep as-is (not our logic)

Third-party binaries and their thin launchers where the wrapper adds no
decision: `claude`, `codex`, `code`, `gh`'s underlying binary, `pnpm`/`pnpx`,
`python3.11`, `hermes*`, `bw` (Bitwarden CLI), `greptile-mcp`, `qwen-code`'s
npm entry. Wrappers whose guard logic is ours (the `gh` guard, the `claude`
guard) get their guard ported to Omega at Tier 2 while the wrapped binary
stays.

## Order

1. Tier 1 in this repo, `tools/<name>/`, same-name drop-ins: `ledger` first
   (smallest, most used), then `quiet-build`, `quiet-measure`, `spark-window`
   (the big one at 378 lines), then the rest of the table.
2. Tier 2 as syscall use matures, one tool per branch, each keeping its name.
3. Tier 3 opens when `execve` + `wait4` land; `renda-harness` is the first
   target and the biggest win.
4. Tier 4 lanes open per service; the Rust ones stay until then.

Every replacement keeps the old tool archived at `<name>.pre-omega-<date>`
and a receipt: same inputs, same output, byte for byte, before the swap.
