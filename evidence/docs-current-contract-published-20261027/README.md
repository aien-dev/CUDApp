# Publication identity mapping

This is the authoritative mapping for the local commit IDs in the earlier docs
repair receipts. Command-line HTTPS Git publication had no credential helper;
the authorized GitHub native Git-object API published the reviewed snapshot
instead. Its commit metadata differs, but its complete tree is identical.

- Local final verified snapshot: `263a11de500efc8c7974ec3d1e78b875aa17687b`.
- Published snapshot: `62cefb65105addcf587bbad15f26eb856b38f1b9`.
- Both complete trees: `51f81152d91aba35093d21fa8ac8461241f88a54`.
- Base: `67f3135ef88d8f520dfc15a2b2bc3264f53f0d6a`.

Every uploaded blob SHA and the resulting complete tree SHA were checked against
the local snapshot. The commit adding this mapping changes only this file.
The source repair, tests and all recorded receipts are unchanged.

Earlier IDs `8b02205`, `d56c5c7`, `b2c78f9`, `865c4d2` and `263a11d` identify local
pre-publication snapshots, not independently published branches. The rebased
receipt's instruction to use its new IDs for published source is superseded by
this mapping. Use `62cefb65105addcf587bbad15f26eb856b38f1b9` for the byte-identical
published source and the PR's final head for exact CI status.

Local focused tests and figure consistency passed; the x86_64 full build/suite
failures remain disclosed in `../docs-current-contract-rebased-20261027/`.
No native ARM64 or GPU execution is claimed by this mapping. CI is separate
head-specific evidence. Publication does not authorize merge or deployment.
