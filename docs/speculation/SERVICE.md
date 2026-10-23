# Persistent native policy process

`native/speculation/service.om` connects the existing policy to a persistent
stdin/stdout process. It is handwritten Omega, including framing, lifecycle
validation and timing. The native model engine must own and acknowledge all
physical prefill, verification, rollback and token-consumption operations.

This is a real Linux AArch64 process interface, tested by a second native Omega
process through two pipes. It is not a model adapter or an AienOS device gate.

## Qualified compiler profile

This entrypoint uses the existing native-host compiler from Omega commit
`f0499f415571bdd7496b6de391096915e78f7556`, independently reconstructed through
its existing seed chain. Compiler SHA-256:

```
e5d60db892c8f21cc8c9128af26d8bded0e96d7f92946f5a549397c43a872649
```

Its published native-host gate covers file, process, pipe, clock and socket
operations. This is a distinct profile from the baseline compiler used for the
original pure components; no union with another compiler profile is assumed.
The dependency is qualified here without editing another owner's compiler.

For a reconstruction, export the exact pinned seed and original fixture paths,
including their Git file index, and run its existing `seed/host/check.sh`.
Require nine positive groups, 41 negative controls and **70** original fixtures.
A partial source export can misleadingly print a passing comparison of zero
fixtures; zero does not qualify this dependency.

Compile these ordered units with that compiler:

- Worker: `native/speculation/policy.om`, `native/speculation/service.om`.
- Native test client: `native/speculation/policy.om`,
  `tests/speculation/support.om`, `tests/speculation/service-test.om`.

Run the test client with the worker executable's absolute path as its sole
argument. The client starts only that host worker, exercises both pipes, and
reaps it. Existing shell tools coordinate development reconstruction only;
the delivered worker invokes no shell, foreign helper or model runtime.

## Wire format

All fields are unsigned 64-bit little-endian words. Requests have five header
words: magic `827343699` (SCP1), version `1`, operation, sequence, body length in
words. Sequences start at one and advance after each framed response, including
semantic refusals. Header and body reads tolerate partial reads and EINTR.

Responses have seven header words: magic, version, operation, sequence, status,
body length in words, elapsed policy nanoseconds. Timing starts after the full
request arrives and ends before the response is written. The engine must also
measure total round-trip overhead, including transport, and include that cost
in the round. The elapsed field alone is not full adaptation overhead.

| Operation | Request body | Successful response body |
| --- | --- | --- |
| 1 initialize | 64 identity words, 6 profile words, 77 cost-row words, 31 prior words | Empty |
| 2 begin | warm flag, require-device flag, sampling mode | Empty |
| 3 select | context, remaining output, memory budget, epoch, mode, fixed length | Five decision words from CONTRACT.md |
| 4 observe | accepted drafts, emitted tokens, full elapsed microseconds, EOS, complete | Emitted, consumed, pending-token, done counters |
| 5 consume | Empty; caller acknowledges consumption by the real model owner | Empty |
| 6 cancel | Empty; caller owns actual rollback and cancellation | Empty |
| 7 end | Empty; request must be complete with no active round | Empty |
| 8 exit | Empty; no request may be active | Empty, then process exit 0 |

Initialize is accepted once per process. Begin requires initialization or a
previously ended/cancelled request. Warm begin retains the policy's learned
statistics under its original identity rules. Reconfiguration requires a new
process. The process performs no shared learning with other instances.

Semantic errors return the policy's status and no payload. Broken framing,
unknown operations, bad versions/sequences or incomplete frames terminate the
process without a response for that frame. EOF is clean only between requests;
EOF during an active request exits 75. One process accepts at most one million
commands. There is no automatic fallback after process or protocol failure.
The model caller must abort the affected trial and preserve its failure record.

The process itself does not discover code/model hashes or authenticate a device.
Supplied identities and acknowledgements must be derived from the real owner;
their presence on the wire cannot prove that the model executed correctly.

## Exact existing-engine attachment point

The inspected 27B engine is sovereign-core `speed2/specdecode` at
`68fa1683874899eb0b6b4ab68bb2b9b19fcbcbb1`. Its existing
`qwen38_omega/spec.rs::decode_rounds` takes one fixed `k` for the request.
It has no external per-round budget callback. Its server reads `AIEN_Q38_SPEC_K`
when loading, and its `qwen38_spec suite` parses the entire plan before loading.
Those interfaces cannot implement adaptive per-round sizing merely by changing
command arguments between calls. This source supports chains up to eight drafts;
16 and 31 must remain explicitly unsupported until independently qualified.

A narrow engine connection would call select before drafting, use the returned
budget, preserve existing MTP/verification/complete-state ownership, and report
observe only after actual commit or rollback. It must handle natural EOS,
cancellation and output exhaustion before reporting the exact emitted/consumed
prefix, then acknowledge any correction token at its real consumption boundary.
Native timing must cover draft, verify, commit, host, policy/transport and setup.
First-divergence checks remain mandatory; this interface adds no numerical waiver.

That connection would require new code in the existing Rust engine. The current
experiment's Omega-only rule has not been waived. A decision on that boundary is
pending; no Rust connection, model launch or hardware trial is delivered here.
A strictly Omega attachment instead depends on the existing native ABI/runtime
owner. The presence of native host facilities does not establish that ABI.
