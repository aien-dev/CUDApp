# Bounded review disposition

The existing PR desk independently inspected the original policy and report
decision/statistics paths and replayed the policy executable after checking its
digest. That replay passed 106 checks. It published draft PR #67. This review
did not cover the complete collector, replay, model adapter or device execution;
full final-source acceptance remains pending.

Four observations were returned and addressed in the contract:

1. The most recent round supplies a single observed cost sample. This choice was
   already described; its responsiveness and noise tradeoff are now explicit.
2. Recovery probing uses eight completed rounds. The contract now states the
   eligibility rule, initial rationale and development-only tuning requirement.
   The README was corrected: the counter includes all rounds, not only ordinary
   rounds. This is at most one recovery probe per eight completed rounds.
3. Latency and TTFT quantiles are independently sorted. They are not attached to
   the median-ratio pair. The paired ratio calculation is unchanged.
4. The unbounded interval endpoint is explicitly documented as the uint64-max
   sentinel, not a measurement.

Corrections to the review's explanatory notes: the report permits elapsed time
up to 1,000,000,000 microseconds (1,000 seconds), not one second. Each cost phase
is bounded to 1,000,000,000 microseconds; their sum is at most 6,000,000,000.
Unsupported sampling explicitly selects ordinary mode; unsupported chain-profile
types are refused. The public contract and code govern these behaviors.

The author's subsequent final-source audit found the missing-baseline-identity
gap. Source `1b5414577368e58ed562964a4e61785faad75f2c` fixes it. The current
receipt contains 299 report checks and a ninth compiled bypass control; the
earlier receipt is retained. This later source delta needs independent review.
No review statement here qualifies the hardware experiment or closes an issue.
