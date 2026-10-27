# Independent GPU evidence, bounded first slice

Issue: https://github.com/aien-dev/CUDApp/issues/91#issuecomment-6407638311
Initial inspection: b5e0da48ee7b8a64ae0b51068769fa0659b2e33a
Refreshed base: c73afb25cce8d22e039287a8ab2e4cd4c7aecf67

## Design

Make the gap between an owner-operated receipt and independent replication
explicit. Supply a collection protocol, a structural manifest schema and an
empty NOT_RUN template. The schema must not issue a physical PASS verdict.
A separate operator and reviewer decide what the supplied evidence supports.

A prose-only checklist would leave evidence shape ambiguous. A new runtime,
GPU launcher or automatic attestation engine would duplicate existing owners
and exceed this slice. This documentation/data contract reuses the existing
native vector harness and leaves missing executable work to its .om owners.
No device code, build gate, driver, historic receipt or public claim is changed.

## Plan and verification

1. Inspect current public main, #79/#88/#91, the native vector harness and the
   existing teardown receipt. Extend #91 without replacing its original scope.
2. Specify exact-source rebuilding, challenge custody, external CPU checking,
   safe controls, process/device evidence and public/raw artifact commitments.
3. Add a schema with no PASS output and an explicitly empty template. Check
   schema validity and reject missing controls, fabricated template evidence,
   unknown fields, malformed hashes and COLLECTED rows without evidence.
4. Validate local links and JSON; run repository checks without weakening them.
   Record host failures separately from schema checks. All GPU work is NOT_RUN.
5. Obtain a fresh independent review, scan new public content, and submit a
   draft PR after review. The PR prepares the work; it does not close #91.

## Follow-on work, not implemented here

The existing harness has fixed salts, an in-process CPU checker, tail checks
that disappear at maximum capacity, and no explicit withheld-submission arm.
Implement independently supplied challenge input/output transport, separate
oracle custody, complete guards and the missing control in seeded .om. Extend
the existing tests rather than creating a second runtime or permanent C path.
The manifest is a collection envelope, not a shipped semantic validator:
artifact references, real digests, chronology and truth still need checking.
