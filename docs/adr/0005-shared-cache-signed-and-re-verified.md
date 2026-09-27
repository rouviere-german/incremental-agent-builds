# 5. A shared cache entry is a signed candidate, re-verified at the moment it is used

- Status: accepted
- Date: 2026-09-27

## Context

A constructive trace — `(rule, version, key, ordered dependency hashes) → result hash` — lets one store
reuse what another already computed. That is the difference between a cold CI job paying for 81 LLM
calls and paying for none.

It is also the highest-value attack surface in the repo. An accepted entry becomes a *value* that
downstream artifacts are built from and that a publish step will write out. An attacker who can get one
entry accepted controls generated documentation, with no code execution and no API access needed. Cache
poisoning in build systems is a real, exploited class of bug.

Two questions have to be answered separately:

1. **Is this entry authentic** — did someone I trust really produce it, over the bytes it claims?
2. **Does it apply here** — are my current inputs the ones it was computed from?

A design that answers only (1) trusts a teammate's snapshot of a *different* workspace. A design that
answers only (2) trusts whatever is in the bundle.

## Decision

An imported entry is a **candidate**, never a result, and both questions are answered — at different
times.

**At import** (`import_bundle`), an entry is accepted only if all of these hold, and each failure is
counted by reason so a rejection is observable rather than silent:

- the body is well formed for its kind (a parseable key id whose kind matches the rule name, dependency
  hashes that are SHA-256 or the explicit `missing` marker, a SHA-256 result hash);
- it carries a signer and a signature (`unsigned` is its own rejection reason, distinct from a bad one);
- the signer is on the local allowlist (`APP_CACHE_TRUSTED_KEYS`);
- the Ed25519 signature verifies over a **domain-separated** canonical encoding of the body, so a
  signature made for any other purpose or format version cannot be replayed as a cache entry;
- the blob it names is in the bundle, hashes to the claimed result, and is canonical JSON.

**At use** (in the scheduler, before an entry is allowed to stand in for an execution):

- the rule name and `version` still match the local rule;
- **every recorded dependency is verified against this store's current values**, by the same walk a
  local verifying trace goes through;
- the value blob is present locally;
- **the signature is checked against the allowlist again.**

Two further restrictions:

- **Only locally executed traces are exported** (`origin == "executed"`). A store never re-signs a
  result it merely imported, so trust cannot launder itself along a chain of stores.
- **Volatile tool steps are never exported.** Their result depends on something outside the workspace,
  so it is not a function of the dependency hashes an entry claims.

## Consequences

- Revocation is retroactive and needs no cache purge. Because the signature is re-checked at use,
  removing a key from the allowlist immediately stops every entry it signed from being used, including
  entries already imported. Tested in
  `tests/unit/test_shared_cache.py::test_revoking_a_signer_stops_its_entries_being_used`.
- A signed entry from a trusted teammate that does not match local inputs is simply not used; the rule
  runs. Sharing never trades correctness for speed.
- Measured detection: 200/200 poisoned entries rejected across five attack shapes (unsigned, forged
  body, foreign signer, foreign body re-labelled with a trusted signer id, blob that does not match its
  hash), with 0 accepted — `bench/bench_robustness.py`.
- Key distribution is out of scope. The allowlist is an environment variable of base64 public keys and
  `iab keygen` prints a pair; there is no key server, no rotation protocol and no revocation list beyond
  editing the allowlist. For the intended use (a team and its CI) that is proportionate, and it is
  stated rather than implied.
- A store trusts its own signing key implicitly (it is added to the allowlist at startup), so an
  export/import round trip on one machine works without extra configuration.
- Verifying signatures twice costs one extra Ed25519 verification per candidate actually considered —
  microseconds against an LLM call, and only on keys where an entry might be used.
