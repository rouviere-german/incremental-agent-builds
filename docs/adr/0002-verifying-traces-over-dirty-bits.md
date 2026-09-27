# 2. Verifying traces with runtime discovery, not dirty bits over a declared graph

- Status: accepted
- Date: 2026-09-27

## Context

The engine has to decide, per key, whether a recorded result may be reused. There are three
established designs, and they differ in what they can express, not just in speed.

**Dirty bits over a static graph** (make, most orchestration frameworks). Dependencies are declared
up front; a change marks its dependents dirty transitively. This is the cheapest and the most common,
and it cannot express the thing this repo is about: `ModuleSummary` needs the `ModuleDoc` of the
modules its own outline turned out to import. The set of keys it needs *is a value it computed*. A
declared graph would have to over-approximate — "every module summary depends on every module doc" —
which makes a docstring edit invalidate all 80 summaries instead of the 5 that quote it. On the
reference workload that turns a 6-call commit into an 80-call commit.

**Verifying traces** (Shake; *Build Systems à la Carte* §4). Each execution records the dependencies
it actually read with the hash of what each returned. On rebuild, the recorded dependencies are
brought up to date and compared. Dependencies are discovered at runtime, so a value-dependent set is
natural.

**Constructive traces** (cloud build systems). Record `(inputs) → result` so *any* store's result can
be reused. Strictly more sharing, but it cannot answer "is my own recorded result still valid?"
without re-deriving the inputs, and accepting someone else's entry is a trust decision.

## Decision

Verifying traces are the primary mechanism. Constructive traces are layered on top as an optional,
signed shared cache (ADR 0005), not as the way a store validates its own results.

Two consequences of choosing verifying traces are load-bearing and deliberate:

1. **Recorded dependencies are walked in order, with early exit.** A key requested *after* a value
   that has since changed may not be requested at all this time, so it must not be built. Eagerly
   checking the whole recorded set would build keys the current build does not need — and which may no
   longer be buildable. Dependencies read together (`ctx.need_all`) form one group and are checked in
   parallel, which recovers most of the lost concurrency.

2. **Rule code is not hashed; the rule declares a `version` integer.** Hashing a rule's transitive
   code in Python is unreliable: a rule closes over module-level constants, helper functions and
   imported code, and there is no dependable way to hash "everything this function's behaviour depends
   on". A version integer is explicit, reviewable in a diff, and cheap to compare. The cost is real:
   changing what a rule computes without bumping its version leaves stale values behind.

## Consequences

- A value-dependent dependency set is expressible, which is what produces the measured 92.7% saving
  on LLM calls over the 30-commit replay: a docstring edit re-runs the summaries that quote it, not
  all of them.
- Verification costs one pass over the recorded dependencies per key, including re-hashing every input
  file. That is the price of the no-op rebuild: 322 ms p95 for 1,000 rules, of which ~57 ms is loading
  and authenticating the trace rows.
- Correctness now rests on a contract that a type checker cannot enforce — every input goes through
  `BuildContext`. That gets its own runtime guard and its own ADR (0004).
- Forgetting to bump a `version` is a real failure mode. It is documented in docs/SEMANTICS.md and in
  the README non-goals rather than hidden, and `--force` exists to recover without deleting the store.
- The planner can only *approximate* what a build will do, because it must not run rules to find out.
  Its dirty set is an upper bound (it cannot see early cutoff), which is why an approval derived from
  a plan is a cost ceiling rather than a prediction.
