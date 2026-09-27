# 3. LLM outputs are values in the store, keyed by a fingerprint of the whole request

- Status: accepted
- Date: 2026-09-27

## Context

An LLM call is the expensive step. Everything about the incremental story depends on being able to
*not* make one, which means treating a model's answer as a cacheable value. That is only sound if the
cache key covers everything that determines the request — and the two obvious keys do not.

Keying on the artifact's explicit input (`@cache def summarize(source)`) misses the prompt template,
the model id, and any other file the prompt quotes. Keying on the SHA-256 of the rendered prompt
catches the template and the quoted files, but **not the model id**, because the model id is not part
of the prompt text. Both then serve the previous answer with no error. Over the 30-commit replay this
is not hypothetical: the input-hash cache serves 91 stale artifacts and the prompt-hash cache 2, both
measured against a clean rebuild in `bench/bench_savings.py`.

There is also a choice about *where* the prompt lives. If the template is a Python string, editing it
is a code change, invisible to the trace (see ADR 0002 on not hashing code).

## Decision

An LLM step is a memoized function call whose fingerprint is the SHA-256 of the canonical JSON of:

- a format version (so the scheme itself can change without silently matching old entries),
- the template id **and the SHA-256 of the template text**,
- the model id,
- `max_tokens`,
- the SHA-256 of the system prompt,
- the hash of the output JSON Schema,
- the redaction flag,
- the value hash of **every** variable.

Equal fingerprints therefore mean byte-identical provider requests.

Supporting decisions that make this hold:

- **Templates are workspace files** (`templates/<id>.txt`), so editing a prompt is an ordinary file
  change that the trace records like any other input.
- **Model ids come from the workspace config** (`iab.toml`), read through `ctx.config` at
  per-key granularity, so switching one role's model invalidates only that role's artifacts — 2 rules
  and 1 LLM call on the reference workload, instead of 83.
- **Rendering is deterministic**: placeholders are substituted with `<untrusted-data>` envelopes whose
  id is a hash of their content, so there is nothing random or time-dependent in a prompt.
- **A template that drops a variable is an error.** Every placeholder must have a variable and every
  variable must be used, so a template edit cannot silently stop sending an input the trace still
  records.
- **The output is schema-validated before it becomes a value**, and the schema's hash is in the
  fingerprint, so tightening a schema invalidates outputs that were only valid under the old one.

## Consequences

- Reusing a recorded answer is sound, and the reference workload's derived docs are reproducible
  offline with `FakeProvider` — which is what makes the savings benchmark a fair comparison rather than
  an estimate: the calls a clean rebuild makes are exactly the calls the incremental build avoided.
- The engine never re-asks the model to check an answer. It stores the first one and reuses it. For a
  real model that is an assumption, not a proof, and it is stated as such in docs/SEMANTICS.md and in
  the README non-goals.
- A prompt template edit is correctly expensive: editing `module_summary` invalidates every module
  summary (82 of 83 calls on that commit). This is the honest cost of correctness, and it is exactly
  what the input-hash cache gets wrong for free.
- Prompt caching, streaming and batch requests are not wired in. They are orthogonal cost levers on
  the calls that *do* happen, and they are listed as non-goals.
- The provider interface must not leak anything the fingerprint does not cover. `LLMRequest` carries an
  explicit `model`, and build steps always pin it rather than relying on a role mapping, so a
  deployment-level model default can never change a value behind the trace's back.
