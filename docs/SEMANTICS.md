# Build semantics

What a trace means, what it covers, what makes reuse sound, and the one contract the whole thing
rests on. The vocabulary follows Mokhov, Mitchell and Peyton Jones, *Build Systems à la Carte*
(ICFP 2018), and the mechanisms are the ones Shake and Salsa use; the implementation here is original.

## Keys, rules and values

A **key** is a frozen pydantic model. Its `kind` names the rule that computes it and its fields are
the rule's argument. The canonical key id is `<kind>:<canonical JSON of the fields>`, so two keys are
equal exactly when their ids are, and an id alone is enough to reconstruct the key when a trace is
verified.

A **rule** is `async def f(ctx: BuildContext, key: K) -> JsonValue`, declared with `@rule(K,
version=n)`. A **value** is a JSON value. Its hash is the SHA-256 of its canonical JSON encoding, so
two values hash equally exactly when they are equal as JSON — key order and whitespace never matter.
Values live in a content-addressed store; the trace database holds only hashes.

## Verifying traces

Each execution of a rule records a **verifying trace**: the key, the rule name and version, the
ordered groups of dependencies the rule read with the hash of what each returned, and the hash of the
result.

```
module_summary:{"path":"pkg/core/engine.py"}  rule=module_summary version=1
  group 0: rule module_outline:{"path":"pkg/core/engine.py"} = 3f9a…
  group 1: rule module_doc:{"path":"pkg/io/reader.py"}       = 71c2…
           rule module_doc:{"path":"pkg/util/ids.py"}        = 9e04…
  group 2: config iab.toml#models.summary                    = a1b8…
  group 3: file  templates/module_summary.txt                = 5d77…
  group 4: llm   <fingerprint>                               = c40e…
  → result 8b21…
```

Dependencies are **grouped** by the call that produced them: `ctx.need_all([...])` records one group,
so verification can bring those keys up to date in parallel. Groups are checked in the order they
were recorded.

To bring a key up to date:

1. no trace → execute;
2. the rule's name or `version` differs from the trace → execute;
3. the key is in the force set → execute;
4. otherwise walk the recorded groups in order. For a rule dependency, bring that key up to date
   first and compare its current result hash with the recorded one. For a file, glob or config
   dependency, re-read and re-hash it. For an LLM or tool step, check that the recorded step still
   holds the recorded output (and, for a volatile tool, that it has not expired). **The first
   mismatch stops the walk and the rule executes.**
5. everything matched → reuse the recorded result without running the rule.

### Why the order matters

Dependencies are discovered at runtime, so the graph is not static: what a rule needs can depend on a
value it computed a moment earlier. Consider a rule that reads a selector file and then needs one of
two other keys based on it. Its trace records `file selector` then `rule A`. If the selector changes,
the walk stops at the selector, the rule re-runs, and it asks for `rule B` instead. Had verification
checked the whole recorded set eagerly, it would have built `A` — a key the current build does not
need at all. That is not merely wasted work: `A` may not even be buildable any more.

This is why verification is a sequential walk with early exit, not a set comparison, and why
`ctx.need_all` exists for the cases where parallelism is safe because the keys were requested
together.

### Early cutoff

When a rule re-runs and its new result hash equals the recorded one, dependents see an unchanged hash
and are not re-run. The rule paid for itself; nothing downstream did.

This is the mechanism behind most of the savings, and the rules are written to make it fire:

| Rule | Deliberately ignores | So this edit stops here |
|---|---|---|
| `ModuleOutline` | blank lines, trailing space, comments | a reformat or a comment |
| `ModuleDoc` | everything but the module docstring's first paragraph | a function docstring |
| `ModuleApi` | all docstrings | any documentation edit |
| `ApiDiff` | anything but added/removed/changed signatures | a docstring or layout change |

`EngineConfig(early_cutoff=False)` is an ablation: it makes every re-executed key count as changed.
Values stay correct, work goes up (measured 1.23× rules executed over the 30-commit replay, up to 14×
on a single commit).

## What a trace covers, and what it does not

A trace covers every input the rule read **through its `BuildContext`**:

| `ctx` call | Recorded as | Invalidated by |
|---|---|---|
| `need(key)` / `need_all(keys)` | rule dependency, value hash | that key's value changing |
| `read(path)` / `read_bytes(path)` | file dependency, content hash | the file's bytes changing, or it appearing/disappearing |
| `glob(pattern)` | glob dependency, hash of the sorted match list | a match being added or removed (not by a match's content) |
| `config(dotted_key)` | config dependency, hash of that one value | that key's value changing — **not** other edits to the file |
| `tool(name, args)` | tool step fingerprint → output hash | args, declared input files, or the tool's `version` changing |
| `llm(template, vars, ...)` | LLM step fingerprint → output hash | the template text, model id, `max_tokens`, system prompt, output schema, redaction flag or any variable changing |

It does **not** cover:

- **The rule's source code.** Only the declared `version` is in the trace. Changing what a rule
  computes without bumping its version leaves stale values behind. This is a deliberate trade-off
  ([ADR 0002](adr/0002-verifying-traces-over-dirty-bits.md)): hashing a closure's transitive code is
  unreliable in Python, and a version integer makes the contract explicit and reviewable.
- **Anything read outside `ctx`.** The clock, the RNG, an environment variable, an already-open file
  handle, a network socket. This is the hermeticity contract below.
- **Other keys in the config file.** `ctx.config("models.summary")` depends on that value only. That
  is the point — a model-id change for one role does not invalidate the others — and it means the
  config file is parsed once per build and tracked per key.

### Config-value granularity, concretely

The `model_id` commits in the replay change one entry in `iab.toml`. A file-level dependency would
invalidate every rule that reads any config value: all 80 module summaries, the package summary and
the release note. Tracking the individual value invalidates only the artifacts produced with that
role's model — **2 rules executed, 1 LLM call**, versus 83 calls for a clean rebuild.

## LLM and tool steps

A step is a memoized function call, keyed by a fingerprint, stored beside the traces.

An **LLM step's** fingerprint is the SHA-256 of the canonical JSON of: format version, template id,
template SHA-256, model id, `max_tokens`, system-prompt SHA-256, output-schema hash, the redaction
flag, and the value hash of every variable. Two calls with equal fingerprints produce byte-identical
provider requests. That is what makes reusing a recorded output sound, and it is also why the two
naive baselines fail: a prompt-hash cache cannot see the model id, because the model id is not prompt
text.

A **tool step's** fingerprint covers the tool name, its `version`, its canonical arguments and the
content hashes of the workspace files it declared as inputs. Build tools are read-only by
construction — the registry refuses any tool declaring a side-effecting capability.

A **volatile** tool (`ttl_seconds` set) also depends on something outside the workspace. Its recorded
result is trusted for the TTL; after that, verification re-runs the tool and compares the result hash,
so an unchanged answer still cuts off. Volatile steps are never exported to the shared cache.

`--force key` re-runs that rule's body *and* its steps, bypassing step memoization for that key. It is
the escape hatch for "I do not trust this recorded output", so it must not silently serve it back.

## Constructive traces and the shared cache

A verifying trace answers *"is my recorded result still valid?"*. A **constructive trace** answers
*"has anyone computed this key for exactly these dependency values?"* (*Build Systems à la Carte*,
§4). Exported entries map `(rule, version, key, ordered dependency hashes)` to a result hash, and
travel with the steps they rely on and the value blobs.

Importing one is safe only because both halves are checked:

- **at import**: the signer is on the local allowlist, the Ed25519 signature verifies over the
  canonical entry, the named blob is in the bundle, hashes to the claimed result, and is canonical
  JSON. Anything else is rejected and counted by reason.
- **at use**: the entry is a *candidate*. The build verifies every recorded dependency against its own
  current values, exactly like a verifying trace, and re-checks the signature against the allowlist at
  that moment — so removing a key revokes its entries retroactively.

Only traces a store computed itself (`origin == "executed"`) are exported, so trust cannot launder
itself through a chain of stores.

## Scheduling

The scheduler is *suspending* (*Build Systems à la Carte*, §5): each key is computed by its own
asyncio task, started the first time anything needs it, and each key is computed at most once per
build. A rule that needs another key **releases its concurrency slot** while it waits, so `jobs`
bounds the number of rule bodies actually running without deadlocking on a deep chain — with
`jobs=1`, a chain of depth 50 still completes.

**Cycles** cannot be ruled out statically, because the dependency set is a runtime value. A waits-for
graph over suspended keys is checked before each suspension: a key that (transitively) waits for
itself raises `CycleError` naming the ring, instead of deadlocking. A rule that needs its own key is
caught immediately. Keys not in the cycle still build.

The check is exact reachability, so its cost grows with the number of simultaneously suspended keys.
The cheap precondition — a target that waits for nothing cannot close a ring — removes the graph
search in the common case, which is what took a 1,000-rule no-op rebuild from 437 ms p95 to 310 ms on
the machine in the README.

**Failures** are per key: a rule that raises fails that key and everything that needs it, while
unrelated targets still complete. A **budget exhaustion** aborts the whole build, but traces of keys
that already finished are kept, so a retry resumes instead of starting over.

## The hermeticity contract

> Every input a rule reads goes through its `BuildContext`.

Everything above depends on this one sentence. A rule that reads a file directly produces a value that
is not a function of its recorded dependencies, and the engine will happily reuse it after that file
changes.

The contract is checked, not assumed:

- **at runtime, in tests**: a `sys.addaudithook` hook (PEP 578) marks the task's context while a rule
  body runs. `BuildContext` methods take a short-lived I/O permit around their own reads. Any audited
  file, directory, process, socket or database event outside a permit is a violation — recorded in
  `warn` mode, raised as `HermeticityError` in `strict` mode. The property tests run strict.
- **statically, in CI**: a semgrep rule flags the reads an audit hook cannot see (`time.time`,
  `random`, `os.environ`) inside rule modules.

What the guard cannot see is documented in
[ADR 0004](adr/0004-hermeticity-guard-not-sandbox.md): reading the clock or the RNG, an
already-open handle, and anything a native extension does. **It is a guard against honest mistakes,
not a sandbox.**

## Determinism assumptions, stated plainly

| Assumption | Status |
|---|---|
| A rule is a pure function of what it read through `ctx` | Enforced in test mode by the audit hook, in CI by semgrep. Not enforced in production. |
| Equal LLM requests produce equal outputs | Exact for `FakeProvider`. For a real model it is an assumption: the engine stores the first answer and reuses it, and never re-asks to check. Temperature is not settable on current Claude models, but sampling is still not a guarantee. |
| A tool is a pure function of its args and declared inputs | Enforced for the AST tool (it receives source text, never a path). A volatile tool declares its impurity with a TTL. |
| Recomputing a value yields the same hash | Checked: after a CAS integrity failure the value is recomputed and the engine raises if the new hash differs, which turns a non-deterministic rule into a loud error rather than a silent divergence. |

## Where each property is verified

| Property | Verified by |
|---|---|
| Clean equivalence after every edit | `tests/property/test_incremental_semantics.py`, `bench/bench_savings.py` |
| Minimality (no rerun without a changed dependency) | the same two, against `engine/oracle.py` |
| Early cutoff fires and stops propagation | `tests/unit/test_scheduler.py` |
| Verification order (a dropped key is not built) | `tests/unit/test_scheduler.py::test_a_key_no_longer_needed_is_not_rebuilt` |
| Fingerprint covers every request component | `tests/unit/test_llm_step.py` |
| Cycles raise instead of deadlocking | `tests/property/test_cycles.py`, `bench/bench_robustness.py` |
| Tamper and poisoning detection | `tests/property/test_tamper.py`, `bench/bench_robustness.py` |
| Hermeticity guard behaviour and limits | `tests/unit/test_hermeticity.py` |
