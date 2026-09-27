# 4. An audit-hook hermeticity guard in test mode, and it is not a sandbox

- Status: accepted
- Date: 2026-09-27

## Context

Every correctness claim in this repo reduces to one sentence: *every input a rule reads goes through
its `BuildContext`*. A rule that reads a file directly produces a value that is not a function of its
recorded dependencies, and the engine will reuse it after that file changes. No type checker can
enforce this — `open()` is available inside any rule body.

The options:

1. **Trust the contract.** Document it and rely on review. Cheap, and it fails silently: a stale value
   looks exactly like a correct one.
2. **Audit hook** (PEP 578, `sys.addaudithook`). CPython raises audit events for `open`, `os.*`,
   `socket.*`, `subprocess.Popen`, `sqlite3.connect` and more. A hook can tell whether the current
   task is inside a rule body and outside a `BuildContext` call.
3. **Real isolation.** Run each rule in a subprocess with seccomp/AppArmor, or in a container. Actually
   enforceable, and it changes the programming model completely: values must be serialized across a
   boundary, concurrency becomes process-level, and a build of 287 tiny rules pays process spawn costs
   that dwarf the work.

## Decision

Use the audit hook (option 2), with three modes, and be explicit that it is not option 3.

- A `ContextVar` marks the current task as executing a specific rule's body. `BuildContext` methods
  wrap their own reads in a short-lived I/O permit. Any watched audit event while the marker is set
  and the permit is not is a violation.
- `strict` raises `HermeticityError`, failing that key. The property tests run strict, so a rule that
  cheats fails the suite rather than producing a subtly wrong value.
- `warn` records violations and counts them on the build result and in a Prometheus counter. This is
  the production default: a latent hermeticity bug should be visible, not fatal.
- `off` disables it, for benchmarks where the hook's per-event cost is measurable.
- New scheduler tasks are started in a copied context with the marker cleared, so engine bookkeeping is
  never mistaken for a rule body.
- A semgrep rule in CI covers the important class the hook *cannot* see.

## Consequences

**What the guard catches.** Reading a file, globbing, spawning a process, opening a socket or a
database from inside a rule body. That is the realistic mistake — someone reaches for `pathlib` instead
of `ctx.read` — and it is caught with the key named, in the test suite, before it can produce a stale
value.

**What it cannot catch, stated plainly:**

- **Reads that raise no audit event.** `time.time()`, `random.random()`, `os.environ` lookups (the
  mapping is already populated), and reads from a file handle opened *before* the rule started. The
  semgrep rule in `.semgrep/rules.yml` flags the first three in rule modules; nothing catches the
  fourth.
- **Native extensions.** A C extension doing its own syscalls bypasses the hook entirely.
- **Hostile rule code.** A rule that wants to evade the guard can: audit hooks cannot be removed once
  installed, but a determined author can read through a native library or a pre-opened handle. The
  guard assumes rule authors are honest and occasionally careless.
- **Anything outside a rule body.** Tool handlers run in a worker thread with the marker cleared,
  because they are *expected* to do I/O; their purity is enforced by their interface instead (declared
  inputs, args-only, read-only capability).
- **A rule that goes through the context and then ignores the answer.** The guard checks *where* an
  input came from, not what the rule did with it. A rule that catches the exception from a failed
  `ctx.llm`, `ctx.tool` or `ctx.need` and returns a fallback is hermetic and still wrong: the value it
  produced is a function of the provider being down, which is nothing the trace can see. That one is
  handled by recording rather than by the guard — a call that raises enters the trace with a sentinel
  hash no real value can equal, so the key re-runs until the call succeeds
  (`tests/unit/test_failed_context_calls.py`, docs/SEMANTICS.md). A rule *should* let the exception
  propagate; the sentinel is what keeps a rule that does not from caching a degraded artifact.

**Therefore the threat model does not rely on it.** The controls that face untrusted input are the
workspace jail (`realpath`, no `..`, size caps), the subprocess-isolated AST parser, the read-only tool
registry, schema validation of model output, the approval gates and the budgets. The hermeticity guard
protects *correctness against mistakes*, and the README, THREAT_MODEL and docs/SEMANTICS.md all say so
rather than implying containment it does not provide.

**Cost.** The hook runs on every watched audit event process-wide once installed. It is a
`ContextVar` read and a set membership test in the common case where no rule is running. The `off` mode
exists so benchmarks measure the engine and not the hook.
