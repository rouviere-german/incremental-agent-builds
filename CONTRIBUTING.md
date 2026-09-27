# Contributing

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv can install it).

```bash
uv sync --frozen
uv run python tasks.py check      # lint, typecheck, test, audit, baseline
```

If the repository sits on a small or slow drive, keep the virtual environment elsewhere:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "C:/Users/<you>/.venvs/<repo>"
uv sync --frozen
```

## Tasks

`tasks.py` is the cross-platform task runner; the Makefile only delegates to it.

| Target | What it runs |
|---|---|
| `lint` | `ruff check` and `ruff format --check` |
| `format` | `ruff format` and `ruff check --fix` |
| `typecheck` | `mypy --strict` over src, tests, scripts, bench and tasks.py |
| `test` | pytest with branch coverage; fails below 85% |
| `bench` | the three benchmark scripts with regression thresholds from `bench/thresholds.json` |
| `demo` | the offline demo (no API key) |
| `audit` | bandit and pip-audit |
| `baseline` | `scripts/check_baseline.py` |

## Rules for changes

- **Tests first for behavior.** Mark tests `unit` or `integration`. Use hypothesis for properties
  (invariants, round trips, oracles) rather than hand-picked examples alone.
- **Offline by default.** Tests, demo and bench must not touch the network or need
  `ANTHROPIC_API_KEY`; `tests/conftest.py` blocks non-loopback sockets. Use `FakeProvider` or
  `ReplayProvider` for model behavior.
- **Evidence for claims.** A new README claim needs a row in `CLAIMS.md` pointing to a test,
  bench script or CI job. Performance numbers must be copied from real `bench/` output together
  with the command that produced them.
- **Threat model.** New entry points, tools or artifacts need a STRIDE row in
  `docs/THREAT_MODEL.md` with a control and a test.
- **Decisions.** Record significant trade-offs as ADRs in `docs/adr/`.
- **No pickle.** Values are JSON and verified on read; nothing in this repo deserializes code.
- **Every input through `BuildContext`.** A rule that reads a file, the clock or the environment
  directly produces a value that is not a function of its trace, and the engine will reuse it after
  that input changes. The audit hook fails such a rule in the test suite (`guard="strict"`); a
  semgrep rule covers what the hook cannot see. If a rule needs a new kind of input, add a
  `BuildContext` method that records it, not a direct read.
- **Bump `version` when a rule changes meaning.** Rule code is not hashed (ADR 0002). Changing what
  a rule computes without bumping its `version` leaves stale values in every existing store.
- **Keep the fingerprint complete.** Anything that changes an LLM request must be in
  `LLMStepSpec.components`, with a case in
  `tests/unit/test_llm_step.py::test_every_part_of_the_request_changes_the_fingerprint`.
- **New evaluation numbers come from `bench/`.** The savings, detection-rate and timing figures in
  README.md are pasted from `bench/bench_savings.py`, `bench/bench_robustness.py` and
  `bench/bench_baseline.py`, with a threshold in `bench/thresholds.json` carrying a `_note` that
  says where the limit came from.
- **No secrets in git.** Configuration goes in `.env` (ignored); document new settings in
  `.env.example` (a test enforces that it lists every setting).
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:`).
