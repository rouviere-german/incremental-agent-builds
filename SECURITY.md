# Security policy

## Supported versions

Only the latest commit on `main` is supported. This is a portfolio project, not a product with
release branches.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability
reporting ("Report a vulnerability" under the repository's Security tab). Include:

- the affected commit and component;
- steps to reproduce, ideally as a failing test;
- the impact you expect.

You can expect an acknowledgement within 7 days and a fix or a written assessment within 30 days.
Credit is given in the fix commit unless you ask otherwise.

## Scope

In scope: the code in this repository, its container image and its CI configuration. Findings that
are especially welcome, because they attack the guarantees this repo exists to make:

- a way to make the engine **serve a value a clean rebuild would not produce** (a stale value), or to
  make it reuse a result after something it depends on changed;
- a way to get a **shared-cache entry accepted** that was not signed by an allowlisted key over the
  bytes it claims, or to make an accepted entry stand in for a key whose local dependencies differ;
- a way to make a **CAS blob or trace row** verify when it should not;
- a way to run an **effect without an approval bound to its exact payload**, or to consume an approval
  for a different payload, plan or principal;
- a way for content in a workspace file to make the engine **act** (read outside the jail, run a
  command, reach the network) rather than merely influence generated text.

Out of scope: denial of service by traffic volume; findings that need a compromised host, write
access to the trace store, or hostile rule code (see the non-goals in README.md and
`docs/adr/0004-hermeticity-guard-not-sandbox.md`); the *content* of generated documentation being
influenced by injected text within its schema; and issues in third-party dependencies that are
already publicly tracked (please report those upstream).

## How this repository is hardened

- Threats, controls and the tests that prove them: [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).
- Every security claim is linked to evidence in [CLAIMS.md](CLAIMS.md).
- CI runs ruff, mypy --strict, pytest with a coverage floor, bandit, semgrep (registry packs plus
  `.semgrep/rules.yml`), pip-audit, gitleaks, a trivy image scan and a syft CycloneDX SBOM.
- Secrets come only from the environment; only `.env.example` is committed.
- Stored values are JSON and verified on read; there is no pickle-family serialization anywhere.
- Source files are treated as untrusted content: they reach the model only as quoted data, the model
  is offered no tools, and untrusted Python is parsed in a separate interpreter with a timeout and an
  output cap.
- Shared build results are Ed25519-signed, accepted only from an allowlisted key, and re-verified
  against local inputs at the moment they are used.
