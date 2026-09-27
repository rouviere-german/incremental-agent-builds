# Claims and evidence

Every claim this repository makes, and the test, benchmark or CI job that backs it.
`scripts/check_baseline.py` verifies that each backticked reference resolves to a real test function,
file or CI job, so a claim whose evidence is deleted fails the build.

Numbers come from `bench/` runs on Windows 10, Intel i7-8750H (6 cores), Python 3.12.14, recorded in
`bench/results/*.json` with the machine that produced them. They are not estimates.

## The primary pattern: incremental rebuilds that match a clean build

| Claim | Evidence | Notes |
|---|---|---|
| After every source edit, the incremental store holds exactly the keys a build from an empty store produces, with equal values | `tests/property/test_incremental_semantics.py::test_incremental_build_equals_a_clean_rebuild_after_every_edit`, `.github/workflows/ci.yml#property-soak` | 1,000 random rule programs × edit sequences in the soak profile, 200 in standard CI, 60 locally |
| The same holds with one worker and a single target | `tests/property/test_incremental_semantics.py::test_equivalence_holds_for_a_single_target_and_one_job` | A blocking scheduler would deadlock here |
| The same holds for 30 real commits over an 80-module package: 30 / 30 commits equal to a clean rebuild | `bench/bench_savings.py`, `tests/integration/test_demo_and_replay.py::test_every_commit_leaves_the_store_equal_to_a_clean_rebuild`, `.github/workflows/ci.yml#savings` | `replay_correctness.clean_equivalent_pct_min = 100` is a CI threshold |
| No rule ever re-runs without at least one direct dependency whose value changed | `tests/property/test_incremental_semantics.py::test_incremental_build_equals_a_clean_rebuild_after_every_edit`, `tests/integration/test_demo_and_replay.py::test_no_rule_re_ran_without_a_changed_dependency`, `bench/bench_savings.py` | Judged by an oracle built from two clean builds, not by the engine's own bookkeeping |
| The minimality oracle actually reports a violation when there is one | `tests/unit/test_oracle_and_budgets.py::test_a_rerun_with_no_changed_dependency_is_reported` | Otherwise every property above would be vacuous |
| The clean-equivalence oracle reports missing keys, differing values and corrupt blobs | `tests/unit/test_oracle_and_budgets.py::test_a_key_present_in_only_one_build_is_reported`, `tests/unit/test_oracle_and_budgets.py::test_a_differing_value_is_reported`, `tests/unit/test_oracle_and_budgets.py::test_a_corrupt_blob_is_reported_rather_than_ignored` | Same reason |
| Dependencies are discovered at runtime: a key the build no longer needs is not rebuilt | `tests/unit/test_scheduler.py::test_a_key_no_longer_needed_is_not_rebuilt`, `tests/unit/test_scheduler.py::test_a_new_glob_match_is_discovered_and_a_removed_one_disappears` | The reason verification is an ordered walk with early exit |
| An unchanged rebuild runs no rule body at all | `tests/unit/test_scheduler.py::test_an_unchanged_rebuild_runs_nothing`, `tests/integration/test_cli.py::test_plan_then_build_then_a_second_build_that_does_nothing` | Counted per rule, not inferred from stats |
| Rewriting a file with identical content costs nothing (content hashes, not mtimes) | `tests/unit/test_oracle_and_budgets.py::test_rewriting_a_file_with_identical_content_costs_nothing` | |
| Early cutoff stops propagation when a re-run produces the same value | `tests/unit/test_scheduler.py::test_early_cutoff_stops_propagation_when_the_value_is_unchanged` | |
| A whitespace-only commit executes exactly one rule and makes zero LLM calls | `tests/integration/test_cli.py::test_a_whitespace_edit_costs_one_rule_and_no_llm_call`, `tests/integration/test_demo_and_replay.py::test_a_whitespace_commit_executes_one_rule_and_makes_no_llm_call` | |
| A tool version bump invalidates the keys that used it | `tests/unit/test_build_tools.py::test_a_tool_version_bump_invalidates_its_memoized_results` | |
| A rule version bump invalidates its traces | `tests/unit/test_scheduler.py::test_a_rule_version_bump_invalidates_its_traces` | |
| `--force` re-runs one rule's body and its steps, and still cuts off if the value is unchanged | `tests/unit/test_scheduler.py::test_force_re_runs_one_rule_and_nothing_else`, `tests/unit/test_scheduler.py::test_a_tool_result_is_memoized_by_its_fingerprint` | |
| Concurrency does not change results | `tests/unit/test_scheduler.py::test_concurrent_jobs_produce_the_same_values_as_one_job` | 1 job versus 8 |
| A failing rule fails its dependents only; unrelated targets still build | `tests/unit/test_scheduler.py::test_a_failing_rule_fails_its_dependents_and_not_the_rest` | |

## Savings

| Claim | Evidence | Notes |
|---|---|---|
| 92.7% of LLM calls and 88.8% of tokens saved across the 30-commit replay (181 vs 2,488 calls; 215,857 vs 1,929,747 tokens) | `bench/bench_savings.py`, `.github/workflows/ci.yml#savings` | Floors of 0.80 / 0.75 enforced in CI |
| Median per-commit saving of 97.6% of calls and 91.2% of tokens, against a spec target of 80% | `bench/bench_savings.py` | Floors of 0.85 / 0.80 enforced |
| Savings hold commit by commit, not only in aggregate | `tests/integration/test_demo_and_replay.py::test_the_incremental_build_costs_far_less_than_clean_rebuilds` | |
| A model-id change costs 2 rule executions and 1 LLM call instead of 83 | `bench/bench_savings.py` (per-commit table), `tests/unit/test_plan.py::test_a_changed_config_value_makes_only_its_readers_dirty` | Config values are tracked per key, not per file |
| Early cutoff is load-bearing: without it, 1.23× the rules execute over the replay, up to 14× on a single commit | `bench/bench_savings.py` (`replay_ablations`), `tests/integration/test_demo_and_replay.py::test_the_ablations_cost_strictly_more_than_the_full_system` | Floors enforced in CI |
| Step memoization is load-bearing: without it, 2.65× the LLM calls | `bench/bench_savings.py` (`replay_ablations`) | Floor of 2.0 enforced |
| Disabling early cutoff or step memoization changes cost, never values | `tests/property/test_incremental_semantics.py::test_disabling_early_cutoff_keeps_values_correct`, `tests/property/test_incremental_semantics.py::test_disabling_step_memoization_keeps_values_correct` | |

## Staleness versus naive caches

| Claim | Evidence | Notes |
|---|---|---|
| This system serves 0 stale artifacts over the replay | `bench/bench_savings.py` (`system_stale_artifacts_max = 0`), `tests/integration/test_demo_and_replay.py::test_the_system_never_serves_a_stale_value` | Compared against the clean rebuild's values |
| An input-hash cache serves 91 stale artifacts (70 still stale after 30 commits) | `bench/bench_savings.py` (`naive_cache_staleness`) | A floor of 20 is enforced, so the baseline must keep failing |
| A prompt-hash cache serves 2 stale artifacts, both on model-id commits | `bench/bench_savings.py`, `tests/integration/test_demo_and_replay.py::test_the_naive_caches_go_stale_where_this_system_does_not` | |
| An input-hash cache misses a template edit; a prompt-hash cache catches it | `tests/unit/test_naive_baseline.py::test_the_input_hash_baseline_misses_a_template_edit`, `tests/unit/test_naive_baseline.py::test_the_prompt_hash_baseline_catches_a_template_edit` | |
| Neither baseline can see a model-id change | `tests/unit/test_naive_baseline.py::test_neither_baseline_sees_a_model_id_change` | The model id is not prompt text |
| An input-hash cache misses an edit to an imported module's docstring that the prompt quotes | `tests/unit/test_naive_baseline.py::test_the_input_hash_baseline_misses_an_imported_modules_docstring_edit` | |
| The comparison is fair: a baseline artifact is byte-identical to the engine's value when it is not stale | `tests/unit/test_naive_baseline.py::test_a_baseline_artifact_is_byte_identical_to_the_engines_value` | Both send the same requests to the same provider |

## Performance

| Claim | Evidence | Notes |
|---|---|---|
| A no-op rebuild of 1,000 rules takes 322 ms p95 (spec target: under 500 ms) | `bench/bench_baseline.py`, `.github/workflows/ci.yml#bench` | CI enforces 1,200 ms for shared-runner noise; `bench/thresholds.json` explains why |
| Planning 1,000 rules takes 86 ms p95 | `bench/bench_baseline.py` | |
| Loading and authenticating 1,000 trace rows takes 57 ms p95 | `bench/bench_baseline.py` | The floor cost of any build |
| An LLM step fingerprint costs 31 µs p95 | `bench/bench_baseline.py` | |
| Batched audit appends keep one fsync for a whole build's entries | `bench/bench_baseline.py` (`audit_append_many_200`) | 200 entries in 23 ms p95 versus 10 ms each individually |

## LLM steps

| Claim | Evidence | Notes |
|---|---|---|
| The step fingerprint covers template text, model id, `max_tokens`, system prompt, output schema, the redaction flag and every variable | `tests/unit/test_llm_step.py::test_every_part_of_the_request_changes_the_fingerprint`, `tests/unit/test_llm_step.py::test_the_fingerprint_components_name_what_they_cover` | Each component is a separate parametrized case |
| Reordering a variable's keys does not change the fingerprint | `tests/unit/test_llm_step.py::test_reordering_a_variables_keys_does_not_change_the_fingerprint` | Canonical JSON, so equality is semantic |
| A template that drops a variable is an error, not a silently different prompt | `tests/unit/test_llm_step.py::test_a_template_that_drops_a_variable_is_an_error` | |
| Output schemas must forbid extra fields | `tests/unit/test_llm_step.py::test_an_output_schema_must_forbid_extra_fields` | |
| The fake provider is a pure function of the request, so the pipeline is reproducible offline | `tests/unit/test_naive_baseline.py::test_the_fake_provider_is_a_function_of_the_request` | What makes the benchmark comparison exact |

## Security controls

| Claim | Evidence | Notes |
|---|---|---|
| Source text reaches the model only as quoted untrusted data, and cannot close or forge its envelope | `tests/unit/test_llm_step.py::test_source_text_cannot_close_or_forge_an_envelope`, `tests/unit/test_envelope.py::test_content_cannot_open_or_close_an_envelope` | |
| The model is offered no tools at all, so an injection has nothing to drive | `tests/unit/test_llm_step.py::test_a_request_offers_the_model_no_tools` | Structural, not behavioural |
| Containment does not depend on the model resisting an injection | `tests/unit/test_naive_baseline.py::test_a_gullible_model_returns_injected_json_verbatim` | A deliberately obedient model still cannot act |
| A build tool declaring any side-effecting capability cannot be registered | `tests/unit/test_build_tools.py::test_a_tool_with_any_side_effecting_capability_is_refused` | Build steps structurally cannot have effects |
| Tool arguments are validated strictly and never echoed back | `tests/unit/test_build_tools.py::test_invalid_arguments_report_their_location_only` | |
| Every CAS read is SHA-256 verified; any mutation is detected | `tests/property/test_tamper.py::test_any_flipped_byte_in_a_blob_is_detected`, `tests/property/test_tamper.py::test_truncated_and_extended_blobs_are_detected`, `tests/property/test_tamper.py::test_a_blob_swapped_for_another_is_detected`, `bench/bench_robustness.py` | 200/200 in the bench |
| A corrupt value is recomputed and must hash the same | `tests/unit/test_cas.py::test_writing_over_a_corrupt_blob_repairs_it`, `tests/integration/test_demo_and_replay.py::test_the_demo_runs_end_to_end_offline` | The demo asserts 1 integrity event and 1 regeneration |
| A CAS digest can never be used as a path | `tests/unit/test_cas.py::test_only_lowercase_hex_names_are_accepted` | |
| Every trace and step row is MAC-authenticated; an edited or moved row is treated as absent | `tests/property/test_tamper.py::test_editing_any_trace_column_fails_the_mac`, `tests/property/test_tamper.py::test_a_trace_row_copied_to_another_key_fails_the_mac`, `bench/bench_robustness.py` | 200/200 in the bench |
| Shared-cache entries must carry a valid Ed25519 signature from an allowlisted key | `tests/property/test_tamper.py::test_unsigned_and_untrusted_entries_are_rejected`, `tests/property/test_tamper.py::test_editing_a_signed_cache_entry_breaks_its_signature`, `bench/bench_robustness.py` | 200/200 rejected across five attack shapes, 0 accepted |
| An imported entry is re-verified against local inputs before use | `tests/unit/test_shared_cache.py::test_an_imported_entry_is_not_used_when_a_dependency_differs` | |
| Removing a key from the allowlist revokes its already-imported entries | `tests/unit/test_shared_cache.py::test_revoking_a_signer_stops_its_entries_being_used` | |
| A store never re-exports results it imported | `tests/unit/test_shared_cache.py::test_only_locally_executed_traces_are_re_exported` | |
| A volatile tool step is never shared | `tests/unit/test_shared_cache.py::test_a_volatile_tool_step_is_never_exported` | |
| One bad entry does not reject a whole bundle | `tests/unit/test_shared_cache.py::test_a_malformed_entry_body_is_rejected_individually` | |
| Signatures are domain-separated, so one made for another purpose never verifies | `tests/unit/test_signing.py::test_the_signed_bytes_carry_a_domain_prefix` | |
| Effect sinks are never memoized and run only after a successful build | `tests/unit/test_effects.py::test_publishing_after_a_failed_build_is_refused` | |
| A publish approval is bound to the payload hash; one changed byte requires a new approval | `tests/unit/test_effects.py::test_an_approval_for_one_payload_does_not_release_another` | |
| An approval is single-use and does not transfer to another principal | `tests/unit/test_effects.py::test_an_approval_is_single_use`, `tests/unit/test_effects.py::test_an_approval_does_not_transfer_to_another_principal` | |
| A requester cannot approve its own plan or publish | `tests/unit/test_effects.py::test_a_requester_cannot_approve_its_own_plan`, `tests/integration/test_api.py::test_a_requester_cannot_approve_its_own_plan_over_http`, `tests/integration/test_cli.py::test_publishing_needs_an_approval_from_a_second_principal` | |
| A plan approval is bound to the plan hash, including the inputs the approver saw | `tests/unit/test_plan.py::test_the_plan_hash_changes_with_anything_an_approver_saw`, `tests/unit/test_effects.py::test_an_approval_for_one_plan_does_not_authorize_a_different_one` | |
| A plan above the cost threshold is not executed without approval | `tests/integration/test_api.py::test_a_costly_plan_needs_an_approval_before_it_runs` | |
| An approved plan bounds the build's budget | `tests/unit/test_plan.py::test_an_estimated_plan_gets_a_budget_bounded_by_its_estimate`, `tests/unit/test_plan.py::test_a_tight_ceiling_always_wins_over_the_estimate` | |
| Step, token and USD budgets abort a build and keep finished traces | `tests/unit/test_oracle_and_budgets.py::test_a_step_budget_aborts_the_build_and_keeps_finished_traces`, `tests/unit/test_oracle_and_budgets.py::test_a_token_budget_stops_further_steps` | |
| Workspace reads are jailed: no `..`, absolute paths, backslashes or NUL; symlinks out are refused and invisible | `tests/unit/test_workspace.py::test_unsafe_paths_are_refused`, `tests/unit/test_workspace.py::test_a_symlink_out_of_the_jail_is_refused_and_invisible` | |
| The in-memory workspace used by the property tests behaves exactly like the on-disk one | `tests/unit/test_workspace.py::test_glob_matches_the_same_paths_in_both_workspaces` | Same cases run against both |
| Untrusted Python is parsed in a separate process with a timeout and an output cap | `tests/unit/test_ast_tool.py::test_a_worker_that_never_answers_is_killed_and_reported`, `tests/unit/test_ast_tool.py::test_a_worker_that_floods_stdout_is_cut_off_at_the_cap` | |
| A worker that dies between requests is replaced without failing the build, and a timeout is not retried | `tests/unit/test_ast_tool.py::test_a_worker_that_dies_after_answering_is_replaced_transparently`, `tests/unit/test_ast_tool.py::test_a_timeout_is_not_retried` | |
| A rule that reads outside `BuildContext` fails in strict mode and is counted in warn mode | `tests/unit/test_hermeticity.py::test_strict_mode_fails_the_rule_that_read_behind_the_engines_back`, `tests/unit/test_hermeticity.py::test_warn_mode_records_the_violation_and_completes_the_build` | Limits documented in `docs/adr/0004-hermeticity-guard-not-sandbox.md` |
| The guard does not fire on the engine's own reads | `tests/unit/test_hermeticity.py::test_context_reads_are_permitted_in_strict_mode` | |
| Executions, reuse, cache imports, approvals and effects are recorded in a hash-chained log | `tests/integration/test_api.py::test_every_operation_is_recorded_in_a_verifiable_chain`, `tests/integration/test_cli.py::test_the_audit_log_of_a_build_verifies` | |
| A tampered audit log fails verification and names the line | `tests/integration/test_cli.py::test_a_tampered_audit_log_fails_verification`, `tests/unit/test_audit.py::test_any_single_byte_change_is_detected` | |
| Batched appends produce the same chain as individual ones | `tests/unit/test_audit.py::test_a_valid_chain_verifies`, `bench/bench_baseline.py` | |
| No pickle-family serialization anywhere in the source tree | `tests/unit/test_no_pickle.py::test_source_tree_has_no_pickle_family_serialization`, `.github/workflows/ci.yml#semgrep` | |

## Cycles and failures

| Claim | Evidence | Notes |
|---|---|---|
| Every injected dependency cycle raises `CycleError` instead of deadlocking | `tests/property/test_cycles.py::test_every_injected_cycle_raises_cycle_error`, `bench/bench_robustness.py` | Lengths 1–7, 1–4 jobs; 60/60 in the bench |
| A rule that needs its own key is a cycle of length one | `tests/property/test_cycles.py::test_a_self_dependency_is_a_cycle_of_length_one` | |
| Keys outside a cycle still build | `tests/property/test_cycles.py::test_keys_outside_the_cycle_still_build` | |
| A reported cycle names a real ring | `tests/property/test_cycles.py::test_every_injected_cycle_raises_cycle_error` | Asserts the path closes on itself |

## The HTTP API

| Claim | Evidence | Notes |
|---|---|---|
| Each route requires its own scope; a wrong scope is 403 and no token is 401 | `tests/integration/test_api.py::test_a_route_needs_its_own_scope` | All eight routes, three cases each |
| Token verification pins the algorithm and requires every standard claim | `tests/unit/test_api_auth.py::test_an_unsigned_alg_none_token_is_refused`, `tests/unit/test_api_auth.py::test_a_token_using_a_different_algorithm_is_refused`, `tests/unit/test_api_auth.py::test_every_required_claim_is_required` | |
| An over-long token lifetime is refused | `tests/unit/test_api_auth.py::test_a_token_whose_lifetime_exceeds_the_limit_is_refused` | |
| A too-short signing secret is refused by configuration | `tests/unit/test_api_auth.py::test_settings_refuse_a_jwt_secret_that_is_too_short_to_sign_with` | RFC 7518 §3.2 |
| Unknown scope names grant nothing | `tests/unit/test_api_auth.py::test_an_unknown_scope_name_grants_nothing` | |
| Without a configured secret the routes fail closed with 503 | `tests/integration/test_api.py::test_the_routes_fail_closed_without_a_configured_secret` | Never unauthenticated |
| `/readyz` reports the store being writable and the signing keys being loaded | `tests/integration/test_api.py::test_readyz_reports_the_store_and_the_signing_keys`, `tests/integration/test_api.py::test_readyz_is_not_ready_without_signing_keys` | |
| `/metrics` requires a key and exports the build counters | `tests/integration/test_api.py::test_metrics_require_a_key_and_report_the_build_counters` | `rules_executed`, `rules_reused`, `early_cutoffs`, `llm_tokens_saved`, `cache_import_rejections`, `build_duration_seconds` |
| Engine errors become fixed 4xx messages that echo no workspace content | `tests/integration/test_api.py::test_an_unknown_target_is_a_client_error`, `tests/integration/test_api.py::test_invalid_target_arguments_are_a_client_error` | |
| Oversized bodies and bundles, and over-long target lists, are refused | `tests/integration/test_api.py::test_an_oversized_body_is_refused`, `tests/integration/test_api.py::test_too_many_targets_are_refused_by_validation` | |
| The schema is not exposed by default | `tests/integration/test_api.py::test_the_service_does_not_expose_its_schema_by_default` | |

## The offline demo and the CLI

| Claim | Evidence | Notes |
|---|---|---|
| `uv run python tasks.py demo` runs end to end with no API key and no network | `tests/integration/test_demo_and_replay.py::test_the_demo_runs_end_to_end_offline`, `.github/workflows/ci.yml#demo` | The CI job sets no key; `tests/conftest.py` blocks non-loopback sockets |
| The demo shows savings, tamper detection, cache poisoning rejection and the approval gate | `tests/integration/test_demo_and_replay.py::test_the_demo_runs_end_to_end_offline`, `tests/integration/test_demo_and_replay.py::test_the_demo_prints_a_report` | Every printed number is asserted |
| `iab init` generates the 80-module reference workspace | `tests/integration/test_cli.py::test_init_generates_the_reference_workspace` | |
| `iab explain` names the changed input behind a rebuild | `tests/integration/test_cli.py::test_explain_says_why_a_key_would_re_run` | |
| `iab cache export` / `import` round-trip, and reject an untrusted signer | `tests/integration/test_cli.py::test_cache_export_and_import_round_trip`, `tests/integration/test_cli.py::test_importing_a_bundle_from_an_untrusted_signer_rejects_every_entry` | |
| `iab keygen` produces a usable Ed25519 pair | `tests/integration/test_cli.py::test_keygen_prints_a_usable_key_pair` | |
| `iab token` mints a token the service accepts, and refuses an unknown scope | `tests/integration/test_cli.py::test_token_needs_a_secret_and_then_mints_a_verifiable_token`, `tests/integration/test_cli.py::test_token_refuses_an_unknown_scope` | |
| `iab serve` configures uvicorn without an access log or server header | `tests/integration/test_cli.py::test_serve_configures_uvicorn_safely` | |

## The reference workload

| Claim | Evidence | Notes |
|---|---|---|
| The generated package has 80 modules and an acyclic import graph with realistic fan-out | `tests/unit/test_workload.py::test_the_generated_package_has_eighty_modules_and_an_acyclic_import_graph`, `tests/unit/test_workload.py::test_some_modules_are_imported_by_many_so_doc_edits_fan_out` | |
| The generator is deterministic | `tests/unit/test_workload.py::test_the_generator_is_deterministic` | Same seed, byte-identical workspace |
| The history has 30 commits covering 12 change kinds, and every one really changes the workspace | `tests/unit/test_workload.py::test_there_are_thirty_commits_covering_every_change_kind`, `tests/unit/test_workload.py::test_every_commit_changes_the_workspace_and_says_what_it_did` | So the replay is not measuring a no-op |
| A whitespace commit changes no module's AST outline | `tests/unit/test_workload.py::test_a_whitespace_commit_changes_no_outline` | Why early cutoff fires |
| A model-id commit changes only the config file; a template commit only that template | `tests/unit/test_workload.py::test_a_model_id_commit_changes_only_the_config`, `tests/unit/test_workload.py::test_a_template_commit_changes_only_that_template` | |
| Derived rules ignore what should not reach their dependents | `tests/unit/test_workload.py::test_module_api_drops_docstrings_so_doc_edits_do_not_reach_the_release_note`, `tests/unit/test_ast_tool.py::test_comments_and_layout_do_not_change_the_outline` | |

## Engineering baseline

| Claim | Evidence | Notes |
|---|---|---|
| ruff lint and format are clean | `.github/workflows/ci.yml#lint` | |
| mypy `--strict` is clean over src, tests, scripts, bench and tasks.py | `.github/workflows/ci.yml#typecheck` | |
| Test coverage is at or above 85% | `.github/workflows/ci.yml#test` | `fail_under = 85` in pyproject.toml; measured 94% |
| bandit, semgrep (registry packs plus repo rules), pip-audit and gitleaks all run | `.github/workflows/ci.yml#bandit`, `.github/workflows/ci.yml#semgrep`, `.github/workflows/ci.yml#audit`, `.github/workflows/ci.yml#secrets` | |
| The container is non-root, read-only, capability-dropped, digest-pinned, scanned by trivy, and ships an SBOM | `.github/workflows/ci.yml#docker` | Plus a smoke test asserting the HEALTHCHECK reports healthy as UID 10001 |
| Every benchmark has a regression threshold enforced in CI | `bench/thresholds.json`, `.github/workflows/ci.yml#bench`, `.github/workflows/ci.yml#savings` | Each threshold carries a `_note` saying where it came from |
| The repository satisfies the shared portfolio baseline | `scripts/check_baseline.py`, `.github/workflows/ci.yml#baseline` | 523 checks including this file's references |
| Tests never reach the network | `tests/conftest.py` | An autouse fixture blocks non-loopback `socket.connect` |
| The Anthropic SDK is not imported by any offline path | `tests/unit/test_anthropic_provider.py::test_offline_modules_do_not_import_the_sdk` | Checked in a subprocess |

## Claims deliberately not made

| Not claimed | Why | Where stated |
|---|---|---|
| Sandboxed execution of rules or tools | The jail and the audit hook contain careless code reading hostile content, not hostile code | README non-goals, `docs/adr/0004-hermeticity-guard-not-sandbox.md` |
| Detection of a changed rule body without a version bump | Rule code is not hashed | README non-goals, `docs/adr/0002-verifying-traces-over-dirty-bits.md` |
| Determinism of a real model | Exact for `FakeProvider`; an assumption for a live API | README non-goals, `docs/SEMANTICS.md` |
| Concurrent builds from several processes on one store | Builds are serialized per engine; sharing is via signed bundles | README non-goals |
| Distributed execution, prompt caching, streaming, batch requests | Out of scope; orthogonal cost levers | README non-goals |
