"""Dry-run planning: the dirty set, the projected cost, and the budget it authorizes.

A plan runs no rule, so it cannot see early cutoff: its dirty set is an *upper bound* on what a
build will execute. These tests pin that relationship (plan >= executed), the reasons reported, how
never-built keys are priced, and the fact that the plan hash changes whenever anything a caller
approved would change.
"""

from __future__ import annotations

import pytest

from incremental_agent_builds.agentic.budget import BudgetLimits
from incremental_agent_builds.engine.plan import explain, make_plan
from tests.engine_helpers import (
    Counted,
    Doubled,
    Echo,
    Sum,
    counting_rules,
    make_engine,
    workspace_with,
)

pytestmark = pytest.mark.unit

_CEILING = BudgetLimits.of(steps=1000, tokens=1_000_000, usd=10.0)


def test_a_never_built_target_is_new_and_unestimated() -> None:
    engine = make_engine(workspace_with(), counting_rules()[0])
    plan = make_plan(engine, [Doubled(path="in/a.txt")])
    assert [item.status for item in plan.items] == ["new"]
    assert plan.items[0].reason == "new key"
    assert plan.unestimated_keys == 1
    assert plan.clean_keys == 0


def test_an_up_to_date_workspace_plans_no_work() -> None:
    engine = make_engine(workspace_with(), counting_rules()[0])
    engine.build_sync([Doubled(path="in/a.txt")])
    plan = make_plan(engine, [Doubled(path="in/a.txt")])
    assert plan.items == ()
    assert plan.clean_keys == 2
    assert plan.projected_tokens == 0
    assert plan.projected_llm_steps == 0


def test_a_changed_file_makes_its_chain_dirty_with_reasons() -> None:
    workspace = workspace_with()
    engine = make_engine(workspace, counting_rules()[0])
    engine.build_sync([Doubled(path="in/a.txt")])
    workspace.write("in/a.txt", "9")

    plan = make_plan(engine, [Doubled(path="in/a.txt")])
    by_key = {item.key_id: item for item in plan.items}
    assert set(by_key) == {Echo(path="in/a.txt").key_id, Doubled(path="in/a.txt").key_id}
    assert by_key[Echo(path="in/a.txt").key_id].reason == "file in/a.txt changed"
    assert by_key[Doubled(path="in/a.txt").key_id].reason.endswith("may change")
    assert all(item.status == "dirty" for item in plan.items)
    assert all(item.estimated for item in plan.items)


def test_a_changed_config_value_makes_only_its_readers_dirty() -> None:
    workspace = workspace_with()
    engine = make_engine(workspace, counting_rules()[0])
    engine.build_sync([Doubled(path="in/a.txt")])
    workspace.write("iab.toml", '[demo]\nfactor = 3\nmodel = "fake-a"\n')

    plan = make_plan(engine, [Doubled(path="in/a.txt")])
    assert [item.key_id for item in plan.items] == [Doubled(path="in/a.txt").key_id]
    assert "config" in plan.items[0].reason


def test_forcing_a_key_makes_it_dirty_with_that_reason() -> None:
    engine = make_engine(workspace_with(), counting_rules()[0])
    engine.build_sync([Doubled(path="in/a.txt")])
    plan = make_plan(engine, [Doubled(path="in/a.txt")], force=[Echo(path="in/a.txt")])
    by_key = {item.key_id: item.reason for item in plan.items}
    assert by_key[Echo(path="in/a.txt").key_id] == "forced"
    assert plan.force == (Echo(path="in/a.txt").key_id,)


def test_the_dirty_set_is_an_upper_bound_on_what_the_build_executes() -> None:
    """``Counted`` cuts off, so the plan lists it as dirty but the build reuses its dependents."""
    workspace = workspace_with()
    engine = make_engine(workspace, counting_rules()[0])
    engine.build_sync([Counted(modulus=2), Sum(pattern="in/*.txt")])
    workspace.write("in/a.txt", "3")

    plan = make_plan(engine, [Counted(modulus=2), Sum(pattern="in/*.txt")])
    result = engine.build_sync([Counted(modulus=2), Sum(pattern="in/*.txt")])
    assert result.stats.executed <= len(plan.items), "a plan must never under-count"


def test_a_plan_prices_new_keys_from_the_mean_recorded_cost_of_their_rule() -> None:
    workspace = workspace_with()
    engine = make_engine(workspace, counting_rules()[0])
    engine.build_sync([Echo(path="in/a.txt")])
    plan = make_plan(engine, [Echo(path="in/b.txt")])
    assert [item.status for item in plan.items] == ["new"]
    assert plan.items[0].estimated, "a rule with recorded runs can price a new key of that rule"
    assert plan.unestimated_keys == 0


def test_the_plan_hash_changes_with_anything_an_approver_saw() -> None:
    workspace = workspace_with()
    engine = make_engine(workspace, counting_rules()[0])
    engine.build_sync([Doubled(path="in/a.txt")])
    workspace.write("in/a.txt", "9")
    base = make_plan(engine, [Doubled(path="in/a.txt")]).plan_hash

    assert make_plan(engine, [Doubled(path="in/a.txt")]).plan_hash == base
    assert make_plan(engine, [Echo(path="in/a.txt")]).plan_hash != base
    assert (
        make_plan(engine, [Doubled(path="in/a.txt")], force=[Echo(path="in/a.txt")]).plan_hash
        != base
    )
    # A different edit that dirties the same keys at the same projected cost must still produce a
    # different plan hash: an approval is bound to the inputs the approver saw, not just the cost.
    workspace.write("in/a.txt", "10")
    assert make_plan(engine, [Doubled(path="in/a.txt")]).plan_hash != base


def test_approval_is_required_above_the_threshold_or_when_unestimated() -> None:
    engine = make_engine(workspace_with(), counting_rules()[0])
    plan = make_plan(engine, [Doubled(path="in/a.txt")])
    assert plan.unestimated_keys == 1
    assert plan.approval_required(threshold_usd_micros=10**9, unestimated_needs_approval=True)
    assert not plan.approval_required(threshold_usd_micros=10**9, unestimated_needs_approval=False)
    assert plan.approval_required(threshold_usd_micros=-1, unestimated_needs_approval=False)


def test_an_unestimated_plan_gets_the_full_ceiling_budget() -> None:
    engine = make_engine(workspace_with(), counting_rules()[0])
    plan = make_plan(engine, [Doubled(path="in/a.txt")])
    assert plan.budget(factor=1.5, ceiling=_CEILING) == _CEILING


def test_an_estimated_plan_gets_a_budget_bounded_by_its_estimate() -> None:
    workspace = workspace_with()
    engine = make_engine(workspace, counting_rules()[0])
    engine.build_sync([Doubled(path="in/a.txt")])
    workspace.write("in/a.txt", "9")
    plan = make_plan(engine, [Doubled(path="in/a.txt")])
    budget = plan.budget(factor=1.5, ceiling=_CEILING)
    assert budget.not_looser_than(_CEILING)
    assert budget.max_tokens < _CEILING.max_tokens


def test_a_tight_ceiling_always_wins_over_the_estimate() -> None:
    workspace = workspace_with()
    engine = make_engine(workspace, counting_rules()[0])
    engine.build_sync([Doubled(path="in/a.txt")])
    workspace.write("in/a.txt", "9")
    tight = BudgetLimits.of(steps=1, tokens=10, usd=0.0)
    assert make_plan(engine, [Doubled(path="in/a.txt")]).budget(factor=99.0, ceiling=tight) == tight


def test_a_plan_serializes_the_numbers_an_approver_needs() -> None:
    engine = make_engine(workspace_with(), counting_rules()[0])
    payload = make_plan(engine, [Echo(path="in/a.txt")]).to_json()
    assert set(payload) == {
        "plan_sha256",
        "targets",
        "force",
        "dirty",
        "dirty_count",
        "clean_count",
        "projected_llm_steps",
        "projected_tokens",
        "projected_usd_micros",
        "unestimated_keys",
        "store_run",
        "input_sha256",
    }
    assert isinstance(payload["dirty"], list)
    assert payload["dirty_count"] == 1


def test_explain_walks_the_chain_down_to_the_changed_input() -> None:
    workspace = workspace_with()
    engine = make_engine(workspace, counting_rules()[0])
    engine.build_sync([Doubled(path="in/a.txt")])
    workspace.write("in/a.txt", "9")

    chain = explain(engine, Doubled(path="in/a.txt"))
    assert len(chain) == 2
    assert chain[0].startswith(Doubled(path="in/a.txt").key_id)
    assert chain[-1].endswith("file in/a.txt changed")


def test_explain_says_so_when_a_key_is_up_to_date() -> None:
    engine = make_engine(workspace_with(), counting_rules()[0])
    engine.build_sync([Doubled(path="in/a.txt")])
    assert explain(engine, Doubled(path="in/a.txt")) == [
        f"{Doubled(path='in/a.txt').key_id}: up to date"
    ]


def test_a_key_whose_rule_disappeared_is_dirty_with_that_reason() -> None:
    from incremental_agent_builds.engine.rules import RuleSet

    workspace = workspace_with()
    rules, _ = counting_rules()
    engine = make_engine(workspace, rules)
    engine.build_sync([Doubled(path="in/a.txt")])
    echo = rules.get("echo")
    assert echo is not None
    trimmed = make_engine(workspace, RuleSet([echo]), store=engine.store, cas=engine.cas)
    plan = make_plan(trimmed, [Echo(path="in/a.txt")])
    assert plan.items == (), "the surviving key is still clean"
    assert "no rule defines this key" in " ".join(explain(trimmed, 'doubled:{"path":"in/a.txt"}'))
