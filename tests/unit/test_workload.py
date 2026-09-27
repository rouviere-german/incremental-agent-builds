"""The reference workload: normalization, the rules' dependency shape, and the commit history.

The workload is what the savings benchmark measures, so its two design decisions matter and are
tested here:

* each derived value ignores the things that should not reach its dependents. ``ModuleOutline``
  ignores layout and comments; ``ModuleDoc`` keeps only the module docstring; ``ModuleApi`` drops
  docstrings. That is what turns a whitespace commit into one cheap rule execution and a
  function-docstring edit into a rebuild that never reaches importers.
* every commit kind in the scripted history really changes the workspace, so the replay is not
  measuring a no-op.
"""

from __future__ import annotations

import json

import pytest

from incremental_agent_builds.engine._ast_worker import outline
from incremental_agent_builds.workload.generator import (
    CONFIG_PATH,
    DEFAULT_MODELS,
    TEMPLATES,
    WorkloadState,
    api_snapshot_text,
    generate_model,
    generate_state,
    write_workspace,
)
from incremental_agent_builds.workload.history import COMMITS, apply_commit
from incremental_agent_builds.workload.model import ModuleSpec, render_module
from incremental_agent_builds.workload.rules import (
    BASELINE_API_PATH,
    api_diff,
    module_api,
    normalize_outline,
    path_to_dotted,
    resolve_imports,
    summary_variables,
    workload_rules,
)

pytestmark = pytest.mark.unit


# --- analysis helpers ---------------------------------------------------------------------------


def test_path_to_dotted_maps_a_module_path_to_its_import_name() -> None:
    assert path_to_dotted("pkg/core/engine.py") == "pkg.core.engine"
    assert path_to_dotted("pkg/__init__.py") == "pkg.__init__"


@pytest.mark.parametrize(
    ("imports", "expected"),
    [
        (
            [{"module": "pkg.core", "level": 0, "names": ["engine"]}],
            ["pkg.core", "pkg.core.engine"],
        ),
        ([{"module": "os", "level": 0, "names": []}], []),
        ([{"module": "pkg", "level": 0, "names": []}], ["pkg"]),
        ([{"module": "", "level": 1, "names": ["sibling"]}], ["pkg.core", "pkg.core.sibling"]),
        ([{"module": "io", "level": 2, "names": ["reader"]}], ["pkg.io", "pkg.io.reader"]),
        ([{"module": "other.pkg", "level": 0, "names": ["x"]}], []),
    ],
)
def test_only_in_package_imports_become_dependencies(
    imports: list[dict[str, object]], expected: list[str]
) -> None:
    assert resolve_imports("pkg.core.engine", imports, "pkg") == expected


def test_normalize_outline_produces_the_module_outline_value() -> None:
    source = (
        '"""Doc line.\n\nSecond paragraph."""\n\n'
        "from pkg.io import reader\n\n\n"
        'def run(x: int) -> str:\n    """Run it."""\n    return ""\n'
    )
    value = normalize_outline("pkg/core/engine.py", outline(source), "pkg")
    assert value["module"] == "pkg.core.engine"
    assert value["path"] == "pkg/core/engine.py"
    assert value["doc"] == "Doc line.\n\nSecond paragraph."
    assert value["imports"] == ["pkg.io", "pkg.io.reader"]
    assert value["symbols"] == [
        {"kind": "function", "name": "run", "signature": "def run(x: int) -> str", "doc": "Run it."}
    ]
    assert value["error"] is None


def test_normalize_outline_carries_a_parse_error_as_a_value() -> None:
    value = normalize_outline("pkg/core/bad.py", {"error": "SyntaxError"}, "pkg")
    assert value["error"] == "SyntaxError"
    assert value["symbols"] == []


def test_module_api_drops_docstrings_so_doc_edits_do_not_reach_the_release_note() -> None:
    documented = normalize_outline(
        "pkg/a.py", outline('def run() -> None:\n    """First."""\n'), "pkg"
    )
    reworded = normalize_outline(
        "pkg/a.py", outline('def run() -> None:\n    """Second, different."""\n'), "pkg"
    )
    assert documented != reworded
    assert module_api(documented) == module_api(reworded)


def test_module_api_keeps_signatures_including_method_signatures() -> None:
    value = normalize_outline(
        "pkg/a.py",
        outline(
            'class Store:\n    """S."""\n\n    def put(self, k: str) -> None:\n        """P."""\n'
        ),
        "pkg",
    )
    api = module_api(value)
    assert api["symbols"] == [
        {
            "kind": "class",
            "name": "Store",
            "signature": "class Store",
            "methods": ["def put(self, k: str) -> None"],
        }
    ]


def test_api_diff_reports_added_removed_and_changed_symbols() -> None:
    def snapshot(signature: str) -> dict[str, object]:
        return {
            "modules": {
                "pkg.a": {"symbols": [{"kind": "function", "name": "run", "signature": signature}]}
            }
        }

    unchanged = api_diff(snapshot("def run() -> None"), snapshot("def run() -> None"))
    assert unchanged == {"added": [], "removed": [], "changed": []}

    changed = api_diff(snapshot("def run() -> None"), snapshot("def run(x: int) -> None"))
    assert changed["changed"] == ["pkg.a.run: def run() -> None -> def run(x: int) -> None"]

    added = api_diff({"modules": {}}, snapshot("def run() -> None"))
    assert added["added"] == ["pkg.a.run: def run() -> None"]
    removed = api_diff(snapshot("def run() -> None"), {"modules": {}})
    assert removed["removed"] == ["pkg.a.run: def run() -> None"]


def test_summary_variables_send_only_the_outline_and_imported_docs() -> None:
    """The prompt's variables are exactly the trace's inputs: nothing hidden reaches the model."""
    value = normalize_outline("pkg/a.py", outline('def run() -> None:\n    """R."""\n'), "pkg")
    variables = summary_variables(value, [{"module": "pkg.b", "doc": "B."}])
    assert set(variables) == {"module", "outline", "imported"}
    assert variables["outline"] == {
        "doc": "",
        "symbols": [{"name": "run", "signature": "def run() -> None", "doc": "R."}],
    }


# --- the rule set -------------------------------------------------------------------------------


def test_the_rule_set_covers_every_derived_artifact() -> None:
    assert workload_rules().kinds == (
        "api_diff",
        "api_snapshot",
        "docs_site",
        "glossary",
        "module_api",
        "module_doc",
        "module_list",
        "module_outline",
        "module_summary",
        "package_summary",
        "release_note",
    )


# --- the generator ------------------------------------------------------------------------------


def test_the_generator_is_deterministic() -> None:
    assert generate_state(7).files() == generate_state(7).files()
    assert generate_state(7).files() != generate_state(8).files()


def test_the_generated_package_has_eighty_modules_and_an_acyclic_import_graph() -> None:
    model = generate_model()
    assert len(model.modules) == 80
    order = {module.dotted(model.root): index for index, module in enumerate(model.modules)}
    for module in model.modules:
        for imported in module.imports:
            assert order[imported] < order[module.dotted(model.root)], (
                "imports must point backwards"
            )


def test_some_modules_are_imported_by_many_so_doc_edits_fan_out() -> None:
    model = generate_model()
    counts = [len(model.importers_of(m.dotted(model.root))) for m in model.modules]
    assert max(counts) >= 5, "the savings benchmark needs a realistic fan-out"


def test_the_workspace_contains_the_package_templates_config_and_baseline() -> None:
    files = generate_state().files()
    assert CONFIG_PATH in files
    assert BASELINE_API_PATH in files
    for name in TEMPLATES:
        assert f"templates/{name}.txt" in files
    assert files["pkg/__init__.py"].startswith('"""')
    assert 'root = "pkg"' in files[CONFIG_PATH]
    for role, model in DEFAULT_MODELS.items():
        assert f'{role} = "{model}"' in files[CONFIG_PATH]


def test_the_baseline_api_snapshot_matches_the_rules_normalization() -> None:
    """The release note diffs against this file, so it must be produced the same way."""
    model = generate_model()
    snapshot = json.loads(api_snapshot_text(model))
    assert set(snapshot["modules"]) == {m.dotted(model.root) for m in model.modules}


def test_a_style_change_alters_the_text_but_not_the_ast() -> None:
    model = generate_model()
    module = model.modules[0]
    before = render_module(module, model.root)
    module.style += 1
    after = render_module(module, model.root)
    assert before != after, "a style change must change the file"
    assert outline(before) == outline(after), "and must not change the outline"


def test_write_workspace_writes_only_changed_files_and_deletes_removed_ones(
    tmp_path: object,
) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    state = generate_state()
    first = write_workspace(tmp_path, state)
    assert len(first) == len(state.files())
    assert write_workspace(tmp_path, state, previous=state.files()) == []

    previous = state.files()
    state.model.modules[0].style += 1
    changed = write_workspace(tmp_path, state, previous=previous)
    assert changed == [state.model.modules[0].path(state.model.root)]

    previous = state.files()
    removed = state.model.modules.pop()
    deleted = write_workspace(tmp_path, state, previous=previous)
    assert removed.path(state.model.root) in deleted
    assert not tmp_path.joinpath(*removed.path(state.model.root).split("/")).exists()


# --- the commit history -------------------------------------------------------------------------


def test_there_are_thirty_commits_covering_every_change_kind() -> None:
    assert len(COMMITS) == 30
    kinds = {commit.kind for commit in COMMITS}
    assert kinds == {
        "whitespace",
        "comment",
        "function_doc",
        "module_doc",
        "signature",
        "add_function",
        "remove_function",
        "add_method",
        "rename",
        "new_module",
        "template",
        "model_id",
    }


def test_every_commit_changes_the_workspace_and_says_what_it_did() -> None:
    state = generate_state()
    files = state.files()
    for index in range(len(COMMITS)):
        kind, description = apply_commit(state, index)
        after = state.files()
        assert after != files, f"commit {index} ({kind}) changed nothing"
        assert description, f"commit {index} has no description"
        files = after


def test_a_whitespace_commit_changes_no_outline() -> None:
    state = generate_state()
    before = {m.path(state.model.root): outline(t) for m, t in _rendered(state)}
    index = next(i for i, c in enumerate(COMMITS) if c.kind == "whitespace")
    apply_commit(state, index)
    after = {m.path(state.model.root): outline(t) for m, t in _rendered(state)}
    assert before == after


def test_a_model_id_commit_changes_only_the_config() -> None:
    state = generate_state()
    before = state.files()
    index = next(i for i, c in enumerate(COMMITS) if c.kind == "model_id")
    apply_commit(state, index)
    after = state.files()
    assert {k for k in after if after[k] != before.get(k)} == {CONFIG_PATH}


def test_a_template_commit_changes_only_that_template() -> None:
    state = generate_state()
    before = state.files()
    index = next(i for i, c in enumerate(COMMITS) if c.kind == "template")
    apply_commit(state, index)
    after = state.files()
    changed = {k for k in after if after[k] != before.get(k)}
    assert len(changed) == 1
    assert next(iter(changed)).startswith("templates/")


def _rendered(state: WorkloadState) -> list[tuple[ModuleSpec, str]]:
    files = state.files()
    return [(m, files[m.path(state.model.root)]) for m in state.model.modules]
