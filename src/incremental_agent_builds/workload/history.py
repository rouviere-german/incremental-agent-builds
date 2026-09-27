"""The scripted 30-commit history applied to the generated workspace.

Each commit edits the :class:`WorkloadState` deterministically and returns a description. The mix
covers the change kinds a docs pipeline sees in practice: layout-only and comment-only edits,
function and module docstring edits (some on widely imported modules, so importers' summaries
depend on them through "referenced files"), signature changes, added and removed functions and
methods, a module rename, new modules, prompt-template edits and model-id changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from incremental_agent_builds.workload.generator import WorkloadState
from incremental_agent_builds.workload.model import FunctionSpec, ModuleSpec

__all__ = ["COMMITS", "Commit", "apply_commit"]


@dataclass(frozen=True, slots=True)
class Commit:
    kind: str
    apply: Callable[[WorkloadState], str]


def _module(state: WorkloadState, index: int) -> ModuleSpec:
    modules = state.model.modules
    return modules[index % len(modules)]


def _most_imported(state: WorkloadState, rank: int) -> ModuleSpec:
    root = state.model.root
    ranked = sorted(
        state.model.modules,
        key=lambda m: (-len(state.model.importers_of(m.dotted(root))), m.dotted(root)),
    )
    return ranked[rank]


def _name(state: WorkloadState, module: ModuleSpec) -> str:
    return module.dotted(state.model.root)


def whitespace(index: int) -> Commit:
    def apply(state: WorkloadState) -> str:
        module = _module(state, index)
        module.style += 1
        return f"reformat {_name(state, module)} (blank lines and trailing spaces only)"

    return Commit("whitespace", apply)


def comment(index: int, text: str) -> Commit:
    def apply(state: WorkloadState) -> str:
        module = _module(state, index)
        module.comments.append(text)
        return f"add a comment to {_name(state, module)}"

    return Commit("comment", apply)


def function_doc(index: int, suffix: str) -> Commit:
    def apply(state: WorkloadState) -> str:
        module = _module(state, index)
        target = module.functions[0]
        target.doc = target.doc.rstrip(".") + ". " + suffix
        return f"reword the docstring of {_name(state, module)}.{target.name}"

    return Commit("function_doc", apply)


def module_doc(rank: int, sentence: str) -> Commit:
    def apply(state: WorkloadState) -> str:
        module = _most_imported(state, rank)
        module.doc = module.doc + " " + sentence
        importers = len(state.model.importers_of(_name(state, module)))
        return f"extend the module docstring of {_name(state, module)} (imported by {importers})"

    return Commit("module_doc", apply)


def signature(index: int) -> Commit:
    def apply(state: WorkloadState) -> str:
        module = _module(state, index)
        target = module.functions[-1]
        target.params.append("verbose: bool = False")
        return f"add a keyword parameter to {_name(state, module)}.{target.name}"

    return Commit("signature", apply)


def return_type(index: int) -> Commit:
    def apply(state: WorkloadState) -> str:
        module = _module(state, index)
        target = module.functions[0]
        target.returns = "dict[str, str]" if target.returns != "dict[str, str]" else "list[str]"
        return f"change the return type of {_name(state, module)}.{target.name}"

    return Commit("signature", apply)


def add_function(index: int, name: str) -> Commit:
    def apply(state: WorkloadState) -> str:
        module = _module(state, index)
        module.functions.append(
            FunctionSpec(
                name=name, params=["path: str"], returns="bool", doc="Check the path exists."
            )
        )
        return f"add {_name(state, module)}.{name}"

    return Commit("add_function", apply)


def remove_function(index: int) -> Commit:
    def apply(state: WorkloadState) -> str:
        module = _module(state, index)
        removed = module.functions.pop()
        return f"remove {_name(state, module)}.{removed.name}"

    return Commit("remove_function", apply)


def add_method(occurrence: int) -> Commit:
    def apply(state: WorkloadState) -> str:
        with_classes = [m for m in state.model.modules if m.classes]
        module = with_classes[occurrence % len(with_classes)]
        cls = module.classes[0]
        cls.methods.append(
            FunctionSpec(
                name="describe", params=["self"], returns="str", doc="Describe the object."
            )
        )
        return f"add {cls.name}.describe in {_name(state, module)}"

    return Commit("add_method", apply)


def rename(index: int, new_name: str) -> Commit:
    def apply(state: WorkloadState) -> str:
        module = _module(state, index)
        old = _name(state, module)
        module.name = new_name
        new = _name(state, module)
        for other in state.model.modules:
            if old in other.imports:
                other.imports = sorted(new if item == old else item for item in other.imports)
        return f"rename {old} to {new}"

    return Commit("rename", apply)


def new_module(sub: str, name: str, imports: list[int]) -> Commit:
    def apply(state: WorkloadState) -> str:
        targets = sorted({_name(state, _module(state, i)) for i in imports})
        module = ModuleSpec(
            sub=sub,
            name=name,
            doc=f"{name.capitalize()} helpers for the {sub} layer, added in this release.",
            imports=targets,
            functions=[
                FunctionSpec(
                    name=f"run_{name}", params=["data: bytes"], returns="bytes", doc="Run it."
                ),
                FunctionSpec(
                    name=f"check_{name}", params=["key: str"], returns="bool", doc="Check it."
                ),
            ],
        )
        state.model.modules.append(module)
        return f"add module {_name(state, module)}"

    return Commit("new_module", apply)


def template(name: str, extra: str) -> Commit:
    def apply(state: WorkloadState) -> str:
        state.templates[name] = state.templates[name] + extra
        return f"edit the {name} prompt template"

    return Commit("template", apply)


def model_id(role: str, model: str) -> Commit:
    def apply(state: WorkloadState) -> str:
        state.models[role] = model
        return f"switch models.{role} to {model}"

    return Commit("model_id", apply)


COMMITS: tuple[Commit, ...] = (
    whitespace(5),
    function_doc(12, "Returns early on empty input."),
    module_doc(0, "All functions are safe to call from several threads."),
    signature(20),
    comment(33, "TODO: revisit the error messages."),
    whitespace(41),
    new_module("io", "compress", [5, 16]),
    function_doc(47, "Raises ValueError on malformed input."),
    template("module_summary", "Prefer plain words over jargon.\n"),
    model_id("release", "claude-sonnet-5"),
    rename(25, "header_utils"),
    whitespace(58),
    module_doc(1, "Configuration is read once at start-up."),
    return_type(63),
    comment(70, "Performance-sensitive: avoid extra copies."),
    add_function(15, "path_exists"),
    function_doc(3, "The result is cached per process."),
    model_id("package", "claude-sonnet-5"),
    whitespace(77),
    remove_function(30),
    template("release_note", "Mention migration steps for breaking changes.\n"),
    module_doc(5, "See the design notes in the repository."),
    comment(9, "Kept for backwards compatibility."),
    new_module("text", "highlight", [30, 31]),
    function_doc(52, "Timeouts are in seconds."),
    signature(8),
    whitespace(22),
    module_doc(2, "Errors are reported, never swallowed."),
    add_method(0),
    function_doc(66, "Returns None when nothing matches."),
)


def apply_commit(state: WorkloadState, index: int) -> tuple[str, str]:
    """Apply commit ``index`` (0-based) and return (kind, description)."""
    commit = COMMITS[index]
    return commit.kind, commit.apply(state)
