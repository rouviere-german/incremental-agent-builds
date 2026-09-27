"""Deterministic generator of the reference workspace (an 80-module Python package).

Everything is derived from a seed with ``random.Random``, so two runs produce byte-identical
workspaces. Modules import only modules generated before them (the import graph is acyclic), and a
few early modules are imported by many others, which gives docstring edits a realistic fan-out.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from incremental_agent_builds.core.fileio import atomic_write_text
from incremental_agent_builds.engine._ast_worker import outline
from incremental_agent_builds.workload.model import (
    ClassSpec,
    FunctionSpec,
    ModuleSpec,
    PackageModel,
)
from incremental_agent_builds.workload.rules import (
    BASELINE_API_PATH,
    module_api,
    normalize_outline,
)

__all__ = [
    "CONFIG_PATH",
    "DEFAULT_MODELS",
    "TEMPLATES",
    "WorkloadState",
    "generate_state",
    "write_workspace",
]

CONFIG_PATH = "iab.toml"
DEFAULT_MODELS = {
    "summary": "claude-opus-5",
    "package": "claude-opus-5",
    "release": "claude-opus-5",
}

SUBPACKAGES: dict[str, list[str]] = {
    "core": ["engine", "registry", "config", "events", "errors", "state", "hooks", "context", "plugins", "lifecycle"],
    "io": ["reader", "writer", "paths", "archive", "stream", "codec", "buffer", "watcher", "locks", "formats"],
    "net": ["client", "server", "retry", "session", "headers", "routing", "sockets", "proxy", "auth", "pool"],
    "text": ["tokenize", "normalize", "template", "render", "markdown", "slug", "wrap", "diff", "search", "spell"],
    "numeric": ["vector", "matrix", "stats", "interp", "solver", "geometry", "units", "rounding", "sampling", "bounds"],
    "util": ["cache", "timing", "flags", "ids", "dates", "collections", "env", "validate", "backoff", "hashing"],
    "data": ["schema", "table", "query", "index", "migrate", "rows", "joins", "filters", "export", "catalog"],
    "cli": ["main", "commands", "prompts", "colors", "progress", "args", "output", "settings", "shell", "completion"],
}  # fmt: skip
VERBS = ["load", "save", "parse", "build", "render", "fetch", "merge", "split", "check", "resolve",
         "format", "compute", "update", "create", "remove", "scan", "open", "close", "apply", "convert"]  # fmt: skip
NOUNS = ["config", "record", "path", "token", "batch", "header", "table", "value", "entry", "report",
         "index", "buffer", "event", "item", "session", "plan", "field", "chunk", "key", "range"]  # fmt: skip
PURPOSES = ["for the caller", "from the current source", "into a normalized form",
            "without side effects", "using the configured defaults", "for later lookups",
            "and reports problems", "in a single pass"]  # fmt: skip
PARAMS = ["path: str", "data: bytes", "limit: int = 100", "strict: bool = False", "name: str",
          "items: list[str]", "timeout: float = 5.0", "encoding: str = 'utf-8'",
          "retries: int = 3", "key: str"]  # fmt: skip
RETURNS = ["str", "int", "bool", "None", "dict[str, str]", "list[str]", "bytes", "float"]
ADJECTIVES = ["small", "composable", "thread-safe", "lazy", "strict", "portable", "incremental"]

TEMPLATES: dict[str, str] = {
    "module_summary": (
        "Summarize the Python module {{module}} for the package's developer documentation.\n"
        "Say what the module is for and name its most important public functions and classes.\n\n"
        "Public API of the module, with the first line of each docstring:\n{{outline}}\n\n"
        "Module docstrings of the internal modules it imports:\n{{imported}}\n\n"
        "Write at most three sentences and up to five key points.\n"
    ),
    "package_summary": (
        "Write the overview page for the package {{package}}.\n"
        "Group the modules by purpose and highlight the entry points a new user needs.\n\n"
        "Module summaries:\n{{modules}}\n\n"
        "Write one paragraph and up to eight highlights.\n"
    ),
    "release_note": (
        "Write release notes for the public API changes below.\n"
        "Call out breaking changes (removed symbols or changed signatures) first.\n\n"
        "Changes since the last release:\n{{changes}}\n\n"
        "Write a short title and one bullet per change.\n"
    ),
}


@dataclass
class WorkloadState:
    """Everything the commit history edits: the package model, templates and config."""

    model: PackageModel
    templates: dict[str, str] = field(default_factory=lambda: dict(TEMPLATES))
    models: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODELS))
    baseline_api: str = ""
    extras: dict[str, str] = field(default_factory=dict)

    def config_text(self) -> str:
        lines = [
            "# Workspace configuration read by the build rules.",
            "[package]",
            f'root = "{self.model.root}"',
            f'name = "{self.model.name}"',
            "",
            "[models]",
        ]
        lines += [f'{role} = "{model}"' for role, model in sorted(self.models.items())]
        return "\n".join(lines) + "\n"

    def files(self) -> dict[str, str]:
        files = self.model.files()
        files.update({f"templates/{name}.txt": text for name, text in self.templates.items()})
        files[CONFIG_PATH] = self.config_text()
        files[BASELINE_API_PATH] = self.baseline_api
        files.update(self.extras)
        return files


def _function(rng: random.Random, used: set[str]) -> FunctionSpec:
    while True:
        name = f"{rng.choice(VERBS)}_{rng.choice(NOUNS)}"
        if name not in used:
            used.add(name)
            break
    params = rng.sample(PARAMS, rng.randint(1, 3))
    params.sort(key=lambda p: "=" in p)  # parameters without defaults first
    verb, noun = name.split("_", 1)
    doc = f"{verb.capitalize()} the {noun} {rng.choice(PURPOSES)}."
    return FunctionSpec(name=name, params=params, returns=rng.choice(RETURNS), doc=doc)


def _class(rng: random.Random, module_name: str) -> ClassSpec:
    name = "".join(part.capitalize() for part in module_name.split("_")) + rng.choice(
        ["Manager", "Store", "Builder", "Handler", "Registry"]
    )
    used: set[str] = set()
    methods = []
    for _ in range(rng.randint(1, 3)):
        fn = _function(rng, used)
        fn.params = ["self", *fn.params]
        methods.append(fn)
    doc = f"A {rng.choice(ADJECTIVES)} {name[0].lower() + name[1:]} for {module_name} work."
    return ClassSpec(name=name, base="object", doc=doc, methods=methods)


def generate_model(seed: int = 7, *, root: str = "pkg", name: str = "acme-toolkit") -> PackageModel:
    # Reproducible fixtures, not cryptography: the same seed must give the same workspace.
    rng = random.Random(seed)  # noqa: S311  # nosec B311
    modules: list[ModuleSpec] = []
    for sub, names in SUBPACKAGES.items():
        for module_name in names:
            index = len(modules)
            imports: list[str] = []
            if index:
                count = rng.choice([0, 1, 1, 2, 2, 3])
                # Earlier modules are likelier targets, so a few become widely imported.
                pool = [m.dotted(root) for m in modules]
                weights = [1.0 / (1 + i) ** 0.7 for i in range(len(pool))]
                for _ in range(count):
                    choice = rng.choices(pool, weights=weights)[0]
                    if choice not in imports:
                        imports.append(choice)
                imports.sort()
            used: set[str] = set()
            functions = [_function(rng, used) for _ in range(rng.randint(2, 4))]
            classes = [_class(rng, module_name)] if rng.random() < 0.5 else []
            title = module_name.replace("_", " ").capitalize()
            doc = (
                f"{title} helpers for the {sub} layer. "
                f"Provides {rng.choice(ADJECTIVES)} building blocks used across the package."
            )
            comments = (
                [f"{sub}/{module_name}: keep the public API small."] if rng.random() < 0.3 else []
            )
            modules.append(
                ModuleSpec(
                    sub=sub,
                    name=module_name,
                    doc=doc,
                    imports=imports,
                    functions=functions,
                    classes=classes,
                    comments=comments,
                )
            )
    return PackageModel(root=root, name=name, subpackages=list(SUBPACKAGES), modules=modules)


def api_snapshot_text(model: PackageModel) -> str:
    """The ApiSnapshot value for ``model``, computed in-process with the rules' normalization."""
    modules: dict[str, Any] = {}
    files = model.files()
    for module in model.modules:
        path = module.path(model.root)
        raw = outline(files[path])
        normalized = normalize_outline(path, raw, model.root)
        api = module_api(normalized)
        modules[str(api["module"])] = {"symbols": api["symbols"]}
    return json.dumps({"modules": modules}, indent=1, sort_keys=True) + "\n"


def generate_state(seed: int = 7) -> WorkloadState:
    model = generate_model(seed)
    return WorkloadState(model=model, baseline_api=api_snapshot_text(model))


def write_workspace(
    root: Path, state: WorkloadState, *, previous: dict[str, str] | None = None
) -> list[str]:
    """Write the workspace files; with ``previous``, write only changed files and delete removed
    ones. Returns the changed paths."""
    files = state.files()
    changed: list[str] = []
    for path, text in sorted(files.items()):
        if previous is not None and previous.get(path) == text:
            continue
        atomic_write_text(root.joinpath(*path.split("/")), text)
        changed.append(path)
    if previous is not None:
        for path in sorted(set(previous) - set(files)):
            root.joinpath(*path.split("/")).unlink(missing_ok=True)
            changed.append(path)
    return changed
