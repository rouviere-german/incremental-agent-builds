"""An in-memory model of the generated reference package and its deterministic rendering.

The workload generator builds a :class:`PackageModel`; the scripted commit history edits the model
(or the templates and config) and re-renders only the files that changed. Rendering is a pure
function of the model, so every edit is exactly reproducible. ``style`` selects one of several
layouts that differ only in whitespace, which is how whitespace-only commits are produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ClassSpec", "FunctionSpec", "ModuleSpec", "PackageModel", "render_module"]


@dataclass
class FunctionSpec:
    name: str
    params: list[str]
    returns: str
    doc: str


@dataclass
class ClassSpec:
    name: str
    base: str
    doc: str
    methods: list[FunctionSpec] = field(default_factory=list)


@dataclass
class ModuleSpec:
    sub: str
    name: str
    doc: str
    imports: list[str] = field(default_factory=list)
    functions: list[FunctionSpec] = field(default_factory=list)
    classes: list[ClassSpec] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)
    style: int = 0

    def path(self, root: str) -> str:
        return f"{root}/{self.sub}/{self.name}.py"

    def dotted(self, root: str) -> str:
        return f"{root}.{self.sub}.{self.name}"


@dataclass
class PackageModel:
    root: str
    name: str
    subpackages: list[str]
    modules: list[ModuleSpec]

    def by_dotted(self, dotted: str) -> ModuleSpec:
        for module in self.modules:
            if module.dotted(self.root) == dotted:
                return module
        raise KeyError(dotted)

    def importers_of(self, dotted: str) -> list[ModuleSpec]:
        return [m for m in self.modules if dotted in m.imports]

    def files(self) -> dict[str, str]:
        rendered = {m.path(self.root): render_module(m, self.root) for m in self.modules}
        rendered[f"{self.root}/__init__.py"] = f'"""The {self.name} package."""\n'
        for sub in self.subpackages:
            rendered[f"{self.root}/{sub}/__init__.py"] = f'"""The {sub} layer."""\n'
        return rendered


def _body(function: FunctionSpec, indent: str, trailing: str) -> list[str]:
    first = function.params[0].split(":")[0].split("=")[0].strip() if function.params else ""
    lines = [f'{indent}    """{function.doc}"""']
    if first and first not in {"self", "cls"}:
        lines.append(f"{indent}    result = {first}{trailing}")
        lines.append(f"{indent}    return result{trailing}")
    else:
        lines.append(f"{indent}    return None{trailing}")
    return lines


def _function(function: FunctionSpec, indent: str, trailing: str) -> list[str]:
    params = ", ".join(function.params)
    header = f"{indent}def {function.name}({params}) -> {function.returns}:"
    return [header, *_body(function, indent, trailing)]


def render_module(module: ModuleSpec, root: str) -> str:
    """Render a module. ``style`` changes only blank lines and trailing spaces (not the AST)."""
    gap = 2 + (module.style % 2)
    trailing = "  " if module.style % 3 == 2 else ""
    lines = [f'"""{module.doc}"""', "", "from __future__ import annotations", ""]
    for dotted in module.imports:
        package, _, name = dotted.rpartition(".")
        lines.append(f"from {package} import {name}{trailing}")
    if module.imports:
        lines.append("")
    if module.style % 4 == 3:
        lines.append("")
    for comment in module.comments:
        lines.append(f"# {comment}")
    if module.comments:
        lines.append("")
    blocks: list[list[str]] = [_function(fn, "", trailing) for fn in module.functions]
    for cls in module.classes:
        block = [f"class {cls.name}({cls.base}):", f'    """{cls.doc}"""']
        for method in cls.methods:
            block.append("")
            block.extend(_function(method, "    ", trailing))
        blocks.append(block)
    for index, block in enumerate(blocks):
        if index:
            lines.extend([""] * gap)
        lines.extend(block)
    if module.imports:
        names = ", ".join(dotted.rpartition(".")[2] for dotted in module.imports)
        lines.extend([""] * gap)
        lines.append(f"_USES = ({names},)")
    return "\n".join(lines) + "\n"
