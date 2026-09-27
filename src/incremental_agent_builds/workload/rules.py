"""Rules of the reference workload: derived documentation for a changing Python package.

Dependency structure (all edges are discovered at runtime through BuildContext)::

    DocsSite -> PackageSummary -> ModuleSummary(p) -> ModuleOutline(p) -> ast_outline tool -> file
             |                                     -> ModuleDoc(q) for each imported module q
             -> Glossary -> ModuleOutline(p)
             -> ReleaseNote -> ApiDiff -> ApiSnapshot -> ModuleApi(p) -> ModuleOutline(p)
                                      -> release/baseline_api.json
    ModuleList (glob) is needed by every aggregate, so new, renamed and deleted modules are seen.

Early cutoff happens at several levels: ``ModuleOutline`` ignores layout and comments,
``ModuleDoc`` ignores everything but the module docstring (so a function-docstring edit does not
reach importers), ``ModuleApi`` ignores docstrings (so doc edits do not reach the release note),
and ``ApiDiff`` only changes when a signature does.

The three LLM rules (module summary, package summary, release note) read their model id from
``iab.toml`` (fine-grained config dependencies) and send source-derived text only as quoted,
untrusted data with a strict output schema.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from incremental_agent_builds.engine.context import BuildContext
from incremental_agent_builds.engine.keys import BuildKey
from incremental_agent_builds.engine.rules import Rule, RuleSet, rule

__all__ = [
    "ApiDiff",
    "ApiSnapshot",
    "DocsSite",
    "Glossary",
    "ModuleApi",
    "ModuleDoc",
    "ModuleList",
    "ModuleOutline",
    "ModuleSummary",
    "ModuleSummaryOut",
    "PackageSummary",
    "PackageSummaryOut",
    "ReleaseNote",
    "ReleaseNoteOut",
    "api_diff",
    "module_api",
    "normalize_outline",
    "path_to_dotted",
    "resolve_imports",
    "workload_rules",
]

BASELINE_API_PATH = "release/baseline_api.json"
DEFAULT_ROOT = "pkg"


# --- keys ---------------------------------------------------------------------------------------


class ModuleList(BuildKey):
    pass


class ModuleOutline(BuildKey):
    path: str


class ModuleDoc(BuildKey):
    path: str


class ModuleApi(BuildKey):
    path: str


class ModuleSummary(BuildKey):
    path: str


class PackageSummary(BuildKey):
    pass


class Glossary(BuildKey):
    pass


class ApiSnapshot(BuildKey):
    pass


class ApiDiff(BuildKey):
    pass


class ReleaseNote(BuildKey):
    pass


class DocsSite(BuildKey):
    pass


# --- LLM output schemas --------------------------------------------------------------------------


class ModuleSummaryOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1, max_length=1200)
    key_points: list[str] = Field(max_length=6)


class PackageSummaryOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    overview: str = Field(min_length=1, max_length=2000)
    highlights: list[str] = Field(max_length=8)


class ReleaseNoteOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=200)
    bullets: list[str] = Field(max_length=20)


# --- pure analysis helpers (shared with the generator and the naive baseline) --------------------


def path_to_dotted(path: str) -> str:
    return path.removesuffix(".py").replace("/", ".")


def resolve_imports(module: str, imports: list[dict[str, Any]], root: str) -> list[str]:
    """Dotted names an import list may refer to inside ``root`` (submodules and packages)."""
    package_parts = module.split(".")[:-1]
    found: set[str] = set()
    for item in imports:
        level = int(item.get("level", 0))
        name = str(item.get("module", ""))
        names = [str(n) for n in item.get("names", [])]
        if level:
            base_parts = (
                package_parts[: len(package_parts) - (level - 1)] if level > 1 else package_parts
            )
            base = ".".join([*base_parts, *([name] if name else [])])
        else:
            base = name
        candidates = [f"{base}.{n}" for n in names] + [base] if names else [base]
        for candidate in candidates:
            if candidate == root or candidate.startswith(root + "."):
                found.add(candidate)
    return sorted(found)


def normalize_outline(path: str, raw: dict[str, Any], root: str) -> dict[str, JsonValue]:
    """The ModuleOutline value: layout- and comment-independent view of one module."""
    module = path_to_dotted(path)
    if "error" in raw:
        return {
            "module": module,
            "path": path,
            "doc": "",
            "imports": [],
            "symbols": [],
            "error": str(raw["error"]),
        }
    symbols: list[JsonValue] = []
    for symbol in raw.get("symbols", []):
        entry: dict[str, JsonValue] = {
            "kind": str(symbol["kind"]),
            "name": str(symbol["name"]),
            "signature": str(symbol["signature"]),
            "doc": str(symbol.get("doc", "")),
        }
        if symbol.get("methods"):
            entry["methods"] = [
                {
                    "name": str(m["name"]),
                    "signature": str(m["signature"]),
                    "doc": str(m.get("doc", "")),
                }
                for m in symbol["methods"]
            ]
        symbols.append(entry)
    return {
        "module": module,
        "path": path,
        "doc": str(raw.get("doc", "")).strip(),
        "imports": list(resolve_imports(module, list(raw.get("imports", [])), root)),
        "symbols": symbols,
        "error": None,
    }


def module_api(outline: dict[str, Any]) -> dict[str, JsonValue]:
    """Signatures only (no docstrings): the ModuleApi value."""
    symbols: list[JsonValue] = []
    for symbol in outline.get("symbols", []):
        entry: dict[str, JsonValue] = {
            "kind": symbol["kind"],
            "name": symbol["name"],
            "signature": symbol["signature"],
        }
        if symbol.get("methods"):
            entry["methods"] = [m["signature"] for m in symbol["methods"]]
        symbols.append(entry)
    return {"module": outline["module"], "symbols": symbols}


def _signatures(api: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for symbol in api.get("symbols", []):
        text = str(symbol["signature"])
        if symbol.get("methods"):
            text += " { " + "; ".join(str(m) for m in symbol["methods"]) + " }"
        result[str(symbol["name"])] = text
    return result


def api_diff(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, JsonValue]:
    """Added, removed and changed public symbols between two API snapshots."""
    added: list[JsonValue] = []
    removed: list[JsonValue] = []
    changed: list[JsonValue] = []
    old_modules = baseline.get("modules", {})
    new_modules = current.get("modules", {})
    for module in sorted(set(old_modules) | set(new_modules)):
        old = _signatures(old_modules.get(module, {}))
        new = _signatures(new_modules.get(module, {}))
        for name in sorted(set(old) | set(new)):
            qualified = f"{module}.{name}"
            if name not in old:
                added.append(f"{qualified}: {new[name]}")
            elif name not in new:
                removed.append(f"{qualified}: {old[name]}")
            elif old[name] != new[name]:
                changed.append(f"{qualified}: {old[name]} -> {new[name]}")
    return {"added": added, "removed": removed, "changed": changed}


def _as_dict(value: JsonValue) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return value


# --- rules ---------------------------------------------------------------------------------------


async def _root(ctx: BuildContext) -> str:
    root = await ctx.config("package.root")
    return root if isinstance(root, str) and root else DEFAULT_ROOT


async def _model(ctx: BuildContext, role: str) -> str:
    model = await ctx.config(f"models.{role}")
    if not isinstance(model, str) or not model:
        raise ValueError(f"iab.toml must set models.{role}")
    return model


@rule(ModuleList)
async def module_list(ctx: BuildContext, key: ModuleList) -> JsonValue:
    root = await _root(ctx)
    paths = await ctx.glob(f"{root}/**/*.py")
    return [p for p in paths if not p.endswith("/__init__.py")]


@rule(ModuleOutline)
async def module_outline(ctx: BuildContext, key: ModuleOutline) -> JsonValue:
    root = await _root(ctx)
    raw = await ctx.tool("ast_outline", {"path": key.path})
    return normalize_outline(key.path, _as_dict(raw), root)


@rule(ModuleDoc)
async def module_doc(ctx: BuildContext, key: ModuleDoc) -> JsonValue:
    outline = _as_dict(await ctx.need(ModuleOutline(path=key.path)))
    doc = str(outline["doc"]).split("\n\n")[0].strip()
    return {"module": outline["module"], "doc": doc}


@rule(ModuleApi)
async def module_api_rule(ctx: BuildContext, key: ModuleApi) -> JsonValue:
    return module_api(_as_dict(await ctx.need(ModuleOutline(path=key.path))))


async def _modules(ctx: BuildContext) -> list[str]:
    value = await ctx.need(ModuleList())
    return [str(p) for p in value] if isinstance(value, list) else []


def summary_variables(outline: dict[str, Any], imported: list[JsonValue]) -> dict[str, JsonValue]:
    """Variables of the module_summary prompt (shared with the naive baseline)."""
    symbols: list[JsonValue] = [
        {"name": s["name"], "signature": s["signature"], "doc": s["doc"]}
        for s in outline.get("symbols", [])
    ]
    return {
        "module": outline["module"],
        "outline": {"doc": outline["doc"], "symbols": symbols},
        "imported": imported,
    }


@rule(ModuleSummary)
async def module_summary(ctx: BuildContext, key: ModuleSummary) -> JsonValue:
    outline = _as_dict(await ctx.need(ModuleOutline(path=key.path)))
    known = {path_to_dotted(p): p for p in await _modules(ctx)}
    imported_paths = [known[name] for name in outline["imports"] if name in known]
    docs = await ctx.need_all([ModuleDoc(path=p) for p in imported_paths])
    model = await _model(ctx, "summary")
    result = await ctx.llm(
        "module_summary",
        summary_variables(outline, list(docs)),
        model=model,
        schema=ModuleSummaryOut,
        max_tokens=800,
    )
    return {
        "module": outline["module"],
        "path": key.path,
        "summary": result.summary,
        "key_points": list(result.key_points),
    }


@rule(PackageSummary)
async def package_summary(ctx: BuildContext, key: PackageSummary) -> JsonValue:
    paths = await _modules(ctx)
    summaries = await ctx.need_all([ModuleSummary(path=p) for p in paths])
    name = await ctx.config("package.name")
    modules: list[JsonValue] = [
        {"module": s["module"], "summary": s["summary"]} for s in map(_as_dict, summaries)
    ]
    model = await _model(ctx, "package")
    result = await ctx.llm(
        "package_summary",
        {"package": str(name or "package"), "modules": modules},
        model=model,
        schema=PackageSummaryOut,
        max_tokens=1500,
    )
    return {"overview": result.overview, "highlights": list(result.highlights)}


@rule(Glossary)
async def glossary(ctx: BuildContext, key: Glossary) -> JsonValue:
    paths = await _modules(ctx)
    outlines = await ctx.need_all([ModuleOutline(path=p) for p in paths])
    terms: list[tuple[str, str, str]] = []
    for outline in map(_as_dict, outlines):
        for symbol in outline["symbols"]:
            if symbol["doc"]:
                terms.append((str(symbol["name"]), str(outline["module"]), str(symbol["doc"])))
    terms.sort()
    return {"terms": [{"term": t, "module": m, "definition": d} for t, m, d in terms]}


@rule(ApiSnapshot)
async def api_snapshot(ctx: BuildContext, key: ApiSnapshot) -> JsonValue:
    paths = await _modules(ctx)
    apis = await ctx.need_all([ModuleApi(path=p) for p in paths])
    modules: dict[str, JsonValue] = {}
    for api in map(_as_dict, apis):
        modules[str(api["module"])] = {"symbols": api["symbols"]}
    return {"modules": modules}


@rule(ApiDiff)
async def api_diff_rule(ctx: BuildContext, key: ApiDiff) -> JsonValue:
    import json  # noqa: PLC0415

    current = _as_dict(await ctx.need(ApiSnapshot()))
    text = await ctx.read(BASELINE_API_PATH)
    baseline = json.loads(text) if text else {"modules": {}}
    return api_diff(baseline if isinstance(baseline, dict) else {"modules": {}}, current)


@rule(ReleaseNote)
async def release_note(ctx: BuildContext, key: ReleaseNote) -> JsonValue:
    diff = _as_dict(await ctx.need(ApiDiff()))
    if not any(diff[kind] for kind in ("added", "removed", "changed")):
        return {"title": "No API changes since the last release", "bullets": [], "generated": False}
    model = await _model(ctx, "release")
    result = await ctx.llm(
        "release_note", {"changes": diff}, model=model, schema=ReleaseNoteOut, max_tokens=1200
    )
    return {"title": result.title, "bullets": list(result.bullets), "generated": True}


def _render_module_page(summary: dict[str, Any]) -> str:
    points = "\n".join(f"- {p}" for p in summary["key_points"])
    return f"# `{summary['module']}`\n\n{summary['summary']}\n\n{points}\n"


@rule(DocsSite)
async def docs_site(ctx: BuildContext, key: DocsSite) -> JsonValue:
    package = _as_dict(await ctx.need(PackageSummary()))
    terms = _as_dict(await ctx.need(Glossary()))
    note = _as_dict(await ctx.need(ReleaseNote()))
    paths = await _modules(ctx)
    summaries = [_as_dict(s) for s in await ctx.need_all([ModuleSummary(path=p) for p in paths])]
    pages: dict[str, JsonValue] = {}
    links = "\n".join(f"- [`{s['module']}`](modules/{s['module']}.md)" for s in summaries)
    highlights = "\n".join(f"- {h}" for h in package["highlights"])
    pages["index.md"] = (
        f"# Overview\n\n{package['overview']}\n\n{highlights}\n\n## Modules\n\n{links}\n"
    )
    glossary_lines = "\n".join(
        f"- **{t['term']}** (`{t['module']}`): {t['definition']}" for t in terms["terms"]
    )
    pages["glossary.md"] = f"# Glossary\n\n{glossary_lines}\n"
    bullets = "\n".join(f"- {b}" for b in note["bullets"])
    pages["release-notes.md"] = f"# {note['title']}\n\n{bullets}\n"
    for summary in summaries:
        pages[f"modules/{summary['module']}.md"] = _render_module_page(summary)
    return {"pages": pages}


WORKLOAD_RULES: tuple[Rule[Any], ...] = (
    module_list,
    module_outline,
    module_doc,
    module_api_rule,
    module_summary,
    package_summary,
    glossary,
    api_snapshot,
    api_diff_rule,
    release_note,
    docs_site,
)


def workload_rules() -> RuleSet:
    return RuleSet(WORKLOAD_RULES)
