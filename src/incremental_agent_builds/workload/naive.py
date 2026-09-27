"""Baselines: the same documentation pipeline with a memoizing cache and no dependency tracking.

Two cache-key policies are measured, because they fail differently:

``input-hash``
    The common ``@cache def summarize(source)`` pattern: an artifact is keyed by the task name and
    its *explicit* input (the module's own source text, the list of module summaries, the API
    diff). Template text, the model id and the docstrings of imported modules are read inside the
    function, so they are hidden dependencies and edits to them are silently missed.
``prompt-hash``
    Keyed by the SHA-256 of the fully rendered prompt. Template and referenced-file edits change the
    prompt and are caught; the model id is not part of the prompt text, so switching models still
    serves the old model's output.

Both send exactly the requests the build engine would send (same rendering, same provider), so a
fresh (non-stale) artifact is byte-identical to the engine's value and staleness is measured by
direct comparison.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, JsonValue

from incremental_agent_builds.agentic.providers.base import LLMProvider
from incremental_agent_builds.core.canonical import canonical_json, sha256_hex
from incremental_agent_builds.engine._ast_worker import outline
from incremental_agent_builds.engine.llm import build_request, schema_json
from incremental_agent_builds.workload.rules import (
    BASELINE_API_PATH,
    ModuleSummaryOut,
    PackageSummaryOut,
    ReleaseNoteOut,
    api_diff,
    module_api,
    normalize_outline,
    path_to_dotted,
    summary_variables,
)

__all__ = ["CacheMode", "NaivePipeline"]

CacheMode = Literal["input-hash", "prompt-hash"]


@dataclass
class NaivePipeline:
    provider: LLMProvider
    mode: CacheMode
    redact: bool = True
    cache: dict[str, dict[str, JsonValue]] = field(default_factory=dict)
    provider_calls: int = 0
    tokens: int = 0

    def _llm(  # noqa: PLR0917 - mirrors the engine's ctx.llm signature on purpose
        self,
        task: str,
        explicit_input: str,
        template: str,
        variables: Mapping[str, JsonValue],
        model: str,
        schema: type[BaseModel],
        max_tokens: int,
    ) -> dict[str, JsonValue]:
        """One cached LLM call. The result is an object because every schema here is one."""
        request = build_request(
            template_text=template,
            variables=variables,
            model=model,
            max_tokens=max_tokens,
            output_schema=schema_json(schema),
            redact=self.redact,
        )
        if self.mode == "input-hash":
            key = sha256_hex(f"{task}\x00{explicit_input}")
        else:
            prompt = "".join(getattr(block, "text", "") for block in request.messages[-1].content)
            key = sha256_hex(f"{task}\x00{request.system}\x00{prompt}")
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        response = self.provider.complete(request)
        self.provider_calls += 1
        self.tokens += response.usage.total_tokens
        payload = response.parsed if response.parsed is not None else json.loads(response.text)
        value: dict[str, JsonValue] = schema.model_validate_json(json.dumps(payload)).model_dump(
            mode="json"
        )
        self.cache[key] = value
        return value

    def run(self, files: Mapping[str, str]) -> dict[str, JsonValue]:
        """Produce every LLM artifact for the given workspace snapshot."""
        config: dict[str, Any] = tomllib.loads(files["iab.toml"])
        root = str(config["package"]["root"])
        models: dict[str, str] = config["models"]
        templates = {
            name: files[f"templates/{name}.txt"]
            for name in ("module_summary", "package_summary", "release_note")
        }
        paths = sorted(
            p
            for p in files
            if p.startswith(f"{root}/") and p.endswith(".py") and not p.endswith("__init__.py")
        )
        outlines = {p: normalize_outline(p, outline(files[p]), root) for p in paths}
        known = {path_to_dotted(p): p for p in paths}
        artifacts: dict[str, JsonValue] = {}
        summaries: list[JsonValue] = []
        for path in paths:
            current = outlines[path]
            imported: list[JsonValue] = []
            for name in current["imports"]:  # type: ignore[union-attr]
                if name in known:
                    other = outlines[known[name]]
                    doc = str(other["doc"]).split("\n\n")[0].strip()
                    imported.append({"module": other["module"], "doc": doc})
            result = self._llm(
                "module_summary",
                files[path],
                templates["module_summary"],
                summary_variables(current, imported),
                models["summary"],
                ModuleSummaryOut,
                800,
            )
            value: JsonValue = {
                "module": current["module"],
                "path": path,
                "summary": result["summary"],
                "key_points": result["key_points"],
            }
            artifacts[f"module:{path}"] = value
            summaries.append({"module": current["module"], "summary": result["summary"]})
        package = self._llm(
            "package_summary",
            canonical_json(summaries),
            templates["package_summary"],
            {"package": str(config["package"].get("name", "package")), "modules": summaries},
            models["package"],
            PackageSummaryOut,
            1500,
        )
        artifacts["package"] = package
        current_api: dict[str, dict[str, Any]] = {"modules": {}}
        for path in paths:
            api = module_api(outlines[path])
            current_api["modules"][str(api["module"])] = {"symbols": api["symbols"]}
        baseline_text = files.get(BASELINE_API_PATH, "")
        baseline = json.loads(baseline_text) if baseline_text else {"modules": {}}
        diff = api_diff(baseline, current_api)
        if not any(diff[kind] for kind in ("added", "removed", "changed")):
            note: JsonValue = {
                "title": "No API changes since the last release",
                "bullets": [],
                "generated": False,
            }
        else:
            generated = self._llm(
                "release_note",
                canonical_json(diff),
                templates["release_note"],
                {"changes": diff},
                models["release"],
                ReleaseNoteOut,
                1200,
            )
            note = {
                "title": generated["title"],
                "bullets": generated["bullets"],
                "generated": True,
            }
        artifacts["release"] = note
        return artifacts
