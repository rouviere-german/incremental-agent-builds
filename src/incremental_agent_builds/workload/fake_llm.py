"""Deterministic stand-in for the model in tests, the demo and benchmarks.

The response is a pure function of the request hash, the way a cached model answer would be: the
same request always yields the same output, and any change to the request (template text, model
id, a variable) yields a different output. Outputs are plausible documentation text built from
the quoted variables, tagged with a prefix of the request hash, and conform to the requested
schema. Token usage is estimated by :class:`FakeProvider` (4 characters per token).

``gullible=True`` mimics a model that follows instructions injected into source files: if quoted
data contains ``INJECT-JSON:{...}``, that object is returned verbatim. Tests use it to show that
schema validation and the absence of tools, not the model's judgement, contain the injection.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from pydantic import JsonValue

from incremental_agent_builds.agentic.envelope import parse_envelopes
from incremental_agent_builds.agentic.errors import ProviderError
from incremental_agent_builds.agentic.providers.base import LLMRequest, LLMResponse, TextBlock
from incremental_agent_builds.agentic.providers.fake import FakeProvider, PolicyResult

__all__ = ["DocsResponder", "make_fake_provider"]

_INJECT = re.compile(r"INJECT-JSON:(\{.*?\})\s*$", re.MULTILINE)


def _variables(request: LLMRequest) -> dict[str, Any]:
    text = "".join(
        block.text for block in request.messages[-1].content if isinstance(block, TextBlock)
    )
    found: dict[str, Any] = {}
    for item in parse_envelopes(text):
        name = item.source.removeprefix("var:")
        try:
            found[name] = json.loads(item.content)
        except json.JSONDecodeError:
            found[name] = item.content
    return found


def _names(items: list[Any], limit: int = 3) -> str:
    names = [str(item.get("name", "")) for item in items if isinstance(item, dict)][:limit]
    return ", ".join(names) if names else "no public symbols"


class DocsResponder:
    """FakeProvider policy for the three documentation prompts."""

    def __init__(self, *, gullible: bool = False) -> None:
        self.gullible = gullible

    def __call__(self, request: LLMRequest, rng: random.Random, turn: int) -> PolicyResult:
        del rng, turn
        digest = request.sha256()
        tag = digest[:8]
        variables = _variables(request)
        title = str((request.output_schema or {}).get("title", ""))
        if self.gullible:
            for value in variables.values():
                match = _INJECT.search(value if isinstance(value, str) else json.dumps(value))
                if match:
                    return self._respond(request, json.loads(match.group(1)))
        if title == "ModuleSummaryOut":
            payload = self._module(variables, tag)
        elif title == "PackageSummaryOut":
            payload = self._package(variables, tag)
        elif title == "ReleaseNoteOut":
            payload = self._release(variables, tag)
        else:
            return ProviderError(f"fake responder has no script for schema {title!r}")
        return self._respond(request, payload)

    @staticmethod
    def _respond(request: LLMRequest, payload: dict[str, JsonValue]) -> LLMResponse:
        return LLMResponse(
            model=request.model or "claude-opus-5",
            stop_reason="end_turn",
            text=json.dumps(payload, sort_keys=True),
            parsed=payload,
        )

    @staticmethod
    def _module(variables: dict[str, Any], tag: str) -> dict[str, JsonValue]:
        module = str(variables.get("module", "module"))
        outline = variables.get("outline") or {}
        symbols = outline.get("symbols", []) if isinstance(outline, dict) else []
        imported = variables.get("imported") or []
        doc = (
            str(outline.get("doc", "")).split(". ")[0].rstrip(".")
            if isinstance(outline, dict)
            else ""
        )
        uses = ", ".join(str(i.get("module")) for i in imported if isinstance(i, dict))
        summary = (
            f"{module} provides {len(symbols)} public symbols ({_names(symbols)}). {doc}."
            + (f" It builds on {uses}." if uses else "")
            + f" [{tag}]"
        )
        points: list[JsonValue] = [
            f"{s.get('name')}: {s.get('doc') or 'undocumented'}"
            for s in symbols[:4]
            if isinstance(s, dict)
        ]
        return {"summary": summary, "key_points": points}

    @staticmethod
    def _package(variables: dict[str, Any], tag: str) -> dict[str, JsonValue]:
        modules = variables.get("modules") or []
        layers = sorted(
            {
                str(m.get("module", "")).split(".")[1]
                for m in modules
                if isinstance(m, dict) and "." in str(m.get("module", ""))
            }
        )
        overview = (
            f"{variables.get('package', 'The package')} has {len(modules)} modules across "
            f"{len(layers)} layers ({', '.join(layers)}). [{tag}]"
        )
        highlights: list[JsonValue] = [
            f"{m.get('module')}: {str(m.get('summary', ''))[:80]}"
            for m in modules[:5]
            if isinstance(m, dict)
        ]
        return {"overview": overview, "highlights": highlights}

    @staticmethod
    def _release(variables: dict[str, Any], tag: str) -> dict[str, JsonValue]:
        changes = variables.get("changes") or {}
        added = list(changes.get("added", [])) if isinstance(changes, dict) else []
        removed = list(changes.get("removed", [])) if isinstance(changes, dict) else []
        changed = list(changes.get("changed", [])) if isinstance(changes, dict) else []
        title = f"{len(removed) + len(changed)} breaking, {len(added)} added [{tag}]"
        bullets: list[JsonValue] = [f"Breaking: {c}" for c in (removed + changed)[:10]]
        bullets += [f"New: {a}" for a in added][: 20 - len(bullets)]
        return {"title": title, "bullets": bullets}


def make_fake_provider(*, gullible: bool = False) -> FakeProvider:
    return FakeProvider(DocsResponder(gullible=gullible))
