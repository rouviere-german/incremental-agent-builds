"""LLM steps: prompt rendering, fingerprints and the fixed system prompt.

A rule calls ``ctx.llm(template_id, vars, model=..., schema=...)``. The template is a workspace
file (``templates/<id>.txt``), so editing it is an ordinary file change. Placeholders are written
``{{name}}``; every variable is rendered as an ``<untrusted-data>`` envelope (see
``agentic/envelope.py``), so source text reaches the model only as quoted data and cannot close
its container. Optional PII/secret redaction runs on variable content before rendering.

The step fingerprint covers everything that determines the request: template id and SHA-256,
model id, ``max_tokens``, the system prompt hash, the output schema hash, the redaction flag and
the value hash of every variable. Two calls with equal fingerprints send byte-identical requests,
which is what makes reusing a recorded output sound ("LLM outputs are values", ADR 0003).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel, JsonValue

from incremental_agent_builds.agentic.envelope import UNTRUSTED_DATA_POLICY, UntrustedData
from incremental_agent_builds.agentic.providers.base import ChatMessage, LLMRequest
from incremental_agent_builds.core.canonical import canonical_json, sha256_hex
from incremental_agent_builds.core.redaction import redact_text
from incremental_agent_builds.engine.errors import LLMStepError
from incremental_agent_builds.engine.hashing import value_hash

__all__ = [
    "SYSTEM_PROMPT",
    "TEMPLATE_ID",
    "LLMStepSpec",
    "build_request",
    "render_prompt",
    "schema_json",
    "template_path",
]

TEMPLATE_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PLACEHOLDER = re.compile(r"\{\{\s*([a-z][a-z0-9_]{0,63})\s*\}\}")

SYSTEM_PROMPT = (
    "You produce derived documentation artifacts for a software package. "
    + UNTRUSTED_DATA_POLICY
    + " Respond only with a JSON object that matches the requested schema. You have no tools."
)


def template_path(template_id: str) -> str:
    if not TEMPLATE_ID.fullmatch(template_id):
        raise LLMStepError(f"invalid template id {template_id!r}")
    return f"templates/{template_id}.txt"


def schema_json(schema: type[BaseModel]) -> dict[str, JsonValue]:
    if schema.model_config.get("extra") != "forbid":
        raise LLMStepError(f"output schema {schema.__name__} must set extra='forbid'")
    result: dict[str, JsonValue] = schema.model_json_schema()
    return result


def _var_text(value: JsonValue) -> str:
    return value if isinstance(value, str) else canonical_json(value)


def render_prompt(template: str, variables: Mapping[str, JsonValue], *, redact: bool) -> str:
    """Substitute ``{{name}}`` placeholders with untrusted-data envelopes.

    Every placeholder must have a variable and every variable must be used, so a template edit
    that drops an input is caught instead of silently sending a different prompt.
    """
    names = set(_PLACEHOLDER.findall(template))
    missing = names - variables.keys()
    unused = variables.keys() - names
    if missing or unused:
        raise LLMStepError(
            f"template/variable mismatch (missing: {sorted(missing)}, unused: {sorted(unused)})"
        )

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        text = _var_text(variables[name])
        if redact:
            text = redact_text(text)
        return UntrustedData(source=f"var:{name}", content=text).render()

    return _PLACEHOLDER.sub(substitute, template)


@dataclass(frozen=True, slots=True)
class LLMStepSpec:
    template_id: str
    template_sha256: str
    model: str
    max_tokens: int
    schema_sha256: str
    redact: bool
    var_hashes: tuple[tuple[str, str], ...]

    @classmethod
    def create(
        cls,
        *,
        template_id: str,
        template_text: str,
        model: str,
        max_tokens: int,
        schema: Mapping[str, JsonValue],
        redact: bool,
        variables: Mapping[str, JsonValue],
    ) -> LLMStepSpec:
        return cls(
            template_id=template_id,
            template_sha256=sha256_hex(template_text),
            model=model,
            max_tokens=max_tokens,
            schema_sha256=value_hash(dict(schema)),
            redact=redact,
            var_hashes=tuple(sorted((name, value_hash(v)) for name, v in variables.items())),
        )

    def components(self) -> dict[str, JsonValue]:
        return {
            "v": 1,
            "template_id": self.template_id,
            "template_sha256": self.template_sha256,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system_sha256": sha256_hex(SYSTEM_PROMPT),
            "schema_sha256": self.schema_sha256,
            "redact": self.redact,
            "vars": dict(self.var_hashes),
        }

    @property
    def fingerprint(self) -> str:
        return sha256_hex(canonical_json(self.components()))


def build_request(
    *,
    template_text: str,
    variables: Mapping[str, JsonValue],
    model: str,
    max_tokens: int,
    output_schema: Mapping[str, JsonValue],
    redact: bool,
) -> LLMRequest:
    """The exact provider request of an LLM step. No tools are ever offered to the model."""
    return LLMRequest(
        system=SYSTEM_PROMPT,
        messages=(ChatMessage.user(render_prompt(template_text, variables, redact=redact)),),
        output_schema=dict(output_schema),
        max_tokens=max_tokens,
        model=model,
        role="build",
    )
