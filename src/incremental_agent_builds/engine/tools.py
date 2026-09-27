"""Read-only build tools, called by rules (never chosen by a model).

A build tool is a deterministic function of its validated arguments and of the workspace files it
declares as inputs. BuildContext reads those inputs through the workspace jail (recording them as
file dependencies), fingerprints ``(tool, version, args, input hashes)`` and memoizes the result
by that fingerprint, exactly like an LLM step.

A *volatile* tool (``ttl_seconds`` set) also depends on something outside the workspace, such as a
remote index. Its recorded result is trusted for ``ttl_seconds``; after that, verifying a trace
re-runs the tool and compares the result hash, so an unchanged answer still cuts off.

The registry grants only :attr:`Capability.READ_DATA`: registering a tool that declares any
side-effect capability fails, which keeps build steps free of side effects (effects are separate,
approval-gated sinks; see ``effects.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, JsonValue, ValidationError

from incremental_agent_builds.agentic.errors import ToolArgumentError, ToolNotAllowedError
from incremental_agent_builds.agentic.providers.base import TOOL_NAME_PATTERN
from incremental_agent_builds.agentic.tools import Capability
from incremental_agent_builds.core.canonical import canonical_json, sha256_hex

__all__ = ["BuildTool", "BuildToolRegistry", "ToolHandler", "tool_fingerprint"]

type ToolHandler[A: BaseModel] = Callable[[A, Mapping[str, bytes | None]], JsonValue]

_READ_ONLY = frozenset({Capability.READ_DATA})


def tool_fingerprint(
    name: str, version: int, args: Mapping[str, JsonValue], input_hashes: Mapping[str, str]
) -> str:
    return sha256_hex(
        canonical_json(
            {
                "v": 1,
                "tool": name,
                "version": version,
                "args": dict(args),
                "inputs": dict(input_hashes),
            }
        )
    )


@dataclass(frozen=True, slots=True)
class BuildTool[A: BaseModel]:
    name: str
    description: str
    args_model: type[A]
    handler: ToolHandler[A]
    inputs: Callable[[A], Sequence[str]] = field(default=lambda args: ())
    version: int = 1
    capabilities: frozenset[Capability] = _READ_ONLY
    ttl_seconds: float | None = None

    def __post_init__(self) -> None:
        import re  # noqa: PLC0415 - tiny, only at definition time

        if not re.fullmatch(TOOL_NAME_PATTERN, self.name):
            raise ValueError(f"invalid tool name {self.name!r}")
        if self.args_model.model_config.get("extra") != "forbid":
            raise ValueError(f"tool {self.name} args model must set extra='forbid'")
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

    @property
    def volatile(self) -> bool:
        return self.ttl_seconds is not None

    def validate(self, raw: Mapping[str, Any]) -> tuple[A, dict[str, JsonValue]]:
        """Strictly validate arguments; return the model and its canonical JSON form."""
        try:
            args = self.args_model.model_validate(dict(raw), strict=True)
        except ValidationError as exc:
            locations = sorted({".".join(str(p) for p in e["loc"]) or "$" for e in exc.errors()})
            raise ToolArgumentError(
                f"invalid arguments for {self.name} at: {', '.join(locations)}"
            ) from exc
        canonical: dict[str, JsonValue] = args.model_dump(mode="json")
        return args, canonical


class BuildToolRegistry:
    """Tools available to rules. Only read-only tools can be registered."""

    def __init__(self, tools: Iterable[BuildTool[Any]] = ()) -> None:
        self._tools: dict[str, BuildTool[Any]] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: BuildTool[Any]) -> None:
        extra = tool.capabilities - _READ_ONLY
        if extra or not tool.capabilities:
            names = ", ".join(sorted(c.value for c in extra)) or "none"
            raise ToolNotAllowedError(
                f"build tool {tool.name} must be read-only (extra capabilities: {names})"
            )
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name} is already registered")
        self._tools[tool.name] = tool

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def get(self, name: str) -> BuildTool[Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotAllowedError(f"build tool {name} is not registered")
        return tool
