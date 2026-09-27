"""Rules: async functions that compute the value of one key type through a BuildContext.

```python
class ModuleOutline(BuildKey):
    path: str

@rule(ModuleOutline, version=1)
async def module_outline(ctx: BuildContext, key: ModuleOutline) -> JsonValue:
    text = await ctx.read(key.path)
    ...
```

The ``version`` is part of every trace. Bump it when the rule's code changes meaning: traces
recorded by another version never verify, so the rule re-runs. Rule code itself is not hashed (see
docs/SEMANTICS.md, "What a trace covers").
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import JsonValue, ValidationError

from incremental_agent_builds.engine.errors import RuleDefinitionError
from incremental_agent_builds.engine.keys import BuildKey, split_key_id

if TYPE_CHECKING:
    from incremental_agent_builds.engine.context import BuildContext

__all__ = ["Rule", "RuleFn", "RuleSet", "rule"]

type RuleFn[K: BuildKey] = Callable[["BuildContext", K], Awaitable[JsonValue]]

# A bound on the key cache, so a long-lived process building millions of distinct keys cannot grow
# it without limit. Past the bound, ids are simply parsed each time.
_KEY_CACHE_LIMIT = 200_000


@dataclass(frozen=True, slots=True)
class Rule[K: BuildKey]:
    key_type: type[K]
    fn: RuleFn[K]
    version: int

    @property
    def name(self) -> str:
        return self.key_type.kind


def rule[K: BuildKey](key_type: type[K], *, version: int = 1) -> Callable[[RuleFn[K]], Rule[K]]:
    """Decorator that turns ``async def f(ctx, key) -> JsonValue`` into a :class:`Rule`."""
    if not (isinstance(key_type, type) and issubclass(key_type, BuildKey)):
        raise RuleDefinitionError("rule() needs a BuildKey subclass")
    if version < 0:
        raise RuleDefinitionError("rule version must be >= 0")

    def decorate(fn: RuleFn[K]) -> Rule[K]:
        if not inspect.iscoroutinefunction(fn):
            raise RuleDefinitionError(f"rule for {key_type.kind} must be an async function")
        return Rule(key_type=key_type, fn=fn, version=version)

    return decorate


class RuleSet:
    """The rules of one build. Each key kind has exactly one rule."""

    def __init__(self, rules: Iterable[Rule[Any]]) -> None:
        self._rules: dict[str, Rule[Any]] = {}
        for item in rules:
            if item.name in self._rules:
                raise RuleDefinitionError(f"duplicate rule for key kind {item.name!r}")
            self._rules[item.name] = item
        # Verifying a trace turns every recorded rule dependency back into a key object, so the same
        # ids are parsed over and over within one build and across builds. The rule set is immutable
        # once constructed, so the result only depends on the id.
        self._key_cache: dict[str, BuildKey | None] = {}

    def __contains__(self, kind: object) -> bool:
        return kind in self._rules

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._rules))

    def rule_for(self, key: BuildKey) -> Rule[Any]:
        found = self._rules.get(key.kind)
        if found is None or not isinstance(key, found.key_type):
            raise RuleDefinitionError(f"no rule for key {key.key_id}")
        return found

    def get(self, kind: str) -> Rule[Any] | None:
        return self._rules.get(kind)

    def key_from_id(self, key_id: str) -> BuildKey | None:
        """Rebuild a key from its id; None if its rule no longer exists or the id is malformed.

        Results are memoized per rule set (see ``_key_cache``). Keys are frozen, so handing the same
        object to several callers is safe.
        """
        if key_id in self._key_cache:
            return self._key_cache[key_id]
        key = self._parse_key_id(key_id)
        if len(self._key_cache) < _KEY_CACHE_LIMIT:
            self._key_cache[key_id] = key
        return key

    def _parse_key_id(self, key_id: str) -> BuildKey | None:
        try:
            kind, args = split_key_id(key_id)
        except ValueError:
            return None
        found = self._rules.get(kind)
        if found is None:
            return None
        try:
            key = found.key_type.model_validate(args)
        except ValidationError:
            return None
        return key if key.key_id == key_id else None
