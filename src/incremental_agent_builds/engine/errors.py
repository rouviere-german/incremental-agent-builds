"""Exception hierarchy of the incremental build engine."""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "BuildAbortedError",
    "BuildEngineError",
    "CacheImportError",
    "CycleError",
    "EffectError",
    "HermeticityError",
    "IntegrityError",
    "LLMStepError",
    "RuleDefinitionError",
    "RuleFailedError",
    "ToolStepError",
    "WorkspaceError",
]


class BuildEngineError(Exception):
    """Base class for every error raised by the build engine."""


class RuleDefinitionError(BuildEngineError):
    """A rule, key type or rule set is malformed (duplicate rule, unknown key type, bad value)."""


class CycleError(BuildEngineError):
    """A rule needs a key that is (transitively) waiting for the rule itself.

    ``cycle`` lists key ids in dependency order and starts and ends with the same key.
    """

    def __init__(self, cycle: Sequence[str]) -> None:
        self.cycle = tuple(cycle)
        super().__init__("dependency cycle: " + " -> ".join(self.cycle))


class RuleFailedError(BuildEngineError):
    """A rule raised. Waiters of the key (and of keys that need it) get this error."""

    def __init__(self, key_id: str, cause: BaseException) -> None:
        self.key_id = key_id
        self.cause = cause
        super().__init__(f"rule for {key_id} failed: {type(cause).__name__}: {cause}")


class BuildAbortedError(BuildEngineError):
    """The whole build stopped (budget exhausted, cancelled). Completed traces are kept."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"build aborted: {reason}")


class WorkspaceError(BuildEngineError):
    """A path escaped the workspace jail, was malformed, or a file could not be read."""


class IntegrityError(BuildEngineError):
    """Stored bytes or records did not verify (CAS hash mismatch, bad trace MAC)."""


class HermeticityError(BuildEngineError):
    """A rule body performed I/O outside BuildContext (strict hermeticity mode)."""


class LLMStepError(BuildEngineError):
    """An LLM step failed: provider error, refusal, or output that failed schema validation."""


class ToolStepError(BuildEngineError):
    """A build tool failed, timed out, exceeded its output cap or was not allowed."""


class CacheImportError(BuildEngineError):
    """A shared-cache bundle could not be parsed at all (individual bad entries are rejected)."""


class EffectError(BuildEngineError):
    """An effect sink refused to run (failed build, missing or mismatched approval)."""
