"""Incremental build engine for LLM and tool pipelines.

Rules are async functions of a :class:`BuildKey`; every input goes through a
:class:`BuildContext`, which records a verifying trace. Rebuilds re-run only rules with a changed
dependency, and early cutoff stops propagation when a re-run produces the same value.
"""

from incremental_agent_builds.engine.context import BuildContext
from incremental_agent_builds.engine.engine import BuildEngine, BuildResult, EngineConfig
from incremental_agent_builds.engine.errors import (
    BuildAbortedError,
    BuildEngineError,
    CycleError,
    HermeticityError,
    IntegrityError,
    LLMStepError,
    RuleFailedError,
    ToolStepError,
    WorkspaceError,
)
from incremental_agent_builds.engine.keys import BuildKey
from incremental_agent_builds.engine.rules import Rule, RuleSet, rule

__all__ = [
    "BuildAbortedError",
    "BuildContext",
    "BuildEngine",
    "BuildEngineError",
    "BuildKey",
    "BuildResult",
    "CycleError",
    "EngineConfig",
    "HermeticityError",
    "IntegrityError",
    "LLMStepError",
    "Rule",
    "RuleFailedError",
    "RuleSet",
    "ToolStepError",
    "WorkspaceError",
    "rule",
]
