"""BuildContext: the only door through which a rule reads anything.

Each method records what it read and the hash of what it returned, in call order, as one
dependency group. Those groups become the rule's verifying trace. ``need_all`` records its keys as
a single group, so verification can bring them up to date in parallel.

A call that *raises* is recorded too, as a dependency with the sentinel hash
:data:`~.hashing.ERROR_HASH`. A rule is expected to let the exception propagate, but one that
catches it and returns a degraded value would otherwise leave a trace that never mentions the
failed call: the next build would verify it, find every recorded input unchanged and serve the
degraded value with the provider or tool healthy again. The sentinel can never equal a real hash,
so the key re-runs until the call succeeds.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, JsonValue, ValidationError

from incremental_agent_builds.agentic.errors import (
    BudgetExceededError,
    ProviderError,
    ToolArgumentError,
    ToolNotAllowedError,
)
from incremental_agent_builds.agentic.pricing import cost_usd_micros
from incremental_agent_builds.agentic.providers.base import acomplete
from incremental_agent_builds.core.canonical import CanonicalJSONError, canonical_json
from incremental_agent_builds.engine.errors import (
    IntegrityError,
    LLMStepError,
    RuleDefinitionError,
    ToolStepError,
    WorkspaceError,
)
from incremental_agent_builds.engine.hashing import ERROR_HASH
from incremental_agent_builds.engine.hermetic import io_permit
from incremental_agent_builds.engine.keys import BuildKey
from incremental_agent_builds.engine.llm import (
    LLMStepSpec,
    build_request,
    schema_json,
    template_path,
)
from incremental_agent_builds.engine.store import Dep, DepGroups, DepKind, StepRecord, TraceUsage
from incremental_agent_builds.engine.tools import tool_fingerprint
from incremental_agent_builds.engine.workspace import validate_relative_path

if TYPE_CHECKING:
    from incremental_agent_builds.engine.session import BuildSession, _Slot

__all__ = ["BuildContext"]

_CONFIG_KEY = r"^[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*$"


def _validate_config_key(dotted_key: str) -> None:
    import re  # noqa: PLC0415 - deferred: only ctx.config needs the regex engine

    if not re.fullmatch(_CONFIG_KEY, dotted_key):
        raise WorkspaceError(f"invalid config key {dotted_key!r}")


class BuildContext:
    def __init__(
        self, session: BuildSession, key: BuildKey, slot: _Slot, *, recording: bool = True
    ) -> None:
        self._session = session
        self.key = key
        self._slot = slot
        self._recording = recording
        self._groups: list[tuple[Dep, ...]] = []
        self._llm_steps = 0
        self._provider_calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._usd_micros = 0
        self._closed = False

    # --- bookkeeping --------------------------------------------------------------------------

    def close(self) -> None:
        self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise RuleDefinitionError("BuildContext used after its rule returned")

    def _record(self, *deps: Dep) -> None:
        if self._recording:
            self._groups.append(tuple(deps))

    def _record_failure(self, kind: str, ref: str) -> None:
        """Record that a context call raised, so the key can never be reused on this trace."""
        self._record(Dep(DepKind.ERROR, f"{kind}:{ref}", ERROR_HASH))
        if self._recording:
            self._session.audit("context_call_failed", {"key": self.key.key_id, "call": kind})

    def dep_groups(self) -> DepGroups:
        return tuple(self._groups)

    def usage(self) -> TraceUsage:
        return TraceUsage(
            llm_steps=self._llm_steps,
            provider_calls=self._provider_calls,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            usd_micros=self._usd_micros,
        )

    @asynccontextmanager
    async def _suspended(self) -> AsyncIterator[None]:
        self._slot.release()
        try:
            yield
        finally:
            await self._slot.acquire()

    # --- other keys ---------------------------------------------------------------------------

    async def need(self, key: BuildKey) -> JsonValue:
        """The up-to-date value of another key (records a rule dependency)."""
        return (await self.need_all([key]))[0]

    async def need_all(self, keys: Sequence[BuildKey]) -> list[JsonValue]:
        """Values of several keys, built concurrently and recorded as one dependency group."""
        self._check_open()
        if not keys:
            return []
        unique = list({key.key_id: key for key in keys}.values())
        session = self._session
        try:
            async with self._suspended():
                results = await session.ensure_many(unique, self.key.key_id)
                values = {
                    key.key_id: await session.value_of(key, result.hash)
                    for key, result in zip(unique, results, strict=True)
                }
        except BaseException:
            self._record_failure("need", unique[0].key_id)
            raise
        self._record(*(Dep(DepKind.RULE, result.key_id, result.hash) for result in results))
        return [values[key.key_id] for key in keys]

    # --- workspace inputs ---------------------------------------------------------------------

    async def read_bytes(self, path: str) -> bytes | None:
        """File content (None if missing). Jailed to the workspace and fingerprinted."""
        self._check_open()
        try:
            validate_relative_path(path)
            with io_permit():
                data, digest = self._session.file_input(path)
        except BaseException:
            self._record_failure("read", path)
            raise
        self._record(Dep(DepKind.FILE, path, digest))
        return data

    async def read(self, path: str) -> str | None:
        """UTF-8 text of a workspace file (None if missing)."""
        data = await self.read_bytes(path)
        if data is None:
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceError(f"{path} is not valid UTF-8") from exc

    async def glob(self, pattern: str) -> list[str]:
        """Sorted workspace paths matching ``pattern``; adding or removing a match invalidates."""
        self._check_open()
        try:
            with io_permit():
                paths, digest = self._session.glob_input(pattern)
        except BaseException:
            self._record_failure("glob", pattern)
            raise
        self._record(Dep(DepKind.GLOB, pattern, digest))
        return list(paths)

    async def config(self, dotted_key: str) -> JsonValue | None:
        """One value from the workspace config file (``iab.toml``), tracked by key, not file."""
        self._check_open()
        try:
            _validate_config_key(dotted_key)
            with io_permit():
                value, digest = self._session.config_input(dotted_key)
        except BaseException:
            self._record_failure("config", dotted_key)
            raise
        self._record(
            Dep(DepKind.CONFIG, f"{self._session.config.config_path}#{dotted_key}", digest)
        )
        return value

    # --- tools --------------------------------------------------------------------------------

    async def tool(self, name: str, args: Mapping[str, JsonValue]) -> JsonValue:
        """Run a registered read-only tool (memoized by tool, version, args and input hashes)."""
        self._check_open()
        try:
            return await self._tool_call(name, args)
        except BaseException:
            self._record_failure("tool", name)
            raise

    async def _tool_call(self, name: str, args: Mapping[str, JsonValue]) -> JsonValue:
        session = self._session
        try:
            tool = session.engine.tools.get(name)
            parsed, canonical = tool.validate(args)
        except (ToolNotAllowedError, ToolArgumentError) as exc:
            raise ToolStepError(str(exc)) from exc
        inputs: dict[str, bytes | None] = {}
        input_hashes: dict[str, str] = {}
        for path in tool.inputs(parsed):
            inputs[path] = await self.read_bytes(path)
            input_hashes[path] = session.file_input(path)[1]
        fingerprint = tool_fingerprint(tool.name, tool.version, canonical, input_hashes)
        step = session.step(fingerprint)
        value: JsonValue = None
        hit = False
        if (
            step is not None
            and step.kind == "tool"
            and session.config.step_memo
            and self.key.key_id not in session.force
            and (tool.ttl_seconds is None or session.now - step.recorded_at <= tool.ttl_seconds)
        ):
            try:
                with io_permit():
                    value = session.engine.cas.get_json(step.result_hash)
                hit = True
                session.stats.tool_memo_hits += 1
            except IntegrityError as exc:
                session.integrity("cas", fingerprint, str(exc))
        if not hit or step is None:
            try:
                with io_permit():
                    value = await asyncio.to_thread(tool.handler, parsed, inputs)
                digest = session.engine.cas.put_json(value)
            except ToolStepError:
                raise
            except CanonicalJSONError as exc:
                raise ToolStepError(f"tool {name} returned a non-JSON value") from exc
            except Exception as exc:
                raise ToolStepError(f"tool {name} failed: {type(exc).__name__}") from exc
            session.stats.tool_runs += 1
            meta: dict[str, JsonValue] = {
                "tool": tool.name,
                "version": tool.version,
                "args": canonical,
                "inputs": dict(input_hashes),
            }
            if tool.ttl_seconds is not None:
                meta["ttl_seconds"] = tool.ttl_seconds
            step = StepRecord(fingerprint, "tool", digest, meta, session.engine.clock())
            session.record_step(step)
        self._record(Dep(DepKind.TOOL, fingerprint, step.result_hash))
        return value

    # --- LLM steps ----------------------------------------------------------------------------

    async def llm[M: BaseModel](
        self,
        template_id: str,
        variables: Mapping[str, JsonValue],
        *,
        model: str,
        schema: type[M],
        max_tokens: int = 1024,
    ) -> M:
        """One LLM step. Variables enter the prompt only as quoted untrusted data; the output
        must validate against ``schema``; no tools are offered to the model."""
        self._check_open()
        try:
            return await self._llm_call(
                template_id, variables, model=model, schema=schema, max_tokens=max_tokens
            )
        except BaseException:
            self._record_failure("llm", template_id)
            raise

    async def _llm_call[M: BaseModel](
        self,
        template_id: str,
        variables: Mapping[str, JsonValue],
        *,
        model: str,
        schema: type[M],
        max_tokens: int,
    ) -> M:
        session = self._session
        template_bytes = await self.read_bytes(template_path(template_id))
        if template_bytes is None:
            raise LLMStepError(f"template {template_id!r} not found")
        try:
            template_text = template_bytes.decode("utf-8")
            canonical_json(dict(variables))
        except (UnicodeDecodeError, CanonicalJSONError) as exc:
            raise LLMStepError(f"template {template_id!r} or its variables are invalid") from exc
        output_schema = schema_json(schema)
        spec = LLMStepSpec.create(
            template_id=template_id,
            template_text=template_text,
            model=model,
            max_tokens=max_tokens,
            schema=output_schema,
            redact=session.config.redact_prompts,
            variables=variables,
        )
        fingerprint = spec.fingerprint
        session.stats.llm_steps += 1
        self._llm_steps += 1
        result = self._memoized_output(fingerprint, schema)
        if result is None:
            result = await self._call_provider(
                spec, template_text, variables, output_schema, schema
            )
        validated, step = result
        self._record(Dep(DepKind.LLM, fingerprint, step.result_hash))
        return validated

    def _step_tokens(self, step: StepRecord) -> tuple[int, int, int]:
        meta = step.meta

        def number(name: str) -> int:
            value = meta.get(name, 0)
            return int(value) if isinstance(value, int | float) else 0

        return number("input_tokens"), number("output_tokens"), number("usd_micros")

    def _memoized_output[M: BaseModel](
        self, fingerprint: str, schema: type[M]
    ) -> tuple[M, StepRecord] | None:
        session = self._session
        if not session.config.step_memo or self.key.key_id in session.force:
            return None
        step = session.step(fingerprint)
        if step is None or step.kind != "llm":
            return None
        try:
            with io_permit():
                stored = session.engine.cas.get_json(step.result_hash)
            validated = schema.model_validate_json(json.dumps(stored))
        except IntegrityError as exc:
            session.integrity("cas", fingerprint, str(exc))
            return None
        except ValidationError:
            session.integrity("llm_step", fingerprint, "stored output fails its schema")
            return None
        input_tokens, output_tokens, usd = self._step_tokens(step)
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._usd_micros += usd
        session.stats.llm_memo_hits += 1
        session.stats.tokens_saved += input_tokens + output_tokens
        return validated, step

    async def _call_provider[M: BaseModel](
        self,
        spec: LLMStepSpec,
        template_text: str,
        variables: Mapping[str, JsonValue],
        output_schema: dict[str, JsonValue],
        schema: type[M],
    ) -> tuple[M, StepRecord]:
        session = self._session
        provider = session.engine.provider
        if provider is None:
            raise LLMStepError("no LLM provider is configured for this build")
        request = build_request(
            template_text=template_text,
            variables=variables,
            model=spec.model,
            max_tokens=spec.max_tokens,
            output_schema=output_schema,
            redact=spec.redact,
        )
        try:
            session.budget.reserve_step()
        except BudgetExceededError as exc:
            raise session.abort(f"budget exhausted before an LLM step: {exc}") from exc
        async with session.llm_slots:
            try:
                with io_permit():
                    response = await acomplete(provider, request)
            except ProviderError as exc:
                if exc.usage is not None:
                    billed = exc.model or spec.model
                    self._charge(billed, exc.usage.input_tokens, exc.usage.output_tokens)
                    self._check_budget(billed, exc.usage.input_tokens, exc.usage.output_tokens)
                raise LLMStepError(f"provider error: {type(exc).__name__}: {exc}") from exc
        session.stats.provider_calls += 1
        self._provider_calls += 1
        usd = self._charge(
            response.model, response.usage.input_tokens, response.usage.output_tokens
        )
        payload: Any = response.parsed
        if payload is None:
            try:
                payload = json.loads(response.text)
            except json.JSONDecodeError as exc:
                raise LLMStepError("model output is not JSON") from exc
        try:
            validated = schema.model_validate_json(json.dumps(payload))
        except ValidationError as exc:
            raise LLMStepError(
                f"model output failed schema {schema.__name__} ({exc.error_count()} errors)"
            ) from exc
        output: JsonValue = validated.model_dump(mode="json")
        digest = session.engine.cas.put_json(output)
        meta: dict[str, JsonValue] = {
            "template_id": spec.template_id,
            "model": spec.model,
            "served_model": response.model,
            "fallback": response.served_by_fallback,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "usd_micros": usd,
            "components": spec.components(),
        }
        step = StepRecord(spec.fingerprint, "llm", digest, meta, session.engine.clock())
        session.record_step(step)
        self._input_tokens += response.usage.input_tokens
        self._output_tokens += response.usage.output_tokens
        self._usd_micros += usd
        self._check_budget(
            response.model, response.usage.input_tokens, response.usage.output_tokens
        )
        return validated, step

    def _charge(self, model: str, input_tokens: int, output_tokens: int) -> int:
        session = self._session
        usd = cost_usd_micros(model, input_tokens, output_tokens)
        session.stats.input_tokens += input_tokens
        session.stats.output_tokens += output_tokens
        session.stats.usd_micros += usd
        return usd

    def _check_budget(self, model: str, input_tokens: int, output_tokens: int) -> None:
        from incremental_agent_builds.agentic.providers.base import Usage  # noqa: PLC0415

        session = self._session
        try:
            session.budget.record_usage(
                model=model, usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens)
            )
        except BudgetExceededError as exc:
            raise session.abort(f"budget exhausted: {exc}") from exc
