"""Signed constructive traces: share build results between stores (machines, CI, teammates).

A verifying trace answers "is my recorded result still valid?". A *constructive* trace answers
"has anyone computed this key for exactly these dependency values?" (Build Systems a la Carte,
section 4). Exported entries map ``(rule, version, key, ordered dependency hashes)`` to a result
hash, together with the LLM/tool step records those traces rely on and the value blobs.

Every entry is Ed25519-signed by the exporting store. On import, an entry is accepted only if:
its signer is on the local allowlist, the signature verifies over the canonical entry, the value
blob it names is in the bundle and hashes to the claimed result, and the blob is canonical JSON.
Accepted trace entries are *candidates*: a build uses one only after checking every recorded
dependency against its own current values, exactly like a verifying trace, and re-verifies the
signature against the allowlist at that moment (so revoking a key also revokes its entries).

Only traces this store computed itself (``origin == "executed"``) are exported, so a store never
re-signs results it merely imported. Volatile tool steps are never exported.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from incremental_agent_builds.core.canonical import canonical_json, sha256_hex
from incremental_agent_builds.engine.cas import ContentStore, decode_json_blob
from incremental_agent_builds.engine.errors import CacheImportError, IntegrityError
from incremental_agent_builds.engine.hashing import MISSING, is_sha256_hex
from incremental_agent_builds.engine.keys import split_key_id
from incremental_agent_builds.engine.metrics import BuildMetrics
from incremental_agent_builds.engine.signing import SigningKey, TrustStore
from incremental_agent_builds.engine.store import (
    ConstructiveEntry,
    DepKind,
    StepRecord,
    Trace,
    TraceStore,
    decode_deps,
    encode_deps,
)

__all__ = [
    "BUNDLE_FORMAT",
    "ImportReport",
    "export_bundle",
    "import_bundle",
]

BUNDLE_FORMAT = "iab-cache-bundle"
MAX_BUNDLE_BYTES = 64 * 1024 * 1024

RejectReason = Literal[
    "malformed",
    "unsigned",
    "untrusted_signer",
    "bad_signature",
    "blob_missing",
    "blob_mismatch",
    "blob_not_canonical",
]


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["trace", "step"]
    body: dict[str, JsonValue]
    signer: str = ""
    signature: str = ""


class _Bundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["iab-cache-bundle"]
    version: Literal[1]
    entries: list[_Entry] = Field(max_length=200_000)
    blobs: dict[str, str] = Field(default_factory=dict)


@dataclass
class ImportReport:
    bundle_sha256: str
    accepted_traces: int = 0
    accepted_steps: int = 0
    rejected: Counter[str] = field(default_factory=Counter)

    @property
    def rejected_total(self) -> int:
        return sum(self.rejected.values())

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "bundle_sha256": self.bundle_sha256,
            "accepted_traces": self.accepted_traces,
            "accepted_steps": self.accepted_steps,
            "rejected": dict(sorted(self.rejected.items())),
            "rejected_total": self.rejected_total,
        }


def _trace_body(trace: Trace) -> dict[str, JsonValue]:
    deps: JsonValue = json.loads(encode_deps(trace.deps))
    return {
        "kind": "trace",
        "key_id": trace.key_id,
        "rule": trace.rule,
        "version": trace.version,
        "deps": deps,
        "result_hash": trace.result_hash,
        "usage": dict(trace.usage.to_json()),
    }


def _step_body(step: StepRecord) -> dict[str, JsonValue]:
    return {
        "kind": "step",
        "fingerprint": step.fingerprint,
        "step_kind": step.kind,
        "result_hash": step.result_hash,
        "meta": dict(step.meta),
    }


def export_bundle(
    store: TraceStore,
    cas: ContentStore,
    signer: SigningKey,
    *,
    key_ids: Iterable[str] | None = None,
) -> dict[str, JsonValue]:
    """Sign and export locally computed traces, the steps they use and their value blobs."""
    traces = store.load_traces()
    wanted = set(key_ids) if key_ids is not None else None
    steps = store.load_steps()
    entries: list[JsonValue] = []
    blobs: dict[str, JsonValue] = {}
    step_refs: set[str] = set()
    for key_id in sorted(traces):
        trace = traces[key_id]
        if trace.origin != "executed" or (wanted is not None and key_id not in wanted):
            continue
        if not cas.contains(trace.result_hash):
            continue
        body = _trace_body(trace)
        entries.append(
            {"type": "trace", "body": body, "signer": signer.key_id, "signature": signer.sign(body)}
        )
        blobs[trace.result_hash] = base64.b64encode(cas.get_bytes(trace.result_hash)).decode()
        for group in trace.deps:
            for dep in group:
                if dep.kind in (DepKind.LLM, DepKind.TOOL):
                    step_refs.add(dep.ref)
    for fingerprint in sorted(step_refs):
        step = steps.get(fingerprint)
        if step is None or "ttl_seconds" in step.meta or not cas.contains(step.result_hash):
            continue
        body = _step_body(step)
        entries.append(
            {"type": "step", "body": body, "signer": signer.key_id, "signature": signer.sign(body)}
        )
        blobs[step.result_hash] = base64.b64encode(cas.get_bytes(step.result_hash)).decode()
    return {"format": BUNDLE_FORMAT, "version": 1, "entries": entries, "blobs": blobs}


def _valid_trace_body(body: Mapping[str, Any]) -> bool:
    try:
        kind, _ = split_key_id(str(body["key_id"]))
        deps = decode_deps(json.dumps(body["deps"]))
    except (KeyError, ValueError, TypeError):
        return False
    if body.get("kind") != "trace" or kind != body.get("rule"):
        return False
    if not isinstance(body.get("version"), int) or not isinstance(body.get("usage"), dict):
        return False
    for group in deps:
        for dep in group:
            if dep.hash != MISSING and not is_sha256_hex(dep.hash):
                return False
    return is_sha256_hex(str(body.get("result_hash")))


def _valid_step_body(body: Mapping[str, Any]) -> bool:
    return (
        body.get("kind") == "step"
        and is_sha256_hex(str(body.get("fingerprint")))
        and body.get("step_kind") in {"llm", "tool"}
        and is_sha256_hex(str(body.get("result_hash")))
        and isinstance(body.get("meta"), dict)
        and "ttl_seconds" not in body["meta"]
    )


def _check_blob(result_hash: str, blobs: Mapping[str, str]) -> tuple[bytes | None, str | None]:
    encoded = blobs.get(result_hash)
    if encoded is None:
        return None, "blob_missing"
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None, "blob_mismatch"
    if sha256_hex(data) != result_hash:
        return None, "blob_mismatch"
    try:
        decode_json_blob(result_hash, data)
    except IntegrityError:
        return None, "blob_not_canonical"
    return data, None


def import_bundle(
    raw: bytes,
    *,
    store: TraceStore,
    cas: ContentStore,
    trust: TrustStore,
    metrics: BuildMetrics | None = None,
    clock_now: float = 0.0,
) -> ImportReport:
    """Verify and import a bundle. Bad entries are rejected individually and counted."""
    if len(raw) > MAX_BUNDLE_BYTES:
        raise CacheImportError(f"bundle exceeds {MAX_BUNDLE_BYTES} bytes")
    try:
        bundle = _Bundle.model_validate_json(raw)
    except ValidationError as exc:
        raise CacheImportError(f"not a {BUNDLE_FORMAT} v1 document") from exc
    report = ImportReport(bundle_sha256=sha256_hex(raw))
    accepted_traces: list[ConstructiveEntry] = []
    accepted_steps: list[StepRecord] = []
    local_steps = store.load_steps()

    def reject(reason: RejectReason) -> None:
        report.rejected[reason] += 1
        if metrics is not None:
            metrics.cache_import_rejections_total.labels(reason).inc()

    for entry in bundle.entries:
        body = entry.body
        valid = _valid_trace_body(body) if entry.type == "trace" else _valid_step_body(body)
        if not valid:
            reject("malformed")
            continue
        if not entry.signature or not entry.signer:
            reject("unsigned")
            continue
        if not trust.trusts(entry.signer):
            reject("untrusted_signer")
            continue
        if not trust.verify(entry.signer, body, entry.signature):
            reject("bad_signature")
            continue
        result_hash = str(body["result_hash"])
        data, problem = _check_blob(result_hash, bundle.blobs)
        if problem is not None or data is None:
            reject(problem or "blob_missing")  # type: ignore[arg-type]
            continue
        cas.put_bytes(data)
        if entry.type == "trace":
            entry_id = sha256_hex(canonical_json({"body": body, "signer": entry.signer}))
            accepted_traces.append(
                ConstructiveEntry(
                    entry_id, str(body["key_id"]), dict(body), entry.signer, entry.signature
                )
            )
        else:
            fingerprint = str(body["fingerprint"])
            if fingerprint in local_steps:
                continue  # never overwrite a locally recorded step
            meta = dict(body["meta"]) if isinstance(body["meta"], dict) else {}
            meta["origin"] = f"imported:{entry.signer}"
            accepted_steps.append(
                StepRecord(fingerprint, str(body["step_kind"]), result_hash, meta, clock_now)
            )
    report.accepted_traces = store.put_constructive(accepted_traces)
    store.put_steps(accepted_steps)
    report.accepted_steps = len(accepted_steps)
    return report
