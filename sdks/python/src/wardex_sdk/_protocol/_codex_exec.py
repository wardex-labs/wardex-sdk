"""Thin wrapper mapping the native `codex exec --json` parser to a frozen dataclass."""

from __future__ import annotations

from dataclasses import dataclass

from .. import _wardex_native


@dataclass(frozen=True, slots=True)
class CodexExecEvent:
    kind: str
    thread_id: str | None = None
    message: str | None = None
    item_id: str | None = None
    item_type: str | None = None
    text: str | None = None
    item_json: bytes | None = None
    #: `turn.completed` carried a usage object at all.
    has_usage: bool = False
    #: Semconv-inclusive, as Codex reports them (cached tokens inside the total).
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    usage_totals_unpaired: bool = False
    usage_overflowed: bool = False


def parse_line(data: bytes) -> CodexExecEvent | None:
    raw = _wardex_native.protocol.parse_codex_exec_line(data)
    if raw is None:
        return None
    return CodexExecEvent(
        kind=raw.kind,
        thread_id=raw.thread_id,
        message=raw.message,
        item_id=raw.item_id,
        item_type=raw.item_type,
        text=raw.text,
        item_json=bytes(raw.item_json) if raw.item_json is not None else None,
        has_usage=raw.has_usage,
        input_tokens=raw.input_tokens,
        output_tokens=raw.output_tokens,
        cache_read_tokens=raw.cache_read_tokens,
        cache_creation_tokens=raw.cache_creation_tokens,
        reasoning_output_tokens=raw.reasoning_output_tokens,
        usage_totals_unpaired=raw.usage_totals_unpaired,
        usage_overflowed=raw.usage_overflowed,
    )
