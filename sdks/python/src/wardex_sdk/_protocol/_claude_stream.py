"""Thin wrapper mapping the native claude stream-json parser to a frozen dataclass."""

from __future__ import annotations

from dataclasses import dataclass

from .. import _wardex_native


@dataclass(frozen=True, slots=True)
class AgentStreamEvent:
    kind: str
    session_id: str | None = None
    model: str | None = None
    message_id: str | None = None
    stop_reason: str | None = None
    parent_tool_use_id: str | None = None
    #: Which CALL a `tool_result` answers, never which sub-agent produced the
    #: line. One CLI line carries both and they are different questions.
    tool_result_id: str | None = None
    subtype: str | None = None
    task_id: str | None = None
    task_status: str | None = None
    task_tool_use_id: str | None = None
    content_json: bytes | None = None
    tool_uses: tuple[tuple[str, str, bytes], ...] = ()
    #: Semconv-inclusive (cache tiers added back in by the native parser).
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    #: A usage sub-counter arrived without its total; the total is None, not
    #: invented — the assembler counts the condition.
    usage_totals_unpaired: bool = False
    usage_overflowed: bool = False
    num_turns: int | None = None
    total_cost_usd: float | None = None
    duration_ms: int | None = None
    duration_api_ms: int | None = None
    is_error: bool = False


def parse_line(data: bytes, outbound: bool) -> AgentStreamEvent | None:
    raw = _wardex_native.protocol.parse_claude_stream_line(data, outbound)
    if raw is None:
        return None
    return AgentStreamEvent(
        kind=raw.kind,
        session_id=raw.session_id,
        model=raw.model,
        message_id=raw.message_id,
        stop_reason=raw.stop_reason,
        parent_tool_use_id=raw.parent_tool_use_id,
        tool_result_id=raw.tool_result_id,
        subtype=raw.subtype,
        task_id=raw.task_id,
        task_status=raw.task_status,
        task_tool_use_id=raw.task_tool_use_id,
        content_json=bytes(raw.content_json) if raw.content_json is not None else None,
        tool_uses=tuple((i, n, bytes(b)) for i, n, b in raw.tool_uses),
        input_tokens=raw.input_tokens,
        output_tokens=raw.output_tokens,
        cache_read_tokens=raw.cache_read_tokens,
        cache_creation_tokens=raw.cache_creation_tokens,
        usage_totals_unpaired=raw.usage_totals_unpaired,
        usage_overflowed=raw.usage_overflowed,
        num_turns=raw.num_turns,
        total_cost_usd=raw.total_cost_usd,
        duration_ms=raw.duration_ms,
        duration_api_ms=raw.duration_api_ms,
        is_error=raw.is_error,
    )
