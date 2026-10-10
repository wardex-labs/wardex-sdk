"""The openai-agents adapter's shared span pieces, and the span kinds past its first mapping.

`_openai_agents.py` maps the run, the agent, the handoff, the turn, the function tool, the
guardrail and the MCP list-tools step. This module holds what that mapping and the kinds below
both open spans with — the agent current on a task, `_open_child`, the framework's own start
instant and its error sentences — and the kinds themselves. Nothing here imports the mapping
module, so the two never form an import cycle.

* The model call — `ResponseSpanData`, and `GenerationSpanData` for a Chat Completions, LiteLLM or
  any-llm model. No span and no usage: the wire owns the call. What it contributes is the JOIN,
  the function calls it requested, for the tool spans that follow to recover their call id from.
  A Chat Completions message names them under `tool_calls`; a streamed one is handed over as a
  `Response` dump whose `output` holds `function_call` items. The source of a recovered id is
  labelled (`_CALL_ID_SOURCES`).
* Hosted tools — web search, file search, code interpreter, image generation, a hosted MCP call, a
  server-side tool search, and a shell call the provider ran (its output arrives in the same
  response). The provider runs them inside the model call, so the framework opens no function
  span for them. One `execute_tool` each, read off the response that carried it: its call id is
  the item's own `id`; its interval is the model call that ran it, because the provider reports
  no per-tool timing (`wardex.openai_agents.hosted_tool` says so); its status is the item's own.
  Its arguments and results stay the wire span's rather than being copied a second time. The
  wire span holds them as a `server_tool_call` part with that same id for web search, file
  search, code interpreter, image generation and a hosted MCP call; a tool search and a shell
  call are on the wire span too, but as parts the wire parser does not map, so no id joins them.
* The step kinds (`STEP_KINDS`) — a host's `custom_span`, the framework's sandbox spans, and the
  voice pipeline's speech, speech-group and transcription spans. One `execute_step` each, from
  the span's start to its end — or to the close of the agent or tool it sits under, if it
  outlives that: it is then cut there and says `child_span_unclosed`. NOT pinned: the framework
  starts and finishes some of these on different tasks, and keeps several open at once on one
  task, where a pin's stack discipline would not hold. So work inside one nests under the agent
  or tool around it, not under the step. A custom span's `data` mapping is recorded as the
  step's input, shaped like a tool's arguments.
* What the framework withheld — `trace_include_sensitive_data=False` strips a tool's arguments
  and result, and every model response, from the framework's spans. The tool span and the agent
  span then carry `wardex.openai_agents.sensitive_data_withheld`, so an empty payload, a missing
  call id or an absent hosted tool reads as withheld rather than as never having existed. Read
  off the run's own config (`RunCall.content_withheld`), never off a missing response, which a
  failed, cancelled or cut call leaves too.
* What this adapter drops — a span it cannot place, whose run it never saw start, or that arrived
  after its run ended. Each reason is a counter and one WARNING per process (`_dropped`).
"""

from __future__ import annotations

import contextvars
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from .._assembly import (
    ConversationContext,
    Limitation,
    SpanIntent,
    ToolAttributes,
    UnitKind,
    report_once,
)
from .._enums import StatusCode, ToolExecutionType, ToolType
from ._context import AdapterContext, Placement, RunHandle
from ._openai_agents_entry import RUN_CALL
from ._payload import _shaped_payload

_FRAMEWORK = "openai_agents"

#: The agent entry CURRENT on this task — set at `AgentSpanData` start on the task that opened it,
#: restored to the enclosing one at its end. Every task the framework spawns underneath (the model
#: task, one per tool call, one per guardrail, a nested run's loop) copies the context and so
#: inherits it: the same mechanism that carries the pin, applied to the adapter's own per-agent
#: bookkeeping. Keyed this way rather than on the trace because a trace is not one agent: the
#: framework's parallelization pattern gathers several `Runner.run`s under one `with trace(...)`,
#: and any "current agent" held on the trace would be whichever started last.
_CURRENT_AGENT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "wardex_openai_agents_current_agent", default=None
)


def _open_child(
    ctx: AdapterContext,
    kind: UnitKind,
    *,
    intent: SpanIntent,
    subject: str,
    site: str,
    describe: Any,
    start_ns: int | None = None,
    conversation: ConversationContext | None = None,
) -> RunHandle:
    """The ONE way a child unit opens under a run — agent, handoff marker, tool, guardrail, MCP step
    alike — so that a sixth site cannot forget what every child owes.

    What every child owes is the adapter's half of a refused agent pin: a child opened while that
    agent is current hangs under whatever IS ambient — the session, or an earlier agent — at 1.0, so
    the child says the edge is not what it looks like. The registry marks only the refused unit
    itself. This used to be a four-line check copied at four of five sites; the fifth (the MCP
    list-tools step) had none. `confirm_active` is here for the same reason: a site that opens is a
    site that counts. `conversation` is stated only by a top-level agent (`call_conversation`).
    """
    h = ctx.open_run(
        kind,
        intent=intent,
        placement=Placement.NESTED,
        subject=subject,
        start_ns=start_ns,
        conversation=conversation,
        describe=describe,
    )
    current = _CURRENT_AGENT.get()
    if current is not None and not current.get("pinned", True):
        h.note(Limitation.CORRELATION_CONFLICT)
    ctx.confirm_active(site)
    return h


def _started_ns(ctx: AdapterContext, span: Any) -> int | None:
    """The framework's own start instant, or None when unreadable.

    `started_at` comes from the provider's `time_iso()`, a hook a host may
    replace. The framework's own is aware UTC; a NAIVE string is read as UTC
    as well, because `datetime.timestamp()` would otherwise read it as the
    process's local time and shift the instant by the zone offset. A string
    that does not parse is counted and answered None: the span then takes
    wardex's own clock, which is late but not wrong.
    """
    raw = span.started_at
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        ctx.count("started_at_unparsed")
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000_000_000)


def _error_message(span: Any) -> str | None:
    err = span.error
    if isinstance(err, dict):
        message = err.get("message")
        return str(message) if message is not None else None
    return None


def _error_data(span: Any) -> dict[str, Any]:
    err = span.error
    data = err.get("data") if isinstance(err, dict) else None
    return data if isinstance(data, dict) else {}


# -- the model call: no span, no usage -------------------------------------------

#: How a tool span's call id was recovered, by the kind of model span that requested the call.
_CALL_ID_SOURCES: dict[str, str] = {
    "ResponseSpanData": "response_output_match",
    "GenerationSpanData": "generation_output_match",
}


def _model_end(adapter: Any, run: dict[str, Any], span: Any) -> None:
    """The wire owns the LLM call. What the framework's model span contributes is the JOIN: the
    response id (a Responses model only), and the function calls the call requested, so the tool
    spans that follow can carry the call id the wire span echoes in the next turn's input. `usage`
    is never read — the same tokens are on the wire span, and two sources bill twice. A Responses
    model's hosted tools become spans here (`_hosted_tools`)."""
    ctx = adapter._ctx
    if ctx is None:
        return
    agent = _CURRENT_AGENT.get()
    if agent is None:
        ctx.count("response_without_agent")
        return
    sd = span.span_data
    kind = type(sd).__name__
    held = sd.output if kind == "GenerationSpanData" else sd.response
    # Withheld only where the run's own config says the framework strips content: a missing
    # response alone is also a failed, cancelled or cut call, which withheld nothing.
    call = RUN_CALL.get()
    if held is None and span.error is None and call is not None and call.content_withheld:
        agent["withheld"] = True
        ctx.count("model_output_withheld")
    agent["calls_from"] = _CALL_ID_SOURCES.get(kind)
    if kind == "GenerationSpanData":
        agent["response_id"] = None
        agent["calls"] = _generation_calls(held)
        return
    calls: dict[tuple[str, str], list[str]] = {}
    agent["calls"] = calls
    rid = held.id if held is not None else None
    agent["response_id"] = str(rid) if rid is not None else None
    if rid is None:
        ctx.count("response_id_unavailable")
    if held is None:
        return
    for item in held.output or ():
        if getattr(item, "type", None) != "function_call":
            continue
        key = (_tool_trace_name(item), str(item.arguments))
        calls.setdefault(key, []).append(str(item.call_id))
    _hosted_tools(adapter, span, agent, held)


def _trace_name(name: Any, namespace: Any) -> str:
    """The framework's own spelling of a tool call's span name
    (`_tool_identity.tool_trace_name`): `namespace.name` for a call under
    `tool_namespace()`, the bare `name` otherwise — and bare when the
    namespace EQUALS the name, the reserved synthetic shape a deferred
    top-level tool arrives in. `FunctionSpanData.name` is this string, so
    the call-id match keys the request side the same way."""
    name = str(name)
    if isinstance(namespace, str) and namespace and namespace != name:
        return f"{namespace}.{name}"
    return name


def _tool_trace_name(item: Any) -> str:
    return _trace_name(item.name, getattr(item, "namespace", None))


def _generation_calls(output: Any) -> dict[tuple[str, str], list[str]]:
    """The function calls a `GenerationSpanData.output` requested, keyed like `_model_end`'s.

    Two shapes, both the framework's own: a non-streamed Chat Completions call hands over the
    message dump, whose calls are `tool_calls[].function`; a streamed one hands over the dump of
    the `Response` it assembled from the chunks, whose calls are `function_call` items in
    `output`. Anything else — a custom tool call, a value of another shape — requests no function
    tool and is skipped.
    """
    calls: dict[tuple[str, str], list[str]] = {}
    for entry in output or ():
        if not isinstance(entry, Mapping):
            continue
        for tc in entry.get("tool_calls") or ():
            fn = tc.get("function") if isinstance(tc, Mapping) else None
            if isinstance(fn, Mapping) and tc.get("id") is not None:
                key = (str(fn.get("name")), str(fn.get("arguments")))
                calls.setdefault(key, []).append(str(tc["id"]))
        for item in entry.get("output") or ():
            if isinstance(item, Mapping) and item.get("type") == "function_call":
                key = (
                    _trace_name(item.get("name"), item.get("namespace")),
                    str(item.get("arguments")),
                )
                calls.setdefault(key, []).append(str(item.get("call_id")))
    return calls


# -- hosted tools ------------------------------------------------------------------

#: Output item types the PROVIDER executes, as the tool name and type their span carries. A hosted
#: MCP call is named after the MCP tool it called; a file search reads a store, the rest act.
_HOSTED: dict[str, tuple[str, ToolType]] = {
    "web_search_call": ("web_search", ToolType.EXTENSION),
    "file_search_call": ("file_search", ToolType.DATASTORE),
    "code_interpreter_call": ("code_interpreter", ToolType.EXTENSION),
    "image_generation_call": ("image_generation", ToolType.EXTENSION),
    "mcp_call": ("mcp", ToolType.EXTENSION),
    "tool_search_call": ("tool_search", ToolType.EXTENSION),
    "shell_call": ("shell", ToolType.EXTENSION),
}


def _hosted_tools(adapter: Any, span: Any, agent: dict[str, Any], response: Any) -> None:
    """One `execute_tool` per server-executed item in `response.output`, closed at once.

    A `shell_call` counts only when its `shell_call_output` is in the same response — the
    provider ran it; a local shell's output arrives in the NEXT request and has a function span of
    its own. A `tool_search_call` counts unless it asks the client to execute it. Opened on the
    model task, under the agent that asked, at the model call's start, and closed at its end.
    """
    ctx = adapter._ctx
    items = list(response.output or ())
    shell_done = {
        getattr(item, "call_id", None)
        for item in items
        if getattr(item, "type", None) == "shell_call_output"
    }
    start_ns = None
    with ctx.guard("hosted_started_at"):
        start_ns = _started_ns(ctx, span)
    for item in items:
        kind = getattr(item, "type", None)
        if not isinstance(kind, str) or kind not in _HOSTED:
            continue
        if kind == "shell_call" and getattr(item, "call_id", None) not in shell_done:
            continue
        if kind == "tool_search_call" and getattr(item, "execution", None) == "client":
            continue
        _hosted_tool(ctx, item, kind, agent, start_ns)


def _hosted_tool(
    ctx: AdapterContext, item: Any, kind: str, agent: dict[str, Any], start_ns: int | None
) -> None:
    name, tool_type = _HOSTED[kind]
    if kind == "mcp_call":
        name = str(item.name)
    # The item's own `id`, for every kind: the key a wire span's server tool part carries. A
    # shell call's `call_id` is the model's, and no wire part is keyed by it.
    raw_id = getattr(item, "id", None)
    call_id = str(raw_id) if raw_id is not None else None
    server = getattr(item, "server_label", None) if kind == "mcp_call" else None
    turn = agent.get("turn")
    response_id = agent.get("response_id")

    def describe(h: RunHandle) -> None:
        h.draft.set_tool(
            ToolAttributes(
                name=name,
                call_id=call_id,
                type=tool_type,
                execution_type=ToolExecutionType.NETWORK,
            )
        )
        h.draft.set_extra("wardex.framework", _FRAMEWORK)
        h.draft.set_extra("wardex.openai_agents.hosted_tool", True)
        if turn is not None:
            h.draft.set_extra("wardex.openai_agents.turn", int(turn))
        if response_id is not None:
            h.draft.set_extra("wardex.openai_agents.response_id", str(response_id))
        if call_id is not None:
            h.draft.set_extra("wardex.openai_agents.tool_call_id_source", "response_output_item")
        if server is not None:
            h.draft.set_extra("wardex.openai_agents.mcp.server", str(server))

    h = _open_child(
        ctx,
        UnitKind.CALL,
        intent=SpanIntent.EXECUTE_TOOL,
        subject=name,
        site="hosted_tool",
        describe=describe,
        start_ns=start_ns,
    )
    status = getattr(item, "status", None)
    if status == "failed" or getattr(item, "error", None) is not None:
        h.close(status=StatusCode.ERROR, error_type="hosted_tool_failed")
    elif status in (None, "completed"):
        h.close()
    else:
        # `in_progress`, `searching`, `incomplete`, ...: the provider reported no outcome.
        h.close(status=StatusCode.UNSET)


# -- the step kinds ------------------------------------------------------------------

#: Span kinds recorded as an `execute_step` and nothing more, and the step's name: the kind's own
#: fixed word, or None for the span's own `name` (a custom span is named by whoever opened it).
STEP_KINDS: dict[str, str | None] = {
    "CustomSpanData": None,
    "SpeechSpanData": "speech",
    "SpeechGroupSpanData": "speech_group",
    "TranscriptionSpanData": "transcription",
}


def _step_start(adapter: Any, run: dict[str, Any], span: Any) -> None:
    ctx = adapter._ctx
    if ctx is None:
        return
    sd = span.span_data
    fixed = STEP_KINDS.get(type(sd).__name__)
    name = fixed if fixed is not None else str(sd.name)

    def describe(h: RunHandle) -> None:
        h.draft.set_extra("wardex.step.name", name)
        h.draft.set_extra("wardex.framework", _FRAMEWORK)

    h = _open_child(
        ctx,
        UnitKind.STEP,
        intent=SpanIntent.EXECUTE_STEP,
        subject=name,
        site="step",
        describe=describe,
    )
    ctx.slot(span)["handle"] = h


def _step_end(adapter: Any, run: dict[str, Any], span: Any) -> None:
    ctx = adapter._ctx
    if ctx is None:
        return
    h = ctx.slot(span).get("handle")
    if h is None:
        ctx.count("step_end_unmatched")
        return
    sd = span.span_data
    # Read at the END: a custom span's opener may fill `data` while it runs.
    data = getattr(sd, "data", None) if STEP_KINDS.get(type(sd).__name__) is None else None
    if data:
        h.record_input(_shaped_payload(data, ctx.record_budget))
    message = _error_message(span)
    if message is not None:
        # The framework's own sentence when it is one; otherwise the opener's own words, which are
        # not a framework sentence this adapter failed to map, so nothing is reported for them.
        error_type = _ERROR_TABLE.get(message, ("openai_agents_error", False))[0]
        h.close(status=StatusCode.ERROR, error_type=error_type)
    else:
        h.close()


# -- what this adapter drops -----------------------------------------------------------

#: Where a span is LOST, as the line that says so. An end that finds no start of its own
#: (`agent_end_unmatched`, `tool_end_unmatched`, `guardrail_end_unmatched`, `step_end_unmatched`) is
#: only counted: its start was lost to one of the reasons below, or to a callback failure
#: `_contained` reported, and that line already named this span. A second line would make one loss
#: read as two.
_DROPPED: dict[str, str] = {
    "trace_lookup_miss": (
        "a span arrived on a task or thread where the framework's current trace is not its own, "
        "so wardex could not place it and did not record it"
    ),
    "span_without_run": (
        "a span arrived for a run whose start wardex never saw (its trace started before "
        "wardex.init(), or the run's root failed to open), so it was not recorded"
    ),
    "span_kind_ignored": (
        "the framework opened a span of a kind this adapter does not map, so it was not recorded"
    ),
    "span_after_run": (
        "a span arrived after its run had ended (from a task the run left running), so it was "
        "not recorded"
    ),
}


def _run_missing(ctx: AdapterContext | None, trace: Any) -> None:
    """A span whose run is not open: one that ENDED (`_close_run` leaves `ended` on the run's
    slot) or one wardex never saw start. Read with `peek`, so asking allocates nothing."""
    run = ctx.peek(trace) if ctx is not None else None
    _dropped(ctx, "span_after_run" if run is not None and run.get("ended") else "span_without_run")


def _dropped(ctx: AdapterContext | None, reason: str, kind: str | None = None) -> None:
    """A span this adapter did not record: counted every time, and said ONCE per reason per
    process. The words are wardex's own and the span kind's class name, never the span's content.
    """
    if ctx is None:
        return
    ctx.count(reason)
    seen = f" (first seen: {kind})" if kind is not None else ""
    report_once(
        f"openai-agents adapter: {_DROPPED[reason]}{seen}; the counter "
        f"adapters.openai_agents.{reason} counts every such span",
        key=f"adapters.openai_agents.{reason}",
    )


# -- failure mapping ---------------------------------------------------------------

#: The framework's span error message -> (wardex error type, fatal to the run).
_ERROR_TABLE: dict[str, tuple[str, bool]] = {
    "Max turns exceeded": ("max_turns_exceeded", True),
    "Guardrail tripwire triggered": ("guardrail_tripwire", True),
    "Tool execution cancelled": ("tool_cancelled", False),
    "Error running tool": ("tool_error", True),
    "Error running tool (non-fatal)": ("tool_error_handled", False),
    "Multiple handoffs requested": ("multiple_handoffs_requested", False),
    "Error in agent run": ("agent_run_error", True),
    "Error in call_model_input_filter": ("model_behavior_error", True),
    "Invalid JSON provided": ("model_behavior_error", True),
    "Invalid JSON": ("model_behavior_error", True),
}
_MODEL_BEHAVIOR_PREFIXES = ("Program ", "Tool approval ", "Invalid input filter")


def _classify_error(adapter: Any, message: str) -> tuple[str, bool]:
    known = _ERROR_TABLE.get(message)
    if known is not None:
        return known
    if message.endswith(" not found") or message.startswith(_MODEL_BEHAVIOR_PREFIXES):
        return ("model_behavior_error", True)
    ctx = adapter._ctx
    if ctx is not None:
        ctx.count("error_message_unmapped")
    # ONE fixed key, and the message stays OFF the line. The framework sets a HOST-supplied string
    # as a function span's error message (the reason an `on_approval` callback returns for a
    # rejected tool call), so a key built from it would grow `report_once`'s process-global table by
    # one entry per distinct reason — the bound the function exists to keep — and printing it would
    # put host content on stderr outside the masking pipeline. The counter above carries the volume;
    # the span carries the type.
    report_once(
        "openai-agents adapter: the framework reported an error message this adapter "
        "does not map; the span carries error_type=openai_agents_error and the counter "
        "adapters.openai_agents.error_message_unmapped counts every occurrence",
        key="adapters.openai_agents.unmapped",
    )
    return ("openai_agents_error", False)
