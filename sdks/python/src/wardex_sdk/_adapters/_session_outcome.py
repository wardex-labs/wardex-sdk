"""What the Claude CLI's own reports say about how a session and its calls ended.

Four rules the Agent SDK adapter applies, kept out of the assembler because each
is a reading of the CLI's evidence rather than bookkeeping about it: whether a
reader that stopped was cancelled or broke, what the session root reports given
the CLI's `result` and whether that result answers the turn still in flight, how
a call the closing hook never reached ended, and whether a stream line is one
nothing in the assembler has a rule for. Every one of them answers "unknown"
when the evidence is missing; none of them invents an OK.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from .._assembly import counters
from .._enums import StatusCode
from .._protocol._claude_stream import AgentStreamEvent
from ._session_state import _OpenTool

#: Inbound stream-json `type`s the native parser builds events from. A line of
#: one of these types can still parse to nothing (a `system` subtype with no
#: rule, a sub-agent's plain `user` message), and that is a decision about a
#: KNOWN line, not a line nobody understood. Mirrors the parser's own match;
#: `test_agent_sdk_session_outcome.py` fails if the two drift apart.
PARSED_STREAM_TYPES = frozenset({"system", "assistant", "user", "stream_event", "result"})

#: Inbound `type`s the assembler knowingly builds nothing from, because there is
#: nothing span-shaped in them: the SDK's own control protocol (hook and MCP
#: requests reach the assembler through the hooks and the tool wrapper, not as
#: stream content), the SessionStore mirror the SDK peels off, the CLI's
#: liveness and per-tool heartbeats, and rate-limit notices.
IGNORED_STREAM_TYPES = frozenset(
    {
        "control_request",
        "control_response",
        "control_cancel_request",
        "transcript_mirror",
        "keep_alive",
        "tool_progress",
        "rate_limit_event",
    }
)


def note_unparsed_line(msg: Any) -> None:
    """Count a line the parser dropped when its TYPE is one nothing here knows.

    The day the CLI renames `assistant`, every chat span disappears while the
    root still reports OK, its turns and its cost; this counter is the one place
    that difference shows. Known types that parse to nothing are deliberate and
    are not counted, which is what keeps a non-zero value meaningful.
    """
    kind = msg.get("type") if isinstance(msg, dict) else None
    if kind not in PARSED_STREAM_TYPES and kind not in IGNORED_STREAM_TYPES:
        counters.bump("adapters.assembler.stream_line_unrecognized")


def reader_stopped(exc: BaseException) -> bool:
    """Did the CONSUMER stop reading, as opposed to the transport failing?

    `ClaudeSDKClient` keeps the CLI alive between turns, so when the host leaves
    `async with` its reader task is parked on the next line, and `disconnect()`
    ends it by cancelling that task. That is the documented way to close a
    multi-turn session, not a broken pipe; reporting it as a transport error
    turned the root of every cleanly closed session into ERROR. `GeneratorExit`
    is the same fact from the other side: whoever drove the iterator closed it.

    Stopping is not succeeding. The session then closes with no error, and its
    root reports whatever the CLI's `result` for the turn in flight said, or
    UNSET when that result never came (`root_status`).

    TOTAL, because it runs inside the tee's `except` with the host's exception
    in flight. trio's `Cancelled` is read off the already-imported module, never
    imported: the SDK runs on trio through anyio, and a trio cancel is not an
    `asyncio.CancelledError`.
    """
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return True
    trio_cancelled = getattr(sys.modules.get("trio"), "Cancelled", None)
    return isinstance(trio_cancelled, type) and isinstance(exc, trio_cancelled)


def awaiting_after(awaiting: bool, ev: AgentStreamEvent) -> bool:
    """Does the CLI still owe a `result` once this inbound event has arrived?

    A `result` settles the turn. A MAIN-THREAD assistant message or stream
    chunk means a turn is running, including one the CLI started by itself
    with no write from the host, which it does when a background agent that
    finished after the last result wakes the session (the SDK's own reader
    reopens its run on the same signal). A sub-agent's messages do not: they
    belong to a call inside the turn, not to a turn of their own.
    """
    if ev.kind == "session_result":
        return False
    if ev.kind in ("assistant_turn", "stream_delta") and ev.parent_tool_use_id is None:
        return True
    return awaiting


def root_status(
    result: AgentStreamEvent | None, awaiting_result: bool, error: str | None
) -> tuple[StatusCode, bool]:
    """The session root's status, and whether the session was cut short.

    Three outcomes, and only the CLI's own `result` can produce OK:

      * the transport FAILED (`error`): ERROR, cut short, whatever had arrived;
      * the CLI answered the turn in flight: its own `is_error` decides, also
        when the session ended by its reader being cancelled, which is how
        `ClaudeSDKClient` closes a finished session;
      * anything else ended the session before its last turn's result (a cancel
        mid-turn, a close with no result at all, a second turn abandoned after
        the first one finished): nobody observed the outcome, so UNSET, cut
        short.

    `awaiting_result` is what tells the second case from the third in a
    multi-turn session, where `result` may be the PREVIOUS turn's.
    """
    if error is not None:
        return StatusCode.ERROR, True
    if result is not None and not awaiting_result:
        return (StatusCode.ERROR if result.is_error else StatusCode.OK), False
    return StatusCode.UNSET, True


def close_from_stream(
    tool: _OpenTool, observed: tuple[bytes, bool, int], meta: tuple[str, bytes, int] | None
) -> tuple[StatusCode, str | None, int]:
    """Settle a hook-opened call from the `tool_result` the stream carried.

    `observed` is the call's `stream_result`. Returns `(status, error_type,
    end_ns)` and fills the record in. The closing hook never came, but the CLI
    did report the outcome, so the span says what the CLI said: OK or ERROR
    from the block's own `is_error`, the result bytes as output, the end at the
    instant the result arrived. The stream's name and byte-exact input (`meta`,
    the call's `stream_tool_meta` entry) win over the hook's re-serialized
    ones, as they do when the hook does close a call.

    A sole-live guess the opening hook made is settled: the call's id arrived on
    THIS session's transport, one of the proofs `_OpenTool.sole_inferred` lists.
    """
    content, is_error, arrived_ns = observed
    if meta is not None:
        stream_name, stream_input, _announced_ns = meta
        if stream_name:
            tool.name = stream_name
        if stream_input:
            tool.input_data = stream_input
    tool.output_data = content
    tool.closed_by_stream = True
    tool.sole_inferred = False
    if is_error:
        return StatusCode.ERROR, "tool_error", arrived_ns
    return StatusCode.OK, None, arrived_ns
