"""How MCP traffic reads on a span: a `tools/call` result's output, its outcome, and whether
wardex could read it at all — and which of the peer's own words may name a span or a revision.

Kept apart from `_mcp_stdio.py`, which correlates the pipe, because this is the one place that
holds an opinion about what a `CallToolResult` looks like — and that opinion is the thing an MCP
revision can change under it. The revisions this reader is written against (2024-11-05 through
2026-07-28) agree on the shape it reads: `content` (a list of blocks), an optional
`structuredContent`, an optional `isError` boolean, `_meta`, and — from 2026-07-28 — `resultType`,
which is `"complete"` for a final result and absent before that revision.

Two rules, and the second is the one that used to be missing.

* Nothing the server returned is dropped. A result that is `content` and nothing else (beside
  `isError` and a `"complete"` `resultType`, which the status and the shape already carry) ships
  its `content` list as the output, exactly as before. Any other key — `structuredContent`, `_meta`,
  a key a later revision adds — ships the whole result object instead. A server answering with
  `structuredContent` alone used to produce an OK span whose output was `[]`.
* A result this reader cannot vouch for claims nothing. Not an object; a `resultType` other than
  `"complete"` (`input_required` is an interim answer, and the call goes on in a retry under a new
  JSON-RPC id); an `isError` that is not a boolean, or spelled some other way; a `content` that is
  not a list; or neither `content` nor `structuredContent`. Such a result ships whole as the output
  with status UNSET — neither the success nor the failure of a call nobody saw finish — and the
  span carries `Limitation.SEMANTIC_PARSE_FAILED`: framing succeeded, the meaning did not.

One exception, and it outranks the second rule: `isError: true` is the server saying the tool
failed, so that result is ERROR (`tool_error`) whatever else its shape lacks.

The peer also supplies the tool name and the protocol version, and both are text it chose. A tool
name becomes part of a span NAME only when it reads as one (`span_tool_name`); a version reaches a
span only when it reads as a revision identifier (`readable_version`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .._assembly import counters, guard, report_once
from .._enums import StatusCode

#: The keys a plain result may carry and still ship its `content` list as the output: the status
#: says what `isError` said, and a `"complete"` `resultType` says only that the result is final.
_PLAIN_KEYS = frozenset({"content", "isError", "resultType"})


@dataclass(frozen=True, slots=True)
class ToolResult:
    """One `tools/call` result as the span reports it."""

    output: bytes
    status: StatusCode
    #: `tool_error` when the server set `isError: true` — the value the MCP semantic conventions
    #: name for a successful JSON-RPC call whose result reports a failed tool.
    error_type: str | None
    #: The reader could not vouch for the shape: the output is the raw result, the status UNSET.
    uninterpreted: bool


def read_tool_result(raw: bytes) -> ToolResult:
    """Read a `tools/call` result, already re-encoded compact by the native parser.

    That re-encoding is why a decode failure here is wardex's own and goes through `guard`: the
    bytes came out of `serde_json`, not off the pipe. Either way the result ships uninterpreted.
    """
    result = None
    with guard("interceptors.mcp_stdio.tool_result"):
        result = json.loads(raw)
    if not isinstance(result, dict):
        return _uninterpreted(raw)
    content = result.get("content")
    if isinstance(content, list) and set(result) <= _PLAIN_KEYS:
        output = json.dumps(content, ensure_ascii=False).encode("utf-8")
    else:
        output = raw
    is_error = result.get("isError", False)
    if is_error is True:
        return ToolResult(output, StatusCode.ERROR, "tool_error", False)
    if (
        is_error is not False
        or result.get("resultType", "complete") != "complete"
        or _respells_is_error(result)
        or ("content" in result and not isinstance(content, list))
        or ("content" not in result and "structuredContent" not in result)
    ):
        return _uninterpreted(raw)
    return ToolResult(output, StatusCode.OK, None, False)


def _uninterpreted(raw: bytes) -> ToolResult:
    return ToolResult(raw, StatusCode.UNSET, None, True)


def _respells_is_error(result: dict[str, Any]) -> bool:
    """Does the result carry `isError` under another spelling (`is_error`, `IsError`)?

    Such a key may well say the tool failed, and reading only the exact spelling would then ship
    the failure as a success. Comparing case- and separator-free is cheap and claims no more than
    that the key looks like the one this reader knows.
    """
    return any(
        key != "isError" and key.replace("_", "").replace("-", "").lower() == "iserror"
        for key in result
    )


#: The longest tool name that may become part of a span name: the 128 characters the MCP schema
#: (2025-11-25) asks tool names to stay within. A longer one is still the tool's name on the span's
#: tool attributes; it just does not name the span.
MAX_SPAN_TOOL_NAME = 128

#: The longest protocol version that reaches a span. Every revision so far is a ten-character date.
MAX_VERSION_CHARS = 32
_VERSION_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-_")


def span_tool_name(name: object) -> str | None:
    """`name` if it may name a span (`execute_tool {name}`), else None.

    A span name is a low-cardinality label that every backend indexes and shows, and this one comes
    from the peer. So it must be a non-empty string of at most `MAX_SPAN_TOOL_NAME` printable
    characters: a dict, a number or a list would have been `str()`-ed into the name, and a newline
    or a tab would break every view that shows names on one line. A name that fails stays on the
    span's tool attributes when it is a string, and the span keeps its transport name.
    """
    if not isinstance(name, str) or not name or len(name) > MAX_SPAN_TOOL_NAME:
        return None
    return name if name.isprintable() and name.strip() else None


def readable_version(value: object) -> str | None:
    """`value` if it reads as an MCP revision identifier, else None.

    Revisions are dates (`2025-06-18`) today, and the official SDK warns that later identifiers
    need not be, so the test is a shape rather than a date: at most `MAX_VERSION_CHARS` letters,
    digits, dots, hyphens and underscores. That keeps any revision name a peer could mean while
    refusing what is not one — free text, a value two megabytes long that would ride every later
    span of the subprocess, or an address or card number a peer put there.
    """
    if not isinstance(value, str) or not value or len(value) > MAX_VERSION_CHARS:
        return None
    return value if set(value) <= _VERSION_CHARS else None


def json_object(data: bytes | None) -> dict[str, Any] | None:
    """`data` parsed as a JSON object, or None for anything else (absent, not an object).

    The bytes are the native parser's compact re-encoding, never the pipe's, so a decode failure
    is wardex's own: counted by `guard`, and the span still ships, unread.
    """
    if not data:
        return None
    parsed = None
    with guard("interceptors.mcp_stdio.json_object"):
        parsed = json.loads(data)
    return parsed if isinstance(parsed, dict) else None


def negotiated_version(result: bytes) -> str | None:
    """The `protocolVersion` an `initialize` result settled on, or None — and None, counted and
    said, when the result names one that does not read as a revision."""
    parsed = json_object(result)
    if parsed is None or "protocolVersion" not in parsed:
        return None
    version = readable_version(parsed["protocolVersion"])
    if version is None:
        refuse_version()
    return version


def refuse_version() -> None:
    """Count and say once that a peer's protocol version was left off the spans."""
    counters.bump("interceptors.mcp_stdio.protocol_version_unreadable")
    report_once(
        "an MCP peer named a protocol version that does not read as a revision identifier (not "
        "a string, too long, or free text), so spans do not carry it. Counted under "
        "interceptors.mcp_stdio.protocol_version_unreadable.",
        key="interceptors.mcp_stdio.protocol_version_unreadable",
    )
