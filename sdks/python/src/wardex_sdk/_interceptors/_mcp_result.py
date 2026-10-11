"""How an MCP `tools/call` result reads on a span: its output, its outcome, and whether wardex
could read it at all.

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
  JSON-RPC id); an `isError` that is not a boolean, or spelled some other way; or neither `content`
  nor `structuredContent`. Such a result ships whole as the output with status UNSET — neither the
  success nor the failure of a call nobody saw finish — and the span carries
  `Limitation.SEMANTIC_PARSE_FAILED`: framing succeeded, the meaning did not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .._assembly import guard
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
    is_error = result.get("isError", False)
    if (
        result.get("resultType", "complete") != "complete"
        or not isinstance(is_error, bool)
        or _respells_is_error(result)
        or ("content" not in result and "structuredContent" not in result)
    ):
        return _uninterpreted(raw)
    content = result.get("content")
    if isinstance(content, list) and set(result) <= _PLAIN_KEYS:
        output = json.dumps(content, ensure_ascii=False).encode("utf-8")
    else:
        output = raw
    if is_error:
        return ToolResult(output, StatusCode.ERROR, "tool_error", False)
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
