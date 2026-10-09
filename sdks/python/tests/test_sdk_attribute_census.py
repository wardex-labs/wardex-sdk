"""Every span attribute the SDK writes, and whose value it holds.

Masking is for what the host application and its traffic put on a span. A
value the SDK computed is neither, and the rules cannot tell the two apart: the
OpenAI Agents adapter's MCP tool-list digest is sixteen hex digits of a SHA-256,
for about one tool list in eighteen thousand all sixteen are decimal and pass
the card checksum, and the card rule then rewrote the digest and wrote
`credit_card` into the span's record of what it masked -- on every run against
that server, since the digest of a tool list does not change.

The walk in the Rust core leaves an attribute alone when the SDK computes its
value, by the key the SDK writes it under and the exact form it writes there
(`SDK_VALUE_ATTRS` in `crates/wardex-pipeline/src/pii/walk.rs`); everything
else is judged. Which attributes those are is a fact about this package's
source, so this file reads it from the source: every `set_extra(key, value)`
call under `wardex_sdk`, by AST, with key and value spelled as the code spells
them. `_SITES` says where each call's value comes from. A new call fails here
until someone decides that; a call that is gone fails until its row goes.

The four answers, and what each one is held to below:

* `COMPUTED` -- text the SDK computes. The walk must list its key, and a value
  the producer really makes that the card rule fires on ships as written, with
  no record, on both wires. The same value anywhere else is still masked.
* `LITERAL` -- text the SDK picks from a fixed set. No rule may fire on any
  member, so no exemption is needed and none exists.
* `NUMBER` -- a count or flag the SDK writes as a number. The walk reads only
  text, so no rule reads it.
* `HOST` -- the host's, its framework's or its traffic's: a node name, a
  response id, a thread id, a CLI's measurement. Judged like any host text,
  which a card number under each such key proves.

`set_extra` is how SDK code writes a span attribute. The other ways an
attribute reaches a span carry no computed text: the builder's
`gen_ai.operation.name` and the snapshot's `wardex.limitations` are closed
vocabularies no rule fires on (held below, and name by name in the Rust
census), the client's scope tags and user fields are the host's, and the
binding flattens typed blocks whose values the host, its framework or its
traffic supplied.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import re
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from typing import Any

import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub, _wardex_native
from wardex_sdk.transport import Transport

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "wardex_sdk"
_REPO = pathlib.Path(__file__).resolve().parents[3]
_WALK = _REPO / "crates" / "wardex-pipeline" / "src" / "pii" / "walk.rs"
_RUST_CENSUS = _REPO / "crates" / "wardex-pipeline" / "tests" / "sdk_generated_fields.rs"

COMPUTED = "computed"
LITERAL = "literal"
NUMBER = "number"
HOST = "host"

_OA = "_adapters/_openai_agents.py"
_LG = "_adapters/_langgraph.py"
_AS = "_adapters/_assembler.py"
_SEAM = "_interceptors/_seam.py"
_GENAI = "_semantics/_genai.py"
_CX = "_adapters/_codex_exec.py"

#: Every `set_extra` call in the package: (module, key as written, value as
#: written) and where the value comes from. One row per call, so a key written
#: from three places has three rows.
_SITES: list[tuple[str, str, str, str]] = [
    # -- the OpenAI Agents adapter --
    (_OA, "'wardex.framework'", "_FRAMEWORK", LITERAL),
    (_OA, "'wardex.framework'", "_FRAMEWORK", LITERAL),
    (_OA, "'wardex.framework'", "_FRAMEWORK", LITERAL),
    (_OA, "'wardex.framework'", "_FRAMEWORK", LITERAL),
    (_OA, "'wardex.framework'", "_FRAMEWORK", LITERAL),
    (_OA, "'wardex.framework'", "_FRAMEWORK", LITERAL),
    (_OA, "'wardex.step.name'", "'mcp.list_tools'", LITERAL),
    (_OA, "'wardex.openai_agents.tool_call_id_source'", "'response_output_match'", LITERAL),
    # `hash_canonical(sorted(tool names))[:16]`.
    (_OA, "'wardex.openai_agents.mcp.tools_hash'", "digest", COMPUTED),
    (_OA, "'wardex.openai_agents.mcp.tools_count'", "count", NUMBER),
    (_OA, "'wardex.openai_agents.turns'", "int(run.get('turn_max') or 0)", NUMBER),
    (_OA, "'wardex.openai_agents.agents'", "int(run.get('agent_count') or 0)", NUMBER),
    (_OA, "'wardex.openai_agents.turns'", "int(entry.get('turns') or 0)", NUMBER),
    (_OA, "'wardex.openai_agents.tools_count'", "len(tools) if tools else 0", NUMBER),
    (_OA, "'wardex.openai_agents.handoffs_count'", "len(handoffs) if handoffs else 0", NUMBER),
    (_OA, "'wardex.openai_agents.turn'", "int(turn)", NUMBER),
    (_OA, "'wardex.openai_agents.turn'", "int(turn)", NUMBER),
    (_OA, "'wardex.openai_agents.resumed'", "True", NUMBER),
    (_OA, "'wardex.evaluation.triggered'", "True", NUMBER),
    (_OA, "'wardex.evaluation.triggered'", "False", NUMBER),
    # The framework's trace and group ids, the provider's response ids, the
    # host's MCP server name and turn limit.
    (_OA, "'wardex.openai_agents.trace_id'", "trace_id", HOST),
    (_OA, "'wardex.openai_agents.group_id'", "shadowed", HOST),
    (_OA, "'wardex.openai_agents.last_response_id'", "str(last)", HOST),
    (_OA, "'wardex.openai_agents.response_id'", "str(response_id)", HOST),
    (_OA, "'wardex.openai_agents.response_id'", "str(response_id)", HOST),
    (_OA, "'wardex.openai_agents.mcp.server'", "str(mcp['server'])", HOST),
    (_OA, "'wardex.openai_agents.mcp.server'", "server if server is not None else ''", HOST),
    (_OA, "'wardex.openai_agents.max_turns'", "max_turns", HOST),
    # -- the Codex CLI adapter --
    # The process's exit code, the turn's usage totals off the --json stream,
    # and what the OTel bridge counted: whether the version is the verified
    # one, how many warm-up requests there were and how long they took.
    (_CX, "'wardex.codex.exit_code'", "returncode", NUMBER),
    (_CX, "'wardex.codex.turn.input_tokens'", "usage.input_tokens", NUMBER),
    (_CX, "'wardex.codex.turn.output_tokens'", "usage.output_tokens", NUMBER),
    (_CX, "'wardex.codex.turn.cache_read_input_tokens'", "usage.cache_read_tokens", NUMBER),
    (
        _CX,
        "'wardex.codex.turn.reasoning_output_tokens'",
        "usage.reasoning_output_tokens",
        NUMBER,
    ),
    (_CX, "'wardex.codex.version_verified'", "view.version == VERIFIED_VERSION", NUMBER),
    (_CX, "'wardex.codex.warmup.requests'", "len(view.warmups)", NUMBER),
    (
        _CX,
        "'wardex.codex.warmup.duration_ms'",
        "sum((max(0, w.end_ns - w.start_ns) for w in view.warmups)) // 1000000",
        NUMBER,
    ),
    # What Codex itself wrote: its thread id, an item's type, the version it
    # reports and the message of a failed turn.
    (_CX, "'wardex.codex.thread_id'", "reading.thread_id", HOST),
    (_CX, "'wardex.codex.item_type'", "ev.item_type", HOST),
    (_CX, "'wardex.codex.version'", "view.version", HOST),
    (_CX, "'wardex.codex.error'", "reading.failures[-1][:500]", HOST),
    # -- the LangGraph adapter --
    (_LG, "'wardex.framework'", "_FRAMEWORK", LITERAL),
    (_LG, "'wardex.framework'", "_FRAMEWORK", LITERAL),
    (_LG, "'wardex.framework'", "_FRAMEWORK", LITERAL),
    (_LG, "'wardex.langgraph.remote'", "'true'", LITERAL),
    # The graph's node names, LangGraph's task ids, step numbers, triggers and
    # checkpoint namespaces, and the host's thread id and `Command` target.
    (_LG, "'wardex.step.name'", "task.name", HOST),
    (_LG, "'wardex.step.task_id'", "str(task.id)", HOST),
    (_LG, "'wardex.step.index'", "index", HOST),
    (_LG, "'wardex.step.trigger'", "','.join((str(t) for t in triggers))", HOST),
    (_LG, "'wardex.step.namespace'", "str(ns)", HOST),
    (_LG, "'wardex.langgraph.thread_id'", "thread_id", HOST),
    (_LG, "'wardex.langgraph.command_goto'", "goto", HOST),
    # -- the Agent SDK adapter --
    (_AS, "'wardex.agent.prompt_source'", "sess.pending_prompt_source", LITERAL),
    (_AS, "'wardex.step.name'", "step_name", LITERAL),
    # What the CLI reports: its version, its span names and allow-listed
    # attributes, the provider's request id, and its own counts and timings.
    (_AS, "OTEL_EXTRA_PREFIX + 'cli_version'", "cli_version", HOST),
    (_AS, "OTEL_EXTRA_PREFIX + 'span'", "span.name", HOST),
    (_AS, "OTEL_EXTRA_PREFIX + 'request_id'", "llm.response_id", HOST),
    (_AS, "OTEL_EXTRA_PREFIX + 'tool_duration_ms'", "duration_ms", HOST),
    (_AS, "key", "value", HOST),
    (_AS, "'wardex.agent.num_turns'", "result.num_turns", HOST),
    (_AS, "'wardex.agent.cost_usd'", "result.total_cost_usd", HOST),
    (_AS, "'wardex.agent.api_duration_ms'", "result.duration_api_ms", HOST),
    # -- the byte seam --
    (_SEAM, "'network.protocol.version'", "'websocket'", LITERAL),
    (_SEAM, "'ws.messages.sent'", "txn.ws_messages_sent", NUMBER),
    (_SEAM, "'ws.messages.received'", "txn.ws_messages_received", NUMBER),
    # A direction's byte count, set only when that direction was counted whole.
    (_SEAM, "'ws.bytes.sent'", "sent", NUMBER),
    (_SEAM, "'ws.bytes.received'", "received", NUMBER),
    (_SEAM, "USAGE_DROPPED_KEY", "dropped", NUMBER),
    # The peer's close code and HTTP version, the gRPC fields and provider
    # extras read off the wire, and the messages the provider was sent.
    (_SEAM, "'ws.close_code'", "code", HOST),
    (_SEAM, "'network.protocol.version'", "txn.version", HOST),
    (_SEAM, "key", "value", HOST),
    (_SEAM, "key", "value", HOST),
    (_SEAM, "'gen_ai.output.messages'", "om", HOST),
    (_SEAM, "'gen_ai.input.messages'", "im", HOST),
    (_SEAM, "'gen_ai.system_instructions'", "si", HOST),
    # -- the semantics layer --
    # The conversation a Responses request body names: the provider's id for a
    # conversation it holds, put in the request by the host or its framework.
    # Traffic, so judged like any other text the request carried.
    (_GENAI, "REQUEST_CONVERSATION_KEY", "stated", HOST),
    (_GENAI, "REQUEST_CONVERSATION_KEY", "stated", HOST),
    # -- the published API: `Span.set_attribute` and a snapshot's attributes --
    ("_tracing.py", "key", "value", HOST),
    ("_assembly/_snapshot.py", "key", "value", HOST),
]

#: For each COMPUTED key, its producer, run for real on an input whose output
#: the card rule fires on.
_COMPUTED_SAMPLES: dict[str, Callable[[], str]] = {
    "wardex.openai_agents.mcp.tools_hash": lambda: importlib.import_module(
        "wardex_sdk._adapters._openai_agents"
    )._tools_digest(sorted(("get_weather", "search_docs_v28401"))),
}

#: LITERAL values the AST cannot read off the call, and where the set comes from.
_LITERAL_SETS: dict[tuple[str, str], Callable[[], tuple[str, ...]]] = {
    # Assigned only `_PROMPT_STREAM` or `_PROMPT_HOOK` (or None, which is
    # never written) -- `test_the_prompt_source_is_only_ever_one_of_two_names`.
    (_AS, "sess.pending_prompt_source"): lambda: (
        importlib.import_module("wardex_sdk._adapters._session_state")._PROMPT_STREAM,
        importlib.import_module("wardex_sdk._adapters._session_state")._PROMPT_HOOK,
    ),
    # A CLI span name mapped through `_STEP_NAMES`, or the literal the
    # conflicted-LLM path passes.
    (_AS, "step_name"): lambda: (
        *importlib.import_module("wardex_sdk._adapters._otel_merge")._STEP_NAMES.values(),
        "llm_request",
    ),
}

_CARD = "4111-1111-1111-1111"


def _scan() -> list[tuple[str, str, str]]:
    out = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "set_extra"
            ):
                assert len(node.args) == 2 and not node.keywords, ast.unparse(node)
                key, value = (ast.unparse(a) for a in node.args)
                out.append((path.relative_to(_SRC).as_posix(), key, value))
    return out


def _module(rel: str) -> Any:
    return importlib.import_module("wardex_sdk." + rel.removesuffix(".py").replace("/", "."))


def _resolve(rel: str, src: str) -> str | None:
    """The text an expression always evaluates to, or None when it depends on
    the call: a string literal, a module constant, or a sum of those."""

    def ev(node: ast.expr) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            value = getattr(_module(rel), node.id, None)
            return value if isinstance(value, str) else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = ev(node.left), ev(node.right)
            return None if left is None or right is None else left + right
        return None

    return ev(ast.parse(src, mode="eval").body)


def _keys(origin: str) -> set[str]:
    out = set()
    for rel, key, _, o in _SITES:
        if o == origin:
            resolved = _resolve(rel, key)
            assert resolved is not None, (rel, key)
            out.add(resolved)
    return out


def _walk_exempt_keys() -> set[str]:
    """The keys in the walk's `SDK_VALUE_ATTRS`, read from its source."""
    text = _WALK.read_text(encoding="utf-8")
    consts = dict(re.findall(r'const (\w+): &str = "([^"]*)";', text))
    block = re.search(r"const SDK_VALUE_ATTRS: [^=]+= &\[(.*?)\n\];", text, re.S)
    assert block is not None, "SDK_VALUE_ATTRS is gone from walk.rs"
    names = re.findall(r"^\s*\((\w+),", block.group(1), re.M)
    assert names, block.group(1)
    return {consts[n] for n in names}


# ==========================================================================
# The census
# ==========================================================================


def test_every_set_extra_call_in_the_sdk_is_classified():
    scanned = Counter(_scan())
    expected = Counter((rel, key, value) for rel, key, value, _ in _SITES)
    assert scanned - expected == Counter(), "calls with no row: classify them in _SITES"
    assert expected - scanned == Counter(), "rows with no call: remove them from _SITES"
    assert {o for *_, o in _SITES} == {COMPUTED, LITERAL, NUMBER, HOST}


def test_a_computed_key_is_never_also_written_from_anything_else():
    computed = _keys(COMPUTED)
    others = {_resolve(rel, key) for rel, key, _, o in _SITES if o != COMPUTED}
    assert computed and not computed & others, computed & others


def test_the_walk_exempts_exactly_the_computed_keys():
    # The conversation id is the one exemption no `set_extra` call writes: it
    # is where the OTLP mapping spells the typed conversation field.
    assert _walk_exempt_keys() == _keys(COMPUTED) | {"gen_ai.conversation.id"}
    census = _RUST_CENSUS.read_text(encoding="utf-8")
    block = census.split("const SDK_COMPUTED_ATTRS", 1)[1].split("];", 1)[0]
    for key in _keys(COMPUTED):
        assert f'"{key}"' in block, f"{key} has no row in the Rust census"
    assert set(_COMPUTED_SAMPLES) == _keys(COMPUTED)


def test_a_literal_value_is_read_off_the_call_or_named_here():
    for rel, _, value, origin in _SITES:
        if origin != LITERAL:
            continue
        if (rel, value) in _LITERAL_SETS:
            assert _resolve(rel, value) is None, (rel, value)
        else:
            assert _resolve(rel, value) is not None, (rel, value)


def test_the_prompt_source_is_only_ever_one_of_two_names():
    allowed = {"_PROMPT_STREAM", "_PROMPT_HOOK", "None"}
    seen = set()
    for path in _SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, (ast.AnnAssign, ast.AugAssign))
                else []
            )
            for t in targets:
                if isinstance(t, ast.Attribute) and t.attr == "pending_prompt_source":
                    seen.add(ast.unparse(node.value) if node.value is not None else "None")
    assert seen and seen <= allowed, seen


def test_an_increment_step_name_is_only_ever_a_mapped_name_or_llm_request():
    tree = ast.parse((_SRC / _AS).read_text(encoding="utf-8"))
    passed = {
        ast.unparse(node.args[2])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_emit_increment"
    }
    # The loop over `view.increments`, and the conflicted-LLM path.
    assert passed == {"step_name", "'llm_request'"}, passed
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and ast.unparse(node.target) == "(step_name, span)"
    ]
    assert [ast.unparse(n.iter) for n in loops] == ["view.increments"]
    merge = ast.parse((_SRC / "_adapters/_otel_merge.py").read_text(encoding="utf-8"))
    appended = {
        ast.unparse(node.args[0])
        for node in ast.walk(merge)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and ast.unparse(node.func.value).endswith(".increments")
    }
    assert appended == {"(_STEP_NAMES[name], span)"}, appended


# ==========================================================================
# What each answer is held to, through the real pipeline
# ==========================================================================


class _Wires(Transport):
    def __init__(self) -> None:
        self.envelopes: list[bytes] = []
        self.otlp: list[bytes] = []

    def export(self, envelope: Any, *, timeout: float | None = None) -> None:
        self.otlp.extend(self.encode(envelope, compress=False))
        self.envelopes.append(
            _wardex_native.codec.encode_envelope(
                envelope,
                self._pii_mode,
                list(self._pii_disabled),
                self._limits,
                **self._pii_names(),
            )
        )


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    _hub.reset_for_test()
    yield
    _hub.reset_for_test()


def _ship(attrs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """One hand-named span carrying `attrs`, encoded with the default rules:
    the envelope span and the OTLP span, as decoded dicts."""
    wires = _Wires()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wardex.init(transport=wires, intercept=False)
    try:
        with wardex.span("census") as span:
            for key, value in attrs.items():
                span.set_attribute(key, value)
        wardex.flush()
    finally:
        wardex.close()
    (env,) = [
        it["span"]
        for b in wires.envelopes
        for it in _wardex_native.codec.decode_envelope(b)["items"]
        if "span" in it and it["span"]["name"] == "census"
    ]
    (otlp,) = [
        sp
        for b in wires.otlp
        for rs in _wardex_native.codec.decode_otlp_traces(b)["resource_spans"]
        for ss in rs["scope_spans"]
        for sp in ss["spans"]
        if sp["name"] == "census"
    ]
    return env, otlp


def _extra(env: dict[str, Any]) -> dict[str, Any]:
    return {kv["key"]: kv["value"] for kv in env["extra"]}


def _rules(env: dict[str, Any]) -> list[str]:
    return env.get("capture_integrity", {}).get("redaction_rules", [])


@pytest.mark.parametrize("key", sorted(_COMPUTED_SAMPLES))
def test_a_computed_value_ships_as_written_and_unrecorded(key):
    value = _COMPUTED_SAMPLES[key]()
    # The card rule takes it anywhere else: under a host key of the same span
    # family, the neighbouring MCP server name.
    env, otlp = _ship({"wardex.openai_agents.mcp.server": value})
    assert _rules(env) == ["credit_card"]
    assert value not in repr(env) and value not in repr(otlp)
    env, otlp = _ship({key: value})
    assert _extra(env)[key] == value
    assert _rules(env) == []
    assert otlp["attributes"][key] == value
    assert not any(k.startswith("wardex.redact") for k in otlp["attributes"])


def test_no_rule_fires_on_any_literal_the_sdk_writes():
    pairs: set[tuple[str, str]] = set()
    for rel, key, value, origin in _SITES:
        if origin != LITERAL:
            continue
        name = _resolve(rel, key)
        texts = _LITERAL_SETS[(rel, value)]() if (rel, value) in _LITERAL_SETS else ()
        for text in texts or (_resolve(rel, value),):
            assert name and isinstance(text, str) and text, (rel, key, value)
            pairs.add((name, text))
    assert len(pairs) >= 10
    # Each under its own key, since a name rule reads the key.
    for key, text in sorted(pairs):
        env, otlp = _ship({key: text})
        assert _extra(env)[key] == text, (key, text, _extra(env)[key])
        assert _rules(env) == [], (key, text)
        assert otlp["attributes"][key] == text


def test_no_rule_fires_on_the_vocabularies_written_outside_set_extra():
    from wardex_sdk._assembly import Limitation
    from wardex_sdk._assembly._vocab import SpanIntent

    attrs = {
        "wardex.limitations": ",".join(m.value for m in Limitation),
        "gen_ai.operation.name": ",".join(sorted({i.operation.value for i in SpanIntent})),
    }
    env, otlp = _ship(attrs)
    assert _rules(env) == []
    assert {k: _extra(env)[k] for k in attrs} == attrs


def test_a_card_number_under_every_host_key_is_masked():
    keys = sorted(
        {k for rel, key, _, o in _SITES if o == HOST and (k := _resolve(rel, key)) is not None}
    )
    assert len(keys) >= 20
    env, otlp = _ship(dict.fromkeys(keys, _CARD))
    assert "credit_card" in _rules(env)
    assert {k: v for k, v in _extra(env).items() if v == _CARD} == {}
    assert _CARD not in repr(otlp)
