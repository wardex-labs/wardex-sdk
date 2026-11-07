"""The pydantic-ai attribute tables, read against pydantic-ai's own spans.

The fixtures under `fixtures/pydantic_ai/` are span dumps measured on
`pydantic-ai-slim` 2.66.0 (instrumentation schema version 5 unless the name
says otherwise) through an SDK `InMemorySpanExporter`, trimmed to the span's
name, kind, parent name, status code, attributes and exception events, with
long strings cut at 60 characters. Nothing here imports pydantic-ai, an
OpenTelemetry tracer or a wardex registry: the module under test is pure, and
that is what lets every rule be checked on the framework's real output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from wardex_sdk._adapters import _pydantic_ai_attrs as pa
from wardex_sdk._assembly import is_declared_extra_key
from wardex_sdk._enums import (
    AgentType,
    OperationName,
    ProviderName,
    StatusCode,
    ToolExecutionType,
    ToolType,
)

_DIR = Path(__file__).parent / "fixtures" / "pydantic_ai"
_SCENARIOS = sorted(p.stem for p in _DIR.glob("*.json"))


def _spans(scenario: str) -> list[dict[str, Any]]:
    return json.loads((_DIR / f"{scenario}.json").read_text(encoding="utf-8"))["spans"]


def _all_spans() -> list[tuple[str, dict[str, Any]]]:
    return [(s, span) for s in _SCENARIOS for span in _spans(s)]


def _span(scenario: str, name: str, index: int = 0) -> dict[str, Any]:
    return [s for s in _spans(scenario) if s["name"] == name][index]


def _error_type(span: dict[str, Any]) -> str | None:
    """What a surface delivers: the last exception event's type, or None."""
    found = None
    for event in span["events"]:
        if event["name"] == "exception":
            found = event["attributes"].get("exception.type")
    return found


def _outcome(span: dict[str, Any], **kw: bool) -> tuple[StatusCode, str | None]:
    attrs = span["attributes"]
    return pa.outcome(pa.classify(attrs), span["status"] == "ERROR", _error_type(span), attrs, **kw)


_EXPECTED_ROLE = {"invoke_agent": "run", "chat": "chat", "execute_tool": "tool"}


def test_the_fixtures_are_here():
    # A glob that matched nothing would turn every parametrized test below
    # into zero tests, which pytest reports as a pass.
    assert len(_SCENARIOS) == 16


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_every_measured_span_is_classified_by_its_operation_name(scenario):
    for span in _spans(scenario):
        assert pa.classify(span["attributes"]) == _EXPECTED_ROLE[span["name"].split(" ")[0]]


def test_the_name_is_never_what_classifies_a_span():
    # A user span named like pydantic's, and a validation-failure tool span
    # whose name is a tool's: the operation name decides both.
    assert pa.classify({"logfire.msg": "x"}) == "other"
    assert pa.classify({"gen_ai.operation.name": "execute_tool"}) == "tool"
    assert pa.classify({"gen_ai.operation.name": "decide"}) == "other"
    assert pa.classify({}) == "other"


def test_a_failed_attempt_span_is_recognised_by_its_attempt_index():
    # `PAI/_model_request_attempts.py` `record_attempt_span`: the model's
    # attributes plus the attempt index, and no operation name on purpose.
    attempt = {
        "gen_ai.provider.name": "openai",
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4o-mini",
        "server.address": "api.openai.com",
        "pydantic_ai.model_request.attempt": 0,
    }
    assert pa.classify(attempt) == "attempt"
    assert pa.unknown_keys("attempt", attempt) == []
    assert pa.extras_for("attempt", attempt) == [("wardex.pydantic_ai.model_request.attempt", 0)]
    assert pa.outcome("attempt", True, None, attempt) == (StatusCode.ERROR, "FailedAttempt")
    assert pa.outcome("attempt", True, "httpx.ConnectError", attempt) == (
        StatusCode.ERROR,
        "ConnectError",
    )


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_no_measured_key_is_unknown(scenario):
    """Every key pydantic wrote is copied, typed, recorded or knowingly dropped.

    The reverse of the drift signal: if this fails after a framework upgrade,
    a key appeared that no table names, and the adapter is counting it.
    """
    for span in _spans(scenario):
        role = pa.classify(span["attributes"])
        assert pa.unknown_keys(role, span["attributes"]) == [], span["name"]


def test_a_key_no_table_names_is_reported_as_unknown_and_never_copied():
    attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "m",
        "pydantic_ai.brand_new": "x",
        "gen_ai.request.top_k": 3,
    }
    assert pa.unknown_keys("chat", attrs) == ["gen_ai.request.top_k", "pydantic_ai.brand_new"]
    assert [k for k, _ in pa.extras_for("chat", attrs)] == []


def test_a_new_cache_key_is_unknown_rather_than_admitted_by_prefix():
    attrs = {"gen_ai.operation.name": "chat", "pydantic_ai.cache.prompt_text": "secret"}
    assert pa.unknown_keys("chat", attrs) == ["pydantic_ai.cache.prompt_text"]
    assert pa.extras_for("chat", attrs) == []


def test_an_other_span_reports_nothing_as_drift():
    # The span itself is dropped (and counted once, by the adapter); its keys
    # are the host's own, not pydantic's schema.
    assert pa.unknown_keys("other", {"anything": 1, "user.key": "v"}) == []
    assert pa.extras_for("other", {"gen_ai.usage.input_tokens": 3}) == []


@pytest.mark.parametrize("role", ["run", "chat", "tool", "attempt"])
def test_every_known_key_has_a_fate(role):
    for key in pa.KNOWN_KEYS[role]:
        assert pa.key_fate(role, key) != "unknown", key


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_every_extra_is_a_declared_wardex_key_with_a_scalar_value(scenario):
    for span in _spans(scenario):
        attrs = span["attributes"]
        for key, value in pa.extras_for(pa.classify(attrs), attrs):
            assert is_declared_extra_key(key), key
            assert key.startswith("wardex.pydantic_ai."), key
            assert isinstance(value, (str, int, float, bool)), (key, value)
        assert pa.rejected_keys(pa.classify(attrs), attrs) == []


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_usage_is_demoted_under_the_bridge_prefix_with_its_value_kept(scenario):
    """`gen_ai.usage.*` on chat and `gen_ai.aggregated_usage.*` on the run both
    move under `wardex.pydantic_ai.otel.`, verbatim; none keeps its own name, so
    no backend's `gen_ai.usage.` collector prices a call the wire already did."""
    for span in _spans(scenario):
        attrs = span["attributes"]
        extras = dict(pa.extras_for(pa.classify(attrs), attrs))
        usage = {
            k: v
            for k, v in attrs.items()
            if k.startswith(("gen_ai.usage.", "gen_ai.aggregated_usage."))
        }
        for key, value in usage.items():
            assert extras[f"wardex.pydantic_ai.otel.{key}"] == value
        assert not [k for k in extras if k.startswith("gen_ai.")]


def test_the_measured_usage_keys_are_the_ones_demoted():
    chat = _span("h_function_model_ids_usage", "chat function:fn:")["attributes"]
    run = _span("h_function_model_ids_usage", "invoke_agent ids")["attributes"]
    assert (
        dict(pa.extras_for("chat", chat)).items()
        >= {
            "wardex.pydantic_ai.otel.gen_ai.usage.input_tokens": 11,
            "wardex.pydantic_ai.otel.gen_ai.usage.output_tokens": 7,
            "wardex.pydantic_ai.otel.gen_ai.usage.cache_read.input_tokens": 3,
            "wardex.pydantic_ai.otel.gen_ai.usage.details.cache_read_tokens": 3,
        }.items()
    )
    assert (
        dict(pa.extras_for("run", run)).items()
        >= {
            "wardex.pydantic_ai.otel.gen_ai.aggregated_usage.input_tokens": 31,
            "wardex.pydantic_ai.otel.gen_ai.aggregated_usage.output_tokens": 8,
            "wardex.pydantic_ai.run_id": run["gen_ai.agent.call.id"],
        }.items()
    )


def test_run_usage_under_the_non_aggregated_names_is_demoted_too():
    # `use_aggregated_usage_attribute_names=False` puts the run total under
    # `gen_ai.usage.*` on `invoke_agent` (`PAI/models/instrumented.py`).
    run = {"gen_ai.operation.name": "invoke_agent", "gen_ai.usage.input_tokens": 9}
    assert pa.extras_for("run", run) == [("wardex.pydantic_ai.otel.gen_ai.usage.input_tokens", 9)]


def test_a_usage_value_that_is_not_a_count_is_rejected_not_written():
    attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.usage.input_tokens": True,
        "gen_ai.usage.output_tokens": "7",
        "pydantic_ai.cache.hit_ratio": [0.5],
        "operation.cost": 0.01,
    }
    assert pa.extras_for("chat", attrs) == [("wardex.pydantic_ai.operation.cost", 0.01)]
    assert pa.rejected_keys("chat", attrs) == [
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "pydantic_ai.cache.hit_ratio",
    ]


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_no_content_ever_reaches_an_extra(scenario):
    for span in _spans(scenario):
        attrs = span["attributes"]
        role = pa.classify(attrs)
        content_values = {v for k, v in attrs.items() if k in pa.CONTENT_KEYS}
        for key, value in pa.extras_for(role, attrs):
            assert value not in content_values, key
        for key in pa.CONTENT_KEYS & set(attrs):
            assert pa.key_fate(role, key) in ("content", "dropped"), key


def test_a_run_records_its_history_and_its_answer_as_content():
    run = _span("b_one_tool", "invoke_agent tooly")["attributes"]
    assert pa.content_for("run", run) == (run["pydantic_ai.all_messages"], run["final_result"])


def test_a_tool_records_its_arguments_and_its_result_as_content():
    tool = _span("h_function_model_ids_usage", "execute_tool add")["attributes"]
    assert pa.content_for("tool", tool) == ('{"x":1,"y":2}', "3")
    # At START only the arguments exist (`gen_ai.tool.call.arguments` is a
    # start attribute; the result is set at end).
    start = {k: v for k, v in tool.items() if k != "gen_ai.tool.call.result"}
    assert pa.content_for("tool", start) == ('{"x":1,"y":2}', None)


def test_a_chat_records_no_content():
    # The wire span under the chat unit carries the bodies; the run carries the
    # history once. The chat's own message JSON is O(history) per request.
    chat = _span("h_function_model_ids_usage", "chat function:fn:")["attributes"]
    assert "gen_ai.input.messages" in chat
    assert pa.content_for("chat", chat) == (None, None)


def test_content_switched_off_records_nothing():
    run = _span("d3_tool_runtime_error_include_content_false", "invoke_agent crashing_nocontent")
    tool = _span("d3_tool_runtime_error_include_content_false", "execute_tool boom2")
    assert pa.content_for("run", run["attributes"])[1] is None
    assert pa.content_for("tool", tool["attributes"]) == (None, None)


def test_the_chat_block_reads_the_request_and_response_and_never_usage():
    chat = _span("h_function_model_ids_usage", "chat function:fn:")["attributes"]
    block = pa.chat_block(chat, subject="ignored")
    assert block.operation is OperationName.CHAT
    assert block.provider == "function"
    assert block.request_model == "function:fn:"
    assert block.response_model == "function:fn:"
    assert block.response_id == "resp_123"
    assert block.finish_reasons == ("tool_call",)
    assert block.max_tokens == 50
    assert block.temperature == 0.2
    assert (block.input_tokens, block.output_tokens, block.cache_read_input_tokens) == (
        None,
        None,
        None,
    )


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_no_chat_block_carries_a_token_count(scenario):
    for span in _spans(scenario):
        if pa.classify(span["attributes"]) != "chat":
            continue
        block = pa.chat_block(span["attributes"], subject="m")
        assert block.input_tokens is None
        assert block.output_tokens is None
        assert block.cache_read_input_tokens is None
        assert block.cache_creation_input_tokens is None
        assert block.reasoning_output_tokens is None


def test_the_chat_block_falls_back_to_the_subject_before_the_model_is_named():
    # At open, a chat span carries only the operation name and the baggage
    # keys; `gen_ai.request.model` arrives by `set_attributes` afterwards.
    block = pa.chat_block({"gen_ai.operation.name": "chat"}, subject="gpt-4o-mini")
    assert block.request_model == "gpt-4o-mini"
    assert block.provider is None


def test_a_known_provider_becomes_the_enum_member_and_any_other_stays_as_written():
    assert pa.chat_block({"gen_ai.provider.name": "openai"}, subject="m").provider is (
        ProviderName.OPENAI
    )
    assert pa.chat_block({"gen_ai.provider.name": "google-gla"}, subject="m").provider == (
        "google-gla"
    )


def test_the_chat_block_reads_streaming_ttft_and_frozen_sequences():
    chat = _span("c1_run_stream", "chat test")["attributes"]
    assert isinstance(pa.chat_block(chat, subject="m").time_to_first_chunk_s, float)
    # An SDK span freezes a list attribute into a tuple; both read the same.
    frozen = {"gen_ai.response.finish_reasons": ("stop",), "gen_ai.request.seed": True}
    block = pa.chat_block(frozen, subject="m")
    assert block.finish_reasons == ("stop",)
    assert block.seed is None  # a bool is not a seed


def test_the_server_is_read_from_the_model_base_url_attributes():
    assert pa.server_of({"server.address": "api.openai.com", "server.port": 443}) == (
        "api.openai.com",
        443,
    )
    assert pa.server_of({"server.port": "443"}) == (None, None)


def test_the_tool_block_carries_the_call_id_or_none_never_a_made_up_one():
    tool = _span("h_function_model_ids_usage", "execute_tool add")["attributes"]
    block = pa.tool_block(tool)
    assert (block.name, block.call_id) == ("add", "call_abc")
    assert block.type is ToolType.FUNCTION
    assert block.execution_type is ToolExecutionType.IN_PROCESS
    output_function = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "final_result"}
    assert pa.tool_block(output_function).call_id is None
    assert pa.tool_block({}, fallback_name="from_span_name").name == "from_span_name"


def test_the_agent_block_names_a_nested_run_a_sub_agent_of_its_parent():
    outer = _span("f_nested_agents", "invoke_agent outer")["attributes"]
    inner = _span("f_nested_agents", "invoke_agent inner")["attributes"]
    top = pa.agent_block(outer, nested=False, parent_agent="ignored")
    assert (top.name, top.agent_type, top.parent_agent) == ("outer", AgentType.PRIMARY, None)
    sub = pa.agent_block(inner, nested=True, parent_agent="outer")
    assert (sub.name, sub.agent_type, sub.parent_agent) == ("inner", AgentType.SUB_AGENT, "outer")


@pytest.mark.parametrize(
    ("scenario", "name", "expected"),
    [
        ("e1_call_deferred", "execute_tool later", (StatusCode.UNSET, None)),
        ("e2_approval_required", "execute_tool dangerous", (StatusCode.UNSET, None)),
        # Schema versions 3-4 spell a deferral as ERROR + an exception event.
        ("e4_call_deferred_version4", "execute_tool later4", (StatusCode.UNSET, None)),
        ("d1_tool_model_retry", "execute_tool flaky", (StatusCode.ERROR, "ToolRetryError")),
        ("d2_tool_runtime_error", "execute_tool boom", (StatusCode.ERROR, "RuntimeError")),
        ("d2_tool_runtime_error", "invoke_agent crashing", (StatusCode.ERROR, "RuntimeError")),
        ("l_cancel_mid_tool", "execute_tool slow", (StatusCode.ERROR, "CancelledError")),
        ("b_one_tool", "execute_tool add", (StatusCode.OK, None)),
        ("b_one_tool", "chat test", (StatusCode.OK, None)),
        ("b_one_tool", "invoke_agent tooly", (StatusCode.OK, None)),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_the_outcome_of_each_measured_ending(scenario, name, expected):
    assert _outcome(_span(scenario, name)) == expected


def test_the_retried_call_succeeds_on_its_second_attempt():
    assert _outcome(_span("d1_tool_model_retry", "execute_tool flaky", 1)) == (StatusCode.OK, None)


def test_a_run_whose_tool_deferred_is_unset_not_ok():
    run = _span("e1_call_deferred", "invoke_agent deferring")
    assert _outcome(run) == (StatusCode.OK, None)
    assert _outcome(run, deferred_child=True) == (StatusCode.UNSET, None)


def test_a_run_ended_while_a_child_was_live_is_unset():
    # pydantic leaves a cancelled run UNSET; the adapter holds its close
    # until the child ends and then says it did not finish.
    run = _span("l_cancel_mid_tool", "invoke_agent cancelled")
    assert _outcome(run, held=True) == (StatusCode.UNSET, None)


def test_an_error_beats_a_deferral_on_the_run_and_has_a_type_even_when_unstated():
    attrs = {"gen_ai.operation.name": "invoke_agent"}
    assert pa.outcome("run", True, None, attrs, deferred_child=True) == (StatusCode.ERROR, "error")
    assert pa.outcome("other", True, "X", {}) == (StatusCode.UNSET, None)


def test_the_deferral_is_kept_as_an_extra():
    tool = _span("e2_approval_required", "execute_tool dangerous")["attributes"]
    assert ("wardex.pydantic_ai.tool.deferral", "ApprovalRequired") in pa.extras_for("tool", tool)


def test_the_retry_attributes_are_kept_as_extras():
    tool = _span("d1_tool_model_retry", "execute_tool flaky")["attributes"]
    assert dict(pa.extras_for("tool", tool)) == {
        "wardex.pydantic_ai.tool.retry.cause": "model_retry",
        "wardex.pydantic_ai.tool.retry.attempt": 1,
        "wardex.pydantic_ai.tool.retry.max_retries": 1,
    }


def test_schema_versions_5_and_6_read_identically():
    """Measured: the two versions differ only in message roles inside the
    content JSON, which this module never parses. Ids differ per run."""
    v5, v6 = _spans("g_version_5"), _spans("g_version_6")
    assert len(v5) == len(v6)
    ids = {"wardex.pydantic_ai.run_id"}
    for a, b in zip(v5, v6, strict=True):
        ra, rb = pa.classify(a["attributes"]), pa.classify(b["attributes"])
        assert ra == rb
        assert set(a["attributes"]) == set(b["attributes"])
        assert {k: pa.key_fate(ra, k) for k in a["attributes"]} == {
            k: pa.key_fate(rb, k) for k in b["attributes"]
        }
        ea = {k: v for k, v in pa.extras_for(ra, a["attributes"]) if k not in ids}
        eb = {k: v for k, v in pa.extras_for(rb, b["attributes"]) if k not in ids}
        assert ea == eb
        assert _outcome(a) == _outcome(b)
        if ra == "chat":
            assert pa.chat_block(a["attributes"], subject="m") == pa.chat_block(
                b["attributes"], subject="m"
            )


def test_every_fixture_key_is_accounted_for_once():
    """A census over every measured span: each key has exactly one fate, and
    the union of fates over a role is a subset of what that role knows plus
    the prefix rules (usage, `logfire.*`)."""
    for _, span in _all_spans():
        attrs = span["attributes"]
        role = pa.classify(attrs)
        for key in attrs:
            fate = pa.key_fate(role, key)
            assert fate in ("typed", "extra", "content", "dropped"), (span["name"], key)
            if key not in pa.KNOWN_KEYS[role]:
                assert key.startswith(("gen_ai.usage.", "gen_ai.aggregated_usage.", "logfire."))


def test_an_operation_name_that_is_not_a_string_classifies_as_other_without_raising():
    """An attribute value may be a sequence, which is unhashable: the table lookup is skipped."""
    assert pa.classify({"gen_ai.operation.name": ["chat"]}) == "other"
    assert pa.classify({"gen_ai.operation.name": 3}) == "other"


def test_a_deferral_spelled_only_by_its_exception_type_is_control_flow():
    """Schema versions 3-4: an ERROR tool span whose exception event names the deferral type,
    with no deferral attribute. Not a failure: the tool closes UNSET with no error type."""
    for raised in ("CallDeferred", "pydantic_ai.exceptions.ApprovalRequired"):
        assert pa.is_deferral({}, raised) is True
        assert pa.outcome("tool", True, raised, {}) == (StatusCode.UNSET, None)
    assert pa.outcome("tool", True, "ValueError", {}) == (StatusCode.ERROR, "ValueError")


def test_the_websocket_transport_attribute_is_a_named_extra_not_drift():
    """Written on the chat span when the user opts into `openai_transport='websocket'`."""
    attrs = {"gen_ai.operation.name": "chat", "pydantic_ai.transport": "websocket"}
    assert pa.key_fate("chat", "pydantic_ai.transport") == "extra"
    assert list(pa.unknown_keys("chat", attrs)) == []
    assert pa.extras_for("chat", attrs) == [("wardex.pydantic_ai.transport", "websocket")]
    assert is_declared_extra_key("wardex.pydantic_ai.transport")


def test_a_call_whose_run_was_cut_is_unset_unless_it_failed():
    """`held` on a chat or tool: its run already ended (cancelled) while the call was in flight.
    pydantic leaves such a model call UNSET with no exception; OK would claim an answer."""
    assert pa.outcome("chat", False, None, {}, held=True) == (StatusCode.UNSET, None)
    assert pa.outcome("tool", False, None, {}, held=True) == (StatusCode.UNSET, None)
    assert pa.outcome("tool", True, "CancelledError", {}, held=True) == (
        StatusCode.ERROR,
        "CancelledError",
    )
    assert pa.outcome("chat", False, None, {}) == (StatusCode.OK, None)
