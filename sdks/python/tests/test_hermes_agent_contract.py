"""Hermes's plugin contract as the adapter reads it: names, schema versions, payload fields.

The payload samples are written here by hand. They carry the field names and id shapes Hermes
sends, and no content from a real run.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from wardex_sdk._adapters._hermes_agent_events import (
    CONTRACT,
    KNOWN_SCHEMAS,
    NOT_REGISTERED,
    AuxEnd,
    AuxStart,
    CallEnter,
    CallError,
    CallReport,
    ChildStart,
    ChildStop,
    ContractRow,
    Drift,
    DriftNote,
    Finalize,
    KeyKind,
    Surface,
    ToolEnter,
    ToolReport,
    TurnAnswer,
    TurnEnd,
    TurnStart,
    parse,
)
from wardex_sdk._adapters._hermes_agent_turns import CloseCause, Trigger

_NAMES = json.loads(
    (Path(__file__).parent / "fixtures" / "hermes" / "contract_names.json").read_text()
)
_ROWS = {r.name: r for r in CONTRACT}
_OBS = {"telemetry_schema_version": "hermes.observer.v1"}
_MW = {**_OBS, "middleware_schema_version": "hermes.middleware.v1"}
_USAGE = {
    "input_tokens": 120,
    "output_tokens": 8,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0,
    "reasoning_tokens": 0,
    "prompt_tokens": 120,
    "total_tokens": 128,
    "request_count": 1,
    "cache_write_by_ttl": {},
}
_CORE_USAGE = (
    ("gen_ai.usage.input_tokens", 120),
    ("gen_ai.usage.output_tokens", 8),
    ("gen_ai.usage.cache_read.input_tokens", 0),
    ("gen_ai.usage.cache_creation.input_tokens", 0),
    ("gen_ai.usage.reasoning.output_tokens", 0),
)
_TOOL_CALLS = SimpleNamespace(
    tool_calls=[SimpleNamespace(id="call_1", name="read_file", arguments='{"path": "x"}')]
)

#: One complete payload per row (field names as Hermes sends them) and the record it must give.
_SAMPLES: dict[str, tuple[dict[str, Any], object]] = {
    "pre_llm_call": (
        {**_OBS, "session_id": "s-1", "turn_id": "t-1", "platform": "cli", "is_first_turn": True},
        TurnStart("s-1", "t-1", platform="cli"),
    ),
    "llm_execution": (
        {
            **_MW,
            "session_id": "s-1",
            "turn_id": "t-1",
            "api_request_id": "s-1:api:1",
            "model": "lab-model",
            "api_mode": "chat_completions",
            "api_call_count": 1,
        },
        CallEnter("s-1", "t-1", "s-1:api:1", "lab-model", "chat_completions"),
    ),
    "post_api_request": (
        {
            **_OBS,
            "session_id": "s-1",
            "turn_id": "t-1",
            "api_request_id": "s-1:api:1",
            "usage": _USAGE,
            "assistant_message": _TOOL_CALLS,
            "assistant_tool_call_count": 1,
        },
        CallReport("s-1", "t-1", "s-1:api:1", _CORE_USAGE, (("call_1", "read_file"),)),
    ),
    "api_request_error": (
        {
            **_OBS,
            "session_id": "s-1",
            "turn_id": "t-1",
            "api_request_id": "s-1:api:1",
            "retry_count": 0,
            "max_retries": 1,
            "retryable": True,
            "reason": "server_error",
            "status_code": 500,
        },
        CallError("s-1", "t-1", "s-1:api:1", "server_error"),
    ),
    "tool_execution": (
        {
            **_MW,
            "session_id": "s-1",
            "turn_id": "t-1",
            "api_request_id": "s-1:api:1",
            "tool_call_id": "call_1",
            "tool_name": "read_file",
            "args": {"path": "x"},
        },
        ToolEnter(
            "read_file",
            session_id="s-1",
            turn_id="t-1",
            api_request_id="s-1:api:1",
            tool_call_id="call_1",
            args={"path": "x"},
        ),
    ),
    "post_tool_call": (
        {
            **_OBS,
            "session_id": "s-1",
            "turn_id": "t-1",
            "tool_call_id": "call_1",
            "tool_name": "read_file",
            "status": "ok",
            "error_type": None,
            "result": "r",
            "duration_ms": 3,
        },
        ToolReport("s-1", "t-1", "call_1", "ok", None, "r"),
    ),
    "pre_auxiliary_call": (
        {
            **_OBS,
            "session_id": "s-1",
            "turn_id": "t-1",
            "api_request_id": "aux-0f3a",
            "retry_count": 0,
            "aux_task": "title_generation",
            "model": "lab-aux-title",
        },
        AuxStart("s-1", "aux-0f3a", 0, "t-1", "title_generation", "lab-aux-title"),
    ),
    "post_auxiliary_call": (
        {
            **_OBS,
            "session_id": "s-1",
            "turn_id": "t-1",
            "api_request_id": "aux-0f3a",
            "retry_count": 0,
            "error_type": "BadRequestError",
            "usage": _USAGE,
        },
        AuxEnd("s-1", "aux-0f3a", 0, "t-1", "BadRequestError", _CORE_USAGE),
    ),
    "subagent_start": (
        {
            **_OBS,
            "parent_session_id": "s-1",
            "parent_turn_id": "t-1",
            "child_session_id": "s-2",
            "child_subagent_id": "sa-1",
            "child_role": "leaf",
            "parent_subagent_id": None,
        },
        ChildStart("s-1", "t-1", "s-2", "sa-1", "leaf"),
    ),
    "subagent_stop": (
        {**_OBS, "child_session_id": "s-2", "child_status": "completed", "duration_ms": 9},
        ChildStop("s-2"),
    ),
    "post_llm_call": (
        {
            **_OBS,
            "session_id": "s-1",
            "task_id": "task-1",
            "turn_id": "t-1",
            "model": "lab-model",
            "platform": "cli",
            "side_question_token": None,
            "review_signals": None,
        },
        TurnAnswer("s-1", "t-1"),
    ),
    "on_session_end": (
        {
            **_OBS,
            "session_id": "s-1",
            "turn_id": "t-1",
            "completed": True,
            "failed": False,
            "interrupted": False,
            "turn_exit_reason": "text_response(finish_reason=stop)",
        },
        TurnEnd("s-1", "t-1", True, False, False, "text_response(finish_reason=stop)"),
    ),
    "on_session_finalize": (
        {"session_id": "s-1", "reason": "shutdown"},
        Finalize("s-1", "shutdown"),
    ),
}

#: Fields Hermes sends that hold model content, end-user identity or the chain's own callable.
_NEVER_READ = {
    "request": {"messages": [{"content": "SECRET"}]},
    "original_request": {"messages": []},
    "messages": ["SECRET"],
    "conversation_history": ["SECRET"],
    "response": {"assistant_message": {"content": "SECRET"}},
    "system_prompt": "SECRET",
    "sender_id": "SECRET",
    "child_goal": "SECRET",
    "child_summary": "SECRET",
    "error_message": "SECRET",
}


# -- names -------------------------------------------------------------------------------------


def test_every_hermes_name_is_registered_or_declined_with_a_reason() -> None:
    registered = {r.name for r in CONTRACT}
    hermes = set(_NAMES["valid_hooks"]) | set(_NAMES["valid_middleware"])
    assert registered | set(NOT_REGISTERED) == hermes
    assert not registered & set(NOT_REGISTERED)
    assert len(registered) == len(CONTRACT)
    assert all(reason.strip() for reason in NOT_REGISTERED.values())


def test_middleware_rows_are_execution_kinds_only() -> None:
    middleware = {r.name for r in CONTRACT if r.surface is Surface.MIDDLEWARE}
    assert middleware == {"llm_execution", "tool_execution"}
    assert middleware <= set(_NAMES["valid_middleware"])


def test_recorded_schema_versions_are_the_known_ones() -> None:
    assert dict(KNOWN_SCHEMAS) == _NAMES["schema_versions"]
    assert {r.schema_field for r in CONTRACT} - {None} <= set(KNOWN_SCHEMAS)


def test_every_row_has_a_sample() -> None:
    assert set(_SAMPLES) == set(_ROWS)


# -- parse -------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(_SAMPLES))
def test_a_complete_payload_gives_its_record(name: str) -> None:
    payload, expected = _SAMPLES[name]
    assert parse(_ROWS[name], payload) == (expected, None)


def _required(row: ContractRow) -> list[str]:
    return [f.payload for f in row.record.FIELDS if f.required]


@pytest.mark.parametrize(
    ("name", "field"),
    [(r.name, f) for r in CONTRACT for f in _required(r)],
    ids=lambda v: v,
)
def test_a_missing_required_field_drops_the_event_and_names_the_field(
    name: str, field: str
) -> None:
    payload = {k: v for k, v in _SAMPLES[name][0].items() if k != field}
    assert parse(_ROWS[name], payload) == (None, DriftNote(Drift.FIELD, name, f"missing {field}"))


def test_a_mistyped_field_counts_as_missing() -> None:
    payload = {**_SAMPLES["api_request_error"][0], "turn_id": 7, "api_request_id": None}
    event, drift = parse(_ROWS["api_request_error"], payload)
    assert event is None
    assert drift == DriftNote(Drift.FIELD, "api_request_error", "missing turn_id, api_request_id")


@pytest.mark.parametrize("name", [r.name for r in CONTRACT if r.schema_field is not None])
def test_an_unknown_schema_version_keeps_the_event_and_says_so(name: str) -> None:
    row = _ROWS[name]
    payload, expected = _SAMPLES[name]
    assert row.schema_field is not None
    stated = {**payload, row.schema_field: "hermes.v9"}
    detail = f"{row.schema_field}='hermes.v9'"
    assert parse(row, stated) == (expected, DriftNote(Drift.SCHEMA, name, detail))
    del stated[row.schema_field]
    assert parse(row, stated)[1] == DriftNote(Drift.SCHEMA, name, f"{row.schema_field}=None")


def test_a_field_drift_wins_over_a_schema_drift() -> None:
    payload = {**_SAMPLES["pre_llm_call"][0], "telemetry_schema_version": "hermes.v9"}
    del payload["turn_id"]
    assert parse(_ROWS["pre_llm_call"], payload) == (
        None,
        DriftNote(Drift.FIELD, "pre_llm_call", "missing turn_id"),
    )


def _all_values(record: object) -> list[object]:
    out: list[object] = []
    pending = [getattr(record, f.name) for f in dataclasses.fields(record)]  # type: ignore[arg-type]
    while pending:
        v = pending.pop()
        out.append(v)
        if isinstance(v, (tuple, list)):
            pending.extend(v)
        elif isinstance(v, dict):
            pending.extend(v.values())
    return out


@pytest.mark.parametrize("name", list(_SAMPLES))
def test_content_identity_and_next_call_never_reach_a_record(name: str) -> None:
    calls: list[object] = []
    payload = {**_SAMPLES[name][0], **_NEVER_READ, "next_call": lambda *a: calls.append(a)}
    event, drift = parse(_ROWS[name], payload)
    assert drift is None
    assert event is not None
    names = {f.name for f in dataclasses.fields(event)}  # type: ignore[arg-type]
    assert not names & (set(_NEVER_READ) | {"next_call"})
    assert "SECRET" not in _all_values(event)
    assert calls == []


#: The turn's own input and output: read into one record each, for the payload path only.
_TURN_IO = {"user_message": "SECRET", "assistant_response": "SECRET"}
_TURN_IO_READ_BY = {"pre_llm_call": "user_message", "post_llm_call": "assistant_response"}


@pytest.mark.parametrize("name", list(_SAMPLES))
def test_the_turn_input_and_output_reach_only_their_own_records(name: str) -> None:
    event, drift = parse(_ROWS[name], {**_SAMPLES[name][0], **_NEVER_READ, **_TURN_IO})
    assert drift is None
    assert event is not None
    held = [f.name for f in dataclasses.fields(event) if getattr(event, f.name) == "SECRET"]  # type: ignore[arg-type]
    assert held == ([_TURN_IO_READ_BY[name]] if name in _TURN_IO_READ_BY else [])


def test_the_turn_input_and_output_are_carried_as_hermes_sent_them() -> None:
    parts = [{"type": "text", "text": "Describe this."}, {"type": "image_url", "image_url": {}}]
    start, _ = parse(_ROWS["pre_llm_call"], {**_SAMPLES["pre_llm_call"][0], "user_message": parts})
    assert isinstance(start, TurnStart)
    assert start.user_message is parts
    payload = {**_SAMPLES["post_llm_call"][0], "assistant_response": "Hello."}
    assert parse(_ROWS["post_llm_call"], payload) == (TurnAnswer("s-1", "t-1", "Hello."), None)
    absent = {**payload, "assistant_response": None}
    assert parse(_ROWS["post_llm_call"], absent) == (TurnAnswer("s-1", "t-1"), None)


def test_requested_tool_calls_keep_ids_and_names_only() -> None:
    row = _ROWS["post_api_request"]
    base = _SAMPLES["post_api_request"][0]
    as_mapping = {"tool_calls": [{"id": "call_1", "name": "read_file", "arguments": "{}"}]}
    for message in (_TOOL_CALLS, as_mapping):
        event, _ = parse(row, {**base, "assistant_message": message})
        assert isinstance(event, CallReport)
        assert event.requested == (("call_1", "read_file"),)
    broken = SimpleNamespace(tool_calls=[SimpleNamespace(id="", name="x"), SimpleNamespace(id="c")])
    for message in (broken, SimpleNamespace(tool_calls=None), SimpleNamespace()):
        event, _ = parse(row, {**base, "assistant_message": message})
        assert isinstance(event, CallReport)
        assert event.requested == ()
    event, _ = parse(row, {k: v for k, v in base.items() if k != "assistant_message"})
    assert isinstance(event, CallReport)
    assert event.requested == ()


def test_usage_keeps_integer_counts_of_known_keys() -> None:
    row = _ROWS["post_api_request"]
    base = _SAMPLES["post_api_request"][0]
    usage = {"prompt_tokens": 5, "output_tokens": True, "reasoning_tokens": "3", "other": 1}
    event, _ = parse(row, {**base, "usage": usage})
    assert isinstance(event, CallReport)
    assert event.usage == (("gen_ai.usage.input_tokens", 5),)
    event, _ = parse(row, {**base, "usage": "not a mapping"})
    assert isinstance(event, CallReport)
    assert event.usage == ()


def test_the_input_count_is_the_prompt_total_with_its_cached_part() -> None:
    """Hermes's `input_tokens` is the uncached part of the prompt; wardex's input count includes
    the cache, as every other source's does, so it is Hermes's `prompt_tokens`."""
    row = _ROWS["post_api_request"]
    base = _SAMPLES["post_api_request"][0]
    usage = {
        "input_tokens": 700,
        "cache_read_tokens": 300,
        "cache_write_tokens": 200,
        "prompt_tokens": 1200,
    }
    event, _ = parse(row, {**base, "usage": usage})
    assert isinstance(event, CallReport)
    assert dict(event.usage)["gen_ai.usage.input_tokens"] == 1200


# -- what an event routes by -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("retry_count", "max_retries", "retryable"),
    [(0, 3, True), (2, 3, True), (0, 1, True), (0, 3, False)],
    ids=["retries-left", "last-retry", "one-attempt", "not-retryable"],
)
def test_a_failed_call_never_predicts_the_end_of_its_turn(
    retry_count: int, max_retries: int, retryable: bool
) -> None:
    # After the last retry Hermes falls back or parks the turn on its recovery ladder.
    payload = {
        **_SAMPLES["api_request_error"][0],
        "retry_count": retry_count,
        "max_retries": max_retries,
        "retryable": retryable,
    }
    event, drift = parse(_ROWS["api_request_error"], payload)
    assert drift is None
    assert isinstance(event, CallError)
    assert event.trigger() is Trigger.CALL_FAILED


@pytest.mark.parametrize(
    ("reason", "trigger"),
    [
        ("shutdown", Trigger.SHUTDOWN),
        (None, Trigger.SHUTDOWN),
        ("new_session", Trigger.FINALIZE),
        ("session_boundary", Trigger.FINALIZE),
        ("cli_close", Trigger.FINALIZE),
    ],
    ids=["shutdown", "no-reason", "gateway-new", "cli-new", "viewer-close"],
)
def test_only_a_shutdown_finalize_is_a_shutdown(reason: str | None, trigger: Trigger) -> None:
    assert Finalize("s-1", reason).trigger() is trigger


@pytest.mark.parametrize(
    ("status", "error_type", "abandoned"),
    [
        ("timeout", "tool_timeout", True),
        ("cancelled", "tool_interrupted", True),
        ("cancelled", "keyboard_interrupt", True),
        ("error", "thread_missing_result", True),
        ("ok", None, False),
        ("error", "FileNotFoundError", False),
    ],
    ids=["timeout", "interrupt", "keyboard", "no-result", "ok", "failed"],
)
def test_a_tool_report_says_when_hermes_gave_up_on_the_worker(
    status: str, error_type: str | None, abandoned: bool
) -> None:
    assert ToolReport("s-1", "t-1", "call_1", status, error_type).abandoned() is abandoned


@pytest.mark.parametrize(
    ("exit_reason", "blocked"),
    [
        ("blocked_by_plugin_pre_llm_call", True),
        ("text_response(finish_reason=stop)", False),
        (None, False),
    ],
    ids=["block", "answer", "none"],
)
def test_an_end_says_when_another_plugin_blocked_the_prompt(
    exit_reason: str | None, blocked: bool
) -> None:
    assert TurnEnd("s-1", "t-1", True, exit_reason=exit_reason).blocked() is blocked


@pytest.mark.parametrize(
    ("flags", "cause"),
    [
        ((True, False, False), CloseCause.ENDED_COMPLETED),
        ((False, True, False), CloseCause.ENDED_FAILED),
        ((False, False, True), CloseCause.ENDED_INTERRUPTED),
        ((False, False, False), CloseCause.ENDED_INCOMPLETE),
        ((True, True, False), CloseCause.ENDED_FAILED),
        ((True, False, True), CloseCause.ENDED_INTERRUPTED),
        ((False, True, True), CloseCause.ENDED_FAILED),
    ],
    ids=[
        "completed",
        "failed",
        "interrupted",
        "none",
        "completed+failed",
        "completed+interrupted",
        "failed+interrupted",
    ],
)
def test_the_end_cause_comes_from_hermes_flags(
    flags: tuple[bool, bool, bool], cause: CloseCause
) -> None:
    completed, failed, interrupted = flags
    end = TurnEnd("s-1", "t-1", completed, failed, interrupted, "text_response(finish_reason=stop)")
    assert end.cause() is cause


def test_an_end_without_a_turn_applies_to_the_session() -> None:
    payload = {**_OBS, "session_id": "s-1", "completed": False, "interrupted": True}
    event, drift = parse(_ROWS["on_session_end"], payload)
    assert drift is None
    assert isinstance(event, TurnEnd)
    assert (event.key_kind(), event.key()) == (KeyKind.SESSION, "s-1")
    assert event.cause() is CloseCause.ENDED_INTERRUPTED


def test_an_auxiliary_call_outside_any_turn_keys_no_turn() -> None:
    payload = {k: v for k, v in _SAMPLES["pre_auxiliary_call"][0].items() if k != "turn_id"}
    event, drift = parse(_ROWS["pre_auxiliary_call"], payload)
    assert drift is None
    assert isinstance(event, AuxStart)
    assert event.key() is None
    assert event.attempt() == ("aux-0f3a", 0)


@pytest.mark.parametrize("name", list(_SAMPLES))
def test_every_event_routes_by_its_own_key(name: str) -> None:
    event = _SAMPLES[name][1]
    expected = {
        "subagent_start": (KeyKind.PARENT_TURN, "t-1"),
        "subagent_stop": (KeyKind.SESSION, "s-2"),
        "on_session_finalize": (KeyKind.SESSION, "s-1"),
    }.get(name, (KeyKind.TURN, "t-1"))
    assert (event.key_kind(), event.key()) == expected  # type: ignore[attr-defined]
