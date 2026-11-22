"""`wardex_sdk.hermes_agent.register(ctx)`: what Hermes gets registered, in which order wardex
starts, and what a second call, a second context or a missing destination change.

Every test drives a `FakePluginContext` the way Hermes drives a plugin context, and asserts what a
user sees: the registrations Hermes holds, the spans exported, the lines wardex says.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import wardex_sdk
from _hermes_fake import FakePluginContext
from _hermes_scripted import FakeHermes
from _hermes_support import (
    exported,
    forget_attachments,
    hermes,
    hermes_installed,
    on_loop,
    plugin,
    received,
    start_wardex,
    status,
)
from wardex_sdk._adapters._hermes_agent_attach import HOST_ISOLATION_LINE
from wardex_sdk._adapters._hermes_agent_events import CONTRACT, NOT_REGISTERED
from wardex_sdk._enums import AdapterState
from wardex_sdk.hermes_agent import HOOKS, MIDDLEWARE, register
from wardex_sdk.testing import RecordingTransport


@pytest.fixture(autouse=True)
def _no_contexts_carried_over():
    forget_attachments()
    yield
    forget_attachments()


@pytest.fixture
def installed(tmp_path: Path):
    with hermes_installed(tmp_path):
        yield


def _one_turn(h: FakeHermes) -> None:
    turn = h.turn()
    h.model_call(turn)
    h.end(turn)


def _turns(nodes) -> list[str]:
    return [n.name for n in nodes if n.name.startswith("invoke_agent")]


def test_hooks_and_middleware_are_every_contract_row_and_no_declined_name():
    assert sorted(HOOKS + MIDDLEWARE) == sorted(r.name for r in CONTRACT)
    assert set(MIDDLEWARE) == {"llm_execution", "tool_execution"}
    assert not set(HOOKS + MIDDLEWARE) & set(NOT_REGISTERED)


def test_register_registers_one_callback_per_contract_row_and_nothing_else(installed):
    start_wardex()
    with plugin() as ctx:
        assert list(ctx.hooks) == list(HOOKS), "registered in the order HOOKS lists"
        assert {k: len(v) for k, v in ctx.hooks.items()} == dict.fromkeys(HOOKS, 1)
        assert {k: len(v) for k, v in ctx.middleware.items()} == dict.fromkeys(MIDDLEWARE, 1)
        assert ctx.warnings == []
        assert status().state is AdapterState.INSTALLED


def test_registering_twice_on_one_context_registers_once(installed):
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        register(ctx)
        assert {len(v) for v in [*ctx.hooks.values(), *ctx.middleware.values()]} == {1}
        on_loop(_one_turn, h)
    assert _turns(exported(transport)) == ["invoke_agent hermes"]


def test_two_contexts_on_one_plugin_manager_give_one_set_of_spans(installed):
    """Two contexts that share Hermes's hook and middleware tables (a plugin loaded twice into
    one manager) register twice, and the turn still has one span per thing that happened."""
    transport = start_wardex()
    with plugin() as first:
        second = FakePluginContext(hooks=first.hooks, middleware=first.middleware)
        register(second)
        assert {len(v) for v in [*first.hooks.values(), *first.middleware.values()]} == {2}
        with hermes(first) as h:
            on_loop(_one_turn, h)
        second.unload()
        assert first.failures == []
    names = sorted(n.name for n in exported(transport))
    assert names.count("invoke_agent hermes") == 1
    assert names.count("chat lab-model") == 1


def test_two_profile_homes_get_one_registration_set_each_and_one_init(
    installed, monkeypatch, tmp_path
):
    transport = RecordingTransport()
    real_init = wardex_sdk.init
    calls: list[tuple] = []

    def init(*args, **kwargs):
        calls.append((args, kwargs))
        real_init(transport=transport)

    monkeypatch.setattr(wardex_sdk, "init", init)
    with plugin(home=tmp_path / "a") as a, plugin(home=tmp_path / "b") as b:
        assert calls == [((), {})]
        assert all(len(a.hooks[n]) == 1 and len(b.hooks[n]) == 1 for n in HOOKS)
        with hermes(a) as ha, hermes(b) as hb:
            on_loop(_one_turn, ha)
            on_loop(_one_turn, hb)
    nodes = exported(transport)
    roots = [n for n in nodes if n.name == "invoke_agent hermes"]
    assert len({n.trace_id for n in roots}) == 2
    assert not [k for n in nodes for k in n.attributes if "profile" in k]


def test_in_the_plugin_host_process_nothing_is_registered_and_one_line_is_said(monkeypatch, caplog):
    monkeypatch.setenv("HERMES_PLUGIN_HOST_PROCESS", "1")
    ctx = FakePluginContext()
    with caplog.at_level(logging.WARNING, logger="wardex_sdk"):
        register(ctx)
        register(FakePluginContext())
    assert ctx.hooks == {} and ctx.middleware == {} and ctx.leases == []
    assert [r.getMessage() for r in caplog.records].count(HOST_ISOLATION_LINE) == 1
    assert not wardex_sdk.diagnostics().initialized, "nothing started where nothing is seen"


def test_register_initializes_wardex_from_the_environment_when_the_host_has_not(
    installed, monkeypatch
):
    transport = RecordingTransport()
    real_init = wardex_sdk.init
    calls: list[tuple] = []

    def init(*args, **kwargs):
        calls.append((args, kwargs))
        real_init(transport=transport)

    monkeypatch.setattr(wardex_sdk, "init", init)
    with plugin() as ctx, hermes(ctx) as h:
        assert calls == [((), {})], "the environment's configuration, nothing passed"
        on_loop(_one_turn, h)
    assert _turns(exported(transport)) == ["invoke_agent hermes"]


def test_an_embedders_earlier_init_is_used_as_it_is(installed, monkeypatch):
    transport = start_wardex()

    def init(*args, **kwargs):
        raise AssertionError("register must not initialize wardex a second time")

    monkeypatch.setattr(wardex_sdk, "init", init)
    with plugin() as ctx, hermes(ctx) as h:
        on_loop(_one_turn, h)
    assert _turns(exported(transport)) == ["invoke_agent hermes"]


@pytest.mark.no_destination
def test_with_no_destination_the_callbacks_are_registered_and_inert_until_a_later_init(installed):
    with plugin() as ctx, hermes(ctx) as h:
        assert not wardex_sdk.diagnostics().initialized
        assert set(ctx.hooks) == set(HOOKS) and set(ctx.middleware) == set(MIDDLEWARE)
        on_loop(_one_turn, h)
        assert ctx.failures == []
        assert {r.value for r in ctx.returns} == {None}

        transport = start_wardex()
        on_loop(_one_turn, h)
        assert {len(v) for v in [*ctx.hooks.values(), *ctx.middleware.values()]} == {1}
    assert _turns(exported(transport)) == ["invoke_agent hermes"]


def test_a_second_init_records_through_the_same_callbacks(installed):
    first = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        on_loop(_one_turn, h)
        assert _turns(exported(first)) == ["invoke_agent hermes"]
        wardex_sdk.close()
        on_loop(_one_turn, h)  # wardex closed: Hermes's turn runs, nothing is recorded
        second = start_wardex()
        on_loop(_one_turn, h)
        assert {len(v) for v in [*ctx.hooks.values(), *ctx.middleware.values()]} == {1}
        assert ctx.failures == []
    assert _turns(exported(second)) == ["invoke_agent hermes"]
    assert _turns(received(first)) == ["invoke_agent hermes"], "the closed client got one turn"


def test_unloading_the_plugin_forgets_its_context(installed):
    start_wardex()
    with plugin():
        assert status().state is AdapterState.INSTALLED
    assert status().state is AdapterState.UNSUPPORTED
