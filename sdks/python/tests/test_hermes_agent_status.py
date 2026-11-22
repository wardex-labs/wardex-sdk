"""What `wardex.diagnostics()` says about the Hermes adapter, and the lines wardex says once.

Hermes loads the plugin after `init()` may already have run, and states its contract version in
each payload rather than in a package version, so the adapter's row is read at diagnostics time:
not attached yet, attached and measured, attached on a version this release was not measured on,
or attached with events dropped or registrations refused.
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
    start_wardex,
    status,
)
from wardex_sdk._adapters._hermes_agent_attach import HOST_ISOLATION_LINE
from wardex_sdk._assembly import counters
from wardex_sdk._enums import AdapterState

MEASURED = ("hermes.observer.v1", "hermes.middleware.v1")


@pytest.fixture(autouse=True)
def _installed(tmp_path: Path):
    forget_attachments()
    with hermes_installed(tmp_path):
        yield
    forget_attachments()


def _one_turn(h: FakeHermes) -> None:
    turn = h.turn()
    h.model_call(turn)
    h.end(turn)


def test_before_hermes_attaches_the_adapter_reads_unsupported_with_the_enable_command():
    start_wardex()
    row = status()
    assert row.state is AdapterState.UNSUPPORTED
    assert row.measured is None
    assert "hermes plugins enable wardex" in row.detail
    assert "plugins.isolation: host" in row.detail


def test_attached_before_any_payload_reads_installed_with_nothing_measured_yet():
    start_wardex()
    with plugin():
        row = status()
    assert (row.state, row.measured, row.measured_versions, row.detail) == (
        AdapterState.INSTALLED,
        None,
        (),
        "",
    )


def test_attached_on_the_measured_contract_reads_installed_and_measured():
    start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        on_loop(_one_turn, h)
        row = status()
    assert row.state is AdapterState.INSTALLED
    assert (row.measured, row.measured_versions) == (True, MEASURED)


def test_an_unmeasured_contract_version_is_said_once_and_recording_continues(caplog):
    transport = start_wardex()
    with caplog.at_level(logging.WARNING, logger="wardex_sdk"):
        with (
            plugin() as ctx,
            hermes(ctx, schema={"telemetry_schema_version": "hermes.observer.v2"}) as h,
        ):
            on_loop(_one_turn, h)
            on_loop(_one_turn, h)
            row = status()
    said = [r.getMessage() for r in caplog.records if "hermes.observer.v2" in r.getMessage()]
    assert len(said) == 1, said
    assert row.state is AdapterState.INSTALLED
    assert (row.measured, row.measured_versions) == (False, MEASURED)
    assert "'hermes.observer.v2'" in row.detail
    roots = [n for n in exported(transport) if n.name == "invoke_agent hermes"]
    assert len(roots) == 2, "a newer contract version is still recorded"


def test_an_event_missing_a_required_field_is_dropped_counted_and_named(caplog):
    counters.reset()
    start_wardex()
    with caplog.at_level(logging.WARNING, logger="wardex_sdk"), plugin() as ctx:
        ctx.invoke_hook("pre_llm_call", session_id="s-1")
        ctx.invoke_hook("pre_llm_call", session_id="s-1")
        row = status()
    assert counters.get("adapters.hermes_agent.dropped.pre_llm_call") == 2
    said = [r.getMessage() for r in caplog.records if "dropped" in r.getMessage()]
    assert len(said) == 1 and "pre_llm_call" in said[0] and "turn_id" in said[0]
    assert "pre_llm_call (missing turn_id)" in row.detail
    assert ctx.failures == []


class _RefusesMiddleware(FakePluginContext):
    """A Hermes that takes hooks and refuses middleware (its load guard timed out half way)."""

    def register_middleware(self, kind, callback):
        return None


def test_a_refused_registration_is_said_once_and_named(caplog):
    start_wardex()
    ctx = _RefusesMiddleware()
    with caplog.at_level(logging.WARNING, logger="wardex_sdk"):
        wardex_sdk.hermes_agent.register(ctx)
    said = [r.getMessage() for r in caplog.records if "refused" in r.getMessage()]
    assert len(said) == 1 and "llm_execution, tool_execution" in said[0]
    row = status()
    assert row.state is AdapterState.INSTALLED, "the hooks it took still record"
    assert "llm_execution, tool_execution" in row.detail
    ctx.unload()


def test_a_context_that_refused_everything_reads_unsupported_and_names_the_refusal():
    start_wardex()
    ctx = FakePluginContext()
    ctx.abandon_load()
    wardex_sdk.hermes_agent.register(ctx)
    row = status()
    assert row.state is AdapterState.UNSUPPORTED
    assert "pre_llm_call" in row.detail and "tool_execution" in row.detail


def test_the_last_restated_status_survives_close():
    start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        on_loop(_one_turn, h)
        wardex_sdk.close()
    row = status()
    assert (row.state, row.measured, row.measured_versions) == (
        AdapterState.INSTALLED,
        True,
        MEASURED,
    )


def test_in_the_plugin_host_process_the_adapter_declines_with_the_host_line(monkeypatch):
    monkeypatch.setenv("HERMES_PLUGIN_HOST_PROCESS", "1")
    start_wardex()
    row = status()
    assert row.state is AdapterState.UNSUPPORTED
    assert row.detail == HOST_ISOLATION_LINE
