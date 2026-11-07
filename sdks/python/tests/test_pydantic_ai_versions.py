"""The two version checks the pydantic-ai adapter rests on.

The measured range is the whole `pydantic-ai-slim` 2.x major (`_adapters/_measured.py`): the
framework releases minors weekly, and what the adapter reads is its OpenTelemetry spans, which
carry their own schema version. So a 3.x release is unmeasured and said once at `init()`, and a
schema version other than 5 is said once at install and written on every run unit.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

import wardex_sdk as wardex
from _pydantic_ai_support import extras, one, user_owned
from wardex_sdk._adapters._measured import installed_version
from wardex_sdk._assembly import counters
from wardex_sdk._config import AdaptersConfig
from wardex_sdk._enums import AdapterName, AdapterState
from wardex_sdk.testing import RecordingTransport

pytestmark = pytest.mark.usefixtures("fresh_counters")

_ONLY = AdaptersConfig(enabled=(AdapterName.PYDANTIC_AI,))


@pytest.fixture
def transport() -> Iterator[RecordingTransport]:
    recording = RecordingTransport()
    yield recording
    wardex.close()


def _status():
    (status,) = [s for s in wardex.diagnostics().adapters if s.name == "pydantic_ai"]
    return status


def test_a_release_outside_the_measured_major_installs_and_says_so_once(
    transport, monkeypatch, capsys
):
    real = installed_version
    monkeypatch.setattr(
        "wardex_sdk._adapters.installed_version",
        lambda dist: "3.0.0" if dist == "pydantic-ai-slim" else real(dist),
    )
    with user_owned():
        wardex.init(transport=transport, adapters=_ONLY)
        status = _status()
        assert status.state is AdapterState.INSTALLED
        assert (status.version, status.measured) == ("3.0.0", False)
        assert status.measured_versions == ("2.x",)
        lines = [line for line in capsys.readouterr().err.splitlines() if "pydantic_ai" in line]
        assert len(lines) == 1
        assert "pydantic-ai-slim 3.0.0 (tested: 2.x)" in lines[0]
        assert counters.get("adapters.pydantic_ai.unmeasured_version") == 1


def test_a_release_inside_the_measured_major_says_nothing(transport, capsys):
    with user_owned():
        wardex.init(transport=transport, adapters=_ONLY)
        status = _status()
        assert status.measured is True
        assert "pydantic_ai" not in capsys.readouterr().err


@pytest.mark.parametrize(("version", "said"), [(6, True), (5, False)], ids=["v6", "v5"])
def test_the_instrumentation_schema_version_is_checked_and_written_on_the_run(
    transport, capsys, version, said
):
    with user_owned(version=version):
        wardex.init(transport=transport, adapters=_ONLY)
        Agent(TestModel(), name="versioned").run_sync("hi")
        wardex.flush()
        spans = [s for e in transport.envelopes for s in e.spans]
        run = one(spans, "invoke_agent versioned")
        assert extras(run)["wardex.pydantic_ai.schema_version"] == version
        err = capsys.readouterr().err
        assert ("instrumentation schema version 6" in err) is said
        assert len([line for line in err.splitlines() if "pydantic_ai" in line]) == int(said)
        assert counters.get("adapters.pydantic_ai.schema_version") == (1 if said else 0)
        status = _status()
        assert status.state is AdapterState.INSTALLED
        assert status.measured is (not said)
        assert ("instrumentation schema version 6" in status.detail) is said
