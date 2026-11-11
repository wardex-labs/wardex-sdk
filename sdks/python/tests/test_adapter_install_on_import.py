"""An adapter installs when the application imports its framework, not at `init()`.

Importing every installed framework at `init()` cost an application that uses
none of them over a second of startup, thousands of modules and well over a
hundred megabytes, and imported code the application never asked for. Here
`init()` only reads package metadata; the adapter installs right after the
application's own import of the framework finishes, on whichever thread made
it, and `wardex.diagnostics()` says which adapters are still waiting.

The framework in most of these tests is a package written to a temporary
directory with its own dist-info, registered in the place of a shipped adapter,
so each test controls when (and whether) it is imported.
"""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
import textwrap
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

import wardex_sdk as wardex
from wardex_sdk import AdapterName, AdapterState
from wardex_sdk._adapters import _ADAPTERS, _Registration
from wardex_sdk._adapters._base import AdapterInterface
from wardex_sdk._adapters._measured import MEASURED, Measured
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk.testing import capture_gaps

# Unique per test module so a fake framework never collides with a real one.
_FW = "wardexfakefw"
_NS = "wardexfakens"


class _Recorder(AdapterInterface):
    """Stands in for the LangGraph adapter; records what it saw when it installed."""

    built: list[_Recorder] = []
    fail_build = False

    def __init__(self) -> None:
        if _Recorder.fail_build:
            raise RuntimeError("cannot build")
        self._installed = False
        self.thread: str | None = None
        self.saw_complete: bool | None = None
        _Recorder.built.append(self)

    def name(self) -> str:
        return AdapterName.LANGGRAPH.value

    def install(self, client, ctx=None) -> None:  # noqa: ANN001
        self.thread = threading.current_thread().name
        module = sys.modules.get(_FW) or sys.modules.get(f"{_NS}.sub")
        self.saw_complete = bool(getattr(module, "COMPLETE", False))
        self._installed = True

    def uninstall(self) -> None:
        self._installed = False


def _write_dist(root: Path, dist: str, version: str) -> None:
    info = root / f"{dist.replace('-', '_')}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n")


@pytest.fixture
def framework(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A regular-package framework `wardexfakefw` (dist `wardexfakefw`, 1.0),
    on sys.path, registered as the LangGraph adapter's framework."""
    pkg = tmp_path / _FW
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        textwrap.dedent(
            """
            import os
            if os.environ.get("WARDEXFAKEFW_RAISE"):
                raise RuntimeError("the framework's own import failed")
            COMPLETE = True
            """
        )
    )
    _write_dist(tmp_path, _FW, "1.0")
    ns_sub = tmp_path / _NS / "sub"
    ns_sub.mkdir(parents=True)
    (ns_sub / "__init__.py").write_text("COMPLETE = True\n")
    _write_dist(tmp_path, _NS, "1.0")

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    monkeypatch.setitem(_ADAPTERS, AdapterName.LANGGRAPH, _Registration(_FW, _FW, _Recorder, None))
    monkeypatch.setitem(MEASURED, AdapterName.LANGGRAPH, (Measured(_FW, ("1.0",)),))
    _Recorder.built = []
    _Recorder.fail_build = False
    reset_reports_for_test()
    meta_path_before = list(sys.meta_path)
    try:
        yield tmp_path
    finally:
        wardex.close()
        for name in [m for m in sys.modules if m.split(".")[0] in (_FW, _NS)]:
            del sys.modules[name]
        assert sys.meta_path == meta_path_before, "wardex left an import hook behind"


def _init() -> None:
    wardex.init(
        transport=wardex.NoOpTransport(),
        intercept=False,
        adapters=wardex.AdaptersConfig(enabled=(AdapterName.LANGGRAPH,)),
    )


def _state() -> AdapterState:
    return next(s for s in wardex.diagnostics().adapters if s.name == "langgraph").state


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def lines() -> Iterator[_Lines]:
    handler = _Lines()
    logger = logging.getLogger("wardex_sdk")
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


def test_init_does_not_import_a_framework_the_app_has_not_imported(framework):
    _init()
    assert _FW not in sys.modules, "init() imported the framework"
    assert _Recorder.built == [], "the adapter was built before its framework was imported"
    status = next(s for s in wardex.diagnostics().adapters if s.name == "langgraph")
    assert status.state is AdapterState.WAITING_FOR_IMPORT
    assert (status.framework, status.version) == (_FW, "1.0")
    assert capture_gaps() == (), "an adapter waiting for its framework is not a capture gap"

    importlib.import_module(_FW)

    assert _state() is AdapterState.INSTALLED
    (adapter,) = _Recorder.built
    assert adapter.saw_complete, "the adapter installed before the framework finished importing"


def test_a_framework_imported_before_init_installs_at_init(framework):
    importlib.import_module(_FW)
    _init()
    assert _state() is AdapterState.INSTALLED
    assert len(_Recorder.built) == 1


def test_a_framework_imported_on_another_thread_installs_there(framework):
    _init()
    worker = threading.Thread(target=importlib.import_module, args=(_FW,), name="importer")
    worker.start()
    worker.join()
    assert _state() is AdapterState.INSTALLED
    (adapter,) = _Recorder.built
    assert (adapter.thread, adapter.saw_complete) == ("importer", True)


def test_concurrent_imports_install_the_adapter_once(framework):
    _init()
    start = threading.Barrier(8)

    def run() -> None:
        start.wait()
        importlib.import_module(_FW)

    workers = [threading.Thread(target=run) for _ in range(8)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert _state() is AdapterState.INSTALLED
    assert len(_Recorder.built) == 1


def test_close_before_the_import_leaves_nothing_to_install(framework):
    _init()
    wardex.close()
    importlib.import_module(_FW)
    assert _Recorder.built == []
    assert _state() is AdapterState.WAITING_FOR_IMPORT, "the account of the last init() is kept"


def test_a_namespace_framework_installs_after_its_first_real_module(framework, monkeypatch):
    monkeypatch.setitem(_ADAPTERS, AdapterName.LANGGRAPH, _Registration(_NS, _NS, _Recorder, None))
    monkeypatch.setitem(MEASURED, AdapterName.LANGGRAPH, (Measured(_NS, ("1.0",)),))
    _init()
    assert _state() is AdapterState.WAITING_FOR_IMPORT
    importlib.import_module(f"{_NS}.sub")
    assert _state() is AdapterState.INSTALLED
    (adapter,) = _Recorder.built
    assert adapter.saw_complete


def test_an_adapter_that_fails_at_import_time_never_raises_into_the_import(framework, lines):
    _init()
    _Recorder.fail_build = True
    module = importlib.import_module(_FW)
    assert module.COMPLETE
    assert _state() is AdapterState.FAILED
    assert any("did not finish installing" in m for m in lines.messages), lines.messages


def test_a_framework_whose_own_import_fails_is_retried_on_the_next_import(framework, monkeypatch):
    _init()
    monkeypatch.setenv("WARDEXFAKEFW_RAISE", "1")
    with pytest.raises(RuntimeError, match="the framework's own import failed"):
        importlib.import_module(_FW)
    assert _Recorder.built == []
    assert _state() is AdapterState.WAITING_FOR_IMPORT
    monkeypatch.delenv("WARDEXFAKEFW_RAISE")
    sys.modules.pop(_FW, None)
    importlib.import_module(_FW)
    assert _state() is AdapterState.INSTALLED


def test_an_untested_version_is_said_when_the_adapter_installs(framework, lines, monkeypatch):
    monkeypatch.setitem(MEASURED, AdapterName.LANGGRAPH, (Measured(_FW, ("0.5",)),))
    _init()
    assert not any("not tested against" in m for m in lines.messages)
    importlib.import_module(_FW)
    said = [m for m in lines.messages if "not tested against" in m]
    assert len(said) == 1
    assert f"{_FW} 1.0 (tested: 0.5.x)" in said[0]


def test_the_framework_module_keeps_its_own_loader(framework):
    """Whatever wardex wraps the loader with is gone once the module runs:
    `importlib.resources`, `inspect.getsource` and `reload` read it."""
    _init()
    module = importlib.import_module(_FW)
    assert type(module.__loader__).__name__ == "SourceFileLoader"
    assert module.__spec__.loader is module.__loader__
    importlib.reload(module)


_REAL = textwrap.dedent(
    """
    import json, sys, time
    import wardex_sdk as wardex
    frameworks = ("agents", "langgraph", "claude_agent_sdk")
    wardex.init(transport=wardex.NoOpTransport())
    imported = sorted(m for m in frameworks if m in sys.modules)
    waiting = sorted(
        s.name for s in wardex.diagnostics().adapters if s.state.value == "waiting_for_import"
    )
    import agents
    after = {s.name: s.state.value for s in wardex.diagnostics().adapters}
    print(json.dumps({"imported": imported, "waiting": waiting, "after": after}))
    wardex.close()
    """
)


def test_init_imports_none_of_the_installed_frameworks():
    """The real frameworks, in a fresh interpreter: none is imported by
    `init()`, each is reported waiting, and importing one installs its adapter."""
    env = dict(os.environ)
    env.pop("WARDEX_DISABLED", None)
    out = subprocess.run(
        [sys.executable, "-c", _REAL], capture_output=True, text=True, env=env, timeout=600
    )
    assert out.returncode == 0, out.stderr
    report = __import__("json").loads(out.stdout.strip().splitlines()[-1])
    assert report["imported"] == []
    assert report["waiting"] == ["anthropic_agent_sdk", "langgraph", "openai_agents"]
    assert report["after"]["openai_agents"] == "installed"
    assert report["after"]["langgraph"] == "waiting_for_import"


_CROSS_THREAD_IMPORTS = textwrap.dedent(
    """
    import sys, threading, time
    sys.path.insert(0, sys.argv[1])
    from wardex_sdk._adapters._registry import AdapterRegistry

    registry = AdapterRegistry()

    def install_z():
        import pkgx  # imports a framework another thread is in the middle of importing

    registry.install_on_import("pkgz", install_z)
    registry.install_on_import("pkgy", lambda: None)
    b = threading.Thread(target=lambda: __import__("pkgx"), daemon=True)
    a = threading.Thread(target=lambda: __import__("pkgz"), daemon=True)
    b.start()
    time.sleep(0.05)
    a.start()
    a.join(10)
    b.join(10)
    print(a.is_alive() or b.is_alive())
    """
)


def test_installs_on_two_threads_whose_imports_cross_never_hang_the_imports(tmp_path):
    """Thread B imports `pkgx`, which imports the watched `pkgy`; thread A
    imports the watched `pkgz`, whose install imports `pkgx`. Without wardex,
    A waits for B's `pkgx` and both finish; an install that held a lock across
    its own imports kept B from running `pkgy`'s install and hung both."""
    for name, body in {
        "pkgx": "import time\ntime.sleep(0.3)\nimport pkgy\n",
        "pkgy": "",
        "pkgz": "import time\ntime.sleep(0.1)\n",
    }.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text(body)
    out = subprocess.run(
        [sys.executable, "-c", _CROSS_THREAD_IMPORTS, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", "an import thread hung"


def test_close_during_an_install_leaves_that_adapter_uninstalled(framework, monkeypatch):
    """`close()` on another thread while an import's install runs: the install
    finishing after `close()` is undone, so nothing stays patched."""
    _init()
    started, release = threading.Event(), threading.Event()
    real_install = _Recorder.install

    def slow_install(self, client, ctx=None):  # noqa: ANN001
        started.set()
        release.wait(10)
        real_install(self, client, ctx)

    monkeypatch.setattr(_Recorder, "install", slow_install)
    importer = threading.Thread(target=importlib.import_module, args=(_FW,))
    importer.start()
    assert started.wait(10)
    closer = threading.Thread(target=wardex.close)
    closer.start()
    closer.join(10)
    release.set()
    importer.join(10)
    assert not closer.is_alive() and not importer.is_alive()
    (adapter,) = _Recorder.built
    assert adapter._installed is False
    from wardex_sdk._adapters._registry import get_registry

    assert not get_registry().is_installed(AdapterName.LANGGRAPH.value)
