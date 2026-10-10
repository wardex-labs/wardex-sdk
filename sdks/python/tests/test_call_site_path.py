"""The file a decorator records as a span's call site leaves without the folders above
the import root.

A decorated function's source path used to leave as `co_filename`, usually absolute, so
`/Users/alice.kim/work/acme-acquisition-2027/support_bot/agent.py` reached the envelope's
`Span.call_site.file` and OTLP's `code.file.path` with the OS user name and the private
project folder in it. The file is now relative to the folder the module's top-level
package was imported from (Sentry's `filename_for_module` rule), and wherever that cannot
be placed it is the file name alone: never an absolute path.

The end-to-end cases build real trees under a fake home folder, import them the way a
host would, and read what the decorator recorded and what both encoders put on the wire.
The Windows cases drive the pure function with `ntpath`: string-level checks of the rule
on any OS, not a run on Windows.
"""

from __future__ import annotations

import importlib
import ntpath
import os
import posixpath
import runpy
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from wardex_sdk import _hub, _wardex_native
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._decorators import _call_site_file, workflow
from wardex_sdk._types import Envelope
from wardex_sdk.transport._base import Transport
from wardex_sdk.transport._codec import decode, encode

USER = "alice.kim"
PROJECT = "acme-acquisition-2027"

_SOURCE = "import wardex_sdk as wardex\n\n\n@wardex.workflow\ndef answer_ticket():\n    return 1\n"


class _Recording(Transport):
    def __init__(self) -> None:
        self.envelopes: list[Envelope] = []

    def export(self, envelope: Envelope) -> None:
        self.envelopes.append(envelope)


@pytest.fixture
def recording() -> _Recording:
    _hub.reset_for_test()
    t = _Recording()
    _hub.set_client(Client(WardexConfig(backend=BackendConfig(api_key="k")), t))
    return t


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The import root under a fake home folder, on `sys.path`; every module imported
    while the test runs is dropped from `sys.modules` after it."""
    root = tmp_path / "Users" / USER / "work" / PROJECT
    root.mkdir(parents=True)
    monkeypatch.syspath_prepend(str(root))
    before = set(sys.modules)
    yield root
    for name in set(sys.modules) - before:
        del sys.modules[name]


def _write(path: Path, text: str = _SOURCE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _exported_file(t: _Recording, fn) -> str:
    """Call `fn`, then return the call-site file both wire encodings carry, asserting
    they agree and that neither holds the user name or the project folder."""
    fn()
    _hub.get_client().flush()
    env = t.envelopes[-1]
    otlp = _wardex_native.codec.encode_otlp_traces(env)
    site = decode(encode(env))["items"][0]["span"]["call_site"]
    attrs = _wardex_native.codec.decode_otlp_traces(otlp)["resource_spans"][0]["scope_spans"][0][
        "spans"
    ][0]["attributes"]
    assert attrs["code.file.path"] == site["file"]
    for needle in (USER, PROJECT):
        assert needle.encode() not in otlp
        assert needle not in repr(site)
    assert not os.path.isabs(site["file"])
    return site["file"]


# -- end to end, through the decorator and both encoders ------------------------------


def test_a_module_in_a_package_reports_its_path_from_the_package_root(project, recording):
    _write(project / "support_bot" / "__init__.py", "")
    _write(project / "support_bot" / "agent.py")
    mod = importlib.import_module("support_bot.agent")

    assert _exported_file(recording, mod.answer_ticket) == "support_bot/agent.py"
    site = recording.envelopes[-1].spans[0].call_site
    assert (site.line, site.function, site.module) == (4, "answer_ticket", "support_bot.agent")


def test_a_nested_package_keeps_every_folder_below_the_root(project, recording):
    _write(project / "support_bot" / "__init__.py", "")
    _write(project / "support_bot" / "tools" / "__init__.py", "")
    _write(project / "support_bot" / "tools" / "search.py")
    mod = importlib.import_module("support_bot.tools.search")

    assert _exported_file(recording, mod.answer_ticket) == "support_bot/tools/search.py"


def test_a_top_level_module_reports_its_file_name(project, recording):
    _write(project / "ticket_helpers.py")
    mod = importlib.import_module("ticket_helpers")

    assert _exported_file(recording, mod.answer_ticket) == "ticket_helpers.py"


def test_a_main_script_reports_its_file_name(project, recording):
    """`python agent.py`: the module is `__main__`, so there is no package to be
    relative to, and only the file name leaves."""
    script = _write(project / "agent.py")
    ns = runpy.run_path(str(script), run_name="__main__")

    assert ns["answer_ticket"].__module__ == "__main__"
    assert _exported_file(recording, ns["answer_ticket"]) == "agent.py"


def test_a_function_with_no_module_reports_no_absolute_path(project, recording):
    """`exec` with globals that carry no `__name__` makes a function whose
    `__module__` is None: nothing to place the file by, so the file name alone."""
    ns: dict = {}
    exec(compile("def made():\n    return 1\n", str(project / "made.py"), "exec"), ns)
    assert ns["made"].__module__ is None

    assert _exported_file(recording, workflow(ns["made"])) == "made.py"


def test_a_module_name_no_package_on_this_process_answers_to(project, recording):
    ns: dict = {"__name__": "never_imported.generated"}
    exec(compile("def made():\n    return 1\n", str(project / "gen.py"), "exec"), ns)

    assert _exported_file(recording, workflow(ns["made"])) == "gen.py"


def test_a_namespace_package_reports_its_file_name(project, recording):
    """A namespace package has no `__file__` to find the root by."""
    _write(project / "ns_bot" / "agent.py")
    mod = importlib.import_module("ns_bot.agent")
    assert sys.modules["ns_bot"].__file__ is None

    assert _exported_file(recording, mod.answer_ticket) == "agent.py"


def test_code_from_another_tree_under_a_borrowed_module_name(project, recording, tmp_path):
    """`functools.wraps` copies `__module__` and not `__code__`: a wrapper whose code
    lives elsewhere under the home folder claims the host's package. The path does
    not start at that package's root, so only the file name leaves."""
    _write(project / "support_bot" / "__init__.py", "")
    lib = _write(
        tmp_path / "Users" / USER / "lib" / "retrying.py",
        "import functools\n\n\ndef retry(fn):\n"
        "    @functools.wraps(fn)\n    def inner(*a, **k):\n        return fn(*a, **k)\n\n"
        "    return inner\n",
    )
    ns: dict = {}
    exec(compile(lib.read_text(), str(lib), "exec"), ns)
    _write(project / "support_bot" / "agent.py", "def answer_ticket():\n    return 1\n")
    target = importlib.import_module("support_bot.agent")
    wrapped = ns["retry"](target.answer_ticket)
    assert wrapped.__module__ == "support_bot.agent"

    assert _exported_file(recording, workflow(wrapped)) == "retrying.py"


def test_a_package_path_that_climbs_above_the_root_reports_its_file_name(project, recording):
    """A package that extends its own `__path__` with `../..` loads a submodule from
    a folder beside the root; the remainder would name it, so only the file name."""
    _write(
        project / "support_bot" / "__init__.py",
        "import os\n__path__.append(os.path.join(os.path.dirname(__file__), '..', '..', "
        "'plugins'))\n",
    )
    _write(project.parent / "plugins" / "billing.py")
    mod = importlib.import_module("support_bot.billing")
    assert ".." in mod.__file__

    assert _exported_file(recording, mod.answer_ticket) == "billing.py"


def test_a_package_imported_from_a_zip_reports_its_path_inside_the_archive(
    project, recording, monkeypatch
):
    archive = project / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("zipped_bot/__init__.py", "")
        z.writestr("zipped_bot/agent.py", _SOURCE)
    monkeypatch.syspath_prepend(str(archive))
    mod = importlib.import_module("zipped_bot.agent")

    assert _exported_file(recording, mod.answer_ticket) == "zipped_bot/agent.py"


def test_a_package_reached_through_a_symlinked_root(project, recording, tmp_path, monkeypatch):
    real = tmp_path / "Users" / USER / "real-checkout"
    _write(real / "linked_bot" / "__init__.py", "")
    _write(real / "linked_bot" / "agent.py")
    link = project / "checkout"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.syspath_prepend(str(link))
    mod = importlib.import_module("linked_bot.agent")

    assert _exported_file(recording, mod.answer_ticket) == "linked_bot/agent.py"


# -- the pure function: every fallback, and Windows-shaped paths ----------------------

_POSIX_ROOT = "/Users/alice.kim/work/acme-acquisition-2027"


def _pkg(file: str | None) -> dict:
    return {"support_bot": SimpleNamespace(__file__=file)}


@pytest.mark.parametrize(
    ("path", "module", "modules", "expected"),
    [
        (
            f"{_POSIX_ROOT}/support_bot/agent.py",
            "support_bot.agent",
            _pkg(f"{_POSIX_ROOT}/support_bot/__init__.py"),
            "support_bot/agent.py",
        ),
        # a compiled file is named by its source
        (
            f"{_POSIX_ROOT}/support_bot/agent.pyc",
            "support_bot.agent",
            _pkg(f"{_POSIX_ROOT}/support_bot/__init__.py"),
            "support_bot/agent.py",
        ),
        (f"{_POSIX_ROOT}/helpers.py", "helpers", {}, "helpers.py"),
        (f"{_POSIX_ROOT}/agent.py", "__main__", {}, "agent.py"),
        (f"{_POSIX_ROOT}/agent.py", None, {}, "agent.py"),
        (f"{_POSIX_ROOT}/agent.py", "", {}, "agent.py"),
        (f"{_POSIX_ROOT}/agent.py", 42, {}, "agent.py"),
        # the top-level package was never imported
        (f"{_POSIX_ROOT}/support_bot/agent.py", "support_bot.agent", {}, "agent.py"),
        # a namespace package
        (f"{_POSIX_ROOT}/support_bot/agent.py", "support_bot.agent", _pkg(None), "agent.py"),
        # a package whose `__init__` is compiled
        (
            f"{_POSIX_ROOT}/support_bot/tools/search.py",
            "support_bot.tools.search",
            _pkg(f"{_POSIX_ROOT}/support_bot/__init__.cpython-312-darwin.so"),
            "support_bot/tools/search.py",
        ),
        # a package imported through a relative `sys.path` entry
        (
            "support_bot/agent.py",
            "support_bot.agent",
            _pkg("support_bot/__init__.py"),
            "support_bot/agent.py",
        ),
        # a top-level module that is a plain file: two folders up is one too many
        (
            f"{_POSIX_ROOT}/agent.py",
            "support_bot.agent",
            _pkg(f"{_POSIX_ROOT}/support_bot.py"),
            "agent.py",
        ),
        # code beside the package under the same root, claiming the package's name
        (
            f"{_POSIX_ROOT}/vendor/retrying.py",
            "support_bot.agent",
            _pkg(f"{_POSIX_ROOT}/support_bot/__init__.py"),
            "retrying.py",
        ),
        # a package imported from `/`: everything else on the disk is "below" that root
        (
            "/home/alice.kim/lib/retrying.py",
            "support_bot.agent",
            _pkg("/support_bot/__init__.py"),
            "retrying.py",
        ),
        # an editable install that maps the package onto a folder of another name
        (
            f"{_POSIX_ROOT}/src/sb_impl/agent.py",
            "support_bot.agent",
            _pkg(f"{_POSIX_ROOT}/src/sb_impl/__init__.py"),
            "agent.py",
        ),
        # a module name with an empty first part, matched by an empty folder name
        (
            "/acme-acquisition-2027//alice.kim/agent.py",
            ".agent",
            {"": SimpleNamespace(__file__="/acme-acquisition-2027//__init__.py")},
            "agent.py",
        ),
        # the package folder appears only in the middle of the path
        (
            f"/Users/alice.kim/backup{_POSIX_ROOT}/support_bot/agent.py",
            "support_bot.agent",
            _pkg(f"{_POSIX_ROOT}/support_bot/__init__.py"),
            "agent.py",
        ),
        # a root that is a prefix of a sibling folder's name
        (
            f"{_POSIX_ROOT}-old/support_bot/agent.py",
            "support_bot.agent",
            _pkg(f"{_POSIX_ROOT}/support_bot/__init__.py"),
            "agent.py",
        ),
        (
            f"{_POSIX_ROOT}/support_bot/../../plugins/agent.py",
            "support_bot.agent",
            _pkg(f"{_POSIX_ROOT}/support_bot/__init__.py"),
            "agent.py",
        ),
    ],
)
def test_the_rule_on_posix_paths(path, module, modules, expected):
    got = _call_site_file(path, module, modules, posixpath)
    assert got == expected
    assert USER not in got and PROJECT not in got
    assert not posixpath.isabs(got)


def test_a_module_whose_file_attribute_raises_reports_its_file_name():
    class _Hostile:
        @property
        def __file__(self):
            raise RuntimeError("no")

    modules = {"support_bot": _Hostile()}
    path = f"{_POSIX_ROOT}/support_bot/agent.py"
    assert _call_site_file(path, "support_bot.agent", modules, posixpath) == "agent.py"


_WIN_ROOT = "C:\\Users\\bob.smith\\work\\acme-acquisition-2027"


@pytest.mark.parametrize(
    ("path", "module", "modules", "expected"),
    [
        (
            f"{_WIN_ROOT}\\support_bot\\agent.py",
            "support_bot.agent",
            _pkg(f"{_WIN_ROOT}\\support_bot\\__init__.py"),
            "support_bot\\agent.py",
        ),
        (
            f"{_WIN_ROOT}\\support_bot\\tools\\search.py",
            "support_bot.tools.search",
            _pkg(f"{_WIN_ROOT}\\support_bot\\__init__.py"),
            "support_bot\\tools\\search.py",
        ),
        (f"{_WIN_ROOT}\\helpers.py", "helpers", {}, "helpers.py"),
        (f"{_WIN_ROOT}\\agent.py", "__main__", {}, "agent.py"),
        (f"{_WIN_ROOT}\\agent.py", None, {}, "agent.py"),
        # a root spelled with forward slashes (a `sys.path` entry written that way)
        (
            "C:/Users/bob.smith/work/acme-acquisition-2027\\support_bot\\agent.py",
            "support_bot.agent",
            _pkg("C:/Users/bob.smith/work/acme-acquisition-2027\\support_bot\\__init__.py"),
            "support_bot\\agent.py",
        ),
        # separators mixed so that a cut on `\\` alone would land at the drive
        (
            "C:\\Users\\bob.smith/work/support_bot/agent.py",
            "support_bot.agent",
            _pkg("C:\\Users\\bob.smith/work/support_bot/__init__.py"),
            "support_bot\\agent.py",
        ),
        (
            f"D:\\other{_WIN_ROOT[2:]}\\support_bot\\agent.py",
            "support_bot.agent",
            _pkg(f"{_WIN_ROOT}\\support_bot\\__init__.py"),
            "agent.py",
        ),
        # a package "folder" that is a drive: its name is not the package's
        (
            "C:\\Users\\bob.smith\\agent.py",
            "support_bot.agent",
            _pkg("C:\\__init__.py"),
            "agent.py",
        ),
        (
            "\\\\fileserver\\home\\bob.smith\\proj\\support_bot\\agent.py",
            "support_bot.agent",
            _pkg("\\\\fileserver\\home\\bob.smith\\proj\\support_bot\\__init__.py"),
            "support_bot\\agent.py",
        ),
    ],
)
def test_the_rule_on_windows_shaped_paths(path, module, modules, expected):
    """String-level: `ntpath` stands in for `os.path` on Windows. Not a Windows run."""
    got = _call_site_file(path, module, modules, ntpath)
    assert got == expected
    assert "bob.smith" not in got
    assert not ntpath.isabs(got) and not ntpath.splitdrive(got)[0]


def test_the_file_is_placed_once_at_decoration_not_per_call(project, recording, monkeypatch):
    """The call path pays nothing: the file is placed when the decorator runs, and
    no call afterwards places it again."""
    import wardex_sdk._decorators as decorators

    placed: list[str] = []
    real = decorators._call_site_file

    def counting(*args):
        placed.append(args[0])
        return real(*args)

    monkeypatch.setattr(decorators, "_call_site_file", counting)
    _write(project / "support_bot" / "__init__.py", "")
    _write(project / "support_bot" / "agent.py")
    mod = importlib.import_module("support_bot.agent")
    for _ in range(3):
        mod.answer_ticket()
    assert len(placed) == 1
