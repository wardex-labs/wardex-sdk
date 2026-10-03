"""Where the path of one of the host's source files may go when it leaves the process.

Two texts carry the host's files: a span's call site (`code.file.path`) and an
exception's stack trace. One rule places both, so the OS user name and every
folder above the import root stay on the machine while the package path a
developer needs to find the code leaves with the span. A stack trace carries
paths in more places than its frame lines — an `OSError` names the file it
could not open, a source line can hold a path literal — so it also has this
process's home folder written as `~` wherever else the text carries it.

Imports nothing from the package, so any layer that has to place a host path
can ask this module without importing the span machinery.
"""

from __future__ import annotations

import os
import re
import sys
import traceback
from collections.abc import Mapping
from types import TracebackType
from typing import Any


def _call_site_file(
    path: str,
    module: object,
    modules: Mapping[str, Any] | None = None,
    pathmod: Any = os.path,
) -> str:
    """`path` relative to the folder its top-level package was imported from.

    `support_bot.agent` defined in `/Users/alice/work/acme/support_bot/agent.py`
    gives `support_bot/agent.py`: the OS user name and every folder above the
    import root stay on the machine, and the package path a developer needs to
    find the code leaves with the span. A top-level module, `__main__`
    included, gives its file name alone.

    This is Sentry's `filename_for_module` (`sentry_sdk/utils.py`) for every
    file that lies inside its top-level package's folder, and the file name
    alone for every other one: the result is always `<package>/<path inside
    the package>` or a bare file name, never an absolute path. Where Sentry
    sends the absolute path (no module name, a top-level package missing from
    `modules`, a namespace package with no `__file__`, any error), this sends
    the file name. It also refuses what Sentry's cut lets through: a top-level
    module that is not a package (two folders up from its file is one too
    many), a path outside the package's folder (Sentry cuts wherever the root
    first appears in the string, and `functools.wraps` copies `__module__`
    without `__code__`, so the two can name different trees), a package
    folder not named after the package, and a remainder that climbs out
    through `..`.

    `modules` and `pathmod` are injectable so the rule is testable against
    Windows-shaped paths (`ntpath`) on any OS. `pathmod.altsep` is folded into
    `pathmod.sep` first, so a mixed-separator path cannot cut at the wrong
    folder.
    """
    sep, altsep = pathmod.sep, pathmod.altsep
    if altsep:
        path = path.replace(altsep, sep)
    if path.endswith(".pyc"):
        path = path[:-1]
    name = pathmod.basename(path)
    if not isinstance(module, str) or not module:
        return name
    base = module.split(".", 1)[0]
    if not base or base == module:
        return name
    try:
        # A failure here is a reason to send less, never to fail the host's
        # decoration: a module object can raise anything from attribute access.
        base_file = (sys.modules if modules is None else modules)[base].__file__
    except Exception:
        return name
    if not isinstance(base_file, str):
        return name
    if altsep:
        base_file = base_file.replace(altsep, sep)
    package_dir, _, init = base_file.rpartition(sep)
    if not init.startswith("__init__.") or package_dir.rpartition(sep)[2] != base:
        return name
    inside = path[len(package_dir) + 1 :] if path.startswith(package_dir + sep) else ""
    if not inside or ".." in inside.split(sep):
        return name
    return base + sep + inside


#: A frame line of a formatted traceback (and a `SyntaxError`'s own location).
_FRAME_FILE = re.compile(r'File "(.+?)", line (?=\d)')


def _stacktrace(exc: Exception, tb: TracebackType | None) -> str:
    """The traceback as Python prints it, with every frame's file placed by the
    call-site rule: a path relative to its package's import root, or the bare
    file name, never an absolute path. Locals are never captured."""
    text = "".join(traceback.TracebackException(type(exc), exc, tb).format())
    files = _frame_files(exc, tb)

    def place(m: re.Match[str]) -> str:
        path = m[1]
        return f'File "{files.get(path) or _call_site_file(path, None)}", line '

    return _scrub_home(_FRAME_FILE.sub(place, text))


def _frame_files(exc: BaseException, tb: TracebackType | None) -> dict[str, str]:
    """Each frame's file, placed with the module its frame ran under.

    Walks the whole graph the formatter prints — `__cause__`, `__context__` and
    an exception group's members — because a chained traceback's frames are
    paths too. A file this walk misses still leaves as its bare name.
    """
    files: dict[str, str] = {}
    seen: set[int] = set()
    todo: list[tuple[BaseException, TracebackType | None]] = [(exc, tb)]
    while todo:
        e, t = todo.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        for frame, _ in traceback.walk_tb(t):
            path = frame.f_code.co_filename
            if path not in files:
                files[path] = _call_site_file(path, frame.f_globals.get("__name__"))
        members = getattr(e, "exceptions", ())
        for nxt in (e.__cause__, e.__context__, *(members if isinstance(members, tuple) else ())):
            if isinstance(nxt, BaseException):
                todo.append((nxt, nxt.__traceback__))
    return files


def _scrub_home(text: str) -> str:
    """`text` with this process's home folder written as `~`.

    The frame paths are already placed; this catches the home folder wherever
    else the text carries it — an `OSError` names the file it could not open,
    and a source line can hold a path literal. Also in its `repr` form, which
    doubles a Windows backslash.
    """
    home = os.path.expanduser("~")
    if len(home) < 2 or not os.path.isabs(home):
        return text
    forms = {home, home.replace("\\", "\\\\")}
    pattern = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    return re.sub(f"(?:{pattern})(?!\\w)", "~", text)
