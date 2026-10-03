"""Where the path of one of the host's source files may go when it leaves the process.

Two texts carry the host's files: a span's call site (`code.file.path`) and an
exception's stack trace. One rule places both, so the OS user name and every
folder above the import root stay on the machine while the package path a
developer needs to find the code leaves with the span. A stack trace carries
paths in more places than its frame lines — an `OSError` names the file it
could not open, a source line can hold a path literal — so it also has this
process's home folder written as `~` wherever it starts a path. And a stack
trace is recorded on every span its exception leaves, so how much of it is
formatted is bounded here too.

Imports nothing from the package, so any layer that has to place a host path
can ask this module without importing the span machinery.
"""

from __future__ import annotations

import os
import re
import sys
import traceback
from collections.abc import Iterator, Mapping
from types import MappingProxyType, TracebackType
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


#: The frames each traceback in a stack trace keeps: the ones nearest where its
#: exception was raised. Every span an exception leaves records it, so with no
#: bound a recursion through a decorated function, N spans deep, formatted N
#: traces of up to 2N frames each while the host was unwinding: seconds of the
#: host's time and megabytes of text for one failure, the cost growing with the
#: square of the depth. With it a span pays for at most this many frames per
#: traceback, however deep the stack.
_MAX_FRAMES = 64


def _stacktrace(exc: Exception, tb: TracebackType | None) -> str:
    """The traceback as Python prints it, with every frame's file placed by the
    call-site rule: a path relative to its package's import root, or the bare
    file name, never an absolute path. Locals are never captured.

    A file is placed on the lines where the formatter writes one, each frame's
    `File "…", line N` and a `SyntaxError`'s own location, and nowhere else:
    a message or a source line that itself says `File "<path>", line N` is the
    host's text and reaches the trace as written, the same text that
    `exception.message` carries.

    Each traceback in it, a chained one included, keeps its `_MAX_FRAMES`
    frames nearest the raise, the frames Python's own `limit=-_MAX_FRAMES`
    prints, and a cut one says how many earlier frames it left out. The span's
    own traceback is cut before anything is read: the formatter is handed it
    from its first kept frame on, so its cost is the kept frames, not the
    depth. A chained one is cut by the formatter's `limit`, which still steps
    through the frames it drops.
    """
    head, left_out = _kept_frames(tb)
    summary = traceback.TracebackException(type(exc), exc, head, limit=-_MAX_FRAMES)
    _own_stacks(summary, exc, left_out, _frame_files(exc, head))
    return _scrub_home("".join(summary.format()))


def _placed(path: str, files: Mapping[str, str]) -> str:
    """`path` placed with the module its frame ran under, or as its bare name."""
    return files.get(path) or _call_site_file(path, None)


def _kept_frames(tb: TracebackType | None) -> tuple[TracebackType | None, int]:
    """Where `tb`'s `_MAX_FRAMES` frames nearest the raise begin, and how many come before.

    Follows `tb_next` and nothing else, so no frame, line or code position is
    read for a frame that is not kept.
    """
    depth = 0
    node = tb
    while node is not None:
        depth += 1
        node = node.tb_next
    left_out = max(depth - _MAX_FRAMES, 0)
    head = tb
    for _ in range(left_out):
        head = head.tb_next  # type: ignore[union-attr]  # depth counted it
    return head, left_out


#: Formatted frames, looked up by everything the stdlib's formatter reads; see `_Frames`.
_FORMATTED: dict[tuple[Any, ...], str] = {}
#: How many formatted frames `_FORMATTED` holds before it starts over.
_FORMATTED_MAX = 1024


class _Frames(traceback.StackSummary):
    """One traceback's kept frames, printed the way the stdlib prints them, with three changes.

    Each frame's file is placed (`files`, else the bare name) on the frame's
    own `File "…", line N` line, and nothing else of the frame is touched: its
    source line is the host's code as written. A cut traceback starts with a
    line saying how many earlier frames it left out. And a frame is formatted
    once, not once per span: the stdlib's frame formatting is most of what
    recording costs (from 3.13 it parses the source line twice to place the
    `^^^` markers), and when one exception leaves N nested spans the same
    frames come back N times. The formatted text is looked up by everything
    the formatter reads — the file, the line and column positions, the
    function name and the source lines — so the same frame prints the same
    text and a changed source line formats afresh.
    """

    left_out = 0
    files: Mapping[str, str] = MappingProxyType({})

    def format(self, **kwargs: Any) -> list[str]:
        # The stdlib formats with the files as recorded, so it folds a
        # recursion's repeats exactly as it would; each frame's text is then
        # placed. Every frame's text starts with its `File "<file>", line `,
        # so matching the frames' own files against that start finds the one
        # line to place; the longest goes first, in case one file's name
        # starts with another's.
        paths = sorted({frame.filename for frame in self}, key=len, reverse=True)
        lines = [self._place(text, paths) for text in super().format(**kwargs)]
        if self.left_out:
            plural = "s" if self.left_out > 1 else ""
            lines.insert(0, f"  [{self.left_out} earlier frame{plural} not recorded]\n")
        return lines

    def _place(self, text: str, paths: list[str]) -> str:
        """One frame's formatted text with the file on its first line placed.

        Text that starts with no frame's file is left as it is: a `[Previous
        line repeated …]` line, and a frame 3.13+ prints as `<stdin>`, which
        names no folder.
        """
        for path in paths:
            start = f'  File "{path}", line '
            if text.startswith(start):
                return f'  File "{_placed(path, self.files)}", line {text[len(start) :]}'
        return text

    def format_frame_summary(self, frame_summary: traceback.FrameSummary, **kwargs: Any) -> str:
        # 3.11+ only: 3.10's `format` formats each frame inline, cheaply.
        if frame_summary.locals is not None:  # never captured here; not looked up if it were
            return super().format_frame_summary(frame_summary, **kwargs)
        key = (
            frame_summary.filename,
            frame_summary.lineno,
            getattr(frame_summary, "end_lineno", None),
            getattr(frame_summary, "colno", None),
            getattr(frame_summary, "end_colno", None),
            frame_summary.name,
            frame_summary.line,
            getattr(frame_summary, "_original_lines", None),  # what 3.13+ reads
            getattr(frame_summary, "_original_line", None),  # what 3.11 and 3.12 read
            tuple(sorted(kwargs.items())),
        )
        text = _FORMATTED.get(key)
        if text is None:
            text = super().format_frame_summary(frame_summary, **kwargs)
            if len(_FORMATTED) >= _FORMATTED_MAX:
                _FORMATTED.clear()
            _FORMATTED[key] = text
        return text


def _own_stacks(
    summary: traceback.TracebackException,
    exc: BaseException,
    n: int,
    files: Mapping[str, str],
) -> None:
    """Give every traceback in `summary` its `_Frames`, with how many frames it
    left out and where its files go, and place a `SyntaxError`'s own location.

    `summary` mirrors the exception graph it was built from — a `__cause__`, a
    `__context__` and a group's members each have their own summary, absent
    where the formatter will not print one — so the two are walked together.
    """
    todo: list[tuple[Any, BaseException, int]] = [(summary, exc, n)]
    while todo:
        s, e, left_out = todo.pop()
        frames = _Frames(s.stack)
        frames.left_out = left_out
        frames.files = files
        s.stack = frames
        if issubclass(type(e), SyntaxError):
            _place_location(s, files)
        pairs = [(s.__cause__, e.__cause__), (s.__context__, e.__context__)]
        members = getattr(s, "exceptions", None)  # 3.11+, and set only for a group
        if members:
            pairs += zip(members, e.exceptions, strict=False)
        for sub, sub_exc in pairs:
            if sub is not None and sub_exc is not None:
                todo.append((sub, sub_exc, _kept_frames(sub_exc.__traceback__)[1]))


def _place_location(s: Any, files: Mapping[str, str]) -> None:
    """Place the `File "<file>", line N` a `SyntaxError` prints for where it was found.

    Every supported Python formats that line first, so the summary's own
    `format_exception_only` is wrapped to place the first line it gives, and
    only that line: the source and the message after it are the host's text.
    Setting the summary's `filename` would place it too, but from 3.14 the
    formatter may open that file (when the error holds no copy of its source)
    to look for a misspelt keyword, and a bare name would open whatever file
    of that name the working folder holds.
    """
    if s.lineno is None:  # no location line; the file goes in the message as `(<file>)`
        return
    path = s.filename or "<string>"
    start = f'  File "{path}", line '
    stdlib = s.format_exception_only

    def format_exception_only(**kwargs: Any) -> Iterator[str]:
        for i, line in enumerate(stdlib(**kwargs)):
            if i == 0 and line.startswith(start):
                line = f'  File "{_placed(path, files)}", line {line[len(start) :]}'
            yield line

    s.format_exception_only = format_exception_only


def _frame_files(exc: BaseException, head: TracebackType | None) -> dict[str, str]:
    """Each kept frame's file, placed with the module its frame ran under.

    Walks the whole graph the formatter prints — `__cause__`, `__context__` and
    an exception group's members — because a chained traceback's frames are
    paths too. A file this walk misses still leaves as its bare name.
    """
    files: dict[str, str] = {}
    seen: set[int] = set()
    todo: list[tuple[BaseException, TracebackType | None]] = [(exc, head)]
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
                todo.append((nxt, _kept_frames(nxt.__traceback__)[0]))
    return files


#: What may stand right before the home folder for it to START a path: the
#: start of the text, whitespace, a quote, an opening bracket, a `key=value` or
#: list separator, or a `file://` scheme. Anything else, a letter, a dot or a
#: slash included, means the match lies inside a longer path, such as a copy of
#: the home folder under `/backup`.
_HOME_STARTS = r"(?:\A|(?<=[\s'\"`(\[{<=,;:])|(?<=file://))"

#: What may stand right after it for the match to BE the home folder: anything
#: that cannot go on a folder's name. A letter, a digit, `_`, `-` or `+` can
#: (`alice-old`, `alice2`), and so can a `.` with one of those after it
#: (`alice.bak`): that folder is a different one. Everything else ends the
#: path — a separator, a quote, a space, a bracket, the end of the text, and
#: the comma or full stop of the sentence the path stands in.
_HOME_ENDS = r"(?![\w+\-]|\.[\w+\-])"


def _scrub_home(text: str) -> str:
    """`text` with this process's home folder written as `~` where it starts a path.

    The frame paths are already placed; this catches the home folder where else
    the text carries it — an `OSError` names the file it could not open, a
    message names a folder in a sentence, and a source line can hold a path
    literal. Also in its `repr` form, which doubles a Windows backslash.

    Only where the text plainly names the home folder, so that no other folder
    is claimed to be it: one whose name merely begins with the home folder's,
    or a copy of it under another folder, is left as written. A space, a comma
    or a full stop after the home folder ends it, because in an exception's
    message that is the sentence going on (`permission denied for
    /Users/alice.`) far more often than a folder's name, and the OS user name
    must not leave with the sentence. The one folder that costs is a sibling
    named like the home folder plus a space (`/Users/alice 2`), written `~ 2`.
    """
    seps = os.sep + (os.altsep or "")
    home = os.path.expanduser("~").rstrip(seps)
    if len(home) < 2 or not os.path.isabs(home):
        return text
    forms = {home, home.replace("\\", "\\\\")}
    if not any(f in text for f in forms):
        return text  # the common case, and a substring search costs far less than the pattern
    pattern = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    return re.sub(f"{_HOME_STARTS}(?:{pattern}){_HOME_ENDS}", "~", text)
