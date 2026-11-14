"""What wardex can say about itself: the types `wardex.diagnostics()` returns.

Kept apart from `_types.py` on purpose. The adapters file their own status, and
the import rules forbid an adapter module from naming `_types` at all — that is
where the span machinery lives, and an adapter that can name a span id can mint
a parent edge. Nothing here is span machinery: these are plain records of what
was installed and what was found, so they sit at the vocabulary layer, where
both the adapters and the public surface can reach them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ._enums import AdapterState

__all__ = ["AdapterStatus", "Diagnostics", "Loss"]


@dataclass(frozen=True, slots=True)
class AdapterStatus:
    """One adapter's state in this process, and the framework version it found.

    `measured` is whether this wardex release was tested against the installed
    `version`: True or False when it was checked, None when there was nothing to
    check (the adapter is not installed, or its framework is a CLI whose
    version is not read at `init()`). `measured_versions` is what was tested,
    as "0.22.x". `detail` is the line wardex said about this adapter, if it
    said one, so the reason is readable after stderr has scrolled away.
    """

    name: str
    state: AdapterState
    framework: str | None = None
    version: str | None = None
    measured: bool | None = None
    measured_versions: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Loss:
    """One kind of thing wardex did not record, counted since the process started.

    `reason` is a closed vocabulary (`queue_overflow`, `internal_sdk_error`,
    ...) and `unit` says what was counted (`span`, `request`, `adapter`, ...).
    `nature` is `"loss"` when the thing is gone, `"partial"` when it was
    recorded incomplete, and `"none"` when nothing was lost because the drop
    was asked for (a `before_send_envelope` hook that returned None).
    `meaning` and `next_step` are the sentences wardex uses for this reason
    everywhere it reports one. `where` breaks the reason down by the place in
    wardex that counted it (`internal_sdk_error` by where it raised, an adapter
    that did not install as `adapters.<name>`), across all of the reason's
    units.

    The counts only grow: sending them to a receiver does not reset them, so
    two readings can be subtracted to get what was lost between them. They are
    exact for this process whether or not they were sent. `undeliverable` is
    how many of `count` never will be: the report carrying them was dropped
    because more reports awaited the receiver's confirmation than wardex holds
    (counted itself as `ledger_overflow`).
    """

    reason: str
    unit: str
    count: int
    nature: str
    meaning: str = ""
    next_step: str = ""
    where: tuple[tuple[str, int], ...] = ()
    undeliverable: int = 0


@dataclass(frozen=True, slots=True)
class Diagnostics:
    """What `wardex.diagnostics()` returns: wardex's own account of this process.

    `initialized` is True between `init()` and `close()`. `adapters` is what
    became of each adapter this release ships at the most recent `init()`, and
    it is kept after `close()`, so a check made once the run is over still
    reads it; before the first `init()` it is empty. `losses` counts every
    kind of loss since the process started, one entry per reason and unit
    that has happened at least once; a forked child starts its own from zero.
    `config_problems` is every line the most recent `init()` said about its
    configuration on stderr -- a variable it could not use, a kill switch, no
    destination -- so a check made later reads why `initialized` is False.

    The rest is what a bug report needs, and none of it is a secret or a
    payload. `wardex_version`, `python_version` and `platform` say what runs;
    `native` is `"loaded"` or why the compiled module is unavailable.
    `config` is the most recent `init()`'s resolved configuration as
    `(setting, value)` pairs, with keys, header values and credentials
    masked. `transport` is the exporter's class and `destination` where it
    sends, as scheme, host, port and path only. `export_tracked` says whether
    the exporter reports on its own deliveries; when it does,
    `last_export_success` (Unix seconds), `last_export_error` and
    `retry_queued` (requests held for a retry) are what it reports, each None
    when it has nothing to say.
    `capture_policy` (the `capture_mode` in force), `interceptors` (the byte
    seams installed), `patched` (each function wardex replaced, as
    `module.Class.attribute`) and `worker_threads` (wardex's threads alive
    now) describe what wardex is doing inside the process. `report()` renders
    all of it as plain text.
    """

    initialized: bool
    adapters: tuple[AdapterStatus, ...] = ()
    losses: tuple[Loss, ...] = ()
    config_problems: tuple[str, ...] = ()
    wardex_version: str = ""
    python_version: str = ""
    platform: str = ""
    native: str = ""
    config: tuple[tuple[str, str], ...] = ()
    transport: str = ""
    destination: str = ""
    last_export_success: float | None = None
    last_export_error: str | None = None
    export_tracked: bool = False
    retry_queued: int | None = None
    capture_policy: str = ""
    interceptors: tuple[str, ...] = ()
    patched: tuple[str, ...] = ()
    worker_threads: tuple[str, ...] = ()

    def report(self) -> str:
        """Everything above as plain text, safe to paste into a bug report:
        no keys, no header values, no captured payloads. One fact per line, in
        a fixed order, so two reports compare line by line."""
        yes_no = "yes" if self.initialized else "no"
        lines = [
            "wardex diagnostics",
            f"wardex-sdk: {self.wardex_version or 'unknown'}",
            f"python: {self.python_version or 'unknown'}",
            f"platform: {self.platform or 'unknown'}",
            f"native module: {self.native or 'unknown'}",
            f"initialized: {yes_no}",
            f"capture mode: {self.capture_policy or 'unknown'}",
            f"transport: {self.transport or 'none'}",
            f"destination: {self.destination or 'none'}",
            f"last export: {_last_export(self)}",
            f"worker threads: {', '.join(self.worker_threads) or 'none'}",
            f"interceptors ({len(self.interceptors)}): {', '.join(self.interceptors) or 'none'}",
            f"patched symbols ({len(self.patched)}):",
        ]
        lines += [f"  {name}" for name in self.patched]
        lines.append(f"adapters ({len(self.adapters)}):")
        lines += [f"  {_adapter_line(status)}" for status in self.adapters]
        lines.append(f"config ({len(self.config)}):")
        lines += [f"  {name} = {value}" for name, value in self.config]
        lines.append(f"config problems ({len(self.config_problems)}):")
        lines += [f"  {line}" for line in self.config_problems]
        lines.append(f"losses ({len(self.losses)}):")
        lines += [f"  {_loss_line(loss)}" for loss in self.losses]
        return "\n".join(lines) + "\n"


def _last_export(d: Diagnostics) -> str:
    if not d.export_tracked:
        return "not tracked by this transport"
    parts = []
    if d.last_export_success is not None:
        parts.append(f"succeeded at {_utc(d.last_export_success)}")
    if d.last_export_error is not None:
        parts.append(f"last failure: {d.last_export_error}")
    if d.retry_queued:
        parts.append(f"{d.retry_queued} request(s) held for a retry")
    return "; ".join(parts) or "none yet"


def _utc(unix: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(unix))


def _adapter_line(status: AdapterStatus) -> str:
    line = f"{status.name}: {status.state.value}"
    if status.framework:
        line += f" ({status.framework} {status.version or 'version unknown'}"
        if status.measured is not None:
            line += ", tested" if status.measured else ", untested release"
        line += ")"
    return line


def _loss_line(loss: Loss) -> str:
    where = ", ".join(f"{place} x{n}" for place, n in loss.where)
    line = f"{loss.reason}: {loss.count} {loss.unit}(s), {loss.nature}"
    if where:
        line += f" (at {where})"
    return f"{line}. {loss.next_step}".rstrip()
