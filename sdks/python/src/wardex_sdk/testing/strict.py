"""`assert_no_capture_gaps()`: fail when wardex cannot say it saw everything.

wardex never raises into the application it observes, so in production a gap
in the record — a span the buffer evicted, an adapter that did not install, an
adapter running on a framework release this wardex was not tested against, an
error wardex swallowed inside itself — is said on stderr and counted, and the
application carries on. That is right for production and wrong for a test
suite: there, the change that caused the gap is in front of the person who can
undo it, and a line on stderr is easy to scroll past. This module turns the
same account into a failure, at a moment the caller chooses.

THE CALLER CHOOSES THE MOMENT, because wardex cannot. A gap can happen at any
point of a run, and the only process-wide hook left at the end — `atexit` —
cannot change the exit code. So the check is a call: one line at the end of a
test or a smoke script, or `pytest --wardex-strict`, which makes that call
around every test (see `pytest_plugin.py`).

WHAT COUNTS AS A GAP, and what does not:

- every `loss` and `partial` in `wardex.diagnostics().losses` — including an
  internal wardex error, which is wardex's own fault and is reported as such;
- an adapter that found its framework and did not install (`UNSUPPORTED`,
  `FAILED`), or whose framework import is answered by a local package
  (`SHADOWED`);
- an adapter installed on a framework release this wardex was not tested
  against: it records, and parts of what it records may be wrong.

Not a gap: a drop the host asked for (`nature == "none"`, a
`before_send_envelope` hook that returned None), and an adapter whose framework
is absent or that `enabled=` left out. Anything else can be accepted only by
naming it in `allow`.

No third-party import: like the rest of this package, it reads the same under
pytest, under unittest and in a script.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .. import _runtime, diagnostics
from .._diagnostics import AdapterStatus, Diagnostics, Loss
from .._enums import AdapterState

__all__ = [
    "UNTESTED_FRAMEWORK_VERSION",
    "CaptureGap",
    "CaptureGapError",
    "assert_no_capture_gaps",
    "capture_gaps",
]

#: The reason an adapter installed on an untested framework release is reported under.
UNTESTED_FRAMEWORK_VERSION = "untested_framework_version"

_ADAPTER_REASONS = {
    AdapterState.UNSUPPORTED: "adapter_unsupported",
    AdapterState.FAILED: "adapter_failed",
    AdapterState.SHADOWED: "adapter_shadowed",
}


@dataclass(frozen=True, slots=True)
class CaptureGap:
    """One thing wardex did not record, or could not vouch for.

    `reason` is a loss reason (`queue_overflow`, ...) or one of the adapter
    reasons (`adapter_unsupported`, `adapter_failed`, `adapter_shadowed`,
    `untested_framework_version`). `subject` is what it is about: the unit
    counted for a loss, the adapter's name for an adapter. `count` is how many
    since the baseline, None for an adapter. `line` is the whole sentence: what
    happened, how many, and what to do.
    """

    reason: str
    subject: str
    count: int | None
    line: str

    def allowed_by(self, allow: frozenset[str]) -> bool:
        """`allow` names a reason, or a reason scoped to one subject (`reason:subject`)."""
        return self.reason in allow or f"{self.reason}:{self.subject}" in allow


class CaptureGapError(AssertionError):
    """Raised by `assert_no_capture_gaps`; `gaps` holds what was found."""

    def __init__(self, gaps: tuple[CaptureGap, ...], *, header: str) -> None:
        self.gaps = gaps
        super().__init__(render(gaps, header=header))


def capture_gaps(
    *, since: Diagnostics | None = None, allow: Iterable[str] = ()
) -> tuple[CaptureGap, ...]:
    """Every gap now, or every gap that appeared after `since`, minus what `allow` names.

    `since` is an earlier `wardex.diagnostics()` reading. Loss counts are
    subtracted, so only what was lost after it is reported; an adapter state
    already present in it is not reported again.
    """
    return gaps_between(diagnostics(), since=since, allow=allow)


def gaps_between(
    now: Diagnostics, *, since: Diagnostics | None, allow: Iterable[str] = ()
) -> tuple[CaptureGap, ...]:
    """The gaps in `now` that are not in `since`: `capture_gaps` on a reading
    the caller already holds, so the same reading can be its next baseline and
    nothing recorded in between two checks falls between them."""
    allowed = frozenset(allow)
    found = _adapter_gaps(now, since) + _loss_gaps(now, since)
    return tuple(gap for gap in found if not gap.allowed_by(allowed))


def assert_no_capture_gaps(*, since: Diagnostics | None = None, allow: Iterable[str] = ()) -> None:
    """Raise `CaptureGapError` if wardex missed anything it should have recorded.

    Call it at the end of a test or a smoke run, before or after
    `wardex.close()`: the loss counts and the last `init()`'s adapter states
    both survive it. It also fails when wardex was never initialized in this
    process, because then there was nothing to check, and a pass would say that
    wardex saw everything when it saw nothing.
    """
    if _runtime.runtime().client is None:
        raise CaptureGapError(
            (),
            header=(
                "wardex was never initialized in this process, so there was nothing to "
                "check. Call wardex.init() in the code under test before asserting that "
                "it recorded everything"
            ),
        )
    gaps = capture_gaps(since=since, allow=allow)
    if gaps:
        raise CaptureGapError(gaps, header="wardex could not say it recorded everything")


def render(gaps: tuple[CaptureGap, ...], *, header: str) -> str:
    """The failure text: one header, then one line per gap, then how to accept one."""
    if not gaps:
        return header
    lines = [f"{header}:"]
    lines += [f"  - {gap.line}" for gap in gaps]
    example = gaps[0].reason
    lines.append(
        f"To accept one on purpose, pass allow=({example!r},) or "
        f"--wardex-strict-allow={example} (scope it with {example}:<subject>)."
    )
    return "\n".join(lines)


def _adapter_gaps(now: Diagnostics, since: Diagnostics | None) -> tuple[CaptureGap, ...]:
    seen = {_identity(s) for s in since.adapters} if since is not None else set()
    gaps = []
    for status in now.adapters:
        reason = _adapter_reason(status)
        if reason is None or _identity(status) in seen:
            continue
        line = status.detail or f"{status.name} adapter is {status.state.value}"
        gaps.append(CaptureGap(reason, status.name, None, line))
    return tuple(gaps)


def _adapter_reason(status: AdapterStatus) -> str | None:
    if status.state in _ADAPTER_REASONS:
        return _ADAPTER_REASONS[status.state]
    if status.state is AdapterState.INSTALLED and status.measured is False:
        return UNTESTED_FRAMEWORK_VERSION
    return None


def _identity(status: AdapterStatus) -> tuple[object, ...]:
    return (status.name, status.state, status.version, status.measured)


def _loss_gaps(now: Diagnostics, since: Diagnostics | None) -> tuple[CaptureGap, ...]:
    before = {(loss.reason, loss.unit): loss for loss in since.losses} if since else {}
    down = [s.name for s in now.adapters if s.state in _ADAPTER_REASONS]
    gaps = []
    for loss in now.losses:
        if loss.nature == "none":
            continue
        earlier = before.get((loss.reason, loss.unit))
        places = _where_delta(loss, earlier)
        echoed = {
            p.format(name)
            for p in _ADAPTER_ECHOES.get((loss.reason, loss.unit), ())
            for name in down
        }
        count = loss.count - (earlier.count if earlier is not None else 0)
        count -= sum(n for place, n in places.items() if place in echoed)
        if count <= 0:
            continue
        places = {place: n for place, n in places.items() if place not in echoed}
        gaps.append(
            CaptureGap(loss.reason, loss.unit, count, _loss_line(loss, earlier, count, places))
        )
    return tuple(gaps)


#: The ledger counts an adapter that did not install as well: as
#: `instrumentation_unavailable` under `adapters.<name>`, and, when it raised,
#: as `internal_sdk_error` under the guard it raised through. Those are the
#: adapter's own gap, already said once under its name with the line that
#: says what to do, so they are not said again: a check that accepts
#: `adapter_failed:<name>` must not then fail on the same cause twice. What is
#: left of a count after them is still a gap.
_ADAPTER_ECHOES = {
    ("instrumentation_unavailable", "adapter"): ("adapters.{}",),
    ("internal_sdk_error", "failure"): ("adapters.{}.load", "adapters.{}.install"),
}


def _loss_line(loss: Loss, earlier: Loss | None, count: int, places: dict[str, int]) -> str:
    # The vocabulary's sentences end in a period; this line puts its own.
    what = (loss.meaning or loss.reason).rstrip(".")
    line = f"{loss.reason}: {count} {loss.unit}(s). {what}"
    if places:
        line += " (at " + ", ".join(f"{place} x{n}" for place, n in places.items()) + ")"
    line += "."
    if loss.next_step:
        line += f" {loss.next_step.rstrip('.')}."
    lost = loss.undeliverable - (earlier.undeliverable if earlier is not None else 0)
    if lost > 0:
        line += (
            f" The count is exact here, but {lost} of them will never reach the receiver: "
            "the report carrying them was dropped (ledger_overflow)."
        )
    return line


def _where_delta(loss: Loss, earlier: Loss | None) -> dict[str, int]:
    before = dict(earlier.where) if earlier is not None else {}
    return {
        place: n - before.get(place, 0) for place, n in loss.where if n - before.get(place, 0) > 0
    }
