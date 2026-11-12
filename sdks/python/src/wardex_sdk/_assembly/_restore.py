"""The calling task's ambient carriers as plain (variable, value) pairs, for an exit with no frames.

`Unit.activate()` installs two ContextVars on the calling task: the scope fork that carries the
unit's span context, and the ambient-unit entry `UnitRegistry.current()` reads. Its exit resets
both through Python frames: the activation's generator, `_Carrier.remove`, and `activate_span`'s
generator under it. An exit that meets the interpreter's recursion limit cannot push those frames,
because every frame a handler pushes raises the same `RecursionError` again. Left installed, the
dead unit stays the parent of whatever the task opens next, an edge to a span that never ships.

So a caller takes this restore point BEFORE it activates, while frames are free, and at the limit
writes it back with `var.set(value)` for each pair: a C call, which pushes no frame. It then sets
`abandoned` on the carrier `activate()` yielded, a slot store that pushes no frame either, so the
suspended activation, finalized when the exit drops it, skips its own reset. Without that the
finalizer needed about five frames there, and at the tightest limits the interpreter reported it
as an unraisable `RecursionError` on the host's stderr.

Values rather than Tokens, because the Tokens live inside the activation's generators where no
caller can reach them. Setting the earlier value is what resetting those Tokens does, on the task
that took the point. The scope is read through `get_current_scope()`, the same call
`activate_span` makes before it forks, so the value written back is the one its reset restores.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from .. import _hub
from ._units import _ambient_unit


def ambient_restore_point() -> tuple[tuple[ContextVar[Any], Any], ...]:
    """What `Unit.activate()` will overwrite on this task, as `(var, value)` pairs.

    Take it on the task that activates, before the activation; write it back only on that task,
    and only where the activation's own exit cannot run.
    """
    return (
        (_hub._current_scope, _hub.get_current_scope()),
        (_ambient_unit, _ambient_unit.get()),
    )
