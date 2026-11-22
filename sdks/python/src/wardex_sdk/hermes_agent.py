"""The Hermes Agent plugin entry: Hermes calls `register(ctx)` when it loads the wardex plugin.

A Hermes plugin's `__init__.py` needs one line, `from wardex_sdk.hermes_agent import register`.
`register` initializes wardex from the environment unless the host already did (the key and the
endpoint are the usual `WARDEX_*` variables), then registers one callback per hook and middleware
kind in `HOOKS` and `MIDDLEWARE` on Hermes's plugin context. Each Hermes turn is then one trace.
No Hermes code changes, and none of Hermes is imported.

`HOOKS` and `MIDDLEWARE` are the names registered, in registration order, for a plugin manifest
(`plugin.yaml`) to list. `register` never raises into Hermes.
"""

from __future__ import annotations

from ._adapters._hermes_agent_attach import attach
from ._adapters._hermes_agent_events import CONTRACT, Surface

__all__ = ["HOOKS", "MIDDLEWARE", "register"]

#: The hook names `register` registers, in order.
HOOKS: tuple[str, ...] = tuple(row.name for row in CONTRACT if row.surface is Surface.HOOK)
#: The execution middleware kinds `register` registers, in order.
MIDDLEWARE: tuple[str, ...] = tuple(
    row.name for row in CONTRACT if row.surface is Surface.MIDDLEWARE
)


def register(ctx: object) -> None:
    """Attach wardex to one Hermes plugin context. Called by Hermes; safe to call twice."""
    from . import diagnostics, flush, init

    attach(ctx, init=init, initialized=lambda: diagnostics().initialized, flush=flush)
