"""Shared pieces of the Hermes adapter's tests: a fake Hermes install, wardex with the adapter on,
the agent's loop thread, and the exported spans.

Hermes itself cannot be installed here (`_hermes_fake.py` says why), so `hermes_installed` writes
the smallest thing the adapter registry's probe reads as an installed Hermes: a `hermes-agent`
distribution record and an empty `hermes_cli` package beside it, imported as Hermes's own is by
the time it loads a plugin.

`FakeHermes` treats the thread that calls it as Hermes's loop thread, and the adapter keeps the
turn pinned there in a task-local variable. `on_loop` gives each scripted loop a thread and an
empty context of its own, so no test leaks a pinned turn into the next.
"""

from __future__ import annotations

import contextvars
import importlib
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

import wardex_sdk as wardex
from _hermes_fake import FakePluginContext, ScriptedProvider
from _hermes_scripted import FakeHermes
from wardex_sdk import _hub
from wardex_sdk._adapters import _hermes_agent_attach as attach_module
from wardex_sdk._config import AdaptersConfig
from wardex_sdk._diagnostics import AdapterStatus
from wardex_sdk._enums import AdapterName
from wardex_sdk.hermes_agent import register
from wardex_sdk.testing import RecordingTransport, SpanNode, read_spans

HERMES_VERSION = "0.23.3"

_T = TypeVar("_T")


@contextmanager
def hermes_installed(root: Path, *, metadata: bool = True) -> Iterator[Path]:
    """A `hermes-agent` distribution and its `hermes_cli` package on `sys.path`, for as long as
    the block runs. `metadata=False` leaves the distribution record out: Hermes run from a
    checkout or a Nix closure on `PYTHONPATH`."""
    site = root / "fake-hermes-site"
    package = site / "hermes_cli"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    if metadata:
        record = site / f"hermes_agent-{HERMES_VERSION}.dist-info"
        record.mkdir()
        (record / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: hermes-agent\nVersion: {HERMES_VERSION}\n"
        )
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    # Hermes is running when it loads a plugin, so its package is always imported by then.
    importlib.import_module("hermes_cli")
    try:
        yield site
    finally:
        # The suite's teardown closes the client but leaves adapters installed, so an adapter a
        # test's `init()` installed would outlive this install and the next test would find its
        # route already taken. Closing wardex uninstalls it; it is a no-op when nothing is open.
        wardex.close()
        sys.path.remove(str(site))
        sys.modules.pop("hermes_cli", None)
        importlib.invalidate_caches()


def forget_attachments() -> None:
    """Hermes plugin contexts are process-wide facts; each test starts with none."""
    with attach_module._ATTACH_LOCK:
        attach_module._ATTACHED.clear()
        attach_module._REFUSED.clear()


def start_wardex(**kwargs: Any) -> RecordingTransport:
    """`init()` with the Hermes adapter alone and a recording transport."""
    transport = RecordingTransport()
    wardex.init(
        transport=transport,
        adapters=AdaptersConfig(enabled=(AdapterName.HERMES_AGENT,)),
        **kwargs,
    )
    return transport


@contextmanager
def plugin(home: Path | None = None, **kwargs: Any) -> Iterator[FakePluginContext]:
    """One Hermes plugin context with wardex registered on it, unloaded at the end."""
    ctx = FakePluginContext(home=home, **kwargs)
    register(ctx)
    try:
        yield ctx
    finally:
        ctx.unload()


@contextmanager
def hermes(ctx: FakePluginContext, **kwargs: Any) -> Iterator[FakeHermes]:
    """A scripted Hermes over `ctx`, with its provider running."""
    with ScriptedProvider() as provider:
        yield FakeHermes(ctx, provider, **kwargs)


def on_loop(fn: Callable[..., _T], *args: Any, name: str = "hermes-loop") -> _T:
    """Run `fn` as one Hermes loop: a thread of its own in an empty context."""
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn(*args)
        except BaseException as exc:  # handed to the caller below
            box["error"] = exc

    thread = threading.Thread(target=contextvars.Context().run, args=(run,), name=name)
    thread.start()
    thread.join(60)
    if thread.is_alive():
        raise TimeoutError(f"{name} did not finish")
    if "error" in box:
        raise box["error"]
    return box["value"]


def exported(transport: RecordingTransport) -> tuple[SpanNode, ...]:
    """Every span the transport received, after everything buffered was sent."""
    client = _hub.get_client()
    if client is not None:
        client._settle()
    wardex.flush()
    return read_spans([s for e in transport.envelopes for s in e.spans])


def received(transport: RecordingTransport) -> tuple[SpanNode, ...]:
    """What the transport holds now, with no flush from the caller."""
    return read_spans([s for e in transport.envelopes for s in e.spans])


def wait_for(check: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.02)
    return check()


def status() -> AdapterStatus:
    """The Hermes adapter's row in `wardex.diagnostics()`."""
    rows = [a for a in wardex.diagnostics().adapters if a.name == AdapterName.HERMES_AGENT.value]
    assert len(rows) == 1, wardex.diagnostics().adapters
    return rows[0]


def one(nodes: tuple[SpanNode, ...], name: str) -> SpanNode:
    found = [n for n in nodes if n.name == name]
    assert len(found) == 1, (name, sorted(n.name for n in nodes))
    return found[0]


def parent_of(nodes: tuple[SpanNode, ...], node: SpanNode) -> SpanNode | None:
    return next((n for n in nodes if n.span_id == node.parent_id), None)


def extras(node: SpanNode) -> dict[str, Any]:
    """The span's `wardex.hermes_agent.*` attributes, prefix removed."""
    prefix = "wardex.hermes_agent."
    return {k[len(prefix) :]: v for k, v in node.attributes.items() if k.startswith(prefix)}


def typed_usage(node: SpanNode) -> dict[str, Any]:
    """The span's typed `gen_ai.usage.*` attributes: the numbers a backend sums."""
    return {k: v for k, v in node.attributes.items() if k.startswith("gen_ai.usage.")}


def _line(node: SpanNode, quiet: frozenset[str] = frozenset()) -> str:
    status = node.status.name if node.status is not None else "-"
    error = node.attributes.get("error.type")
    if error is not None:
        status += f"({error})"
    strategy = node.strategy.name.lower() if node.strategy is not None else "-"
    line = f"{node.name} | {strategy} {node.confidence} | {status}"
    markers = sorted(m.name.lower() for m in node.limitations if m.name.lower() not in quiet)
    return line + (" | " + ", ".join(markers) if markers else "")


#: The markers a stream cut by an interrupt may or may not carry, by how much of it had arrived.
CUT_STREAM_MARKERS = frozenset(
    {"provider_inferred", "reassembled_from_stream", "stream_usage_unavailable"}
)

#: An expected tree: a span's outline line and its children, as `render` draws them.
Tree = tuple[str, tuple["Tree", ...]]


def _drawn(line: str, kids: list[list[str]], depth: int) -> list[str]:
    # Siblings sort by everything they draw, so threads finishing in another order, or two
    # siblings with the same line and different children, always draw the same outline.
    ordered = sorted(kids)
    return ["  " * depth + line] + [row for lines in ordered for row in lines]


def outline(nodes: tuple[SpanNode, ...], quiet: frozenset[str] = frozenset()) -> str:
    """Every exported tree as indented lines: name, parent source and confidence, status (with
    `error.type`), markers. Markers named in `quiet` are left out: the seam's account of a stream
    cut mid-chunk depends on how much of it had arrived."""
    ids = {n.span_id for n in nodes}

    def draw(node: SpanNode, depth: int) -> list[str]:
        kids = [draw(n, depth + 1) for n in nodes if n.parent_id == node.span_id]
        return _drawn(_line(node, quiet), kids, depth)

    roots = sorted(draw(r, 0) for r in nodes if r.parent_id not in ids)
    return "\n".join(row for lines in roots for row in lines)


def render(*roots: Tree) -> str:
    """The outline the trees in `roots` draw: what `outline` must return for them."""

    def draw(tree: Tree, depth: int) -> list[str]:
        line, kids = tree
        return _drawn(line, [draw(k, depth + 1) for k in kids], depth)

    return "\n".join(row for lines in sorted(draw(r, 0) for r in roots) for row in lines)


def links(transport: RecordingTransport, node: SpanNode, nodes: tuple[SpanNode, ...]) -> list:
    """The span's links as (reason, the linked span's name), in the order the span carries them."""
    by_id = {n.span_id: n.name for n in nodes}
    raw = next(s for e in transport.envelopes for s in e.spans if s.context.span_id == node.span_id)
    return [(link.reason, by_id.get(link.span_id, "<not exported>")) for link in raw.links]
