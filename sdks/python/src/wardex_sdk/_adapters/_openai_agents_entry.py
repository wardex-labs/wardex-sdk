"""The OpenAI Agents run's ENTRY: the one value the framework's trace never carries.

`Runner.run(conversation_id=...)` names the provider-held conversation a run
continues. The framework hands that id to every model call (it is each
request's `conversation`) and never to its trace, so no `TracingProcessor`
callback can read it, and the run's own spans used to carry no conversation
while its LLM calls carried this one.

The three public entry points — `Runner.run`, `run_sync` and `run_streamed` —
are wrapped to READ the argument, and do nothing else: no argument, result or
exception is changed. They are the framework's documented surface, not its run
loop, which is why this is not the internal patching the adapter's own
docstring rejects. Each call puts its id in `RUN_REQUEST` for the length of the
call, on the task or thread the host called from. Every callback of the run
fires there or on a task created from there — `run_sync`'s task on the
thread's loop, `run_streamed`'s run-loop task, the model, tool and guardrail
tasks — and each of those copied the context when it was created, so a
callback reads the id of the call that started its run and never another's:
two runs gathered on one loop, or on two threads, each read their own.

Precedence is the `group_id` rule, unchanged (`framework_conversation`): the
host's own `wardex.conversation(...)` wins over everything. A `group_id` stays
the run's conversation when one is set, and the requests' own id then rides
along on each LLM call (`wardex.openai.conversation_id`, written by the
semantics layer). Only with no `group_id` does the call's id become the run's
conversation.
"""

from __future__ import annotations

import contextvars
import functools
from collections.abc import Callable
from inspect import iscoroutinefunction, signature
from typing import Any

from .._assembly import ConversationContext, report_once
from ._context import AdapterContext
from ._conversation import framework_conversation

#: The run's entry points, and whether each is a coroutine function.
ENTRY_POINTS: tuple[tuple[str, bool], ...] = (
    ("run", True),
    ("run_sync", False),
    ("run_streamed", False),
)


class RunRequest:
    """One entry-point call's conversation id. Compared by IDENTITY: the run
    whose root took it records the object, so a top-level agent can tell "my
    call's id is on my run's root already" from "my run's root is a trace the
    host opened around this call"."""

    __slots__ = ("conversation_id",)

    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id


#: The entry-point call the current task or thread is inside, or None.
RUN_REQUEST: contextvars.ContextVar[RunRequest | None] = contextvars.ContextVar(
    "wardex_openai_agents_run_request", default=None
)


# -- what the open of a run and of its top-level agents reads ----------------


def run_conversation(
    ctx: AdapterContext, group_id: str | None
) -> tuple[ConversationContext | None, str | None, RunRequest | None]:
    """`(the conversation the run's root opens with, the group id the host
    shadowed, the entry-point call this root took its id from)`.

    A `group_id` is the run's word when set, exactly as before this hook
    existed. With none, the `conversation_id` the run was started with is,
    under the same rule: the host's own conversation wins over it, and an
    absent id opens no conversation.
    """
    request = RUN_REQUEST.get()
    if group_id is None and request is not None:
        conversation, _ = framework_conversation(
            ctx, request.conversation_id, shadowed_counter="conversation_id_shadowed_by_host"
        )
        return conversation, None, request
    conversation, shadowed = framework_conversation(
        ctx, group_id, shadowed_counter="group_id_shadowed_by_host"
    )
    return conversation, shadowed, request


def call_conversation(ctx: AdapterContext, run: dict[str, Any]) -> ConversationContext | None:
    """The conversation a TOP-LEVEL agent states for its entry-point call, or None.

    None when the call named no id, and when the run's root already took this
    call's id: the agent then inherits it like every other child. What is left
    is a run whose root is a trace the HOST opened around the call — `with
    trace(...)` holding one `Runner.run` or several gathered ones — which fires
    no `on_trace_start` inside the call. That root is the host's and belongs to
    no one call, so each call's top-level agents state its id, and their tools,
    handoffs and LLM calls inherit it. The precedence is the root's: a
    `group_id` the host's trace states wins (`run["stated"]`, set at the root's
    open), and so does the host's own `wardex.conversation(...)`.
    """
    request = RUN_REQUEST.get()
    if request is None or run.get("request") is request:
        return None
    if run.get("stated"):
        ctx.count("conversation_id_shadowed_by_group_id")
        return None
    conversation, _ = framework_conversation(
        ctx, request.conversation_id, shadowed_counter="conversation_id_shadowed_by_host"
    )
    return conversation


# -- the hook ------------------------------------------------------------------


def install_entry_hook(ctx: AdapterContext, live: Callable[[], bool]) -> None:
    """Group 3 of the adapter's probe, declined on its own without touching
    the processor.

    Without it every span still ships; what is lost is a run's
    `conversation_id` on the run's own spans, which then reaches its LLM calls
    alone (they read it off the request). Through `ctx.patches`, so uninstall
    hands each classmethod back by identity and a patch another library laid
    over ours is left in place. `live` answers whether the adapter is still
    installed: a wrapper someone kept a reference to only passes calls through
    once it is not.
    """
    run_mod = None
    surface = None
    with ctx.guard("entry_surface"):
        import agents.run as run_mod

        surface = _entry_surface(run_mod)
    if surface is None:
        report_once(
            "openai-agents adapter: Runner entry points unrecognized; a run's "
            "conversation_id will reach its LLM calls but not its agent, tool and "
            "handoff spans",
            key="adapters.openai_agents.unsupported_entry_surface",
        )
        ctx.count("unsupported_entry_surface")
        return
    state_cls = getattr(run_mod, "RunState", None)
    state_cls = state_cls if isinstance(state_cls, type) else None
    runner = run_mod.Runner
    for name, awaited in ENTRY_POINTS:
        reader = _Reader(ctx, live, state_cls, surface[name])
        ctx.patches.patch(runner, name, _wrap(vars(runner)[name], reader, awaited=awaited))


def _position(names: list[str], name: str) -> int | None:
    return names.index(name) if name in names else None


def _entry_surface(run_mod: Any) -> dict[str, tuple[int | None, int | None]] | None:
    """Where each entry point takes `input` and `conversation_id` positionally
    (None: by keyword only), or None when the surface is not the one measured.

    Each is a classmethod in `Runner`'s OWN namespace — the raw descriptor is
    what gets wrapped and handed back — `run` a coroutine function and the
    other two plain ones, each naming both parameters. `run_streamed` takes
    `conversation_id` positionally as well (measured on 0.22), which is why a
    position is read off the signature rather than assumed.
    """
    runner = getattr(run_mod, "Runner", None)
    if not isinstance(runner, type):
        return None
    out: dict[str, tuple[int | None, int | None]] = {}
    for name, awaited in ENTRY_POINTS:
        raw = vars(runner).get(name)
        if not isinstance(raw, classmethod) or iscoroutinefunction(raw.__func__) is not awaited:
            return None
        params = list(signature(raw.__func__).parameters.values())[1:]
        if not {"input", "conversation_id"} <= {p.name for p in params}:
            return None
        positional = [
            p.name for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        out[name] = (_position(positional, "input"), _position(positional, "conversation_id"))
    return out


class _Reader:
    """What one wrapped entry point needs to read a call: the adapter's
    context and liveness, the framework's `RunState`, and where the entry point
    takes `input` and `conversation_id` positionally."""

    __slots__ = ("ctx", "live", "state_cls", "input_at", "conversation_at")

    def __init__(
        self,
        ctx: AdapterContext,
        live: Callable[[], bool],
        state_cls: type | None,
        at: tuple[int | None, int | None],
    ) -> None:
        self.ctx = ctx
        self.live = live
        self.state_cls = state_cls
        self.input_at, self.conversation_at = at

    def request(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> RunRequest | None:
        """The conversation the call names, or None.

        A run resumed from a `RunState` continues the conversation the state
        recorded unless the call names another: the framework's own rule
        (`conversation_id or run_state._conversation_id`), read the same way so
        the resumed half carries the id its requests carry. An absent or empty
        id names nothing, and wardex does not make one up.
        """
        stated = _argument(args, kwargs, "conversation_id", self.conversation_at)
        if not stated and self.state_cls is not None:
            state = _argument(args, kwargs, "input", self.input_at)
            if isinstance(state, self.state_cls):
                stated = getattr(state, "_conversation_id", None)
        if stated is None or stated == "":
            return None
        return RunRequest(str(stated))

    def enter(
        self, original: Any, cls: type, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[list[Any], contextvars.Token[RunRequest | None] | None]:
        """`([the framework's call, ready to make], the token to reset)`.

        The read runs under the adapter's guard: a call whose id could not be
        read runs with none rather than not at all.
        """
        call = functools.partial(original.__get__(None, cls), *args, **kwargs)
        if not self.live():
            return [call], None
        request = None
        with self.ctx.guard("run_entry"):
            request = self.request(args, kwargs)
        return [call], RUN_REQUEST.set(request)

    def leave(self, token: contextvars.Token[RunRequest | None] | None) -> None:
        if token is not None:
            with self.ctx.guard("run_exit"):
                RUN_REQUEST.reset(token)


def _argument(args: tuple[Any, ...], kwargs: dict[str, Any], name: str, at: int | None) -> Any:
    if name in kwargs:
        return kwargs[name]
    if at is not None and at < len(args):
        return args[at]
    return None


def _wrap(original: Any, reader: _Reader, *, awaited: bool) -> classmethod:  # type: ignore[type-arg]
    """The wrapped entry point: the call's id in `RUN_REQUEST`, then the
    framework's own classmethod, bound to the class it was called on.

    The wrapper's frame holds none of the call's arguments while the run
    executes. The framework raises an error whose data it redacted only from
    frames that own no payload — it clears its own locals and drops the
    traceback first — and a wrapper frame still holding `input` and `context`
    would hand them to anything that reads a traceback's locals, the way an
    error tracker does. So the call is made ready in a one-item list, the
    argument names are deleted, and the list is emptied by the expression that
    makes the call.
    """
    if awaited:

        async def entry(cls: type, *args: Any, **kwargs: Any) -> Any:
            box, token = reader.enter(original, cls, args, kwargs)
            del args, kwargs
            try:
                return await box.pop()()
            finally:
                reader.leave(token)

    else:

        def entry(cls: type, *args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
            box, token = reader.enter(original, cls, args, kwargs)
            del args, kwargs
            try:
                return box.pop()()
            finally:
                reader.leave(token)

    functools.update_wrapper(entry, original.__func__)
    return classmethod(entry)
