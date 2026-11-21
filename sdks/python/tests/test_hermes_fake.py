"""The fake Hermes host behaves like Hermes v0.23.3 where a plugin can tell.

The Hermes adapter's tests run against these fakes because Hermes itself cannot be installed in
the test environment. A fake that drifts from Hermes would let the adapter pass here and fail in a
real process, so these tests pin what a plugin observes: which thread a callback runs on, which
context it sees, which arguments it receives, and how a middleware chain treats its callbacks.

The plugin under test is `Recorder`, defined here. It does what the adapter will do with a context
variable: set it inside the first `llm_execution` frame and read it everywhere else.
"""

from __future__ import annotations

import contextvars
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

import wardex_sdk as wardex
from _hermes_fake import (
    MIDDLEWARE_V1,
    OBSERVER_V1,
    VALID_HOOKS,
    Answer,
    DownstreamError,
    FakePluginContext,
    Lane,
    Reply,
    ScriptedProvider,
    Usage,
    lane_of,
)
from _hermes_replay import SCENARIOS, Replay, load_fixture
from _hermes_scripted import FakeHermes
from wardex_sdk import _hub
from wardex_sdk._enums import SpanKind, StatusCode
from wardex_sdk.testing import RecordingTransport

PROBE: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "hermes_fake_probe", default=()
)
SET = ("llm",)


@dataclass(frozen=True)
class Seen:
    surface: str  # "hook", "mw" or "attempt"
    name: str
    thread: str
    probe: tuple[str, ...]
    args: tuple[str, ...]
    session: str | None


class Recorder:
    """A plugin that logs every delivery. With `stack`, it pushes one id per execution frame and
    pops it on the way out, as the lab's recording plugin did, so probes compare with fixtures.
    Without it, it sets `PROBE` once, in the first `llm_execution` frame, and never resets it."""

    def __init__(self, *, stack: bool = False) -> None:
        self.stack = stack
        self.seen: list[Seen] = []
        self._lock = threading.Lock()
        self._frames = 0

    def install(self, ctx: FakePluginContext) -> None:
        for hook in sorted(VALID_HOOKS):
            ctx.register_hook(hook, self._hook(hook))
        for kind in ("llm_request", "llm_execution", "tool_request", "tool_execution"):
            ctx.register_middleware(kind, self._middleware(kind))

    def note(self, surface: str, name: str, args: dict[str, Any]) -> None:
        with self._lock:
            self.seen.append(
                Seen(
                    surface,
                    name,
                    threading.current_thread().name,
                    PROBE.get(),
                    tuple(sorted(args)),
                    args.get("session_id"),
                )
            )

    def hooks(self, name: str | None = None) -> list[Seen]:
        return [s for s in self.seen if s.surface == "hook" and name in (None, s.name)]

    def _hook(self, hook: str) -> Callable[..., None]:
        def callback(**kwargs: Any) -> None:
            self.note("hook", hook, kwargs)

        callback.__name__ = f"recorder_{hook}"
        return callback

    def _middleware(self, kind: str) -> Callable[..., Any]:
        def callback(**kwargs: Any) -> Any:
            self.note("mw", kind, kwargs)
            if kind.endswith("_request"):
                return None
            if not self.stack:
                if kind == "llm_execution" and not PROBE.get():
                    PROBE.set(SET)
                return kwargs["next_call"]()
            with self._lock:
                self._frames += 1
                frame = f"{kind.split('_')[0]}:{self._frames}"
            token = PROBE.set((*PROBE.get(), frame))
            try:
                return kwargs["next_call"]()
            finally:
                PROBE.reset(token)

        callback.__name__ = f"recorder_mw_{kind}"
        return callback


class ProbedProvider(ScriptedProvider):
    """Notes the thread and context each request leaves from."""

    def __init__(self, recorder: Recorder, *args: Any) -> None:
        super().__init__(*args)
        self.recorder = recorder

    def send(self, answer: Any = None, *, model: str = "lab-model") -> Any:
        self.recorder.note("attempt", "send", {})
        return super().send(answer, model=model)


def in_loop(fn: Callable[[], Any]) -> Any:
    """Run `fn` as an agent loop thread would: in a context the test's own thread never sees."""
    return contextvars.copy_context().run(fn)


@pytest.fixture
def hermes() -> Any:
    recorder = Recorder()
    ctx = FakePluginContext()
    recorder.install(ctx)
    with ProbedProvider(recorder) as provider:
        yield FakeHermes(ctx, provider), recorder


def _after_first_frame(recorder: Recorder) -> list[Seen]:
    first = next(i for i, s in enumerate(recorder.seen) if s.name == "llm_execution")
    return recorder.seen[first + 1 :]


# -- threads and contexts ----------------------------------------------------------------------


def test_the_context_set_in_the_first_model_frame_reaches_every_thread_copied_after_it(hermes):
    fake, recorder = hermes

    def turn() -> None:
        t = fake.turn(title=True)
        call = fake.model_call(t, tool_calls=["read_file"])
        fake.tool(call, "read_file")
        both = fake.model_call(t, tool_calls=["search", "fetch"])
        fake.parallel_tools(both)
        fake.model_call(t)
        fake.aux(t)
        fake.end(t)

    in_loop(turn)
    title = [s for s in recorder.seen if s.thread == "auto-title"]
    later = [
        s for s in _after_first_frame(recorder) if s.name not in _AUX and s.thread != "auto-title"
    ]
    assert [(s.name, s.probe) for s in recorder.hooks()[:2]] == [
        ("on_session_start", ()),
        ("pre_llm_call", ()),
    ]
    attempts = [s for s in recorder.seen if s.surface == "attempt" and s.thread != "auto-title"]
    assert len(attempts) == 3
    assert all(re.fullmatch(r"Thread-\d+ \(<lambda>\)", s.thread) for s in attempts)
    workers = {s.thread for s in recorder.seen if s.name == "tool_request"}
    assert len(workers) == 3
    assert all(w.startswith("ThreadPoolExecutor-") for w in workers)
    assert {s.thread for s in recorder.hooks("pre_tool_call")} == {
        "hermes-hook-recorder_pre_tool_call"
    }
    assert {s.thread for s in recorder.hooks("on_stream_end")} == {
        "plugin-stream-hook:on_stream_end"
    }
    assert {s.probe for s in later} == {SET}
    assert [s.surface for s in title] == ["attempt"]
    assert title[0].probe == ()
    assert [(s.name, s.probe) for s in recorder.hooks() if s.name in _AUX] == [
        ("pre_auxiliary_call", ()),
        ("post_auxiliary_call", ()),
    ]


_AUX = ("pre_auxiliary_call", "post_auxiliary_call")


def test_post_tool_call_lands_after_the_frame_on_the_sequential_path_and_inside_on_the_parallel():
    recorder = Recorder(stack=True)
    ctx = FakePluginContext()
    recorder.install(ctx)
    with ScriptedProvider() as provider:
        fake = FakeHermes(ctx, provider)

        def turn() -> None:
            t = fake.turn()
            fake.tool(fake.model_call(t, tool_calls=["read_file"]), "read_file")
            fake.parallel_tools(fake.model_call(t, tool_calls=["a", "b"]))

        in_loop(turn)
    sequential, *parallel = recorder.hooks("post_tool_call")
    assert sequential.probe == ()
    assert sorted(s.probe for s in parallel) == [("tool:4",), ("tool:5",)]


def test_a_joined_child_sees_the_parent_context_from_its_first_event(hermes):
    fake, recorder = hermes

    def turn() -> list[Any]:
        t = fake.turn()
        kids = fake.delegate(fake.model_call(t, tool_calls=["delegate_task"]), 2)
        fake.model_call(t)
        fake.end(t)
        return kids

    kids = in_loop(turn)
    for kid in kids:
        own = [s for s in recorder.seen if s.session == kid.session_id]
        assert own[0].name == "on_session_start"
        assert {s.probe for s in own} == {SET}
    child_attempts = [s for s in recorder.seen if s.surface == "attempt"][1:-1]
    assert len(child_attempts) == 4
    assert all(s.thread.startswith("ThreadPoolExecutor-") for s in child_attempts)
    stops = recorder.hooks("subagent_stop")
    assert len(stops) == 2
    assert all(s.thread.startswith("ThreadPoolExecutor-") for s in stops)


def test_background_children_run_after_the_parent_turn_ends(hermes):
    fake, recorder = hermes

    def turn() -> list[Any]:
        t = fake.turn()
        kids = fake.delegate(fake.model_call(t, tool_calls=["delegate_task"]), 1, joined=False)
        fake.model_call(t)
        fake.end(t)
        return kids

    (kid,) = in_loop(turn)
    fake.run_background()
    names = [s.name for s in recorder.hooks()]
    parent_end = names.index("on_session_end")
    first_child = next(i for i, s in enumerate(recorder.hooks()) if s.session == kid.session_id)
    assert first_child > parent_end
    (stop,) = recorder.hooks("subagent_stop")
    assert stop.thread == "async-delegate_0"
    assert stop.probe == SET


def test_a_hook_runs_in_a_copy_and_middleware_and_caller_hooks_run_in_place():
    ctx = FakePluginContext()
    hook_token = ("hook",)
    finalize_token = ("finalize",)
    ctx.register_hook("pre_llm_call", lambda **_: PROBE.set(hook_token) and None)
    ctx.register_hook("on_session_finalize", lambda **_: PROBE.set(finalize_token) and None)
    Recorder().install(ctx)

    def turn(fake: FakeHermes) -> list[tuple[str, ...]]:
        t = fake.turn()
        after_hook = PROBE.get()
        fake.model_call(t, (Reply.OK,))
        after_frame = PROBE.get()
        fake.finalize(t.session_id)
        return [after_hook, after_frame, PROBE.get()]

    with ScriptedProvider() as provider:
        fake = FakeHermes(ctx, provider)
        assert in_loop(lambda: turn(fake)) == [(), SET, finalize_token]
    assert lane_of("pre_llm_call") is Lane.BOUNDED
    assert lane_of("on_session_finalize") is Lane.CALLER


def test_a_gateway_job_runs_in_its_own_copy(hermes):
    fake, _ = hermes

    def job() -> tuple[str, tuple[str, ...]]:
        PROBE.set(("job",))
        return threading.current_thread().name, PROBE.get()

    assert fake.gateway_job(job) == ("hermes-gateway_1", ("job",))
    assert PROBE.get() == ()


# -- the execution chain -----------------------------------------------------------------------


def _chain(*callbacks: Callable[..., Any]) -> FakePluginContext:
    ctx = FakePluginContext()
    for cb in callbacks:
        ctx.register_middleware("llm_execution", cb)
    return ctx


def _run(ctx: FakePluginContext, terminal: Callable[[Any], Any] = lambda p: ("done", p)) -> Any:
    return ctx.run_execution_chain("llm_execution", terminal, request={"n": 1})


def test_with_no_middleware_the_chain_is_hermes_own_call():
    assert _run(FakePluginContext()) == ("done", {"n": 1})


def test_callbacks_nest_in_registration_order_and_may_replace_the_payload():
    order: list[str] = []

    def outer(next_call, **_):
        order.append("outer in")
        result = next_call({"n": 2})
        order.append("outer out")
        return result

    def inner(request, next_call):
        order.append(f"inner in {request['n']}")
        result = next_call()
        order.append("inner out")
        return result

    assert _run(_chain(outer, inner)) == ("done", {"n": 2})
    assert order == ["outer in", "inner in 2", "inner out", "outer out"]


def test_a_downstream_raise_reaches_the_callback_wrapped_and_leaves_the_chain_unwrapped():
    seen: list[BaseException] = []

    def watch(next_call, **_):
        try:
            return next_call()
        except DownstreamError as exc:
            seen.append(exc)
            raise

    def terminal(_):
        raise TimeoutError("provider")

    ctx = _chain(watch)
    with pytest.raises(TimeoutError, match="provider"):
        _run(ctx, terminal)
    assert isinstance(seen[0].original, TimeoutError)
    assert seen[0].__cause__ is seen[0].original
    assert ctx.failures == []


def test_a_callback_that_raises_before_next_call_is_skipped():
    def broken(**_):
        raise KeyError("plugin bug")

    calls: list[Any] = []
    ctx = _chain(broken)
    assert _run(ctx, lambda p: calls.append(p) or "done") == "done"
    assert calls == [{"n": 1}]
    assert [(f.surface, type(f.error)) for f in ctx.failures] == [("Middleware", KeyError)]


def test_a_callback_that_raises_after_next_call_returned_yields_the_downstream_result():
    def late(next_call, **_):
        next_call()
        raise KeyError("plugin bug")

    ctx = _chain(late)
    assert _run(ctx) == ("done", {"n": 1})
    assert len(ctx.failures) == 1


def test_next_call_is_single_use():
    def twice(next_call, **_):
        next_call()
        return next_call()

    ctx = _chain(twice)
    assert _run(ctx) == ("done", {"n": 1})
    assert "more than once" in str(ctx.failures[0].error)


def test_a_callback_that_raises_its_own_error_after_a_downstream_failure_reraises_it():
    def replace(next_call, **_):
        try:
            next_call()
        except DownstreamError:
            raise KeyError("plugin bug") from None

    def terminal(_):
        raise TimeoutError("provider")

    ctx = _chain(replace)
    with pytest.raises(KeyError):
        _run(ctx, terminal)
    assert len(ctx.failures) == 1


def test_execution_callbacks_get_both_schema_versions_and_next_call():
    seen: dict[str, Any] = {}

    def look(**kwargs):
        seen.update(kwargs)
        return kwargs["next_call"]()

    _run(_chain(look))
    assert seen["telemetry_schema_version"] == OBSERVER_V1
    assert seen["middleware_schema_version"] == MIDDLEWARE_V1
    assert callable(seen["next_call"])


def test_a_request_callback_gets_its_own_copy_and_replaces_the_payload_only_by_returning_it():
    ctx = FakePluginContext()
    seen: list[dict[str, Any]] = []

    def mutate(request, original_request):
        request["model"] = "mutated"
        original_request["model"] = "mutated"

    def look(request, original_request):
        seen.append(request)
        return {"request": {**request, "seen": True}}

    ctx.register_middleware("llm_request", mutate)
    ctx.register_middleware("llm_request", look)
    out = ctx.run_request_chain("llm_request", "request", request={"model": "m"})
    assert seen == [{"model": "m"}]
    assert out == {"model": "m", "seen": True}


# -- delivery ----------------------------------------------------------------------------------


def test_hook_arguments_are_narrowed_to_the_declared_names():
    ctx = FakePluginContext()
    got: list[dict[str, Any]] = []
    ctx.register_hook(
        "subagent_start", lambda child_session_id: got.append({"c": child_session_id})
    )
    ctx.register_hook("subagent_start", lambda **kw: got.append(kw))
    ctx.register_hook("gateway_platform_event", lambda **kw: got.append(kw))
    ctx.invoke_hook("subagent_start", child_session_id="s", child_role="leaf")
    ctx.invoke_hook("gateway_platform_event", event="x")
    assert got == [
        {"c": "s"},
        {"child_session_id": "s", "child_role": "leaf", "telemetry_schema_version": OBSERVER_V1},
        {"event": "x"},
    ]


def test_stream_hooks_are_not_narrowed_so_a_narrow_signature_fails_and_is_logged():
    ctx = FakePluginContext()
    ctx.register_hook("on_stream_start", lambda session_id: None)
    ctx.invoke_hook("on_stream_start", session_id="s", turn_id="t")
    ctx.drain_streams()
    assert [(f.name, type(f.error)) for f in ctx.failures] == [("on_stream_start", TypeError)]


def test_a_registration_after_the_load_timed_out_is_refused_with_a_warning():
    ctx = FakePluginContext()
    ctx.abandon_load()
    assert ctx.register_hook("pre_llm_call", lambda **_: None) is None
    assert ctx.register_middleware("llm_execution", lambda **_: None) is None
    assert ctx.on_unload(lambda: None) is None
    assert ctx.hooks == {} and ctx.middleware == {} and ctx.leases == []
    assert len(ctx.warnings) == 3


def test_an_unknown_name_warns_and_is_still_delivered():
    ctx = FakePluginContext()
    got: list[str] = []
    ctx.register_hook("on_future_event", lambda **_: got.append("hook"))
    ctx.register_hook("pre_llm_call", lambda **_: None, hold_stream=True)
    ctx.invoke_hook("on_future_event")
    assert got == ["hook"]
    assert ctx.warnings == [
        "unknown hook 'on_future_event'",
        "hold_stream on 'pre_llm_call' ignored",
    ]
    with pytest.raises(TypeError):
        ctx.on_unload("not callable")


def test_registering_twice_nests_two_frames_and_delivers_each_hook_twice():
    """Hermes keys nothing by plugin, so a plugin whose `register` runs twice sees every frame
    and hook twice (measured on v0.23.3 by loading the plugin twice)."""
    first, second = Recorder(), Recorder()
    ctx = FakePluginContext()
    first.install(ctx)
    second.install(ctx)
    depth: list[int] = []
    level = contextvars.ContextVar("level", default=0)

    def count(next_call, **_):
        token = level.set(level.get() + 1)
        try:
            depth.append(level.get())
            return next_call()
        finally:
            level.reset(token)

    ctx.register_middleware("llm_execution", count)
    ctx.register_middleware("llm_execution", count)
    with ScriptedProvider() as provider:
        fake = FakeHermes(ctx, provider)
        in_loop(lambda: fake.model_call(fake.turn(), (Reply.OK,)))
    assert depth == [1, 2]
    assert len(first.hooks("pre_llm_call")) == len(second.hooks("pre_llm_call")) == 1
    assert len(ctx.hooks["pre_llm_call"]) == 2


def test_unload_disposes_newest_first_and_stops_delivery():
    ctx = FakePluginContext()
    order: list[str] = []
    ctx.on_unload(lambda: order.append("first"))
    ctx.register_hook("pre_llm_call", lambda **_: order.append("hook"))
    ctx.on_unload(lambda: order.append("last"))
    ctx.unload()
    ctx.invoke_hook("pre_llm_call")
    assert order == ["last", "first"]


def test_a_bounded_hook_that_overruns_is_abandoned_then_skipped_and_pre_tool_call_fails_closed():
    ctx = FakePluginContext(hook_timeout_s=0.05)
    gate = threading.Event()

    def stuck(**_):
        gate.wait(5)

    ctx.register_hook("pre_llm_call", stuck)
    ctx.register_hook("pre_tool_call", stuck)
    try:
        assert ctx.invoke_hook("pre_llm_call") == []
        assert ctx.invoke_hook("pre_llm_call") == []
        assert ctx.invoke_hook("pre_tool_call") == [{"action": "block", "message": "timed out"}]
    finally:
        gate.set()
    assert ctx.timeouts == [("pre_llm_call", "stuck"), ("pre_tool_call", "stuck")]
    assert ctx.warnings == ["hook 'pre_llm_call' callback stuck skipped after a timeout"]


def test_every_return_is_recorded_and_finalize_hands_hermes_the_non_none_ones(hermes):
    fake, recorder = hermes
    fake.ctx.register_hook("on_session_finalize", lambda **_: "goodbye")
    in_loop(lambda: fake.end(fake.turn()))
    assert fake.finalize() == ["goodbye"]
    recorded = {r.callback: r.value for r in fake.ctx.returns}
    assert recorded.pop("<lambda>") == "goodbye"
    assert set(recorded.values()) == {None}


def test_a_schema_override_and_dropped_fields_reach_every_payload():
    recorder = Recorder()
    ctx = FakePluginContext()
    recorder.install(ctx)
    seen: list[dict[str, Any]] = []
    ctx.register_hook("post_api_request", lambda **kw: seen.append(kw))
    ctx.register_middleware("llm_execution", lambda **kw: seen.append(kw) or kw["next_call"]())
    schema = {"telemetry_schema_version": "hermes.observer.v2", "middleware_schema_version": "m2"}
    with ScriptedProvider() as provider:
        fake = FakeHermes(ctx, provider, schema=schema, drop_fields=frozenset({"turn_id"}))
        in_loop(lambda: fake.model_call(fake.turn(), (Reply.OK,)))
    frame, post = seen
    assert frame["middleware_schema_version"] == "m2"
    assert post["telemetry_schema_version"] == "hermes.observer.v2"
    assert all("turn_id" not in s.args for s in recorder.hooks())


# -- the provider ------------------------------------------------------------------------------


def test_the_provider_records_every_scripted_attempt(hermes):
    fake, _ = hermes
    used = Usage(5, 6, cached_tokens=2)
    in_loop(lambda: fake.model_call(fake.turn(), (Reply.UNAVAILABLE, Reply.OK_STREAM), usage=used))
    attempts = fake.provider.attempts
    assert [(a.status, a.stream, a.usage, a.model) for a in attempts] == [
        (503, False, None, "lab-model"),
        (200, True, used, "lab-model"),
    ]


def test_the_provider_answers_in_script_order_without_a_named_answer():
    with ScriptedProvider([Reply.BAD_REQUEST, Answer(Reply.OK)]) as provider:
        assert [provider.post(), provider.post()] == [400, 200]


def test_a_turn_whose_retries_run_out_fires_the_error_hooks_and_no_session_end(hermes):
    fake, recorder = hermes
    call = in_loop(lambda: fake.fail(fake.turn()))
    assert call.status == 500
    names = [s.name for s in recorder.hooks()]
    assert names[-2:] == ["transform_api_error_classification", "api_request_error"]
    assert "on_session_end" not in names
    assert recorder.hooks("transform_api_error_classification")[0].thread == "MainThread"


def test_an_interrupted_stream_raises_out_of_the_frame_and_ends_the_session_interrupted(hermes):
    fake, recorder = hermes
    seen: list[dict[str, Any]] = []
    fake.ctx.register_hook("on_session_end", lambda **kw: seen.append(kw))
    in_loop(lambda: fake.interrupt(fake.turn()))
    assert [a.reply for a in fake.provider.attempts] == [Reply.HOLD]
    assert recorder.hooks("post_api_request") == []
    assert seen[0]["interrupted"] is True


def test_wardex_records_one_wire_span_per_attempt_from_the_model_worker():
    """The point of reaching the provider over real sockets: wardex's interceptor sees each
    attempt on the worker thread, under the span that was current when the frame began."""
    transport = RecordingTransport()
    wardex.init(transport=transport)
    try:
        with ScriptedProvider() as provider:
            fake = FakeHermes(FakePluginContext(), provider)

            def turn() -> None:
                with wardex.span("turn"):
                    fake.model_call(fake.turn(), (Reply.UNAVAILABLE, Reply.OK_STREAM))

            in_loop(turn)
        _hub.get_client()._settle()
        wardex.flush()
    finally:
        wardex.close()
    spans = [sp for env in transport.envelopes for sp in env.spans]
    (outer,) = [s for s in spans if s.name == "turn"]
    wire = [s for s in spans if s.kind is SpanKind.CLIENT]
    assert sorted(s.status.value for s in wire) == [StatusCode.ERROR.value, StatusCode.OK.value]
    assert {s.parent_span_id for s in wire} == {outer.context.span_id}


# -- the recorded fixtures ---------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_each_fixture_replays_in_recorded_order_with_recorded_threads_and_contexts(scenario):
    fixture = load_fixture(scenario)
    recorder = Recorder(stack=True)
    ctx = FakePluginContext()
    recorder.install(ctx)
    root = fixture.header["root"]["probe"]

    def play() -> None:
        if root is not None:
            PROBE.set(tuple(root))
        with ScriptedProvider() as provider:
            Replay(ctx, provider).run(scenario)
            assert len(provider.attempts) == sum(
                1 for e in fixture.events if e.kind == "http" and e.name == "request"
            )

    in_loop(play)
    want = [
        (e.kind, e.name, e.thread, e.probe or ())
        for e in fixture.events
        if e.kind == "hook" or (e.kind == "mw" and e.phase != "exit")
    ]
    got = [(s.surface, s.name, s.thread, s.probe) for s in recorder.seen]
    assert got == want
    hooks = [e for e in fixture.events if e.kind == "hook"]
    for event, seen in zip(hooks, recorder.hooks(), strict=True):
        assert set(seen.args) == set(event.payload) | {"telemetry_schema_version"}
    assert ctx.failures == []
