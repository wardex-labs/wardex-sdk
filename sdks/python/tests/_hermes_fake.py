"""A fake Hermes Agent plugin host and a scripted Hermes. No Hermes import.

Hermes needs Python 3.11 or newer with pins that resolve only on 3.14, and it is not on PyPI, so
the adapter's automated tests cannot install it. These fakes reproduce only what a plugin can
observe, each piece taken from Hermes v0.23.3 (tag `v0.23.3`, commit `caafe53`) or from a lab
measurement against it. Every behaviour cites its source so a reader can re-check it against a
newer Hermes. "Source" means read, not run; "measured" means a lab run of the real Hermes showed it.

* `FakePluginContext`: what `register(ctx)` may call (`register_hook`, `register_middleware`,
  `on_unload`) and what Hermes does with the registrations (`invoke_hook`, `run_request_chain`,
  `run_execution_chain`, `unload`).
* `ScriptedProvider`: a loopback chat-completions server answering each attempt from a script,
  reached with real sockets from the thread the caller runs on.

Two drivers play turns through a context: `FakeHermes` in `_hermes_scripted.py`, a scripted turn
loop for cases no fixture recorded, and `Replay` in `_hermes_replay.py`, which plays the recorded
fixtures (`tests/fixtures/hermes/`).
"""

from __future__ import annotations

import contextvars
import http.client
import http.server
import inspect
import json
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

OBSERVER_V1 = "hermes.observer.v1"
MIDDLEWARE_V1 = "hermes.middleware.v1"

#: `hermes_cli/plugins.py::VALID_HOOKS` at v0.23.3.
VALID_HOOKS = frozenset(
    """agent_loop_stopped api_request_error gateway_platform_event kanban_task_blocked
    kanban_task_claimed kanban_task_completed on_human_input_request on_human_input_resolved
    on_interim_message on_kanban_dispatch_tick on_kanban_task_updated on_kanban_worker_exited
    on_kanban_worker_spawned on_kanban_worker_stale_claim on_room_member_activity
    on_session_delete on_session_end on_session_finalize on_session_open on_session_reset
    on_session_start on_skill_lifecycle on_stream_delta on_stream_end on_stream_start
    post_api_request post_approval_response post_auxiliary_call post_compaction
    post_curator_run post_gateway_admission post_llm_call post_tool_call pre_api_request
    pre_approval_request pre_auxiliary_call pre_command pre_gateway_dispatch pre_llm_call
    pre_tool_call pre_transcription pre_verify subagent_start subagent_stop
    transform_api_error_classification transform_llm_output transform_terminal_output
    transform_tool_result""".split()
)
#: `hermes_cli/middleware.py::VALID_MIDDLEWARE`.
VALID_MIDDLEWARE = frozenset({"llm_request", "llm_execution", "tool_request", "tool_execution"})


class Lane(Enum):
    """The thread a hook callback runs on, and whose context it sees."""

    BOUNDED = (
        "bounded"  # fresh `hermes-hook-<cb>` thread, COPY of the caller's context, caller waits
    )
    CALLER = "caller"  # the caller's own thread, its own context, no copy
    STREAM = "stream"  # long-lived `plugin-stream-hook:<hook>` thread, copy, caller does NOT wait


#: The lane of every hook, as one table. Source: `hermes_cli/plugins_dispatch.py`
#: (`_HOOK_TIMEOUT_BOUNDED_HOOKS`, `_HOOK_TIMEOUT_FAIL_CLOSED_HOOKS` and `_HOOK_CALLER_THREAD_HOOKS`
#: read by `_hook_uses_callback_timeout`; any hook in neither set runs inline on the caller) and
#: `agent/plugin_stream_hooks.py::enqueue_plugin_stream_hook` for the three stream observers.
#: Measured for the hooks the fixtures hold: every bounded hook arrived on `hermes-hook-*`, every
#: stream hook on `plugin-stream-hook:*`; `subagent_start` on the parent's tool worker,
#: `subagent_stop` on the tool worker (joined) or `async-delegate_N` (background) and
#: `transform_api_error_classification` on the loop thread (fixtures
#: `parallel_tools`, `two_subagents_joined`, `two_subagents`, `failed_turn`).
#: `on_session_finalize` is caller-lane from source only
#: (`hermes_cli/lifecycle.py::finalize_session`).
HOOK_LANES: Mapping[str, Lane] = {
    **{name: Lane.CALLER for name in VALID_HOOKS},
    **{
        name: Lane.BOUNDED
        for name in (
            "post_tool_call transform_terminal_output transform_tool_result transform_llm_output "
            "pre_llm_call post_llm_call pre_api_request post_api_request api_request_error "
            "pre_auxiliary_call post_auxiliary_call pre_verify on_session_start on_session_end "
            "on_session_open post_gateway_admission pre_command post_compaction pre_tool_call"
        ).split()
    },
    **{name: Lane.STREAM for name in ("on_stream_start", "on_stream_delta", "on_stream_end")},
}
#: `plugins_dispatch.py::_HOOK_CALLBACK_TIMEOUT_SECS` and `_HOOK_TIMEOUT_SUPPRESSION_SECONDS`.
#: `plugins.hook_callback_timeout: 0` (documented: "set 0 to disable") runs every bounded hook
#: inline on the caller instead (`_hook_uses_callback_timeout`): `hook_timeout_s=0` here.
HOOK_TIMEOUT_S = 30.0
SUPPRESSION_S = 60.0
_SKIPPED = object()


def lane_of(hook: str) -> Lane:
    """Unknown names are stored by Hermes and dispatched inline, like any unlisted hook."""
    return HOOK_LANES.get(hook, Lane.CALLER)


def narrow(callback: Callable[..., Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    """`PluginDispatchMixin._hook_callback_kwargs`: all for `**kwargs`, else the declared names."""
    try:
        params = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        return dict(payload)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(payload)
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return {k: v for k, v in payload.items() if k in params and params[k].kind in kinds}


def conversation_of(turn: Any) -> str:
    """The `conversation_id` Hermes states on a turn's tool events (`agent/prompt_cache_scope.py::
    conversation_id`): the id a compression rotation kept, else a delegated child's parent's
    (`tools/delegate_tool.py` stamps it at spawn), else the session id. Duck-typed on the scripted
    `Turn`: `conversation`, `parent`, `session_id`."""
    if turn.conversation:
        return turn.conversation
    return conversation_of(turn.parent) if turn.parent is not None else turn.session_id


def middleware_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """`hermes_cli/middleware.py::middleware_payload`: both schema versions, unless already set."""
    out = dict(payload)
    out.setdefault("telemetry_schema_version", OBSERVER_V1)
    out.setdefault("middleware_schema_version", MIDDLEWARE_V1)
    return out


class DownstreamError(Exception):
    """`hermes_cli/middleware.py::_DownstreamExecutionError`: what a callback sees when the work
    below it raised. Raised `from` the original; the chain unwraps it again on the way out."""

    def __init__(self, original: BaseException) -> None:
        super().__init__(str(original))
        self.original = original


@dataclass
class Lease:
    """`hermes_cli/plugins_ledger.py::PluginRegistration`: one registration and its inverse."""

    kind: str
    key: str
    release: Callable[[], None]
    active: bool = True

    def dispose(self) -> None:
        if not self.active:
            return
        self.active = False
        self.release()


@dataclass(frozen=True)
class Failure:
    """One callback exception Hermes caught and logged (`_report_hook_failure`)."""

    surface: str  # "Hook" or "Middleware", as Hermes's log line says
    name: str
    callback: str
    error: BaseException


@dataclass(frozen=True)
class Returned:
    """One hook callback's return value. Hermes acts on non-None returns of several hooks
    (`pre_llm_call` context or block, `pre_tool_call` block, transforms), so tests assert None."""

    hook: str
    callback: str
    value: Any
    thread: str


def _name(cb: Callable[..., Any]) -> str:
    return getattr(cb, "__name__", repr(cb))


@dataclass
class FakePluginContext:
    """`hermes_cli/plugins_context.py::PluginContext`, reduced to what a plugin can call, plus the
    host side: delivering hooks and running middleware chains as `PluginDispatchMixin` does."""

    plugin_id: str = "wardex"
    home: Path | None = None  # one context per profile home (`plugins.py::get_plugin_manager`)
    hook_timeout_s: float = HOOK_TIMEOUT_S
    hooks: dict[str, list[Callable[..., Any]]] = field(default_factory=dict)
    middleware: dict[str, list[Callable[..., Any]]] = field(default_factory=dict)
    leases: list[Lease] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    returns: list[Returned] = field(default_factory=list)
    timeouts: list[tuple[str, str]] = field(default_factory=list)
    abandoned: bool = False
    _suppressed: dict[tuple[str, int], float] = field(default_factory=dict)
    _running: set[tuple[str, int, str | None]] = field(default_factory=set)
    _streams: dict[tuple[str, int], _StreamDispatcher] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # -- what `register(ctx)` may call ----------------------------------------------------------

    def register_hook(
        self, hook_name: str, callback: Callable[..., Any], *, hold_stream: bool = False
    ) -> Lease | None:
        """`track_hook`: unknown names warn and are still stored; `hold_stream` is honoured only
        on `transform_llm_output` (`plugins_dispatch.py::track_hook`)."""
        if self._ignored("register_hook"):
            return None
        if hold_stream and hook_name != "transform_llm_output":
            self.warnings.append(f"hold_stream on {hook_name!r} ignored")
        return self._track("hook", hook_name, callback, self.hooks, VALID_HOOKS)

    def register_middleware(self, kind: str, callback: Callable[..., Any]) -> Lease | None:
        """Unknown kinds warn and are stored (`plugins_context.py::register_middleware`)."""
        if self._ignored("register_middleware"):
            return None
        return self._track("middleware", kind, callback, self.middleware, VALID_MIDDLEWARE)

    def on_unload(self, callback: Callable[[], None]) -> Lease | None:
        """Runs at unload in reverse acquisition order, interleaved with registration teardown."""
        if self._ignored("on_unload"):
            return None
        if not callable(callback):
            raise TypeError("on_unload callback must be callable")
        lease = Lease("on_unload", _name(callback), callback)
        self.leases.append(lease)
        return lease

    # -- what Hermes does to the registrations --------------------------------------------------

    def abandon_load(self) -> None:
        """`PluginContext._abandon_load`: the load overran `plugins.load_timeout_seconds`, and every
        later registrar call is a warned no-op (`plugins_load_guard.py`)."""
        self.abandoned = True

    def unload(self) -> None:
        """Dispose every lease, newest first; a failing `on_unload` callback is logged."""
        for lease in reversed(self.leases):
            try:
                lease.dispose()
            except Exception as exc:
                self.failures.append(Failure("Unload", lease.key, lease.kind, exc))
        self.leases.clear()

    def invoke_hook(self, hook_name: str, **kwargs: Any) -> list[Any]:
        """`PluginDispatchMixin.invoke_hook`: the non-None results, each callback isolated."""
        if hook_name != "gateway_platform_event":
            kwargs.setdefault("telemetry_schema_version", OBSERVER_V1)
        lane = lane_of(hook_name)
        callbacks = list(self.hooks.get(hook_name, []))
        if lane is Lane.STREAM:
            for cb in callbacks:
                self._stream_for(hook_name, cb).put(contextvars.copy_context(), kwargs)
            return []
        results: list[Any] = []
        for cb in callbacks:
            try:
                if lane is Lane.BOUNDED and self.hook_timeout_s > 0:
                    ret = self._bounded(hook_name, cb, kwargs)
                    if ret is _SKIPPED:
                        if hook_name == "pre_tool_call":  # fail closed (`_run_hook_callbacks`)
                            results.append({"action": "block", "message": "timed out"})
                        continue
                else:
                    ret = cb(**narrow(cb, kwargs))
                    self._returned(hook_name, cb, ret)
                if ret is not None:
                    results.append(ret)
            except (Exception, SystemExit) as exc:
                self.failures.append(Failure("Hook", hook_name, _name(cb), exc))
                if hook_name == "pre_tool_call":
                    results.append({"action": "block", "message": f"{_name(cb)} raised"})
        return results

    def drain_streams(self, timeout: float = 5.0) -> None:
        """Wait until every queued stream-hook event was delivered. Hermes never waits; a test that
        asserts on stream deliveries calls this first."""
        for dispatcher in list(self._streams.values()):
            dispatcher.drain(timeout)

    def run_request_chain(
        self, kind: str, payload_key: str, trace: list[dict[str, str]] | None = None, **kwargs: Any
    ) -> Any:
        """`_apply_request_chain` + `invoke_middleware(_payload_key=...)`: each callback gets its
        own copy of the payload; a dict returned under `payload_key` replaces it for the next and
        appends one entry to `trace` (its `source`, `reason` and `name` strings, or
        `{"source": "plugin"}`), which Hermes hands the execution frame as `middleware_trace`."""
        kwargs = middleware_payload(kwargs)
        original_key = "original_" + payload_key
        kwargs.setdefault(original_key, kwargs[payload_key])
        for cb in list(self.middleware.get(kind, [])):
            call = {
                **kwargs,
                payload_key: _copy(kwargs[payload_key]),
                original_key: _copy(kwargs[original_key]),
            }
            try:
                ret = cb(**narrow(cb, call))
            except (Exception, SystemExit) as exc:
                self.failures.append(Failure("Middleware", kind, _name(cb), exc))
                continue
            if isinstance(ret, dict) and isinstance(ret.get(payload_key), dict):
                kwargs[payload_key] = ret[payload_key]
                if trace is not None:
                    said = {
                        k: v
                        for k in ("source", "reason", "name")
                        if isinstance(v := ret.get(k), str) and v
                    }
                    trace.append(said or {"source": "plugin"})
        return kwargs[payload_key]

    def run_execution_chain(self, kind: str, terminal: Callable[[Any], Any], **kwargs: Any) -> Any:
        """`hermes_cli/middleware.py::_run_execution_chain`, on the CALLER's thread, no copy.

        `next_call` rides in the kwargs; `next_call()` with no argument passes the payload on; a
        second call raises; a downstream raise reaches the callback as `DownstreamError` and leaves
        the chain unwrapped. A callback's own raise is logged and fails open: the downstream result
        if `next_call` succeeded, the same raise if it was called and failed, else the rest of the
        chain runs as if the callback were absent. Callbacks nest in registration order.
        """
        payload_key = "request" if "request" in kwargs else "args"
        callbacks = list(self.middleware.get(kind, []))
        if not callbacks:
            return terminal(kwargs[payload_key])

        def call_at(index: int, payload: Any) -> Any:
            if index >= len(callbacks):
                return terminal(payload)
            callback = callbacks[index]
            state: dict[str, Any] = {"called": False, "ok": False, "result": None}

            def next_call(next_payload: Any = None) -> Any:
                if state["called"]:
                    raise RuntimeError(
                        f"Middleware {kind!r} callback {_name(callback)} called next_call() more "
                        "than once; downstream execution is single-use"
                    )
                state["called"] = True
                try:
                    state["result"] = call_at(
                        index + 1, payload if next_payload is None else next_payload
                    )
                    state["ok"] = True
                    return state["result"]
                except Exception as exc:
                    raise DownstreamError(exc) from exc

            call_kwargs = middleware_payload(kwargs)
            call_kwargs[payload_key] = payload
            call_kwargs["next_call"] = next_call
            try:
                return callback(**narrow(callback, call_kwargs))
            except DownstreamError as exc:
                raise exc.original  # noqa: B904 - Hermes re-raises the original inside the handler
            except Exception as exc:
                self.failures.append(Failure("Middleware", kind, _name(callback), exc))
                if state["ok"]:
                    return state["result"]
                if state["called"]:
                    raise
                return call_at(index + 1, payload)

        return call_at(0, kwargs[payload_key])

    # -- internals ------------------------------------------------------------------------------

    def _ignored(self, method: str) -> bool:
        if self.abandoned:
            self.warnings.append(f"{method}() after the load timed out; ignored")
        return self.abandoned

    def _track(
        self,
        kind: str,
        key: str,
        callback: Callable[..., Any],
        mapping: dict[str, list[Callable[..., Any]]],
        valid: frozenset[str],
    ) -> Lease:
        """`PluginContext._track_callback`."""
        if key not in valid:
            self.warnings.append(f"unknown {kind} {key!r}")
        mapping.setdefault(key, []).append(callback)

        def release() -> None:
            entries = mapping.get(key, [])
            if callback in entries:
                entries.remove(callback)

        lease = Lease(kind, key, release)
        self.leases.append(lease)
        return lease

    def _returned(self, hook: str, cb: Callable[..., Any], value: Any) -> None:
        with self._lock:
            self.returns.append(Returned(hook, _name(cb), value, threading.current_thread().name))

    def _bounded(self, hook: str, cb: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
        """`_run_hook_callback_bounded`: a fresh daemon thread runs the callback in
        `copy_context()` of the caller, who waits up to the timeout and then abandons it. A call
        is skipped while the callback is suppressed after a timeout, or while the same callback
        still runs for the same call identity (`_hook_call_identity`: `tool_call_id`, else
        `turn_id`)."""
        key = (hook, id(cb))
        identity = next(
            (v for f in ("tool_call_id", "turn_id") if isinstance(v := kwargs.get(f), str) and v),
            None,
        )
        gate = (hook, id(cb), identity)
        with self._lock:
            if self._suppressed.get(key, 0.0) > time.monotonic() or gate in self._running:
                self.warnings.append(
                    f"hook {hook!r} callback {_name(cb)} skipped after previous timeout or "
                    "while still running"
                )
                return _SKIPPED
            self._running.add(gate)
        context = contextvars.copy_context()
        done = threading.Event()
        outcome: dict[str, Any] = {}

        def runner() -> None:
            try:
                outcome["value"] = context.run(cb, **narrow(cb, kwargs))
                self._returned(hook, cb, outcome["value"])
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                with self._lock:
                    self._running.discard(gate)
                done.set()

        threading.Thread(target=runner, name=f"hermes-hook-{_name(cb)}"[:40], daemon=True).start()
        if not done.wait(self.hook_timeout_s):
            with self._lock:
                self._suppressed[key] = time.monotonic() + SUPPRESSION_S
                self.timeouts.append((hook, _name(cb)))
            return _SKIPPED
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("value")

    def _stream_for(self, hook: str, cb: Callable[..., Any]) -> _StreamDispatcher:
        with self._lock:
            key = (hook, id(cb))
            if key not in self._streams:
                self._streams[key] = _StreamDispatcher(self, hook, cb)
            return self._streams[key]


class _StreamDispatcher:
    """`agent/plugin_stream_hooks.py`: one queue and one daemon worker per callback. The worker
    runs each event in a copy of the enqueuer's context, with the payload NOT narrowed."""

    def __init__(self, host: FakePluginContext, hook: str, cb: Callable[..., Any]) -> None:
        self._host, self._hook, self._cb = host, hook, cb
        self._items: list[tuple[contextvars.Context, dict[str, Any]]] = []
        self._cond = threading.Condition()
        self._busy = False
        threading.Thread(target=self._work, name=f"plugin-stream-hook:{hook}", daemon=True).start()

    def put(self, context: contextvars.Context, payload: dict[str, Any]) -> None:
        with self._cond:
            self._items.append((context, dict(payload)))
            self._cond.notify_all()

    def drain(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._items or self._busy:
                if not self._cond.wait(max(0.0, deadline - time.monotonic())):
                    raise TimeoutError(f"stream hook {self._hook} did not drain")

    def _work(self) -> None:
        while True:
            with self._cond:
                while not self._items:
                    self._cond.wait()
                context, payload = self._items.pop(0)
                self._busy = True
            try:
                value = context.copy().run(self._cb, **payload)
                self._host._returned(self._hook, self._cb, value)
            except Exception as exc:
                self._host.failures.append(Failure("Hook", self._hook, _name(self._cb), exc))
            finally:
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value)) if isinstance(value, (dict, list)) else value


# -- the provider ------------------------------------------------------------------------------


class Reply(Enum):
    OK = "ok"  # 200 JSON chat completion with usage
    OK_STREAM = "ok_stream"  # 200 server-sent events, usage in the last chunk
    UNAVAILABLE = "503"
    SERVER_ERROR = "500"
    BAD_REQUEST = "400"
    HOLD = "hold"  # 200 stream headers and one chunk, then nothing until `release()`


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0  # `prompt_tokens_details.cached_tokens`, the cache tier


@dataclass(frozen=True)
class Answer:
    reply: Reply
    usage: Usage | None = None
    tool_calls: tuple[tuple[str, str], ...] = ()  # (id, name) the model asks for


@dataclass(frozen=True)
class Attempt:
    """One request as the server saw it."""

    reply: Reply
    status: int
    usage: Usage | None
    model: str | None
    stream: bool


_STATUS = {Reply.UNAVAILABLE: 503, Reply.SERVER_ERROR: 500, Reply.BAD_REQUEST: 400}


class ScriptedProvider:
    """A loopback OpenAI-compatible server. Each request is answered by the `Answer` named in its
    `x-fake-answer` header (an index into `answers`), or else by the next one in `script` order.
    Requests go out from the CALLING thread over a real socket, so wardex's socket interceptor
    records one wire span per attempt on that thread, as it does under Hermes."""

    def __init__(self, script: Iterable[Answer | Reply] = ()) -> None:
        self.script: list[Answer] = [_answer(a) for a in script]
        self.answers: list[Answer] = []
        self.attempts: list[Attempt] = []
        self._next = 0
        self._lock = threading.Lock()
        self._released = threading.Event()
        self._httpd: http.server.ThreadingHTTPServer | None = None

    def __enter__(self) -> ScriptedProvider:
        provider = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                provider._answer(self, body, self.headers.get("x-fake-answer"))

            def log_message(self, *args: object) -> None:
                pass

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()

    @property
    def base_url(self) -> str:
        assert self._httpd is not None, "use the provider as a context manager"
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def release(self) -> None:
        """Let every held answer finish."""
        self._released.set()

    def post(self, answer: Answer | Reply | None = None, *, model: str = "lab-model") -> int:
        """One complete attempt from the current thread; returns the HTTP status."""
        exchange = self.send(answer, model=model)
        status = exchange.head()
        exchange.finish()
        return status

    def send(self, answer: Answer | Reply | None = None, *, model: str = "lab-model") -> Exchange:
        """Send one request from the current thread; read its answer with the returned handle."""
        headers = {"Content-Type": "application/json"}
        stream = False
        if answer is not None:
            chosen = _answer(answer)
            stream = chosen.reply in (Reply.OK_STREAM, Reply.HOLD)
            with self._lock:
                self.answers.append(chosen)
                headers["x-fake-answer"] = str(len(self.answers) - 1)
        assert self._httpd is not None, "use the provider as a context manager"
        host, port = self._httpd.server_address[:2]
        conn = http.client.HTTPConnection(str(host), int(port), timeout=30)
        body = json.dumps({"model": model, "stream": stream, "messages": []})
        conn.request("POST", "/v1/chat/completions", body, headers)
        return Exchange(conn)

    def _answer(self, handler: http.server.BaseHTTPRequestHandler, body: dict, index: str | None):
        with self._lock:
            if index is not None:
                answer = self.answers[int(index)]
            else:
                answer = self.script[self._next]
                self._next += 1
            n = len(self.attempts) + 1
            usage = answer.usage or Usage(1000 + n, 10 + n)
            status = _STATUS.get(answer.reply, 200)
            stream = answer.reply in (Reply.OK_STREAM, Reply.HOLD)
            ok = status == 200
            self.attempts.append(
                Attempt(answer.reply, status, usage if ok else None, body.get("model"), stream)
            )
        if not ok:
            _send(handler, status, "application/json", {"error": {"message": f"scripted {status}"}})
            return
        model = body.get("model") or "lab-model"
        message: dict[str, Any] = {
            "role": "assistant",
            "content": None if answer.tool_calls else "ok",
        }
        calls = [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}
            for cid, name in answer.tool_calls
        ]
        if calls:
            message["tool_calls"] = calls
        finish = "tool_calls" if calls else "stop"
        wire_usage = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.prompt_tokens + usage.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": usage.cached_tokens},
        }
        head = {"id": f"chatcmpl-fake-{n}", "created": 1, "model": model}
        if not stream:
            choice = {"index": 0, "message": message, "finish_reason": finish}
            completion = {**head, "object": "chat.completion", "choices": [choice]}
            _send(handler, 200, "application/json", {**completion, "usage": wire_usage})
            return
        chunks = [
            {"index": 0, "delta": {**message}, "finish_reason": None},
            {"index": 0, "delta": {}, "finish_reason": finish},
        ]
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        _chunk(handler, {**head, "object": "chat.completion.chunk", "choices": chunks[:1]})
        if answer.reply is Reply.HOLD:
            self._released.wait(30)
        _chunk(handler, {**head, "object": "chat.completion.chunk", "choices": chunks[1:]})
        _chunk(
            handler, {**head, "object": "chat.completion.chunk", "choices": [], "usage": wire_usage}
        )
        _write_chunk(handler, b"data: [DONE]\n\n")
        _write_chunk(handler, b"")


class Exchange:
    """One request in flight: `head()` reads the status line, `finish()` the body, `cut()` closes
    the socket mid-answer (an interrupted stream)."""

    def __init__(self, conn: http.client.HTTPConnection) -> None:
        self._conn = conn
        self._response: http.client.HTTPResponse | None = None

    def head(self) -> int:
        self._response = self._conn.getresponse()
        return self._response.status

    def finish(self) -> None:
        assert self._response is not None
        self._response.read()
        self._conn.close()

    def cut(self) -> None:
        if self._conn.sock is not None:
            self._conn.sock.close()
        self._conn.close()


def _answer(a: Answer | Reply) -> Answer:
    return a if isinstance(a, Answer) else Answer(a)


def _send(handler: http.server.BaseHTTPRequestHandler, status: int, kind: str, obj: Any) -> None:
    data = json.dumps(obj).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", kind)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _chunk(handler: http.server.BaseHTTPRequestHandler, obj: Any) -> None:
    _write_chunk(handler, f"data: {json.dumps(obj)}\n\n".encode())


def _write_chunk(handler: http.server.BaseHTTPRequestHandler, data: bytes) -> None:
    try:
        handler.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        handler.wfile.flush()
    except OSError:
        handler.close_connection = True  # the client cut the stream (an interrupt)


class ProviderStatusError(Exception):
    """A provider's HTTP error as the OpenAI client raises it; the class name is what Hermes
    reports as `error_type` (`openai.InternalServerError` in the failed-turn recording)."""

    status_code = 0


class InternalServerError(ProviderStatusError):
    status_code = 500


class BadRequestError(ProviderStatusError):
    status_code = 400


#: Hermes's `exit_reason` and the error it raises when a turn is interrupted mid-stream
#: (`agent/chat_completion_helpers.py`: `InterruptedError("Agent interrupted during streaming API
#: call")`; measured: the frame raised with that text, then `on_session_end(interrupted=True)`).
INTERRUPTED_TEXT = "Agent interrupted during streaming API call"


__all__ = [
    "Answer",
    "Attempt",
    "BadRequestError",
    "DownstreamError",
    "Exchange",
    "FakePluginContext",
    "Failure",
    "HOOK_LANES",
    "INTERRUPTED_TEXT",
    "InternalServerError",
    "Lane",
    "Lease",
    "MIDDLEWARE_V1",
    "OBSERVER_V1",
    "ProviderStatusError",
    "Reply",
    "Returned",
    "ScriptedProvider",
    "Usage",
    "VALID_HOOKS",
    "VALID_MIDDLEWARE",
    "lane_of",
    "middleware_payload",
    "narrow",
]
