"""`FakeHermes`: Hermes's turn loop as a script, for the cases no recorded fixture holds.

Each method fires the hooks and middleware chains one step of a real Hermes turn fires, on the
threads Hermes uses for it, with payloads shaped like the recorded ones (`_hermes_replay.py`
plays the recordings themselves). The thread that calls a method is the agent's loop thread.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import itertools
import secrets
import threading
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from _hermes_fake import (
    INTERRUPTED_TEXT,
    OBSERVER_V1,
    Answer,
    BadRequestError,
    Exchange,
    FakePluginContext,
    InternalServerError,
    ProviderStatusError,
    Reply,
    ScriptedProvider,
    Usage,
)

TOOL_RESULT = "<synthetic result>"


@dataclass
class Turn:
    session_id: str
    task_id: str
    turn_id: str
    platform: str = ""
    parent: Turn | None = None
    subagent_id: str | None = None
    calls: int = 0
    title: _TitleThread | None = None


@dataclass(frozen=True)
class Call:
    turn: Turn
    api_request_id: str
    tool_calls: tuple[tuple[str, str], ...]  # (tool_call_id, name) the model asked for
    status: int | None  # the last attempt's HTTP status; None when cut


class _TitleThread:
    """`agent/memory_provider.py::spawn_context_thread`: the context is copied when the thread is
    CONSTRUCTED, at turn start before the first middleware call (measured: fixture
    `auxiliary_title`)."""

    def __init__(self) -> None:
        self._context = contextvars.copy_context()

    def start(self, fn: Callable[[], Any]) -> threading.Thread:
        thread = threading.Thread(target=self._context.run, args=(fn,), name="auto-title")
        thread.start()
        return thread


@dataclass
class FakeHermes:
    """Hermes's turn loop reduced to the calls, threads and payloads a plugin sees.

    The thread calling a method is the agent's loop thread. Payload fields and id shapes follow
    the recorded fixtures: turn `<session>:<task>:<hex>`, call `<turn_id>:api:<n>`, tool call
    `call_lab_<n>_<k>`, child task `sa-<i>-<hex>`, auxiliary call `aux-<hex>`.
    """

    ctx: FakePluginContext
    provider: ScriptedProvider
    model: str = "lab-model"
    schema: dict[str, str] = field(default_factory=dict)  # override a schema version field
    drop_fields: frozenset[str] = frozenset()  # leave fields out of every payload
    _seq: Iterator[int] = field(default_factory=lambda: itertools.count(6))
    _gate: threading.Event = field(default_factory=threading.Event)
    _background: list[threading.Thread] = field(default_factory=list)

    def turn(self, session_id: str | None = None, *, title: bool = False) -> Turn:
        """`on_session_start` then `pre_llm_call`, both bounded hooks (fixture `single_turn`).
        With `title`, the turn-start title thread is constructed now, before any middleware runs."""
        sid = session_id or f"20261122_0000{next(self._seq) % 100:02d}_{secrets.token_hex(3)}"
        task = str(uuid.uuid4())
        turn = Turn(sid, task, f"{sid}:{task}:{secrets.token_hex(4)}")
        self._open(turn)
        if title:
            turn.title = _TitleThread()
        return turn

    def model_call(
        self,
        turn: Turn,
        attempts: Sequence[Reply] = (Reply.OK_STREAM,),
        *,
        tool_calls: Sequence[str] = (),
        on_worker: bool | None = None,
        usage: Usage | None = None,
    ) -> Call:
        """One logical call: `llm_request`, `pre_api_request`, then one `llm_execution` frame on
        this thread holding every attempt (retries never leave the frame: fixture `retry`). Each
        attempt is a real request from a worker started inside the frame with a copied context
        (`chat_completion_helpers.py::_context_thread_target`); a child turn's attempts run on its
        own loop thread, as recorded in `two_subagents`. Spent retries raise out of the frame and
        fire `transform_api_error_classification` and `api_request_error`, and no end event."""
        turn.calls += 1
        ids = self._call_ids(turn, f"{turn.turn_id}:api:{turn.calls}")
        n = next(self._seq)
        asked = tuple((f"call_lab_{n}_{k}", name) for k, name in enumerate(tool_calls))
        used = usage or Usage(1000 + n, 10 + n)
        request = self.ctx.run_request_chain(
            "llm_request", "request", request={"model": self.model, "messages": []}, **ids
        )
        self._emit("pre_api_request", {**ids, "retry_count": 0})
        worker = turn.parent is None if on_worker is None else on_worker
        statuses: list[int | None] = []

        def terminal(payload: Any) -> Any:
            for reply in attempts:
                answer = (
                    Answer(reply, used, asked) if reply in (Reply.OK, Reply.OK_STREAM) else None
                )
                statuses.append(self._attempt(turn, answer or Answer(reply), worker))
            if statuses[-1] is None:
                raise InterruptedError(INTERRUPTED_TEXT)
            if statuses[-1] != 200:
                error = InternalServerError if statuses[-1] >= 500 else BadRequestError
                raise error(f"Error code: {statuses[-1]}")
            return SimpleNamespace(tool_calls=asked)

        try:
            self.ctx.run_execution_chain(
                "llm_execution", terminal, request=request, original_request=request, **ids
            )
        except ProviderStatusError as exc:
            self.ctx.invoke_hook(
                "transform_api_error_classification",
                provider="custom",
                model=self.model,
                status_code=str(exc.status_code),
                error_type=type(exc).__name__,
            )
            self._emit(
                "api_request_error",
                {
                    **ids,
                    "retry_count": 0,
                    "max_retries": len(attempts) - 1,
                    "retryable": exc.status_code >= 500,
                    "reason": "server_error" if exc.status_code >= 500 else "bad_request",
                    "status_code": str(exc.status_code),
                    "error_type": type(exc).__name__,
                },
            )
            return Call(turn, ids["api_request_id"], asked, statuses[-1])
        reported = _usage(used)
        self._emit(
            "post_api_request",
            {
                **ids,
                "response_model": self.model,
                "finish_reason": "tool_calls" if asked else "stop",
                "assistant_tool_call_count": len(asked),
                "usage": reported,
                "response": {
                    "finish_reason": "tool_calls" if asked else "stop",
                    "assistant_message": {
                        "tool_calls": [{"id": i, "name": nm, "arguments": "{}"} for i, nm in asked]
                    },
                },
            },
        )
        return Call(turn, ids["api_request_id"], asked, statuses[-1])

    def fail(self, turn: Turn, attempts: Sequence[Reply] = (Reply.SERVER_ERROR,) * 2) -> Call:
        """Retries spent: the frame raises, `api_request_error` fires, and Hermes fires NO
        `on_session_end` (`agent/turn_recovery.py::_failed_turn_result`; fixture `failed_turn`)."""
        return self.model_call(turn, attempts)

    def interrupt(self, turn: Turn) -> Call:
        """A held stream cut by an interrupt: the frame raises `InterruptedError`, then
        `on_session_end(interrupted=True)`, with no `post_api_request` (fixture `interrupted`)."""
        try:
            return self.model_call(turn, (Reply.HOLD,))
        except InterruptedError:
            self._emit(
                "on_session_end",
                {
                    **self._turn_ids(turn),
                    "completed": False,
                    "failed": False,
                    "interrupted": True,
                    "turn_exit_reason": "interrupted_during_api_call",
                },
            )
            return Call(turn, f"{turn.turn_id}:api:{turn.calls}", (), None)

    def tool(self, call: Call, name: str, *, status: str = "ok", tool_call_id: str = "") -> None:
        """One tool on a fresh one-worker pool, context copied at submit on this thread
        (`agent/tool_executor.py::_run_sequential_tool_execution_middleware`). `post_tool_call`
        lands after the frame exits on this path (fixture `inline_seq`)."""
        tcid = tool_call_id or next(i for i, nm in call.tool_calls if nm == name)
        copied = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(copied.run, self._tool_job, call, tcid, name, status, None).result()

    def parallel_tools(self, call: Call, names: Sequence[str] = ()) -> None:
        """One pool, one job per call, each context copied at submit (`_ConcurrentToolBatch.
        submit_all`); the frames overlap, and `post_tool_call` lands inside each (fixture
        `parallel_tools`)."""
        picked = [(i, nm) for i, nm in call.tool_calls if not names or nm in names]
        barrier = threading.Barrier(len(picked), timeout=10)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(picked)) as pool:
            jobs = [
                pool.submit(
                    contextvars.copy_context().run, self._tool_job, call, i, nm, "ok", barrier
                )
                for i, nm in picked
            ]
            for job in jobs:
                job.result()

    def delegate(
        self,
        call: Call,
        children: int = 2,
        *,
        joined: bool = True,
        child: Callable[[FakeHermes, Turn], None] | None = None,
    ) -> list[Turn]:
        """A `delegate_task` frame. Inside it, per child: `subagent_start` on this tool worker
        (caller lane), and the child's loop on a pool thread whose context is copied inside the
        frame (fixtures `two_subagents_joined`, `two_subagents`). Joined: the frame waits, then
        fires `subagent_stop` on the worker. Background: the frame returns at once; the children
        and an `async-delegate_N` thread (copied inside the frame, fires `subagent_stop`) wait for
        `run_background()`."""
        tcid = next(i for i, nm in call.tool_calls if nm == "delegate_task")
        kids: list[Turn] = []
        script = child or _default_child

        def body() -> None:
            futures = []
            for k in range(children):
                kid = self._child(call.turn, k)
                kids.append(kid)
                self.ctx.invoke_hook(
                    "subagent_start",
                    **self._payload(
                        {
                            "parent_session_id": call.turn.session_id,
                            "parent_turn_id": call.turn.turn_id,
                            "parent_subagent_id": call.turn.subagent_id,
                            "child_session_id": kid.session_id,
                            "child_subagent_id": kid.subagent_id,
                            "child_role": "leaf",
                        }
                    ),
                )
                pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                run = self._run_child if joined else self._run_child_later
                futures.append(pool.submit(contextvars.copy_context().run, run, kid, script))
                pool.shutdown(wait=False)
            if joined:
                self._stop(futures, kids)
                return
            stopper = threading.Thread(
                target=contextvars.copy_context().run,
                args=(self._stop, futures, kids),
                name=f"async-delegate_{len(self._background)}",
            )
            self._background.append(stopper)
            stopper.start()

        copied = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            job = pool.submit(
                copied.run, self._tool_job, call, tcid, "delegate_task", "ok", None, body
            )
            job.result()
        return kids

    def run_background(self, timeout: float = 30.0) -> None:
        """Let background children run. They start after the parent turn ended (fixture
        `two_subagents`)."""
        self._gate.set()
        for thread in self._background:
            thread.join(timeout)

    def aux(
        self,
        turn: Turn,
        *,
        from_loop: bool = False,
        answer: Reply = Reply.OK,
        aux_task: str = "title_generation",
        model: str = "lab-aux-title",
        wait: bool = True,
    ) -> threading.Thread | None:
        """`pre_auxiliary_call`, one request, `post_auxiliary_call` (`agent/auxiliary_hooks.py`),
        with no middleware frame (fixture `auxiliary_title`). `from_loop` runs it on this thread
        (not measured: no lab scenario produced one); otherwise on the turn-start title thread."""
        ids = {**self._turn_ids(turn), "api_request_id": f"aux-{uuid.uuid4().hex}"}
        base = {
            **ids,
            "api_mode": "chat_completions",
            "api_call_count": 1,
            "aux_task": aux_task,
            "model": model,
            "platform": turn.platform,
            "provider": "custom",
            "retry_count": 0,
            "streaming": False,
        }

        def run() -> None:
            self._emit("pre_auxiliary_call", base)
            used = Usage(1000 + next(self._seq), 17)
            status = self.provider.post(Answer(answer, used), model=model)
            ok = status == 200
            self._emit(
                "post_auxiliary_call",
                {
                    **base,
                    "response_model": model if ok else None,
                    "finish_reason": "stop" if ok else None,
                    "usage": _usage(used) if ok else None,
                    "error_type": None if ok else "BadRequestError",
                    "error": None if ok else {"type": "BadRequestError"},
                },
            )

        if from_loop:
            run()
            return None
        assert turn.title is not None, "turn(title=True) constructs the title thread"
        thread = turn.title.start(run)
        if wait:
            thread.join(30)
        return thread

    def end(
        self,
        turn: Turn,
        *,
        completed: bool = True,
        interrupted: bool = False,
        failed: bool = False,
        reason: str = "text_response(finish_reason=stop)",
    ) -> None:
        """`transform_llm_output`, `post_llm_call`, `on_session_end`: the normal close in fixture
        `single_turn`."""
        light = {
            "session_id": turn.session_id,
            "turn_id": turn.turn_id,
            "model": self.model,
            "platform": turn.platform,
        }
        self._emit("transform_llm_output", light)
        self._emit("post_llm_call", {**light, "task_id": turn.task_id})
        self._emit(
            "on_session_end",
            {
                **self._turn_ids(turn),
                "completed": completed,
                "failed": failed,
                "interrupted": interrupted,
                "turn_exit_reason": reason,
            },
        )
        self.ctx.drain_streams()

    def finalize(self, session_id: str | None = None, reason: str = "shutdown") -> list[Any]:
        """`on_session_finalize` on the calling thread (`hermes_cli/cli_shutdown.py` ->
        `lifecycle.finalize_session`); Hermes uses the non-None results as messages."""
        return self.ctx.invoke_hook(
            "on_session_finalize", session_id=session_id, platform="cli", reason=reason
        )

    def gateway_job(self, fn: Callable[[], Any], n: int = 1) -> Any:
        """One gateway job: its own thread, inside `copy_context().run` (`gateway/run.py`, source
        only: gateway messages were not run in the lab)."""
        result: dict[str, Any] = {}
        copied = contextvars.copy_context()
        thread = threading.Thread(
            target=lambda: result.update(value=copied.run(fn)), name=f"hermes-gateway_{n}"
        )
        thread.start()
        thread.join(60)
        return result.get("value")

    # -- internals ------------------------------------------------------------------------------

    def _payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in payload.items() if k not in self.drop_fields}
        for key, value in self.schema.items():
            if key == "telemetry_schema_version" or key in out:
                out[key] = value
        return out

    def _emit(self, hook: str, payload: dict[str, Any]) -> list[Any]:
        out = {"telemetry_schema_version": OBSERVER_V1, **payload}
        return self.ctx.invoke_hook(hook, **self._payload(out))

    def _turn_ids(self, turn: Turn) -> dict[str, Any]:
        return {
            "session_id": turn.session_id,
            "task_id": turn.task_id,
            "turn_id": turn.turn_id,
            "model": self.model,
            "platform": turn.platform,
        }

    def _call_ids(self, turn: Turn, api_request_id: str) -> dict[str, Any]:
        ids = {
            **self._turn_ids(turn),
            "api_request_id": api_request_id,
            "api_mode": "chat_completions",
            "api_call_count": turn.calls,
            "provider": "custom",
        }
        if "middleware_schema_version" in self.schema:
            ids["middleware_schema_version"] = self.schema["middleware_schema_version"]
        return self._payload(ids)

    def _open(self, turn: Turn) -> None:
        self._emit(
            "on_session_start",
            {"session_id": turn.session_id, "model": self.model, "platform": turn.platform},
        )
        parent = turn.parent.session_id if turn.parent else ""
        self._emit(
            "pre_llm_call",
            {**self._turn_ids(turn), "parent_session_id": parent, "is_first_turn": True},
        )

    def _attempt(self, turn: Turn, answer: Answer, worker: bool) -> int | None:
        stream = answer.reply in (Reply.OK_STREAM, Reply.HOLD)
        light = {
            "session_id": turn.session_id,
            "turn_id": turn.turn_id,
            "model": self.model,
            "provider": "custom",
            "iteration": turn.calls,
            "surface": "cli",
        }
        held: list[Exchange] = []

        def run() -> int:
            if stream:
                self._emit("on_stream_start", light)
            exchange = self.provider.send(answer, model=self.model)
            status = exchange.head()
            if answer.reply is Reply.HOLD:
                held.append(exchange)
                return status
            exchange.finish()
            if stream:
                self._emit("on_stream_end", {**light, "finished": True})
            return status

        if not worker:
            status: int | None = run()
        else:
            box: dict[str, Any] = {}
            copied = contextvars.copy_context()
            thread = threading.Thread(target=lambda: box.update(status=copied.run(run)))
            thread.start()
            thread.join(30)
            status = box.get("status")
        for exchange in held:
            exchange.cut()
            self.provider.release()
            status = None
        return status

    def _tool_job(self, call, tcid, name, status, barrier, body=None) -> None:
        turn = call.turn
        ids = {
            "session_id": turn.session_id,
            "task_id": turn.task_id,
            "turn_id": turn.turn_id,
            "api_request_id": call.api_request_id,
            "tool_call_id": tcid,
            "tool_name": name,
            "conversation_id": turn.session_id,
        }
        ids = self._payload({"telemetry_schema_version": OBSERVER_V1, **ids})
        args = self.ctx.run_request_chain(
            "tool_request", "args", args={"input": "<synthetic>"}, **ids
        )
        post = {
            **ids,
            "args": args,
            "result": TOOL_RESULT,
            "duration_ms": 1,
            "status": status,
            "error_type": None if status == "ok" else "tool_error",
            "error_message": None if status == "ok" else "<synthetic error>",
        }

        def terminal(final_args: Any) -> Any:
            self._emit("pre_tool_call", {**ids, "args": final_args})
            if body is not None:
                body()
            if barrier is not None:
                barrier.wait()
                self._emit("post_tool_call", post)
            self._emit("transform_tool_result", post)
            return TOOL_RESULT

        self.ctx.run_execution_chain(
            "tool_execution", terminal, args=args, original_args=args, **ids
        )
        if barrier is None:
            self._emit("post_tool_call", post)

    def _child(self, parent: Turn, k: int) -> Turn:
        sid = f"20261122_0000{next(self._seq) % 100:02d}_{secrets.token_hex(3)}"
        agent = f"sa-{k}-{secrets.token_hex(4)}"
        return Turn(sid, agent, f"{sid}:{agent}:{secrets.token_hex(4)}", "subagent", parent, agent)

    def _run_child(self, kid: Turn, script: Callable[[FakeHermes, Turn], None]) -> None:
        self._open(kid)
        script(self, kid)

    def _run_child_later(self, kid: Turn, script: Callable[[FakeHermes, Turn], None]) -> None:
        self._gate.wait(30)
        self._run_child(kid, script)

    def _stop(self, futures: list[concurrent.futures.Future], kids: list[Turn]) -> None:
        for future, kid in zip(futures, kids, strict=True):
            future.result(30)
            self.ctx.invoke_hook(
                "subagent_stop",
                **self._payload(
                    {
                        "parent_session_id": kid.parent.session_id if kid.parent else None,
                        "parent_turn_id": kid.parent.turn_id if kid.parent else None,
                        "child_session_id": kid.session_id,
                        "child_role": "leaf",
                        "child_status": "completed",
                        "duration_ms": 1,
                    }
                ),
            )


def _default_child(hermes: FakeHermes, kid: Turn) -> None:
    """The recorded child: one tool round, then an answer (`two_subagents`)."""
    hermes.tool(hermes.model_call(kid, (Reply.OK,), tool_calls=["read_file"]), "read_file")
    hermes.model_call(kid, (Reply.OK,))
    hermes.end(kid)


def _usage(used: Usage) -> dict[str, int]:
    """Hermes's normalized `usage` block, as recorded in `post_api_request`. Hermes's
    `input_tokens` is the uncached part of the prompt: its `normalize_usage` subtracts the cache
    buckets from a chat-shaped total, and its `prompt_tokens` adds them back."""
    return {
        "input_tokens": used.prompt_tokens - used.cached_tokens,
        "output_tokens": used.completion_tokens,
        "cache_read_tokens": used.cached_tokens,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "request_count": 1,
        "prompt_tokens": used.prompt_tokens,
        "total_tokens": used.prompt_tokens + used.completion_tokens,
    }
