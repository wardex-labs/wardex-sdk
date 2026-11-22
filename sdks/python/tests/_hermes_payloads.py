"""Events `FakeHermes` does not play: the ones Hermes sends with ids it does not have.

Hermes coerces an id it lacks to "" before any hook or middleware sees it
(`model_tools.py::_CallIds.hook_kwargs`, `agent/auxiliary_hooks.py::_parent_turn_identity`). These
helpers fire such events through the same chains and on the same threads as Hermes v0.23.3 does,
built from `FakeHermes`'s own pieces. Each cites the Hermes source it follows; none was measured in
the lab except where a docstring says so.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import hashlib
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from _hermes_fake import OBSERVER_V1, Answer, Reply, Usage, conversation_of
from _hermes_scripted import TOOL_RESULT, Call, FakeHermes, Turn, _usage


def prelude_tool(h: FakeHermes, turn: Turn, name: str = "read_file") -> None:
    """A scripted prelude's tool (`agent/turn_scripted_prelude.py`), run before the turn's first
    model call. Hermes resets `_current_api_request_id` to "" at turn start
    (`agent/turn_context.py`), so the frame and its report carry `api_request_id=""`; the tool
    call id is Hermes's own (`prelude_call_id`). It runs on a tool worker like any sequential
    tool. Measured on a real Hermes: the tool ran and its result reached the model."""
    h.tool(Call(turn, "", ((f"call_prelude_{uuid.uuid4().hex[:8]}", name),), None), name)


def dispatch(h: FakeHermes, turn: Turn, name: str, inner: Callable[[], None] | None = None) -> None:
    """`model_tools.py::handle_function_call` as code in `execute_code` reaches it on its RPC
    thread (`tools/code_execution_rpc.py::_default_dispatch`, `tools/code_kernel.py::_invoke`):
    the turn's session, turn and conversation ids, and no `tool_call_id` or `api_request_id`.
    Request middleware, `pre_tool_call`, the `tool_execution` frame, then `post_tool_call` and
    `transform_tool_result` after it, on this thread. `inner` runs inside the frame."""
    ids = {
        "telemetry_schema_version": OBSERVER_V1,
        "task_id": turn.task_id,
        "session_id": turn.session_id,
        "tool_call_id": "",
        "turn_id": turn.turn_id,
        "api_request_id": "",
        "conversation_id": conversation_of(turn),
        "tool_name": name,
    }
    args = h.ctx.run_request_chain("tool_request", "args", args={"input": "<synthetic>"}, **ids)
    h.ctx.invoke_hook("pre_tool_call", **ids, args=args)

    def terminal(final_args: Any) -> Any:
        if inner is not None:
            inner()
        return TOOL_RESULT

    h.ctx.run_execution_chain("tool_execution", terminal, args=args, original_args=args, **ids)
    post = {
        **ids,
        "args": args,
        "result": TOOL_RESULT,
        "duration_ms": 1,
        "status": "ok",
        "error_type": None,
        "error_message": None,
    }
    h.ctx.invoke_hook("post_tool_call", **post)
    h.ctx.invoke_hook("transform_tool_result", **post)


def code_calls(h: FakeHermes, call: Call, run: Callable[[], None]) -> None:
    """`execute_code` asked for by `call`: its frame on a tool worker, and inside it the RPC
    thread Hermes starts with a copy of the worker's context
    (`tools/code_execution_tool.py`: `propagate_context_to_thread(_rpc_poll_loop)`), where `run`
    makes the code's tool calls (`dispatch`). The code tool waits for the thread."""
    tcid = next(i for i, nm in call.tool_calls if nm == "execute_code")

    def body() -> None:
        rpc = threading.Thread(target=contextvars.copy_context().run, args=(run,), name="rpc")
        rpc.start()
        rpc.join(30)

    copied = contextvars.copy_context()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(copied.run, h._tool_job, call, tcid, "execute_code", "ok", None, body).result()


def turnless_aux(
    h: FakeHermes, *, aux_task: str = "title_generation", model: str = "lab-aux-title"
) -> None:
    """An auxiliary call with no current turn, on the calling thread (`agent/auxiliary_hooks.py`):
    `_parent_turn_identity` sends `session_id`, `task_id`, `turn_id` and `platform` as "". Cron,
    gateway idle work, a call after the turn ended, and an MCP server's sampling request on the
    MCP loop thread all send this shape (measured: MCP sampling and a title call after a turn)."""
    base = {
        "session_id": "",
        "task_id": "",
        "turn_id": "",
        "platform": "",
        "aux_task": aux_task,
        "api_request_id": f"aux-{uuid.uuid4().hex}",
        "retry_count": 0,
        "api_call_count": 1,
        "model": model,
        "provider": "custom",
        "api_mode": "chat_completions",
        "streaming": False,
    }
    h._emit("pre_auxiliary_call", base)
    used = Usage(1000, 17)
    h.provider.post(Answer(Reply.OK, used), model=model)
    h._emit(
        "post_auxiliary_call",
        {
            **base,
            "response_model": model,
            "finish_reason": "stop",
            "usage": _usage(used),
            "error_type": None,
            "error": None,
        },
    )


def compress(turn: Turn) -> str:
    """Hermes compresses the context mid-turn (`agent/conversation_compression.py`): the agent's
    `session_id` rotates, and `conversation_id`, the lineage's root, stays. Every later event of
    the turn states the new session id. Returns it."""
    turn.conversation = conversation_of(turn)
    turn.session_id = f"{turn.session_id}_c{uuid.uuid4().hex[:4]}"
    return turn.session_id


# -- what Hermes does to the tool calls a model call reported, before it dispatches them -------
# `agent/turn_tool_validation.py::validate_tool_calls` runs after `post_api_request` has stated
# the calls as the provider sent them. Each helper returns the `Call` as Hermes then dispatches
# it; the fake's `post_api_request` keeps stating the original.


def reused_ids_renamed(call: Call) -> Call:
    """`agent/message_sanitization.py::uniquify_tool_call_ids`: a provider that names every call
    `call_0` turn after turn reuses an id the session already holds, and Hermes runs the call
    under `<id>_d<n>` instead."""
    return replace(call, tool_calls=tuple((f"{i}_d2", n) for i, n in call.tool_calls))


def provider_ids_normalized(call: Call) -> Call:
    """`agent/message_sanitization.py::normalize_provider_tool_call_ids`: a parallel batch whose
    ids are all provider-minted (`chatcmpl-tool-*`, which vLLM sends) runs under
    `call_<12 hex of the id's sha256>`. The fake's own ids stand in for the provider's here; the
    smoke row `tool_ids_normalized` drives the real prefix through a real Hermes."""
    assert len(call.tool_calls) > 1, "Hermes normalizes a batch of two or more"

    def digest(i: str) -> str:
        return f"call_{hashlib.sha256(i.encode()).hexdigest()[:12]}"

    return replace(call, tool_calls=tuple((digest(i), n) for i, n in call.tool_calls))


def runs_as(call: Call, asked: str, name: str) -> Call:
    """A call Hermes runs under another tool name and the same id:
    `agent/agent_runtime_helpers.py::repair_tool_call` (a misspelled name repaired to a
    registered tool) or a legacy alias (`model_tools.py::_LEGACY_TOOL_ALIASES`, `todo` runs as
    `todo_list`). A deferred tool asked for by its own name is routed through the `tool_call`
    bridge and unwrapped again, so it runs under the name asked (measured)."""
    renamed = tuple((i, name if n == asked else n) for i, n in call.tool_calls)
    return replace(call, tool_calls=renamed)


#: `tools/tool_search_catalog.py::TOOL_CALL_NAME`.
BRIDGE = "tool_call"


def bridge_split(call: Call, tools: Sequence[str]) -> Call:
    """One `tool_call` bridge call carrying `tools`, as Hermes dispatches it: split after
    `post_api_request` into one call per tool (`agent/tool_call_batches.py::
    expand_local_tool_batches`), the first under the asked id and the rest under new ids
    (`deterministic_call_id`, not derived from the asked one), each run under its tool's own
    name (`agent/tool_executor.py::_unwrap_tool_search_call`). One tool is not split, and still
    runs under its own name. Measured on a real Hermes with two `todo_list` calls."""
    tcid = next(i for i, n in call.tool_calls if n == BRIDGE)
    split = [(tcid if k == 0 else f"call_split_{k}", t) for k, t in enumerate(tools)]
    rest = [(i, n) for i, n in call.tool_calls if i != tcid]
    return replace(call, tool_calls=(*split, *rest))


def connector_batch(h: FakeHermes, call: Call, connectors: Sequence[str]) -> None:
    """A `tool_call` bridge call whose entries are all connector tools. Hermes does not split
    it, so its frame runs under the bridge's own name, and inside that frame
    `tools/connectors/dispatch.py::dispatch_connector_batch` runs each entry through
    `model_tools.py::handle_function_call`: its own request middleware, `pre_tool_call`,
    `tool_execution` frame and `post_tool_call`, on the same worker, under the wrapper's
    `api_request_id` and the id `<wrapper id>#<position>`. From the source only: a connector
    needs an outside host, so none was run in the lab."""
    tcid = next(i for i, n in call.tool_calls if n == BRIDGE)

    def entries() -> None:
        for k, name in enumerate(connectors):
            h._tool_job(call, f"{tcid}#{k}", name, "ok", None)

    copied = contextvars.copy_context()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(copied.run, h._tool_job, call, tcid, BRIDGE, "ok", None, entries).result()


def gave_up(h: FakeHermes, turn: Turn) -> None:
    """A turn Hermes stops as partial and never ends (`agent/turn_tool_validation.py`): the
    model names a tool that does not exist three times in a row. Each call answers, its frame
    returns and `post_api_request` fires; no tool runs; then Hermes returns with no
    `post_llm_call`, no `on_session_end` and no `api_request_error`, so the turn stays pinned on
    its loop thread. An answer cut at the output cap (`agent/turn_truncation.py::end_turn`)
    leaves the same events behind but for the call's finish reason. Measured on a real Hermes
    for both."""
    for _ in range(3):
        h.model_call(turn, tool_calls=["lab_no_such_tool"])


def slow_tool(h: FakeHermes, call: Call, name: str, seconds: float) -> None:
    """`FakeHermes.tool` whose handler takes `seconds`, so its frame stays pinned that long."""
    tcid = next(i for i, nm in call.tool_calls if nm == name)
    copied = contextvars.copy_context()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        job = (call, tcid, name, "ok", None, lambda: time.sleep(seconds))
        pool.submit(copied.run, h._tool_job, *job).result()


# -- agents an embedder or a tool runs inside a live turn ---------------------------------------


def tool_starts_agent(h: FakeHermes, call: Call, name: str, agent: Callable[[], None]):
    """A plugin tool whose handler starts `agent` on a thread that carries a copy of the tool's
    own context, as Hermes's own tools do (`tools/thread_context.py::
    propagate_context_to_thread`), and returns at once. The thread is built inside the tool's
    frame and returned unstarted, so a test starts it after the frame exited: the agent then
    runs while the turn that called the tool is live and between two of its frames."""
    tcid = next(i for i, nm in call.tool_calls if nm == name)
    built: list[threading.Thread] = []

    def body() -> None:
        copied = contextvars.copy_context()
        built.append(threading.Thread(target=copied.run, args=(agent,), name="lab-helper"))

    copied = contextvars.copy_context()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(copied.run, h._tool_job, call, tcid, name, "ok", None, body).result()
    return built[0]
