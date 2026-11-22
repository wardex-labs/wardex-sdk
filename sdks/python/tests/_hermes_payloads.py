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
import threading
import uuid
from collections.abc import Callable
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
