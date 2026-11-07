"""The pydantic-ai bridge, run against the shared adapter conformance suite, in both modes.

Everything here is the SUBJECT; the claims are in `wardex_sdk.testing.conformance` and are the
ones the other adapters answer next door. Every check runs twice:

* `wardex_owned` — instrumentation is off, so the adapter turns it on. Its one seam is
  `Agent._instrument_default`, patched from `False` to wardex's own settings and handed back by
  identity (`False` is a singleton).
* `user_owned` — the host turned it on with its own SDK provider. The seam is that provider's
  processor tuple, patched to carry wardex's processor and handed back by identity; the provider
  starts with one host processor so the tuple is never the empty-tuple singleton.
  `Agent._instrument_default` does not change in this mode and is not a seam (the identity check
  would call it a patch that did nothing).

The workload makes no HTTP call. Every framework `chat` span ships as a unit, so the outer model
names each of its requests (`o-0`, `o-1`), and its function opens one stand-in child per request
where the wire CHAT would be: that child must nest under the chat unit, which is the claim the
bridge exists for. The outer model's first answer calls two async tools, each running an inner
agent, so the two chains below need two different answers to "which unit is current" at once.

`usage_expected` is `"totals"`: `FunctionModel` reports usage, no wire request is issued under any
chat unit in the harness, so the core promotes each chat's demoted numbers into the typed block
(marked `no_wire_evidence`). No cache tier is reported.
"""

from __future__ import annotations

import asyncio
import re
import threading
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from _pydantic_ai_support import MODES, PerRequestModel, instrumentation_mode
from wardex_sdk._adapters._context import Placement
from wardex_sdk._adapters._pydantic_ai import PydanticAiAdapter
from wardex_sdk._assembly import SpanIntent, ToolAttributes, UnitKind
from wardex_sdk._enums import AdapterName, ToolExecutionType, ToolType
from wardex_sdk.testing import AdapterConformanceSuite, AdapterSubject, StalledRun


@pytest.fixture(params=MODES)
def instrumentation(request):
    with instrumentation_mode(request.param) as setup:
        yield setup


def _answered(messages: list[Any]) -> bool:
    return any(
        isinstance(part, ToolReturnPart)
        for message in messages
        for part in getattr(message, "parts", ())
    )


def _probe(live: Any, model: PerRequestModel) -> None:
    """A unit opened from inside the model body, where the wire CHAT would be. Only when
    installed: the zero-point run opens nothing of wardex's own."""
    if live.ctx is None:
        return
    name = f"probe-{model.stem}-{model.n}"

    def describe(h: Any) -> None:
        h.draft.set_tool(
            ToolAttributes(
                name=name, type=ToolType.FUNCTION, execution_type=ToolExecutionType.IN_PROCESS
            )
        )

    handle = live.ctx.open_run(
        UnitKind.CALL,
        intent=SpanIntent.EXECUTE_TOOL,
        placement=Placement.NESTED,
        subject=name,
        describe=describe,
    )
    handle.close()


def _inner(label: str) -> Agent:
    async def answer(messages: list[Any], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(f"inner {label}")])

    return Agent(FunctionModel(answer, model_name=label), name=f"inner_{label}")


def _outer(live: Any) -> Agent:
    model: PerRequestModel

    async def answer(messages: list[Any], info: Any) -> ModelResponse:
        _probe(live, model)
        if _answered(messages):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(
            parts=[
                ToolCallPart("delegate_a", {}, tool_call_id="call_a"),
                ToolCallPart("delegate_b", {}, tool_call_id="call_b"),
            ]
        )

    model = PerRequestModel("o", answer)
    outer = Agent(model, name="outer")
    inner_a, inner_b = _inner("a"), _inner("b")

    @outer.tool_plain
    async def delegate_a() -> str:
        return (await inner_a.run("go")).output

    @outer.tool_plain
    async def delegate_b() -> str:
        return (await inner_b.run("go")).output

    return outer


def workload(live: Any) -> Any:
    return _outer(live).run_sync("hi").output


def stall(live: Any) -> StalledRun:
    """A run parked inside an async tool on a background thread, its first chat finished."""
    gate = threading.Event()
    entered = threading.Event()

    async def answer(messages: list[Any], info: Any) -> ModelResponse:
        if _answered(messages):
            return ModelResponse(parts=[TextPart("resumed")])
        return ModelResponse(parts=[ToolCallPart("wait", {}, tool_call_id="call_w")])

    agent = Agent(FunctionModel(answer, model_name="s"), name="stalled")

    @agent.tool_plain
    async def wait() -> str:
        entered.set()
        await asyncio.to_thread(gate.wait, 10)
        return "ok"

    result: dict[str, Any] = {}

    def drive() -> None:
        try:
            result["out"] = agent.run_sync("hi").output
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            result["exc"] = exc

    thread = threading.Thread(target=drive, daemon=True)
    thread.start()
    assert entered.wait(10), "the tool never started; the stall never happened"

    def resume() -> Any:
        gate.set()
        thread.join(15)
        assert "exc" not in result, result
        return result.get("out")

    return StalledRun(root="invoke_agent stalled", resume=resume)


#: The tree the workload builds, as span-name chains from the root down.
_CHAINS = (
    ("invoke_agent outer", "execute_tool delegate_a", "invoke_agent inner_a", "chat a"),
    ("invoke_agent outer", "execute_tool delegate_b", "invoke_agent inner_b", "chat b"),
    ("invoke_agent outer", "chat o-0", "execute_tool probe-o-0"),
    ("invoke_agent outer", "chat o-1", "execute_tool probe-o-1"),
)


@pytest.fixture
def subject(instrumentation) -> AdapterSubject:
    if instrumentation.mode == "wardex_owned":

        def seams() -> dict[str, object]:
            return {"Agent._instrument_default": vars(Agent)["_instrument_default"]}

    else:

        def seams() -> dict[str, object]:
            return {"processors": instrumentation.processors()}

    return AdapterSubject(
        name=AdapterName.PYDANTIC_AI.value,
        module="wardex_sdk._adapters._pydantic_ai",
        factory=PydanticAiAdapter,
        seams=seams,
        workload=workload,
        chains=_CHAINS,
        stall=stall,
        detect_package="pydantic_ai",
        usage_expected="totals",
    )


@pytest.mark.parametrize("check", AdapterConformanceSuite.CHECKS)
def test_conformance(subject, check):
    AdapterConformanceSuite(subject).run(check)


#: What the README's framework table calls each kind of span, and the span name it is.
_README_NOUNS = {"agent run": "invoke_agent ", "model call": "chat ", "tool call": "execute_tool "}


def test_the_readme_row_names_exactly_the_span_kinds_the_declared_tree_has():
    """Docs truth: each item the README's table says becomes a span is a kind of span the
    conformance subject's tree declares, and the tree has no kind the row leaves out."""
    readme = (Path(__file__).resolve().parents[3] / "README.md").read_text(encoding="utf-8")
    (row,) = [line for line in readme.splitlines() if line.startswith("| Pydantic AI ")]
    found = re.search(r"\| each (.+?), read from", row)
    assert found is not None, row
    nouns = {n.strip() for n in re.split(r",| and ", found.group(1)) if n.strip()}
    assert nouns == set(_README_NOUNS)
    names = {name for chain in _CHAINS for name in chain}
    for noun, prefix in _README_NOUNS.items():
        assert any(name.startswith(prefix) for name in names), noun
    assert all(name.startswith(tuple(_README_NOUNS.values())) for name in names)
