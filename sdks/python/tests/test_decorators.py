import asyncio

from wardex_sdk import _hub
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._tracing import agent as agent_deco
from wardex_sdk._tracing import task, tool, trace, workflow
from wardex_sdk._types import AgentAttributes, InternalEnvelope, ToolAttributes
from wardex_sdk.transport._base import Transport


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[InternalEnvelope] = []

    def export(self, envelope: InternalEnvelope) -> None:
        self.envelopes.append(envelope)


def _setup() -> _Recording:
    _hub.reset_for_test()
    t = _Recording()
    _hub.set_client(Client(WardexConfig(backend=BackendConfig(api_key="k")), t))
    return t


def _all_spans(t: _Recording):
    _hub.get_client().flush()
    return {sp.name: sp for sp in t.envelopes[0].spans}


def test_workflow_decorator_sync():
    t = _setup()

    @workflow(name="research-graph")
    def run():
        return 42

    with trace("s"):
        assert run() == 42
    spans = _all_spans(t)
    sp = spans["research-graph"]
    assert sp.workflow_name == "research-graph"
    assert ("gen_ai.operation.name", "invoke_workflow") in sp.extra
    assert sp.call_site is not None and sp.call_site.function == "run"


def test_tool_decorator_sets_attrs():
    t = _setup()

    @tool(name="web_search", tool=ToolAttributes(name="web_search"))
    def search():
        return "ok"

    with trace("s"):
        search()
    sp = _all_spans(t)["web_search"]
    assert sp.tool is not None and sp.tool.name == "web_search"
    assert ("gen_ai.operation.name", "execute_tool") in sp.extra


def test_task_decorator_is_sugar_only_no_op():
    t = _setup()

    @task(name="summarize")
    def do():
        return 1

    with trace("s"):
        do()
    sp = _all_spans(t)["summarize"]
    assert sp.extra == ()


def test_agent_decorator_async():
    t = _setup()

    @agent_deco(name="researcher", agent=AgentAttributes(name="researcher"))
    async def run_agent():
        return "done"

    async def main():
        with trace("s"):
            return await run_agent()

    assert asyncio.run(main()) == "done"
    sp = _all_spans(t)["researcher"]
    assert sp.agent is not None and sp.agent.name == "researcher"
    assert ("gen_ai.operation.name", "invoke_agent") in sp.extra
