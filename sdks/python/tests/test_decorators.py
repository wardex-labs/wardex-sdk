import asyncio

from wardex_sdk import _hub
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._tracing import agent as agent_deco
from wardex_sdk._tracing import conversation, step, tool, workflow
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

    with conversation("s"):
        assert run() == 42
    spans = _all_spans(t)
    sp = spans["research-graph"]
    assert sp.workflow_name == "research-graph"
    assert ("gen_ai.operation.name", "invoke_workflow") in sp.extra
    assert sp.call_site is not None and sp.call_site.function == "run"


def test_tool_decorator_sets_attrs():
    t = _setup()

    @tool(name="web_search", attributes=ToolAttributes(name="web_search"))
    def search():
        return "ok"

    with conversation("s"):
        search()
    sp = _all_spans(t)["web_search"]
    assert sp.tool is not None and sp.tool.name == "web_search"
    assert ("gen_ai.operation.name", "execute_tool") in sp.extra


def test_step_decorator_maps_execute_step():
    """`@step` spans carry an operation like every other decorator's — the old
    `task()` mapped none, and its spans vanished from any dashboard keyed on
    `gen_ai.operation.name`. The span NAME stays the given name, exactly as
    workflow/agent/tool names come out."""
    t = _setup()

    @step(name="summarize")
    def do():
        return 1

    with conversation("s"):
        do()
    sp = _all_spans(t)["summarize"]
    assert ("gen_ai.operation.name", "execute_step") in sp.extra


def test_agent_decorator_async():
    t = _setup()

    @agent_deco(name="researcher", attributes=AgentAttributes(name="researcher"))
    async def run_agent():
        return "done"

    async def main():
        with conversation("s"):
            return await run_agent()

    assert asyncio.run(main()) == "done"
    sp = _all_spans(t)["researcher"]
    assert sp.agent is not None and sp.agent.name == "researcher"
    assert ("gen_ai.operation.name", "invoke_agent") in sp.extra


# ==========================================================================
# bare form: @wardex.tool with no parentheses, name defaults to fn.__name__
# ==========================================================================


def test_bare_decorators_default_the_name_to_the_function():
    t = _setup()

    @tool
    def web_search():
        return "ok"

    @workflow
    def research_graph():
        return web_search()

    @step
    def summarize():
        return 1

    @agent_deco
    def researcher():
        return summarize()

    with conversation("s"):
        assert research_graph() == "ok"
        assert researcher() == 1
    spans = _all_spans(t)
    assert ("gen_ai.operation.name", "execute_tool") in spans["web_search"].extra
    assert ("gen_ai.operation.name", "invoke_workflow") in spans["research_graph"].extra
    assert spans["research_graph"].workflow_name == "research_graph"
    assert ("gen_ai.operation.name", "execute_step") in spans["summarize"].extra
    assert ("gen_ai.operation.name", "invoke_agent") in spans["researcher"].extra


def test_keyword_form_name_defaults_to_the_function_when_omitted():
    t = _setup()

    @tool(attributes=ToolAttributes(name="lookup"))
    def lookup():
        return "ok"

    with conversation("s"):
        lookup()
    sp = _all_spans(t)["lookup"]
    assert sp.tool is not None and sp.tool.name == "lookup"


def test_bare_decorator_on_an_async_function():
    t = _setup()

    @tool
    async def fetch():
        return "ok"

    async def main():
        with conversation("s"):
            return await fetch()

    assert asyncio.run(main()) == "ok"
    sp = _all_spans(t)["fetch"]
    assert ("gen_ai.operation.name", "execute_tool") in sp.extra
