"""The LangGraph adapter, run against the shared adapter conformance suite.

Everything here is the SUBJECT — six seams, one workload, the tree that
workload must produce, and a run left open mid-stream. The claims are in
`wardex_sdk.testing.conformance` and are the same ones the Agent SDK adapter
answers next door.

The graph builders come from `test_langgraph_adapter`, which is where the
adapter-specific tests that go BEYOND these invariants still live: the retry
granularity claim, the `confirm_active` sites, the surface probe, the
control-flow classification and the AST rule about framework identifiers. None
of those is a shared invariant, and none of them moved.
"""

from __future__ import annotations

import pytest

from test_langgraph_adapter import chain
from wardex_sdk._adapters._langgraph import LangGraphAdapter
from wardex_sdk._enums import AdapterName
from wardex_sdk.testing import AdapterConformanceSuite, AdapterSubject, StalledRun

_N_NODES = 3


def seams() -> dict[str, object]:
    """The six attributes the adapter replaces, by their live values.

    Read through the framework's own modules rather than off the adapter: what
    has to be restored is the FRAMEWORK's attribute, and asking the adapter what
    it patched would take its word for the very thing under test.
    """
    from langgraph.prebuilt.tool_node import ToolNode
    from langgraph.pregel import _runner
    from langgraph.pregel import main as pregel_mod

    return {
        "Pregel.stream": pregel_mod.Pregel.stream,
        "Pregel.astream": pregel_mod.Pregel.astream,
        "_runner.run_with_retry": _runner.run_with_retry,
        "_runner.arun_with_retry": _runner.arun_with_retry,
        "ToolNode._run_one": ToolNode._run_one,
        "ToolNode._arun_one": ToolNode._arun_one,
    }


def workload(live):  # noqa: ANN001, ANN201
    """A linear graph whose every node opens one wire-shaped span.

    `leaves` follows the context, which is what makes the same function usable
    for the zero-point run: with nothing installed there is no context to open a
    span through, and a workload that opened one anyway would emit spans the
    adapter had nothing to do with.
    """
    leaves = live.ctx is not None
    return chain(live.ctx, _N_NODES, name="Conformance", leaves=leaves).invoke({"trail": []})


def stall(live) -> StalledRun:  # noqa: ANN001
    """A `stream()` the host has pumped once and not finished.

    The run's SESSION unit lives for as long as the host iterates, so this is
    the shape where a shutdown arrives with a span still open — and `resume` is
    the host carrying on afterwards, which must not see an exception wardex
    invented.
    """
    it = chain(live.ctx, _N_NODES, name="Stalled", leaves=False).stream({"trail": []})
    next(it)
    return StalledRun(root="invoke_workflow Stalled", resume=lambda: list(it))


@pytest.fixture
def subject() -> AdapterSubject:
    return AdapterSubject(
        name=AdapterName.LANGGRAPH.value,
        module="wardex_sdk._adapters._langgraph",
        factory=LangGraphAdapter,
        seams=seams,
        workload=workload,
        chains=tuple(
            ("invoke_workflow Conformance", f"execute_step n{i}", f"execute_tool leaf-n{i}")
            for i in range(_N_NODES)
        ),
        stall=stall,
        detect_package="langgraph",
    )


@pytest.mark.parametrize("check", AdapterConformanceSuite.CHECKS)
def test_conformance(subject, check):
    AdapterConformanceSuite(subject).run(check)
