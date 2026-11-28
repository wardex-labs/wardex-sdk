"""E2E: what wardex records while another instrumentation library is on in the same process.

Not collected by pytest (the name misses `python_files`). It runs one adapter's conformance
workload twice in this process, through the real adapter registry: first with wardex alone,
then with the neighbour turned on. It exits 0 only when both hold:

* the two trees are the same: the same spans under the same parents with the same
  capture markers;
* the neighbour recorded spans of its own during the second run, so "the same" cannot mean
  "the neighbour was never on".

    python sdks/python/tests/e2e_neighbour.py <neighbour> [--out result.json]

The workloads and their in-process stand-in models are the conformance tests' own, imported
from the files beside this one. No network, no key.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: neighbour -> (conformance module, adapter class in it, the distribution that is the neighbour)
NEIGHBOURS = {
    "openinference_openai_agents": (
        "test_openai_agents_conformance",
        "OpenAIAgentsAdapter",
        "openinference-instrumentation-openai-agents",
    ),
    "openinference_langchain": (
        "test_langgraph_conformance",
        "LangGraphAdapter",
        "openinference-instrumentation-langchain",
    ),
    "logfire_pydantic_ai": ("test_pydantic_ai_conformance", "PydanticAiAdapter", "logfire"),
}

_SEEN: list[str] = []


def _processor():
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    class Keep(SpanExporter):
        def export(self, spans):  # noqa: ANN001, ANN201
            _SEEN.extend(span.name for span in spans)
            return SpanExportResult.SUCCESS

    return SimpleSpanProcessor(Keep())


def prepare(name: str) -> None:
    """What both runs need before either: nothing may leave this machine."""
    if name == "openinference_openai_agents":
        from agents.tracing import set_trace_processors

        # The framework's default processor posts each run to the real API.
        set_trace_processors([])


def turn_on(name: str) -> None:
    if name == "openinference_openai_agents":
        from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider()
        provider.add_span_processor(_processor())
        OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)
    elif name == "openinference_langchain":
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider()
        provider.add_span_processor(_processor())
        LangChainInstrumentor().instrument(tracer_provider=provider)
    elif name == "logfire_pydantic_ai":
        import logfire

        logfire.configure(
            send_to_logfire=False, console=False, additional_span_processors=[_processor()]
        )
        logfire.instrument_pydantic_ai()
    else:
        raise SystemExit(f"no such neighbour: {name}")


def observe(module, factory) -> list:  # noqa: ANN001
    """The workload's tree as (span, its parent, its capture markers), sorted."""
    from wardex_sdk.testing.harness import installed_adapter, read_spans

    with installed_adapter(factory) as live:
        module.workload(live)
        nodes = read_spans(live.spans)
    by_id = {node.span_id: node for node in nodes}
    return sorted(
        [
            node.name,
            by_id[node.parent_id].name if node.parent_id in by_id else None,
            sorted(str(getattr(mark, "value", mark)) for mark in node.limitations),
        ]
        for node in nodes
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("neighbour", choices=sorted(NEIGHBOURS))
    parser.add_argument("--out", help="where to write the result as JSON")
    args = parser.parse_args(argv)
    module_name, factory_name, _ = NEIGHBOURS[args.neighbour]
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)

    prepare(args.neighbour)
    alone = observe(module, factory)
    turn_on(args.neighbour)
    before = len(_SEEN)
    beside = observe(module, factory)
    theirs = _SEEN[before:]

    failed = []
    if not alone:
        failed.append("wardex recorded nothing on its own")
    if alone != beside:
        failed.append("the tree differs with the neighbour on")
    if not theirs:
        failed.append("the neighbour recorded no span, so it was not on")
    for title, tree in (("alone", alone), ("beside the neighbour", beside)):
        print(f"--- wardex {title}: {len(tree)} spans")
        for name, parent, marks in tree:
            print(f"  {name}  <-  {parent}  {marks or ''}")
    print(f"--- the neighbour recorded {len(theirs)} spans: {theirs[:8]}")
    for line in failed:
        print(f"FAILED: {line}")
    result = {
        "ok": not failed,
        "failed": failed,
        "checks": {"spans_alone": len(alone), "spans_beside": len(beside), "theirs": len(theirs)},
    }
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    print("ok" if not failed else "not ok")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
