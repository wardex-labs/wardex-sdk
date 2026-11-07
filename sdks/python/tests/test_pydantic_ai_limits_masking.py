"""Content from pydantic-ai's spans goes through wardex's capture limits and masking.

The framework's content attributes (a tool's arguments and result, a run's whole message
history) reach a unit only through the capped record path, never as an extra: `set_extra` is
uncapped, and the history is O(run length). Masking happens in the native codec at encode, so the
masking test decodes BOTH encoded wires (the wardex envelope and OTLP) instead of reading the
in-process spans, which are pre-masking by design.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

import wardex_sdk as wardex
from _pydantic_ai_support import MODES, instrumentation_mode, one
from wardex_sdk import _wardex_native
from wardex_sdk._adapters._pydantic_ai import PydanticAiAdapter
from wardex_sdk._config import AdaptersConfig
from wardex_sdk._enums import AdapterName
from wardex_sdk._limits import LimitsConfig
from wardex_sdk.testing import RecordingClient, installed_adapter
from wardex_sdk.transport import Transport

_EMAIL = "john.doe@acme.com"
_CARD = "4111-1111-1111-1111"


@pytest.fixture(params=MODES)
def instrumentation(request):
    with instrumentation_mode(request.param) as setup:
        yield setup


class _CappedClient(RecordingClient):
    """A recording client with a limits config: `context_for` reads `client.config.limits`, and
    the base double's `config = None` can only exercise the core default."""

    class _Config:
        debug = False

        def __init__(self, limits: LimitsConfig) -> None:
            self.limits = limits

    def __init__(self, limits: LimitsConfig) -> None:
        super().__init__(limits=limits)
        self.config = self._Config(limits)


def _agent(argument: str, result: str) -> Agent:
    def answer(messages: list[Any], info: Any) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("lookup", {"q": argument}, tool_call_id="c1")])

    agent = Agent(FunctionModel(answer, model_name="m"), name="capped")

    @agent.tool_plain
    def lookup(q: str) -> str:
        return result

    return agent


@pytest.mark.parametrize("budget", [64, None], ids=["capped", "default"])
def test_a_tool_payload_over_the_budget_is_cut_and_says_so(instrumentation, budget):
    argument, result = "a" * 500, "r" * 500
    client = (
        _CappedClient(LimitsConfig(max_body_bytes=budget))
        if budget is not None
        else RecordingClient()
    )
    with installed_adapter(PydanticAiAdapter, client=client) as live:
        _agent(argument, result).run_sync("go")
        tool = one(live.spans, "execute_tool lookup")
        run = one(live.spans, "invoke_agent capped")
    whole_in, whole_out = f'{{"q":"{argument}"}}'.encode(), result.encode()
    if budget is None:
        assert (tool.input_data, tool.output_data) == (whole_in, whole_out)
        assert not tool.capture_integrity.truncated
        return
    assert len(tool.input_data) == budget and whole_in.startswith(tool.input_data)
    assert len(tool.output_data) == budget and whole_out.startswith(tool.output_data)
    assert tool.capture_integrity.truncated is True
    assert len(run.input_data) == budget, "the run's history is capped the same way"
    assert run.capture_integrity.truncated is True
    assert all(len(str(v)) < 200 for s in live.spans for _, v in s.extra), "no content in extras"


class _Wires(Transport):
    """Both encodings of every envelope, exactly as a real transport would send them."""

    def __init__(self) -> None:
        self.envelopes: list[bytes] = []
        self.otlp: list[bytes] = []

    def export(self, envelope: Any, *, timeout: float | None = None) -> None:
        self.otlp.extend(self.encode(envelope, compress=False))
        self.envelopes.append(
            _wardex_native.codec.encode_envelope(
                envelope,
                self._pii_mode,
                list(self._pii_disabled),
                self._limits,
                **self._pii_names(),
            )
        )

    def envelope_spans(self) -> dict[str, dict[str, Any]]:
        return {
            item["span"]["name"]: item["span"]
            for blob in self.envelopes
            for item in _wardex_native.codec.decode_envelope(blob)["items"]
            if "span" in item
        }

    def otlp_spans(self) -> dict[str, dict[str, Any]]:
        return {
            span["name"]: span
            for blob in self.otlp
            for rs in _wardex_native.codec.decode_otlp_traces(blob)["resource_spans"]
            for ss in rs["scope_spans"]
            for span in ss["spans"]
        }


def test_pii_in_a_prompt_and_a_tool_argument_is_masked_on_both_wires(instrumentation):
    wires = _Wires()
    wardex.init(transport=wires, adapters=AdaptersConfig(enabled=(AdapterName.PYDANTIC_AI,)))
    try:
        _agent(_EMAIL, f"card {_CARD}").run_sync(f"mail {_EMAIL}")
        wardex.flush()
    finally:
        wardex.close()

    envelope = wires.envelope_spans()
    tool, run = envelope["execute_tool lookup"], envelope["invoke_agent capped"]
    assert tool["input_data"] == b'{"q":"[EMAIL]"}'
    assert tool["output_data"] == b"card ****-****-****-1111"
    assert set(tool["capture_integrity"]["redaction_rules"]) == {"email", "credit_card"}
    assert tool["capture_integrity"]["redacted"] is True
    assert b"mail [EMAIL]" in run["input_data"]
    assert run["capture_integrity"]["redacted"] is True

    otlp = wires.otlp_spans()
    tool_attrs = otlp["execute_tool lookup"]["attributes"]
    assert tool_attrs["gen_ai.tool.call.arguments"] == '{"q":"[EMAIL]"}'
    assert tool_attrs["gen_ai.tool.call.result"] == "card ****-****-****-1111"
    assert tool_attrs["wardex.redacted"] is True

    for blob in (*wires.envelopes, *wires.otlp):
        assert _EMAIL.encode() not in blob
        assert _CARD.encode() not in blob


def test_a_raised_message_reaches_no_attribute_on_either_wire(instrumentation):
    """pydantic writes the raised message into the status description and the exception event;
    wardex reads neither, so only the type travels."""

    def answer(messages: list[Any], info: Any) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart("boom", {}, tool_call_id="c1")])

    agent = Agent(FunctionModel(answer, model_name="m"), name="raising")

    @agent.tool_plain
    def boom() -> str:
        raise RuntimeError("secret=abc")

    wires = _Wires()
    wardex.init(transport=wires, adapters=AdaptersConfig(enabled=(AdapterName.PYDANTIC_AI,)))
    try:
        with pytest.raises(RuntimeError):
            agent.run_sync("go")
        wardex.flush()
    finally:
        wardex.close()
    assert wires.envelope_spans()["execute_tool boom"]["name"] == "execute_tool boom"
    for blob in (*wires.envelopes, *wires.otlp):
        assert b"secret=abc" not in blob
