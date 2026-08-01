import pytest

from wardex_sdk import _hub
from wardex_sdk._client import Client
from wardex_sdk._config import WardexConfig
from wardex_sdk._enums import OperationName, ProviderName, SpanKind, StatusCode
from wardex_sdk._tracing import span, trace
from wardex_sdk._types import GenAIAttributes, InternalEnvelope
from wardex_sdk.transport._base import Transport


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[InternalEnvelope] = []

    def export(self, envelope: InternalEnvelope) -> None:
        self.envelopes.append(envelope)


def _setup() -> _Recording:
    _hub.reset_for_test()
    t = _Recording()
    _hub.set_client(Client(WardexConfig(api_key="k"), t))
    return t


def test_span_records_to_client_on_exit():
    t = _setup()
    with trace("session"):
        with span("llm-call", kind=SpanKind.CLIENT) as s:
            s.set_gen_ai(
                GenAIAttributes(operation=OperationName.CHAT, provider=ProviderName.ANTHROPIC)
            )
            s.input_data = b'{"model":"x"}'
            s.set_status(StatusCode.OK)
    _hub.get_client().flush()
    spans = t.envelopes[0].spans
    names = {sp.name for sp in spans}
    assert "session" in names and "llm-call" in names
    llm = next(sp for sp in spans if sp.name == "llm-call")
    assert llm.gen_ai is not None and llm.gen_ai.operation == OperationName.CHAT
    assert llm.input_data == b'{"model":"x"}'
    assert llm.status == StatusCode.OK


def test_child_span_has_parent_of_active():
    t = _setup()
    with trace("session"):
        with span("child"):
            pass
    _hub.get_client().flush()
    spans = {sp.name: sp for sp in t.envelopes[0].spans}
    assert spans["child"].parent_span_id is not None
    assert spans["child"].parent_span_id.value == spans["session"].context.span_id.value
    assert spans["child"].context.trace_id.value == spans["session"].context.trace_id.value


def test_trace_assigns_conversation_id():
    t = _setup()
    with trace("session"):
        with span("inner"):
            pass
    _hub.get_client().flush()
    inner = next(sp for sp in t.envelopes[0].spans if sp.name == "inner")
    assert inner.conversation is not None and inner.conversation.conversation_id


def test_trace_op_surfaces_in_extra():
    t = _setup()
    from wardex_sdk._enums import OperationName

    with trace("session", op=OperationName.INVOKE_WORKFLOW):
        pass
    _hub.get_client().flush()
    sess = next(sp for sp in t.envelopes[0].spans if sp.name == "session")
    assert ("gen_ai.operation.name", "invoke_workflow") in sess.extra


def test_set_attribute_appears_in_console_output(capsys):
    import wardex_sdk
    from wardex_sdk import _hub

    _hub.reset_for_test()
    wardex_sdk.init(transport=wardex_sdk.ConsoleTransport(), api_key="k")
    with wardex_sdk.trace("s"):
        with wardex_sdk.span("inner") as sp:
            sp.set_attribute("code.git.head_sha", "a1b2c3d")
            sp.set_attribute("turn", 1)
    wardex_sdk.flush()
    out = capsys.readouterr().out
    assert "code.git.head_sha" in out
    assert "a1b2c3d" in out


# ==========================================================================
# A wardex bug inside `wardex.span()` costs the span, never the block
# ==========================================================================


def test_a_bug_opening_a_manual_span_still_runs_the_hosts_block(monkeypatch):
    """`wardex.span()` is the SDK's published context manager, so its `with`
    body is the host's own code. Latching the ambient scope, resolving the edge
    and building the draft all run before that body — a defect in any of them
    used to take the block with it.
    """
    import wardex_sdk._tracing as tracing
    from wardex_sdk.assembly._diag import reset_reports_for_test

    _setup()
    reset_reports_for_test()
    monkeypatch.setattr(
        tracing, "resolve_parentage", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("broken"))
    )

    ran = []
    with span("llm-call") as s:
        ran.append(s)
        s.input_data = b"the host wrote this"
        s.set_status(StatusCode.OK)
        value = "the host's own"

    assert len(ran) == 1
    assert value == "the host's own"
    assert ran[0].input_data == b"the host wrote this"


def test_a_bug_opening_a_manual_span_emits_nothing_and_says_why(capsys):
    """No span, and one line saying so — the difference between wardex being
    broken here and wardex never having been installed.
    """
    import wardex_sdk._tracing as tracing
    from wardex_sdk.assembly._diag import reset_reports_for_test

    t = _setup()
    reset_reports_for_test()
    real = tracing.resolve_parentage

    def blow(*a, **k):
        raise RuntimeError("broken")

    tracing.resolve_parentage = blow
    capsys.readouterr()
    try:
        with span("llm-call") as s:
            s.set_status(StatusCode.OK)
    finally:
        tracing.resolve_parentage = real

    err = capsys.readouterr().err
    _hub.get_client().flush()
    assert t.envelopes == []
    assert len([line for line in err.splitlines() if line.strip()]) == 1
    assert "will not be recorded" in err, err


def test_the_hosts_exception_leaves_a_manual_span_as_the_same_object():
    """Identity, and `BaseException` included: wardex may not become the library
    in the process that eats a real Ctrl-C.
    """
    _setup()
    for exc in (ValueError("host"), KeyboardInterrupt()):
        with pytest.raises(type(exc)) as caught:
            with span("llm-call"):
                raise exc
        assert caught.value is exc
