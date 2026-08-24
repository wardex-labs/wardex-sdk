import pytest

from wardex_sdk import _hub
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._enums import OperationName, ProviderName, SpanKind, StatusCode
from wardex_sdk._tracing import conversation, span
from wardex_sdk._types import Envelope, GenAIAttributes
from wardex_sdk.transport._base import Transport


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[Envelope] = []

    def export(self, envelope: Envelope) -> None:
        self.envelopes.append(envelope)


def _setup() -> _Recording:
    _hub.reset_for_test()
    t = _Recording()
    _hub.set_client(Client(WardexConfig(backend=BackendConfig(api_key="k")), t))
    return t


def test_span_records_to_client_on_exit():
    t = _setup()
    with conversation("session"):
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
    with conversation("session"):
        with span("child"):
            pass
    _hub.get_client().flush()
    spans = {sp.name: sp for sp in t.envelopes[0].spans}
    assert spans["child"].parent_span_id is not None
    assert spans["child"].parent_span_id.value == spans["session"].context.span_id.value
    assert spans["child"].context.trace_id.value == spans["session"].context.trace_id.value


def test_conversation_assigns_conversation_id():
    t = _setup()
    with conversation("session"):
        with span("inner"):
            pass
    _hub.get_client().flush()
    inner = next(sp for sp in t.envelopes[0].spans if sp.name == "inner")
    assert inner.conversation is not None and inner.conversation.conversation_id


def test_conversation_op_surfaces_in_extra():
    t = _setup()
    from wardex_sdk._enums import OperationName

    with conversation("session", op=OperationName.INVOKE_WORKFLOW):
        pass
    _hub.get_client().flush()
    sess = next(sp for sp in t.envelopes[0].spans if sp.name == "session")
    assert ("gen_ai.operation.name", "invoke_workflow") in sess.extra


def test_set_attribute_appears_in_console_output(capsys):
    import wardex_sdk
    from wardex_sdk import _hub

    _hub.reset_for_test()
    wardex_sdk.init(transport=wardex_sdk.ConsoleTransport(), backend=BackendConfig(api_key="k"))
    with wardex_sdk.conversation("s"):
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
    from wardex_sdk._assembly._diag import reset_reports_for_test

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
    from wardex_sdk._assembly._diag import reset_reports_for_test

    t = _setup()
    reset_reports_for_test()
    # `resolve_observed`, because that is what the open path calls: a manual
    # span OBSERVES a parent it did not open, so it asks the same question the
    # byte seams ask — including whether that parent's unit has already closed
    # (design §10.3). Injecting into the symbol the path no longer calls would
    # leave this test green while proving nothing.
    real = tracing.resolve_observed

    def blow(*a, **k):
        raise RuntimeError("broken")

    tracing.resolve_observed = blow
    capsys.readouterr()
    try:
        with span("llm-call") as s:
            s.set_status(StatusCode.OK)
    finally:
        tracing.resolve_observed = real

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


def test_a_bug_in_wardexs_own_span_does_not_silence_the_work_inside_it(monkeypatch):
    """The loss this closes, measured end to end on the default capture mode.

    `capture_mode=AGENT` captures traffic that was issued while a local wardex
    span was ambient. A `wardex.span()` that failed to open leaves nothing
    ambient — so every HTTP request the host makes inside its block was dropped
    at the byte seam, with no counter and no marker. One bug in wardex's own
    span turned into total silence for exactly the work that span was opened to
    watch, and the run read as one that never happened.
    """
    from types import SimpleNamespace

    import wardex_sdk._tracing as tracing
    from wardex_sdk._assembly import latch_ambient
    from wardex_sdk._assembly._diag import reset_reports_for_test
    from wardex_sdk._enums import CaptureMode
    from wardex_sdk._interceptors._seam import ByteSeamInterceptor

    class Seam(ByteSeamInterceptor):
        def _select_tracker(self, obj):
            raise NotImplementedError

        def _resolve_timing(self, obj, st):
            raise NotImplementedError

        def name(self):
            return "probe"

        def install(self, client, ctx=None):
            self._client = client

        def uninstall(self):
            pass

    def kept(broken: bool) -> bool:
        _hub.reset_for_test()
        client = Client(
            WardexConfig(capture_mode=CaptureMode.AGENT, backend=BackendConfig(api_key="k")),
            _Recording(),
        )
        _hub.set_client(client)
        reset_reports_for_test()
        seam = Seam()
        seam._client = client
        if broken:
            monkeypatch.setattr(
                tracing,
                "resolve_parentage",
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("broken")),
            )
        with span("my agent turn"):
            # What a real socket write does: latch the carrier the host is on.
            txn = SimpleNamespace(parent=latch_ambient().span_context)
            # The gate is a module function since the deferred-parse split;
            # ask it with the same inputs the seam would seal.
            from wardex_sdk._assembly import capture_mode_of
            from wardex_sdk._interceptors import _seam as seam_mod

            return seam_mod._should_capture(
                seam._transport_prefilter(SimpleNamespace()),
                txn,
                None,
                mode=capture_mode_of(seam._client),
            )

    assert kept(False) is True
    assert kept(True) is True, "a wardex bug at the top silenced everything under it"


# ==========================================================================
# conversation(): a conversation id, not a trace root
# ==========================================================================


def test_conversation_uses_an_explicit_id_verbatim():
    """A multi-turn chat app passes its own session id so its turns share ONE
    conversation; `id=None` mints a fresh uuid4 per block."""
    t = _setup()
    with conversation("turn-1", id="chat-777"):
        with span("inner"):
            pass
    _hub.get_client().flush()
    inner = next(sp for sp in t.envelopes[0].spans if sp.name == "inner")
    assert inner.conversation is not None
    assert inner.conversation.conversation_id == "chat-777"


def test_conversation_joins_the_ambient_trace_as_a_child():
    """The reason it is not called "trace": no new trace_id is minted here."""
    t = _setup()
    with span("outer") as outer:
        with conversation("turn"):
            pass
    _hub.get_client().flush()
    spans = {sp.name: sp for sp in t.envelopes[0].spans}
    assert spans["turn"].context.trace_id.value == outer.context.trace_id.value
    assert spans["turn"].parent_span_id is not None
    assert spans["turn"].parent_span_id.value == outer.context.span_id.value


# ==========================================================================
# the CM objects are not decorators — the async footgun is closed
# ==========================================================================


@pytest.mark.parametrize("factory", [lambda: span("x"), lambda: conversation("x")])
def test_the_cm_object_refuses_to_decorate(factory):
    """`@contextmanager` objects are ContextDecorators, and `@span("x")` on an
    `async def` silently closed the span before any awaited work ran. Calling
    the CM object is refused, naming the decorators that do it right."""
    _setup()
    cm = factory()
    with pytest.raises(TypeError, match="@workflow/@agent/@step/@tool"):
        cm(lambda: None)


# ==========================================================================
# scope tags/user reach exported spans (the stratum used to be write-only)
# ==========================================================================


def test_set_tag_lands_on_exported_spans():
    import wardex_sdk

    t = _setup()
    wardex_sdk.set_tag("tenant", "acme")
    with span("llm-call"):
        pass
    _hub.get_client().flush()
    sp = next(s for s in t.envelopes[0].spans if s.name == "llm-call")
    assert ("tenant", "acme") in sp.extra


def test_a_span_local_attribute_wins_over_the_scope_tag():
    import wardex_sdk

    t = _setup()
    wardex_sdk.set_tag("tenant", "scope-says")
    with span("llm-call") as s:
        s.set_attribute("tenant", "span-says")
    _hub.get_client().flush()
    sp = next(s for s in t.envelopes[0].spans if s.name == "llm-call")
    assert ("tenant", "span-says") in sp.extra
    assert ("tenant", "scope-says") not in sp.extra


def test_set_user_maps_to_user_attributes():
    import wardex_sdk
    from wardex_sdk import UserInfo

    t = _setup()
    wardex_sdk.set_user(
        UserInfo(id="u-1", email="u@example.com", username="ada", ip_address="10.1.2.3")
    )
    with span("llm-call"):
        pass
    _hub.get_client().flush()
    sp = next(s for s in t.envelopes[0].spans if s.name == "llm-call")
    assert ("user.id", "u-1") in sp.extra
    assert ("user.email", "u@example.com") in sp.extra
    assert ("user.name", "ada") in sp.extra
    assert ("client.address", "10.1.2.3") in sp.extra


def test_set_user_skips_none_fields():
    import wardex_sdk
    from wardex_sdk import UserInfo

    t = _setup()
    wardex_sdk.set_user(UserInfo(id="u-1"))
    with span("llm-call"):
        pass
    _hub.get_client().flush()
    sp = next(s for s in t.envelopes[0].spans if s.name == "llm-call")
    assert ("user.id", "u-1") in sp.extra
    keys = {k for k, _ in sp.extra}
    assert "user.email" not in keys
    assert "user.name" not in keys
    assert "client.address" not in keys


def test_set_user_none_clears_the_user():
    import wardex_sdk
    from wardex_sdk import UserInfo

    t = _setup()
    wardex_sdk.set_user(UserInfo(id="u-1", email="u@example.com"))
    wardex_sdk.set_user(None)
    with span("llm-call"):
        pass
    _hub.get_client().flush()
    sp = next(s for s in t.envelopes[0].spans if s.name == "llm-call")
    keys = {k for k, _ in sp.extra}
    assert not any(k.startswith("user.") for k in keys)
    assert "client.address" not in keys


def test_isolation_scope_tags_do_not_leak_to_spans_outside():
    import wardex_sdk

    t = _setup()
    with wardex_sdk.isolation_scope():
        wardex_sdk.set_tag("request", "r-1")
        with span("inside"):
            pass
    with span("outside"):
        pass
    _hub.get_client().flush()
    spans = {sp.name: sp for sp in t.envelopes[0].spans}
    assert ("request", "r-1") in spans["inside"].extra
    assert ("request", "r-1") not in spans["outside"].extra


def test_a_current_scope_tag_overrides_the_isolation_scopes():
    """The Global → Isolation → Current layering lands the overriding value."""
    import wardex_sdk

    t = _setup()
    wardex_sdk.set_tag("env", "isolation-says")
    with _hub.new_scope() as current:
        current.set_tag("env", "current-says")
        with span("llm-call"):
            pass
    _hub.get_client().flush()
    sp = next(s for s in t.envelopes[0].spans if s.name == "llm-call")
    assert ("env", "current-says") in sp.extra
    assert ("env", "isolation-says") not in sp.extra


def test_snapshots_are_not_stamped_with_scope_tags():
    import wardex_sdk
    from wardex_sdk._types import ToolDefinitionSet

    t = _setup()
    wardex_sdk.set_tag("tenant", "acme")
    with conversation("s"):
        wardex_sdk.capture_state_snapshot(
            turn_index=0,
            conversation_state=b"{}",
            tool_definitions=ToolDefinitionSet(),
        )
    _hub.get_client().flush()
    snapshots = t.envelopes[0].state_snapshots
    assert snapshots, "the snapshot must still be captured"
    for snap in snapshots:
        assert ("tenant", "acme") not in (snap.attributes or ())
