from wardex_sdk import _enums


def test_pii_mode_has_only_implemented_members():
    """REDACT and HASH were selectable names whose only behavior was to raise;
    no selectable no-ops — each returns as a member when it ships."""
    assert {m.value for m in _enums.PIIMode} == {"mask", "off"}


def test_adapter_name_has_a_member_iff_its_adapter_ships():
    """LANGCHAIN and OPENAI_AGENTS were phantom members — names a user could
    select that installed nothing at all. Same doctrine as InterceptorName's
    GRPC/WEBSOCKET/SSE removals: a name that cannot be spelled needs no
    validation, and each returns as a member when its adapter ships —
    OPENAI_AGENTS has; LANGCHAIN has not."""
    assert {m.value for m in _enums.AdapterName} == {
        "anthropic_agent_sdk",
        "langgraph",
        "openai_agents",
    }
    assert not hasattr(_enums.AdapterName, "LANGCHAIN")


def test_provider_name_is_open_enum_string():
    assert _enums.ProviderName.ANTHROPIC.value == "anthropic"
    assert _enums.ProviderName.GCP_GEMINI.value == "gcp.gemini"


def test_operation_name_values():
    assert _enums.OperationName.CHAT.value == "chat"
    assert _enums.OperationName.INVOKE_WORKFLOW.value == "invoke_workflow"


def test_span_kind_carries_all_five_otel_kinds():
    """PRODUCER/CONSUMER included: the SDK pitches Celery/Kafka propagation and
    could not express the kinds those spans are. The values are the proto enum's
    (SPAN_KIND_* names, appended, never renumbered) and the codec maps them
    mechanically off the schema."""
    assert {m.value for m in _enums.SpanKind} == {
        "internal",
        "client",
        "server",
        "producer",
        "consumer",
    }
