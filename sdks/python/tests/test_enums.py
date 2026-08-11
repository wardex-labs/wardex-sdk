from wardex_sdk import _enums


def test_pii_mode_has_only_implemented_members():
    """REDACT and HASH were selectable names whose only behavior was to raise;
    no selectable no-ops — each returns as a member when it ships."""
    assert {m.value for m in _enums.PIIMode} == {"mask", "off"}


def test_provider_name_is_open_enum_string():
    assert _enums.ProviderName.ANTHROPIC.value == "anthropic"
    assert _enums.ProviderName.GCP_GEMINI.value == "gcp.gemini"


def test_operation_name_values():
    assert _enums.OperationName.CHAT.value == "chat"
    assert _enums.OperationName.INVOKE_WORKFLOW.value == "invoke_workflow"
