from wardex_sdk import _enums


def test_retention_class_values():
    assert _enums.RetentionClass.SUMMARY_ONLY.value == "summary_only"
    assert _enums.RetentionClass.REPLAYABLE.value == "replayable"
    assert _enums.RetentionClass.FORENSIC.value == "forensic"


def test_provider_name_is_open_enum_string():
    assert _enums.ProviderName.ANTHROPIC.value == "anthropic"
    assert _enums.ProviderName.GCP_GEMINI.value == "gcp.gemini"


def test_operation_name_values():
    assert _enums.OperationName.CHAT.value == "chat"
    assert _enums.OperationName.INVOKE_WORKFLOW.value == "invoke_workflow"
