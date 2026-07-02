import wardex_sdk
from wardex_sdk import _hub
from wardex_sdk._types import ToolDefinitionSet


def setup_function():
    _hub.reset_for_test()


def test_init_console_and_trace_flush(capsys):
    wardex_sdk.init(transport=wardex_sdk.ConsoleTransport(), api_key="k")
    with wardex_sdk.trace("s"):
        with wardex_sdk.span("inner") as sp:
            sp.input_data = b"hi"
    wardex_sdk.flush()
    out = capsys.readouterr().out
    assert "inner" in out


def test_capture_state_snapshot_recorded():
    wardex_sdk.init(api_key="k")
    with wardex_sdk.trace("s"):
        wardex_sdk.capture_state_snapshot(
            turn_index=0,
            conversation_state=b'{"messages":[]}',
            tool_definitions=ToolDefinitionSet(),
        )
    wardex_sdk.close()


def test_public_exports_exist():
    for name in (
        "init",
        "trace",
        "span",
        "capture_state_snapshot",
        "set_tag",
        "set_user",
        "isolation_scope",
        "new_scope",
        "flush",
        "close",
        "Transport",
        "NoOpTransport",
        "ConsoleTransport",
        "UserInfo",
        "InputRef",
    ):
        assert hasattr(wardex_sdk, name), name


def test_capture_state_snapshot_with_input_refs():
    wardex_sdk.init(api_key="k")
    with wardex_sdk.trace("s"):
        wardex_sdk.capture_state_snapshot(
            turn_index=1,
            conversation_state=b'{"messages":[]}',
            input_refs=[("POORCODE.md", "sha256:9f8e")],
            attributes={"code.git.head_sha": "a1b2c3d"},
        )
    wardex_sdk.close()
