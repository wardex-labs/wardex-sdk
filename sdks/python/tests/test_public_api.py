import pytest

import wardex_sdk
from wardex_sdk import _hub
from wardex_sdk._config import BackendConfig
from wardex_sdk._types import ToolDefinitionSet


def setup_function():
    _hub.reset_for_test()


def test_init_console_and_trace_flush(capsys):
    wardex_sdk.init(
        transport=wardex_sdk.ConsoleTransport(),
        backend=BackendConfig(api_key="k"),
        intercept=False,
    )
    with wardex_sdk.trace("s"):
        with wardex_sdk.span("inner") as sp:
            sp.input_data = b"hi"
    wardex_sdk.flush()
    out = capsys.readouterr().out
    assert "inner" in out


def test_capture_state_snapshot_recorded():
    wardex_sdk.init(backend=BackendConfig(api_key="k"), intercept=False)
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


def test_limits_config_is_public():
    assert "LimitsConfig" in wardex_sdk.__all__
    assert wardex_sdk.LimitsConfig().max_body_bytes is None


def test_capture_state_snapshot_with_input_refs():
    wardex_sdk.init(backend=BackendConfig(api_key="k"), intercept=False)
    with wardex_sdk.trace("s"):
        wardex_sdk.capture_state_snapshot(
            turn_index=1,
            conversation_state=b'{"messages":[]}',
            input_refs=[("POORCODE.md", "sha256:9f8e")],
            attributes={"code.git.head_sha": "a1b2c3d"},
        )
    wardex_sdk.close()


def test_the_interceptor_package_no_longer_re_exports_the_seams():
    """The privatized package's surface is the composition root's entry point.

    The lazy `SSLInterceptor` re-export existed to keep a PUBLIC spelling
    (`from wardex_sdk.interceptors import SSLInterceptor`) alive without
    dragging the TLS seam into every import of the package. The package is
    `wardex_sdk._interceptors` now, so there is no public spelling left to
    keep, and the class is reached through its defining module — which is what
    the isolation suite's substitution needed from the lazy hook anyway. The
    resolution is asserted, not just the `__all__` line: a PEP 562 hook left
    behind would keep the old surface reachable while claiming it is gone.
    """
    from wardex_sdk import _interceptors

    assert _interceptors.__all__ == ["install_configured_interceptors"]
    with pytest.raises(AttributeError):
        _ = _interceptors.SSLInterceptor
    with pytest.raises(AttributeError):
        _ = _interceptors.InterceptorRegistry
