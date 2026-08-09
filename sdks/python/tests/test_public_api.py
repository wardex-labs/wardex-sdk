import pytest

import wardex_sdk
from wardex_sdk import _hub
from wardex_sdk._config import BackendConfig
from wardex_sdk._types import ToolDefinitionSet


def setup_function():
    _hub.reset_for_test()


def test_init_console_and_trace_flush(capsys):
    wardex_sdk.init(transport=wardex_sdk.ConsoleTransport(), backend=BackendConfig(api_key="k"))
    with wardex_sdk.trace("s"):
        with wardex_sdk.span("inner") as sp:
            sp.input_data = b"hi"
    wardex_sdk.flush()
    out = capsys.readouterr().out
    assert "inner" in out


def test_capture_state_snapshot_recorded():
    wardex_sdk.init(backend=BackendConfig(api_key="k"))
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


def test_capture_limits_is_public():
    assert "CaptureLimits" in wardex_sdk.__all__
    assert wardex_sdk.CaptureLimits().max_body_bytes is None


def test_capture_state_snapshot_with_input_refs():
    wardex_sdk.init(backend=BackendConfig(api_key="k"))
    with wardex_sdk.trace("s"):
        wardex_sdk.capture_state_snapshot(
            turn_index=1,
            conversation_state=b'{"messages":[]}',
            input_refs=[("POORCODE.md", "sha256:9f8e")],
            attributes={"code.git.head_sha": "a1b2c3d"},
        )
    wardex_sdk.close()


def test_interceptors_still_exports_ssl_interceptor_lazily(monkeypatch):
    """The public name survived the eager import going away, and stayed lazy.

    `wardex_sdk.interceptors` has no leading underscore and has carried this
    name on its `__all__` since the seam existed, so dropping it would break a
    pinned caller's import in a refactor. Both halves are asserted because the
    reason the eager import went is that it dragged `_ssl` — and the native
    extension under it — into every import of this package, including the
    teardown paths that exist to work without one: the name resolves, and it
    still is not in the module dict afterwards, so nothing was bound at import.
    """
    from wardex_sdk import interceptors
    from wardex_sdk.interceptors import _ssl

    assert "SSLInterceptor" in interceptors.__all__
    assert interceptors.SSLInterceptor is _ssl.SSLInterceptor
    assert "SSLInterceptor" not in vars(interceptors)

    # Resolved through the MODULE on every access, which is what the isolation
    # suite's substitution needs: a name bound at import time would hand back
    # the real seam and measure nothing.
    sentinel = object()
    monkeypatch.setattr(_ssl, "SSLInterceptor", sentinel)
    assert interceptors.SSLInterceptor is sentinel


def test_interceptors_rejects_an_unknown_attribute():
    from wardex_sdk import interceptors

    with pytest.raises(AttributeError, match="Nope"):
        _ = interceptors.Nope
