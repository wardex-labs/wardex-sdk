"""Unit tests for the gRPC _emit_span branch — _build_grpc_fields + integration verification."""

from hpack import Encoder

from wardex_sdk._enums import StatusCode
from wardex_sdk.assembly import Limitation
from wardex_sdk.interceptors._ssl import _build_grpc_fields
from wardex_sdk.interceptors._trackers import _Txn


def _msg(payload: bytes, compressed: int = 0) -> bytes:
    return bytes([compressed]) + len(payload).to_bytes(4, "big") + payload


def _txn(grpc_status, req=b"", resp=b"", grpc_message=None) -> _Txn:
    return _Txn(
        method="POST",
        path="/echo.Echo/Say",
        status=200,
        request_body=req,
        response_body=resp,
        parent=None,
        start_ns=0,
        end_ns=1,
        ttfb_ms=0.0,
        version="2",
        content_type="application/grpc",
        grpc_status=grpc_status,
        grpc_message=grpc_message,
    )


_BASE = (("network.protocol.version", "2"),)


def test_ok_status_and_rpc_attrs():
    txn = _txn(0, req=_msg(b"abc"), resp=_msg(b"xyz"))
    name, status, error_type, extra, lims = _build_grpc_fields(txn, _BASE, ())
    assert name == "gRPC /echo.Echo/Say"
    assert status is StatusCode.OK
    assert error_type is None
    assert ("rpc.system", "grpc") in extra
    assert ("rpc.service", "echo.Echo") in extra
    assert ("rpc.method", "Say") in extra
    assert ("rpc.grpc.request.message_count", 1) in extra
    assert ("rpc.grpc.response.message_count", 1) in extra
    assert ("rpc.grpc.status_code", 0) in extra
    assert lims == ()


def test_error_status_maps_name():
    name, status, error_type, extra, lims = _build_grpc_fields(_txn(5), _BASE, ())
    assert status is StatusCode.ERROR
    assert error_type == "NOT_FOUND"
    assert ("rpc.grpc.status_code", 5) in extra


def test_missing_status_falls_back_to_ok_with_marker():
    name, status, error_type, extra, lims = _build_grpc_fields(_txn(None), _BASE, ())
    assert status is StatusCode.OK
    assert error_type is None
    assert Limitation.GRPC_STATUS_UNAVAILABLE in lims


def test_compressed_marker():
    txn = _txn(0, req=_msg(b"abc", compressed=1))
    _, _, _, _, lims = _build_grpc_fields(txn, _BASE, ())
    # Census rename (design §6.5.1): grpc_compressed -> PAYLOAD_COMPRESSED, and
    # markers are Limitation members rather than free strings since step 3a.
    assert Limitation.PAYLOAD_COMPRESSED in lims


def test_truncated_marker():
    txn = _txn(0, resp=_msg(b"hello")[:7])
    _, _, _, _, lims = _build_grpc_fields(txn, _BASE, ())
    assert Limitation.GRPC_MESSAGE_TRUNCATED in lims


def _frame(ftype: int, flags: int, stream_id: int, payload: bytes) -> bytes:
    return (
        len(payload).to_bytes(3, "big")
        + bytes([ftype, flags])
        + stream_id.to_bytes(4, "big")
        + payload
    )


def test_emit_span_builds_grpc_client_span():
    """Verifies the _Http2Tracker._mk → _build_grpc_fields path at the _Txn level.

    Passes a _Txn built by the tracker directly into _build_grpc_fields without a socket,
    to confirm end-to-end field mapping (socket mocking is handled by test_ssl_interceptor).
    """
    from wardex_sdk.interceptors._trackers import _Http2Tracker

    tracker = _Http2Tracker()
    cenc, senc = Encoder(), Encoder()
    req_block = cenc.encode(
        [
            (b":method", b"POST"),
            (b":path", b"/pkg.Svc/Do"),
            (b"content-type", b"application/grpc"),
        ]
    )
    req = _frame(0x1, 0x4, 1, req_block) + _frame(0x0, 0x1, 1, _msg(b"req"))
    tracker.on_request_bytes(req)
    resp_hdr = senc.encode([(b":status", b"200"), (b"content-type", b"application/grpc")])
    trailers = senc.encode([(b"grpc-status", b"5")])
    resp = (
        _frame(0x1, 0x4, 1, resp_hdr)
        + _frame(0x0, 0x0, 1, _msg(b"resp"))
        + _frame(0x1, 0x4 | 0x1, 1, trailers)
    )
    (txn,) = tracker.on_response_bytes(resp)

    name, status, error_type, extra, lims = _build_grpc_fields(
        txn, (("network.protocol.version", "2"),), ()
    )
    assert name == "gRPC /pkg.Svc/Do"
    assert status is StatusCode.ERROR
    assert error_type == "NOT_FOUND"
    assert ("rpc.service", "pkg.Svc") in extra
    assert ("rpc.method", "Do") in extra


def test_status_message_emitted_when_present():
    """If grpc_message is present, it should be emitted on the span as the
    rpc.grpc.status_message attribute."""
    txn = _txn(5, grpc_message="boom")
    _, _, _, extra, _ = _build_grpc_fields(txn, _BASE, ())
    assert ("rpc.grpc.status_message", "boom") in extra


def test_status_message_absent_when_none():
    """If grpc_message is None, the rpc.grpc.status_message attribute should be
    absent from the span."""
    txn = _txn(0)
    _, _, _, extra, _ = _build_grpc_fields(txn, _BASE, ())
    assert all(k != "rpc.grpc.status_message" for k, _ in extra)


def test_framing_failure_on_an_error_status_still_carries_an_error_type(monkeypatch):
    """The fallback branch must not return `ERROR` with no type.

    `SpanDraft.finish()` refuses that pair and the refusal DELETES the span, so
    a gRPC call whose framing wardex cannot parse AND whose HTTP status is 5xx
    would have vanished entirely. The fallback is plain-h2 fields, so the type
    is the plain-h2 one: the status rendered as a string.
    """
    import wardex_sdk.interceptors._seam as seam_mod

    def _boom(_body):
        raise RuntimeError("unframeable")

    monkeypatch.setattr(seam_mod, "parse_grpc_frames", _boom)
    txn = _txn(None)
    txn.status = 503

    name, status, error_type, extra, lims = _build_grpc_fields(txn, _BASE, ())

    assert name == "HTTP POST /echo.Echo/Say"
    assert status is StatusCode.ERROR
    assert error_type == "503"
    assert Limitation.FRAME_PARSE_FAILED in lims


def test_framing_failure_on_a_2xx_status_has_no_error_type(monkeypatch):
    import wardex_sdk.interceptors._seam as seam_mod

    def _boom(_body):
        raise RuntimeError("unframeable")

    monkeypatch.setattr(seam_mod, "parse_grpc_frames", _boom)

    _, status, error_type, _, lims = _build_grpc_fields(_txn(None), _BASE, ())

    assert status is StatusCode.OK
    assert error_type is None
    assert Limitation.FRAME_PARSE_FAILED in lims
