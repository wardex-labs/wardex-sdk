"""gRPC semantics — framing and trailer status mapped onto span fields.

Pure functions over a transaction the byte seam already parsed. The transaction
is typed `Any` rather than `interceptors._trackers._Txn`: naming that record
here would be an import from a sibling observer package, which is the one edge
this package's layering forbids (see `semantics/__init__.py`). What is actually
required of it is structural — `method`, `path`, `status`, `request_body`,
`response_body`, `grpc_status`, `grpc_message` — and the call site is the byte
seam, which holds the real type.
"""

from __future__ import annotations

from typing import Any

from .._enums import StatusCode
from ..assembly import Limitation, TransportLabel
from ..assembly._vocab import transport_name
from ..protocol import grpc_status_name, parse_grpc_frames


def build_grpc_fields(
    txn: Any,
    extra: tuple[tuple[str, str | int | float | bool], ...],
    limitations: tuple[Limitation, ...],
) -> tuple[
    str,
    StatusCode,
    str | None,
    tuple[tuple[str, str | int | float | bool], ...],
    tuple[Limitation, ...],
]:
    """Assemble gRPC span fields → (name, status_code, error_type, extra, limitations).

    Pure, and deliberately still returning a tuple rather than mutating a draft:
    it is the one branch of the seam with enough protocol logic to be worth
    testing without a socket, and it is where the census scanner's R6 rule finds
    the gRPC markers.

    On a framing-parse failure the span falls back to plain h2 (HTTP) fields plus
    `Limitation.FRAME_PARSE_FAILED`. The caller keys its own label off exactly
    that marker, so the name and the marker cannot disagree about whether this
    was gRPC.

    Markers are `Limitation` members, not free strings, and two of these values
    changed name on the way in (§6.5.1): `grpc_parse_failed` became
    `FRAME_PARSE_FAILED` because a WebSocket framing failure is the same fact,
    and `grpc_compressed` became `PAYLOAD_COMPRESSED` because
    `TransportAttributes.protocol` already carries which protocol it was and
    encoding that into the marker duplicates a field.
    """
    try:
        req = parse_grpc_frames(txn.request_body)
        resp = parse_grpc_frames(txn.response_body)
    except Exception:
        http_status = StatusCode.OK if 200 <= txn.status < 400 else StatusCode.ERROR
        return (
            transport_name(TransportLabel.HTTP, f"{txn.method} {txn.path}"),
            http_status,
            # Plain-h2 fallback, so the plain-h2 error type: the HTTP status as
            # a string. Returning `None` here alongside an ERROR status is what
            # `finish()` deletes the span for.
            str(txn.status) if http_status is StatusCode.ERROR else None,
            extra,
            limitations + (Limitation.FRAME_PARSE_FAILED,),
        )

    name = transport_name(TransportLabel.GRPC, txn.path)
    code = txn.grpc_status
    status_code = StatusCode.ERROR if code not in (0, None) else StatusCode.OK
    error_type = grpc_status_name(code) if status_code is StatusCode.ERROR else None

    # "/pkg.Svc/Method" → service="pkg.Svc", method="Method"
    service, method = "", ""
    trimmed = txn.path.lstrip("/")
    if "/" in trimmed:
        service, method = trimmed.rsplit("/", 1)
    else:
        method = trimmed

    extra = extra + (
        ("rpc.system", "grpc"),
        ("rpc.service", service),
        ("rpc.method", method),
        ("rpc.grpc.request.message_count", len(req.messages)),
        ("rpc.grpc.response.message_count", len(resp.messages)),
    )
    if code is not None:
        extra = extra + (("rpc.grpc.status_code", code),)
    # If grpc-message is present, include it on the span — useful for diagnosing errors
    # (e.g. "NOT_FOUND: collection x missing")
    if txn.grpc_message:
        extra = extra + (("rpc.grpc.status_message", txn.grpc_message),)

    if code is None:
        limitations = limitations + (Limitation.GRPC_STATUS_UNAVAILABLE,)
    if any(m.compressed for m in req.messages) or any(m.compressed for m in resp.messages):
        limitations = limitations + (Limitation.PAYLOAD_COMPRESSED,)
    if req.truncated or resp.truncated:
        limitations = limitations + (Limitation.GRPC_MESSAGE_TRUNCATED,)

    return name, status_code, error_type, extra, limitations
