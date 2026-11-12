"""What the HTTP/2 tracker records about a stream whose latch entry the cap evicted.

An evicted entry that held a parent is a stream wardex latched under agent work and then lost. Its
answer, if it arrives, ships with the eviction marker on its own span. One whose answer never
arrives has no span to carry anything, so the tracker keeps a bounded record of such streams and
counts the ones that close (or fall off the record) unanswered: a model call as
`protocol.http2.evicted_call_unanswered`, any other request as `protocol.http2.evicted_unanswered`.
"""

from __future__ import annotations

import pytest
from hpack import Encoder

from wardex_sdk import _hub
from wardex_sdk._assembly import Limitation, counters
from wardex_sdk._interceptors._trackers import _Http2Tracker
from wardex_sdk._limits import LimitsConfig
from wardex_sdk._types import SpanContext, SpanId, TraceId

pytestmark = pytest.mark.usefixtures("fresh_counters")

_MODEL = ("POST", "/v1/chat/completions")
_OTHER = ("GET", "/health")


def _frame(ftype: int, flags: int, stream_id: int, payload: bytes) -> bytes:
    head = len(payload).to_bytes(3, "big") + bytes([ftype, flags])
    return head + stream_id.to_bytes(4, "big") + payload


class _Wire:
    """One connection's two HPACK encoders and a tracker capped at `max_streams`."""

    def __init__(self, max_streams: int) -> None:
        self.tracker = _Http2Tracker(LimitsConfig(max_streams=max_streams).to_native())
        self._client, self._server = Encoder(), Encoder()

    def open(self, stream_id: int, request: tuple[str, str], parent: SpanContext | None) -> None:
        """A whole request (END_HEADERS | END_STREAM) issued with `parent` as the ambient span."""
        method, path = request
        block = self._client.encode([(b":method", method.encode()), (b":path", path.encode())])
        with _hub.new_scope() as scope:
            scope.active_span_context = parent
            self.tracker.on_request_bytes(_frame(0x1, 0x4 | 0x1, stream_id, block))

    def answer(self, stream_id: int) -> list:
        block = self._server.encode([(b":status", b"200")])
        return self.tracker.on_response_bytes(_frame(0x1, 0x4 | 0x1, stream_id, block))


def _parent() -> SpanContext:
    return SpanContext(TraceId.generate(), SpanId.generate(), is_remote=False)


def _unanswered() -> tuple[int, int]:
    return (
        counters.get("protocol.http2.evicted_call_unanswered"),
        counters.get("protocol.http2.evicted_unanswered"),
    )


@pytest.mark.parametrize(
    ("request_", "counted"),
    [(_OTHER, (0, 1)), (_MODEL, (1, 0))],
    ids=["not-a-model-call", "model-call"],
)
def test_an_evicted_stream_closed_without_its_answer_is_counted(request_, counted):
    wire = _Wire(max_streams=1)
    parent = _parent()
    wire.open(1, request_, parent)
    wire.open(3, _OTHER, parent)
    assert 1 not in wire.tracker._latch, "precondition: the cap evicted stream 1's latch entry"

    wire.tracker.on_connection_close(Limitation.FRAME_PARSE_FAILED)

    assert _unanswered() == counted
    assert counters.get("protocol.http2.evicted_witnessed") == 0


def test_an_evicted_non_model_stream_whose_answer_arrives_is_not_counted_or_witnessed():
    wire = _Wire(max_streams=1)
    parent = _parent()
    wire.open(1, _OTHER, parent)
    wire.open(3, _OTHER, parent)

    (txn,) = wire.answer(1)

    assert txn.parent_evicted and txn.evicted_parent is None, "marked, and nothing to witness"
    wire.tracker.on_connection_close(Limitation.FRAME_PARSE_FAILED)
    assert _unanswered() == (0, 0)
    assert counters.get("protocol.http2.evicted_witnessed") == 0


def test_the_evicted_call_record_overflows_lowest_id_first_and_counts_each_loss_once():
    """cap + 1 model calls evicted under a parent: the record keeps `cap` of them, forgets the
    lowest id and counts it at that moment; close counts the `cap` it still held."""
    cap = 2
    wire = _Wire(max_streams=cap)
    parent = _parent()
    for sid in (1, 3, 5, 7, 9):  # the latch keeps 7 and 9, so 1, 3 and 5 are evicted
        wire.open(sid, _MODEL, parent)
    assert set(wire.tracker._latch) == {7, 9}, "precondition: three entries evicted"

    assert counters.get("protocol.http2.evicted_call_unanswered") == 1
    assert set(wire.tracker._evicted_calls) == {3, 5}

    wire.tracker.on_connection_close(Limitation.FRAME_PARSE_FAILED)

    assert counters.get("protocol.http2.evicted_call_unanswered") == 1 + cap
    assert wire.tracker._evicted_calls == {}


@pytest.mark.parametrize("request_", [_OTHER, _MODEL], ids=["not-a-model-call", "model-call"])
@pytest.mark.parametrize("by_client", [True, False], ids=["client-reset", "server-reset"])
def test_a_reset_stream_the_cap_then_evicts_is_not_counted_as_unanswered(request_, by_client):
    """A RST_STREAM ends the stream for both peers: no answer will come, and none was lost. A host
    that cancels its own request (or a server that refuses one) is not a loss wardex caused, so the
    stream leaves the latch uncounted, whether or not the cap later reaches for it."""
    wire = _Wire(max_streams=1)
    parent = _parent()
    wire.open(1, request_, parent)
    reset = _frame(0x3, 0, 1, (8).to_bytes(4, "big"))  # RST_STREAM, CANCEL
    if by_client:
        wire.tracker.on_request_bytes(reset)
    else:
        wire.tracker.on_response_bytes(reset)
    assert 1 not in wire.tracker._latch and 1 not in wire.tracker._model_paths
    wire.open(3, _OTHER, parent)

    wire.tracker.on_connection_close(Limitation.FRAME_PARSE_FAILED)

    assert _unanswered() == (0, 0)


def test_a_reset_stream_already_on_the_evicted_record_leaves_it_uncounted():
    """The cap evicted stream 1 first; the host cancels it afterwards. The record lets it go."""
    wire = _Wire(max_streams=1)
    parent = _parent()
    wire.open(1, _MODEL, parent)
    wire.open(3, _OTHER, parent)
    assert 1 in wire.tracker._evicted_calls, "precondition: the record holds stream 1"

    wire.tracker.on_request_bytes(_frame(0x3, 0, 1, (8).to_bytes(4, "big")))
    wire.tracker.on_connection_close(Limitation.FRAME_PARSE_FAILED)

    assert wire.tracker._evicted_calls == {}
    assert _unanswered() == (0, 0)
