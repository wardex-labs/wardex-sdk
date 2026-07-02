from wardex_sdk._types import EnvelopeHeader, InternalEnvelope, SdkInfo
from wardex_sdk.transport._console import ConsoleTransport
from wardex_sdk.transport._noop import NoOpTransport


def _envelope():
    sdk = SdkInfo(
        name="wardex.python", version="0.1.0", python_version="3.10", os="darwin", arch="arm64"
    )
    return InternalEnvelope(header=EnvelopeHeader(event_id="e1", api_key="", sdk=sdk, sent_at_ns=0))


def test_noop_captures_nothing_and_closes():
    t = NoOpTransport()
    t.export(_envelope())
    t.flush()
    t.close()


def test_console_writes_to_stream():
    import io

    buf = io.StringIO()
    t = ConsoleTransport(stream=buf)
    t.export(_envelope())
    t.flush()
    out = buf.getvalue()
    assert "InternalEnvelope" in out or "event_id" in out or "e1" in out
