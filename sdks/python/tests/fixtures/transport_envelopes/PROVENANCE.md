# Transport envelope body

A request body exactly as `WardexTransport` would POST it — a
`wardex.v1.Envelope`, protobuf under zstd — produced by driving loopback
traffic through the installed byte seam and encoding what it captured with
the real encoder (masking off). It carries a streaming chat call (SSE), a
non-streaming chat call (JSON) and a WebSocket session. Every value in it is
synthetic: a loopback server, canned responses.

- Produced: 2026-10-01
- wardex-sdk: 0.6.0b1, Python 3.14.3
- Source: `sdks/python/tests/test_observed_transport.py`, which also holds the
  contract the body is checked against on every run
- Regenerate: `WARDEX_REGEN_ENVELOPES=1 uv run pytest sdks/python/tests/test_observed_transport.py`
