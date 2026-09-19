# Adapter envelope bodies

Request bodies exactly as `WardexTransport` would POST them — a
`wardex.v1.Envelope`, protobuf under zstd — each produced by driving the real
adapter and encoding what it emitted with the real encoder. A receiver's test
suite reads these in place of hand-built envelopes. Every value in them is
synthetic: a scripted model, a loopback server, canned tool output.

- Produced: 2026-09-19
- wardex-sdk: 0.6.0b1, Python 3.14.3
- Source: `sdks/python/tests/test_adapter_envelopes.py`, which also holds the
  contract each body is checked against on every run
- Regenerate: `WARDEX_REGEN_ENVELOPES=1 uv run pytest sdks/python/tests/test_adapter_envelopes.py`

| File | Conversation id it states | Framework |
|---|---|---|
| `openai_agents.envelope.zst` | `conv-openai-agents` | openai-agents 0.22.3 |
| `langgraph.envelope.zst` | `conv-langgraph` | langgraph 1.2.11 |
| `agent_sdk.envelope.zst` | `s-1` | none (scripted stream) |
