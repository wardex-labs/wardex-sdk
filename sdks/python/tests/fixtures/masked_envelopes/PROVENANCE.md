# Masked envelope bodies

Request bodies exactly as `WardexTransport` would POST them after masking — a
`wardex.v1.Envelope`, protobuf under zstd — produced by capturing a real
loopback request and encoding it with the real encoder and the policy `init()`
installed. A receiver's test suite reads these to check that it stores what
the masker reported (`CaptureIntegrity.redaction_count`, `redaction_rules`,
`redaction_names`). Every value in them is synthetic.

- Source: `test_a_masked_envelope_body_for_receivers` in
  `sdks/python/tests/test_secret_names.py`, which also holds the contract the
  body is checked against on every run
- Regenerate: `WARDEX_REGEN_ENVELOPES=1 uv run pytest sdks/python/tests/test_secret_names.py`

| File | What it carries |
|---|---|
| `masked_http.envelope.zst` | `HTTP POST /search?q=seoul&appid=…` with `{"api_key": …}`: two values replaced, rules `secret_exact_name` and `secret_last_word`, names `appid` and `api_key` |
