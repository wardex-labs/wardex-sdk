# wardex-sdk

[![PyPI](https://img.shields.io/pypi/v/wardex-sdk)](https://pypi.org/project/wardex-sdk/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/wardex-labs/wardex-sdk/blob/main/LICENSE)

Open-source observability SDK for AI agents — zero-instrumentation capture,
OpenTelemetry-native.

> ⚠️ **Beta.** PII masking is on by default (see below), but the
> SDK is still early: review the caveats below before sending sensitive data
> through it.

## Install

```bash
pip install wardex-sdk
```

## Quickstart

```python
import wardex_sdk as wardex
from wardex_sdk import OtlpHttpTransport

wardex.init(
    transport=OtlpHttpTransport(endpoint="https://<your-collector>/v1/traces"),
    intercept=True,  # zero-instrumentation capture of LLM calls
)

# your app code — OpenAI/Anthropic calls are captured automatically

wardex.close()   # optional — spans auto-flush every 5s, on buffer threshold, and at exit
```

## Status

**Works today**
- Zero-instrumentation capture of LLM HTTP calls (OpenAI, Anthropic) over
  `https`, cleartext `http`, and h2c
- `gen_ai` semantics: model, tokens, parameters, finish reasons, input/output messages
- Transport metrics (TCP/TLS timing, TTFT), gRPC (grpclib), WebSocket (`wss`), MCP stdio
- Export to any OpenTelemetry backend via `OtlpHttpTransport`
- Manual span decorators: `@workflow` / `@agent` / `@task` / `@tool` / `@span`
- PII masking on by default: emails, phone numbers, credit cards (Luhn-verified),
  US SSNs, IP addresses, bank routing numbers, IBANs, and API-key/token secrets
  are masked before anything leaves the process (`pii_mode=PIIMode.OFF` to disable,
  `pii_disabled_categories={PIICategory.IP_ADDRESS}` for per-category opt-out)
- Background batching: automatic flush every 5s / on buffer threshold /
  at exit and on SIGINT/SIGTERM (chained; opt out with `flush_on_signals=False`)

**Not yet (see Roadmap)**
- Framework adapters (LangGraph, Anthropic/OpenAI Agent SDKs)
- Distributed context propagation (W3C traceparent)
- Node/TS and Java SDKs
- After `os.fork()` the worker respawns lazily in the child on first capture;
  spans buffered before the fork may be sent by both processes (duplicates,
  never loss). Under uWSGI enable threads (`--enable-threads`).

## Roadmap

1. ~~PII masking (pre-send safety)~~ — shipped
2. Batching & lifecycle (background worker, at-exit/periodic flush, concurrency)
3. Framework adapters
4. Distributed propagation (W3C)
5. Node/TS and Java SDKs

> PII masking caveats: `before_send` sees pre-masking data (masking runs inside
> the encoder), the Console transport prints raw (local debugging only), and
> non-UTF-8 binary payloads pass through unmasked.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

"Wardex" is a trademark of Wardex Labs.
