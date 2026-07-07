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

wardex.flush()   # manual flush (background batching is on the roadmap)
wardex.close()
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

**Not yet (see Roadmap)**
- Background batching / periodic & at-exit flush (manual `flush()`/`close()` only)
- Framework adapters (LangGraph, Anthropic/OpenAI Agent SDKs)
- Distributed context propagation (W3C traceparent)
- Node/TS and Java SDKs

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
