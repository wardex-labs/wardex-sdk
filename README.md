# wardex-sdk

[![PyPI](https://img.shields.io/pypi/v/wardex-sdk)](https://pypi.org/project/wardex-sdk/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/wardex-labs/wardex-sdk/blob/main/LICENSE)

Open-source observability SDK for AI agents — zero-instrumentation capture,
OpenTelemetry-native.

> ⚠️ **Beta (0.1.0b1).** No PII masking yet: captured prompts and responses
> are sent to your backend in cleartext. Not suitable for production or
> sensitive data. Use in development/staging, or wait for the PII release.

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

**Not yet (see Roadmap)**
- PII masking (prompts sent in cleartext today)
- Background batching / periodic & at-exit flush (manual `flush()`/`close()` only)
- Framework adapters (LangGraph, Anthropic/OpenAI Agent SDKs)
- Distributed context propagation (W3C traceparent)
- Node/TS and Java SDKs

## Roadmap

1. PII masking (pre-send safety)
2. Batching & lifecycle (background worker, at-exit/periodic flush, concurrency)
3. Framework adapters
4. Distributed propagation (W3C)
5. Node/TS and Java SDKs

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

"Wardex" is a trademark of Wardex Labs.
