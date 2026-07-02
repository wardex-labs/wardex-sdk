# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and versions follow PEP 440.

## [Unreleased]

## [0.1.0b1] - 2026-07-02

### Added
- Initial public beta.
- Zero-instrumentation interception of LLM HTTP calls (OpenAI, Anthropic) over
  `https`, cleartext `http`, and h2c; `gen_ai` semantics; transport metrics;
  gRPC (grpclib), WebSocket, and MCP stdio capture.
- OpenTelemetry export via `OtlpHttpTransport`.
- Manual span decorators (`@workflow`/`@agent`/`@task`/`@tool`/`@span`).

### Known limitations
- No PII masking (prompts/responses sent in cleartext).
- Manual flush only; no background batching.
- No framework adapters yet.
