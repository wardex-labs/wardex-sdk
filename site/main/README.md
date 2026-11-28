# Compatibility of the default branch

What the daily run found for wardex built from `main` (version string `0.7.0b1`, commit `5ee7a87720`).

Generated on 28 Nov 2026 from the run records on this branch; nothing here is written by hand. A row marked *watching* is run and reported and never turns the run red. A pass older than 3 days is shown as when it last passed, not as passing. How to reproduce a row: [AGENTS.md](https://github.com/wardex-labs/wardex-sdk/blob/main/AGENTS.md).

## claude-agent-sdk

- Tested range, as this build states it: `claude-agent-sdk` 0.2.x
- Evidence: the framework's Python package runs against recorded output of `claude CLI 2.1.286`, replayed. A pass does not show that a newer build of that program behaves the same.
- Newest release on PyPI when last planned: `0.2.189`

| Check | What | Version | Python | Result | Last run | Last pass |
|---|---|---|---|---|---|---|
| framework release | newest tested release | `claude-agent-sdk 0.2.189` | `3.14` | pass | 28 Nov 2026 | 28 Nov 2026 |
| framework release | newest tested release, oldest Python | `claude-agent-sdk 0.2.189` | `3.10` | pass | 28 Nov 2026 | 28 Nov 2026 |
| framework release | oldest tested release | `claude-agent-sdk 0.2.82` | `3.10` | pass | 28 Nov 2026 | 28 Nov 2026 |

## langgraph

- Tested range, as this build states it: `langgraph` 1.2.x; `langgraph-prebuilt` 1.1.x
- Evidence: the framework itself runs in the test.
- Newest release on PyPI when last planned: `1.2.24`

| Check | What | Version | Python | Result | Last run | Last pass |
|---|---|---|---|---|---|---|
| framework release | newest tested release | `langgraph 1.2.24, langgraph-prebuilt 1.1.3` | `3.14` | pass | 28 Nov 2026 | 28 Nov 2026 |
| framework release | newest tested release, oldest Python | `langgraph 1.2.24` | `3.10` | pass | 28 Nov 2026 | 28 Nov 2026 |
| framework release | oldest tested release | `langgraph 1.2.0, langgraph-prebuilt 1.1.0` | `3.10` | pass | 28 Nov 2026 | 28 Nov 2026 |

## openai-agents

- Tested range, as this build states it: `openai-agents` 0.22.x, 0.23.x
- Evidence: the framework itself runs in the test.
- Newest release on PyPI when last planned: `0.26.0`

| Check | What | Version | Python | Result | Last run | Last pass |
|---|---|---|---|---|---|---|
| framework release | newer than the tested range | `openai-agents 0.24.0` | `3.14` | tests fail | 28 Nov 2026 | - |
| framework release | newer than the tested range | `openai-agents 0.25.0` | `3.14` | tests fail | 28 Nov 2026 | - |
| framework release | newer than the tested range | `openai-agents 0.26.0` | `3.14` | tests fail | 28 Nov 2026 | - |
| framework release | newest tested release | `openai-agents 0.23.1` | `3.14` | pass | 28 Nov 2026 | 28 Nov 2026 |
| framework release | newest tested release, oldest Python | `openai-agents 0.23.1` | `3.10` | pass | 28 Nov 2026 | 28 Nov 2026 |
| framework release | oldest tested release | `openai-agents 0.22.0` | `3.10` | pass | 28 Nov 2026 | 28 Nov 2026 |
| framework release | newest patch of a tested line | `openai-agents 0.22.3` | `3.14` | pass | 28 Nov 2026 | 28 Nov 2026 |

Failing tests on `openai-agents 0.24.0`:

- `test_openai_agents_adapter.py::test_a_run_resumed_after_a_tool_approval_ships_under_its_own_root`
- `test_openai_agents_silent_gaps.py::test_a_disabled_nested_run_inside_a_traced_one_loses_only_its_own_joins`
- `test_openai_agents_silent_gaps.py::test_a_disabling_run_config_inside_an_open_trace_says_exactly_what_is_lost`

Failing tests on `openai-agents 0.25.0`:

- `test_openai_agents_adapter.py::test_a_run_resumed_after_a_tool_approval_ships_under_its_own_root`
- `test_openai_agents_adapter.py::test_a_streamed_input_guardrail_tripwire_reaches_the_consumer`
- `test_openai_agents_silent_gaps.py::test_a_disabled_nested_run_inside_a_traced_one_loses_only_its_own_joins`
- `test_openai_agents_silent_gaps.py::test_a_disabling_run_config_inside_an_open_trace_says_exactly_what_is_lost`

Failing tests on `openai-agents 0.26.0`:

- `test_openai_agents_adapter.py::test_a_run_resumed_after_a_tool_approval_ships_under_its_own_root`
- `test_openai_agents_adapter.py::test_a_streamed_input_guardrail_tripwire_reaches_the_consumer`
- `test_openai_agents_silent_gaps.py::test_a_disabled_nested_run_inside_a_traced_one_loses_only_its_own_joins`
- `test_openai_agents_silent_gaps.py::test_a_disabling_run_config_inside_an_open_trace_says_exactly_what_is_lost`

## pydantic-ai

- Tested range, as this build states it: `pydantic-ai-slim` 2.x from 2.54
- Evidence: the framework itself runs in the test.
- Newest release on PyPI when last planned: `2.77.0`

| Check | What | Version | Python | Result | Last run | Last pass |
|---|---|---|---|---|---|---|
| framework release | newest tested release | `pydantic-ai-slim 2.77.0` | `3.14` | pass | 28 Nov 2026 | 28 Nov 2026 |
| framework release | newest tested release, oldest Python | `pydantic-ai-slim 2.54.0` | `3.10` | pass | 28 Nov 2026 | 28 Nov 2026 |
| framework release | oldest tested release | `pydantic-ai-slim 2.54.0` | `3.10` | pass | 28 Nov 2026 | 28 Nov 2026 |

## What the README says, and what proves it

Each sentence of the README that says wardex was tested, verified or measured against something, and the scheduled check behind it. Where there is none, that is said here instead of left to be assumed.

| The README says | Proven on a schedule by |
|---|---|
| Each adapter is tested against the framework releases in the table above. | framework release points, daily |
| Langfuse and Phoenix are tested destinations | nothing yet: checked by hand against a live receiver; no scheduled run starts one |
| Hermes Agent (through a Hermes plugin; measured on v0.23.3) | nothing yet: replayed on every pull request from payloads recorded on that release; no scheduled run installs Hermes |
| Verified against codex-cli 0.160.0. | nothing yet: replayed on every pull request from output recorded on that release; no scheduled run installs the CLI |
| Prebuilt wheels cover Linux with glibc (x86_64, aarch64), macOS (x86_64, arm64) and Windows (x86_64) | nothing yet: each wheel is built and import-checked at release; the tests run on Linux x86_64 only |
| Measured on one arm64 macOS machine at a load average of about 2, not a guarantee. | nothing yet: one measurement on one machine, stated as such; no scheduled run repeats it |
