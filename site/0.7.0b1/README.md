# Compatibility of wardex 0.7.0b1

What the daily run found for `wardex-sdk==0.7.0b1` as PyPI serves it, with the tests of its own tag `python-v0.7.0b1`.

Generated on 28 Nov 2026 from the run records on this branch; nothing here is written by hand. A row marked *watching* is run and reported and never turns the run red. A pass older than 3 days is shown as when it last passed, not as passing. How to reproduce a row: [AGENTS.md](https://github.com/wardex-labs/wardex-sdk/blob/main/AGENTS.md).

## claude-agent-sdk

- Tested range, as this build states it: `claude-agent-sdk` 0.2.x
- Evidence: the framework's Python package runs against recorded output of `claude CLI 2.1.286`, replayed. A pass does not show that a newer build of that program behaves the same.
- Newest release on PyPI when last planned: `0.2.189`

| Check | What | Version | Python | Result | Last run | Last pass |
|---|---|---|---|---|---|---|
| published wardex release (watching) | newest tested release | `claude-agent-sdk 0.2.189` | `3.14` | pass | 28 Nov 2026 | 28 Nov 2026 |

## langgraph

- Tested range, as this build states it: `langgraph` 1.2.x; `langgraph-prebuilt` 1.1.x
- Evidence: the framework itself runs in the test.
- Newest release on PyPI when last planned: `1.2.24`

| Check | What | Version | Python | Result | Last run | Last pass |
|---|---|---|---|---|---|---|
| published wardex release (watching) | newest tested release | `langgraph 1.2.24, langgraph-prebuilt 1.1.3` | `3.14` | pass | 28 Nov 2026 | 28 Nov 2026 |

## openai-agents

- Tested range, as this build states it: `openai-agents` 0.22.x, 0.23.x
- Evidence: the framework itself runs in the test.
- Newest release on PyPI when last planned: `0.26.0`

| Check | What | Version | Python | Result | Last run | Last pass |
|---|---|---|---|---|---|---|
| published wardex release (watching) | newer than the tested range | `openai-agents 0.26.0` | `3.14` | tests fail | 28 Nov 2026 | - |
| published wardex release (watching) | newest tested release | `openai-agents 0.23.1` | `3.14` | pass | 28 Nov 2026 | 28 Nov 2026 |

Failing tests on `openai-agents 0.26.0`:

- `test_openai_agents_adapter.py::test_a_run_resumed_after_a_tool_approval_ships_under_its_own_root`
- `test_openai_agents_adapter.py::test_a_streamed_input_guardrail_tripwire_reaches_the_consumer`
- `test_openai_agents_silent_gaps.py::test_a_disabled_nested_run_inside_a_traced_one_loses_only_its_own_joins`
- `test_openai_agents_silent_gaps.py::test_a_disabling_run_config_inside_an_open_trace_says_exactly_what_is_lost`

## pydantic-ai

- Tested range, as this build states it: `pydantic-ai-slim` 2.x
- Evidence: the framework itself runs in the test.
- Newest release on PyPI when last planned: `2.77.0`

| Check | What | Version | Python | Result | Last run | Last pass |
|---|---|---|---|---|---|---|
| published wardex release (watching) | newest tested release | `pydantic-ai-slim 2.77.0` | `3.14` | pass | 28 Nov 2026 | 28 Nov 2026 |
