# Hermes Agent replay fixtures

Each file is one Hermes Agent turn, recorded from a real Hermes process and trimmed to what a
plugin can observe: which hooks and middleware chains fired, in what order, on which thread, with
which context and which payload fields. `tests/_hermes_replay.py` plays a file back through a fake
plugin host (`tests/_hermes_fake.py`), so the adapter's tests run against the thread and context
structure of a real Hermes without installing it.

- Hermes version: `v0.23.3` (commit `caafe53`), with a local OpenAI-compatible fake model server.
- Format: JSON Lines, version 1. The header line carries `"format": "wardex.hermes.fixture"` and
  `"version": 1`, and the loader refuses any other version. The fields are described in the
  docstring of `tests/_hermes_replay.py`.
- Mode: every file runs Hermes as a library (`run_conversation` called from a thread named
  `lab-turn`), except `one_tool-cli`, which runs the `hermes chat` command line.

| File | What the turn does |
| --- | --- |
| `single_turn` | one model call, no tools |
| `one_tool`, `one_tool-cli` | one tool call, then an answer |
| `parallel_tools` | three `read_file` calls in one concurrent batch |
| `inline_seq` | two tools in two model rounds, one after the other |
| `mixed_batch` | four tools in one message, run one at a time because one is not parallel-safe |
| `mcp_seq`, `parallel_mcp` | a stdio MCP tool alone, and with two `read_file` calls in one concurrent batch |
| `retry` | a 503 attempt and a successful retry inside one model call |
| `two_subagents_joined` | `delegate_task` with two children, waited for |
| `two_subagents` | `delegate_task` with two background children |
| `auxiliary_title` | the title generator's auxiliary call on its own thread |
| `interrupted` | a stream cut by an interrupt |
| `failed_turn` | retries spent on 500s: the turn fails |

## What was removed

The recording plugin saw full payloads. The committed files keep the ids, the schema versions and
the fields the adapter reads, and drop or replace everything else:

- Message content, prompts, tool arguments, tool results and stream deltas. Where the field must
  exist, it holds `"<synthetic>"`, `"<synthetic result>"` or `"<synthetic error>"`.
- The model request (`{"model": ..., "messages": []}` remains), request headers, the provider's
  `base_url`, and the full `assistant_message` (only each tool call's `id` and `name` remain, with
  `"arguments": "{}"`).
- Usage blocks keep their token counts only.
- The recording harness's own events (runner, plugin load, process exit) and the sockets opened
  before the turn began or on the main thread.
- Child goals and summaries, and gateway sender ids.

Session ids, task ids and turn ids are the values Hermes generated in the recorded run. They hold a
timestamp and random hex, nothing about the machine or the person who ran it.

## What is derived

A line or header entry lists the fields that were not recorded but derived in its `derived` array.
Everything not listed was recorded.

- `on` of a hook that does not run on the caller's thread: the thread that fired it. Hermes
  delivers bounded hooks on a fresh `hermes-hook-*` thread and stream hooks on a
  `plugin-stream-hook:*` thread, so the recording shows only where the callback ran. The firing
  thread is taken from the event's owner: the tool worker for tool hooks (by `tool_call_id`), the
  title thread for auxiliary hooks, the owner of the innermost open frame for stream hooks, and the
  session's loop thread for the rest.
- `copier` and `copy_before` of every header thread: whose context the thread copied, and when.
  They follow the measured thread model (a model worker copies its frame's thread inside the
  frame, a tool worker copies the loop thread at submit, a child loop copies the tool worker right
  after its own `subagent_start`, the title thread copies the loop thread before the first
  `llm_request`). Each thread's recorded `context`, the probe its first event saw, is what the
  replay test checks these against.
- `raised` of an `llm_execution` exit that raised: the error's module and type. The recording
  kept only the exception's text; the type follows from it (Hermes's interrupt text is an
  `InterruptedError`, the provider's 500 an `openai.InternalServerError`, as `api_request_error`
  reports).
- The HTTP attempts of `one_tool-cli`: the command line ran in another process, so its sockets were
  not recorded. One attempt is placed inside each model frame, on a model worker thread, with
  `thread`, `on`, `probe` and `position` derived.

The `stream`, `status` and `usage` of each HTTP attempt come from the fake model server's own log,
matched to the recorded requests in order, and checked against the usage Hermes reported in
`post_api_request`.

## Re-recording

`e2e_hermes_smoke.py` runs a real Hermes against the fake model server and re-records these files.
Re-record when Hermes changes a hook, a payload field or a thread, and keep the format version
unless a field changes meaning.
