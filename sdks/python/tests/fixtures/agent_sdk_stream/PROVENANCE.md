# Agent SDK stream recordings

Inbound stream-json lines exactly as a real `claude` CLI wrote them to the
Agent SDK's transport, one JSON object per line. Each file is one session,
recorded with a prompt chosen to make the CLI split ONE model response across
several `assistant` lines:

| File | Prompt shape | Response 1 on the wire |
|---|---|---|
| `text_then_tool.jsonl` | a sentence, then one Bash call | 2 lines (text, tool_use), one `message.id` |
| `parallel_tools.jsonl` | three Bash calls in one response | 3 lines (one tool_use each), one `message.id` |

What the recordings show, and what the replay tests in
`sdks/python/tests/test_agent_sdk_assembler.py` hold the assembler to:

- Every line of one response carries the same `message.id` and an identical
  copy of the same `usage` object: the copies are repeated, not split across
  lines and not running totals. Counting the response's usage once means
  taking one copy.
- `output_tokens` on those lines is the count the API reported when the
  response STARTED (8 and 16 here), not its final count. The session's real
  output total is only in the `result` line's `usage`.
- Summed once per response, the `assistant` lines' input and cache tokens
  equal the `result` line's own totals.

Provenance:

- Recorded: 2026-10-03
- claude CLI 2.1.286 (the one bundled with claude-agent-sdk 0.2.163), model
  `claude-opus-5-5`, Python 3.14.3, macOS
- Recorder: `sdks/python/tests/e2e_agent_sdk_smoke.py`
- Regenerate (needs a working `claude` login; each run is a real, billed
  model call):

  ```bash
  WARDEX_SMOKE_SCENARIO=text_then_tool WARDEX_RECORD=out.jsonl \
      uv run python sdks/python/tests/e2e_agent_sdk_smoke.py
  ```

  then scrub as below. Ids, timestamps and token counts differ per run, so a
  regenerated file needs the replay tests' expectations re-read from it.

Scrubbed before commit; every other byte of every kept line is as recorded
and the line order is unchanged:

- `control_request` / `control_response` lines removed. They are the SDK's
  hook and initialization traffic, carry absolute paths of the recording
  machine, and the assembler never reads them from the stream.
- `rate_limit_event` lines removed: they describe the recording account's
  quota.
- `system/init` cut down to the fields the parser reads plus the version
  facts above; its `cwd` replaced by `/home/user/project`. The tool, MCP
  server, command, agent, skill and plugin lists were dropped: they describe
  the recording machine, not the CLI's protocol.
- Every `session_id` replaced by a neutral id per file
  (`s-text-then-tool`, `s-parallel-tools`).
