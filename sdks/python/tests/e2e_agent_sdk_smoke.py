"""Manual E2E smoke: real `claude` CLI + wardex adapter + ConsoleTransport.

Run locally (requires claude CLI + auth):
    uv run python sdks/python/tests/e2e_agent_sdk_smoke.py
Also serves as the fixture recorder: set WARDEX_RECORD=/path/to/out.jsonl to
dump raw inbound stream-json lines for parser golden tests.

WARDEX_SMOKE_SCENARIO picks a prompt that makes the CLI split ONE model
response across several `assistant` lines, which is what the replay fixtures
under `fixtures/agent_sdk_stream/` hold:
    text_then_tool  — a sentence, then one Bash call
    parallel_tools  — three Bash calls in one response
A scenario runs with only the Bash tool, no settings sources, in a fresh
temporary directory.
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile

SCENARIOS = {
    "text_then_tool": (
        "First write exactly one short sentence saying you will run a command. "
        "Then call the Bash tool once to run: echo hello. "
        "After the result, reply with the single word: done."
    ),
    "parallel_tools": (
        "In ONE single response, call the Bash tool three times in parallel, "
        "with these three commands: echo one ; echo two ; echo three. "
        "Do not write any text before the tool calls. "
        "After all three results, reply with the single word: done."
    ),
}


async def main() -> int:
    if shutil.which("claude") is None:
        print("claude CLI not found — skipping")
        return 0

    import claude_agent_sdk

    import wardex_sdk
    from wardex_sdk import ConsoleTransport

    record_path = os.environ.get("WARDEX_RECORD")
    scenario = os.environ.get("WARDEX_SMOKE_SCENARIO")
    if scenario is not None and scenario not in SCENARIOS:
        print(f"unknown WARDEX_SMOKE_SCENARIO {scenario!r}; one of {sorted(SCENARIOS)}")
        return 2
    wardex_sdk.init(transport=ConsoleTransport())

    rec = None
    try:
        if record_path:
            from claude_agent_sdk._internal.transport import subprocess_cli

            cls = subprocess_cli.SubprocessCLITransport
            orig_read = cls.read_messages
            rec = open(record_path, "a", encoding="utf-8")

            def read_messages(self):
                inner = orig_read(self)

                async def gen():
                    async for msg in inner:
                        try:
                            rec.write(json.dumps(msg) + "\n")
                        except Exception:
                            pass
                        yield msg

                return gen()

            cls.read_messages = read_messages

        prompt = "What is 2 + 2? Answer with one number."
        options = None
        if scenario is not None:
            prompt = SCENARIOS[scenario]
            options = claude_agent_sdk.ClaudeAgentOptions(
                tools=["Bash"],
                allowed_tools=["Bash(echo:*)", "Bash"],
                cwd=tempfile.mkdtemp(prefix="wardex-smoke-"),
                max_turns=4,
                setting_sources=[],
            )
        async for message in claude_agent_sdk.query(prompt=prompt, options=options):
            print(type(message).__name__)

        print("smoke OK — spans printed above by ConsoleTransport")
        return 0
    finally:
        if rec is not None:
            try:
                rec.close()
            except Exception:
                pass
        wardex_sdk.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
