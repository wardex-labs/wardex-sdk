"""Manual E2E smoke: real `claude` CLI + wardex adapter + ConsoleTransport.

Run locally (requires claude CLI + auth):
    uv run python sdks/python/tests/e2e_agent_sdk_smoke.py
Also serves as the fixture recorder: set WARDEX_RECORD=/path/to/out.jsonl to
dump raw inbound stream-json lines for parser golden tests.
"""

import asyncio
import json
import os
import shutil
import sys


async def main() -> int:
    if shutil.which("claude") is None:
        print("claude CLI not found — skipping")
        return 0

    import claude_agent_sdk

    import wardex_sdk
    from wardex_sdk import ConsoleTransport

    record_path = os.environ.get("WARDEX_RECORD")
    wardex_sdk.init(transport=ConsoleTransport())

    if record_path:
        from claude_agent_sdk._internal.transport import subprocess_cli

        cls = subprocess_cli.SubprocessCLITransport
        orig_read = cls.read_messages
        rec = open(record_path, "a", encoding="utf-8")

        def read_messages(self):
            inner = orig_read(self)

            async def gen():
                async for msg in inner:
                    rec.write(json.dumps(msg) + "\n")
                    yield msg

            return gen()

        cls.read_messages = read_messages

    async for message in claude_agent_sdk.query(prompt="What is 2 + 2? Answer with one number."):
        print(type(message).__name__)

    wardex_sdk.close()
    print("smoke OK — spans printed above by ConsoleTransport")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
