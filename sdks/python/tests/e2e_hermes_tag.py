"""E2E: the Hermes smoke against one release tag of Hermes Agent, from nothing.

`e2e_hermes_smoke.py` needs a Hermes checkout with its environment built and a plugin
directory. This script makes both for one tag and runs the smoke in library and in CLI
mode, so a scheduled run can ask "does wardex still record a real Hermes of this release"
with one command. Not collected by pytest (the name misses `python_files`):

    python sdks/python/tests/e2e_hermes_tag.py --tag v0.23.3 --wheel <wardex wheel> \
        --work <empty dir> [--out result.json]

It clones the tag (depth 1), builds Hermes's environment with Hermes's own builder
(`python -m pm.build_env`, from its CONTRIBUTING), installs the wardex wheel into that
environment, writes the plugin directory from `wardex_sdk.hermes_agent`, and runs the
smoke. The smoke uses loopback ports only and no key. Exit code 0 means both modes held;
1 means the smoke failed; 3 means Hermes could not be fetched or built, which says
nothing about wardex.
"""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

REPO = "https://github.com/NousResearch/hermes-agent"
SMOKE = Path(__file__).with_name("e2e_hermes_smoke.py")
#: The interpreter Hermes's core dependencies ask for at the tags this was measured on.
PYTHON = "3.14"
_TAG = re.compile(r"v\d+(?:\.\d+){1,3}")
_PLUGIN = """\
from wardex_sdk.hermes_agent import HOOKS, MIDDLEWARE
lines = ["name: wardex", 'version: "0.1.0"', "description: wardex, for the smoke.",
         "provides_hooks:", *[f"  - {h}" for h in HOOKS],
         "provides_middleware:", *[f"  - {m}" for m in MIDDLEWARE]]
print("\\n".join(lines))
"""


def run(cmd: list, *, cwd: Path | None = None, timeout: int = 900) -> tuple[int, str]:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    try:
        done = subprocess.run(  # noqa: S603
            [str(part) for part in cmd],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    return done.returncode, (done.stdout + done.stderr)[-6000:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if not _TAG.fullmatch(args.tag):
        raise SystemExit(f"not a release tag: {args.tag!r}")
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    hermes, plugin = work / "hermes", work / "plugin" / "wardex"
    result: dict = {"ok": False, "failed": [], "checks": {}, "versions": {"hermes-agent": args.tag}}

    def finish(code: int) -> int:
        result["ok"] = code == 0
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True), "utf-8")
        print(json.dumps(result, indent=1, sort_keys=True))
        return code

    steps = (
        (["git", "clone", "--quiet", "--depth", "1", "--branch", args.tag, REPO, hermes], None),
        (["uv", "python", "install", PYTHON], None),
    )
    for cmd, cwd in steps:
        code, said = run(cmd, cwd=cwd)
        if code != 0:
            result["failed"].append(f"could not prepare Hermes: {cmd[0]} {cmd[1]}: {said[-300:]}")
            return finish(3)
    code, found = run(["uv", "python", "find", PYTHON])
    python = found.strip().splitlines()[-1] if code == 0 and found.strip() else "python3"
    build = [python, "-m", "pm.build_env", "--source", ".", "--out", ".venv"]
    code, said = run([*build, "--group", "dev", "--group", "test"], cwd=hermes, timeout=1200)
    if code != 0:
        result["failed"].append(f"Hermes's own environment builder failed: {said[-300:]}")
        return finish(3)
    theirs = hermes / ".venv" / "bin" / "python"
    code, said = run(["uv", "pip", "install", "--python", theirs, "--quiet", args.wheel])
    if code != 0:
        result["failed"].append(f"wardex does not install into Hermes's environment: {said[-300:]}")
        return finish(1)
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "__init__.py").write_text(
        "from wardex_sdk.hermes_agent import register\n", encoding="utf-8"
    )
    code, said = run([theirs, "-I", "-c", _PLUGIN])
    if code != 0:
        result["failed"].append(f"the plugin's manifest could not be written: {said[-300:]}")
        return finish(1)
    (plugin / "plugin.yaml").write_text(said.strip() + "\n", encoding="utf-8")

    for mode in ("library", "cli"):
        started = time.monotonic()
        smoke = [theirs, SMOKE, "--hermes", hermes, "--plugin", plugin, "--scenario", "all"]
        code, said = run([*smoke, "--mode", mode], timeout=840)
        held = len(re.findall(r"^\s*ok\b", said, flags=re.M))
        result["checks"][mode] = {"exit": code, "seconds": round(time.monotonic() - started)}
        print(said[-3000:])
        if code != 0:
            result["failed"].append(
                f"the smoke failed in {mode} mode (exit {code}, {held} ok lines)"
            )
    return finish(0 if not result["failed"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
