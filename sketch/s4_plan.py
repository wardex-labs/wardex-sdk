"""Sketch: transport-library points through the real executor. Not a rule of the device yet."""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compat import manifest, plan  # noqa: E402

LIBS = {"httpx": "0.28.1", "aiohttp": "3.9.0", "requests": "2.31.0", "h2": "4.0.0"}
EXTRAS = {"httpx": "[http2]"}
FAMILY = {
    "httpx": ["httpx", "httpcore", "h11", "anyio", "sniffio", "certifi", "idna"],
    "aiohttp": ["aiohttp", "aiosignal", "frozenlist", "multidict", "yarl", "aiohappyeyeballs", "propcache", "attrs"],
    "requests": ["requests", "urllib3", "charset-normalizer", "idna", "certifi"],
    "h2": ["h2", "hpack", "hyperframe"],
}
OTHERS = ["anyio", "hpack", "opentelemetry-api", "opentelemetry-sdk", "syrupy", "uvloop"]
FRAMEWORKS = {"agents", "langgraph", "langchain", "langchain_core", "claude_agent_sdk", "pydantic_ai",
              "fastmcp", "mcp", "boto3", "openai", "logfire"}


def files_for(lib: str) -> list[str]:
    tests = ROOT / "sdks/python/tests"
    out = []
    for path in sorted(glob.glob(str(tests / "test_*.py"))):
        name = Path(path).name
        if name.startswith(("test_compat_", "test_hermes", "test_codex")):
            continue
        imported = set(re.findall(r"^\s*(?:import|from)\s+([a-zA-Z0-9_]+)", Path(path).read_text(), re.M))
        if imported & FRAMEWORKS:
            continue
        if lib in imported:
            out.append(name)
    return out


def main() -> int:
    cutoff = plan.iso(datetime.now(timezone.utc))
    points = []
    for lib, floor in LIBS.items():
        releases = plan.fetch_releases(lib)
        finals = plan._finals(releases, cutoff)
        newest = max(finals, key=plan.vkey)
        for role, version, python in (("floor", floor, "3.10"), ("newest", newest, "3.14")):
            point_cutoff = cutoff
            if role == "floor":
                point_cutoff = min(cutoff, plan.iso(plan.parse_iso(finals[version]["uploaded"]) + timedelta(days=14)))
            base = {
                "key": f"transport--{lib}--{role}", "adapter": lib, "role": role, "targets": {lib: version},
                "floats": [], "python": python, "in_claim": False, "cutoff": point_cutoff,
                "assume_measured": False, "uploaded": {lib: finals[version]["uploaded"]},
            }
            spec = {"extras": {lib: EXTRAS.get(lib, "")}, "imports": (), "optional": (), "evidence": "live", "replay_of": None,
                    "requires": [f"{o}{EXTRAS.get(o, '')}" for o in LIBS if o != lib] + OTHERS,
                    "tests": files_for(lib)}
            point = plan.compile_point(base, kind="published", subject=lib, spec=spec)
            point["probe"] = {"imports": [], "adapter": None}
            if role == "floor":
                point["install"]["era"] = FAMILY[lib]
            points.append(point)
            print(point["key"], python, version, point_cutoff, len(spec["tests"]))
    Path(sys.argv[1]).write_text(json.dumps({"cutoff": cutoff, "points": points}, indent=1), encoding="utf-8")
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
        handle.write("matrix=" + json.dumps({"include": [{"key": p["key"]} for p in points]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
