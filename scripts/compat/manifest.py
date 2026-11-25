"""What the daily compatibility run installs and tests for each adapter.

One entry per adapter in the table of tested versions
(`sdks/python/src/wardex_sdk/_adapters/_measured.py`). A test holds the two
together, and holds every test file that imports a framework to that
framework's entry, so neither an adapter nor a file can drop out unnoticed.

No requirement here carries an upper bound. A cap added to quiet a failure
makes the run test a version users no longer install; a version that fails
stays visible as an open issue instead. A lower bound is a different thing:
it says what a test file was written against, and where it cannot be met (a
floor installed with the dependencies of its own time) that file is left out
and recorded, instead of failing for a reason that is the test's own.
"""

from __future__ import annotations

#: The oldest and newest interpreters the package declares. A floor runs on
#: the first, everything else on the second, so old meets old and new meets new.
PYTHONS = ("3.10", "3.14")
#: Every interpreter a point may fall back to when its target refuses its own.
SUPPORTED_PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")

#: A floor is installed with the dependencies that existed this many days after
#: it was published. A team still on that version locked its dependencies
#: around then; resolving it against today's index builds an environment
#: nobody has, and one whose failures say nothing about wardex.
FLOOR_WINDOW_DAYS = 14

#: How many release lines above the tested range are run. They show where a
#: break starts and how far the range could be widened.
BEYOND_LIMIT = 3

#: Pinned exactly: a test tool's release must not turn every point red on the
#: same morning. Raise these on purpose, in a commit of their own.
TOOLS = ("pytest==9.0.3", "pytest-asyncio==1.3.0", "pytest-timeout==2.4.0")
#: Loaded by name. Plugin autoload is off, because installed frameworks ship
#: pytest plugins of their own and those move with every framework release.
PLUGINS = ("pytest_asyncio.plugin", "pytest_timeout")
#: Seconds one test may take before it is failed rather than left to hang the job.
TEST_TIMEOUT_S = 120
#: Seconds one run of an adapter's files may take. A release that hangs many tests ends
#: here with a result, instead of ending at the job's limit with none.
SUITE_TIMEOUT_S = 1500
#: A first run longer than this is not repeated, so that a point always fits its job.
RERUN_BUDGET_S = 600

TESTS_DIR = "sdks/python/tests"

#: `targets` are the distributions the table of tested versions names, primary
#: first. `extras` are requested on a target so the framework's own bounds on
#: them apply. `requires` is what every test file of the adapter imports
#: beyond the target. An `optional` set is what only some files need: when it
#: cannot be installed beside the target, those files are left out and the
#: rest still run, so a new framework release is not hidden behind a test-only
#: dependency that has not caught up with it. `evidence` says what a pass
#: proves: "live" runs the framework's code; "wrapper-live" runs its Python
#: package against recorded output of the program it drives.
ADAPTERS: dict[str, dict] = {
    "openai_agents": {
        "label": "openai-agents",
        "targets": ("openai-agents",),
        "extras": {},
        "imports": ("agents",),
        "requires": ("h2", "hpack", "httpx[http2]", "httpx2", "mcp"),
        "tests": ("test_openai_agents_*.py", "test_conversation_boundaries.py"),
        "optional": (),
        "evidence": "live",
        "replay_of": None,
    },
    "langgraph": {
        "label": "langgraph",
        "targets": ("langgraph", "langgraph-prebuilt"),
        "extras": {},
        "imports": ("langgraph.graph",),
        "requires": ("langchain-core", "numpy"),
        "tests": ("test_langgraph_*.py",),
        "optional": ({"requires": ("langchain",), "tests": ("test_langgraph_streams.py",)},),
        "evidence": "live",
        "replay_of": None,
    },
    "anthropic_agent_sdk": {
        "label": "claude-agent-sdk",
        "targets": ("claude-agent-sdk",),
        "extras": {},
        "imports": ("claude_agent_sdk",),
        "requires": ("anyio",),
        "tests": (
            "test_agent_sdk_*.py",
            "test_claude_stream_parser.py",
            "test_otel_bridge_usage_keys.py",
        ),
        "optional": (),
        "evidence": "wrapper-live",
        "replay_of": "claude CLI 2.1.286",
    },
    "pydantic_ai": {
        "label": "pydantic-ai",
        "targets": ("pydantic-ai-slim",),
        "extras": {"pydantic-ai-slim": "[openai]"},
        "imports": ("pydantic_ai",),
        "requires": ("opentelemetry-api", "opentelemetry-sdk", "httpx[http2]", "h2", "hpack"),
        "tests": ("test_pydantic_ai_*.py", "test_bridge_core.py", "test_h2c_unended_stream.py"),
        "optional": (
            {
                "requires": ("fastmcp-slim[client]>=4.1", "mcp>=2.3", "anyio"),
                "tests": ("test_pydantic_ai_mcp.py",),
            },
            {"requires": ("boto3",), "tests": ("test_pydantic_ai_bedrock.py",)},
        ),
        "evidence": "live",
        "replay_of": None,
    },
}

#: Adapters with no package version to move: their suites replay recorded runs
#: on every pull request, and that is all a pass proves about them. Listed so
#: the records say so instead of saying nothing.
REPLAY_ONLY: dict[str, dict] = {
    "codex_exec": {"label": "codex", "replay_of": "codex-cli 0.160.0"},
    "hermes_agent": {"label": "hermes-agent", "replay_of": "Hermes Agent v0.23.3"},
}
