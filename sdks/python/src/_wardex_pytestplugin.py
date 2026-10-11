"""The `pytest11` entry point behind `pytest --wardex-strict`.

Registered through the package's `pytest11` entry point, so the flag is the
whole setup: no `conftest.py` line, no fixture, no change to a test.

KEPT OUTSIDE THE `wardex_sdk` PACKAGE ON PURPOSE. pytest imports every
installed plugin when it starts, flag or not, and importing anything under
`wardex_sdk` runs the package's `__init__`, which loads the native extension
and the adapter modules: a cost every pytest run in every project with
wardex-sdk installed would pay, for a check it did not ask for. This module
imports nothing; the check itself (`wardex_sdk.testing.pytest_plugin`) is
imported only when the flag is given. Without the flag, this module adds two
options to `pytest --help` and nothing else runs. `-p no:wardex` keeps even
the options out.
"""

from __future__ import annotations

from typing import Any


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("wardex")
    group.addoption(
        "--wardex-strict",
        action="store_true",
        default=False,
        help=(
            "fail a test during which wardex missed something it should have recorded, "
            "or ran an adapter on a framework release it was not tested against"
        ),
    )
    group.addoption(
        "--wardex-strict-allow",
        action="append",
        default=[],
        metavar="REASON",
        help="a gap reason --wardex-strict accepts, or REASON:SUBJECT; repeatable",
    )


def pytest_configure(config: Any) -> None:
    if config.getoption("wardex_strict", default=False):
        from wardex_sdk.testing.pytest_plugin import StrictMode

        allow = frozenset(config.getoption("wardex_strict_allow", default=[]) or [])
        config.pluginmanager.register(StrictMode(config, allow), "wardex-strict")
