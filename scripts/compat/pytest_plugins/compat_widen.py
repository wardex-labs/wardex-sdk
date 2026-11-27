"""Treat each installed framework release as tested, for one test session.

A point above the tested range asks one question: would the adapter's suite pass if this
release were added to the table of tested versions? On the real table the adapter says
"untested" on such a release, which the suite's "wardex said nothing" assertions would
fail before the tests that answer the question ran. Loading this plugin adds each
installed release's line to the table for the session and puts the table back afterwards.

It lives beside the run and not in the tests tree on purpose: a point that tests a
published wardex release runs that release's own tests, and a release made before this
existed still has to be able to ask the question. It reads three names of the installed
package (`MEASURED`, `installed_version`, `is_measured`) and nothing else.

That `init()` says "untested" on the real table is checked separately, at every point, by
`scripts/compat/attach_probe.py`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest


def widen() -> dict:
    """Add each installed framework's release line to the table of tested versions.

    Returns the entries it replaced, so a caller can put them back."""
    from wardex_sdk._adapters import _measured

    before = dict(_measured.MEASURED)
    for name, entries in before.items():
        widened = []
        for entry in entries:
            version = _measured.installed_version(entry.distribution)
            if version is None or _measured.is_measured(entry, version):
                widened.append(entry)
                continue
            parts = version.split(".")
            width = max((len(minor.split(".")) for minor in entry.minors), default=2)
            line = ".".join(parts[:width])
            widened.append(dataclasses.replace(entry, minors=(*entry.minors, line)))
        _measured.MEASURED[name] = tuple(widened)
    return before


@pytest.fixture(scope="session", autouse=True)
def _compat_widen() -> Iterator[None]:
    from wardex_sdk._adapters import _measured

    before = widen()
    try:
        yield
    finally:
        _measured.MEASURED.update(before)
