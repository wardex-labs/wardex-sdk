#!/usr/bin/env python3
"""Release helper: bump version, roll the changelog, commit, and tag.

Usage:
    python scripts/release.py python 0.1.0b2

Creates commit "chore(release): python-v<version>" and tag "python-v<version>".
Pushing is left to the operator (review first): git push origin main --tags
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "sdks" / "python" / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"


def bump_pyproject(version: str) -> None:
    text = PYPROJECT.read_text()
    new, n = re.subn(r'(?m)^version\s*=\s*".*"', f'version         = "{version}"', text)
    if n != 1:
        sys.exit("could not find a single version line in pyproject.toml")
    PYPROJECT.write_text(new)


def roll_changelog(version: str) -> None:
    text = CHANGELOG.read_text()
    if "## [Unreleased]" not in text:
        sys.exit("CHANGELOG.md missing '## [Unreleased]' section")
    from datetime import date  # local import; release time only

    today = date.today().isoformat()
    text = text.replace(
        "## [Unreleased]",
        f"## [Unreleased]\n\n## [{version}] - {today}",
        1,
    )
    CHANGELOG.write_text(text)


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] != "python":
        sys.exit("usage: python scripts/release.py python <version>")
    version = sys.argv[2]
    tag = f"python-v{version}"
    bump_pyproject(version)
    roll_changelog(version)
    subprocess.run(["git", "add", str(PYPROJECT), str(CHANGELOG)], check=True, cwd=ROOT)
    subprocess.run(["git", "commit", "-m", f"chore(release): {tag}"], check=True, cwd=ROOT)
    subprocess.run(["git", "tag", tag], check=True, cwd=ROOT)
    print(f"created commit and tag {tag}. Review, then: git push origin main --tags")


if __name__ == "__main__":
    main()
