"""Compute today's points: which exact framework versions the daily run tests.

Inputs are the table of tested versions, the manifest, and what PyPI lists at
one cutoff instant. The same three give the same plan, so a run re-attempted
days later resolves what the first attempt resolved. Nothing here is written
into a workflow file: a new framework release changes the plan by existing.

A point is one framework version on one interpreter:

- ``floor``: the lowest release still served inside the tested range, on the
  oldest interpreter, with the dependencies of its own time.
- ``minor_newest``: the newest patch of each tested release line below the
  top one, for a framework whose tree differs from line to line.
- ``claim_top``: the newest release inside the tested range, on the newest
  interpreter; ``claim_top_oldpy`` is the same question on the oldest one.
- ``beyond``: the newest patch of each release line above the tested range.
- ``displaced``: a release that failed while it was one of the points above and
  has since been replaced there by a newer one. It is still inside the tested
  range, so it keeps being run until it passes or the table lists it as bad.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compat import kinds, manifest, model  # noqa: E402

SCHEMA = model.SCHEMA
REPO = Path(__file__).resolve().parents[2]
MEASURED_PY = REPO / "sdks/python/src/wardex_sdk/_adapters/_measured.py"
ENUMS_PY = REPO / "sdks/python/src/wardex_sdk/_enums.py"

_FINAL = re.compile(r"\d+(?:\.\d+)*")
_CLAUSE = re.compile(r"\s*(>=|<=|==|!=|~=|>|<)\s*([0-9]+(?:\.[0-9]+)*)(\.\*)?\s*")


def vkey(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _release(version: str) -> tuple[int, ...]:
    """`vkey` with trailing zeros dropped: "2.54" and "2.54.0" are one release."""
    parts = list(vkey(version))
    while parts and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(text: str) -> datetime:
    return datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def read_measured(measured_py: Path = MEASURED_PY, enums_py: Path = ENUMS_PY) -> dict:
    """The table of tested versions, read from source without importing the package.

    The plan job runs before any wheel exists, and importing the package
    would load the native extension. Returns
    ``{adapter: [{"distribution", "minors", "known_bad", "since"}, ...]}``.
    """
    return parse_measured(
        measured_py.read_text(encoding="utf-8"), enums_py.read_text(encoding="utf-8")
    )


def parse_measured(measured_source: str, enums_source: str) -> dict:
    """`read_measured` on source text, so a release's table can be read from its tag."""
    names: dict[str, str] = {}
    for node in ast.walk(ast.parse(enums_source)):
        if isinstance(node, ast.ClassDef) and node.name == "AdapterName":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant):
                    names[stmt.targets[0].id] = stmt.value.value
    table: dict[str, list[dict]] = {}
    for node in ast.walk(ast.parse(measured_source)):
        target = getattr(node, "target", None) or (getattr(node, "targets", None) or [None])[0]
        if not (isinstance(target, ast.Name) and target.id == "MEASURED"):
            continue
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            entries = []
            for call in value.elts:
                args = [ast.literal_eval(arg) for arg in call.args]
                kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
                entries.append(
                    {
                        "distribution": args[0],
                        "minors": tuple(args[1]) if len(args) > 1 else tuple(kwargs["minors"]),
                        "known_bad": tuple(kwargs.get("known_bad", ())),
                        "since": kwargs.get("since", ""),
                    }
                )
            table[names[key.attr]] = entries
    if not table:
        raise ValueError("no MEASURED table found")
    return table


def fetch_releases(distribution: str) -> list[dict]:
    """Every file PyPI lists for `distribution`, as plain data.

    The JSON API rather than the simple index because it carries upload times
    and yank flags in one request. Each release keeps its files, because a
    release is installable at an instant only if a file had been uploaded by
    then. Tried three times: a plan that cannot be made leaves the whole run
    without a result, and one dropped connection is not worth that.
    """
    url = f"https://pypi.org/pypi/{distribution}/json"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                payload = json.load(response)
            break
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(15 * (attempt + 1))
    releases = []
    for version, files in payload["releases"].items():
        releases.append(
            {
                "version": version,
                "files": [
                    {
                        "uploaded": f["upload_time_iso_8601"][:19] + "Z",
                        "yanked": bool(f.get("yanked")),
                        "requires_python": f.get("requires_python") or "",
                    }
                    for f in files
                ],
            }
        )
    return releases


def admits(requires_python: str, python: str) -> bool:
    """Whether a ``Requires-Python`` value lets `python` ("3.10") install the release.

    Only the comparison forms that appear in practice. A clause this cannot
    read is taken as admitting: the installer is the authority, and a point
    that then fails to install is recorded as exactly that.
    """
    have = vkey(python)
    for clause in filter(None, (part.strip() for part in requires_python.split(","))):
        match = _CLAUSE.fullmatch(clause)
        if match is None:
            continue
        operator, text, wildcard = match.groups()
        # An interpreter is named by two numbers and stands for every patch of it, so a
        # bound is compared on as many numbers as the interpreter has.
        want = vkey(text)[: len(have)]
        want += (0,) * (len(have) - len(want))
        mine = have[: len(vkey(text))] if wildcard else have
        if operator == ">=" and not have >= want:
            return False
        if operator == ">" and not have > want:
            return False
        if operator == "<=" and not have <= want:
            return False
        if operator == "<" and not have < want:
            return False
        if operator == "==" and mine != vkey(text)[: len(mine)]:
            return False
        if operator == "!=" and mine == vkey(text)[: len(mine)]:
            return False
        if operator == "~=" and not (have >= want and have[0] == want[0]):
            return False
    return True


def _finals(releases: list[dict], cutoff: str) -> dict[str, dict]:
    """Final releases installable at `cutoff`: ``{version: {"uploaded", "requires_python"}}``."""
    out = {}
    for release in releases:
        if not _FINAL.fullmatch(release["version"]):
            continue
        live = [f for f in release["files"] if not f["yanked"] and f["uploaded"] <= cutoff]
        if live:
            out[release["version"]] = {
                "uploaded": min(f["uploaded"] for f in live),
                "requires_python": live[0]["requires_python"],
            }
    return out


def _line(version: str, width: int) -> str:
    return ".".join(version.split(".")[:width])


def claims(entry: dict, version: str) -> bool:
    """Whether the table calls final release `version` tested. The rule `is_measured` applies."""
    parts = vkey(version)
    if len(parts) < 2 or version in entry["known_bad"]:
        return False
    if entry.get("since") and _release(version) < _release(entry["since"]):
        return False
    return f"{parts[0]}.{parts[1]}" in entry["minors"] or str(parts[0]) in entry["minors"]


def spelled(entry: dict) -> str:
    """The tested range as the SDK says it to a user: "0.22.x, 0.23.x", "2.x from 2.54"."""
    since = entry.get("since") or ""
    return ", ".join(
        f"{minor}.x from {since}"
        if since == minor or since.startswith(f"{minor}.")
        else f"{minor}.x"
        for minor in entry["minors"]
    )


def compile_point(
    base: dict, *, kind: str, subject: str, spec: dict, sdk: dict | None = None
) -> dict:
    """The one place a planned point is finished: its kind, whether it may turn the run
    red, and the whole recipe the executor follows.

    The executor reads the point and nothing else, so everything a kind's declaration
    says about installing and checking is copied in here. That is what keeps it from
    ever asking what kind of point it is running."""
    return {
        **base,
        "kind": kind,
        "subject": subject,
        "adapters": [base["adapter"]],
        "watching": not kinds.is_red_capable(kind),
        "runner": "linux-x64",
        # Which wardex the point installs: the wheel this run built, or a release from PyPI.
        "sdk": dict(sdk) if sdk else {"origin": "build"},
        "install": {
            "extras": dict(spec["extras"]),
            "requires": list(spec["requires"]),
            "optional": [
                {"requires": list(extra["requires"]), "tests": list(extra["tests"])}
                for extra in spec["optional"]
            ],
            "tools": list(manifest.TOOLS),
        },
        "probe": {"imports": list(spec["imports"]), "adapter": base["adapter"]},
        "check": {
            "how": "suite",
            "files": list(spec["tests"]),
            "plugins": list(manifest.PLUGINS),
            "test_timeout_s": manifest.TEST_TIMEOUT_S,
            "suite_timeout_s": manifest.SUITE_TIMEOUT_S,
            "evidence": spec["evidence"],
            "replay_of": spec["replay_of"],
        },
        "budget_s": manifest.POINT_BUDGET_S,
        "fresh_days": kinds.KINDS[kind].fresh_days,
    }


def _claimed(finals: dict, entry: dict) -> list[str]:
    return sorted((v for v in finals if claims(entry, v)), key=vkey)


def _python_for(finals: dict, version: str, prefer: str) -> str | None:
    """`prefer` when the release admits it, else the nearest interpreter that does."""
    spec = finals[version]["requires_python"]
    if admits(spec, prefer):
        return prefer
    order = manifest.SUPPORTED_PYTHONS
    ranked = order if prefer == manifest.PYTHONS[0] else tuple(reversed(order))
    return next((python for python in ranked if admits(spec, python)), None)


def _prerelease_ahead(releases: list[dict], finals: dict, cutoff: str) -> str | None:
    """A pre-release published beyond the newest final, if there is one. Recorded, not run."""
    newest = max(finals, key=vkey) if finals else None
    ahead = []
    for release in releases:
        version = release["version"]
        head = re.match(r"\d+(?:\.\d+)*", version)
        if head is None or _FINAL.fullmatch(version):
            continue
        if any(not f["yanked"] and f["uploaded"] <= cutoff for f in release["files"]) and (
            newest is None or vkey(head.group()) > vkey(newest)
        ):
            ahead.append(version)
    return ahead[-1] if ahead else None


def plan_adapter(
    name: str, entries: list[dict], releases: dict, cutoff: str, displaced: list[dict] | None = None
) -> dict:
    """One adapter's points and the facts the report states about it.

    `displaced` is what the last report remembered: releases that failed as a
    point and are no longer picked by the rules above."""
    spec = manifest.ADAPTERS[name]
    primary, companions = entries[0], entries[1:]
    finals = _finals(releases[primary["distribution"]], cutoff)
    claimed = _claimed(finals, primary)
    info = {
        "label": spec["label"],
        "evidence": spec["evidence"],
        "replay_of": spec["replay_of"],
        "claim": {entry["distribution"]: list(entry["minors"]) for entry in entries},
        "said": {entry["distribution"]: spelled(entry) for entry in entries},
        # The table's own rows, so the report can ask whether a release is still claimed.
        "entries": [
            {**entry, "minors": list(entry["minors"]), "known_bad": list(entry["known_bad"])}
            for entry in entries
        ],
        "newest": max(finals, key=vkey) if finals else None,
        # The releases inside the range the index serves now, for each distribution a point
        # pins. One the index withdrew cannot be run, so nothing is owed for it.
        "served": {primary["distribution"]: list(claimed)},
        "prerelease_ahead": _prerelease_ahead(releases[primary["distribution"]], finals, cutoff),
        # A problem means the adapter could not be planned and turns the run red. A note is
        # a fact about the plan worth reading (a point that cannot exist) and does not.
        "problems": [],
        "notes": [],
    }
    if not claimed:
        info["problems"].append(
            f"no installable release of {primary['distribution']} lies inside its tested range"
        )
        return {"info": info, "points": []}

    width = max(len(minor.split(".")) for minor in primary["minors"])
    top, floor = claimed[-1], claimed[0]
    oldest_py, newest_py = manifest.PYTHONS

    companion_finals = {
        entry["distribution"]: _finals(releases[entry["distribution"]], cutoff)
        for entry in companions
    }
    for entry in companions:
        served = _claimed(companion_finals[entry["distribution"]], entry)
        info["served"][entry["distribution"]] = served

    def companion_pins(pick: int) -> dict[str, str]:
        pins = {}
        for entry in companions:
            versions = _claimed(companion_finals[entry["distribution"]], entry)
            if versions:
                pins[entry["distribution"]] = versions[pick]
        return pins

    def uploaded(targets: dict[str, str]) -> dict[str, str]:
        return {
            dist: (finals if dist == primary["distribution"] else companion_finals[dist])[version][
                "uploaded"
            ]
            for dist, version in targets.items()
        }

    points: list[dict] = []

    def add(role: str, version: str, prefer: str, *, pins: dict | None = None, line: str = ""):
        python = _python_for(finals, version, prefer)
        if python is None:
            # Above the range this is the framework's own business. Inside it, a release the
            # README calls tested cannot be run at all, and that must not pass in silence.
            said = f"{primary['distribution']} {version} admits no supported interpreter"
            info["notes" if role == "beyond" else "problems"].append(f"{said}; not run")
            return
        targets = {primary["distribution"]: version, **(pins or {})}
        stamps = uploaded(targets)
        in_claim = role != "beyond"
        point_cutoff = cutoff
        if role == "floor":
            window = timedelta(days=manifest.FLOOR_WINDOW_DAYS)
            point_cutoff = min(cutoff, iso(parse_iso(max(stamps.values())) + window))
        base = {
            "key": "--".join(filter(None, (name, role, line))),
            "adapter": name,
            "role": role,
            "targets": targets,
            "floats": [e["distribution"] for e in companions if e["distribution"] not in targets],
            "python": python,
            "in_claim": in_claim,
            "cutoff": point_cutoff,
            "assume_measured": not in_claim,
            "uploaded": stamps,
        }
        points.append(compile_point(base, kind="framework", subject=name, spec=spec))

    add("floor", floor, oldest_py, pins=companion_pins(0))
    lines = sorted({_line(v, width) for v in claimed}, key=vkey)
    for line in lines[:-1] if width > 1 else ():
        newest_in_line = [v for v in claimed if _line(v, width) == line][-1]
        if newest_in_line != floor:
            add("minor_newest", newest_in_line, newest_py, line=line)
    add("claim_top", top, newest_py, pins=companion_pins(-1))
    old_enough = [v for v in claimed if admits(finals[v]["requires_python"], oldest_py)]
    if old_enough:
        add("claim_top_oldpy", old_enough[-1], oldest_py)
    else:
        info["notes"].append(
            f"no tested release of {primary['distribution']} admits Python {oldest_py}"
        )

    above = sorted((v for v in finals if vkey(v) > vkey(top)), key=vkey)
    above = [
        v for v in above if v not in primary["known_bad"] and _line(v, width) != _line(top, width)
    ]
    beyond_lines = sorted({_line(v, width) for v in above}, key=vkey)[-manifest.BEYOND_LIMIT :]
    for line in beyond_lines:
        add("beyond", [v for v in above if _line(v, width) == line][-1], newest_py, line=line)

    for item in displaced or ():
        targets, python = item.get("targets") or {}, item.get("python")
        version = targets.get(primary["distribution"])
        # No longer claimed (listed as bad, withdrawn, the range moved): nothing left to prove.
        if version not in claimed or python not in manifest.SUPPORTED_PYTHONS:
            continue
        pins = {dist: v for dist, v in targets.items() if dist != primary["distribution"]}
        rows = {entry["distribution"]: entry for entry in companions}
        # Every pinned companion is held to its own row, as the primary is: listing the
        # companion's release as bad settles the pair too.
        if any(
            dist not in rows or v not in _claimed(companion_finals[dist], rows[dist])
            for dist, v in pins.items()
        ):
            continue
        # One of the rules above picks it again today: that point speaks for it. Not the
        # floor, which installs the dependencies of the release's own time, not today's.
        if any(
            p["targets"] == targets and p["python"] == python and p["role"] != "floor"
            for p in points
        ):
            continue
        add("displaced", version, python, pins=pins, line=displaced_line(targets, python))
    return {"info": info, "points": points}


def displaced_line(targets: dict, python: str) -> str:
    """The last part of a displaced point's key: the release and the interpreter it failed on."""
    return "-".join([*targets.values(), f"py{python}"])


# --- the published release -------------------------------------------------------

WARDEX = "wardex-sdk"
_WARDEX_VERSION = re.compile(r"(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+))?(?:\.post(\d+))?")
_PRE_RANK = {"a": 0, "b": 1, "rc": 2, None: 3}


def _wardex_key(version: str) -> tuple | None:
    """Order for wardex's own versions, pre-releases included: a final outranks its betas."""
    match = _WARDEX_VERSION.fullmatch(version)
    if not match:
        return None
    return (vkey(match[1]), _PRE_RANK[match[2]], int(match[3] or 0), int(match[4] or 0))


def newest_wardex(releases: list[dict], cutoff: str) -> str | None:
    """The newest wardex release PyPI serves at `cutoff`.

    Pre-releases count: what `pip install --pre wardex-sdk` gives is what a user has, and
    until a final exists a pre-release is all there is. A final, once published, outranks
    every pre-release of the same number, and a post release outranks its final, because
    that is what the installer serves. A development release is nobody's install."""
    live = [
        release["version"]
        for release in releases
        if _wardex_key(release["version"])
        and any(not f["yanked"] and f["uploaded"] <= cutoff for f in release["files"])
    ]
    return max(live, key=_wardex_key, default=None)


def _git(*args: str) -> str:
    done = subprocess.run(  # noqa: S603
        ["git", "-C", str(REPO), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout


def read_tag(version: str) -> dict:
    """What the tag of a published release holds: its commit, its table of tested
    versions and the names of its test files.

    A release is judged by its own table and its own tests, read from its tag and never
    from this checkout: the README a user of that release reads is that tag's."""
    ref = f"python-v{version}"
    try:
        commit = _git("rev-parse", "--verify", "--quiet", f"refs/tags/{ref}^{{commit}}").strip()
    except subprocess.CalledProcessError:
        # A workflow checkout holds one commit. Deepened only when it is already shallow:
        # `--depth` on a full clone would cut its history.
        shallow = _git("rev-parse", "--is-shallow-repository").strip() == "true"
        depth = ["--depth", "1"] if shallow else []
        _git("fetch", "--quiet", *depth, "origin", f"refs/tags/{ref}:refs/tags/{ref}")
        commit = _git("rev-parse", "--verify", f"refs/tags/{ref}^{{commit}}").strip()
    table = parse_measured(
        _git("show", f"{commit}:{MEASURED_PY.relative_to(REPO)}"),
        _git("show", f"{commit}:{ENUMS_PY.relative_to(REPO)}"),
    )
    tests = _git("ls-tree", "--name-only", f"{commit}:{model.TESTS_DIR}").split()
    try:
        override = model.parse_override(_git("show", f"{commit}:{model.OVERRIDE}"), version)
    except (subprocess.CalledProcessError, ValueError):
        # No file is the usual case: the release went through its gate. A file that is
        # not text is no override either, and must not cost the release its points.
        override = None
    return {
        "ref": ref,
        "commit": commit,
        "table": table,
        "tests": sorted(tests),
        "override": override,
    }


def plan_published(version: str, tag: dict, releases: dict, cutoff: str) -> dict:
    """The newest published wardex on the newest release its own table claims, and on the
    newest release above that range.

    The same rule that plans a framework point picks the releases, from the table the
    release shipped with. What differs is in the point: the wheel comes from PyPI, and
    the tests are the ones in the release's tag, where the executor looks the manifest's
    file patterns up. A file this checkout has and the tag has not is simply not there:
    the release never claimed to pass it."""
    points, adapters = [], {}
    sdk = {"origin": "pypi", "version": version, "ref": tag["ref"], "commit": tag["commit"]}
    for name, entries in sorted(tag["table"].items()):
        if not entries or name not in manifest.ADAPTERS:
            continue
        spec = manifest.ADAPTERS[name]
        planned = plan_adapter(name, entries, releases, cutoff)
        info = adapters[name] = planned["info"]
        if not any(
            fnmatch.fnmatch(held, pattern) for held in tag["tests"] for pattern in spec["tests"]
        ):
            info["problems"].append(f"the tag {tag['ref']} holds none of its test files")
            continue
        chosen = [p for p in planned["points"] if p["role"] == "claim_top"]
        chosen += [p for p in planned["points"] if p["role"] == "beyond"][-1:]
        for picked in chosen:
            base = {**picked, "key": f"published--{version}--{picked['key']}"}
            points.append(compile_point(base, kind="published", subject=name, spec=spec, sdk=sdk))
    return {
        "version": version,
        "ref": tag["ref"],
        "commit": tag["commit"],
        "override": tag.get("override"),
        "points": points,
        "adapters": adapters,
    }


def fetch_published(cutoff: str, only: str | None = None) -> dict:
    """`plan_published` on what PyPI and the release's tag say now. Never raises: a day the
    release cannot be read is a day its points are not planned, and that is said."""
    try:
        version = newest_wardex(fetch_releases(WARDEX), cutoff)
        if version is None:
            return {"problems": ["PyPI lists no wardex release"], "points": [], "adapters": {}}
        tag = read_tag(version)
        if only:
            tag["table"] = {name: rows for name, rows in tag["table"].items() if name == only}
        releases = {
            entry["distribution"]: fetch_releases(entry["distribution"])
            for name, entries in tag["table"].items()
            if name in manifest.ADAPTERS
            for entry in entries
        }
        return {"problems": [], **plan_published(version, tag, releases, cutoff)}
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        said = f"the newest wardex release could not be read ({type(error).__name__})"
        return {"problems": [said], "points": [], "adapters": {}}


# --- the other things a run looks at ------------------------------------------------
#
# Each rule below turns a declaration of the manifest and what the plan read from outside
# into points of one kind. They build on the newest tested point the framework rule made
# for an adapter (its release, its interpreter, the instant it was resolved at), so a
# combination is the frameworks a user installs together today, not a second opinion on
# which release is the newest. A rule returns its points, the subjects it planned in
# full, and what it could not plan. Nothing it cannot plan turns a run red.

_BASE = "key adapter role targets floats python in_claim cutoff assume_measured uploaded".split()
_SHA = re.compile(r"[0-9a-f]{40}")
_TAG = re.compile(r"v\d+(?:\.\d+){1,3}")
_NO_SPEC = {
    "extras": {},
    "requires": (),
    "optional": (),
    "imports": (),
    "tests": (),
    "evidence": "live",
    "replay_of": None,
}


def _base(point: dict, **change: object) -> dict:
    """The part of a compiled point a rule decides, to compile another point from."""
    return {**{name: json.loads(json.dumps(point[name])) for name in _BASE}, **change}


def _unique(items: list) -> list:
    return list(dict.fromkeys(items))


def _driven(point: dict, script: str, args: list[str], evidence: str) -> dict:
    """`point`, checked by a script of the tests tree that writes its own verdict."""
    point["check"] = {
        **point["check"],
        "how": "driver",
        "driver": script,
        "args": list(args),
        "files": [],
        "suite_timeout_s": manifest.DRIVER_TIMEOUT_S,
        "evidence": evidence,
        "replay_of": None if evidence == "cli-live" else point["check"]["replay_of"],
    }
    return point


def plan_combination(tops: dict[str, dict]) -> dict:
    """The frameworks of each declared set in one environment, every member at its newest
    tested release, with every member's tests and every member's attach probe."""
    points, covered, problems = [], [], []
    for name, declared in manifest.COMBINATIONS.items():
        members = list(declared["members"])
        missing = [member for member in members if member not in tops]
        if missing:
            problems.append(f"{name}: no newest tested point for {', '.join(missing)}")
            continue
        first, rest = tops[members[0]], [tops[member] for member in members[1:]]
        if len({point["python"] for point in (first, *rest)}) != 1:
            problems.append(f"{name}: its members do not share an interpreter")
            continue
        specs = [manifest.ADAPTERS[member] for member in members]
        targets = {d: v for point in (first, *rest) for d, v in point["targets"].items()}
        base = _base(
            first,
            key=f"combination--{name}",
            targets=targets,
            uploaded={d: s for point in (first, *rest) for d, s in point["uploaded"].items()},
            floats=[
                d
                for d in _unique([d for p in (first, *rest) for d in p["floats"]])
                if d not in targets
            ],
        )
        replayed = next((spec for spec in specs if spec["evidence"] != "live"), None)
        spec = {
            "extras": {d: e for spec in specs for d, e in spec["extras"].items()},
            "requires": _unique([r for spec in specs for r in spec["requires"]]),
            "optional": [extra for spec in specs for extra in spec["optional"]],
            "imports": _unique([i for spec in specs for i in spec["imports"]]),
            "tests": _unique([t for spec in specs for t in spec["tests"]]),
            "evidence": replayed["evidence"] if replayed else "live",
            "replay_of": replayed["replay_of"] if replayed else None,
        }
        point = compile_point(base, kind="combination", subject=name, spec=spec)
        point["adapters"] = members
        point["probe"]["adapters"] = [
            {"adapter": member, "imports": list(manifest.ADAPTERS[member]["imports"])}
            for member in members
        ]
        points.append(point)
        covered.append(name)
    return {"points": points, "covered": covered, "problems": problems}


def plan_neighbour(tops: dict[str, dict]) -> dict:
    """An adapter's newest tested release with another instrumentation library installed,
    checked by the driver that compares the tree alone and beside it."""
    points, covered, problems = [], [], []
    for name, declared in manifest.NEIGHBOURS.items():
        adapter = declared["adapter"]
        if adapter not in tops:
            problems.append(f"{name}: no newest tested point for {adapter}")
            continue
        spec = manifest.ADAPTERS[adapter]
        beside = {
            **spec,
            "requires": (*spec["requires"], *declared["requires"]),
            "optional": (),
            "tests": (),
        }
        base = _base(tops[adapter], key=f"neighbour--{name}")
        point = compile_point(base, kind="neighbour", subject=adapter, spec=beside)
        args = [name, "--out", "{out}/driver.json"]
        points.append(_driven(point, manifest.NEIGHBOUR_DRIVER, args, spec["evidence"]))
        covered.append(adapter)
    return {"points": points, "covered": _unique(covered), "problems": problems}


def plan_prerelease(measured: dict, adapters: dict, releases: dict, cutoff: str) -> dict:
    """The newest pre-release ahead of the newest final, for each adapter that has one."""
    points, covered, problems = [], [], []
    for name, info in sorted(adapters.items()):
        version = info.get("prerelease_ahead")
        covered.append(name)
        if not version:
            continue
        primary, companions = measured[name][0], measured[name][1:]
        release = next(r for r in releases[primary["distribution"]] if r["version"] == version)
        live = [f for f in release["files"] if not f["yanked"] and f["uploaded"] <= cutoff]
        order = tuple(reversed(manifest.SUPPORTED_PYTHONS))
        python = next(
            (py for py in order if any(admits(f["requires_python"], py) for f in live)), None
        )
        if python is None:
            problems.append(f"{name}: {version} admits no supported interpreter")
            covered.pop()
            continue
        base = {
            "key": f"prerelease--{name}",
            "adapter": name,
            "role": "prerelease",
            "targets": {primary["distribution"]: version},
            "floats": [entry["distribution"] for entry in companions],
            "python": python,
            "in_claim": False,
            "cutoff": cutoff,
            "assume_measured": True,
            "uploaded": {primary["distribution"]: min(f["uploaded"] for f in live)},
        }
        spec = manifest.ADAPTERS[name]
        points.append(compile_point(base, kind="prerelease", subject=name, spec=spec))
    return {"points": points, "covered": covered, "problems": problems}


def _two(listed: dict[str, str], claimed: str, order) -> tuple[list[tuple], list[str]]:  # noqa: ANN001
    """The release a sentence names and the newest one beside it, from `{release: when}`."""
    chosen, problems = [], []
    if claimed in listed:
        chosen.append(("claim_top", claimed, True))
    else:
        problems.append(f"{claimed}, the release named as tested, is not listed")
    newest = max(listed, key=order, default=None)
    if newest and newest != claimed:
        chosen.append(("beyond", newest, False))
    return chosen, problems


def plan_cli(tops: dict[str, dict], facts: dict, cutoff: str) -> dict:
    """The programs the adapters record, run for real against the loopback stand-in."""
    points, covered, problems = [], [], []
    for subject, declared in manifest.CLI.items():
        args = [declared["program"], "--out", "{out}/driver.json"]
        if "adapter" in declared:
            if declared["adapter"] not in tops:
                problems.append(f"{subject}: no newest tested point for {declared['adapter']}")
                continue
            spec = {**manifest.ADAPTERS[declared["adapter"]], "optional": (), "tests": ()}
            base = _base(tops[declared["adapter"]], key=f"cli--{subject}")
            point = compile_point(base, kind="cli", subject=subject, spec=spec)
            points.append(_driven(point, manifest.CLI_DRIVER, args, "cli-live"))
            covered.append(subject)
            continue
        package = declared["npm"]
        listed = (facts.get("npm") or {}).get(package)
        if listed is None:
            problems.append(f"{subject}: the npm registry could not be read for {package}")
            continue
        stable = {
            entry["version"]: entry["uploaded"]
            for entry in listed
            if _FINAL.fullmatch(entry["version"]) and entry["uploaded"] <= cutoff
        }
        chosen, missing = _two(stable, declared["claimed"], vkey)
        problems += [f"{subject}: {package} {problem}" for problem in missing]
        for role, version, in_claim in chosen:
            base = {
                "key": f"cli--{subject}--{role}",
                "adapter": subject,
                "role": role,
                "targets": {package: version},
                "floats": [],
                "python": manifest.PYTHONS[1],
                "in_claim": in_claim,
                "cutoff": cutoff,
                "assume_measured": False,
                "uploaded": {package: stable[version]},
            }
            point = compile_point(base, kind="cli", subject=subject, spec=_NO_SPEC)
            point["install"]["npm"] = {package: version}
            point["probe"] = {"imports": [], "adapter": None}
            points.append(_driven(point, manifest.CLI_DRIVER, args, "cli-live"))
        if not missing:
            covered.append(subject)
    return {"points": points, "covered": covered, "problems": problems}


def plan_hermes(facts: dict, cutoff: str) -> dict:
    """The Hermes smoke against the release tag the README names and against the newest."""
    declared = manifest.HERMES
    listed = (facts.get("github_releases") or {}).get(declared["repository"])
    if listed is None:
        said = f"the releases of {declared['repository']} could not be read"
        return {"points": [], "covered": [], "problems": [said]}
    finals = {
        entry["tag"]: entry["published"]
        for entry in listed
        if not entry["prerelease"] and _TAG.fullmatch(entry["tag"]) and entry["published"] <= cutoff
    }
    # The newest by when it was published, not by its number: the repository has carried
    # more than one numbering, and a tag of an older scheme sorts above every newer release.
    chosen, problems = _two(finals, declared["claimed"], lambda tag: (finals[tag], tag))
    points = []
    for role, tag, in_claim in chosen:
        base = {
            "key": f"hermes--{role}",
            "adapter": declared["subject"],
            "role": role,
            "targets": {"hermes-agent": tag},
            "floats": [],
            "python": declared["python"],
            "in_claim": in_claim,
            "cutoff": cutoff,
            "assume_measured": False,
            "uploaded": {"hermes-agent": finals[tag]},
        }
        point = compile_point(base, kind="hermes", subject=declared["subject"], spec=_NO_SPEC)
        # The driver builds Hermes's own environment and installs the wheel into it.
        point["install"]["how"] = "self"
        point["probe"] = {"imports": [], "adapter": None}
        args = ["--tag", "{target}", "--wheel", "{wheel}", "--work", "{work}/hermes"]
        args += ["--out", "{out}/driver.json"]
        points.append(_driven(point, manifest.HERMES_DRIVER, args, "live"))
    covered = [declared["subject"]] if not problems else []
    return {"points": points, "covered": covered, "problems": problems}


def plan_devbranch(measured: dict, tops: dict[str, dict], facts: dict) -> dict:
    """Each declared framework at the head commit of its default branch, with the
    adapter's own tests. The head is what the plan read; the point names the commit."""
    points, covered, problems = [], [], []
    for name, declared in manifest.DEV_BRANCHES.items():
        head = (facts.get("heads") or {}).get(declared["repository"])
        if not isinstance(head, str) or not _SHA.fullmatch(head):
            problems.append(f"{name}: the head of {declared['repository']} could not be read")
            continue
        if name not in tops:
            problems.append(f"{name}: no newest tested point to take its environment from")
            continue
        sources = {
            dist: f"git+{declared['repository']}@{head}" + (f"#subdirectory={sub}" if sub else "")
            for dist, sub in declared["sources"].items()
        }
        base = _base(
            tops[name],
            key=f"devbranch--{name}",
            role="devbranch",
            targets={dist: f"head-{head[:10]}" for dist in sources},
            floats=[e["distribution"] for e in measured[name] if e["distribution"] not in sources],
            in_claim=False,
            assume_measured=True,
            uploaded={},
        )
        spec = manifest.ADAPTERS[name]
        point = compile_point(base, kind="devbranch", subject=name, spec=spec)
        point["install"]["sources"] = sources
        points.append(point)
        covered.append(name)
    return {"points": points, "covered": covered, "problems": problems}


def subjects() -> dict[str, dict]:
    """Every subject an issue can be about, and what to call it. An adapter is one; so is
    a set of frameworks run together, and so is a program that has no tested range."""
    named = {name: {"label": spec["label"]} for name, spec in manifest.ADAPTERS.items()}
    named.update({name: {"label": spec["label"]} for name, spec in manifest.REPLAY_ONLY.items()})
    named.update({name: {"label": spec["label"]} for name, spec in manifest.COMBINATIONS.items()})
    return named


def build_plan(
    measured: dict,
    releases: dict,
    cutoff: str,
    state: dict | None = None,
    *,
    cadence: str = "daily",
    partial: bool = False,
    published: dict | None = None,
    facts: dict | None = None,
) -> dict:
    adapters, points, problems = {}, [], []
    remembered = (state or {}).get("displaced") or {}
    # A run of a cadence the framework kind is not part of plans none of its points and so
    # has nothing to say about an adapter's issue: it must not close what it did not run.
    # Its other rules still build on the newest tested point of each adapter, so those are
    # worked out either way and kept out of the plan.
    due = "framework" in kinds.due(cadence)
    tops: dict[str, dict] = {}
    for name, entries in sorted(measured.items()):
        if not entries:
            if due and name not in manifest.REPLAY_ONLY:
                problems.append(f"{name} has no tested versions and no replay entry")
            continue
        if name not in manifest.ADAPTERS:
            if due:
                problems.append(
                    f"{name} is in the table of tested versions but not in the manifest"
                )
            continue
        planned = plan_adapter(name, entries, releases, cutoff, remembered.get(name))
        top = [p for p in planned["points"] if p["role"] == "claim_top"]
        if top:
            tops[name] = top[0]
        if due:
            adapters[name] = planned["info"]
            points.extend(planned["points"])
    # What this run speaks for. A kind is covered for the subjects it planned in full; the
    # report drops what it remembers of a covered subject's points that are no longer
    # planned, and touches nothing outside the cover. A day the index would not list an
    # adapter's releases covers nothing of that adapter, so it forgets nothing either.
    covered = sorted(name for name, info in adapters.items() if not info["problems"])
    scope = {
        "cadence": cadence,
        "covers": {"framework": covered} if due else {},
        "partial": partial,
        "builds": {},
    }
    # What a watching kind could not plan. Said on the health issue and never a reason for
    # red: `problems` above is read by the verdict, this is not.
    unplanned: list[str] = []
    release = None
    if "published" in kinds.due(cadence) and published:
        unplanned += [f"published: {problem}" for problem in published.get("problems", [])]
        if published.get("version"):
            release = {
                k: published.get(k) for k in ("version", "ref", "commit", "override", "adapters")
            }
            points.extend(published["points"])
            for name, info in sorted(published["adapters"].items()):
                unplanned += [f"published: {name}: {problem}" for problem in info["problems"]]
            scope["covers"]["published"] = sorted(
                name for name, info in published["adapters"].items() if not info["problems"]
            )
            # Points of an older release are that release's history. A run that plans this
            # one forgets nothing of theirs.
            scope["builds"]["published"] = [published["version"]]
    # The other rules. A run of one adapter asks only about that adapter's tested range,
    # and so does a plan made without what the rules read from outside (`facts` is None).
    rules = {
        "combination": lambda: plan_combination(tops),
        "neighbour": lambda: plan_neighbour(tops),
        "prerelease": lambda: plan_prerelease(measured, adapters, releases, cutoff),
        "cli": lambda: plan_cli(tops, facts or {}, cutoff),
        "hermes": lambda: plan_hermes(facts or {}, cutoff),
        "devbranch": lambda: plan_devbranch(measured, tops, facts or {}),
    }
    for kind in kinds.due(cadence) if not partial and facts is not None else ():
        if kind not in rules:
            continue
        made = rules[kind]()
        points.extend(made["points"])
        unplanned += [f"{kind}: {problem}" for problem in made["problems"]]
        scope["covers"][kind] = sorted(made["covered"])
    return {
        "schema": SCHEMA,
        "cutoff": cutoff,
        "cadence": cadence,
        "scope": scope,
        "partial": partial,
        "adapters": adapters,
        "published": release,
        "claims": [dict(claim) for claim in manifest.CLAIMS],
        "subjects": subjects(),
        "replay_only": manifest.REPLAY_ONLY,
        "points": points,
        "problems": problems,
        "unplanned": unplanned,
    }


def _get(url: str, headers: dict | None = None) -> object:
    """One JSON document, tried three times."""
    request = urllib.request.Request(url, headers={"Accept": "application/json", **(headers or {})})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                return json.load(response)
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(15 * (attempt + 1))
    return None


def fetch_npm(package: str) -> list[dict]:
    """Every version the npm registry lists for `package`, with when it was published."""
    listed = _get("https://registry.npmjs.org/" + package.replace("/", "%2F"))["time"]
    return [
        {"version": version, "uploaded": str(when)[:19] + "Z"}
        for version, when in listed.items()
        if version not in ("created", "modified")
    ]


def fetch_github_releases(repository: str) -> list[dict]:
    """The releases of a public repository, newest first as the API lists them.

    The run's own read-only token is sent when there is one: without it the request is
    counted against the address the runner shares with everyone else's."""
    token = os.environ.get("GH_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    listed = _get(f"https://api.github.com/repos/{repository}/releases?per_page=50", headers)
    return [
        {
            "tag": str(entry["tag_name"]),
            "published": str(entry["published_at"])[:19] + "Z",
            "prerelease": bool(entry["prerelease"] or entry["draft"]),
        }
        for entry in listed
    ]


def fetch_head(repository: str) -> str:
    """The commit a repository's default branch points at now."""
    done = subprocess.run(  # noqa: S603
        ["git", "ls-remote", repository, "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return done.stdout.split()[0]


def fetch_facts(cadence: str) -> dict:
    """What the rules of the kinds due read from outside PyPI. Never raises: a source that
    cannot be read is left out, and the rule that needed it says so."""
    facts: dict = {"npm": {}, "github_releases": {}, "heads": {}}
    wanted = kinds.due(cadence)
    failed = (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError)
    if "cli" in wanted:
        for declared in manifest.CLI.values():
            if "npm" in declared:
                try:
                    facts["npm"][declared["npm"]] = fetch_npm(declared["npm"])
                except failed:
                    pass
    if "hermes" in wanted:
        try:
            repository = manifest.HERMES["repository"]
            facts["github_releases"][repository] = fetch_github_releases(repository)
        except failed:
            pass
    if "devbranch" in wanted:
        for declared in manifest.DEV_BRANCHES.values():
            try:
                facts["heads"][declared["repository"]] = fetch_head(declared["repository"])
            except failed:
                pass
    return facts


def cadence_of(asked: str | None, schedule: str | None) -> str:
    """The cadence of this run: the one asked for, else the one of the cron line that
    started it, else the daily one (a run started by hand)."""
    if asked:
        if asked not in set(kinds.CRONS.values()):
            raise SystemExit(f"no such cadence: {asked!r}")
        return asked
    return kinds.CRONS.get(schedule or "", "daily")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="where to write plan.json")
    parser.add_argument("--cutoff", help="UTC instant (ISO 8601); default now")
    parser.add_argument("--only", help="plan one adapter only")
    parser.add_argument("--state", help="state.json of the records branch, when there is one")
    parser.add_argument("--cadence", help="daily or weekly; default from the schedule")
    args = parser.parse_args(argv)
    cadence = cadence_of(args.cadence, os.environ.get("COMPAT_SCHEDULE"))

    cutoff = iso(parse_iso(args.cutoff)) if args.cutoff else iso(datetime.now(timezone.utc))
    measured = read_measured()
    if args.only:
        measured = {args.only: measured[args.only]}
    releases = {
        entry["distribution"]: fetch_releases(entry["distribution"])
        for entries in measured.values()
        for entry in entries
    }
    state_file = Path(args.state) if args.state else None
    state = json.loads(state_file.read_text("utf-8")) if state_file and state_file.is_file() else {}
    # A plan of one adapter says nothing about the others; the report must not read it
    # as the whole run.
    published = fetch_published(cutoff, args.only) if "published" in kinds.due(cadence) else None
    plan = build_plan(
        measured,
        releases,
        cutoff,
        state,
        cadence=cadence,
        partial=bool(args.only),
        published=published,
        facts=None if args.only else fetch_facts(cadence),
    )
    Path(args.out).write_text(json.dumps(plan, indent=1, sort_keys=True), encoding="utf-8")
    for point in plan["points"]:
        targets = ", ".join(f"{d}=={v}" for d, v in point["targets"].items())
        print(f"{point['key']:<44} py{point['python']:<5} {targets}")
    for problem in plan["problems"]:
        print(f"problem: {problem}", file=sys.stderr)
    for problem in plan["unplanned"]:
        print(f"not planned: {problem}", file=sys.stderr)
    # `tree` is the commit whose tests the point runs when that is not this checkout.
    matrix = {
        "include": [
            {"key": p["key"], "python": p["python"], "tree": p["sdk"].get("commit", "")}
            for p in plan["points"]
        ]
    }
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write("matrix=" + json.dumps(matrix, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
