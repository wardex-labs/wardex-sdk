"""Run one point of the daily compatibility run and write its result.

A point is one framework version on one interpreter, in an environment that
holds the built wheel, that framework and nothing another adapter needs. The
steps run in a fixed order and each failure has one name, so the report can
tell "the new release changed what wardex records" from "PyPI was down":

1. resolve and install, at the point's cutoff instant;
2. import the framework without wardex (a release that cannot be imported
   fails for its users with or without wardex);
3. the attach probe, on the real table of tested versions;
4. the adapter's own test files against the installed wheel;
5. a failing file is run once more, whole and in order, to name a flaky test;
6. a failing point inside the tested range is run once more on the packages
   of its last pass with today's wheel: a pass there blames a dependency
   release, a failure there blames a wardex change.

Exits 0 whenever a result was written. The report decides the run's colour;
a point job that fails means only that no result exists.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compat import manifest  # noqa: E402

SCHEMA = 1
REPO = Path(__file__).resolve().parents[2]
PROBE = Path(__file__).with_name("attach_probe.py")
_NO_SOLUTION = ("No solution found", "not satisfiable", "requirements are unsatisfiable")


def run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None, timeout: int = 900):
    """Run `cmd`, never raising: the outcome of every step is data."""
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    try:
        done = subprocess.run(
            [str(part) for part in cmd],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as hung:
        return 124, (hung.stdout or "") if isinstance(hung.stdout, str) else "", "timed out"
    return done.returncode, done.stdout, done.stderr


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def python_of(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def lock_entries(text: str) -> list[str]:
    """A hashed requirements file as one string per requirement, hashes included."""
    entries, current = [], []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0].isspace() and current:
            current.append(line)
        else:
            if current:
                entries.append("\n".join(current))
            current = [line]
    if current:
        entries.append("\n".join(current))
    return entries


def pinned(text: str) -> dict[str, str]:
    """``{name: version}`` for every ``name==version`` in a requirements file."""
    return {
        match[1].lower().replace("_", "-"): match[2]
        for entry in lock_entries(text)
        if (match := re.match(r"([A-Za-z0-9._-]+)(?:\[[^\]]*\])?==([^\s\\;]+)", entry))
    }


def is_pass(exit_code: int, counts: dict) -> bool:
    """A pass is a clean exit that ran something.

    No tests collected, a collection error, a timeout or a teardown error is
    not a pass, whatever the other counts say.
    """
    return exit_code == 0 and counts["passed"] > 0 and not counts["failed"] and not counts["errors"]


def attach_verdict(seen: dict) -> str:
    """What the attach probe's reading means.

    `in_table` is the table's verdict on the installed version; `measured` is
    the adapter's own. An adapter that is not installed is the worst case. One
    whose account disagrees with the table, that said a line on a tested
    release, or stayed silent on an untested one, has broken the promise that
    an untested release is said to be untested.
    """
    if seen["state"] != "installed":
        return "not_installed"
    silent = seen["lines_said"] == 0
    if seen["measured"] is not seen["in_table"] or silent is not seen["in_table"]:
        return "flag_mismatch"
    return "ok"


def without_wardex(entries: list[str]) -> list[str]:
    """A lock's entries minus the wardex wheel, which a replay installs from today's build."""
    return [entry for entry in entries if "wardex_sdk-" not in entry.split("\n")[0]]


def cause_without_replay(*, in_claim: bool, frozen: bool, has_baseline: bool) -> str | None:
    """The cause that needs no second run, or None when only a replay can tell.

    Above the tested range the release is the cause by construction. A floor
    whose environment is frozen (its cutoff lies in the past) and that passed
    before can only have been broken by wardex.
    """
    if not in_claim:
        return "new_version"
    if not has_baseline:
        return "no_baseline"
    if frozen:
        return "ours"
    return None


def replay_can_tell(failed_ids: list[str], replayed_files: list[str]) -> bool:
    """Whether running `replayed_files` on the last pass's packages can explain the failure.

    The replay runs only the files that last pass ran. A failure in a file
    added since would not be run at all, the replay would pass, and a wardex
    change would be blamed on a dependency release.
    """
    failing_files = {node.split("::")[0] for node in failed_ids}
    return bool(failing_files) and failing_files <= set(replayed_files)


class Point:
    def __init__(self, args: argparse.Namespace) -> None:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        self.point = next(p for p in plan["points"] if p["key"] == args.key)
        self.run_cutoff = plan["cutoff"]
        self.spec = manifest.ADAPTERS[self.point["adapter"]]
        self.work = Path(args.work).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.out = Path(args.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        wheels = sorted(Path(args.wheel_dir).resolve().glob("wardex_sdk-*.whl"))
        if len(wheels) != 1:
            raise SystemExit(f"expected exactly one wardex_sdk wheel, found {len(wheels)}")
        self.wheel = wheels[0]
        self.last_green = Path(args.last_green_lock) if args.last_green_lock else None
        self.result: dict = {
            "schema": SCHEMA,
            "key": self.point["key"],
            "adapter": self.point["adapter"],
            "role": self.point["role"],
            "targets": self.point["targets"],
            "in_claim": self.point["in_claim"],
            "cutoff": self.point["cutoff"],
            "uploaded": self.point["uploaded"],
            "evidence": self.spec["evidence"],
            "replay_of": self.spec["replay_of"],
            "isolated": True,
            "os": platform.system().lower(),
            "arch": platform.machine().lower(),
            "runner_image": "-".join(
                filter(None, (os.environ.get("ImageOS"), os.environ.get("ImageVersion")))
            ),
            "sdk": {"wheel": self.wheel.name, "wheel_sha256": sha256(self.wheel)},
            "install": None,
            "resolved": {},
            "control_import": None,
            "attach": None,
            "tests": None,
            "rerun": None,
            "counterfactual": None,
            "outcome": None,
            "cause": None,
        }

    # --- environment ---------------------------------------------------------

    def _requirements(self, optional: tuple[int, ...], *, minimal: bool = False) -> str:
        lines = [str(self.wheel)]
        for dist, version in self.point["targets"].items():
            lines.append(f"{dist}{self.spec['extras'].get(dist, '')}=={version}")
        if not minimal:
            lines += [f"{dist}{self.spec['extras'].get(dist, '')}" for dist in self.point["floats"]]
            lines += self.spec["requires"]
            for index in optional:
                lines += self.spec["optional"][index]["requires"]
        lines += manifest.TOOLS
        return "\n".join(lines) + "\n"

    def _compile(self, venv: Path, requirements: str, lock: Path, *, refresh: bool = False):
        source = self.work / "requirements.in"
        source.write_text(requirements, encoding="utf-8")
        cmd = ["uv", "pip", "compile", source, "-o", lock, "--python", python_of(venv)]
        cmd += ["--generate-hashes", "--no-header", "--no-annotate", "--quiet"]
        cmd += ["--exclude-newer", self.point["cutoff"]]
        if self.point["cutoff"] != self.run_cutoff:
            # The test tools are today's, whatever the era of the framework under them.
            for tool in manifest.TOOLS:
                cmd += ["--exclude-newer-package", f"{tool.split('==')[0]}={self.run_cutoff}"]
        if refresh:
            for dist in self.point["targets"]:
                cmd += ["--refresh-package", dist]
        return run(cmd, cwd=self.work)

    def install(self, venv: Path) -> bool:
        """Resolve, lock with hashes and install. False when the point cannot go on."""
        for attempt in range(3):
            shutil.rmtree(venv, ignore_errors=True)
            code, _, err = run(
                ["uv", "venv", "--python", self.point["python"], venv], cwd=self.work
            )
            if code == 0:
                break
            time.sleep(20 * (attempt + 1))
        else:
            # No interpreter after three tries: the index or the network, not the framework.
            self.result["install"] = {"outcome": "install_network", "detail": err[-600:]}
            return False
        lock = self.out / "lock.txt"
        every = tuple(range(len(self.spec["optional"])))
        kept, detail = every, ""
        for attempt in range(3):
            code, _, err = self._compile(venv, self._requirements(every), lock, refresh=attempt > 0)
            if code == 0 or any(marker in err for marker in _NO_SOLUTION):
                break
            time.sleep(20)
        if code != 0 and any(marker in err for marker in _NO_SOLUTION):
            detail = err[-600:]
            code, _, err = self._compile(venv, self._requirements(()), lock)
            kept = ()
            if code == 0:
                # Add back each optional set that still resolves beside the target.
                for index in every:
                    trial = (*kept, index)
                    if self._compile(venv, self._requirements(trial), lock)[0] == 0:
                        kept = trial
                code, _, err = self._compile(venv, self._requirements(kept), lock)
        if code != 0:
            if not any(marker in err for marker in _NO_SOLUTION):
                outcome = "install_network"
            else:
                # Is it the target that cannot be installed here, or what the tests add to it?
                alone = self._compile(venv, self._requirements((), minimal=True), lock)[0] == 0
                outcome = "test_deps_unresolvable" if alone else "not_installable"
            self.result["install"] = {"outcome": outcome, "detail": (detail or err)[-600:]}
            lock.unlink(missing_ok=True)
            return False
        code, _, err = run(
            ["uv", "pip", "sync", lock, "--python", python_of(venv), "--require-hashes", "--quiet"],
            cwd=self.work,
        )
        if code != 0:
            self.result["install"] = {"outcome": "install_network", "detail": err[-600:]}
            return False
        dropped = [i for i in every if i not in kept]
        self.dropped_tests = [t for i in dropped for t in self.spec["optional"][i]["tests"]]
        self.result["install"] = {
            "outcome": "ok",
            "detail": "",
            "dropped_optional": [list(self.spec["optional"][i]["requires"]) for i in dropped],
        }
        self.result["lock_sha256"] = sha256(lock)
        return self._verify(venv)

    def _verify(self, venv: Path) -> bool:
        """The interpreter, the exact target versions, and the wheel as the thing imported."""
        probe = (
            "import importlib.metadata as m, json, sys, wardex_sdk;"
            "print(json.dumps({'python': sys.version.split()[0], 'sdk': wardex_sdk.__version__,"
            "'file': wardex_sdk.__file__,"
            "'dists': {d.metadata['Name'].lower().replace('_','-'): d.version"
            " for d in m.distributions()}}))"
        )
        # From the repository root, as pytest runs: a source tree on the path would
        # shadow the wheel.
        code, out, err = run([python_of(venv), "-c", probe], cwd=REPO)
        if code != 0:
            self.result["install"] = {"outcome": "attach_failed", "detail": err[-600:]}
            self.result["outcome"] = "attach_failed"
            return False
        seen = json.loads(out.strip().splitlines()[-1])
        self.result["python"] = seen["python"]
        self.result["sdk"]["version"] = seen["sdk"]
        self.result["resolved"] = dict(sorted(seen["dists"].items()))
        uv_version = run(["uv", "--version"])[1].split()
        self.result["uv_version"] = uv_version[1] if len(uv_version) > 1 else ""
        for dist, version in self.point["targets"].items():
            installed = seen["dists"].get(dist)
            print(f"installed {dist} {installed} (target {version})", flush=True)
            assert installed == version, f"{dist}: installed {installed}, the point is {version}"
        inside = Path(seen["file"]).resolve().is_relative_to(venv.resolve())
        print(f"wardex_sdk {seen['sdk']} imported from {seen['file']}", flush=True)
        assert inside, "the tests would import a source tree, not the built wheel"
        return True

    # --- checks --------------------------------------------------------------

    def control_import(self, venv: Path) -> bool:
        statement = "import " + ", ".join(self.spec["imports"])
        code, _, err = run([python_of(venv), "-c", statement], cwd=self.work, timeout=120)
        self.result["control_import"] = "ok" if code == 0 else err.strip().splitlines()[-1][:300]
        return code == 0

    def attach(self, venv: Path) -> bool:
        code, out, err = run(
            [python_of(venv), PROBE, self.point["adapter"], ",".join(self.spec["imports"])],
            cwd=self.work,
            timeout=120,
        )
        if code != 0:
            self.result["attach"] = {"verdict": "probe_failed", "detail": err[-600:]}
            return False
        seen = json.loads(out.strip().splitlines()[-1])
        self.result["attach"] = {**seen, "verdict": attach_verdict(seen)}
        return self.result["attach"]["verdict"] == "ok"

    def files(self) -> list[str]:
        root = REPO / manifest.TESTS_DIR
        dropped = {
            Path(found).name
            for pattern in getattr(self, "dropped_tests", [])
            for found in glob.glob(str(root / pattern))
        }
        chosen = sorted(
            {
                Path(found).name
                for pattern in self.spec["tests"]
                for found in glob.glob(str(root / pattern))
            }
            - dropped
        )
        self.excluded = sorted(dropped)
        return chosen

    def pytest(self, venv: Path, files: list[str], tag: str) -> dict:
        """Run `files` in order against the installed wheel.

        Counts and test ids come from the JUnit report, never from parsing output."""
        junit = self.work / f"junit-{tag}.xml"
        junit.unlink(missing_ok=True)
        cmd = [python_of(venv), "-m", "pytest", *(f"{manifest.TESTS_DIR}/{name}" for name in files)]
        cmd += ["-q", "-p", "no:cacheprovider", f"--timeout={manifest.TEST_TIMEOUT_S}"]
        cmd += [f"--junitxml={junit}", "-o", "junit_family=xunit1"]
        for plugin in manifest.PLUGINS:
            cmd += ["-p", plugin]
        env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
        env.pop("PYTHONPATH", None)
        if self.point["assume_measured"]:
            env["COMPAT_ASSUME_MEASURED"] = "1"
        started = time.monotonic()
        # Never longer than what the point has left, so the last word is always a result.
        limit = int(max(60, min(manifest.SUITE_TIMEOUT_S, self._left() - 60)))
        code, out, err = run(cmd, cwd=REPO, env=env, timeout=limit)
        print(out[-6000:], flush=True)
        if code not in (0, 1):
            print(err[-3000:], flush=True)
        counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
        failed, skipped = [], []
        if junit.exists() and code != 124:
            for case in ET.parse(junit).getroot().iter("testcase"):  # noqa: S314
                where = case.get("file") or (case.get("classname") or "").replace(".", "/") + ".py"
                node = f"{Path(where).name}::{case.get('name')}"
                kinds = {child.tag for child in case}
                if "failure" in kinds:
                    counts["failed"] += 1
                    failed.append(node)
                elif "error" in kinds:
                    counts["errors"] += 1
                    failed.append(node)
                elif "skipped" in kinds:
                    counts["skipped"] += 1
                    skipped.append(node)
                else:
                    counts["passed"] += 1
        return {
            **counts,
            "exit_code": code,
            "ok": is_pass(code, counts),
            "timed_out": code == 124,
            "seconds": round(time.monotonic() - started, 1),
            "failed_ids": sorted(set(failed)),
            "skipped_ids": sorted(set(skipped)),
        }

    def counterfactual(self, files: list[str], failed_ids: list[str]) -> None:
        """Today's wheel and tests on the packages of this point's last pass."""
        then = self.last_green.read_text(encoding="utf-8")
        now = (self.out / "lock.txt").read_text(encoding="utf-8")
        replay = self.work / "last-green.txt"
        replay.write_text("\n".join(without_wardex(lock_entries(then))) + "\n", encoding="utf-8")
        # Only the files that last pass ran: a file whose dependencies were left out then
        # cannot be collected on those packages, and that would read as a wardex change.
        listed = self.last_green.with_name(self.last_green.stem + ".files.json")
        if listed.is_file():
            files = [name for name in files if name in json.loads(listed.read_text("utf-8"))]
        if not replay_can_tell(failed_ids, files):
            self.result["counterfactual"] = {"ran": False}
            self.result["cause"] = "not_determined"
            return
        venv = self.work / "venv-last-green"
        shutil.rmtree(venv, ignore_errors=True)
        steps = (
            ["uv", "venv", "--python", self.point["python"], venv],
            ["uv", "pip", "sync", replay, "--python", python_of(venv), "--require-hashes", "-q"],
            ["uv", "pip", "install", "--python", python_of(venv), "--no-deps", "-q", self.wheel],
        )
        for step in steps:
            if run(step, cwd=self.work)[0] != 0:
                # A release of that day has since been withdrawn: the past cannot be rebuilt.
                self.result["counterfactual"] = {"ran": False}
                self.result["cause"] = "not_determined"
                return
        again = self.pytest(venv, files, "last-green")
        before, after = pinned(then), pinned(now)
        changed = {
            name: [before.get(name), after.get(name)]
            for name in sorted(set(before) | set(after))
            if before.get(name) != after.get(name)
        }
        self.result["counterfactual"] = {"ran": True, "passed": again["ok"], "changed": changed}
        if again["timed_out"]:
            self.result["cause"] = "not_determined"
        else:
            self.result["cause"] = "upstream" if again["ok"] else "ours"

    # --- the point -----------------------------------------------------------

    def execute(self) -> None:
        started = time.monotonic()
        self.deadline = started + manifest.POINT_BUDGET_S
        venv = self.work / "venv"
        try:
            self._steps(venv)
        except Exception as broken:  # noqa: BLE001 - whatever broke, the report must hear of it
            # The harness itself failed (the wrong version got installed, a tool crashed).
            # That is never a pass and never the framework's doing.
            self.result["outcome"] = "harness_error"
            self.result["harness_error"] = f"{type(broken).__name__}: {broken}"[:600]
        finally:
            self.result["duration_s"] = round(time.monotonic() - started, 1)
            (self.out / "result.json").write_text(
                json.dumps(self.result, indent=1, sort_keys=True), encoding="utf-8"
            )
            print(f"outcome: {self.result['outcome']} cause: {self.result['cause']}", flush=True)

    def _left(self) -> float:
        """Seconds this point may still spend before it must write its result."""
        return self.deadline - time.monotonic()

    def _steps(self, venv: Path) -> None:
        result = self.result
        if not self.install(venv):
            result["outcome"] = result["outcome"] or result["install"]["outcome"]
            return
        if not self.control_import(venv):
            result["outcome"] = "framework_import_error"
            return
        if not self.attach(venv):
            result["outcome"] = "attach_failed"
            result["cause"] = "new_version" if not self.point["in_claim"] else None
            return
        files = self.files()
        first = self.pytest(venv, files, "first")
        result["tests"] = {
            **first,
            "files": files,
            "excluded_files": self.excluded,
            "selection_sha256": hashlib.sha256(
                "".join(sha256(REPO / manifest.TESTS_DIR / name) for name in files).encode()
            ).hexdigest(),
            "plugins": list(manifest.PLUGINS),
        }
        if first["ok"]:
            result["outcome"] = "pass"
            return
        # Everything again, in the same order. An adapter patches its framework for the whole
        # process, so a failure that needs an earlier file to set it up is a real one, and a
        # rerun of the failing file alone would pass and call it flaky every day. Not repeated
        # when there is no time left for it: a result beats a second opinion.
        again = first["seconds"] * 1.5 + 120
        if first["exit_code"] == 1 and self._left() > again:
            second = self.pytest(venv, files, "rerun")
            result["rerun"] = {"passed": second["ok"]}
            if second["ok"]:
                result["outcome"] = "flaky"
                return
        result["outcome"] = "fail"
        result["cause"] = cause_without_replay(
            in_claim=self.point["in_claim"],
            frozen=self.point["cutoff"] != self.run_cutoff,
            has_baseline=self.last_green is not None and self.last_green.is_file(),
        )
        if result["cause"] is None:
            if self._left() > again:
                self.counterfactual(files, first["failed_ids"])
            else:
                result["cause"] = "not_determined"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--wheel-dir", required=True)
    parser.add_argument("--work", required=True, help="scratch directory for environments")
    parser.add_argument("--out", required=True, help="where result.json and lock.txt go")
    parser.add_argument("--last-green-lock", help="lock.txt of this point's last pass, if any")
    Point(parser.parse_args(argv)).execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
