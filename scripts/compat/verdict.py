"""What a point's outcome means, and whether a run is red.

One module, so the report, the pages and the release gate cannot come to mean different
things by "passes". The run is red in exactly two cases: a release inside a claim does not
pass, or a planned point left no result. Both apply only to points of a red-capable kind.
A watching point is recorded and reported and is never an argument for red.

Standard library only.
"""

from __future__ import annotations

from compat import kinds

#: A point in one of these states has passed.
GREEN = frozenset({"pass", "flaky"})
#: A point inside a claim in one of these states turns the run red.
RED = frozenset({"fail", "attach_failed", "no_result", "harness_error", "skips_changed"})
#: The release itself was run and did not hold. The other red states say only that it
#: could not be looked at, which is no reason to keep running it once it is replaced.
FAILED = frozenset({"fail", "attach_failed", "skips_changed"})
#: Nothing wardex or its tests did: reported only once it has lasted two days running.
NOT_OURS = frozenset({"install_network", "not_installable", "framework_import_error"})

OUTCOME_WORDS = {
    "pass": "pass",
    "flaky": "pass (a test failed once and passed when its file was run again)",
    "fail": "tests fail",
    "attach_failed": "the adapter did not attach as the SDK says it should",
    "skips_changed": "the set of skipped tests changed",
    "no_result": "no result",
    "harness_error": "the run itself failed",
    "install_network": "could not be installed (index or network)",
    "not_installable": "cannot be installed on this Python",
    "test_deps_unresolvable": "the test dependencies cannot be installed beside it",
    "framework_import_error": "the framework does not import, with or without wardex",
}


def is_watching(result: dict) -> bool:
    """Whether a result can never turn the run red.

    Stamped on the point by the plan, from `kinds.RED_CAPABLE`. A result with no stamp
    is from before kinds existed, when every point was a red-capable framework point."""
    return bool(result.get("watching", False))


def judge(plan: dict, results: dict[str, dict]) -> list[str]:
    """The reasons this run is red. Empty means green."""
    reasons = [f"plan: {problem}" for problem in plan.get("problems", [])]
    floor = kinds.KINDS["framework"].min_points
    for name, info in plan["adapters"].items():
        reasons += [f"plan: {name}: {problem}" for problem in info["problems"]]
        if sum(1 for p in plan["points"] if p["adapter"] == name) < floor:
            reasons.append(f"plan: {name} has fewer than two points")
    if not plan["points"]:
        reasons.append("plan: no points")
    for key, result in results.items():
        if is_watching(result):
            continue
        if result["outcome"] in ("no_result", "harness_error"):
            reasons.append(f"{key}: {OUTCOME_WORDS[result['outcome']]}")
        elif result["in_claim"] and result["outcome"] in RED:
            reasons.append(f"{key}: {OUTCOME_WORDS[result['outcome']]}")
    return reasons
