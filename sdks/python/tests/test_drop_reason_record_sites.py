"""Every drop reason is recorded somewhere, or says why it is not yet.

`wardex.diagnostics().losses` and `pytest --wardex-strict` see only what reaches
the drop ledger. A reason with a meaning and a next step but no record site is a
loss that still ends in a counter nobody reads: strict mode passes over it and a
receiver never hears of it. So each reason is held to having at least one record
site in the package, found by reading the source: `DropReason.<NAME>` spelled in
a module that calls the ledger (`record`, `record_items`, `lose`), outside a
comparison, so a reason chosen by a table or a classifier and then recorded
counts as recorded.

The reasons that have no site yet are listed below with why, and the list only
shrinks: a reason that gains a site fails here until its entry is deleted.
"""

from __future__ import annotations

import ast
import pathlib

from test_drop_ledger_census import _SRC, _scan
from wardex_sdk._assembly._ledger import DropReason

#: Reasons with no record site, and why. Delete an entry when its reason gains one.
_NOT_YET_RECORDED: dict[str, str] = {
    "PARTIAL_SUCCESS": "the OTLP exporter does not read a partial-success answer's rejected count",
    "TRANSPORT_UNSUPPORTED": "the OTLP exporter skips state snapshots without counting them",
    "EXCLUDED_BY_POLICY": "the byte seam's path exclusion is a counter only",
    "RESPONSE_UNOBSERVED": "no seam counts a request whose response never came",
    "REQUEST_UNOBSERVED": "the HTTP/1 and HTTP/2 trackers count this as a counter only",
    "CONNECTION_EVICTED": "an evicted connection ships a marked span; its exchange is not counted",
    "IN_FLIGHT_AT_CLOSE": "uninstall does not count the exchanges it cuts off",
    "UNCLASSIFIED_CONNECTION": "the seam does not count a connection first seen mid-response",
    "LATE_TO_CLOSED_UNIT": "the unit registry counts late data as counters only",
    "LINK_LOST": "the adapter surface counts an unresolved link target as a counter only",
    "PATCH_SUPERSEDED": "a superseded patch is found only at restore, and counted there only",
    "CHILD_UNINITIALIZED": "a spawned child that never initializes cannot be seen from the parent",
    "WS_SAMPLE_CAP": "the WebSocket sample cap marks its span but does not count it",
    "WS_FRAME_CAP": "an over-cap WebSocket frame marks its span but does not count it",
    "DECODED_CAP": "the decoded-size cap is enforced in the native parser, which reports no count",
    "BRIDGE_LATE": "a bridge span after its session closed is filed, then counted when unroutable",
    "BRIDGE_INCOMPLETE": "no site compares the bridge's per-call data with the run's totals",
    "LAUNCH_UNRECOGNIZED": (
        "a launch wardex cannot read is also one it cannot tell was the CLI agent at all, so "
        "counting it would count processes that were never an agent"
    ),
}

_LEDGER_KINDS = frozenset({"record", "lose"})


def _compared(tree: ast.AST) -> set[int]:
    """The ids of every node that is an operand of a comparison."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for operand in (node.left, *node.comparators):
                found.update(id(sub) for sub in ast.walk(operand))
    return found


def _recorded_reasons(root: pathlib.Path) -> dict[str, set[str]]:
    """Each reason spelled outside a comparison in a module that calls the
    ledger, with the modules it is spelled in."""
    recording = {module for (module, kind, _key) in _scan(root) if kind in _LEDGER_KINDS}
    found: dict[str, set[str]] = {}
    for module in sorted(recording):
        if module in {"_assembly/_drop_reasons.py", "_assembly/_explain.py"}:
            continue
        tree = ast.parse((root / module).read_text(encoding="utf-8"))
        compared = _compared(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "DropReason"
                and id(node) not in compared
            ):
                found.setdefault(node.attr, set()).add(module)
    return found


def test_every_drop_reason_has_a_record_site_or_says_why_not():
    recorded = _recorded_reasons(_SRC)
    missing = sorted(
        r for r in DropReason.__members__ if r not in recorded and r not in _NOT_YET_RECORDED
    )
    assert not missing, (
        "drop reasons with no record site: every loss of these kinds is invisible to "
        "wardex.diagnostics().losses and to strict mode. Record each where the loss happens, "
        f"or list it in _NOT_YET_RECORDED with why: {missing}"
    )


def test_the_not_yet_recorded_list_only_shrinks():
    recorded = _recorded_reasons(_SRC)
    stale = sorted(r for r in _NOT_YET_RECORDED if r in recorded)
    unknown = sorted(r for r in _NOT_YET_RECORDED if r not in DropReason.__members__)
    assert not unknown, f"not reasons at all: {unknown}"
    assert not stale, (
        f"these reasons are recorded now ({ {r: sorted(recorded[r]) for r in stale} }); "
        "delete their _NOT_YET_RECORDED entries"
    )


def test_the_scan_sees_a_reason_chosen_by_a_table_and_ignores_a_comparison(tmp_path):
    """The negative control: a reason a classifier returns and the caller then
    records counts; a reason only compared against does not."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "mod.py").write_text(
        "def classify(exc):\n"
        "    return DropReason.NETWORK_ERROR\n"
        "def send(exc, why):\n"
        "    if why is DropReason.QUEUE_OVERFLOW:\n"
        "        return\n"
        "    record(classify(exc))\n",
        encoding="utf-8",
    )
    (src / "quiet.py").write_text("X = DropReason.TABLE_FULL\n", encoding="utf-8")
    found = _recorded_reasons(src)
    assert found == {"NETWORK_ERROR": {"mod.py"}}
