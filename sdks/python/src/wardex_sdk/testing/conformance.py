"""The invariants every framework adapter owes, in one place.

Two adapters shipped before this existed and each proved the same four things
in its own file, in its own style, with its own helpers: that the causal tree
is real, that every span site declares whether it may begin a trace, that an
install is exactly undone, and that a shutdown mid-run ships the run instead of
dropping it. Written twice, they drift; written once, the third adapter costs a
fixture.

WHAT MAKES THIS A GATE RATHER THAN A CHECKLIST
----------------------------------------------

A conformance check that asserts spans EXIST is worth nothing, and so is one
that compares a set of parent/child edges. Both pass on a fully collapsed
causal tree — every intermediate span present, every one of them the root's
child, every edge reading `unit_active` at confidence 1.0 with no marker, one
trace, correct counts — which is precisely what an adapter produces when it
wraps a framework's producer instead of its consumer, or when a release stops
copying the context at task submit. That tree is byte-identical to a healthy
one in every aggregate a suite might compute.

So the tree is asserted as PATHS OF SPAN IDS, `a` is the parent of `b` is the
parent of `c`, by identity, per node. And `check_a_collapsed_tree_fails_this_suite`
builds the collapse from the subject's OWN run and watches the tier half wave
it through while the id half catches it. That check runs for every adapter
wired in, so no subject is certified by a gate nobody has seen fail on its own
workload.

The suite also refuses a subject whose declared tree could not exhibit a
collapse at all — fewer than two chains, or a chain shallower than three, or
two chains sharing an intermediate. That is
`check_the_declared_tree_is_one_a_collapse_could_break`, and it exists because
the cheapest way to pass a tree gate is to declare a tree with no depth in it.

WIRING AN ADAPTER IN

    @pytest.fixture
    def subject():
        return AdapterSubject(...)

    @pytest.mark.parametrize("check", AdapterConformanceSuite.CHECKS)
    def test_conformance(subject, check):
        AdapterConformanceSuite(subject).run(check)

One test per check per adapter, so a failure names the invariant that broke
rather than "conformance".
"""

from __future__ import annotations

import ast
import importlib
import inspect
from collections.abc import Sequence

from .._adapters import _DETECT_PACKAGES, _make_adapter
from .._assembly import Limitation, ParentSource
from .._enums import AdapterName
from .harness import (
    AdapterSubject,
    SpanNode,
    collapse_onto_root,
    exactly_one,
    installed_adapter,
    never_installed,
    parent_name_of,
    read_spans,
)

#: The methods an adapter's causal surface may open a span through. `rejoin` is
#: the one where a framework identifier influences the shape of the tree, and it
#: is a separate name precisely so that one grep is the complete list — the same
#: reason this rule can be written at all.
_OPENERS = frozenset({"enter", "open_run", "rejoin"})

#: Markers that say the EDGE is not what it looks like. A healthy run may carry
#: plenty of others — the Agent SDK's root always reports that a subprocess has
#: no transport timing — so the tier half asks about these four and not about an
#: empty set, which would be a rule the first honest observation breaks.
#:
#: Named `_EDGE_MARKERS` so the limitation census can SEE it. A name without
#: `marker` in it would put four member references in a slot the scanner is not
#: looking at, which is a hiding place whether or not anything is hidden there.
_EDGE_MARKERS = (
    Limitation.PARENT_UNRESOLVED,
    Limitation.UNIT_INFERRED_SOLE,
    Limitation.CORRELATION_CONFLICT,
    Limitation.INSTRUMENTATION_DEGRADED,
)


class AdapterConformanceSuite:
    """Every shared invariant, one method each. See the module docstring."""

    CHECKS: tuple[str, ...] = (
        "check_the_declared_tree_is_one_a_collapse_could_break",
        "check_the_adapter_names_itself_after_its_enum_member",
        "check_install_replaces_every_seam_and_uninstall_restores_it",
        "check_a_restored_seam_captures_nothing",
        "check_the_workload_ships_one_read_tree",
        "check_the_causal_chain_holds_by_span_id",
        "check_a_collapsed_tree_fails_this_suite",
        "check_every_span_site_declares_its_placement",
        "check_close_units_ships_the_open_run_and_stays_installed",
        "check_uninstall_ships_the_open_run_and_never_raises_into_the_host",
    )

    def __init__(self, subject: AdapterSubject) -> None:
        self._subject = subject

    def run(self, check: str) -> None:
        """Run one check by name. Refuses a name that is not in `CHECKS`."""
        assert check in self.CHECKS, (
            f"{check!r} is not a conformance check; expected one of {self.CHECKS}"
        )
        getattr(self, check)()

    # -- the subject itself --------------------------------------------------

    def check_the_declared_tree_is_one_a_collapse_could_break(self) -> None:
        """A gate whose workload cannot fail it is a gate that measures nothing.

        Three properties, and each rules out a way of declaring a tree that
        passes trivially:

          * DEPTH. A chain of two is a root and a child, and collapsing it onto
            the root leaves it unchanged — so the id half and the tier half
            would agree on every workload and the suite would be blind for that
            adapter.
          * BREADTH. One chain lets "the parent is whatever was opened most
            recently" be right by construction. Two chains with DISTINCT
            intermediates need two different answers at once, which no
            process-global register can give.
          * ONE ROOT. Every chain hangs off the same span, or the declared tree
            is two trees and the suite's own `collapse_onto_root()` has no anchor.
        """
        subject = self._subject
        chains = subject.chains
        assert len(chains) >= 2, (
            "a subject must declare at least two chains: with one, 'the parent is "
            "whatever this adapter opened last' is right by construction and the "
            "tree assertion tests the tie-break rather than the mechanism"
        )
        for chain in chains:
            assert len(chain) >= 3, (
                f"chain {chain} is shallower than root -> node -> leaf; collapsing it "
                "onto the root changes nothing, so this suite could not fail on it"
            )
            assert len(set(chain)) == len(chain), f"chain {chain} names a span twice"
            assert chain[0] == subject.root, (
                f"chain {chain} hangs off {chain[0]!r}, not off {subject.root!r}; "
                "one declared tree has one root"
            )
        seconds = [chain[1] for chain in chains]
        assert len(set(seconds)) == len(seconds), (
            f"two chains share the intermediate span {seconds}; distinct intermediates "
            "are what stop a single 'current parent' answer from being right twice"
        )

    def check_the_adapter_names_itself_after_its_enum_member(self) -> None:
        """The registration square: enum, factory, detection and `name()` agree.

        All four, because each pair alone leaves a live failure. An adapter the
        factory cannot build is one `init()` silently never installs. One with
        no detection entry never installs itself for a host that configured
        nothing, which is every host. And one whose `name()` disagrees with its
        enum member is filed by the registry under a name nothing else uses, so
        `sole_live(owner=...)` filters for units all owned by somebody else.
        """
        subject = self._subject
        adapter = subject.factory()
        assert adapter.name() == subject.name
        member = AdapterName(subject.name)  # raises unless the value is a member
        made = _make_adapter(member)
        assert made is not None, f"the adapter factory has no entry for {member}"
        assert type(made) is type(adapter)
        assert _DETECT_PACKAGES.get(member) == subject.detect_package, (
            f"{member} is not registered for auto-detection under {subject.detect_package!r}"
        )

    # -- install / uninstall -------------------------------------------------

    def check_install_replaces_every_seam_and_uninstall_restores_it(self) -> None:
        """Identity in both directions, plus the two idempotence cases.

        `is not` on the way in: a seam still holding its original is a patch
        that silently did nothing, which is the failure mode an install-level
        self-check reports as success. `is` on the way out: `PatchSet` restores
        with a plain `setattr`, so a restore that merely produced an EQUAL
        object would weld wardex's wrapper on for the life of the process.

        The second uninstall emitting no span is the other half of the same
        fact: a teardown that re-drained a table it had already drained would
        ship a second copy of every run it closed.
        """
        subject = self._subject
        before = dict(subject.seams())
        assert before, "a subject must declare the seams its adapter patches"
        with installed_adapter(subject.factory) as live:
            during = dict(subject.seams())
            assert set(during) == set(before), "the seam snapshot changed shape"
            silent = [key for key in before if during[key] is before[key]]
            assert not silent, (
                f"install left {silent} holding the framework's own attribute; a seam "
                "that was not replaced is a patch that did nothing"
            )
            # A second install must be a no-op, not a second layer of wrappers.
            live.adapter.install(live.client, live.ctx)
            assert all(subject.seams()[key] is during[key] for key in before), (
                "installing twice replaced a seam a second time; the inner wrapper "
                "is then unreachable and its restore is lost"
            )

            live.teardown()
            self._assert_restored(before)

            shipped = len(live.spans)
            live.teardown()
            live.adapter.uninstall()
            assert len(live.spans) == shipped, "a second uninstall shipped another span"
            self._assert_restored(before)

    def check_a_restored_seam_captures_nothing(self) -> None:
        """The zero point: every number this suite reports is the adapter's doing.

        Without it the tree checks could be measuring spans some other installed
        adapter, or an earlier test's leaked patch, happens to emit for the same
        workload — and every count would still come out right.

        The client is driven through a LIVE install FIRST and the silence is
        asserted on that same object afterwards. A fresh client nobody wired up
        is empty whatever the adapter does: the assertion would read as if it
        measured something and could not fail.
        """
        subject = self._subject
        before = dict(subject.seams())
        with installed_adapter(subject.factory) as live:
            subject.workload(live)
            assert live.spans, "this client really does receive spans"
            shipped = len(live.spans)

            live.teardown()
            self._assert_restored(before)

            subject.workload(never_installed(subject.factory, live.client))
            assert len(live.spans) == shipped, (
                "a restored seam emitted spans; either the uninstall did not restore "
                "it, or a patch leaked out of an earlier install and this suite is "
                "not measuring what it thinks"
            )

    def _assert_restored(self, before: dict) -> None:
        after = dict(self._subject.seams())
        welded = [key for key in before if after[key] is not before[key]]
        assert not welded, (
            f"uninstall left {welded} holding something other than the framework's own "
            "attribute; the restore is by identity or it is not a restore"
        )

    # -- the causal tree -----------------------------------------------------

    def check_the_workload_ships_one_read_tree(self) -> None:
        """The TIER half: one trace, one root, and every edge wardex READ.

        Blind to a collapse by construction — see the next two checks — and
        kept anyway, because it is the half that catches the failures a
        collapse is not: a run entry nobody wrapped (the tree shatters into one
        trace per captured call), a guess that quietly replaced a read (0.5 and
        `UNIT_INFERRED_SOLE`), and a wardex bug that degraded a span into
        existence.
        """
        self._assert_tiers(self._observe())

    def check_the_causal_chain_holds_by_span_id(self) -> None:
        """The half that matters most: each declared path, asserted by id.

        THAT parent's id, not "some ancestor's". Everything else about a
        collapsed tree is right.
        """
        self._assert_chains(self._observe())

    def check_a_collapsed_tree_fails_this_suite(self) -> None:
        """The gate's own proof of life, on this adapter's own workload.

        The collapse is built from the run that actually happened by moving
        every edge onto the root and touching nothing else, so the tier half is
        provably unchanged — same tiers, same confidences, same markers, same
        trace, same counts — and the demonstration cannot be an artefact of a
        different run.

        A suite that only claimed to catch a collapse would be exactly as
        convincing as the tier half is on its own, which is not at all.
        """
        nodes = self._observe()
        flattened = collapse_onto_root(nodes, root=self._subject.root)
        self._assert_tiers(flattened)  # every tier assertion still passes

        for chain in self._subject.chains:
            leaf = exactly_one(flattened, chain[-1])
            assert parent_name_of(flattened, leaf) == self._subject.root, (
                "the collapse this check builds must actually be a collapse"
            )
        try:
            self._assert_chains(flattened)
        except AssertionError:
            return
        raise AssertionError(
            "the id half passed on a fully collapsed tree, so it is not testing "
            "the causal structure of anything. Every span was reparented onto "
            f"{self._subject.root!r} and the chains still held."
        )

    def _observe(self) -> tuple[SpanNode, ...]:
        subject = self._subject
        with installed_adapter(subject.factory) as live:
            subject.workload(live)
            return read_spans(live.spans)

    def _assert_tiers(self, nodes: Sequence[SpanNode]) -> None:
        subject = self._subject
        declared = {name for chain in subject.chains for name in chain}
        assert sorted(node.name for node in nodes) == sorted(declared), (
            "the shipped spans are not exactly the declared tree; an extra span is "
            "one the subject does not describe, and a missing one is a seam that "
            "did not fire"
        )
        assert len({node.trace_id for node in nodes}) == 1, "one run is one trace"

        roots = [node for node in nodes if node.parent_id is None]
        assert [node.name for node in roots] == [subject.root], (
            f"expected exactly one trace root named {subject.root!r}, got "
            f"{[node.name for node in roots]}; a second root is the shattered-run "
            "shape a site that did not declare its placement produces"
        )
        assert roots[0].strategy is ParentSource.TRACE_ROOT
        assert roots[0].confidence == 1.0

        for node in nodes:
            if node.name == subject.root:
                continue
            assert node.strategy is ParentSource.UNIT_ACTIVE, (
                f"{node.name}: the edge is {node.strategy}, not a scope wardex read"
            )
            assert node.confidence == 1.0, f"{node.name}: confidence {node.confidence}"
        for node in nodes:
            reported = [marker for marker in _EDGE_MARKERS if marker in node.limitations]
            assert not reported, (
                f"{node.name} carries {reported}; a healthy run's edges are read, not "
                "guessed, and nothing about them was lost"
            )

    def _assert_chains(self, nodes: Sequence[SpanNode]) -> None:
        for chain in self._subject.chains:
            for parent, child in zip(chain, chain[1:], strict=False):
                above = exactly_one(nodes, parent)
                below = exactly_one(nodes, child)
                assert below.parent_id == above.span_id, (
                    f"{child!r} is parented to {parent_name_of(nodes, below)!r}, not to {parent!r}"
                )

    # -- placement -----------------------------------------------------------

    def check_every_span_site_declares_its_placement(self) -> None:
        """Read as SOURCE, because there is no runtime moment to observe it at.

        `Placement` is a property of a PATCH SITE — "may this site legitimately
        begin a trace" — and the failure it catches is "the adapter did not wrap
        the framework's run entry", which is a bug in the adapter's shape rather
        than a condition of one invocation. A computed value would let every
        site paper over it independently, so the rule is the LITERAL: an enum
        member spelled at the call.

        Nothing else fails loudly. A nested site that becomes a trace root
        produces spans that are byte-identical to legitimate ones — full
        confidence, no marker — and the run is reported as as many genuine
        traces as it happened to make calls.
        """
        subject = self._subject
        module = importlib.import_module(subject.module)
        tree = ast.parse(inspect.getsource(module))
        sites = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _OPENERS
        ]
        assert sites, (
            f"{subject.module} opens no span through the adapter surface, so this "
            "rule would assert nothing; every span an adapter emits goes through it"
        )
        bad = [site.lineno for site in sites if not _declares_placement(site)]
        assert not bad, (
            f"{subject.module}: a span site does not declare `placement=Placement.X` "
            f"as a literal, at line(s) {bad}"
        )

    # -- shutdown ------------------------------------------------------------

    def check_close_units_ships_the_open_run_and_stays_installed(self) -> None:
        """The signal-handler path. The adapter is NOT uninstalled by it.

        That is the whole difference from a teardown, and it is not cosmetic:
        the process may or may not be about to die, and an adapter that stopped
        capturing because a shutdown signal arrived would stop capturing for a
        program that then carries on.
        """
        subject = self._subject
        before = dict(subject.seams())
        with installed_adapter(subject.factory) as live:
            stalled = subject.stall(live)
            registry = live.registry
            assert registry is not None
            registry.close_units_all(marker=Limitation.UNIT_INTERRUPTED)

            self._assert_shipped(live, stalled.root, Limitation.UNIT_INTERRUPTED)
            assert registry.is_installed(subject.name), "close_units must not uninstall"
            assert all(subject.seams()[key] is not before[key] for key in before), (
                "close_units restored a seam; the adapter must still be installed"
            )
            stalled.resume()  # the host keeps going; this must not raise into it

    def check_uninstall_ships_the_open_run_and_never_raises_into_the_host(self) -> None:
        """The atexit path, and the run that never finished.

        Until a teardown closed its units, a run interrupted mid-flight left
        NOTHING — not a truncated span, not a marked one. What ships now says
        why it is short, and the host's own generator finishes afterwards
        without seeing an exception wardex invented.
        """
        subject = self._subject
        with installed_adapter(subject.factory) as live:
            stalled = subject.stall(live)
            live.teardown()
            self._assert_shipped(live, stalled.root, Limitation.ADAPTER_UNINSTALLED)
            stalled.resume()

    def _assert_shipped(self, live, root: str, marker: Limitation) -> None:  # noqa: ANN001
        shipped = [node for node in read_spans(live.spans) if node.name == root]
        assert len(shipped) == 1, (
            f"expected exactly one {root!r} to ship for an interrupted run, got {len(shipped)}"
        )
        assert marker in shipped[0].limitations, (
            f"{root!r} shipped without {marker}, so nothing downstream can tell it "
            f"from a run that ended normally; it carries {list(shipped[0].limitations)}"
        )


def _declares_placement(site: ast.Call) -> bool:
    """`placement=Placement.SOMETHING`, spelled out at the call."""
    for keyword in site.keywords:
        if keyword.arg != "placement":
            continue
        value = keyword.value
        return (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "Placement"
        )
    return False


__all__ = ["AdapterConformanceSuite"]
