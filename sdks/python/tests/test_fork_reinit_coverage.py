"""Fork-reset coverage — the guard that makes the NEXT holder wire itself.

The bug class: someone adds a process-global mutable holder — a lock, an
Event, a connection/session table — and nobody plumbs it into the fork
child's reset. The state then crosses `os.fork()` carrying the parent's
world: a held lock hangs the child's teardown, a stale table mis-attributes
the child's traffic. The first design of the fork work fell into this class
three times itself (the Runtime lock, the PatchSet locks, the propagation
module's lock), which is why the guard is LOCK-GRANULAR: a class-level rule
("the class has a reset method") passes a class whose reset forgets one of
its own locks.

Two halves, deliberately redundant in opposite directions:

  DECLARATION (AST): every lock/Event/Condition allocation site and every
  empty-container holder site in `src/` must either be NAMED by a fork-reset
  path in its own module (directly, or through a helper that path calls) or
  sit in `_FORK_EXEMPT` beside a one-line reason. This is what fails the
  commit that ADDS an unwired holder — including one in a brand-new module
  with no reset function at all.

  WIRING (graph-walk): declaring a reset is not reaching it. A real install
  with real traffic forks, and the child walks the live object graph from
  `runtime()`: every table it can reach must be empty (or allowlisted with a
  reason), every lock acquirable without blocking. This is what fails the
  commit whose declared reset never gets CALLED.

Both scans read source/objects, not a hand-kept list of files — the person
who forgot to wire a holder is the same person who would have forgotten to
extend the list.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import threading
from collections import deque

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "wardex_sdk"

#: Function names that ARE fork-reset paths. `_at_fork_reinit` is the
#: protocol; the other three are the special entry points the design names
#: (the runtime's own hook, the diagnostics' no-acquire reset, and the
#: receiver's child-safe teardown).
_RESET_FNS = frozenset(
    {
        "_at_fork_reinit",
        "after_in_child",
        "diag_reset_for_new_process",
        "close_inherited_after_fork",
    }
)

#: (file, attribute) -> why this holder does NOT need a fork-reset mention.
#: Every entry is a decision record; an entry without a real reason is a bug
#: in review, not in this table's mechanism.
_FORK_EXEMPT: dict[tuple[str, str], str] = {
    # -- per-draft transients: born and die inside one assembly call; a fork
    #    cannot observe them across its boundary in SDK-owned state.
    ("_assembly/_builder.py", "_markers"): "per-draft transient, dies with its frame",
    ("_assembly/_builder.py", "_extra"): "per-draft transient, dies with its frame",
    ("_assembly/_builder.py", "_links"): "per-draft transient, dies with its frame",
    ("_assembly/_builder.py", "_events"): "per-draft transient, dies with its frame",
    ("_assembly/_snapshot.py", "_input_refs"): "per-draft transient, dies with its frame",
    ("_assembly/_snapshot.py", "_attrs"): "per-draft transient, dies with its frame",
    # -- per-unit object state: the registry-level clear drops the REGISTRY's
    #    references; a unit the host still holds is the host's deliberate
    #    continuation (design §3.10) and ships under the child's pid.
    ("_assembly/_units.py", "_children"): "per-unit state; registry clear + §3.10 judgement",
    ("_assembly/_units.py", "_alias_keys"): "per-unit state; registry clear + §3.10 judgement",
    ("_assembly/_units.py", "_claims"): "per-unit state; registry clear + §3.10 judgement",
    ("_assembly/_units.py", "_open"): "per-unit state; registry clear + §3.10 judgement",
    ("_assembly/_units.py", "_remembered"): "per-unit state; registry clear + §3.10 judgement",
    # -- state that rides an entry another reset drops wholesale.
    ("_interceptors/_close_hook.py", "hooks"): "per-entry; dropped with the entry by clear()",
    ("_adapters/_anthropic_names.py", "tools"): "per-handle; dropped with the handle",
    ("_interceptors/_trackers.py", "_resp_marks"): (
        "per-connection tracker state; rides _ConnectionState, dropped by the seam's clear"
    ),
    ("_interceptors/_trackers.py", "_latch"): (
        "per-connection tracker state; rides _ConnectionState, dropped by the seam's clear"
    ),
    # -- the MCP stdio interceptor: its state rides per-stream closures the
    #    fork either carries validly or never touches; the interceptor object
    #    itself holds no table (design §1.2 row I).
    ("_interceptors/_mcp_stdio.py", "_latch"): (
        "per-_ProcState, held by per-stream closures; inert unless the child "
        "drives the parent's pipes, where the existing caps/markers apply"
    ),
    # -- install records: kept across the fork BY DESIGN (I-fork-4). The
    #    patches they describe crossed in the memory image and still work;
    #    forgetting them would strand the child's teardown.
    ("_adapters/_registry.py", "_installed"): "install record, kept (I-fork-4)",
    ("_adapters/_registry.py", "_contexts"): "install record, kept (I-fork-4)",
    ("_interceptors/_registry.py", "_installed"): "install record, kept (I-fork-4)",
    ("context/_inject.py", "_installed"): "patch record, kept (I-fork-4)",
    ("_assembly/_patchset.py", "_patches"): "restore records, kept (I-fork-4)",
    # -- signal dispositions cross the fork with the signal table itself; the
    #    child's handlers are as installed as the parent's (design §3.8).
    ("_runtime.py", "_prev_handlers"): "signal dispositions survive fork by design (§3.8)",
    # -- the owner-typed scan counts a lock at its allocation site AND at the
    #    site holding the instance; this is the one same-module double count.
    ("_runtime.py", "_RUNTIME"): (
        "the singleton whose own after_in_child replaces the lock this site "
        "would otherwise strand; the allocation site inside Runtime is the "
        "covered one, and this name is the object that owns the reset"
    ),
}


# ==========================================================================
# scanners
# ==========================================================================

_LOCK_FACTORIES = frozenset({"Lock", "RLock", "Event", "Condition"})


def _is_lock_call(value: ast.expr) -> bool:
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    if isinstance(func, ast.Attribute):
        return func.attr in _LOCK_FACTORIES
    return isinstance(func, ast.Name) and func.id in _LOCK_FACTORIES


def _lock_owning_classes(trees: list[ast.Module]) -> frozenset[str]:
    """Every src class whose `__init__` allocates a threading primitive.

    Computed from the same AST pass, not hand-kept: `PatchSet` is the case
    that motivated this (`self._patches = PatchSet(...)` matched neither a
    lock literal nor an empty container, so four unwired PatchSet holders
    passed the declaration scan while their locks crossed the fork), and a
    list would rot the day the next lock-owning helper type lands. One
    level deep on purpose: a class owning such a CLASS shows up because its
    own `__init__` contains the owner call, which `_holder_sites` records
    as a lock site inside that class's module.
    """
    owners: set[str] = set()
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            inits = [
                n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            ]
            for init in inits:
                for sub in ast.walk(init):
                    value = getattr(sub, "value", None)
                    if (
                        isinstance(sub, (ast.Assign, ast.AnnAssign))
                        and value is not None
                        and _is_lock_call(value)
                    ):
                        owners.add(node.name)
    return frozenset(owners)


def _is_lock_owner_call(value: ast.expr, owners: frozenset[str]) -> bool:
    """A constructor call of a type whose instances own a threading lock."""
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    if isinstance(func, ast.Attribute):
        return func.attr in owners
    return isinstance(func, ast.Name) and func.id in owners


def _is_empty_container(value: ast.expr) -> bool:
    if isinstance(value, ast.Dict) and not value.keys:
        return True
    if isinstance(value, (ast.List, ast.Set)) and not getattr(value, "elts", None):
        return True
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in {"dict", "deque", "set", "list"}
        and not value.args
        and not value.keywords
    )


def _holder_sites(tree: ast.Module, owners: frozenset[str] = frozenset()) -> list[tuple[str, str]]:
    """Every (attr, kind) holder allocation in one module.

    Attribute targets (`self.x = ...`, `obj.x = ...`) and module-global
    names; bare locals are frames, not process state. `kind` is "lock" or
    "table". A constructor call of a lock-owning SDK type (`owners`) is a
    lock site too — the lock exists just as surely when `PatchSet()`
    allocates it as when `threading.RLock()` does.
    """
    sites: list[tuple[str, str]] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.depth = 0  # function nesting; 0 = module scope

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

        def _record(self, targets: list[ast.expr], value: ast.expr | None) -> None:
            if value is None:
                return
            if _is_lock_call(value) or _is_lock_owner_call(value, owners):
                kind = "lock"
            elif _is_empty_container(value):
                kind = "table"
            else:
                return
            for t in targets:
                if isinstance(t, ast.Attribute):
                    sites.append((t.attr, kind))
                elif isinstance(t, ast.Name) and self.depth == 0:
                    sites.append((t.id, kind))

        def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
            self._record(node.targets, node.value)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
            self._record([node.target], node.value)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return sites


def _reset_mentioned_names(tree: ast.Module) -> set[str]:
    """Every attribute/name a module's fork-reset paths can reach, one hop deep.

    Direct mentions inside a `_RESET_FNS` function count; so do mentions
    inside any same-module function, method or class constructor that such a
    function CALLS (`CloseRegistry.clear` mentioning `_entries`, `_SpanBuffer`
    mentioning `spans`). One hop, because that is what the code uses; a chain
    deeper than that deserves to fail and be flattened.
    """
    functions: dict[str, list[ast.AST]] = {}
    classes: dict[str, ast.ClassDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.setdefault(node.name, []).append(node)
        elif isinstance(node, ast.ClassDef):
            classes[node.name] = node

    def mentions(nodes: list[ast.AST]) -> set[str]:
        out: set[str] = set()
        for fn in nodes:
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Attribute):
                    out.add(sub.attr)
                elif isinstance(sub, ast.Name):
                    out.add(sub.id)
        return out

    def called_names(nodes: list[ast.AST]) -> set[str]:
        out: set[str] = set()
        for fn in nodes:
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Call):
                    if isinstance(sub.func, ast.Attribute):
                        out.add(sub.func.attr)
                    elif isinstance(sub.func, ast.Name):
                        out.add(sub.func.id)
        return out

    roots: list[ast.AST] = []
    for name in _RESET_FNS:
        roots.extend(functions.get(name, []))
    if not roots:
        return set()
    reached = mentions(roots)
    for called in called_names(roots):
        if called in functions:
            reached |= mentions(functions[called])
        if called in classes:
            inits = [
                n
                for n in classes[called].body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            ]
            reached |= mentions(list(inits))
    return reached


def _scan(src: pathlib.Path) -> list[str]:
    """Every holder site not covered by a reset path or an exemption."""
    violations: list[str] = []
    parsed: list[tuple[str, ast.Module]] = []
    for path in sorted(src.rglob("*.py")):
        rel = path.relative_to(src).as_posix()
        if rel.startswith("testing/"):
            continue  # test doubles hold a test's state, not the SDK's
        parsed.append((rel, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))))
    # Owners are computed over the whole root FIRST: the class that owns the
    # lock (`PatchSet`) and the module that instantiates it are usually not
    # the same file.
    owners = _lock_owning_classes([tree for _, tree in parsed])
    for rel, tree in parsed:
        sites = _holder_sites(tree, owners)
        if not sites:
            continue
        covered = _reset_mentioned_names(tree)
        for attr, kind in sites:
            if attr in covered:
                continue
            if (rel, attr) in _FORK_EXEMPT:
                continue
            violations.append(f"{rel}: {attr} ({kind})")
    return violations


# ==========================================================================
# declaration coverage
# ==========================================================================


def test_every_lock_and_table_site_is_named_by_a_fork_reset_path():
    violations = _scan(_SRC)
    assert violations == [], (
        "process-global holder(s) with no fork-reset wiring and no recorded "
        "exemption:\n  " + "\n  ".join(violations) + "\n"
        "A lock that crosses os.fork() held hangs the child; a table that "
        "crosses it populated mis-attributes the child's traffic. Either name "
        "the attribute in the owning module's _at_fork_reinit (replace locks, "
        "never acquire; drop tables, never emit) or add a _FORK_EXEMPT entry "
        "WITH its reason."
    )


def test_the_scan_can_see_an_unwired_lock(tmp_path):
    """A source-scanning guard is worth its predicate: feed it a module with
    an unwired lock, an unwired table, and a wired lock, and watch it flag
    exactly the unwired two — including through a one-hop helper."""
    fake = tmp_path / "src"
    (fake / "sub").mkdir(parents=True)
    (fake / "sub" / "mod.py").write_text(
        "import threading\n"
        "from collections import deque\n"
        "class Queue:\n"
        "    def __init__(self):\n"
        "        self._lock = threading.Lock()\n"
        "        self._cv = threading.Condition()\n"
        "        self._jobs = deque()\n"
        "        self._wired = threading.RLock()\n"
        "        self._table = {}\n"
        "    def _drop_table(self):\n"
        "        self._table.clear()\n"
        "    def _at_fork_reinit(self):\n"
        "        self._wired = threading.RLock()\n"
        "        self._drop_table()\n",
        encoding="utf-8",
    )
    flagged = _scan(fake)
    assert sorted(flagged) == [
        "sub/mod.py: _cv (lock)",
        "sub/mod.py: _jobs (table)",
        "sub/mod.py: _lock (lock)",
    ], flagged


def test_the_scan_can_see_an_unwired_indirectly_owned_lock(tmp_path):
    """The row-Q regression shape: `self._patches = PatchSet(...)` is a lock
    allocation as surely as `threading.RLock()` is, but it matched neither
    scanner predicate — which is how four unwired PatchSet holders shipped.
    Plant one unwired and one wired indirect holder (owner class defined in
    a DIFFERENT module, as in the real tree) and watch exactly the unwired
    one get flagged."""
    fake = tmp_path / "src"
    fake.mkdir()
    (fake / "patchset.py").write_text(
        "import threading\n"
        "class FakePatchSet:\n"
        "    def __init__(self, owner):\n"
        "        self._lock = threading.RLock()\n"
        "    def _at_fork_reinit(self):\n"
        "        self._lock = threading.RLock()\n",
        encoding="utf-8",
    )
    (fake / "holders.py").write_text(
        "from patchset import FakePatchSet\n"
        "class Unwired:\n"
        "    def __init__(self):\n"
        "        self._patches = FakePatchSet('unwired')\n"
        "class Wired:\n"
        "    def __init__(self):\n"
        "        self._pset = FakePatchSet('wired')\n"
        "    def _at_fork_reinit(self):\n"
        "        self._pset._at_fork_reinit()\n",
        encoding="utf-8",
    )
    flagged = _scan(fake)
    assert flagged == ["holders.py: _patches (lock)"], flagged


def test_a_module_with_no_reset_path_at_all_is_flagged(tmp_path):
    """The exact shape a NEW subsystem lands in: state, no reset function.
    The scan may not read 'no reset path' as 'nothing to check'."""
    fake = tmp_path / "src"
    fake.mkdir()
    (fake / "fresh.py").write_text(
        "import threading\n_pending = {}\n_guard = threading.Lock()\n",
        encoding="utf-8",
    )
    flagged = _scan(fake)
    assert sorted(flagged) == ["fresh.py: _guard (lock)", "fresh.py: _pending (table)"], flagged


def test_every_exemption_still_names_a_real_site():
    """An exemption for code that is gone is a hole the next holder of the
    same name walks through unexamined."""
    live: set[tuple[str, str]] = set()
    parsed: list[tuple[str, ast.Module]] = []
    for path in sorted(_SRC.rglob("*.py")):
        rel = path.relative_to(_SRC).as_posix()
        if rel.startswith("testing/"):
            continue
        parsed.append((rel, ast.parse(path.read_text(encoding="utf-8"))))
    owners = _lock_owning_classes([tree for _, tree in parsed])
    for rel, tree in parsed:
        for attr, _kind in _holder_sites(tree, owners):
            live.add((rel, attr))
    stale = sorted(k for k in _FORK_EXEMPT if k not in live)
    assert stale == [], f"exemptions for sites that no longer exist: {stale}"


# ==========================================================================
# wiring coverage — the graph walk
# ==========================================================================

_RLOCK_TYPE = type(threading.RLock())
_LOCK_TYPE = type(threading.Lock())

#: Attribute names whose containers may be POPULATED in a fresh fork child,
#: each with its reason. Everything else the walk reaches must be empty.
_WALK_ALLOWED_POPULATED: dict[str, str] = {
    "_installed": "install/patch records are kept (I-fork-4)",
    "_contexts": "install records are kept (I-fork-4)",
    "_patches": "restore records are kept (I-fork-4)",
    "_prev_handlers": "signal dispositions survive fork by design (§3.8)",
    "_limits": "resolved config projection — immutable by convention",
    "limits": "resolved config projection on AdapterContext — immutable by convention",
    "_ws_kwargs": "resolved config projection — immutable by convention",
    "_proc_limits": "prebuilt limits kwargs — immutable projection, reset not needed",
    "_reset_at_fork_ids": "the fork latch: deliberately populated BY the reset",
    "_counts": "the child's own post-reset diagnostics (fork_child_reinit)",
    "_afterfork_registry": "multiprocessing's own machinery, not SDK state",
}


def _walk_child_graph() -> dict:
    """Runs IN THE FORK CHILD: walk the SDK object graph, report violations.

    Returns counts too, so the parent can assert the walk actually SAW a
    meaningful population — a walk that reaches nothing passes forever.
    """
    from wardex_sdk import _runtime

    seen: set[int] = set()
    violations: list[str] = []
    locks_checked = 0
    tables_checked = 0

    def is_sdk(obj: object) -> bool:
        module = getattr(type(obj), "__module__", "") or ""
        return module.startswith("wardex_sdk") and not module.startswith("wardex_sdk.testing")

    def check_lock(path: str, lock: object) -> None:
        nonlocal locks_checked
        locks_checked += 1
        acquired = lock.acquire(blocking=False)  # type: ignore[attr-defined]
        if acquired:
            lock.release()  # type: ignore[attr-defined]
        else:
            violations.append(f"{path}: lock not acquirable (inherited held)")

    def check_container(path: str, value: object) -> None:
        nonlocal tables_checked
        tables_checked += 1
        if not value:
            return
        name = path.rsplit(".", 1)[-1]
        if name not in _WALK_ALLOWED_POPULATED:
            violations.append(f"{path}: populated after fork ({len(value)} entries)")  # type: ignore[arg-type]

    def visit(path: str, obj: object, depth: int) -> None:
        if depth > 8 or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, (_RLOCK_TYPE, _LOCK_TYPE)):
            check_lock(path, obj)
            return
        if isinstance(obj, (dict, list, set, frozenset, deque, tuple)):
            if isinstance(obj, (dict, list, set, deque)):
                check_container(path, obj)
            values = obj.values() if isinstance(obj, dict) else obj
            for i, item in enumerate(values):
                if is_sdk(item):
                    visit(f"{path}[{i}]", item, depth + 1)
            return
        if not is_sdk(obj):
            return
        state: dict[str, object] = {}
        instance_dict = getattr(obj, "__dict__", None)
        if isinstance(instance_dict, dict):
            state.update(instance_dict)
        for klass in type(obj).__mro__:
            for slot in getattr(klass, "__slots__", ()):
                try:
                    state[slot] = object.__getattribute__(obj, slot)
                except AttributeError:
                    continue
        for name, value in state.items():
            visit(f"{path}.{name}", value, depth + 1)

    runtime = _runtime.runtime()
    visit("runtime", runtime, 0)
    # The module-global holders are not reachable from the runtime object.
    import sys as _sys

    for module_name, attrs in {
        "wardex_sdk._interceptors._conn_timing": ("_shared_store",),
        "wardex_sdk._interceptors._close_hook": ("_registry",),
        "wardex_sdk._assembly._diag": ("counters", "_REPORTED", "_REPORT_LOCK"),
        "wardex_sdk.context._inject": ("_install_lock", "_installed", "_patches"),
    }.items():
        module = _sys.modules.get(module_name)
        if module is None:
            continue
        for attr in attrs:
            value = getattr(module, attr, None)
            if value is None:
                continue
            if isinstance(value, (_RLOCK_TYPE, _LOCK_TYPE)):
                check_lock(f"{module_name}.{attr}", value)
            elif isinstance(value, (dict, list, set, deque)):
                check_container(f"{module_name}.{attr}", value)
            elif is_sdk(value):
                visit(f"{module_name}.{attr}", value, 0)
    return {
        "violations": violations,
        "locks_checked": locks_checked,
        "tables_checked": tables_checked,
    }


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
@pytest.mark.filterwarnings(
    "ignore:This process.*is multi-threaded, use of fork:DeprecationWarning"
)
def test_fork_walk_reaches_every_holder():
    """Install everything installable, fill every table with real traffic,
    fork, and let the CHILD prove the reset reached what the declaration
    promised: tables empty (or allowlisted with a reason), locks free."""
    import wardex_sdk as wardex
    from wardex_sdk import _hub
    from wardex_sdk._config import BatchingConfig
    from wardex_sdk._enums import CaptureMode
    from wardex_sdk._interceptors._conn_timing import shared_timing_store
    from wardex_sdk._interceptors._registry import get_registry as interceptors
    from wardex_sdk.testing import RecordingTransport

    wardex.init(
        transport=RecordingTransport(),
        intercept=True,
        capture_mode=CaptureMode.ALL,
        batching=BatchingConfig(flush_interval=3600.0, flush_on_signals=False),
    )
    try:
        # Fill what real traffic fills: a tracked connection on the ssl seam
        # (which also registers a close hook), a timing-store slot, buffered
        # spans, an adapter session with a live unit, a server handle.
        seam = interceptors()._installed["ssl"]

        class _Sock:
            def selected_alpn_protocol(self):
                return "http/1.1"

            def getpeername(self):
                return ("127.0.0.1", 443)

            def fileno(self):
                return -1

        sock = _Sock()
        seam._on_request_bytes(
            sock, b"POST /v1/messages HTTP/1.1\r\nHost: h\r\nContent-Length: 0\r\n\r\n"
        )
        shared_timing_store().set_connect(5, 1.25)

        from wardex_sdk._adapters._anthropic_agent_sdk import AnthropicAgentSdkAdapter
        from wardex_sdk._adapters._registry import get_registry as adapters

        # init() auto-detects installed frameworks, so on a machine with the
        # Agent SDK the adapter is ALREADY installed; only build one where it
        # is not (CI without the framework).
        adapter = adapters()._installed.get("anthropic_agent_sdk")
        if adapter is None:
            adapter = AnthropicAgentSdkAdapter()
            adapters().install(adapter, _hub.get_client())
        assembler = adapter._assembler
        assert assembler is not None
        assembler.on_outbound(
            1,
            json.dumps(
                {"type": "user", "session_id": "s-1", "message": {"role": "user", "content": "x"}}
            ),
        )
        assembler.on_inbound(
            1,
            {"type": "system", "subtype": "init", "session_id": "s-1", "model": "m"},
        )
        adapter._names.handle_for("srv")

        from wardex_sdk._enums import SpanKind, StatusCode
        from wardex_sdk._types import InternalSpan, SpanContext, SpanId, TraceId

        _hub.get_client().capture_span(
            InternalSpan(
                context=SpanContext(TraceId.generate(), SpanId.generate()),
                parent_span_id=None,
                name="buffered-before-fork",
                kind=SpanKind.CLIENT,
                start_time_ns=1,
                end_time_ns=2,
                status=StatusCode.OK,
            )
        )

        # Stage a PENDING finalize entry (integration S-4). The traffic above
        # never seals a transaction — request bytes only — so the parse
        # backlog forks empty, and an empty table cannot distinguish "the
        # Client -> FinalizeQueue fork wiring ran" from "it was deleted": the
        # queue's own unit test drives _at_fork_reinit directly and proves
        # nothing about the wiring. Submitted with the worker deliberately
        # never spawned (ensure_alive is not called; submit's wake() on an
        # unspawned worker is a no-op), so the entry deterministically
        # survives to the fork instant; the child's populated-table walk
        # below then turns a missing Client wiring into a red test. The
        # parent's own close() drains the stub through the FALLBACK path.
        class _StubDeferred:
            size = 64
            ctx = None

            def run(self):  # noqa: ANN202 — DeferredSpan duck type
                return None

            def fallback(self, marker):  # noqa: ANN001, ANN202
                return None

        finalize = _hub.get_client()._finalize
        finalize.submit(_StubDeferred(), ({}, None))
        assert finalize.pending() > 0, "the S-4 staging must survive to the fork"

        assert seam._conns and assembler.open_session_count() == 1

        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            code = 0
            try:
                encoded = json.dumps(_walk_child_graph()).encode()
            except BaseException as exc:  # noqa: BLE001 — reported to the parent
                encoded = json.dumps({"error": repr(exc)}).encode()
                code = 1
            try:
                os.write(w, encoded)
                os.close(w)
            finally:
                os._exit(code)
        os.close(w)
        chunks = []
        while True:
            chunk = os.read(r, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        os.close(r)
        _, status = os.waitpid(pid, 0)
        payload = json.loads(b"".join(chunks).decode())
        assert os.waitstatus_to_exitcode(status) == 0, payload
        assert payload["violations"] == [], (
            "the fork reset DECLARED coverage the walk could not confirm:\n  "
            + "\n  ".join(payload["violations"])
        )
        # Non-vacuity: a walk that saw nothing proves nothing. The exact
        # figures float with the SDK's shape; the floor is what matters.
        assert payload["locks_checked"] >= 8, payload
        assert payload["tables_checked"] >= 10, payload
    finally:
        wardex.close()
