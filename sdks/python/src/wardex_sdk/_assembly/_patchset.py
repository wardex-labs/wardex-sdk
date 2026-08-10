"""The SDK's one patch mechanism — design §3.2, invariants I6 and I7.

wardex observes by monkeypatching: `ssl.SSLSocket`, `ssl.SSLObject`,
`socket.socket`, `ssl.SSLContext`, the asyncio event loops, anyio's subprocess
backend, `asyncio.create_subprocess_exec`, and the Claude Agent SDK's transport
class and module functions. Every one of those sites grew its own dict of
originals and its own uninstall loop, and every one of them undid the patch the
same way:

    setattr(target, name, original)

That line is the SDK breaking the host application, which is the one thing it
may never do. It is unconditional in two separate ways, and each is a distinct
bug that outlives `uninstall()`:

  * It does not ask whether the attribute still holds *wardex's* wrapper. If
    any other library patched the same attribute after wardex did — a second
    observability agent, a `mock.patch` in the host's test suite, a retry shim
    — that library's patch is destroyed and a function nobody asked for is
    reinstated. wardex, a component that has just announced it is gone, leaves
    the host running an interception it did not install and cannot see.

  * It does not ask whether the target *had* the attribute of its own. Patching
    an INHERITED method and restoring it with `setattr` converts it into a
    permanent own-attribute shadow: the class now carries a frozen copy of what
    the base class happened to hold at install time, and a later change to the
    base — a monkeypatch, a subclass reload, a library upgrade in a long-lived
    process — no longer reaches it. Nothing about the SDK is installed any more,
    and the damage is invisible.

`PatchSet` is the single mechanism, and the guarantees are these:

  1. THREE TARGET KINDS — a class, a module, and an INSTANCE. The instance case
     is the reason this exists rather than being a tidy-up: patching
     `SubprocessCLITransport.write` on the CLASS instruments every transport in
     the process, including ones the host built for its own purposes and never
     handed to wardex. An instance patch is scoped to the object wardex was
     actually given.

  2. OWN-ATTRIBUTE MEMORY — the record says whether the target had the name in
     its own namespace. If it did, restore is `setattr` back to the exact value
     that was there (the raw descriptor, so a `classmethod`/`staticmethod`
     survives the round trip). If it did not, restore is `delattr`, which is the
     only way to leave no shadow.

  3. IDENTITY-CHECKED RESTORE — an attribute is restored only while it still
     holds the exact wrapper this PatchSet installed. If it does not, someone
     patched over wardex: their patch is left alone, the event is counted, and
     `Limitation.PATCH_SUPERSEDED` becomes available on this set. Leaving the
     other library's patch in place is the whole point; clobbering it is the
     failure mode being removed.

  4. LIFO — patches applied A then B are undone B then A. `patch()` accepts the
     same attribute twice, and when it is used that way the second record's
     `original` IS the first wrapper, because a wrapper is built from the value
     it replaces. Undone oldest-first, the outer restore then writes the INNER
     wrapper back as the attribute's live value: wardex reports a clean
     uninstall and is still in the call path. LIFO is the only order that
     unwinds that, and it is a property of the mechanism rather than a fix for
     one site — no interceptor layers two patches on one attribute in one set
     today (`_socket` and `_conn_timing` both reach `socket.socket`, but for
     disjoint attributes and out of separate PatchSets), and none of them should
     have to know that the order is what keeps it safe.

  5. INDIVIDUAL ISOLATION — each restore runs inside `_diag.guard`, so one
     failing restore cannot abandon the others. A half-uninstalled SDK is worse
     than either end state, and it is what a single `raise` in the middle of the
     old loops produced. `patch()` is guarded for the same reason from the other
     end: a target that refuses the assignment (a metaclass with a raising
     `__setattr__`, a read-only C type) must not throw into the host, and must
     not abandon the patches this set has already applied — they stay in the set
     so that `restore_all()` can still take them out. It reports the outcome as
     `True`/`False` instead; callers are free to ignore it.

  6. REFUSAL BEFORE CORRUPTION — an instance patch is installed only where the
     assignment lands in storage this module can read back: the instance dict,
     or a slot on an instance that has no dict. If `type(target)` resolves the
     name to a DATA descriptor (a property with a setter, a C getset), `setattr`
     writes THROUGH it into the host's own state, where the own-namespace read
     cannot see it — restore would then report a supersession that never
     happened while leaving wardex's object inside the host's fields. wardex
     cannot scope-patch through a data descriptor, and declining is better than
     pretending: the patch is refused, counted, and nothing is written.

  7. WEAK INSTANCE REFERENCES, AND WHAT THEY DO NOT BUY — an instance target is
     held through a `weakref`, so the RECORD does not pin it, and a referent that
     died before `restore_all()` is a patch with nothing left to restore rather
     than an error. Dead records are pruned (and counted as collected) whenever
     `patch()` runs, so the list cannot grow without bound in a long-lived
     process that patches many short-lived objects; a set that stops patching
     keeps its records until `restore_all()`.

     That is the whole of it, and it is less than "a PatchSet is never the reason
     a host object stays alive": the WRAPPER is held strongly (the identity check
     in rule 3 is a comparison against that exact object, so it cannot be weak),
     and the natural instance wrapper is built by closing over `obj.method` — a
     bound method, which pins `obj` through the wrapper this set holds. In that
     ordinary case the weakref buys nothing at all. It buys something only for a
     wrapper that does not reference its target: write instance wrappers so that
     the object arrives as an argument — the way a class-level wrapper receives
     `self` — or capture a weakref to it, not the object.

  8. ONE LOCK — `patch()` and `restore_all()` are serialized by an `RLock`.
     Two concurrent `restore_all()` calls otherwise interleave their pops, which
     breaks LIFO (rule 4), welds a wrapper on, and reports it as a supersession
     by a third party that does not exist — while `len()` reads 0, the number
     that means "fully uninstalled". Re-entrant because teardown in this SDK can
     re-enter through a signal handler (`_runtime.py` installs one).

Nothing here imports anything above `_assembly/` (`tests/test_import_graph.py`
enforces that), so an interceptor, an adapter and a future second-language
binding all reach the same implementation.
"""

from __future__ import annotations

import threading
import weakref
from types import MemberDescriptorType, ModuleType
from typing import Any

from ._diag import counters, guard
from ._integrity import Limitation

_SUPERSEDED_MARKER = Limitation.PATCH_SUPERSEDED
"""The marker a superseded restore reports.

Named as a module constant rather than written at the two use sites so that
`tests/test_limitation_census.py` records this module as the member's emit site
— the census keys on marker-ish slots, and a member reached only through an
f-string or a bare `return` is a member the census cannot see.
"""

_MISSING: Any = object()
"""'the name is absent from this namespace', distinguishable from a stored None."""

_UNREADABLE: Any = object()
"""'asking the host object what it holds raised' — distinguishable from absent.

A host object can refuse to be read: a slotted instance whose `__getattr__`
raises something other than `AttributeError` answers every unresolved name with
that exception, and the read is host code. Conflating it with `_MISSING` would
make wardex `delattr` an attribute it never wrote; conflating it with a value
would make the identity check in rule 3 report a supersession. It is neither, so
it has its own sentinel: `patch()` refuses, and `_restore` leaves the attribute
exactly as it found it.
"""


def _mro_get(klass: type, name: str) -> Any:
    """`name` from `klass`'s MRO by raw dict lookup, or `_MISSING`.

    Type dicts only, never `getattr`: nothing here triggers a descriptor's
    `__get__`, a `__getattr__` hook, or any other host code, and the value comes
    back as it is stored — which is what makes "is this a data descriptor?"
    answerable without asking the object it belongs to.
    """
    for base in klass.__mro__:
        namespace = base.__dict__
        if name in namespace:
            return namespace[name]
    return _MISSING


def _namespace(target: Any, protect: guard) -> Any:
    """`target`'s own namespace, None if it has none, `_UNREADABLE` if it raised.

    `getattr(target, "__dict__", None)` rather than `vars(target)`: `vars()`
    raises on an object with no `__dict__`, and "this object has no dict" is an
    ordinary input rather than a failure, so it must not arrive as an exception.

    The `guard` is not decoration. `getattr` with a default swallows exactly
    `AttributeError`, and on a dict-less instance the lookup falls through to the
    host's `__getattr__`, which is host code that may raise anything at all —
    `hasattr` has the identical hole, which is why `hasattr` appears nowhere in
    this module and every `getattr` on a host object is inside a guard.
    """
    got: Any = _UNREADABLE
    with protect:
        got = getattr(target, "__dict__", None)
    return got


def _own_value(target: Any, name: str, protect: guard) -> Any:
    """What `target` itself holds under `name`, `_MISSING`, or `_UNREADABLE`.

    Both halves of the mechanism ask this one question: `patch` asks it before
    patching, to learn whether restoring means `setattr` or `delattr`, and
    `_restore` asks it after, to check that the value in place is still wardex's.
    They must agree, so they share the implementation.

    Reading the namespace, not `getattr(target, name)`, is what makes
    inherited-vs-own decidable at all — and it is also what keeps a
    `classmethod`/`staticmethod` intact, since `getattr` on a class unwraps the
    descriptor into the plain function and restoring THAT silently changes how
    the attribute binds. A class yields its own mappingproxy and not its bases'
    (`type.__dict__['__dict__']` is a data descriptor on the metaclass, so it
    wins the lookup); a module yields its module dict; an instance yields its
    instance dict.

    The `getattr` fallback is reached only by a target with no `__dict__` at all,
    and only for a name `_may_patch` has already established is a SLOT on a type
    that defines no `__getattr__`. Both halves of that matter. A slot descriptor
    is a data descriptor that returns exactly the object stored in it, with no
    binding and no MRO fallback, so for a slot `getattr` IS the own-namespace
    read; for any other name it is not, because an unset slot and an absent name
    both fall through to the MRO (returning a bound method for a purely inherited
    attribute, which would record `had_own=True` for something the instance never
    had) or to `__getattr__` (returning whatever the host invents). Those targets
    do not reach here: they are refused.
    """
    namespace = _namespace(target, protect)
    if namespace is _UNREADABLE:
        return _UNREADABLE
    got: Any = _UNREADABLE
    with protect:
        if namespace is None:
            got = getattr(target, name, _MISSING)
        else:
            got = namespace.get(name, _MISSING)
    return got


def _is_scopeable(target: Any, name: str, namespace: Any) -> bool:
    """Can `setattr(target, name, ...)` land where `_own_value` will read it back?

    The instance question of rule 6, asked without touching the object: every
    lookup here is a raw type-dict read.

      * A DATA descriptor on `type(target)` wins the assignment. A property with
        a setter writes into the host's own fields; a C getset writes into C
        state. `_own_value` reads neither, so the patch would be invisible to its
        own restore. The one exception is a slot on an instance that has no
        `__dict__`, which is the case `_own_value`'s `getattr` fallback exists
        for: the write goes into the slot and the read comes straight back out.

      * A target with no `__dict__` has no own namespace for a name that is not a
        slot. `setattr` there raises anyway on an ordinary slotted object, but an
        object with a `__setattr__` of its own accepts the write and forwards it
        into state wardex cannot see or undo.

      * A dict-less target whose type defines `__getattr__` is refused even for a
        slot: an UNSET slot raises `AttributeError` from the descriptor, the hook
        answers instead, and the value it invents is recorded as the original.
    """
    entry = _mro_get(type(target), name)
    if _mro_get(type(entry), "__set__") is not _MISSING:
        if not isinstance(entry, MemberDescriptorType):
            return False
        return namespace is None and _mro_get(type(target), "__getattr__") is _MISSING
    if namespace is None:
        return False
    return True


class _Patch:
    """One installed patch and everything `restore_all` needs to undo it.

    `__slots__` because a PatchSet holds one of these per patched attribute for
    the entire life of the process, and because a stray attribute assigned onto
    a patch record would be a fact about the host's namespace that nothing reads.
    """

    __slots__ = ("_ref", "_strong", "had_own", "name", "original", "wrapper")

    def __init__(self, target: Any, name: str, wrapper: Any, original: Any, *, ref: Any) -> None:
        self.name = name
        self.wrapper = wrapper
        self.original = original
        self.had_own = original is not _MISSING
        self._ref = ref
        self._strong = None if ref is not None else target

    def target(self) -> Any:
        """The patched object, or None once a weakly-held referent has died."""
        if self._ref is None:
            return self._strong
        return self._ref()

    def dead(self) -> bool:
        """True once a weakly-held referent has been collected. Never for a strong hold."""
        return self._ref is not None and self._ref() is None


class PatchSet:
    """Every patch one component installed, restorable exactly once, safely.

    `owner` is a stable, low-cardinality label for the component — the same
    contract `_diag.guard`'s `where` has, because it is what the counter keys
    are built from. "interceptors.ssl", not f"{cls.__name__} patches".
    """

    __slots__ = ("_collected", "_debug", "_lock", "_owner", "_patches", "_refused", "_superseded")

    def __init__(self, owner: str, *, debug: bool = False) -> None:
        self._owner = owner
        self._debug = debug
        self._patches: list[_Patch] = []
        self._superseded = 0
        self._collected = 0
        self._refused = 0
        self._lock = threading.RLock()

    def __len__(self) -> int:
        """Records still awaiting restore.

        Including any whose weakly-held target has died since the last `patch()`
        pruned the list — a dead record still has to be popped, so counting it
        here is what makes `len() == 0` mean "this set is done".
        """
        return len(self._patches)

    # --- install ---

    def patch(self, target: Any, name: str, wrapper: Any) -> bool:
        """Install `wrapper` as `target.name`, recording how to undo it.

        `target` is a class, a module or an instance. An instance patch shadows
        the class attribute for that object alone, which is how wardex
        instruments the transport it was handed without instrumenting every
        transport in the process.

        NEVER RAISES. A target that refuses the assignment (a metaclass with a
        raising `__setattr__`, a read-only C type) is a host that has said no,
        not a wardex bug to propagate into it — the failure is counted under
        `<owner>.patch` and reported as `False`. An instance patch that could not
        be scoped without writing through the host's own state (rule 6) is
        refused the same way, before anything is written. `True` means the
        wrapper is installed and the set can take it out again.

        The record is built BEFORE the `setattr` and appended only AFTER it
        succeeds, and both halves are load-bearing. Built after, `original` would
        be wardex's own wrapper, and restore would weld it on permanently.
        Appended before, a patch that never installed would leave a restore entry
        that later `delattr`s an attribute wardex never wrote. Patches this set
        applied EARLIER are untouched by a refusal: they stay in the set, which
        is the difference between "one patch did not install" and "the SDK cannot
        be uninstalled".

        Returns whether the wrapper is installed, and deliberately not the
        original. Callers need the original in order to build `wrapper` in the
        first place, so they read it themselves one line above; handing it back
        here would invite the lazy `self._orig[key]` lookup this module exists to
        delete — the shape that raises `KeyError` into the host when a wrapper
        outlives its uninstall.
        """
        protect = guard(f"{self._owner}.patch", debug=self._debug)
        with self._lock:
            self._prune()
            if not self._may_patch(target, name, protect):
                return False
            original = _own_value(target, name, protect)
            if original is _UNREADABLE:
                self._refuse()
                return False
            record = _Patch(target, name, wrapper, original, ref=self._hold(target, protect))
            applied = False
            with protect:
                setattr(target, name, wrapper)
                applied = True
            if not applied:
                return False
            self._patches.append(record)
            return True

    def _may_patch(self, target: Any, name: str, protect: guard) -> bool:
        """Rule 6, plus its own failure: an unreadable target is a refused one.

        A class or a module is always patchable here — `setattr` on either writes
        into the namespace `_own_value` reads — and a target that refuses the
        write is caught by the guard around the `setattr` itself.
        """
        if isinstance(target, type | ModuleType):
            return True
        namespace = _namespace(target, protect)
        scopeable = False
        if namespace is not _UNREADABLE:
            with protect:
                scopeable = _is_scopeable(target, name, namespace)
        if not scopeable:
            self._refuse()
        return scopeable

    def _refuse(self) -> None:
        """Record a patch this set declined to install. Counted, never silent."""
        self._refused += 1
        counters.bump(f"{self._owner}.patch_refused")

    def _prune(self) -> None:
        """Drop records whose weakly-held target has been collected.

        Called from `patch()`, under the lock. Without it the list is append-only
        for the life of the set: a dead referent frees the host object and leaves
        the record — and with the record the wrapper closure, held strongly for
        the identity check — behind forever. A component that patches one
        short-lived instance per connection would grow that list without bound
        and without counting anything, which is the shape of a leak nobody sees.

        The records are counted as collected here for the same reason `_restore`
        counts them: "no patches were superseded" must not be readable off a set
        whose targets simply evaporated.
        """
        if not self._patches:
            return
        live = [record for record in self._patches if not record.dead()]
        gone = len(self._patches) - len(live)
        if gone:
            self._collected += gone
            self._patches = live

    def _hold(self, target: Any, protect: guard) -> Any:
        """A weakref for an instance; None (meaning: hold it directly) otherwise.

        Classes and modules are held strongly: they are process-lifetime globals
        reached by name, and a weakref to one buys nothing while adding a way for
        a restore to be skipped.

        The capability test is the reference itself. `hasattr(target,
        "__weakref__")` was the pre-check and it is a hole in I6: `hasattr`
        swallows only `AttributeError`, so a slotted instance falls through to the
        host's `__getattr__`, and one that raises anything else raises out of
        `patch()` into the host — which is the failure this whole module exists to
        remove. `weakref.ref` answers the same question by doing the thing, and
        answers it correctly for a slotted class with no `__weakref__` slot: that
        is an ordinary input, so it is counted as a strong hold rather than as a
        failure. The `guard` covers the residue, where holding the target
        strongly is the lesser evil against a patch nobody can undo.
        """
        if isinstance(target, type | ModuleType):
            return None
        ref: Any = None
        with protect:
            try:
                ref = weakref.ref(target)
            except TypeError:
                counters.bump(f"{self._owner}.strong_hold")
        return ref

    # --- restore ---

    def restore_all(self) -> None:
        """Undo every patch, newest first. Idempotent: a second call does nothing.

        The pop-then-restore order is what makes both of those true at once. A
        record leaves the list before its restore is attempted, so a restore that
        raises inside the guard is not retried on a later call and cannot be
        counted twice — and an interrupted `restore_all` leaves exactly the
        patches it has not reached yet.

        The lock is what makes the pop-then-restore window safe against a second
        caller: unserialized, two threads pop a record each and restore them in
        the wrong order (rule 8).
        """
        protect = guard(f"{self._owner}.restore", debug=self._debug)
        with self._lock:
            while self._patches:
                record = self._patches.pop()
                with protect:
                    self._restore(record, protect)

    def _restore(self, record: _Patch, protect: guard) -> None:
        target = record.target()
        if target is None:
            # The host object was collected while patched. Nothing to restore and
            # nothing wrong: the wrapper died with it. Counted rather than
            # ignored so that "no patches were superseded" cannot be read off a
            # set whose targets simply evaporated.
            self._collected += 1
            return

        current = _own_value(target, record.name, protect)
        if current is _UNREADABLE:
            # Asking the host what it holds raised, and `protect` has counted it.
            # wardex does not write to an attribute it could not read: the value
            # in place may be another library's, and there is no way to tell.
            return

        if current is not record.wrapper:
            # Someone else owns this attribute now — they patched over wardex, or
            # removed wardex's patch outright. Either way the value in place is
            # not ours to restore, and writing the original back here would
            # delete their work. Leave it, and say so.
            self._superseded += 1
            counters.bump(f"{self._owner}.{_SUPERSEDED_MARKER.value}")
            return

        if record.had_own:
            setattr(target, record.name, record.original)
        else:
            delattr(target, record.name)

    # --- what the restore observed ---

    @property
    def superseded(self) -> int:
        """How many patches another library had taken over by restore time."""
        return self._superseded

    @property
    def collected(self) -> int:
        """How many weakly-held targets died before their patch was restored."""
        return self._collected

    @property
    def refused(self) -> int:
        """How many patches this set declined to install rather than corrupt (rule 6)."""
        return self._refused

    def limitations(self) -> tuple[Limitation, ...]:
        """The markers this set's restore earned, for a span that can carry them.

        Today that is `PATCH_SUPERSEDED` and nothing else. It is offered rather
        than emitted because supersession is only DETECTABLE at restore time, and
        by then the component has stopped producing spans — so the counter this
        set bumps is the live signal, and this accessor is what a caller that
        does hold a span (an uninstall-time lifecycle span, when one exists) uses
        to attach the fact instead of re-deriving it.
        """
        if self._superseded:
            return (_SUPERSEDED_MARKER,)
        return ()
