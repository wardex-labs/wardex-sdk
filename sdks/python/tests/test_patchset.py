"""`assembly.PatchSet` — the SDK's one patch mechanism.

Every test here is named for a way the SDK used to break the host application,
because that is what the module replaced. The five hand-rolled patch dictionaries
(`_seam._orig`, `_socket`/`_ssl` uninstall, `_mcp_stdio._orig_*`,
`_anthropic_agent_sdk._originals`, `_conn_timing._orig`) all restored with an
unconditional

    setattr(target, name, original)

and that one line carries two distinct defects that outlive `uninstall()`:
it overwrites whatever another library put there afterwards, and it turns an
INHERITED method into a permanent own-attribute shadow. The second was not
hypothetical — `test_ssl_interceptor.py::test_install_uninstall_restores_originals`
was failing on Python 3.14 for exactly that reason, because
`_UnixSelectorEventLoop` no longer overrides `create_connection` and wardex's
connect-timing wrapper stayed welded to it forever after uninstall.
"""

from __future__ import annotations

import gc
import threading
import types
import weakref
from typing import Any

import pytest

from wardex_sdk.assembly import Limitation, PatchSet, counters


class Base:
    def greet(self) -> str:
        return "base"


class Child(Base):
    """Inherits `greet`; defines nothing of its own."""


class Transport:
    def write(self) -> str:
        return "real"


def _superseded_key(owner: str) -> str:
    return f"{owner}.{Limitation.PATCH_SUPERSEDED.value}"


# --------------------------------------------------------------------------
# rule 5, install half — `patch()` never raises into the host either
# --------------------------------------------------------------------------


class RefusesAssignment(type):
    """A metaclass that rejects attribute assignment — the `setattr` raises.

    The host shapes this stands in for are real and are exactly the ones wardex
    reaches for: a C extension type, a module proxy that freezes its namespace,
    a class with a defensive `__setattr__`.
    """

    def __setattr__(cls, name: str, value: Any) -> None:
        raise TypeError("this type does not accept attribute assignment")


class ReadOnly(metaclass=RefusesAssignment):
    def run(self) -> str:
        return "real"


def test_a_target_that_refuses_assignment_does_not_raise_into_the_host():
    """An unpatchable target is a host that said no, not an exception to forward.

    Nothing catches `patch()`. Every `install()` in the SDK sets
    `self._installed = True` on its LAST line, so an exception here escapes into
    the host AND leaves `uninstall()` short-circuiting on a False flag — the
    wrappers already applied are welded on with nothing able to remove them.
    """
    owner = "test.refuses_setattr"
    before = counters.get(f"{owner}.patch")
    patches = PatchSet(owner)
    first = Transport()
    patches.patch(first, "write", lambda: "wardex")

    assert patches.patch(ReadOnly, "run", lambda self: "wardex") is False

    assert counters.get(f"{owner}.patch") == before + 1, "the refused assignment was not counted"
    assert ReadOnly().run() == "real"

    # and the whole point of not raising: what this set already applied is still
    # in the set, so the SDK can still be taken out.
    assert len(patches) == 1
    patches.restore_all()
    assert first.write() == "real"
    assert "write" not in vars(first), "an earlier patch was abandoned by a later failure"


def test_a_patch_that_never_installed_leaves_no_restore_entry():
    """The record is built BEFORE the `setattr` and appended only AFTER it.

    Both halves are load-bearing, and only this one was untested. Appended
    before, a patch that never installed leaves a restore entry, and
    `restore_all` later acts on it — `delattr`ing an attribute wardex never
    wrote, or writing a stale "original" over whatever the host holds now.
    """
    owner = "test.no_entry"
    patches = PatchSet(owner)
    original = ReadOnly.__dict__["run"]

    assert patches.patch(ReadOnly, "run", lambda self: "wardex") is False
    assert len(patches) == 0, "a patch that never installed left a restore entry"

    before = counters.get(f"{owner}.restore")
    patches.restore_all()

    assert counters.get(f"{owner}.restore") == before, (
        "restore_all acted on a patch that was never installed"
    )
    assert ReadOnly.__dict__["run"] is original


# --------------------------------------------------------------------------
# rule 3 — restore is identity-checked
# --------------------------------------------------------------------------


def test_uninstall_does_not_destroy_a_second_librarys_patch():
    """The failure this module exists to remove.

    Another agent, a `mock.patch`, a retry shim — anything that patches the same
    attribute AFTER wardex — used to have its patch deleted by wardex's
    uninstall, which then reinstated a function nobody asked for. wardex, having
    just announced it is gone, left the host running an interception it did not
    install and cannot see.
    """
    owner = "test.superseded"
    original = Transport.write
    before = counters.get(_superseded_key(owner))
    patches = PatchSet(owner)

    def wardex_wrapper(self) -> str:
        return "wardex"

    def other_library_wrapper(self) -> str:
        return "other-library"

    patches.patch(Transport, "write", wardex_wrapper)
    Transport.write = other_library_wrapper  # a second library, patching later

    patches.restore_all()

    try:
        assert Transport.write is other_library_wrapper, (
            "wardex's uninstall clobbered another library's patch"
        )
        assert Transport().write() == "other-library"
        # and it is reported rather than merely survived
        assert patches.superseded == 1
        assert patches.limitations() == (Limitation.PATCH_SUPERSEDED,)
        assert counters.get(_superseded_key(owner)) == before + 1
    finally:
        Transport.write = original


class _TransparentProxy:
    """The shape `wrapt.ObjectProxy` has, and therefore the shape OTel installs.

    Every `opentelemetry-instrumentation-*` package patches through
    `wrap_function_wrapper`, which installs a `wrapt.FunctionWrapper` — an
    `ObjectProxy` that forwards `__eq__` to the object it wraps. So a proxy
    placed OVER wardex's wrapper is a different object that compares EQUAL to
    it.

    Reproduced here rather than imported: `wrapt` is not a dependency of this
    SDK, and the property under test is the proxy's equality behaviour, not
    wrapt's implementation of it.
    """

    def __init__(self, wrapped: Any) -> None:
        self.__wrapped__ = wrapped

    def __eq__(self, other: Any) -> bool:
        return bool(self.__wrapped__ == other)

    def __hash__(self) -> int:
        return hash(self.__wrapped__)

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return "other-library"


def test_a_transparent_proxy_over_wardex_is_still_a_supersession():
    """Identity, not equality — and the whole suite passes if you get it wrong.

    `_restore` asks whether the value in place is still wardex's. Spelled
    `current != record.wrapper` instead of `is not`, every other test in this
    file still passes, because they all install a plain function as the second
    library's patch and `==` on a function degenerates to identity.

    A transparent proxy does not. It compares equal to what it wraps, so under
    `==` wardex concludes nothing changed, restores the original, and deletes
    the other library's patch — the exact bug this module exists to remove,
    against the exact library its own docstring names.
    """
    owner = "test.superseded_proxy"
    original = Transport.write
    before = counters.get(_superseded_key(owner))
    patches = PatchSet(owner)

    def wardex_wrapper(self: Any) -> str:
        return "wardex"

    patches.patch(Transport, "write", wardex_wrapper)
    instrumented = _TransparentProxy(wardex_wrapper)
    assert instrumented == wardex_wrapper, "the proxy under test is not transparent"
    assert instrumented is not wardex_wrapper
    Transport.write = instrumented  # OTel wraps wardex's wrapper

    patches.restore_all()

    try:
        assert Transport.write is instrumented, (
            "wardex's uninstall clobbered an OTel instrumentation wrapper"
        )
        assert Transport().write() == "other-library"
        assert patches.superseded == 1
        assert patches.limitations() == (Limitation.PATCH_SUPERSEDED,)
        assert counters.get(_superseded_key(owner)) == before + 1
    finally:
        Transport.write = original


def test_a_patch_removed_by_someone_else_is_reported_not_reinstated():
    """`del cls.attr` by a third party is supersession too.

    The live attribute is not wardex's wrapper, so writing the original back
    would resurrect an attribute the host deliberately removed.
    """
    owner = "test.superseded_by_delete"
    patches = PatchSet(owner)

    class Local(Base):
        def greet(self) -> str:
            return "local"

    original = Local.greet
    patches.patch(Local, "greet", lambda self: "wardex")
    del Local.greet  # someone else removes it entirely

    patches.restore_all()

    assert "greet" not in Local.__dict__, "wardex resurrected an attribute it did not own"
    assert Local().greet() == "base"
    assert patches.superseded == 1
    assert original is not None


def test_an_untouched_patch_reports_no_supersession():
    """The negative control: `superseded` must not fire on the happy path."""
    patches = PatchSet("test.clean")
    original = Transport.write
    patches.patch(Transport, "write", lambda self: "wardex")

    patches.restore_all()

    assert Transport.write is original
    assert patches.superseded == 0
    assert patches.limitations() == ()


# --------------------------------------------------------------------------
# rule 2 — own-attribute memory
# --------------------------------------------------------------------------


def test_restoring_an_inherited_method_leaves_no_own_attribute_shadow():
    """`setattr` on restore is a bug that survives uninstall.

    Restoring an inherited method by assignment freezes a copy of whatever the
    base held at install time onto the subclass. Nothing looks wrong — the
    subclass answers with the right function — until the base changes and the
    subclass no longer follows it. This is the defect that had wardex's
    connect-timing wrapper permanently attached to `_UnixSelectorEventLoop`.
    """
    patches = PatchSet("test.inherited")
    assert "greet" not in Child.__dict__, "precondition: Child inherits greet"

    patches.patch(Child, "greet", lambda self: "wardex")
    assert "greet" in Child.__dict__, "precondition: the patch shadows the base"

    patches.restore_all()

    assert "greet" not in Child.__dict__, "restore left an own-attribute shadow"
    assert Child().greet() == "base"

    # the proof that the shadow is really gone: a later change to the base must
    # reach the subclass again, which a frozen copy would swallow.
    later = Base.greet
    try:
        Base.greet = lambda self: "changed later"
        assert Child().greet() == "changed later"
    finally:
        Base.greet = later


def test_an_own_method_is_restored_to_the_same_object():
    patches = PatchSet("test.own")

    class Local:
        def greet(self) -> str:
            return "own"

    original = Local.__dict__["greet"]
    patches.patch(Local, "greet", lambda self: "wardex")

    patches.restore_all()

    assert Local.__dict__["greet"] is original


def test_a_classmethod_does_not_come_back_as_a_plain_function():
    """The descriptor, not what `getattr` unwraps it into.

    `interceptors/_mcp_stdio.py` patches `AsyncIOBackend.open_process`, which is
    a `classmethod`. A restore built from `getattr(cls, name)` writes back the
    underlying function, so the attribute silently stops binding the class and
    every later call is handed the wrong first argument.
    """
    patches = PatchSet("test.descriptor")

    class Local:
        @classmethod
        def make(cls) -> str:
            return cls.__name__

    original = Local.__dict__["make"]
    assert isinstance(original, classmethod)

    patches.patch(Local, "make", staticmethod(lambda: "wardex"))
    assert Local.make() == "wardex"

    patches.restore_all()

    assert Local.__dict__["make"] is original
    assert isinstance(Local.__dict__["make"], classmethod)
    assert Local.make() == "Local"


# --------------------------------------------------------------------------
# rule 1 — three target kinds
# --------------------------------------------------------------------------


def test_an_instance_patch_does_not_reach_a_second_instance():
    """The reason instance patching exists at all.

    Patching `SubprocessCLITransport.write` on the CLASS instruments every
    transport in the process, including ones the host built for its own purposes
    and never handed to wardex. An instance patch is scoped to the object wardex
    was actually given.
    """
    patches = PatchSet("test.instance")
    mine, theirs = Transport(), Transport()

    patches.patch(mine, "write", lambda: "wardex")

    assert mine.write() == "wardex"
    assert theirs.write() == "real", "an instance patch leaked onto another instance"
    assert Transport().write() == "real", "an instance patch leaked onto the class"

    patches.restore_all()

    assert "write" not in vars(mine), "the instance kept an own-attribute shadow"
    assert mine.write() == "real"
    assert theirs.write() == "real"


def test_a_module_attribute_round_trips():
    """`asyncio.create_subprocess_exec` and `sdk.query` are module patches."""
    patches = PatchSet("test.module")
    module = types.ModuleType("wardex_test_module")
    module.entry = lambda: "real"
    original = module.entry

    patches.patch(module, "entry", lambda: "wardex")
    assert module.entry() == "wardex"

    patches.restore_all()

    assert module.entry is original


def test_a_module_attribute_wardex_invented_is_deleted_not_left_behind():
    patches = PatchSet("test.module_new")
    module = types.ModuleType("wardex_test_module_new")

    patches.patch(module, "entry", lambda: "wardex")
    patches.restore_all()

    assert not hasattr(module, "entry")


def test_an_own_value_of_none_is_restored_rather_than_deleted():
    """`_MISSING` is not `None`, and the difference is a host attribute's life.

    A host attribute whose own value is legitimately `None` — a handle created
    lazily, a callback the host cleared — is an OWN attribute. Collapse the
    sentinel to `None` and `had_own` becomes False, which turns the restore into
    a `delattr`: wardex deletes a host attribute it merely borrowed, and the
    host's next `module.handle = ...` is preceded by an AttributeError nobody
    can trace back to an observability SDK that has announced it is gone.
    """
    patches = PatchSet("test.none")
    module = types.ModuleType("wardex_test_module_none")
    module.handle = None

    patches.patch(module, "handle", lambda: "wardex")
    patches.restore_all()

    assert "handle" in vars(module), "wardex deleted a host attribute whose own value was None"
    assert module.handle is None


# --------------------------------------------------------------------------
# rule 6 — refused rather than corrupted
# --------------------------------------------------------------------------


class HostWithAProperty:
    """`type(obj)` owns the name through a data descriptor with a setter."""

    def __init__(self) -> None:
        self._value = "real"

    @property
    def write(self) -> Any:
        return self._value

    @write.setter
    def write(self, value: Any) -> None:
        self._value = value


def test_a_data_descriptor_on_the_type_is_refused_not_patched_through():
    """An instance patch that cannot be scoped must not be attempted.

    A property with a setter WINS the assignment: `setattr(obj, "write", w)`
    calls the host's own setter and puts wardex's wrapper inside the host's
    fields. The own-namespace read never sees it, so restore finds "not our
    wrapper", counts a PATCH_SUPERSEDED that no second library caused, and walks
    away leaving wardex's object in the host's state. The one signal this
    mechanism exists to produce, emitted because wardex corrupted the host.
    """
    owner = "test.data_descriptor"
    before = counters.get(f"{owner}.patch_refused")
    patches = PatchSet(owner)
    obj = HostWithAProperty()

    installed = patches.patch(obj, "write", lambda: "wardex")

    assert obj.write == "real", "wardex wrote through the host's own setter"
    assert obj._value == "real", "wardex's wrapper is sitting in the host's own state"
    assert installed is False
    assert len(patches) == 0, "a refused patch left a restore entry"
    assert patches.refused == 1
    assert counters.get(f"{owner}.patch_refused") == before + 1

    patches.restore_all()

    assert patches.superseded == 0, "a patch wardex refused was reported as superseded"
    assert patches.limitations() == ()
    assert obj.write == "real"


def test_a_slot_on_a_dict_less_instance_is_still_patchable():
    """The one data descriptor that is not a refusal, and why.

    A slot IS the instance's own storage: the write goes into it and the read
    comes straight back out, which is the case the `getattr` fallback in
    `_own_value` exists for. Refusing every `__set__` indiscriminately would
    take this with it.
    """
    patches = PatchSet("test.slot_ok")

    class Slotted:
        __slots__ = ("write",)

    obj = Slotted()
    assert patches.patch(obj, "write", lambda: "wardex") is True
    assert obj.write() == "wardex"

    patches.restore_all()

    assert not hasattr(obj, "write")
    assert patches.refused == 0


# --------------------------------------------------------------------------
# rule 4 — LIFO
# --------------------------------------------------------------------------


def test_two_patches_on_one_attribute_are_undone_newest_first():
    """Order is observable, and FIFO leaves wardex's wrapper installed forever.

    Undone oldest-first, the inner restore finds a wrapper it does not recognise
    (superseded — correctly refusing to touch it), and the outer restore then
    writes the INNER wrapper back as the attribute's value. wardex is uninstalled
    and still in the call path.

    The justification this docstring used to carry was false and worth replacing
    rather than deleting: `interceptors/_conn_timing` and `interceptors/_socket`
    do both reach `socket.socket`, but for DISJOINT attributes and out of
    separate PatchSets, so neither the order nor the set is shared and LIFO fixes
    nothing there. The rule belongs to the mechanism instead. `patch()` accepts
    the same attribute twice, a wardex wrapper is by construction built from the
    value it replaces, and no call site should have to know that the unwind order
    is what keeps that safe.
    """
    patches = PatchSet("test.lifo")
    calls: list[str] = []

    class Local:
        def run(self) -> str:
            calls.append("original")
            return "original"

    original = Local.__dict__["run"]

    inner_target = Local.run

    def inner(self) -> str:
        calls.append("inner")
        return inner_target(self)

    patches.patch(Local, "run", inner)

    outer_target = Local.run

    def outer(self) -> str:
        calls.append("outer")
        return outer_target(self)

    patches.patch(Local, "run", outer)

    assert Local().run() == "original"
    assert calls == ["outer", "inner", "original"], "precondition: the layers nest"

    patches.restore_all()

    assert Local.__dict__["run"] is original, "restore was not LIFO — a wrapper survived"
    assert patches.superseded == 0, "LIFO restore must not look like a supersession"
    calls.clear()
    assert Local().run() == "original"
    assert calls == ["original"], "a wrapper is still in the call path after uninstall"


# --------------------------------------------------------------------------
# rule 8 — one lock
# --------------------------------------------------------------------------


def test_two_concurrent_restore_alls_do_not_interleave():
    """Unserialized, LIFO is not a property of this mechanism at all.

    `restore_all` pops a record and then restores it, and the window between the
    two is where a second caller gets in. Two threads uninstalling at once is not
    contrived here: an interpreter shutdown hook, a signal handler and an
    explicit `client.shutdown()` all reach the same `uninstall()`.

    Interleaved, the SECOND thread pops the older record while the first is
    mid-`setattr`, finds a live value that is not its own wrapper, and reports
    `PATCH_SUPERSEDED` — blaming a third-party library that does not exist. The
    first thread then finishes and writes the INNER wrapper back as the
    attribute's value. `len()` reads 0, which is the number that means "fully
    uninstalled", and wardex is still in the call path.

    The host's `__setattr__` is the pause: it is host code, it runs INSIDE the
    restore, and a host that takes a lock or logs there is the ordinary way this
    window gets wide enough to lose.
    """
    entered = threading.Event()
    release = threading.Event()
    armed = False

    class Meta(type):
        def __setattr__(cls, name: str, value: Any) -> None:
            if armed and value is inner:
                entered.set()
                release.wait(5)
            super().__setattr__(name, value)

    class Local(metaclass=Meta):
        def run(self) -> str:
            return "original"

    original = Local.__dict__["run"]
    patches = PatchSet("test.threads")

    inner_target = Local.run

    def inner(self) -> str:
        return inner_target(self)

    outer_target: Any = None

    def outer(self) -> str:
        return outer_target(self)

    patches.patch(Local, "run", inner)
    outer_target = Local.run
    patches.patch(Local, "run", outer)
    assert len(patches) == 2, "precondition: two patches on one attribute"

    armed = True
    first = threading.Thread(target=patches.restore_all, name="wardex-restore-1")
    first.start()
    assert entered.wait(5), "the first restore never reached the host's __setattr__"

    second = threading.Thread(target=patches.restore_all, name="wardex-restore-2")
    second.start()
    second.join(0.5)  # serialized it is blocked here; unserialized it is already done
    release.set()
    first.join(5)
    second.join(5)

    assert Local.__dict__["run"] is original, (
        "two concurrent restore_all calls interleaved: a wardex wrapper is welded on"
    )
    assert patches.superseded == 0, (
        "a supersession was reported against a third party that does not exist"
    )
    assert patches.limitations() == ()
    assert len(patches) == 0
    assert Local().run() == "original"


# --------------------------------------------------------------------------
# rule 5 — one failing restore does not abandon the rest
# --------------------------------------------------------------------------


class RefusesDeletion:
    """A host object that rejects attribute deletion — a restore that raises."""

    def write(self) -> str:
        return "real"

    def __delattr__(self, name: str) -> None:
        raise RuntimeError("this object refuses attribute deletion")


def test_one_raising_restore_does_not_abandon_the_others():
    """A half-uninstalled SDK is worse than either end state.

    The old loops were a bare `for` over a dict: the first exception abandoned
    every patch after it, and which ones survived depended on dict order.
    """
    owner = "test.raising"
    before = counters.get(f"{owner}.restore")
    patches = PatchSet(owner)
    first, hostile, last = Transport(), RefusesDeletion(), Transport()

    patches.patch(first, "write", lambda: "wardex")
    patches.patch(hostile, "write", lambda: "wardex")
    patches.patch(last, "write", lambda: "wardex")

    patches.restore_all()  # must not raise into the caller

    # LIFO: `last` restores, `hostile` raises, `first` must still be restored.
    assert last.write() == "real"
    assert first.write() == "real", "a raising restore abandoned the patches behind it"
    assert counters.get(f"{owner}.restore") == before + 1, "the failure was not counted"


def test_restore_all_is_idempotent():
    patches = PatchSet("test.idempotent")
    original = Transport.write
    patches.patch(Transport, "write", lambda self: "wardex")

    patches.restore_all()
    marker = object()
    Transport.write = marker  # type: ignore[assignment]
    try:
        patches.restore_all()  # the second call must do nothing at all
        assert Transport.write is marker
        assert patches.superseded == 0, "a second restore_all re-examined a finished patch"
    finally:
        Transport.write = original
    assert len(patches) == 0


# --------------------------------------------------------------------------
# rule 7 — weak instance references, and what they do not buy
# --------------------------------------------------------------------------


def test_an_instance_collected_before_uninstall_does_not_raise():
    """A dead referent is a patch with nothing to restore, not an error."""
    patches = PatchSet("test.collected")
    doomed = Transport()
    patches.patch(doomed, "write", lambda: "wardex")

    del doomed
    gc.collect()

    patches.restore_all()  # must not raise

    assert patches.collected == 1
    assert patches.superseded == 0, "a collected target is not a supersession"


def test_a_wrapper_that_does_not_close_over_its_target_leaves_it_collectable():
    """The record itself does not pin the instance — which is all the weakref buys.

    Named for what it proves. It does NOT prove "a PatchSet is never the reason a
    host object stays alive": the wrapper here is written the way the module's
    rule 7 asks for, taking nothing from the object it replaces a method on. The
    test below is the other half.
    """
    patches = PatchSet("test.weak")
    obj = Transport()
    ref = weakref.ref(obj)
    patches.patch(obj, "write", lambda: "wardex")

    del obj
    gc.collect()

    assert ref() is None, "the PatchSet held the host object alive"
    patches.restore_all()


def _wrapper_closing_over_the_original(obj: Any) -> Any:
    """`real = obj.method` inside a factory — the shape every seam in the SDK uses.

    Written inline in the test instead, `del real` would clear the closure CELL
    and free the instance, which is an artefact of the test frame and not of the
    wrapper.
    """
    real = obj.write

    def wrapper() -> str:
        return real()

    return wrapper


def test_a_bound_method_wrapper_pins_the_instance_the_weakref_would_have_freed():
    """The caveat, pinned: the natural wrapper defeats the weak reference.

    `real = obj.method` is how every wrapper in this SDK is built, and on an
    INSTANCE target that bound method holds `obj`. The wrapper is held strongly
    and must be — the identity check in rule 3 is a comparison against that exact
    object — so the target stays alive through the PatchSet for as long as the
    record does, weakref or no weakref.

    This is a limit, not a bug to fix by weakening the wrapper reference; it is
    here so the module's docstring cannot quietly go back to claiming otherwise.
    """
    patches = PatchSet("test.pinned")
    obj = Transport()
    ref = weakref.ref(obj)
    wrapper = _wrapper_closing_over_the_original(obj)

    patches.patch(obj, "write", wrapper)

    del obj
    gc.collect()

    assert ref() is not None, (
        "a bound-method wrapper no longer pins its instance — if that is now true "
        "by design, the module docstring's rule 7 caveat is what needs updating"
    )

    patches.restore_all()
    del wrapper
    gc.collect()
    assert ref() is None, "restore_all dropped the record but something still pins the target"


def test_a_collected_targets_record_does_not_outlive_the_next_patch():
    """A dead weakref frees the host object and leaves everything else behind.

    The record stays in the list, and with it the wrapper closure the set holds
    strongly for the identity check. A component that patches one short-lived
    instance per connection — which is what an instance patch IS for — grows that
    list for the life of the process, counting nothing and restoring nothing.
    """
    patches = PatchSet("test.prune")
    doomed = Transport()
    patches.patch(doomed, "write", lambda: "wardex")

    del doomed
    gc.collect()

    live = Transport()
    patches.patch(live, "write", lambda: "wardex")

    assert len(patches) == 1, "the record of a collected target was left in the set"
    assert patches.collected == 1, "the pruned record was dropped without being counted"

    patches.restore_all()

    assert live.write() == "real"
    assert patches.collected == 1, "the pruned record was counted twice"


def _patch_a_class_nothing_else_holds() -> tuple[PatchSet, weakref.ref, Any]:
    """Patch a throwaway class and hand back no strong reference to it.

    The class must not be reachable from the test's own frame. A local `Local`
    keeps it alive whatever `_hold` does, which is what made the previous version
    of the test below pass against a `_hold` that weakly referenced classes too:
    the assertion could not fail, so it was not testing anything.
    """

    class Local:
        def run(self) -> str:
            return "real"

    original = Local.__dict__["run"]  # a plain function; it does not reference Local
    patches = PatchSet("test.strong")
    patches.patch(Local, "run", lambda self: "wardex")
    return patches, weakref.ref(Local), original


def test_a_class_target_is_held_strongly_so_its_restore_still_happens():
    """Classes and modules are process-lifetime globals reached by name.

    Holding one weakly buys nothing — nobody is waiting to reclaim `socket.socket`
    — and adds a way for a restore to be skipped: a class that is only reachable
    through the PatchSet dies, `restore_all` finds a dead referent, counts it as
    collected, and the patch is simply never undone.
    """
    patches, ref, original = _patch_a_class_nothing_else_holds()
    gc.collect()

    target = ref()
    assert target is not None, (
        "the PatchSet held its class target weakly: the class was collected while "
        "patched, and its restore can now never run"
    )
    assert target.__dict__["run"] is not original, "precondition: the class is patched"

    patches.restore_all()

    assert target.__dict__["run"] is original
    assert patches.collected == 0, "a class target was reported as collected"


def test_an_object_that_cannot_be_referenced_weakly_is_still_patchable():
    """A slotted instance rejects `weakref.ref`; that is an input, not a failure."""
    patches = PatchSet("test.slots")

    class Slotted:
        __slots__ = ("write",)

    obj = Slotted()
    patches.patch(obj, "write", lambda: "wardex")
    assert obj.write() == "wardex"

    patches.restore_all()

    assert not hasattr(obj, "write")


# --------------------------------------------------------------------------
# reading a host object is host code — every read here is exception-safe
# --------------------------------------------------------------------------


class HostileLookup:
    """A host object that answers every unresolved attribute with an exception.

    Not exotic: an SDK client that raises `RuntimeError("closed")` from
    `__getattr__` after teardown, a proxy that turns an unknown name into its
    own error type, a lazy-import shim that re-raises the import failure.

    `__slots__ = ("__dict__",)` gives it an instance dict and NO `__weakref__`
    slot, so it cannot be weakly referenced — which is precisely the object the
    old `hasattr(target, "__weakref__")` pre-check interrogated, and `hasattr`
    swallows only `AttributeError`, so the question went into `__getattr__` and
    the answer came back out of `patch()` into the host.
    """

    __slots__ = ("__dict__",)

    def write(self) -> str:
        return "real"

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"host lookup for {name!r} refuses")


def test_probing_weak_referenceability_does_not_go_through_getattr():
    owner = "test.hostile_weakref"
    before = counters.get(f"{owner}.strong_hold")
    patches = PatchSet(owner)
    obj = HostileLookup()

    assert patches.patch(obj, "write", lambda: "wardex") is True, (
        "the weak-reference capability test raised out of patch() into the host"
    )
    assert obj.write() == "wardex"
    assert counters.get(f"{owner}.strong_hold") == before + 1, (
        "a target that cannot be weakly referenced was not recorded as held strongly"
    )

    patches.restore_all()

    assert "write" not in vars(obj)
    assert obj.write() == "real"


class HostileNamespace:
    """Dict-less, weak-referenceable, and hostile to the own-namespace read.

    `getattr(target, "__dict__", None)` has the same hole as `hasattr`: on an
    object with no instance dict the lookup falls through to `__getattr__`, and
    the default only absorbs `AttributeError`.
    """

    __slots__ = ("__weakref__", "write")

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"host lookup for {name!r} refuses")


def test_a_target_that_cannot_be_read_is_refused_not_propagated():
    owner = "test.hostile_namespace"
    before_read = counters.get(f"{owner}.patch")
    patches = PatchSet(owner)
    obj = HostileNamespace()
    slot = HostileNamespace.__dict__["write"]

    assert patches.patch(obj, "write", lambda: "wardex") is False, (
        "reading the host's own namespace raised out of patch() into the host"
    )

    assert counters.get(f"{owner}.patch") == before_read + 1, "the failed read was not counted"
    assert patches.refused == 1
    assert len(patches) == 0
    with pytest.raises(AttributeError):
        slot.__get__(obj, HostileNamespace)  # nothing was written


class GoesHostile:
    """A host object that stops answering reads AFTER wardex patched it.

    A client that raises from `__getattribute__` once it is closed, an object
    whose teardown has already run — uninstall is exactly when wardex meets
    these, because it runs during the host's own shutdown.
    """

    hostile = False

    def write(self) -> str:
        return "real"

    def __getattribute__(self, name: str) -> Any:
        if name == "__dict__" and type(self).hostile:
            raise RuntimeError("this object no longer answers reads")
        return object.__getattribute__(self, name)


def test_an_attribute_that_cannot_be_read_at_restore_is_left_alone_not_blamed():
    """An unreadable attribute is not one another library took over.

    Reporting PATCH_SUPERSEDED here would be the same false signal rule 6 exists
    to prevent, arriving through the other door — and writing the original back
    without a successful identity check would be the unconditional `setattr` this
    module replaced.
    """
    owner = "test.unreadable_restore"
    before = counters.get(f"{owner}.restore")
    before_superseded = counters.get(_superseded_key(owner))
    patches = PatchSet(owner)
    obj = GoesHostile()

    assert patches.patch(obj, "write", lambda: "wardex") is True

    GoesHostile.hostile = True
    try:
        patches.restore_all()  # must not raise, must not guess
    finally:
        GoesHostile.hostile = False

    assert patches.superseded == 0, "an unreadable attribute was blamed on a third party"
    assert patches.limitations() == ()
    assert counters.get(_superseded_key(owner)) == before_superseded
    assert counters.get(f"{owner}.restore") == before + 1, "the failed read was not counted"
    assert vars(obj)["write"]() == "wardex", "wardex wrote to an attribute it could not read"


class InventsAnOriginal:
    """A dict-less host object whose `__getattr__` answers for an UNSET slot.

    This is what makes `getattr` the wrong own-namespace read for anything but a
    set slot: the hook is consulted exactly when the slot is empty, so the value
    it invents is what a `getattr`-based read records as "the original" — and a
    restore then writes the host's invention into a slot that never held it.
    """

    invented = "invented by the host"
    __slots__ = ("__weakref__", "write")

    def __getattr__(self, name: str) -> Any:
        if name == "__dict__":
            return None
        return InventsAnOriginal.invented


def test_a_getattr_hook_never_becomes_the_recorded_original():
    patches = PatchSet("test.invented")
    obj = InventsAnOriginal()
    slot = InventsAnOriginal.__dict__["write"]

    installed = patches.patch(obj, "write", lambda: "wardex")
    patches.restore_all()

    with pytest.raises(AttributeError):
        slot.__get__(obj, InventsAnOriginal)  # the slot is still empty, as it was
    assert installed is False
    assert patches.refused == 1
    assert patches.superseded == 0


# --------------------------------------------------------------------------
# the mechanism as the seams use it
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "wardex_sdk.interceptors._seam",
        "wardex_sdk.interceptors._ssl",
        "wardex_sdk.interceptors._socket",
        "wardex_sdk.interceptors._conn_timing",
        "wardex_sdk.interceptors._mcp_stdio",
        "wardex_sdk.adapters._anthropic_agent_sdk",
    ],
)
def test_no_patch_site_kept_its_own_dictionary_of_originals(module):
    """The conversion, asserted rather than described.

    A site that keeps its own originals keeps the unconditional `setattr` that
    goes with it, and every guarantee above stops applying to that site alone.

    Read by AST, not by substring: the comments at these sites NAME the old
    attributes to say what they replaced, and a text scan would either fail on
    the prose or force the prose out.
    """
    import ast
    import importlib
    import inspect

    tree = ast.parse(inspect.getsource(importlib.import_module(module)))
    kept = sorted(
        {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr.startswith("_orig")
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        }
    )

    assert kept == [], f"{module} still keeps a hand-rolled originals dict: {kept}"
