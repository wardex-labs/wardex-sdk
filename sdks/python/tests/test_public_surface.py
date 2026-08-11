"""The root namespace IS `__all__` — no leaks, and every name resolves.

`dir(wardex_sdk)` used to carry more than the API: `Client`, `WardexConfig`,
`NATIVE_OK`, eight parentage-forging assembly helpers, and the stdlib names
the module itself imported. None of those were promised, but every one of them
was one attribute access from being pinned by a host — and a pinned accident
is an API. These tests hold the boundary mechanically so the next import added
to `__init__.py` is a decision rather than a leak.
"""

from __future__ import annotations

import enum
import inspect
import pathlib
import typing

import wardex_sdk
import wardex_sdk.context
import wardex_sdk.testing
import wardex_sdk.transport

#: The only non-underscore names allowed in `dir(wardex_sdk)` beyond `__all__`
#: and the dunders: the public subpackages, which appear as attributes of the
#: package once imported. Exactly the three with a user story.
_PUBLIC_SUBMODULES = {"transport", "context", "testing"}


def test_the_root_namespace_leaks_nothing_beyond_all():
    extras = {
        name for name in set(dir(wardex_sdk)) - set(wardex_sdk.__all__) if not name.startswith("_")
    }
    assert extras == _PUBLIC_SUBMODULES, (
        f"dir(wardex_sdk) holds names that are neither __all__ nor private: "
        f"{sorted(extras - _PUBLIC_SUBMODULES)}. A reachable name is a pinnable "
        "name whatever __all__ says — import it under an underscore alias, or "
        "add it to __all__ deliberately."
    )


def test_every_name_in_all_resolves():
    missing = [name for name in wardex_sdk.__all__ if not hasattr(wardex_sdk, name)]
    assert not missing, f"__all__ names nothing importable: {missing}"


# --------------------------------------------------------------------------
# the signature walk — a public callable's parameters speak public names
# --------------------------------------------------------------------------

#: Parameter types a public callable still references without a public
#: spelling. A RECORDED DEBT, not an allowance: each entry is a type whose
#: public future is decided by a later slice of the same API batch, and the
#: set may only shrink. A NEW entry means a callable was published whose
#: signature a user cannot type out — export the type or unpublish the
#: callable instead of widening this.
_KNOWN_UNEXPORTED: dict[str, frozenset[str]] = {
    # `triggers` is typed with the retention vocabulary; the retention group's
    # shape (and whether the trigger enum survives at all) is a later slice.
    "RetentionPolicy": frozenset({"CaptureTrigger"}),
    # `Scope.span_context` is the span machinery, which stays unnameable from
    # user code (I5); the scope surface is retargeted in a later slice.
    "Scope": frozenset({"SpanContext"}),
}


def _wardex_types(tp: object, acc: set[type]) -> None:
    """Every wardex_sdk-defined class reachable inside annotation `tp`."""
    if isinstance(tp, type) and getattr(tp, "__module__", "").startswith("wardex_sdk"):
        acc.add(tp)
    for arg in typing.get_args(tp):
        _wardex_types(arg, acc)


def _parameter_hints(obj: object) -> dict[str, object]:
    """Resolved parameter annotations, or {} where resolution is impossible.

    `typing.get_type_hints` needs every name in the annotation to be resolvable
    against the defining module. A callable whose annotations cannot resolve is
    skipped rather than failed here — the walk covers what it can see, and an
    unresolvable annotation is a typing bug with its own symptoms.
    """
    try:
        target = obj.__init__ if inspect.isclass(obj) else obj
        hints = typing.get_type_hints(target)
    except Exception:
        return {}
    hints.pop("return", None)
    return hints


def test_every_public_callable_signature_speaks_public_names():
    """PARAMETERS only, for now: return/yield coverage extends when the
    tracing and transport surfaces land in their own slices."""
    public = set(wardex_sdk.__all__) | set(wardex_sdk.transport.__all__)
    violations: list[str] = []
    for name in wardex_sdk.__all__:
        obj = getattr(wardex_sdk, name)
        if not callable(obj) or isinstance(obj, enum.EnumMeta):
            continue
        referenced: set[type] = set()
        for hint in _parameter_hints(obj).values():
            _wardex_types(hint, referenced)
        tolerated = _KNOWN_UNEXPORTED.get(name, frozenset())
        for cls in sorted(referenced, key=lambda c: c.__name__):
            if cls.__name__ in public or cls.__name__ in tolerated:
                continue
            violations.append(f"  {name} takes a {cls.__name__} ({cls.__module__})")
    assert not violations, (
        "a public callable's parameters reference wardex types with no public name:\n"
        + "\n".join(violations)
        + "\n\nWHY: a signature a user cannot spell is an API they cannot call\n"
        "correctly — the type is de-facto public the moment the parameter is.\n"
        "Export the type, or record it in _KNOWN_UNEXPORTED with the slice that\n"
        "resolves it."
    )


def test_the_signature_walk_is_not_vacuous():
    """The walk must actually see wardex types, or the rule above asserts
    nothing. `set_user(user: UserInfo)` is the simplest signature it must
    resolve; a walk that cannot see it has gone blind, not clean."""
    referenced: set[type] = set()
    for hint in _parameter_hints(wardex_sdk.set_user).values():
        _wardex_types(hint, referenced)
    assert wardex_sdk.UserInfo in referenced


# --------------------------------------------------------------------------
# py.typed — the wheel's annotations are visible to type checkers
# --------------------------------------------------------------------------


def test_py_typed_ships_next_to_the_package():
    marker = pathlib.Path(wardex_sdk.__file__).parent / "py.typed"
    assert marker.exists(), (
        "py.typed is missing from the installed package; without the PEP 561 "
        "marker every inline annotation in the wheel is invisible to type "
        "checkers (see [tool.maturin] include in pyproject.toml)"
    )
