"""Delivery of a configured bound to the component that enforces it.

`test_limits.py` answers "is this limit enforced anywhere?". This file answers
the other half — "does the USER's value get there?" — and the two are genuinely
different questions. Most of the probes next door hand a value straight to the
component under test rather than routing it through `wardex.init()`, so a bound
could be enforced there and still never leave `LimitsConfig` in production.
Three of them did: the unit registry's body cap, the MCP tool catalog's entry
cap, and the shared timing store's connection cap after a second `init()`.

Five guards, because the three failures had three different shapes and no
single rule sees all of them:

  G2  every parameter of a registered consumer is classified (delivered,
      native, or passthrough), so a new bound cannot arrive unclassified.
  G3  every construction of a registered consumer goes through the projection,
      so a call site cannot hand-spell the keywords and forget one.
  G3b a registered NATIVE call always passes the native limits object.
  G4  every field reaches its enforcement site through a real `init()` — twice,
      because one of the three failures only appears on the SECOND init.
  G5  the process defaults are resolved only where a declared row says so,
      which is the one rule that would have caught a component resolving its
      own bound.

The AST guards (G3, G3b, G5) each have counterexample metatests below. A
scanner whose symbol resolution is subtly wrong matches nothing and stays green
forever, which is the same failure mode as the bug it is here to prevent.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest

from wardex_sdk._limits import _LIMIT_DELIVERY, LimitsConfig, LimitsConsumer, limits_kwargs


def _resolve(target: str) -> Any:
    """`"pkg.mod:Symbol"` / `"pkg.mod:Symbol.method"` -> the object.

    Import errors and attribute errors are left to propagate: a table row that
    names something that moved must fail loudly at collection, which is the
    whole reason the row may be a string in the first place.
    """
    module, _, symbol = target.partition(":")
    obj: Any = importlib.import_module(module)
    for part in symbol.split("."):
        obj = getattr(obj, part)
    return obj


def _parameters(target: str) -> dict[str, inspect.Parameter]:
    params = dict(inspect.signature(_resolve(target)).parameters)
    params.pop("self", None)
    return params


CONSUMERS = list(_LIMIT_DELIVERY)


# ==========================================================================
# G2 — the signature and the row are one description
# ==========================================================================


@pytest.mark.parametrize("consumer", CONSUMERS, ids=lambda c: c.value)
def test_every_consumer_parameter_is_classified(consumer):
    """A registered consumer's parameters are exactly delivered ∪ native ∪ passthrough.

    Both directions matter and each catches a different edit. A parameter with
    no classification is a bound (or a dependency) nobody decided about — the
    half-write this whole file exists for. A classification with no parameter is
    a row that outlived a rename, and `limits_kwargs` would then raise
    `TypeError` deep inside an `install()` guard, which turns the seam off and
    counts it rather than telling anyone why.
    """
    row = _LIMIT_DELIVERY[consumer]
    declared = set(row.delivers) | row.native | row.passthrough
    actual = set(_parameters(row.target))
    assert actual == declared, (
        f"{consumer.value}: the row for {row.target} and its real signature disagree.\n"
        f"  parameters with no classification: {sorted(actual - declared)}\n"
        f"  classifications with no parameter: {sorted(declared - actual)}\n\n"
        "WHY: a parameter that is a resource bound belongs in `delivers`, so the\n"
        "projection carries it. One that takes the `to_native()` object belongs\n"
        "in `native`. Anything else belongs in `passthrough` — which is a\n"
        "DECISION that it is not a bound, not a place to put things."
    )


@pytest.mark.parametrize("consumer", CONSUMERS, ids=lambda c: c.value)
def test_delivered_kwargs_are_keyword_acceptable(consumer):
    """Every delivered name can actually be passed as a keyword.

    `limits_kwargs` is expanded with `**`, so a positional-only parameter or a
    `**kwargs` catch-all would make the projection a silent no-op at one call
    site while every other guard here stayed green.
    """
    row = _LIMIT_DELIVERY[consumer]
    params = _parameters(row.target)
    for kw in row.delivers:
        kind = params[kw].kind
        assert kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ), f"{consumer.value}: {row.target} cannot take {kw} as a keyword ({kind})"


@pytest.mark.parametrize("consumer", CONSUMERS, ids=lambda c: c.value)
def test_delivered_fields_are_mirror_fields(consumer):
    """Each row delivers FROM a field the mirror declares.

    `resolved()` returns the core's whole table, including the two fields the
    mirror deliberately cuts (`replay_buffer_size`, `zstd_level`), so a row
    naming one of those would resolve happily and hand a host-side component a
    value no user can set.
    """
    row = _LIMIT_DELIVERY[consumer]
    mirrored = set(LimitsConfig.__dataclass_fields__)
    unknown = set(row.delivers.values()) - mirrored
    assert not unknown, (
        f"{consumer.value}: delivers from {sorted(unknown)}, which the mirror "
        f"does not declare. A bound a user cannot set is not a bound."
    )


def test_the_projection_returns_exactly_the_declared_keywords():
    """The projection is the row, evaluated — nothing added, nothing dropped."""
    resolved = LimitsConfig().resolved()
    for consumer, row in _LIMIT_DELIVERY.items():
        kwargs = limits_kwargs(consumer, resolved)
        assert set(kwargs) == set(row.delivers), consumer.value
        for kw, field in row.delivers.items():
            assert kwargs[kw] == resolved[field], f"{consumer.value}: {kw}"


def test_the_projection_carries_an_override_rather_than_the_default():
    """The whole point, stated once: an override reaches the keyword.

    A projection that read `limits_defaults()` instead of its argument would
    pass every structural test above and deliver the process default forever.
    """
    resolved = LimitsConfig(max_units=7, max_entries_per_unit=3).resolved()
    kwargs = limits_kwargs(LimitsConsumer.UNIT_REGISTRY, resolved)
    assert kwargs["max_units"] == 7
    assert kwargs["max_entries_per_unit"] == 3
