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

import ast
import importlib
import inspect
import os
import pathlib
from typing import Any, NamedTuple

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


# ==========================================================================
# source access, shared by the three AST guards
# ==========================================================================

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "wardex_sdk"


def _modules() -> dict[str, ast.Module]:
    """Every module in the package, keyed by path relative to `wardex_sdk/`."""
    return {
        path.relative_to(_SRC).as_posix(): ast.parse(
            path.read_text(encoding="utf-8"), filename=str(path)
        )
        for path in sorted(_SRC.rglob("*.py"))
    }


def _aliases(tree: ast.Module) -> dict[str, str]:
    """Local name -> the symbol it was imported AS.

    Without this, `from .._assembly import UnitRegistry as R` renames a
    registered consumer out of every scanner's sight — which is not a
    hypothetical dodge so much as the ordinary way a scanner written against
    spellings goes quietly blind.
    """
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                out[a.asname or a.name] = a.name
        elif isinstance(node, ast.Import):
            for a in node.names:
                out[a.asname or a.name.split(".")[0]] = a.name.split(".")[-1]
    return out


def _called_symbol(call: ast.Call, aliases: dict[str, str]) -> str | None:
    """The name this call names, un-aliased. `None` for anything else."""
    func = call.func
    if isinstance(func, ast.Name):
        return aliases.get(func.id, func.id)
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _calls(tree: ast.AST):
    """Every `ast.Call`, paired with the `ClassDef` it is written inside."""

    def walk(node: ast.AST, cls: ast.ClassDef | None):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Call):
                yield child, cls
            yield from walk(child, child if isinstance(child, ast.ClassDef) else cls)

    yield from walk(tree, None)


#: Symbol name -> the consumer whose row registers it. The name is the class
#: for a constructor row and the METHOD for a method row, which is why the
#: metatest below pins uniqueness: two rows landing on one name would make
#: each other's call sites unauditable.
_BY_SYMBOL = {
    row.target.partition(":")[2].split(".")[-1]: consumer
    for consumer, row in _LIMIT_DELIVERY.items()
}


def _limits_kwargs_member(node: ast.AST) -> str | None:
    """`limits_kwargs(LimitsConsumer.X, ...)` -> `"X"`, else None."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
    if name != "limits_kwargs" or not node.args:
        return None
    first = node.args[0]
    if (
        isinstance(first, ast.Attribute)
        and isinstance(first.value, ast.Name)
        and first.value.id == "LimitsConsumer"
    ):
        return first.attr
    return None


def _prebuilt_attrs(cls: ast.ClassDef | None) -> dict[str, str]:
    """`self.X = limits_kwargs(LimitsConsumer.M, ...)` inside this class -> {X: M}.

    The second sanctioned spelling of the projection (see `_projection_violations`),
    scoped to the class that owns the attribute so a cache built in one class
    cannot excuse a call in another.
    """
    out: dict[str, str] = {}
    if cls is None:
        return out
    for node in ast.walk(cls):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        member = _limits_kwargs_member(value)
        if member is None:
            continue
        for t in targets:
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name):
                if t.value.id == "self":
                    out[t.attr] = member
    return out


# ==========================================================================
# G3 — a registered consumer is only ever built through the projection
# ==========================================================================


def _projection_violations(tree: ast.Module) -> list[tuple[int, str, str]]:
    """Calls to a registered consumer that do not go through `limits_kwargs`.

    TWO spellings are legal, and the set is closed:

      1. `Consumer(..., **limits_kwargs(LimitsConsumer.X, <expr>))` — inline.
      2. `Consumer(..., **self._attr)` where the same class assigns
         `self._attr = limits_kwargs(LimitsConsumer.X, ...)` — the prebuilt
         cache, which exists for the two constructions on a host-rate path
         (`_interceptors/_seam.py`'s WebSocket tracker, one per upgrade, and
         `_interceptors/_mcp_stdio.py`'s per-subprocess state). Those two are
         named here on purpose: a later "simplification" that inlines them back
         would make this guard reject the code it is protecting.

    Either way the member must be the row that registers the callee, so a
    projection borrowed from the wrong consumer is a violation and not a pass.
    """
    aliases = _aliases(tree)
    out: list[tuple[int, str, str]] = []
    for call, cls in _calls(tree):
        symbol = _called_symbol(call, aliases)
        consumer = _BY_SYMBOL.get(symbol) if symbol else None
        if consumer is None:
            continue
        prebuilt = _prebuilt_attrs(cls)
        seen: list[str] = []
        for kw in call.keywords:
            if kw.arg is not None:
                continue
            member = _limits_kwargs_member(kw.value)
            if member is None and isinstance(kw.value, ast.Attribute):
                if isinstance(kw.value.value, ast.Name) and kw.value.value.id == "self":
                    member = prebuilt.get(kw.value.attr)
            if member is not None:
                seen.append(member)
        if not seen:
            out.append((call.lineno, symbol or "?", "no projection expanded into this call"))
        elif consumer.name not in seen:
            out.append(
                (
                    call.lineno,
                    symbol or "?",
                    f"projects {seen} but the callee is registered as {consumer.name}",
                )
            )
    return out


def test_registered_consumers_are_only_built_through_the_projection():
    """Every construction of a registered consumer expands the projection.

    This is the guard that would have caught the original defect at the call
    site rather than in a host's memory profile: `context_for` listed four
    registry keywords by hand and one of them did not exist, and nothing about
    a short list of keywords looks wrong.
    """
    found: dict[str, list[tuple[int, str, str]]] = {}
    for rel, tree in _modules().items():
        bad = _projection_violations(tree)
        if bad:
            found[rel] = bad
    assert not found, (
        "these constructions of a registered limits consumer bypass the projection:\n"
        + "\n".join(
            f"  {rel}:{line} {symbol} — {why}"
            for rel, items in sorted(found.items())
            for line, symbol, why in items
        )
        + "\n\nWHY: a call site that spells the keywords itself owns a COPY of the\n"
        "delivery list, and a copy can be short. Expand\n"
        "`**limits_kwargs(LimitsConsumer.X, resolved)` instead, or — on a path\n"
        "that builds one per host event — a `self._…` cache assigned from it."
    )


def test_registered_symbols_are_uniquely_named():
    """One symbol name per row: the scanners match on the name."""
    assert len(_BY_SYMBOL) == len(_LIMIT_DELIVERY), (
        "two rows resolve to the same symbol name, so a call site cannot be "
        "attributed to either: " + str(sorted(_BY_SYMBOL))
    )


# ==========================================================================
# G3b — a native-limits parameter is never dropped at a registered call
# ==========================================================================

#: Callables that take the `to_native()` object and enforce a bound in Rust.
#: The rows of `_LIMIT_DELIVERY` with a `native` parameter are added below, so
#: this table only has to name the ones that take NOTHING else — the parsers
#: and the semantic extraction entry point.
_NATIVE_ONLY: dict[str, str] = {
    "parse_llm_semantics": "wardex_sdk._protocol:parse_llm_semantics",
    "_Http1Tracker": "wardex_sdk._interceptors._trackers:_Http1Tracker",
    "_Http2Tracker": "wardex_sdk._interceptors._trackers:_Http2Tracker",
    "Http1RequestParser": "wardex_sdk._protocol._http1:Http1RequestParser",
    "Http1ResponseParser": "wardex_sdk._protocol._http1:Http1ResponseParser",
    "Http2Parser": "wardex_sdk._protocol._http2:Http2Parser",
    "JsonRpcParser": "wardex_sdk._protocol:JsonRpcParser",
    "WsParser": "wardex_sdk._protocol:WsParser",
}


def _native_targets() -> dict[str, tuple[str, frozenset[str]]]:
    """Symbol -> (dotted target, the parameters that take the native object)."""
    out = {name: (target, frozenset({"limits"})) for name, target in _NATIVE_ONLY.items()}
    for row in _LIMIT_DELIVERY.values():
        if row.native:
            out[row.target.partition(":")[2].split(".")[-1]] = (row.target, row.native)
    return out


def _native_violations(tree: ast.Module) -> list[tuple[int, str, str]]:
    """Registered native calls that pass no limits object, or pass literal None.

    Positional is fine and has to be: three of these are called positionally
    today. The judgement is made by BINDING the AST arguments to the real
    signature rather than by counting them, because "the fifth argument" is a
    fact about one call site and "the `limits` parameter" is a fact about the
    callee.

    The historical failure this pins: `parse_llm_semantics` was once called
    without limits, so `max_decoded_bytes` bounded nothing and a compressed
    body could decompress without a ceiling.
    """
    aliases = _aliases(tree)
    targets = _native_targets()
    out: list[tuple[int, str, str]] = []
    for call, _cls in _calls(tree):
        symbol = _called_symbol(call, aliases)
        if symbol not in targets:
            continue
        target, params = targets[symbol]
        signature = inspect.signature(_resolve(target))
        by_keyword = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
        starred = any(kw.arg is None for kw in call.keywords)
        if any(isinstance(a, ast.Starred) for a in call.args):
            out.append((call.lineno, symbol, "positional unpacking hides the argument list"))
            continue
        for param in sorted(params):
            if param in by_keyword:
                value: ast.AST | None = by_keyword[param]
            elif starred:
                out.append((call.lineno, symbol, f"{param} must be explicit beside a ** expansion"))
                continue
            else:
                try:
                    bound = signature.bind_partial(*call.args, **by_keyword)
                except TypeError as exc:
                    out.append((call.lineno, symbol, f"arguments do not fit the signature: {exc}"))
                    continue
                value = bound.arguments.get(param)
            if value is None:
                out.append((call.lineno, symbol, f"{param} is not passed"))
            elif isinstance(value, ast.Constant) and value.value is None:
                out.append((call.lineno, symbol, f"{param} is literally None"))
    return out


def test_native_limits_are_never_dropped_at_a_registered_call():
    """A Rust-enforced bound only exists if the native object travels with the call."""
    found: dict[str, list[tuple[int, str, str]]] = {}
    for rel, tree in _modules().items():
        bad = _native_violations(tree)
        if bad:
            found[rel] = bad
    assert not found, (
        "these calls drop the native limits object:\n"
        + "\n".join(
            f"  {rel}:{line} {symbol} — {why}"
            for rel, items in sorted(found.items())
            for line, symbol, why in items
        )
        + "\n\nWHY: the parsers fall back to the CORE defaults when handed None,\n"
        "so the call succeeds, the parse succeeds, and the host's configured\n"
        "ceiling is the one thing that does not happen."
    )


# ==========================================================================
# G5 — the process defaults are resolved only where a row says so
# ==========================================================================

#: File -> how many default resolutions it may contain, and why each one is
#: there. A row is a DECLARATION that this component does not take the host's
#: value — which is a decision, not a default. Adding one without a reason is
#: how the tool catalog spent a release resolving its own bound.
_DEFAULT_RESOLUTION_ALLOWED: dict[str, int] = {
    # `resolved()` itself: the merge of overrides onto the core table.
    "_limits.py": 1,
    # `UnitRegistry`'s per-parameter `None` fallback, pinned by
    # `test_units.py::test_bounds_are_resolved_from_the_core_not_a_python_literal`.
    "_assembly/_units.py": 1,
    # `McpToolCatalog`'s core default. The host's value arrives through
    # `apply_bound`, and `test_anthropic_tool_names.py::
    # test_the_bound_comes_from_the_core` pins this half.
    "_adapters/_anthropic_names.py": 1,
    # `ConnTimingStore`'s `cap=None` fallback.
    "_interceptors/_conn_timing.py": 1,
    # `_ProcState.SNIFF_LIMIT` (the class constant), and the interceptor's
    # pre-install projection cache — an interceptor driven without `install()`
    # is a real shape in this suite.
    "_interceptors/_mcp_stdio.py": 2,
    # `_max_streams`' fallback when there is no native object, and
    # `_WebSocketTracker`'s `sample_cap=None` fallback.
    "_interceptors/_trackers.py": 2,
    # The byte seam's pre-install defaults, whose own comment says why.
    "_interceptors/_seam.py": 1,
}


def _default_resolution_violations(tree: ast.Module) -> list[tuple[int, str]]:
    """Every site that resolves the PROCESS defaults rather than a config.

    Three spellings reach the same place and all three count:
    `LimitsConfig().resolved()`, a name bound to a bare `LimitsConfig()` and
    then resolved, and `limits_defaults()` off the native module. What does NOT
    count is `config.limits.resolved()` or a name bound conditionally — those
    carry the user's values when there are any, which is the whole difference.
    """
    fresh: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
            if isinstance(target, ast.Name) and _is_bare_limits_config(value):
                fresh.add(target.id)
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == "limits_defaults":
                out.append((node.lineno, "limits_defaults()"))
                continue
            if name == "resolved" and isinstance(func, ast.Attribute):
                if _is_bare_limits_config(func.value):
                    out.append((node.lineno, "LimitsConfig().resolved()"))
                elif isinstance(func.value, ast.Name) and func.value.id in fresh:
                    out.append((node.lineno, f"{func.value.id}.resolved()"))
        elif isinstance(node, ast.Attribute) and node.attr == "resolved":
            if isinstance(node.value, ast.Name) and node.value.id == "LimitsConfig":
                out.append((node.lineno, "LimitsConfig.resolved"))
    return out


def _is_bare_limits_config(node: ast.AST) -> bool:
    """`LimitsConfig()` with no arguments — the process defaults, spelled out."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LimitsConfig"
        and not node.args
        and not node.keywords
    )


def test_core_defaults_are_resolved_only_where_declared():
    """Resolving the process defaults is a declared act, not an available one.

    The one rule that catches a component resolving its own bound — which is
    what the MCP tool catalog did, invisibly, past four other guards: it needed
    no new field, no new probe and no new parameter to ignore the host's
    configuration completely.
    """
    counted = {
        rel: len(bad)
        for rel, tree in _modules().items()
        if (bad := _default_resolution_violations(tree))
    }
    assert counted == _DEFAULT_RESOLUTION_ALLOWED, (
        f"default resolutions are at {counted}, and the declared table is "
        f"{_DEFAULT_RESOLUTION_ALLOWED}.\n\n"
        + "\n".join(
            f"  {rel}:{line} {what}"
            for rel, tree in sorted(_modules().items())
            for line, what in _default_resolution_violations(tree)
        )
        + "\n\nWHY: `LimitsConfig().resolved()` and `limits_defaults()` always answer\n"
        "with the PROCESS defaults and are indistinguishable from a configured\n"
        "read at the call site. A component that wants the host's value takes it\n"
        "as an argument. A component that genuinely wants the core default adds a\n"
        "row here WITH its reason — which is then a decision somebody made."
    )


# ==========================================================================
# counterexample metatests — a blind scanner is green forever
# ==========================================================================
#
# Every AST guard in this repository owes a pair of these, and the reason is
# the failure mode rather than thoroughness: a scanner whose symbol resolution
# is subtly wrong matches NOTHING, reports NOTHING, and is indistinguishable
# from a clean tree. That is the same shape as the bug the guards are here for
# — a wiring that looks complete because the missing half leaves no trace — so
# each checker below is fed source it must reject and source it must accept.

_BYPASSES = {
    "hand-spelled keywords": """
def build(sink, resolved):
    return UnitRegistry(
        sink=sink,
        max_units=resolved["max_units"],
        max_entries_per_unit=resolved["max_entries_per_unit"],
    )
""",
    "some other mapping": """
def build(sink, other):
    return UnitRegistry(sink=sink, **other)
""",
    "a self attribute that is not a projection": """
class Seam:
    def __init__(self):
        self._not_limits = {"path": "/"}

    def upgrade(self):
        return _WebSocketTracker(**self._not_limits)
""",
    "a cache built in a different class": """
class Owner:
    def __init__(self):
        self._ws_kwargs = limits_kwargs(LimitsConsumer.WS_TRACKER, self._limits)


class Borrower:
    def upgrade(self):
        return _WebSocketTracker(**self._ws_kwargs)
""",
    "an alias import": """
from .._assembly import UnitRegistry as R


def build(sink):
    return R(sink=sink, max_units=1)
""",
    "the wrong consumer's projection": """
def build(sink, resolved):
    return UnitRegistry(sink=sink, **limits_kwargs(LimitsConsumer.SESSION_ASSEMBLER, resolved))
""",
    "a method row bypassed": """
def bind(names, resolved):
    names.apply_bound(max_entries=resolved["max_entries_per_unit"])
""",
}


@pytest.mark.parametrize("shape", sorted(_BYPASSES), ids=lambda s: s.replace(" ", "-"))
def test_g3_sees_a_bypass_however_it_is_written(shape):
    assert _projection_violations(ast.parse(_BYPASSES[shape])), (
        f"the projection scanner did not see: {shape}"
    )


_PROJECTED = {
    "inline, module function": """
def context_for(client, resolved, sink, debug):
    return UnitRegistry(
        sink=sink, debug=debug, **limits_kwargs(LimitsConsumer.UNIT_REGISTRY, resolved)
    )
""",
    "inline, inside a class": """
class Assembler:
    def __init__(self, client, resolved, sink):
        self._units = UnitRegistry(
            sink=sink, **limits_kwargs(LimitsConsumer.UNIT_REGISTRY, resolved)
        )
        self._names = McpToolCatalog()
        self._names.apply_bound(**limits_kwargs(LimitsConsumer.MCP_TOOL_CATALOG, resolved))
""",
    "inline, a bare function call target": """
class Seam:
    def _acquire(self):
        install_shared_timing(**limits_kwargs(LimitsConsumer.CONN_TIMING, self._limits))
""",
    "prebuilt cache, annotated assignment": """
class Seam:
    def __init__(self):
        self._ws_kwargs: dict[str, int] = limits_kwargs(LimitsConsumer.WS_TRACKER, self._limits)

    def upgrade(self, txn):
        return _WebSocketTracker(path="/", limits=self._native_limits, **self._ws_kwargs)
""",
    "prebuilt cache, plain assignment": """
class Mcp:
    def install(self, client):
        self._proc_limits = limits_kwargs(LimitsConsumer.MCP_PROC_STATE, lim.resolved())

    def _wrap(self):
        return _ProcState(**self._proc_limits, limits=self._native_limits, mode=m, debug=d)
""",
    "the bridge receiver": """
class Adapter:
    def install(self, client, resolved):
        self._bridge = _OtelBridgeReceiver(
            **limits_kwargs(LimitsConsumer.OTEL_BRIDGE_RECEIVER, resolved)
        )
        self._assembler = SessionAssembler(
            client, **limits_kwargs(LimitsConsumer.SESSION_ASSEMBLER, resolved)
        )
""",
}


@pytest.mark.parametrize("shape", sorted(_PROJECTED), ids=lambda s: s.replace(" ", "-"))
def test_g3_does_not_flag_the_shape_the_code_actually_writes(shape):
    assert _projection_violations(ast.parse(_PROJECTED[shape])) == [], (
        f"the projection scanner rejected a sanctioned shape: {shape}"
    )


_DROPPED_NATIVE = {
    "omitted entirely": "p = WsParser()",
    "positional literal None": "p = WsParser(None)",
    "keyword literal None": "p = WsParser(limits=None)",
    "positional None among four": "s = _ProcState(sniff, None, mode, debug)",
    "hidden behind a ** expansion": "s = _ProcState(**everything)",
    "dropped at the semantic parse": "sem = parse_llm_semantics(host, path, req, resp)",
}


@pytest.mark.parametrize("shape", sorted(_DROPPED_NATIVE), ids=lambda s: s.replace(" ", "-"))
def test_g3b_sees_a_dropped_native_limits_however_written(shape):
    assert _native_violations(ast.parse(_DROPPED_NATIVE[shape])), (
        f"the native scanner did not see: {shape}"
    )


_KEPT_NATIVE = {
    "positional": "p = WsParser(limits)",
    "keyword": "p = WsParser(limits=self._native_limits)",
    "positional among four": "s = _ProcState(sniff, self._native_limits, mode, debug)",
    "explicit beside a ** expansion": (
        "s = _ProcState(**self._proc_limits, limits=self._native_limits, mode=m, debug=d)"
    ),
    "the semantic parse": (
        "sem = parse_llm_semantics(host, txn.path, req, resp, self._native_limits)"
    ),
}


@pytest.mark.parametrize("shape", sorted(_KEPT_NATIVE), ids=lambda s: s.replace(" ", "-"))
def test_g3b_accepts_positional_native_limits(shape):
    assert _native_violations(ast.parse(_KEPT_NATIVE[shape])) == [], (
        f"the native scanner rejected a sanctioned shape: {shape}"
    )


_FRESH_DEFAULTS = {
    "inline": 'cap = LimitsConfig().resolved()["max_units"]',
    "through a name": "cfg = LimitsConfig()\nr = cfg.resolved()",
    "the native table, subscripted": 'n = _wardex_native.limits_defaults()["max_units"]',
    "the native table, aliased module": (
        "from . import _wardex_native as n\nd = n.limits_defaults()"
    ),
    "an unbound method reference": "f = LimitsConfig.resolved",
}


@pytest.mark.parametrize("shape", sorted(_FRESH_DEFAULTS), ids=lambda s: s.replace(" ", "-"))
def test_g5_sees_a_fresh_default_resolution_however_written(shape):
    assert _default_resolution_violations(ast.parse(_FRESH_DEFAULTS[shape])), (
        f"the default-resolution scanner did not see: {shape}"
    )


_CONFIG_DERIVED = {
    "straight off the config": "r = config.limits.resolved()",
    "the native object": "n = lim.to_native()",
    "a name bound conditionally": (
        "lim = config.limits if config is not None else LimitsConfig()\n"
        "self._limits = lim.resolved()"
    ),
    "an override is not a default": "r = LimitsConfig(max_units=7).resolved()",
}


@pytest.mark.parametrize("shape", sorted(_CONFIG_DERIVED), ids=lambda s: s.replace(" ", "-"))
def test_g5_accepts_a_config_derived_resolution(shape):
    assert _default_resolution_violations(ast.parse(_CONFIG_DERIVED[shape])) == [], (
        f"the default-resolution scanner rejected a config-derived read: {shape}"
    )


# ==========================================================================
# G4 — the delivery census, through a real init(), TWICE
# ==========================================================================


class _Sites(NamedTuple):
    """The objects a real `init()` actually installed. Not doubles."""

    ssl_seam: Any
    socket_seam: Any
    mcp: Any
    client: Any
    transport: Any
    ctx: Any
    adapter: Any


def _read(obj: Any, name: str) -> Any:
    """`getattr`, but a miss is a LOUD failure rather than a silent None.

    A probe that quietly returns nothing when the attribute it names is gone
    turns this whole census green on evidence it never gathered, which is the
    same failure as the wiring bug it watches for.
    """
    if obj is None:
        raise AssertionError(f"the census reached for {name!r} on nothing at all")
    try:
        return getattr(obj, name)
    except AttributeError as exc:  # pragma: no cover — the failure path is the point
        raise AssertionError(
            f"the census probe for {name!r} does not fit {type(obj).__name__} any more: {exc}"
        ) from exc


def _nat(site: str, field: str):
    """One seam's native `Limits` object, read by field."""
    return lambda s: _read(_read(getattr(s, site), "_native_limits"), field)


def _ws_tracker(s: _Sites) -> Any:
    """A WebSocket tracker built the way the seam builds one, per upgrade."""
    from wardex_sdk._interceptors._trackers import _WebSocketTracker

    return _WebSocketTracker(
        path="/",
        deflate=False,
        parent=None,
        start_ns=0,
        limits=_read(s.ssl_seam, "_native_limits"),
        **_read(s.ssl_seam, "_ws_kwargs"),
    )


def _h2_tracker(s: _Sites) -> Any:
    from wardex_sdk._interceptors._trackers import _Http2Tracker

    return _Http2Tracker(_read(s.ssl_seam, "_native_limits"))


def _proc_state(s: _Sites) -> Any:
    from wardex_sdk._interceptors._mcp_stdio import _ProcState

    return _ProcState(**_read(s.mcp, "_proc_limits"))


#: Field -> every place the configured value must have LANDED, each tagged
#: with how it got there: a consumer's `.value` for a projected keyword,
#: `"native"` for the `to_native()` object, `"resolved_map"` for a component
#: that holds the whole resolved mapping.
_DELIVERY: dict[str, tuple[tuple[str, Any], ...]] = {
    "max_headers": (
        ("native", _nat("ssl_seam", "max_headers")),
        ("native", _nat("socket_seam", "max_headers")),
        ("native", _nat("mcp", "max_headers")),
        ("native", lambda s: _read(_read(s.transport, "_limits"), "max_headers")),
    ),
    "max_body_bytes": (
        ("native", _nat("ssl_seam", "max_body_bytes")),
        ("unit_registry", lambda s: _read(_read(s.ctx, "_units"), "_max_record_bytes")),
        ("unit_registry", lambda s: _read(s.ctx, "record_budget")),
        ("resolved_map", lambda s: _read(s.ctx, "limits")["max_body_bytes"]),
    ),
    "max_opaque_body_bytes": (
        ("native", _nat("ssl_seam", "max_opaque_body_bytes")),
        ("native", _nat("mcp", "max_opaque_body_bytes")),
    ),
    "max_stream_buffer_bytes": (
        ("native", _nat("ssl_seam", "max_stream_buffer_bytes")),
        ("native", _nat("mcp", "max_stream_buffer_bytes")),
    ),
    "max_decoded_bytes": (("native", _nat("ssl_seam", "max_decoded_bytes")),),
    "max_streams": (
        ("native", _nat("ssl_seam", "max_streams")),
        ("native", lambda s: _read(_h2_tracker(s), "_latch_cap")),
    ),
    "max_ws_frame_bytes": (("native", _nat("ssl_seam", "max_ws_frame_bytes")),),
    "ws_sample_bytes": (
        ("native", _nat("ssl_seam", "ws_sample_bytes")),
        ("resolved_map", lambda s: _read(s.ssl_seam, "_limits")["ws_sample_bytes"]),
        ("ws_tracker", lambda s: _read(_ws_tracker(s), "_sample_cap")),
    ),
    "max_connections": (
        ("resolved_map", lambda s: _read(s.ssl_seam, "_limits")["max_connections"]),
        ("conn_timing", lambda s: _read(_shared_store(), "_cap")),
    ),
    "max_sessions": (
        ("session_assembler", lambda s: _read(_read(s.adapter, "_assembler"), "_max_sessions")),
        ("otel_bridge_receiver", lambda s: _read(_read(s.adapter, "_bridge"), "_max_sessions")),
    ),
    "max_session_entries": (
        (
            "session_assembler",
            lambda s: _read(_read(s.adapter, "_assembler"), "_max_session_entries"),
        ),
    ),
    "max_units": (("unit_registry", lambda s: _read(_read(s.ctx, "_units"), "_max_units")),),
    "max_entries_per_unit": (
        ("unit_registry", lambda s: _read(_read(s.ctx, "_units"), "_max_entries_per_unit")),
        ("mcp_tool_catalog", lambda s: _read(_read(s.adapter, "_names"), "_max")),
    ),
    "mcp_sniff_bytes": (("mcp_proc_state", lambda s: _read(_proc_state(s), "_sniff_limit")),),
    "max_extra_keys": (("native", _nat("ssl_seam", "max_extra_keys")),),
    "max_parse_backlog": (
        ("finalize_queue", lambda s: _read(_read(s.client, "_finalize"), "_max_jobs")),
    ),
    "max_parse_backlog_bytes": (
        ("finalize_queue", lambda s: _read(_read(s.client, "_finalize"), "_max_bytes")),
    ),
    "max_buffer_spans": (("resolved_map", lambda s: _read(s.client, "_max_buffer_spans")),),
    "max_buffer_bytes": (("resolved_map", lambda s: _read(s.client, "_max_buffer_bytes")),),
    "max_otel_bridge_body_bytes": (
        ("otel_bridge_receiver", lambda s: _read(_read(s.adapter, "_bridge"), "_max_body_bytes")),
    ),
    "max_otel_bridge_spans_per_session": (
        (
            "otel_bridge_receiver",
            lambda s: _read(_read(s.adapter, "_bridge"), "_max_spans_per_session"),
        ),
    ),
    "max_otlp_attribute_bytes": (
        ("native", lambda s: _read(_read(s.transport, "_limits"), "max_otlp_attribute_bytes")),
    ),
    "max_otlp_request_bytes": (
        ("native", lambda s: _read(_read(s.transport, "_limits"), "max_otlp_request_bytes")),
    ),
    "max_link_targets": (
        ("unit_registry", lambda s: _read(_read(s.ctx, "_units"), "_max_link_targets")),
    ),
}


def _shared_store() -> Any:
    from wardex_sdk._interceptors._conn_timing import shared_timing_store

    return shared_timing_store()


#: Two configurations, every field different from the core default AND from
#: each other. Both halves matter: equal to the core proves nothing about
#: delivery, and equal to each other proves nothing about the SECOND init.
_ROUND_A = LimitsConfig(
    max_headers=7,
    max_body_bytes=4096,
    max_opaque_body_bytes=8192,
    max_stream_buffer_bytes=16384,
    max_decoded_bytes=32768,
    max_streams=3,
    max_ws_frame_bytes=2048,
    ws_sample_bytes=1024,
    max_connections=11,
    max_sessions=13,
    max_session_entries=17,
    max_units=19,
    max_entries_per_unit=23,
    mcp_sniff_bytes=512,
    max_extra_keys=3,
    max_parse_backlog=83,
    max_parse_backlog_bytes=2097152,
    max_buffer_spans=29,
    max_buffer_bytes=65536,
    max_otel_bridge_body_bytes=131072,
    max_otel_bridge_spans_per_session=31,
    max_otlp_attribute_bytes=262144,
    max_otlp_request_bytes=524288,
    max_link_targets=37,
)
_ROUND_B = LimitsConfig(
    max_headers=41,
    max_body_bytes=8192,
    max_opaque_body_bytes=16384,
    max_stream_buffer_bytes=32768,
    max_decoded_bytes=65536,
    max_streams=43,
    max_ws_frame_bytes=4096,
    ws_sample_bytes=2048,
    max_connections=47,
    max_sessions=53,
    max_session_entries=59,
    max_units=61,
    max_entries_per_unit=67,
    mcp_sniff_bytes=1024,
    max_extra_keys=5,
    max_parse_backlog=89,
    max_parse_backlog_bytes=4194304,
    max_buffer_spans=71,
    max_buffer_bytes=131072,
    max_otel_bridge_body_bytes=262144,
    max_otel_bridge_spans_per_session=73,
    max_otlp_attribute_bytes=524288,
    max_otlp_request_bytes=1048576,
    max_link_targets=79,
)

_ENV = (
    "WARDEX_API_KEY",
    "WARDEX_ENDPOINT",
    "WARDEX_SERVICE_NAME",
    "WARDEX_RELEASE",
    "WARDEX_ENVIRONMENT",
    "WARDEX_DEBUG",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
)


def _sites() -> _Sites:
    """The live objects behind the current `init()`. Missing one is a failure.

    Everything here comes off the runtime rather than out of a constructor, so
    a value that reaches a component only because a test handed it there does
    not count — which is what the probe table next door cannot tell apart, and
    says so in its own docstring.
    """
    from wardex_sdk import _hub, _runtime

    rt = _runtime.runtime()
    seams = rt.interceptors._installed
    adapters = rt.adapters._installed
    contexts = rt.adapters._contexts
    for name in ("ssl", "socket", "mcp_stdio"):
        assert name in seams, f"the {name} seam did not install; the census cannot observe it"
    assert "anthropic_agent_sdk" in adapters, (
        "the Agent SDK adapter did not install, so four consumers have no site"
    )
    client = _hub.get_client()
    assert client is not None
    return _Sites(
        ssl_seam=seams["ssl"],
        socket_seam=seams["socket"],
        mcp=seams["mcp_stdio"],
        client=client,
        transport=_read(client, "_transport"),
        ctx=contexts["anthropic_agent_sdk"],
        adapter=adapters["anthropic_agent_sdk"],
    )


def _observe(limits: LimitsConfig) -> dict[str, list[tuple[str, int]]]:
    """One `init()` round trip, every probe read inside it."""
    import wardex_sdk
    from wardex_sdk._config import AdaptersConfig, AnthropicAgentSdkConfig

    wardex_sdk.init(
        intercept=True,
        limits=limits,
        adapters=AdaptersConfig(anthropic_agent_sdk=AnthropicAgentSdkConfig(otel_bridge=True)),
    )
    try:
        sites = _sites()
        return {
            field: [(tag, probe(sites)) for tag, probe in probes]
            for field, probes in _DELIVERY.items()
        }
    finally:
        wardex_sdk.close()


@pytest.fixture(scope="module")
def census():
    """Two round trips, observed. The SECOND is the one that found defect (C).

    A module singleton that survives `close()` — the shared connection-timing
    store — latches its bound at construction, so a census that ran one init
    would have called that delivery green while a re-initialising host got the
    first number for the life of the process. Hence two, with different values.

    The reset either side is not politeness: this census reads process
    globals, so any earlier test in this interpreter that ran `init()` could
    have latched one of them first. The environment is scrubbed here rather
    than through the autouse fixture next door because that one is
    function-scoped and this fixture is built before it.
    """
    from wardex_sdk import _hub

    saved = {name: os.environ.pop(name, None) for name in _ENV}
    _hub.reset_for_test()
    try:
        return {
            "first init": (_ROUND_A.resolved(), _observe(_ROUND_A)),
            "second init": (_ROUND_B.resolved(), _observe(_ROUND_B)),
        }
    finally:
        _hub.reset_for_test()
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


def test_the_two_census_rounds_can_prove_something():
    """Both configurations differ from the core AND from each other.

    Without this the census could pass on values nobody delivered: equal to the
    core default proves nothing about delivery, and equal to each other proves
    nothing about the second init.
    """
    core = LimitsConfig().resolved()
    a, b = _ROUND_A.resolved(), _ROUND_B.resolved()
    for field in LimitsConfig.__dataclass_fields__:
        assert a[field] != core[field], f"{field}: round A equals the core default"
        assert b[field] != core[field], f"{field}: round B equals the core default"
        assert a[field] != b[field], f"{field}: the two rounds are the same number"


@pytest.mark.parametrize("field", sorted(_DELIVERY))
def test_every_limit_reaches_its_enforcement_site(field, census):
    """The configured value is what the enforcement site holds — on both inits."""
    for label, (expected, observed) in census.items():
        sites = observed[field]
        assert sites, (
            f"{field} ({label}): no site observed it at all, so nothing delivers it.\n"
            "A limit with no delivery is a field a user can set that changes nothing."
        )
        want = expected[field]
        wrong = [(tag, value) for tag, value in sites if value != want]
        assert not wrong, (
            f"{field} ({label}): configured {want}, but "
            + ", ".join(f"{tag} holds {value}" for tag, value in wrong)
            + ".\n\nWHY: the value reached `LimitsConfig` and stopped short of the\n"
            "component that enforces it. That is invisible from the outside — the\n"
            "config object keeps reporting the number the user asked for."
        )


def test_delivery_census_covers_every_registered_consumer():
    """Every field has a probe, and every (consumer, keyword) row has one TAGGED.

    Splitting the census by field alone was not enough: a new consumer could
    join an already-probed field and be observed by nobody, which is how a
    delivery guard goes green over a delivery that does not happen.
    """
    assert set(_DELIVERY) == set(LimitsConfig.__dataclass_fields__), (
        "fields with no probe: "
        f"{sorted(set(LimitsConfig.__dataclass_fields__) - set(_DELIVERY))}; "
        f"probes for no field: {sorted(set(_DELIVERY) - set(LimitsConfig.__dataclass_fields__))}"
    )
    tags = {field: {tag for tag, _ in probes} for field, probes in _DELIVERY.items()}
    for consumer, row in _LIMIT_DELIVERY.items():
        for kwarg, field in row.delivers.items():
            assert consumer.value in tags[field], (
                f"{consumer.value} takes {kwarg} from {field}, and no probe observes "
                f"{field} AT that consumer (probes tagged: {sorted(tags[field])})"
            )


def test_a_broken_probe_fails_loudly():
    """A probe that stops fitting must raise, never read as an empty observation.

    The whole census rests on this: `getattr(obj, name, None)` would turn a
    renamed attribute into "nothing to report", and nothing to report is
    indistinguishable from a clean run.
    """
    with pytest.raises(AssertionError, match="does not fit"):
        _read(LimitsConfig(), "_max_record_bytez")
    with pytest.raises(AssertionError, match="nothing at all"):
        _read(None, "_max_units")
