"""Structural conformance — design §3.1 (dependency direction) and §10.4 (C-S1..C-S5).

These tests read the source with `ast`; they never import the modules they
judge. They exist because of the single most instructive failure in the prior
art: an emitter helper was *extracted* into one place, and then a second
constructor and eight streaming branches were left free to bypass it. An
extraction that leaves the bypass reachable is not an extraction. So the rules
below are the extraction — the docstrings in `assembly/` are only its
description.

Two kinds of assertion live here, and the difference matters when one fails:

  HARD RULE — true today, must stay true forever. If you broke one, the fix is
  in your change, not here.

  BUDGET — a ratchet over code that predates the `assembly/` package. Each file
  gets the count it has today; the test fails if a count goes UP or if a file
  not on the list acquires its first occurrence. These numbers may only be
  lowered. The migration (design §11) drives each budget to zero, at which
  point the budget dict is deleted and the rule becomes hard. Adding a line to
  a budget to make your change pass is the one edit this file is not for.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "wardex_sdk"
_PKG = "wardex_sdk"


# --------------------------------------------------------------------------
# source access
# --------------------------------------------------------------------------


def _modules() -> dict[str, ast.Module]:
    """Every module in the package, keyed by path relative to `wardex_sdk/`."""
    out: dict[str, ast.Module] = {}
    for path in sorted(_SRC.rglob("*.py")):
        rel = path.relative_to(_SRC).as_posix()
        out[rel] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return out


def _known_modules() -> set[str]:
    """Dotted names of every module and package inside `wardex_sdk`.

    Used to tell `from .._types import TraceId` (an edge to `wardex_sdk._types`,
    plus a symbol) apart from `from .. import _hub` (an edge to
    `wardex_sdk._hub`). Without it the symbol would be mistaken for a module.
    """
    names = {_PKG}
    for path in _SRC.rglob("*.py"):
        parts = list(path.relative_to(_SRC).parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][: -len(".py")]
        names.add(".".join([_PKG, *parts]))
    return names


def _known_packages() -> set[str]:
    """Dotted names of every package (directory with `__init__.py`) in wardex_sdk.

    A symbol imported from a package may be a re-export that hides the module
    that defines it; a symbol imported from a leaf module cannot.
    """
    return {
        ".".join([_PKG, *path.relative_to(_SRC).parts[:-1]]) for path in _SRC.rglob("__init__.py")
    }


def _imported_modules(rel: str, tree: ast.Module) -> set[str]:
    """Absolute dotted names of every wardex module `rel` imports.

    Relative imports are resolved against `rel`'s own package, and
    `from .. import _hub` is resolved to `wardex_sdk._hub` rather than left as
    a bare name — the rule is about the module graph, not about spelling.

    A symbol pulled out of a PACKAGE — `from .. import trace`, where `trace` is
    re-exported by `wardex_sdk/__init__.py` rather than being a module — yields
    the pseudo-target `wardex_sdk:trace`. Such an import produces no module edge
    at all (the defining module is `_tracing.py`, which is never named), so
    without this a layering rule that forbids `wardex_sdk._tracing` is one
    re-export away from meaning nothing. The colon keeps it un-confusable with a
    real dotted name, so an allowlist has to name it deliberately. Symbols taken
    from a leaf MODULE need no such treatment: `from .._types import TraceId`
    already carries the auditable edge `wardex_sdk._types`.

    Dynamic imports (`importlib.import_module("wardex_sdk.x")`, `__import__`)
    are resolved too when the argument is a string literal, and reported as
    `<dynamic>` when it is not: `adapters/__init__.py` already imports
    `importlib.util`, and design §3.2 gives `assembly/_patchset.py` the job of
    reaching modules lazily, so this idiom is one edit away from being a hole in
    every rule below.
    """
    known = _known_modules()
    packages = _known_packages()
    pkg = [_PKG, *rel.split("/")[:-1]]  # package containing this module
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in known:
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[: len(pkg) - node.level + 1]
                target = [*base, *(node.module.split(".") if node.module else [])]
            elif node.module and (node.module == _PKG or node.module.startswith(_PKG + ".")):
                target = node.module.split(".")
            else:
                continue  # stdlib or third party
            dotted = ".".join(target)
            if dotted in known:
                found.add(dotted)
            for alias in node.names:
                # `from .. import _hub` — the imported name may itself be a module.
                candidate = f"{dotted}.{alias.name}"
                if candidate in known:
                    found.add(candidate)
                elif alias.name != "*" and dotted in packages:
                    found.add(f"{dotted}:{alias.name}")
        elif isinstance(node, ast.Call) and _is_dynamic_import(node):
            found.update(_dynamic_import_target(node))
    return found


def _is_dynamic_import(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id == "__import__":
        return True
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "import_module"
        and isinstance(func.value, ast.Name)
        and func.value.id in ("importlib", "il")
    )


def _dynamic_import_target(node: ast.Call) -> set[str]:
    if not node.args:
        return set()
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        name = first.value
        if name == _PKG or name.startswith(_PKG + "."):
            return {name}
        return set()
    return {"<dynamic>"}  # a non-literal target cannot be audited at all


def _imported_names(tree: ast.Module) -> set[str]:
    """Names this module pulls in by name, including under TYPE_CHECKING.

    A type-only import still puts the symbol in the adapter author's
    vocabulary, and C-S1 is about making the bypass unnameable (I5).
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(a.name for a in node.names)
            if node.module:
                names.add(node.module.rsplit(".", 1)[-1])
        elif isinstance(node, ast.Import):
            names.update(a.name.rsplit(".", 1)[-1] for a in node.names)
    return names


def _local_names(tree: ast.Module, symbol: str) -> set[str]:
    """Every local name bound to `symbol` by an import, including aliases.

    `from .._types import TraceId as _T` must not launder a rule. The scan is
    per-module because the alias is.
    """
    names = {symbol}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names if a.name == symbol)
    return names


def _module_aliases(rel: str, tree: ast.Module) -> set[str]:
    """Every local name bound to a wardex MODULE object, including aliases.

    A rule that only understands `TraceId.generate()` is defeated by
    `_types.TraceId.generate()`, and `from .. import _hub, _wardex_native` is the
    house form in this codebase (`adapters/_assembler.py:17`,
    `interceptors/_mcp_stdio.py:20`, `interceptors/_trackers.py:14`) — so that
    spelling is one token away from code already in the tree. The predicates
    below therefore accept a module-qualified receiver as well as a bare name.

    Resolution is the real thing, not a naming heuristic: a name is collected
    only when the module it would bind actually exists in the package. `_PKG` is
    always included so `import wardex_sdk` + `wardex_sdk._types.TraceId` is
    covered without needing the intermediate binding.
    """
    known = _known_modules()
    pkg = [_PKG, *rel.split("/")[:-1]]
    names = {_PKG}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in known:
                    # `import wardex_sdk._types` binds `wardex_sdk`;
                    # `import wardex_sdk._types as t` binds `t`.
                    names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[: len(pkg) - node.level + 1]
                target = [*base, *(node.module.split(".") if node.module else [])]
            elif node.module and (node.module == _PKG or node.module.startswith(_PKG + ".")):
                target = node.module.split(".")
            else:
                continue
            dotted = ".".join(target)
            for alias in node.names:
                if f"{dotted}.{alias.name}" in known:
                    names.add(alias.asname or alias.name)
    return names


def _reaches_symbol(node: ast.expr, symbol: str, bound: set[str], modules: set[str]) -> bool:
    """True if `node` names `symbol`, however it is spelled.

    Accepts `TraceId` and its import aliases (`bound`), plus any module-qualified
    path ending in `.TraceId` whose leftmost receiver is a wardex module name
    (`_types.TraceId`, `wardex_sdk._types.TraceId`). The rule is about reaching
    the type, not about how the reach is written.
    """
    if isinstance(node, ast.Name):
        return node.id in bound
    if not (isinstance(node, ast.Attribute) and node.attr == symbol):
        return False
    receiver: ast.expr = node.value
    while isinstance(receiver, ast.Attribute):
        receiver = receiver.value
    return isinstance(receiver, ast.Name) and receiver.id in modules


def _calls_traceid_generate(rel: str, tree: ast.Module):
    bound = _local_names(tree, "TraceId")
    modules = _module_aliases(rel, tree)

    def predicate(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate"
            and _reaches_symbol(node.func.value, "TraceId", bound, modules)
        )

    return predicate


def _constructs_internal_span(rel: str, tree: ast.Module):
    bound = _local_names(tree, "InternalSpan")
    modules = _module_aliases(rel, tree)

    def predicate(node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and _reaches_symbol(
            node.func, "InternalSpan", bound, modules
        )

    return predicate


def _passes_parent_span_id(rel: str, tree: ast.Module):
    def predicate(node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and any(
            kw.arg == "parent_span_id" for kw in node.keywords
        )

    return predicate


def _calls_sink(rel: str, tree: ast.Module):
    def predicate(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("capture_span", "capture_snapshot")
        )

    return predicate


# Calls that make a handler self-reporting. `guard` is the sanctioned swallow;
# `bump` is its counter; the rest are the logging spellings already in the tree
# (`interceptors/_seam.py` `parser_disable_log`, stdlib logging, debug prints).
_REPORTING_CALLS = frozenset(
    {
        "guard",
        "bump",
        "parser_disable_log",
        "print",
        "log",
        "debug",
        "info",
        "warn",
        "warning",
        "error",
        "exception",
        "critical",
    }
)


def _is_silent_swallow(rel: str, tree: ast.Module):
    # Needs no per-module context (no import can alias `except: pass`), but keeps
    # the predicate-factory shape the other rules use so _tally stays uniform.
    return _silent_swallow_node


def _silent_swallow_node(node: ast.AST) -> bool:
    """An `except` handler that leaves NO trace that it ran.

    Deliberately not "the body is literally `pass`". Two things go wrong with the
    literal reading. It misses `except Exception: return None`, which is the same
    swallow with different punctuation and is already the shape of 13 handlers in
    `adapters/`/`interceptors/`. Worse, it makes the C-S4 ratchet satisfiable by
    laundering: rewriting `pass` as `return None` lowers the count, turns the
    budget green, and fixes nothing — which would defeat the migration this
    budget exists to drive (design §11 step 9).

    So the question asked is the one the rule's WHY actually asks: after this
    handler runs, is there any evidence anywhere? A handler is NOT silent if it
    re-raises (the failure reaches the host) or calls something that records
    (`guard`, `counters.bump`, any logging spelling). Everything else — `pass`,
    `return`, `return b""`, `self._disabled = True` — is a swallow that is
    indistinguishable from wardex not being installed.
    """
    if not isinstance(node, ast.ExceptHandler):
        return False
    for child in ast.walk(node):
        if isinstance(child, ast.Raise):
            return False
        if isinstance(child, ast.Call):
            func = child.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in _REPORTING_CALLS:
                return False
    return True


def _tally(make_predicate, *, under: tuple[str, ...] | None = None) -> dict[str, int]:
    """{module path: occurrences}, omitting zeros. `make_predicate` is built per module."""
    counts: dict[str, int] = {}
    for rel, tree in _modules().items():
        if under is not None and not rel.startswith(under):
            continue
        predicate = make_predicate(rel, tree)
        n = sum(1 for node in ast.walk(tree) if predicate(node))
        if n:
            counts[rel] = n
    return counts


def _assert_within_budget(
    actual: dict[str, int], budget: dict[str, int], rule: str, why: str
) -> None:
    over = {rel: (n, budget.get(rel, 0)) for rel, n in actual.items() if n > budget.get(rel, 0)}
    assert not over, (
        f"{rule} regressed.\n"
        + "\n".join(f"  {rel}: {n} occurrences, budget {b}" for rel, (n, b) in sorted(over.items()))
        + f"\n\nWHY: {why}\n"
        "These budgets are a ratchet over code that predates wardex_sdk.assembly.\n"
        "They may only be LOWERED. If you need a new occurrence, you need the\n"
        "assembly/ entry point instead — that is the whole point of the rule.\n"
        "See design §10.4 and the migration table in §11."
    )


# --------------------------------------------------------------------------
# design §3.1 — dependency direction (HARD RULES)
# --------------------------------------------------------------------------

# assembly/ sits above the leaf vocabulary and the scope layer and below
# everything that observes anything. This is a positive allowlist so that a new
# module in the package forces an explicit decision instead of silently
# widening the layer.
_ASSEMBLY_MAY_IMPORT = frozenset(
    {
        f"{_PKG}",  # exact: the package object, e.g. `from .. import _hub`
        f"{_PKG}._types",
        f"{_PKG}._enums",
        f"{_PKG}._limits",
        f"{_PKG}._hash",
        f"{_PKG}._config",
        f"{_PKG}._scope",
        f"{_PKG}._hub",
        f"{_PKG}._client",
    }
)

# Whole subpackages assembly/ may reach into. Deliberately separate from the
# exact list: `wardex_sdk` itself must NOT act as a prefix, or every module in
# the SDK would be allowed and the rule would assert nothing.
_ASSEMBLY_MAY_IMPORT_PACKAGES = (f"{_PKG}.context",)

_LAYERING_WHY = (
    "assembly/ is the layer everything else depends ON. If it may import an\n"
    "observer (interceptors/, adapters/, protocol/, semantics/), the one-way\n"
    "arrow in design §3.1 becomes a cycle, the 'adapters cannot reach\n"
    "InternalSpan' guarantee (I5) becomes reachable through assembly, and the\n"
    "package stops being a boundary anyone can reason about."
)


def test_assembly_imports_only_leaf_and_scope_layers():
    violations: list[str] = []
    for rel, tree in _modules().items():
        if not rel.startswith("assembly/"):
            continue
        for target in sorted(_imported_modules(rel, tree)):
            if target.startswith(f"{_PKG}.assembly"):
                continue
            if target in _ASSEMBLY_MAY_IMPORT:
                continue
            if any(
                target == pkg or target.startswith(pkg + ".")
                for pkg in _ASSEMBLY_MAY_IMPORT_PACKAGES
            ):
                continue
            violations.append(f"  {rel} imports {target}")
    assert not violations, (
        "assembly/ reached above its layer:\n"
        + "\n".join(sorted(violations))
        + "\n\nWHY: "
        + _LAYERING_WHY
        + "\n\nIf the dependency is genuinely a leaf, add it to"
        " _ASSEMBLY_MAY_IMPORT in this file and say why in the PR."
    )


def test_adapters_and_interceptors_do_not_import_each_other():
    violations: list[str] = []
    for a, b in (("adapters/", f"{_PKG}.interceptors"), ("interceptors/", f"{_PKG}.adapters")):
        for rel, tree in _modules().items():
            if not rel.startswith(a):
                continue
            for target in sorted(_imported_modules(rel, tree)):
                if target == b or target.startswith(b + "."):
                    violations.append(f"  {rel} imports {target}")
    assert not violations, (
        "adapters/ and interceptors/ are siblings, not a stack:\n"
        + "\n".join(sorted(violations))
        + "\n\nWHY: they are two independent observers of the same run. A direct\n"
        "edge between them is how one layer ends up owning the other's spans,\n"
        "which is exactly the double-instrumentation mess design §8 exists to\n"
        "prevent. Anything they need to share belongs in assembly/."
    )


def test_assembly_internal_modules_stay_private():
    public = sorted(
        rel
        for rel in _modules()
        if rel.startswith("assembly/")
        and not rel.rsplit("/", 1)[-1].startswith("_")
        and rel != "assembly/__init__.py"
    )
    assert not public, (
        f"assembly/ modules must stay underscore-private: {public}\n\n"
        "WHY: assembly.__all__ is a semver-stable boundary (design §3.1). A\n"
        "module without a leading underscore invites `from wardex_sdk.assembly\n"
        "import somemodule`, which freezes an internal layout we intend to keep\n"
        "moving through the migration."
    )


def test_assembly_public_surface_is_declared_and_resolvable():
    import wardex_sdk.assembly as assembly

    assert assembly.__all__, "assembly must declare __all__ — it is the stable surface"
    assert list(assembly.__all__) == sorted(assembly.__all__), "keep __all__ sorted"
    missing = [name for name in assembly.__all__ if not hasattr(assembly, name)]
    assert not missing, f"assembly.__all__ names nothing importable: {missing}"


# --------------------------------------------------------------------------
# C-S1 — adapters/ cannot name the span machinery
# --------------------------------------------------------------------------

_FORBIDDEN_IN_ADAPTERS = frozenset(
    {"InternalSpan", "SpanContext", "TraceId", "SpanId", "Client", "_hub", "capture_state_snapshot"}
)

# The same rule at module granularity, and it is the half that carries the
# weight. A symbol-name check is defeated by one line — `from .. import _types`
# binds no forbidden NAME, and `_types` is where InternalSpan, SpanContext,
# TraceId and SpanId all live, one attribute access away. That spelling is the
# house form here (`adapters/_assembler.py:17`, `interceptors/_mcp_stdio.py:20`,
# `interceptors/_trackers.py:14`), so it is not a contrived bypass. `_hub` was
# caught before only by the accident of being spelled like a symbol.
#
# `wardex_sdk` itself is on the list: `import wardex_sdk` hands an adapter the
# whole package object, and with it every private module by attribute.
_FORBIDDEN_MODULES_IN_ADAPTERS = frozenset(
    {
        _PKG,
        f"{_PKG}._types",
        f"{_PKG}._hub",
        f"{_PKG}._client",
        f"{_PKG}._tracing",
        f"{_PKG}._scope",
    }
)

# Pre-assembly debt. Each entry is the set of forbidden names and module targets
# that file still reaches today; the sets may only SHRINK. Emptied by migration
# steps 6-8, when the Anthropic adapter moves onto AdapterContext + assembly/ and
# this dict is deleted along with the budget machinery.
_CS1_DEBT: dict[str, frozenset[str]] = {
    "adapters/__init__.py": frozenset({"Client", f"{_PKG}._client"}),
    "adapters/_base.py": frozenset({"Client", f"{_PKG}._client"}),
    "adapters/_registry.py": frozenset({"Client", f"{_PKG}._client"}),
    "adapters/_anthropic_agent_sdk.py": frozenset(
        {"Client", f"{_PKG}._client", f"{_PKG}._types", f"{_PKG}._tracing"}
    ),
    "adapters/_assembler.py": frozenset(
        {
            "InternalSpan",
            "SpanContext",
            "SpanId",
            "TraceId",
            "_hub",
            _PKG,
            f"{_PKG}._hub",
            f"{_PKG}._types",
        }
    ),
}


def test_adapters_do_not_import_span_machinery():
    violations: list[str] = []
    for rel, tree in _modules().items():
        if not rel.startswith("adapters/"):
            continue
        used = _imported_names(tree) & _FORBIDDEN_IN_ADAPTERS
        used |= _imported_modules(rel, tree) & _FORBIDDEN_MODULES_IN_ADAPTERS
        new = used - _CS1_DEBT.get(rel, frozenset())
        if new:
            violations.append(f"  {rel} imports {sorted(new)}")
    assert not violations, (
        "C-S1: an adapter module named the span machinery:\n"
        + "\n".join(sorted(violations))
        + "\n\nWHY: this is the product claim, enforced structurally. An adapter\n"
        "that can reach TraceId/SpanContext/InternalSpan/Client can mint a\n"
        "parent edge out of a framework's run id, and the causal tree stops\n"
        "coming from in-process context propagation. Forbidding the IMPORT is\n"
        "stronger than auditing call sites: with the type absent from the\n"
        "module's namespace the bypass has no name to call (design I5, §10.4).\n"
        "Adapters get parentage by asking wardex_sdk.assembly for it."
    )


# --------------------------------------------------------------------------
# C-S2 / C-S3 — one parentage source, one span constructor
# --------------------------------------------------------------------------

_CS2_BUDGET = {
    "_tracing.py": 1,
    "adapters/_assembler.py": 1,
    "interceptors/_mcp_stdio.py": 1,
    "interceptors/_seam.py": 2,
}

_CS3_SPAN_BUDGET = {
    "_tracing.py": 1,
    "adapters/_assembler.py": 4,
    "interceptors/_mcp_stdio.py": 1,
    "interceptors/_seam.py": 2,
}

_CS3_PARENT_BUDGET = {
    "_tracing.py": 1,
    "adapters/_assembler.py": 5,
    "interceptors/_mcp_stdio.py": 1,
    "interceptors/_seam.py": 2,
}


def test_trace_id_is_generated_only_in_parentage():
    """C-S2, in the form that is already true: inside assembly/, exactly one site."""
    inside = _tally(_calls_traceid_generate, under=("assembly/",))
    assert inside == {"assembly/_parentage.py": 1}, (
        f"C-S2: TraceId.generate() inside assembly/ is {inside}, expected exactly\n"
        "{'assembly/_parentage.py': 1}.\n\n"
        "WHY: a new trace id is the statement 'this work has no parent'. One\n"
        "call site is what makes that statement auditable — and it is why the\n"
        "'started a new trace' case can be told apart from the 'expected a\n"
        "parent and lost it' case at all (design I1, I4)."
    )
    _assert_within_budget(
        {k: v for k, v in _tally(_calls_traceid_generate).items() if not k.startswith("assembly/")},
        _CS2_BUDGET,
        "C-S2 (TraceId.generate outside assembly/)",
        "every extra generator is another place a subtree can silently detach\n"
        "into its own trace. Migration step 1 rewrites these six sites onto\n"
        "assembly.resolve_parentage() and empties this budget.",
    )


def test_internal_span_is_constructed_in_one_place():
    inside = _tally(_constructs_internal_span, under=("assembly/",))
    assert inside == {}, (
        f"C-S3: assembly/ constructs InternalSpan at {inside}. Until\n"
        "assembly/_builder.py lands (migration step 3), the assembly package\n"
        "resolves parentage and does not build spans.\n\n"
        "WHY: one constructor is what keeps every span carrying correlation,\n"
        "capture_sources and capture_integrity. Today's two execute_tool\n"
        "variants differ precisely because they are built in two places."
    )
    _assert_within_budget(
        _tally(_constructs_internal_span),
        _CS3_SPAN_BUDGET,
        "C-S3 (InternalSpan constructed outside assembly/_builder.py)",
        "a second constructor is how spans start disagreeing about which\n"
        "forensic fields are mandatory. Migration step 3 routes all of these\n"
        "through assembly.SpanDraft.",
    )


def test_parent_span_id_is_passed_only_from_assembly():
    """C-S3, second half: `parent_span_id=` is assembly/'s keyword.

    Inside assembly/ it is unrestricted — that is where the edge is decided.
    """
    _assert_within_budget(
        {k: v for k, v in _tally(_passes_parent_span_id).items() if not k.startswith("assembly/")},
        _CS3_PARENT_BUDGET,
        "C-S3 (parent_span_id= outside assembly/)",
        "a parent edge written outside assembly/ is an edge nobody resolved:\n"
        "no Evidence, no confidence, no limitation marker when it was a guess\n"
        "(design I1, I4). Ask assembly.resolve_parentage() for a Parentage and\n"
        "let it fill the field.",
    )


# --------------------------------------------------------------------------
# C-S4 — swallow, but never in silence
# --------------------------------------------------------------------------

# design §10.4 scopes C-S4 to adapters/ and interceptors/; assembly/ is held to
# zero from day one. (`_lifecycle.py`, `context/_inject.py`, `context/_asgi.py`,
# `context/_propagate.py` and `context/_wsgi.py` also swallow silently today and
# are out of the rule's declared scope — migration step 9.)
#
# These numbers went UP once, on the commit that widened `_silent_swallow_node`
# from "the body is literally `pass`" to "the handler leaves no trace" (44 -> 57).
# That is the one legitimate reason a budget may rise: the code did not regress,
# the predicate stopped being blind. Reading it as licence to raise a number is
# the misuse this file's header forbids — from here they only go down.
_CS4_BUDGET = {
    "adapters/__init__.py": 1,
    "adapters/_anthropic_agent_sdk.py": 14,
    "adapters/_assembler.py": 2,
    "interceptors/_conn_timing.py": 10,
    "interceptors/_mcp_stdio.py": 13,
    "interceptors/_seam.py": 5,
    "interceptors/_socket.py": 6,
    "interceptors/_ssl.py": 6,
}


def test_no_silent_swallow_in_assembly():
    inside = _tally(_is_silent_swallow, under=("assembly/",))
    assert inside == {}, (
        f"C-S4: silent swallow inside assembly/ at {inside}.\n\n"
        "WHY: assembly/ owns the ONE authorized swallow — assembly._diag.guard(),\n"
        "which always counts and logs with a traceback under config.debug. A\n"
        "bare `except Exception: pass` in the module that defines the rule turns\n"
        "an SDK bug into 'wardex just doesn't capture this' with no evidence\n"
        "anywhere (design I6)."
    )


def test_silent_swallows_do_not_spread():
    _assert_within_budget(
        _tally(_is_silent_swallow, under=("adapters/", "interceptors/")),
        _CS4_BUDGET,
        "C-S4 (silent swallow in adapters/ or interceptors/)",
        "wardex must not raise into the host, but a swallow that leaves no\n"
        "counter and no debug traceback is indistinguishable from wardex not\n"
        "being installed. Wrap the block in assembly._diag.guard(where=...)\n"
        "instead; migration step 9 converts the ones already here.",
    )


# --------------------------------------------------------------------------
# C-S5 — one sink
# --------------------------------------------------------------------------

_CS5_BUDGET = {
    "__init__.py": 1,
    "_tracing.py": 1,
    "adapters/_assembler.py": 4,
    "interceptors/_mcp_stdio.py": 2,
    "interceptors/_seam.py": 2,
}


def test_sink_is_not_called_from_assembly_yet():
    inside = _tally(_calls_sink, under=("assembly/",))
    assert inside == {}, (
        f"C-S5: assembly/ calls the client sink at {inside}. When it does, it is\n"
        "from assembly/_emit.py and nowhere else (migration step 3).\n\n"
        "WHY: one sink is where the capture-mode gate, the limitation markers\n"
        "and 'never hold a wardex lock while emitting' (I11) can be enforced\n"
        "once instead of per caller."
    )
    _assert_within_budget(
        _tally(_calls_sink),
        _CS5_BUDGET,
        "C-S5 (capture_span/capture_snapshot outside assembly/_emit.py)",
        "every direct sink call is a span that skipped the gate and the shared\n"
        "policy, and a place where capture_span can be reached while holding an\n"
        "SDK lock — the reentrancy hazard I11 names.",
    )


# --------------------------------------------------------------------------
# negative controls — the rules above must FAIL on a real bypass
# --------------------------------------------------------------------------
#
# A structural rule that nobody has watched fail is a rule nobody knows the
# shape of. Every predicate here was, at some point in this file's history,
# green against source that did the exact thing it forbids: `_types.TraceId`
# reached the type without naming it, `except Exception: return None` swallowed
# without naming `pass`. These cases pin the spellings that used to escape, so
# the next widening of a predicate is a test change and not a discovery.


def _parse(source: str) -> ast.Module:
    return ast.parse(source)


def _handler(body: str) -> str:
    """A function with one `except Exception:` handler whose body is `body`."""
    return f"def f():\n    try:\n        g()\n    except Exception:\n        {body}\n"


_BYPASS_SPELLINGS = {
    "bare symbol": "from .._types import TraceId, InternalSpan",
    "aliased symbol": "from .._types import TraceId as _T, InternalSpan as _S",
    "module attribute": "from .. import _types",
    "aliased module": "from .. import _types as _t",
    "absolute module": "import wardex_sdk._types",
    "package attribute": "import wardex_sdk",
    "from-package module": "from wardex_sdk import _types",
}


def _bypass_source(header: str) -> str:
    """The same bypass written against whatever `header` bound."""
    if "InternalSpan" in header and " as " not in header:
        body = "def f():\n    return TraceId.generate(), InternalSpan(1, 2)\n"
    elif " as _T" in header:
        body = "def f():\n    return _T.generate(), _S(1, 2)\n"
    elif header.endswith("_types as _t"):
        body = "def f():\n    return _t.TraceId.generate(), _t.InternalSpan(1, 2)\n"
    elif header == "import wardex_sdk._types" or header == "import wardex_sdk":
        body = (
            "def f():\n"
            "    return wardex_sdk._types.TraceId.generate(), "
            "wardex_sdk._types.InternalSpan(1, 2)\n"
        )
    else:
        body = "def f():\n    return _types.TraceId.generate(), _types.InternalSpan(1, 2)\n"
    return header + "\n\n\n" + body


@pytest.mark.parametrize("spelling", sorted(_BYPASS_SPELLINGS))
def test_every_spelling_of_the_span_machinery_bypass_is_seen(spelling):
    """C-S2/C-S3: reaching the type counts, however the reach is written."""
    rel = "adapters/_probe.py"
    tree = _parse(_bypass_source(_BYPASS_SPELLINGS[spelling]))

    traces = [n for n in ast.walk(tree) if _calls_traceid_generate(rel, tree)(n)]
    spans = [n for n in ast.walk(tree) if _constructs_internal_span(rel, tree)(n)]

    assert len(traces) == 1, f"C-S2 blind to {spelling!r}: {_BYPASS_SPELLINGS[spelling]}"
    assert len(spans) == 1, f"C-S3 blind to {spelling!r}: {_BYPASS_SPELLINGS[spelling]}"


@pytest.mark.parametrize(
    "spelling", ["module attribute", "absolute module", "package attribute", "from-package module"]
)
def test_c_s1_sees_a_module_import_not_only_a_symbol_import(spelling):
    """C-S1: `from .. import _types` is the bypass, and it names no symbol."""
    rel = "adapters/_probe.py"
    tree = _parse(_bypass_source(_BYPASS_SPELLINGS[spelling]))

    named = _imported_names(tree) & _FORBIDDEN_IN_ADAPTERS
    modules = _imported_modules(rel, tree) & _FORBIDDEN_MODULES_IN_ADAPTERS

    assert not named, "precondition: this spelling deliberately names no forbidden symbol"
    assert modules, (
        f"C-S1 blind to {spelling!r}: {_BYPASS_SPELLINGS[spelling]} — an adapter\n"
        "can reach TraceId/SpanContext/InternalSpan through the module object."
    )


@pytest.mark.parametrize(
    "body",
    [
        "pass",
        "...",
        "return None",
        "return",
        'return b""',
        "return False",
        "self._disabled = True",
    ],
)
def test_c_s4_sees_a_swallow_that_leaves_no_trace_however_written(body):
    tree = _parse(_handler(body))

    handlers = [n for n in ast.walk(tree) if _silent_swallow_node(n)]

    assert len(handlers) == 1, f"C-S4 blind to a handler whose body is {body!r}"


@pytest.mark.parametrize(
    "body",
    [
        "raise",
        "raise RuntimeError from None",
        "counters.bump('x')",
        "logger.warning('x')",
        "parser_disable_log('x')",
        "with guard('x'):\n            pass",
    ],
)
def test_c_s4_does_not_flag_a_handler_that_reports_itself(body):
    tree = _parse(_handler(body))

    assert not [n for n in ast.walk(tree) if _silent_swallow_node(n)], (
        f"C-S4 false positive on a self-reporting handler: {body!r}"
    )


@pytest.mark.parametrize(
    "call",
    [
        'importlib.import_module("wardex_sdk.interceptors._seam")',
        '__import__("wardex_sdk.adapters._assembler")',
    ],
)
def test_dynamic_imports_are_resolved_like_static_ones(call):
    tree = _parse(f"import importlib\n\n_m = {call}\n")

    found = _imported_modules("assembly/_probe.py", tree)

    assert any(
        t.startswith(f"{_PKG}.") and "interceptors" in t or "adapters" in t for t in found
    ), f"a layering rule can be escaped by writing the import as {call}"


def test_a_non_literal_dynamic_import_is_reported_as_unauditable():
    tree = _parse('import importlib\n\n_m = importlib.import_module("wardex_sdk." + name)\n')

    assert "<dynamic>" in _imported_modules("assembly/_probe.py", tree)


def test_a_re_exported_symbol_from_a_package_is_not_invisible():
    """`from .. import trace` names no module, and `_tracing.py` is above assembly/."""
    tree = _parse("from .. import capture_state_snapshot, trace\n")

    found = _imported_modules("assembly/_probe.py", tree)

    assert f"{_PKG}:trace" in found
    assert f"{_PKG}:capture_state_snapshot" in found
