"""Where a `GenAIAttributes` may be constructed — an AST census (G2).

The inclusive-totals invariant is enforced at construction in Rust
(`TokenUsage::new` refuses to compile without an `InputConvention`) and
counted at the emission choke point (`SpanDraft.set_gen_ai`). What neither
covers is a NEW Python construction site quietly appearing and becoming an
un-reviewed third implementation of "what do these numbers mean" — exactly
how the P3 path escaped the first time: it never went through the type the
guard was on.

So the construction sites are a frozen list. Adding one is allowed — but the
commit that does it must touch this file, and the diff review then asks the
one question that matters: "where is this site's usage normalized?"
"""

from __future__ import annotations

import ast
from pathlib import Path

import wardex_sdk

#: Every module (package-relative) that may construct `GenAIAttributes`,
#: with its number of construction sites.
#:
#: * `_semantics/_genai.py` — `build_gen_ai`, the P1/P2 mapping; usage comes
#:   from `LlmSemantics.usage` (Rust-normalized).
#: * `_adapters/_assembler.py` — the P3 chat draft; usage comes from
#:   `AgentStreamEvent` (Rust-normalized).
#:
#: (`dataclasses.replace` on an existing block is not a construction site:
#: it can only start from a block one of these two produced.)
_FROZEN_SITES = {
    "_semantics/_genai.py": 1,
    "_adapters/_assembler.py": 1,
}


def _construction_sites() -> dict[str, int]:
    root = Path(wardex_sdk.__file__).parent
    found: dict[str, int] = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        count = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            named = (isinstance(func, ast.Name) and func.id == "GenAIAttributes") or (
                isinstance(func, ast.Attribute) and func.attr == "GenAIAttributes"
            )
            if named:
                count += 1
        if count:
            found[path.relative_to(root).as_posix()] = count
    return found


def test_gen_ai_construction_sites_match_the_frozen_list():
    found = _construction_sites()
    assert found == _FROZEN_SITES, (
        f"GenAIAttributes construction sites changed: {found} != {_FROZEN_SITES}. "
        "A new site is a new place the usage-inclusivity contract must hold — "
        "say in this file where that site's numbers are normalized, then "
        "update the frozen list."
    )


def test_the_census_scanner_can_see_a_site():
    """The counter-example meta-test: a scanner that cannot see the shape it
    guards is a green light wired to nothing."""
    sample = "x = GenAIAttributes(operation='chat')\ny = mod.GenAIAttributes(operation='chat')"
    tree = ast.parse(sample)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "GenAIAttributes")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "GenAIAttributes")
        )
    ]
    assert len(calls) == 2
