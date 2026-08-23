"""G1 — every public getter of the native `LlmSemantics` is consumed, and
every consumer's table names a real getter.

The failure this closes: a getter added in Rust with no Python consumer is a
fact the parser extracts and the SDK silently discards (the pre-Responses
usage fields were exactly that class of loss), while a table row with no
getter is a consumer reading None forever. The partition is asserted in BOTH
directions, and the audit sets live here rather than in the wheel — they are
test data, not runtime data.
"""

from __future__ import annotations

import ast
import pathlib

from wardex_sdk import _wardex_native
from wardex_sdk._semantics._genai import _GEN_AI_FIELDS, _PROVIDER_EXTRAS

#: Getters `build_gen_ai` consumes WITH a conversion — enum maps
#: (provider/operation), tuple() rebuilds, and the output_type suppression
#: rule. Everything consumed 1:1 is in `_GEN_AI_FIELDS` (runtime data, since
#: the copy loop reads it); everything here is spelled out in the function
#: body instead, which is what this set audits.
_GEN_AI_TRANSFORMED = frozenset(
    {
        "provider",
        "operation",
        "output_type",
        "stop_sequences",
        "finish_reasons",
        "encoding_formats",
    }
)

#: Getters the seam (or a helper it calls: `embeddings_attrs`,
#: `provider_extras`' usage mirror, `build_gen_ai`'s FFI-condition tallies)
#: consumes directly — the last column that makes the partition of
#: `dir(LlmSemantics)` exhaustive.
_SEAM_CONSUMED = frozenset(
    {
        "decoded_response",
        "reassembled_from_stream",
        "stream_terminated",
        "output_messages",
        "tool_args_unparsed",
        "output_messages_has_unmapped",
        "input_messages",
        "input_messages_has_unmapped",
        "system_instructions",
        "usage_leaves",
        "usage_dropped_count",
        "embedding_dimensions",
        # FFI-value diagnostics (there is no Rust->Python counter channel):
        # read by build_gen_ai, tallied into the diagnostics registry.
        "usage_totals_unpaired",
        "usage_overflowed",
    }
)

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "wardex_sdk"
_CONSUMER_SOURCES = (
    _SRC / "_interceptors" / "_seam.py",
    _SRC / "_semantics" / "_genai.py",
)


def _surface() -> set[str]:
    return {n for n in dir(_wardex_native.protocol.LlmSemantics) if not n.startswith("_")}


def test_the_surface_partition_is_exhaustive_and_disjoint():
    """dir(LlmSemantics) == fields U transformed U provider-extras U seam.

    Both directions: a getter outside every table is an unconsumed fact; a
    table name with no getter is a consumer that can only ever read None.
    The four sets are disjoint so no name is claimed by two consumers with
    different conversions.
    """
    surface = _surface()
    fields = set(_GEN_AI_FIELDS)
    extras = {attr for attr, _ in _PROVIDER_EXTRAS}
    claimed = fields | _GEN_AI_TRANSFORMED | extras | _SEAM_CONSUMED
    assert surface == claimed, (
        f"unconsumed getters: {sorted(surface - claimed)}; "
        f"table names with no getter: {sorted(claimed - surface)}"
    )
    sets = [fields, _GEN_AI_TRANSFORMED, extras, _SEAM_CONSUMED]
    for i, a in enumerate(sets):
        for b in sets[i + 1 :]:
            assert not (a & b), f"claimed twice: {sorted(a & b)}"
    # The partition arithmetic the integration record pins: 20 + 6 + 4 + 14.
    assert (len(fields), len(_GEN_AI_TRANSFORMED), len(extras), len(_SEAM_CONSUMED)) == (
        20,
        6,
        4,
        14,
    )
    assert len(surface) == 44


def test_every_seam_consumed_name_appears_in_a_consumer_source():
    """Membership in `_SEAM_CONSUMED` is a claim about real code: the name
    must occur as `sem.<name>` / `<obj>.<name>` attribute access or as a
    `getattr(x, "<name>", ...)` literal in the seam or the gen_ai module —
    the census's AST technique, so renaming a consumer site breaks the claim
    loudly instead of leaving a stale audit row.
    """
    seen: set[str] = set()
    for source in _CONSUMER_SOURCES:
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                seen.add(node.attr)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                seen.add(node.args[1].value)
    missing = _SEAM_CONSUMED - seen
    assert not missing, (
        f"claimed as seam-consumed but never read in "
        f"{[p.name for p in _CONSUMER_SOURCES]}: {sorted(missing)}"
    )
