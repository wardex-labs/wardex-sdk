"""gen_ai semantics — an LLM parse result mapped onto `GenAIAttributes`.

Pure functions over whatever `protocol.parse_llm_semantics` returned. The seam
supplies the object; this module decides what it means in gen_ai terms.

Three tables partition what this module consumes off `LlmSemantics`:
`_GEN_AI_FIELDS` (copied 1:1 onto `GenAIAttributes`), the transformed getters
(`build_gen_ai`'s enum maps and tuple conversions), and `_PROVIDER_EXTRAS`
(registry-namespaced scalars that ride as extras). The structural guard
(`tests/test_llm_semantics_surface.py`) holds the partition exhaustive against
`dir(LlmSemantics)`, so a getter added in Rust cannot go unconsumed here and a
name added here cannot miss its getter.
"""

from __future__ import annotations

from typing import Any

from .._assembly import counters
from .._enums import OperationName, ProviderName
from .._types import EmbeddingsAttributes, GenAIAttributes

_OPERATION_MAP = {"chat": OperationName.CHAT, "embeddings": OperationName.EMBEDDINGS}
_PROVIDER_MAP = {"openai": ProviderName.OPENAI, "anthropic": ProviderName.ANTHROPIC}

#: LlmSemantics getter -> GenAIAttributes field, SAME name on both sides,
#: value copied as-is. Everything needing a conversion is spelled out in
#: `build_gen_ai` below and audited by `_GEN_AI_TRANSFORMED` in the guard.
_GEN_AI_FIELDS: tuple[str, ...] = (
    "request_model",
    "response_model",
    "response_id",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_output_tokens",
    "temperature",
    "max_tokens",
    "top_p",
    "top_k",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "choice_count",
    "stream",
    "reasoning_level",
    "previous_response_id",
    "response_status",
)

#: LlmSemantics getter -> registry extra key (provider-specific scalars in the
#: semconv `openai.*` namespace — declared in `_vocab._REGISTRY_PREFIXES`).
_PROVIDER_EXTRAS: tuple[tuple[str, str], ...] = (
    ("api_type", "openai.api.type"),
    ("request_service_tier", "openai.request.service_tier"),
    ("response_service_tier", "openai.response.service_tier"),
    ("system_fingerprint", "openai.response.system_fingerprint"),
)

#: The provider-usage mirror family: every scalar leaf of the provider's usage
#: tree, spelling preserved (`wardex.usage.<dotted path>`). U1 holds this
#: family to a set equality with the provider's own leaves — which is why the
#: drop-count key below lives OUTSIDE it: a wardex-made integer inside the
#: family would break the equality, and a provider could one day ship a
#: `usage.dropped_count` leaf of its own under the same spelling.
USAGE_EXTRA_PREFIX = "wardex.usage."
USAGE_DROPPED_KEY = "wardex.usage_leaves.dropped_count"


def has_core_semantics(sem: Any) -> bool:
    """True if at least one core semantic (model, tokens) is present.

    Every field here is RESPONSE-side, which is the whole of what this answers:
    did the reply carry gen_ai truth. A call the provider refused carries none
    of them and is still an LLM call — `identifies_llm_call` is that question.
    """
    return (
        sem.input_tokens is not None
        or sem.output_tokens is not None
        or sem.response_model is not None
    )


def identifies_llm_call(sem: Any) -> bool:
    """True if the REQUEST named a provider, an operation and a model.

    All three, and by truthiness rather than `is not None`: `provider` and
    `operation` cross from Rust as non-Optional strings, so an emptiness check
    is the only one that means anything, and a lone `provider` would make this
    read as "the parser produced something" — which is `sem is not None`, and
    lets a batches endpoint with no model in the body pass for a chat call.

    `getattr` with defaults, unusually for this codebase, because this runs
    inside the seam's fail-open `try` and is driven by hand-built doubles in the
    capture-policy tests; raising here would fail OPEN and capture everything.
    """
    return (
        bool(getattr(sem, "provider", ""))
        and bool(getattr(sem, "operation", ""))
        and getattr(sem, "request_model", None) is not None
    )


def build_gen_ai(sem: Any) -> GenAIAttributes:
    """Convert LlmSemantics → GenAIAttributes."""
    # Normalization conditions cross the FFI as VALUES (there is no
    # Rust->Python counter channel) and are tallied here, where the
    # diagnostics registry lives. `getattr` for the same reason as
    # `identifies_llm_call`: hand-built doubles drive this function inside
    # the seam's fail-open try, and the native getters are pinned by their
    # own tests, so a default cannot mask a missing binding.
    if getattr(sem, "usage_totals_unpaired", False):
        counters.bump("semantics.build_gen_ai.usage_totals_unpaired")
    if getattr(sem, "usage_overflowed", False):
        counters.bump("semantics.build_gen_ai.usage_overflowed")
    copied = {name: getattr(sem, name) for name in _GEN_AI_FIELDS}
    stops = tuple(sem.stop_sequences) if sem.stop_sequences else None
    finishes = tuple(sem.finish_reasons) if sem.finish_reasons else None
    encodings = tuple(sem.encoding_formats) if sem.encoding_formats else None
    return GenAIAttributes(
        operation=_OPERATION_MAP.get(sem.operation, sem.operation),
        provider=_PROVIDER_MAP.get(sem.provider, sem.provider),
        stop_sequences=stops,
        finish_reasons=finishes,
        encoding_formats=encodings,
        # Suppressed when the response half yielded nothing. The Rust parser
        # sets `output_type` on parsed requests, so a refused call would
        # otherwise ship `output_type="text"` next to no tokens, no response
        # model and no output messages — a span asserting it produced text
        # output when the provider produced an error. That is the same false
        # claim this call path exists to stop telling.
        output_type=sem.output_type if has_core_semantics(sem) else None,
        **copied,
    )


def provider_extras(sem: Any) -> list[tuple[str, str | int | float | bool]]:
    """Registry extras + the usage mirror, as (key, scalar) pairs.

    `openai.*` scalars from `_PROVIDER_EXTRAS`, then every usage leaf under
    `wardex.usage.<provider path>`. The dropped-count key is NOT built here —
    the seam sets it beside the marker, so the marker, the count and the
    diagnostics bump travel as one fact or not at all.

    `getattr` with defaults so a hand-built double (`test_capture_policy`'s
    `_Sem`) that answers None for everything still yields `[]`.
    """
    out: list[tuple[str, str | int | float | bool]] = []
    for attr, key in _PROVIDER_EXTRAS:
        value = getattr(sem, attr, None)
        if value is not None:
            out.append((key, value))
    for path, value in getattr(sem, "usage_leaves", None) or ():
        out.append((USAGE_EXTRA_PREFIX + path, value))
    return out


def embeddings_attrs(sem: Any) -> EmbeddingsAttributes | None:
    """The embeddings block, when the request declared a dimension count."""
    dimensions = getattr(sem, "embedding_dimensions", None)
    if dimensions is None:
        return None
    return EmbeddingsAttributes(dimension_count=dimensions)
