"""gen_ai semantics — an LLM parse result mapped onto `GenAIAttributes`.

Pure functions over whatever `protocol.parse_llm_semantics` returned. The seam
supplies the object; this module decides what it means in gen_ai terms.
"""

from __future__ import annotations

from typing import Any

from .._enums import OperationName, ProviderName
from .._types import GenAIAttributes

_OPERATION_MAP = {"chat": OperationName.CHAT, "embeddings": OperationName.EMBEDDINGS}
_PROVIDER_MAP = {"openai": ProviderName.OPENAI, "anthropic": ProviderName.ANTHROPIC}


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
    stops = tuple(sem.stop_sequences) if sem.stop_sequences else None
    finishes = tuple(sem.finish_reasons) if sem.finish_reasons else None
    return GenAIAttributes(
        operation=_OPERATION_MAP.get(sem.operation, sem.operation),
        provider=_PROVIDER_MAP.get(sem.provider, sem.provider),
        request_model=sem.request_model,
        response_model=sem.response_model,
        response_id=sem.response_id,
        input_tokens=sem.input_tokens,
        output_tokens=sem.output_tokens,
        cache_read_input_tokens=sem.cache_read_input_tokens,
        cache_creation_input_tokens=sem.cache_creation_input_tokens,
        reasoning_output_tokens=sem.reasoning_output_tokens,
        temperature=sem.temperature,
        max_tokens=sem.max_tokens,
        top_p=sem.top_p,
        top_k=sem.top_k,
        seed=sem.seed,
        frequency_penalty=sem.frequency_penalty,
        presence_penalty=sem.presence_penalty,
        choice_count=sem.choice_count,
        stop_sequences=stops,
        stream=sem.stream,
        finish_reasons=finishes,
        # Suppressed when the response half yielded nothing. The Rust parser
        # sets `output_type` unconditionally, so a refused call would otherwise
        # ship `output_type="text"` next to no tokens, no response model and no
        # output messages — a span asserting it produced text output when the
        # provider produced an error. That is the same false claim this call
        # path exists to stop telling.
        output_type=sem.output_type if has_core_semantics(sem) else None,
    )
