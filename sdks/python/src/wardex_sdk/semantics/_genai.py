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
    """True if at least one core semantic (model, tokens) is present."""
    return (
        sem.input_tokens is not None
        or sem.output_tokens is not None
        or sem.response_model is not None
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
        output_type=sem.output_type,
    )
