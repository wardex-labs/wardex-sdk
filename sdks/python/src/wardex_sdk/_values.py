"""The rule for every value a host writes onto a span, checked at the host's call.

The exporter encodes what the host wrote one batch later, on the worker, and a
value of the wrong type there used to fail the batch (every other request's
spans with it) or to ship as a note naming the key with the value dropped. Each
rule here runs where the host wrote the value instead, so a mistake raises on
the host's own line with what is accepted, and what passes is what the exporter
can encode. An enum field takes its member or the member's value as a string.

Rank 0: imports nothing from the package but the enums, so `_types` can run
these from its dataclasses' `__post_init__` and `_scope` can share the
attribute rule with the span API.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping
from enum import Enum
from typing import Any, TypeVar

from ._enums import AgentType, OperationName, SpanKind, StatusCode, ToolExecutionType

EnumT = TypeVar("EnumT", bound=Enum)

#: What an attribute value is once `attribute_value` accepted it.
AttributeValue = str | int | float | bool

#: The wire's attribute integers are signed 64-bit.
_INT_MIN, _INT_MAX = -(2**63), 2**63 - 1

_ATTRIBUTE_ACCEPTS = (
    "a str, int, float or bool, a list, tuple or dict of JSON values "
    "(sent as its JSON text), or None to remove the attribute"
)


def attribute_value(api: str, key: object, value: object) -> AttributeValue | None:
    """The value `api` (`set_attribute`, `set_tag`) stores for `key`, or None to remove it.

    * the key must be a `str`;
    * a str, float or bool is kept as it is, and so is an int the wire's
      signed 64-bit integer holds (a wider one is a `ValueError`: send
      `str(value)`); any other number (an object with `__index__` or
      `__float__`, such as a numpy scalar) is stored as the int or float it
      converts to, read here so a conversion that fails fails on this line;
    * a list, tuple or dict is sent as its JSON text, compact and with sorted
      keys, so the same value always produces the same string: the wire has no
      list type for a host attribute. Its size is bounded where every
      attribute's is, at export (`LimitsConfig.max_otlp_attribute_bytes`);
    * None means "remove the attribute": the caller drops what was set under
      `key` instead of sending a null.

    Anything else is a `TypeError` naming what is accepted. Only the value's
    type is read, except by `json.dumps` over a container the host sent and by
    the conversion of a number of a type other than int or float.
    """
    if not isinstance(key, str):
        raise TypeError(
            f"{api}(): the attribute key must be a str, not {_type_name(key)}; "
            f"write {api}(str(key), value)"
        )
    kind = type(value)
    if kind is str or kind is float or kind is bool:
        return value  # type: ignore[return-value]
    if kind is int:
        if _INT_MIN <= value <= _INT_MAX:  # type: ignore[operator]
            return value  # type: ignore[return-value]
        return _wire_int(api, key, value)  # type: ignore[arg-type]
    if value is None:
        return None
    if issubclass(kind, (list, tuple, dict)):
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{api}({key!r}, ...): the {kind.__name__} is not JSON-serialisable ({exc}); "
                f"the value must be {_ATTRIBUTE_ACCEPTS}"
            ) from None
    if issubclass(kind, str):
        return value  # type: ignore[return-value]
    if not issubclass(kind, (bytes, bytearray)) and (
        hasattr(kind, "__index__") or hasattr(kind, "__float__")
    ):
        number = _read_number(api, key, kind, value)
        return number if type(number) is float else _wire_int(api, key, number)
    raise TypeError(
        f"{api}({key!r}, ...): got a {kind.__name__}; the value must be "
        f"{_ATTRIBUTE_ACCEPTS}. Convert it first, e.g. str(value)"
    )


def _wire_int(api: str, key: str, value: int) -> int:
    if _INT_MIN <= value <= _INT_MAX:
        return value
    raise ValueError(
        f"{api}({key!r}, ...): an int of {value.bit_length()} bits does not fit the "
        "64-bit signed integer an attribute carries; send it as text with str(value)"
    )


def _read_number(api: str, key: str, kind: type, value: Any) -> int | float:
    """A number-like host object as an int (`__index__`) or else a float.

    Both are tried because some scalar types declare `__index__` and refuse it
    (a numpy bool); a `TypeError` naming the type when neither reads.
    """
    failure = "no conversion"
    if hasattr(kind, "__index__"):
        try:
            return operator.index(value)
        except Exception as exc:
            failure = type(exc).__name__
    if hasattr(kind, "__float__"):
        try:
            return float(value)
        except Exception as exc:
            failure = type(exc).__name__
    raise TypeError(
        f"{api}({key!r}, ...): the {kind.__name__} looks like a number but reading it "
        f"as one raised {failure}; pass int(value), float(value) or str(value)"
    ) from None


def event_attributes(api: str, attributes: object) -> tuple[tuple[str, AttributeValue], ...]:
    """An event's attributes, each by `attribute_value`'s rule; a None value is left out."""
    if attributes is None:
        return ()
    if not isinstance(attributes, Mapping):
        raise TypeError(
            f"{api}() attributes must be a dict of str keys, not {_type_name(attributes)}"
        )
    pairs = []
    for key, value in attributes.items():
        stored = attribute_value(api, key, value)
        if stored is not None:
            pairs.append((key, stored))
    return tuple(sorted(pairs))


def enum_member(what: str, enum: type[EnumT], value: object) -> EnumT:
    """`value` as a member of `enum`: the member itself, or its value as a string.

    The string is matched case-insensitively against the members' values
    (`"client"` and `"CLIENT"` are `SpanKind.CLIENT`). An unknown string is a
    `ValueError` and any other type a `TypeError`, both naming every accepted value.
    """
    if isinstance(value, enum):
        return value
    accepted = ", ".join(repr(member.value) for member in enum)
    if isinstance(value, str):
        folded = value.lower()
        for member in enum:
            if member.value == folded:
                return member
        raise ValueError(f"{what}: {value!r} is not one of {accepted} (or {enum.__name__}.*)")
    raise TypeError(
        f"{what} must be a {enum.__name__} member or one of {accepted}, not {_type_name(value)}"
    )


# -- the typed blocks' fields --------------------------------------------------

_GEN_AI_TEXT = (
    "request_model",
    "response_model",
    "response_id",
    "response_status",
    "reasoning_level",
    "previous_response_id",
    "tool_definitions_hash",
    "prompt_name",
)
_GEN_AI_INTS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_output_tokens",
    "max_tokens",
    "seed",
    "choice_count",
)
_GEN_AI_FLOATS = (
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "time_to_first_chunk_s",
)
_GEN_AI_TEXT_TUPLES = ("stop_sequences", "encoding_formats", "finish_reasons")


def check_gen_ai(attrs: Any) -> None:
    """`GenAIAttributes.__post_init__`: refuse what the exporter cannot encode.

    Every block an LLM call produces passes here, so the common case (each
    field already its declared type or None) costs one inline type test per
    field; only a field that fails it pays for a call. Integers and floats in
    another numeric type are converted; a list of strings becomes a tuple.
    """
    cls = "GenAIAttributes"
    _check_label(cls, "operation", attrs.operation, optional=False)
    _check_label(cls, "provider", attrs.provider)
    _check_label(cls, "output_type", attrs.output_type)
    for name in _GEN_AI_TEXT:
        value = getattr(attrs, name)
        if value is not None and type(value) is not str:
            _check_text(cls, name, value)
    for name in _GEN_AI_INTS:
        value = getattr(attrs, name)
        if value is not None and type(value) is not int:
            _set_int(attrs, cls, name)
    for name in _GEN_AI_FLOATS:
        value = getattr(attrs, name)
        if value is not None and type(value) is not float:
            _set_float(attrs, cls, name)
    for name in _GEN_AI_TEXT_TUPLES:
        if getattr(attrs, name) is not None:
            _set_text_tuple(attrs, cls, name)
    if attrs.stream is not None and type(attrs.stream) is not bool:
        raise TypeError(f"{cls}.stream must be a bool or None, not {_type_name(attrs.stream)}")
    if isinstance(attrs.system_instructions, str):
        object.__setattr__(attrs, "system_instructions", attrs.system_instructions.encode())
    elif not isinstance(attrs.system_instructions, bytes):
        raise TypeError(
            f"{cls}.system_instructions must be bytes or str (JSON text), "
            f"not {_type_name(attrs.system_instructions)}"
        )


def check_agent(attrs: Any) -> None:
    """`AgentAttributes.__post_init__`."""
    cls = "AgentAttributes"
    _check_text(cls, "name", attrs.name, optional=False)
    for name in ("id", "description", "version", "parent_agent"):
        _check_text(cls, name, getattr(attrs, name))
    if attrs.agent_type is not None:
        _set_member(attrs, cls, "agent_type", AgentType)


def check_tool(attrs: Any) -> None:
    """`ToolAttributes.__post_init__`. `type` stays free text or a `ToolType`."""
    cls = "ToolAttributes"
    _check_text(cls, "name", attrs.name, optional=False)
    _check_text(cls, "call_id", attrs.call_id)
    _check_text(cls, "description", attrs.description)
    _check_label(cls, "type", attrs.type)
    _set_member(attrs, cls, "execution_type", ToolExecutionType)


def _type_name(value: object) -> str:
    return type(value).__name__


def _check_text(cls: str, name: str, value: object, *, optional: bool = True) -> None:
    if isinstance(value, str) or (optional and value is None):
        return
    accepted = "a str or None" if optional else "a str"
    raise TypeError(f"{cls}.{name} must be {accepted}, not {_type_name(value)}")


def _check_label(cls: str, name: str, value: object, *, optional: bool = True) -> None:
    """A field the wire reads as an enum member's value or as free text."""
    if isinstance(value, (str, Enum)) or (optional and value is None):
        return
    raise TypeError(
        f"{cls}.{name} must be an enum member or a str{' or None' if optional else ''}, "
        f"not {_type_name(value)}"
    )


def _set_int(obj: object, cls: str, name: str) -> None:
    value = getattr(obj, name)
    if not isinstance(value, bool):
        try:
            object.__setattr__(obj, name, operator.index(value))
            return
        except TypeError:
            pass
    raise TypeError(f"{cls}.{name} must be an int or None, not {_type_name(value)}")


def _set_float(obj: object, cls: str, name: str) -> None:
    value = getattr(obj, name)
    if not isinstance(value, (bool, str, bytes)):
        try:
            object.__setattr__(obj, name, float(value))
            return
        except (TypeError, ValueError):
            pass
    raise TypeError(f"{cls}.{name} must be a number or None, not {_type_name(value)}")


def _set_text_tuple(obj: object, cls: str, name: str) -> None:
    value = getattr(obj, name)
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        if type(value) is not tuple:
            object.__setattr__(obj, name, tuple(value))
        return
    raise TypeError(
        f"{cls}.{name} must be a tuple or list of str, or None, not {_type_name(value)}"
        + (f" (write ({value!r},) for one value)" if isinstance(value, str) else "")
    )


def _set_member(obj: object, cls: str, name: str, enum: type[Enum]) -> None:
    value = getattr(obj, name)
    if type(value) is not enum:
        object.__setattr__(obj, name, enum_member(f"{cls}.{name}", enum, value))


# -- the span API's fields -------------------------------------------------------


def text(what: str, value: object, *, optional: bool = False) -> Any:
    if isinstance(value, str) or (optional and value is None):
        return value
    raise TypeError(
        f"{what} must be a str{' or None' if optional else ''}, not {_type_name(value)}"
    )


def block(what: str, cls: type, value: object, *, optional: bool = True) -> Any:
    """`value` if it is a `cls` (a typed attribute block); None too when optional."""
    if isinstance(value, cls) or (optional and value is None):
        return value
    raise TypeError(
        f"{what} must be a wardex.{cls.__name__}{' or None' if optional else ''}, "
        f"not {_type_name(value)}"
    )


def operation(what: str, value: object) -> Any:
    if value is None or isinstance(value, (OperationName, str)):
        return value
    raise TypeError(f"{what} must be an OperationName, a str or None, not {_type_name(value)}")


#: The wire's timestamps are unsigned 64-bit nanosecond counts.
_TIMESTAMP_MAX = 2**64 - 1


def timestamp(what: str, value: object) -> int:
    """A span instant: an int of nanoseconds since the epoch the wire can hold.

    Past 2**64 - 1 is a unit slip (`time.time_ns() * 1000`, or seconds read as
    microseconds), a `ValueError` here rather than a batch the exporter cannot
    encode.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        if 0 <= value <= _TIMESTAMP_MAX:
            return int(value)
        raise ValueError(
            f"{what} must be nanoseconds since the epoch between 0 and 2**64 - 1, as "
            f"time.time_ns() returns, not {value!r}"
        )
    got = repr(value) if type(value) is float else _type_name(value)
    raise TypeError(
        f"{what} must be a non-negative int of nanoseconds since the epoch "
        f"(time.time_ns()), not {got}"
    )


def raw(what: str, value: object) -> bytes:
    """`input_data`/`output_data` as the bytes the exporter sends; a str is UTF-8.

    `backslashreplace` keeps a lone surrogate visible as its escape rather than
    raising out of the host's line or replacing it with a character it was not.
    """
    if type(value) is bytes:
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8", "backslashreplace")
    raise TypeError(
        f"{what} must be bytes or str, not {_type_name(value)}; for a JSON value "
        "use Span.set_input(value) / Span.set_output(value)"
    )


def payload(what: str, value: object) -> bytes:
    """str and bytes as `raw`; anything else as its JSON text, UTF-8."""
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        return raw(what, value)
    try:
        encoded = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{what} takes str, bytes or a JSON-serialisable value, and this "
            f"{_type_name(value)} is not one ({exc}); pass json.dumps(value, default=str) "
            "or str(value)"
        ) from None
    return encoded.encode("utf-8", "backslashreplace")


def unshippable(span: Any) -> str | None:
    """The first core field of a finished span the exporter cannot encode, or None.

    The span API's setters refuse every wrong type, so this answers None for a
    span built through them. It is the backstop for one built some other way:
    the exporter encodes these fields for a whole batch at once, and one it
    cannot read fails the batch, every other request's spans with it.
    """
    for field_name, expected in (
        ("name", str),
        ("kind", SpanKind),
        ("status", StatusCode),
        ("status_message", str),
        ("input_data", bytes),
        ("output_data", bytes),
    ):
        if not isinstance(getattr(span, field_name), expected):
            return field_name
    for field_name in ("start_time_ns", "end_time_ns"):
        value = getattr(span, field_name)
        if type(value) is not int or not 0 <= value <= _TIMESTAMP_MAX:
            return field_name
    for field_name in ("error_type", "workflow_name"):
        value = getattr(span, field_name)
        if value is not None and not isinstance(value, str):
            return field_name
    for event in span.events:
        if not isinstance(event.name, str) or type(event.timestamp_ns) is not int:
            return "events"
    return None


def span_name(api: str, name: object) -> None:
    """A span name is a str; a function here is `@wardex.span` written as a decorator."""
    if callable(name) and not isinstance(name, str):
        raise TypeError(
            f"wardex.{api}() opens a block and takes its name first, and got the "
            f"{type(name).__name__} {getattr(name, '__name__', '')!r}: `@wardex.{api}` "
            "is not a decorator. Decorate the function with @wardex.workflow/@agent/"
            f'@step/@tool instead, or open the block inside it: `with wardex.{api}("name"):`.'
        )
    text(f"wardex.{api}() name", name)


# -- a host's exception class ----------------------------------------------------


def exception_type(cls: type) -> str:
    """The class's fully qualified name, bare for a builtin — OTel Python's rule.

    TOTAL, because it runs outside any guard on the host's way out of a block,
    over a class that is the host's: a metaclass can make `__module__` a
    property, or refuse every lookup, and what it raises would leave the block
    in place of the host's own exception. Python's own lookup goes first, so
    the name matches the last line of the stack trace; when it raises or gives
    something that is not a string, the names come from `type`'s own
    descriptors, which read what the class statement stored and run nothing
    of the host's. Only exact `str` values are compared or joined, so a string
    subclass's `__eq__` or `__format__` never runs either.
    """
    try:
        module, qualname = cls.__module__, cls.__qualname__
    except Exception:
        module = qualname = None
    module = _exact_str(module)
    if module is None:
        module = _exact_str(_stored_module(cls))
    qualname = _exact_str(qualname)
    if qualname is None:
        # Never raises: every class has a qualified name, and this getter reads
        # it from the type object itself.
        qualname = str.__str__(type.__dict__["__qualname__"].__get__(cls))
    if module is None or module in ("", "builtins"):
        return qualname
    return f"{module}.{qualname}"


def _stored_module(cls: type) -> object:
    """The `__module__` the class statement stored, or None when it stored none."""
    try:
        return type.__dict__["__module__"].__get__(cls)
    except Exception:
        return None


def _exact_str(value: object) -> str | None:
    """`value` as an exact `str` (a copy, for a subclass), or None if it is no string.

    `type()` and a subclass check against `str` read the object's real type, so
    unlike `isinstance` they never ask the object for its `__class__`.
    """
    return str.__str__(value) if issubclass(type(value), str) else None
