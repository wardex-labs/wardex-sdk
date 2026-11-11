"""What an OTLP/HTTP receiver's 2xx answer says about the spans it was sent.

A 2xx is not proof the spans were kept. OTLP lets a receiver accept a request
and reject some of its spans (`partial_success.rejected_spans`), and a 200 with
an HTML page is what a proxy, a login page or a web app sends to a wrong path.
`read_answer` tells the cases apart from the answer's headers and body, with
nothing but the standard library, so a caller can count or report them.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from typing import NamedTuple

#: How much of an answer is read. An OTLP answer is a few bytes; anything
#: longer is not one, and its first bytes are enough to say so.
MAX_ANSWER_BYTES = 64 * 1024


class Answer(NamedTuple):
    """`kind` is "otlp" (an OTLP response, decoded; empty counts, since an
    empty message is a valid one), "html" (a web page), or "other" (a body
    that is neither). `rejected` and `message` are the partial success's
    rejected span count and text; 0 and "" for anything but "otlp"."""

    kind: str
    content_type: str
    rejected: int = 0
    message: str = ""


def read_answer(content_type: str, content_encoding: str, body: bytes) -> Answer:
    """Classify one 2xx answer to an OTLP/HTTP export. Never raises."""
    media = content_type.split(";", 1)[0].strip().lower()
    try:
        if content_encoding.strip().lower() == "gzip" and body:
            body = gzip.decompress(body)[:MAX_ANSWER_BYTES]
        if not body:
            return Answer("otlp", media)
        if media in ("text/html", "application/xhtml+xml") or body.lstrip()[:1] == b"<":
            return Answer("html", media)
        if media == "application/json":
            return _from_json(media, json.loads(body))
        if media in ("application/x-protobuf", "application/protobuf"):
            return _from_protobuf(media, body)
    except (OSError, EOFError, ValueError, TypeError, AttributeError, IndexError):
        pass
    return Answer("other", media)


def _from_json(media: str, doc: object) -> Answer:
    partial = doc.get("partialSuccess") if isinstance(doc, dict) else None
    if not isinstance(partial, dict):
        return Answer("otlp", media)
    return Answer(
        "otlp",
        media,
        int(partial.get("rejectedSpans") or 0),
        str(partial.get("errorMessage") or ""),
    )


def _from_protobuf(media: str, body: bytes) -> Answer:
    """`ExportTraceServiceResponse { ExportTracePartialSuccess partial_success = 1; }`,
    `ExportTracePartialSuccess { int64 rejected_spans = 1; string error_message = 2; }`."""
    rejected, message = 0, ""
    for number, value in _fields(body):
        if number == 1 and isinstance(value, bytes):
            for inner, field in _fields(value):
                if inner == 1 and isinstance(field, int):
                    rejected = field
                elif inner == 2 and isinstance(field, bytes):
                    message = field.decode("utf-8", "replace")
    return Answer("otlp", media, rejected, message)


def _fields(buf: bytes) -> Iterator[tuple[int, int | bytes]]:
    """Each field of one protobuf message: varints as ints, length-delimited
    fields as bytes. A malformed message raises ValueError."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
            yield number, value
        elif wire == 2:
            size, i = _varint(buf, i)
            if i + size > len(buf):
                raise ValueError("truncated field")
            yield number, buf[i : i + size]
            i += size
        elif wire in (1, 5):
            i += 8 if wire == 1 else 4
        else:
            raise ValueError("unsupported wire type")


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        if i >= len(buf) or shift > 63:
            raise ValueError("truncated varint")
        byte = buf[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, i
        shift += 7
