"""Every transport field, and every scalar a span carries, from schema to OTLP.

A value the SDK measures passes through five hand-written places before a
backend sees it: the schema (`proto/wardex/v1/span.proto`), the dataclass a
producer fills (`_types.py`), the producers themselves (the interceptors), the
envelope encoder and decoder (the binding), and the OTLP mapping
(`crates/wardex-codec/src/otlp/map.rs`). Each is written by hand, so each can
miss a field on its own, and for a long time only one of them failed when it
did. That is how twelve of the sixteen transport values the interceptors
measure — TCP/TLS timing, sizes, connection id and reuse, direction, the MCP
method and id — and `workflow_name` never reached an OTLP backend, while the
envelope shipped fields no producer ever filled under their zero values, as if
they had been observed.

This file holds every stage to the schema, field by field:

* SCHEMA <-> DATACLASS: the fields of each transport message are the fields of
  its Python dataclass, except the ones `_PROTO_ONLY` names.
* SCHEMA <-> OTLP: the mapping's own census (`otlp::map::WIRE_FIELDS`, read
  through the binding so there is one table, not two) covers every field of
  `Span` and every leaf under `Span.transport`.
* ENVELOPE: a sentinel in every leaf survives encode and decode.
* OTLP: a sentinel in every leaf the mapping exports arrives under its key, and
  one in every leaf it declines arrives nowhere.
* PRODUCERS: read off the source. Everything a producer fills is exported; a
  leaf the mapping declines for having no producer really has none; and a
  field whose dataclass default is a value rather than `None` is passed
  explicitly by every producer, so no default can stand in for a reading.

WHAT THIS DOES NOT REACH: the leaves of the other blocks a span carries
(`capture_integrity`, `correlation`, `call_site`, `conversation`, events,
links) are census rows here only as blocks, and each block's projection is
asserted leaf by leaf by its own tests. The envelope header and state
snapshots are not spans and are not here.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path
from typing import Any

import pytest

import wardex_sdk
from test_codec import _env, _span
from wardex_sdk import _types, _wardex_native
from wardex_sdk._enums import Direction, Modality, Protocol
from wardex_sdk._types import (
    A2aMeta,
    GrpcMeta,
    HttpMeta,
    McpMeta,
    SseMeta,
    TransportAttributes,
    TransportTiming,
    WebSocketMeta,
)
from wardex_sdk.transport import _codec

_SCHEMA = Path(__file__).resolve().parents[3] / "proto" / "wardex" / "v1" / "span.proto"

#: The transport messages, each with the dataclass a producer fills.
_TRANSPORT = {
    "TransportAttributes": TransportAttributes,
    "TransportTiming": TransportTiming,
    "HttpMeta": HttpMeta,
    "GrpcMeta": GrpcMeta,
    "WebSocketMeta": WebSocketMeta,
    "McpMeta": McpMeta,
    "SseMeta": SseMeta,
    "A2aMeta": A2aMeta,
}

#: Schema fields with no dataclass field, so nothing in Python can fill them.
#: Each needs its no-producer exclusion in `WIRE_FIELDS` (asserted below).
_PROTO_ONLY = {"GrpcMeta.decoded_payload"}

#: Span scalars whose OTLP home is one attribute; the dataclass attribute has
#: the schema's name.
_SPAN_SCALARS = ("error_type", "server_address", "server_port", "workflow_name")


def _schema() -> dict[str, list[str]]:
    """Message -> its field names, read off the schema file."""
    out: dict[str, list[str]] = {}
    current: str | None = None
    for raw in _SCHEMA.read_text().splitlines():
        line = raw.split("//")[0].strip()
        if m := re.match(r"message (\w+) \{", line):
            current = m.group(1)
            out[current] = []
        elif line == "}":
            current = None
        elif current and "=" in line and not line.startswith("reserved"):
            decl = re.sub(r"^(optional|repeated) ", "", line)
            out[current].append(decl.split()[1])
    return out


def _wire_fields() -> dict[str, tuple[str, str]]:
    return {path: (kind, text) for path, kind, text in _wardex_native.codec.otlp_wire_fields()}


def _leaves() -> list[str]:
    """`Msg.field` for every transport leaf: `TransportAttributes` fields that
    are not themselves one of the transport messages, plus every field of
    those messages."""
    schema = _schema()
    containers = {f for f in schema["TransportAttributes"] if f in _CONTAINERS}
    out = [f"TransportAttributes.{f}" for f in schema["TransportAttributes"] if f not in containers]
    for msg in _TRANSPORT:
        if msg != "TransportAttributes":
            out += [f"{msg}.{f}" for f in schema[msg]]
    return out


#: `TransportAttributes` field -> the message it holds.
_CONTAINERS = {
    "timing": "TransportTiming",
    "http": "HttpMeta",
    "grpc": "GrpcMeta",
    "websocket": "WebSocketMeta",
    "mcp": "McpMeta",
    "sse": "SseMeta",
    "a2a": "A2aMeta",
}

# --- sentinels: one per leaf, each distinct from every other value on the span

_SENTINELS: dict[str, Any] = {
    "TransportAttributes.connection_id": "sentinel-conn",
    "TransportAttributes.protocol": Protocol.GRPC,
    "TransportAttributes.direction": Direction.INBOUND,
    "TransportAttributes.request_size": 4321,
    "TransportAttributes.response_size": 8765,
    "TransportAttributes.request_blob_ref": "sentinel-req-blob",
    "TransportAttributes.response_blob_ref": "sentinel-resp-blob",
    "TransportAttributes.request_modality": Modality.IMAGE,
    "TransportAttributes.response_modality": Modality.AUDIO,
    "TransportAttributes.is_streaming": True,
    "TransportAttributes.connection_reused": True,
    "TransportTiming.tcp_connect_ms": 11.25,
    "TransportTiming.tls_handshake_ms": 22.5,
    "TransportTiming.ttfb_ms": 33.75,
    "TransportTiming.transfer_ms": 44.0,
    "TransportTiming.ttft_ms": 55.5,
    "HttpMeta.method": "PATCH",
    "HttpMeta.url": "http://sentinel-host/x?q=1",
    "HttpMeta.status_code": 418,
    "GrpcMeta.service": "sentinel.Svc",
    "GrpcMeta.method": "SentinelMethod",
    "GrpcMeta.stream_id": 77,
    "GrpcMeta.status_code": 13,
    "GrpcMeta.encoding": "sentinel-enc",
    "WebSocketMeta.opcode": 9,
    "WebSocketMeta.direction": "sentinel-ws-dir",
    "McpMeta.rpc_method": "sentinel/method",
    "McpMeta.rpc_id": "sentinel-rpc-id",
    "SseMeta.event_type": "sentinel-sse-event",
    "A2aMeta.task_id": "sentinel-task",
    "A2aMeta.transport": "sentinel-a2a",
    "Span.error_type": "SentinelError",
    "Span.server_address": "sentinel.server",
    "Span.server_port": 4433,
    "Span.workflow_name": "sentinel-workflow",
}


def _sentinel_span() -> Any:
    def block(msg: str) -> Any:
        cls = _TRANSPORT[msg]
        return cls(**{f.name: _SENTINELS[f"{msg}.{f.name}"] for f in dataclasses.fields(cls)})

    scalars = {
        f.name: _SENTINELS[f"TransportAttributes.{f.name}"]
        for f in dataclasses.fields(TransportAttributes)
        if f.name not in _CONTAINERS
    }
    transport = TransportAttributes(
        **scalars, **{name: block(msg) for name, msg in _CONTAINERS.items()}
    )
    return _span(
        transport=transport, **{name: _SENTINELS[f"Span.{name}"] for name in _SPAN_SCALARS}
    )


def _otlp_value(sentinel: Any) -> Any:
    """What the sentinel reads as among the decoded OTLP attributes."""
    if isinstance(sentinel, (Protocol, Direction, Modality)):
        return sentinel.value
    return sentinel


# --- SCHEMA <-> DATACLASS, SCHEMA <-> OTLP CENSUS ---


@pytest.mark.parametrize("msg", sorted(_TRANSPORT))
def test_each_transport_message_and_its_dataclass_name_the_same_fields(msg: str):
    schema = {f"{msg}.{f}" for f in _schema()[msg]}
    python = {f"{msg}.{f.name}" for f in dataclasses.fields(_TRANSPORT[msg])}
    assert python <= schema, f"dataclass fields the schema does not have: {python - schema}"
    assert schema - python == {p for p in _PROTO_ONLY if p.startswith(f"{msg}.")}


def test_the_otlp_census_covers_every_span_field_and_every_transport_leaf():
    schema = _schema()
    expected = {f"Span.{f}" for f in schema["Span"]}
    expected |= {f"{msg}.{f}" for msg in _TRANSPORT for f in schema[msg]}
    census = _wire_fields()
    assert set(census) == expected
    for path, (kind, text) in census.items():
        assert kind in {"attribute", "projected", "not_exported"}, path
        assert text.strip(), path


def test_every_leaf_has_a_sentinel():
    """A leaf added to the schema without a sentinel here fails on this set
    equality, which is the moment somebody has to say where it goes."""
    leaves = set(_leaves()) - _PROTO_ONLY
    assert leaves | {f"Span.{n}" for n in _SPAN_SCALARS} == set(_SENTINELS)
    # Distinct, so a value found in OTLP names the one leaf it came from. The
    # two booleans cannot be, and are only ever looked up by their key.
    values = [repr(v) for v in _SENTINELS.values() if not isinstance(v, bool)]
    assert len(set(values)) == len(values), "sentinels collide"


# --- ENVELOPE ---


def _decoded_leaf(transport: dict, path: str) -> Any:
    msg, field = path.split(".")
    if msg == "TransportAttributes":
        return transport[field]
    container = next(name for name, m in _CONTAINERS.items() if m == msg)
    return transport[container][field]


def _decoded_form(path: str, sentinel: Any) -> Any:
    """How the envelope decoder spells a sentinel: protocol and direction as
    their proto numbers, a modality by name, everything else as written."""
    if path == "TransportAttributes.protocol":
        return 2  # PROTOCOL_GRPC
    if path == "TransportAttributes.direction":
        return 2  # DIRECTION_INBOUND
    if isinstance(sentinel, Modality):
        return sentinel.value
    return sentinel


@pytest.mark.parametrize("path", sorted(_SENTINELS))
def test_every_leaf_survives_the_envelope(path: str):
    span = _codec.decode(_codec.encode(_env(_sentinel_span())))["items"][0]["span"]
    sentinel = _SENTINELS[path]
    if path.startswith("Span."):
        assert span[path.split(".")[1]] == sentinel
        return
    assert _decoded_leaf(span["transport"], path) == _decoded_form(path, sentinel)


# --- OTLP ---


def _otlp_attributes(span: Any) -> dict[str, Any]:
    data = _wardex_native.codec.encode_otlp_traces(_env(span))
    decoded = _wardex_native.codec.decode_otlp_traces(data)
    return decoded["resource_spans"][0]["scope_spans"][0]["spans"][0]["attributes"]


@pytest.mark.parametrize("path", sorted(_SENTINELS))
def test_every_leaf_reaches_its_otlp_home_or_is_declined_by_name(path: str):
    attrs = _otlp_attributes(_sentinel_span())
    kind, text = _wire_fields()[path]
    want = _otlp_value(_SENTINELS[path])
    if kind == "attribute":
        assert attrs.get(text) == want, f"{path} did not arrive under `{text}`: {attrs!r}"
    elif kind == "not_exported":
        leaked = [k for k, v in attrs.items() if v == want]
        assert not leaked, f"{path} is declined but its value rides under {leaked}"
    else:
        assert path == "TransportAttributes.protocol", f"{path}: projected, check it here"
        assert attrs["network.protocol.name"] == "grpc"


def test_every_declined_leaf_is_declined_for_having_no_producer():
    census = _wire_fields()
    for path in _leaves():
        kind, text = census[path]
        if kind == "not_exported":
            assert text.startswith("no producer"), f"{path}: {text}"


# --- PRODUCERS ---


def _producer_calls() -> list[tuple[str, str, set[str]]]:
    """`(where, class, keyword names)` for every construction of a transport
    dataclass in the SDK's source. A positional argument or `**kwargs` is a
    construction this census cannot read, and fails it."""
    calls = []
    root = Path(wardex_sdk.__file__).parent
    for path in sorted(root.rglob("*.py")):
        if path.name == "_types.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name not in _TRANSPORT:
                continue
            where = f"{path.relative_to(root)}:{node.lineno}"
            assert not node.args, f"{where}: positional arguments to {name}"
            kws = {k.arg for k in node.keywords}
            assert None not in kws, f"{where}: **kwargs to {name}"
            calls.append((where, name, kws))  # type: ignore[arg-type]
    return calls


def _produced() -> set[str]:
    return {
        f"{cls}.{kw}"
        for _, cls, kws in _producer_calls()
        for kw in kws
        if not (cls == "TransportAttributes" and kw in _CONTAINERS)
    }


def test_the_census_sees_the_producers():
    """Guards the guard: an AST walk that silently matched nothing would make
    every producer assertion below pass vacuously."""
    classes = {cls for _, cls, _ in _producer_calls()}
    assert {"TransportAttributes", "TransportTiming", "HttpMeta", "McpMeta"} <= classes
    assert "TransportTiming.tcp_connect_ms" in _produced()


def test_every_value_a_producer_fills_reaches_otlp():
    census = _wire_fields()
    for path in sorted(_produced()):
        kind, text = census[path]
        assert kind != "not_exported", f"{path} is filled by a producer and declined: {text}"


def test_a_leaf_declined_for_having_no_producer_has_none():
    produced = _produced()
    for path, (kind, text) in _wire_fields().items():
        if kind == "not_exported" and text.startswith("no producer") and path in _leaves():
            assert path not in produced, (
                f"{path} now has a producer; decide its OTLP name in `WIRE_FIELDS`"
            )


def test_no_default_stands_in_for_a_reading():
    """A field whose default is a value (`0`, `""`, an enum member) and not
    `None` would ship that value for any producer that forgot it, as if it had
    been observed. Such a field must be passed explicitly by every producer.
    `None` defaults ship as unset and need no such rule."""
    for cls_name, cls in _TRANSPORT.items():
        valued = {
            f.name
            for f in dataclasses.fields(cls)
            if f.default is not dataclasses.MISSING
            and f.default is not None
            and f.name not in _CONTAINERS
        }
        for where, name, kws in _producer_calls():
            if name == cls_name:
                assert valued <= kws, f"{where}: {cls_name} relies on the default of {valued - kws}"


def test_no_default_in_a_transport_dataclass_claims_an_observation():
    """The presence-carrying fields default to `None`, and a timing block left
    at its default says nothing."""
    timing = TransportTiming()
    assert all(getattr(timing, f.name) is None for f in dataclasses.fields(TransportTiming))
    plain = TransportAttributes(
        connection_id="",
        protocol=Protocol.HTTP,
        direction=Direction.OUTBOUND,
    )
    for name in (
        "request_size",
        "response_size",
        "is_streaming",
        "connection_reused",
        "request_modality",
        "response_modality",
    ):
        assert getattr(plain, name) is None, name
    assert not hasattr(_types.TransportAttributes, "chunk_index")
    assert not hasattr(_types.TransportAttributes, "is_final_chunk")
