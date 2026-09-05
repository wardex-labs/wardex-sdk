"""Typed mirror of the core's resource limits.

This dataclass intentionally holds no values: every field defaults to None,
meaning "use the core default". The core (crates/wardex-limits) owns both the
schema and the values, so a limit can never disagree between two declaration
sites. test_limits.py asserts both properties.

The mirror is a strict SUBSET of the core's table, not a copy of it: the core
keeps `replay_buffer_size` and `zstd_level` for its own encoder defaults, but
neither is a knob anything in-process reads off this config, so neither is
declared here — a field a user can set that changes nothing is the failure the
probe table in test_limits.py exists to prevent.

Below the mirror sits the DELIVERY table: which component takes which bound,
under which keyword. It is here rather than beside its consumers because this
module is a leaf of the import graph and the table has to name components in
three packages; `_LIMIT_DELIVERY` names them as strings, and
`tests/test_limits_wiring.py` resolves those strings, checks them against the
real signatures, and checks that every construction goes through
`limits_kwargs()`. A limit that reaches this dataclass and stops there is not
a limit, and it is not a shape any single component can notice on its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, NamedTuple

from ._native import NATIVE_OK, native, unavailable_reason


def _no_core() -> RuntimeError:
    """The error both accessors raise when the extension could not be imported.

    They cannot answer with a value. The core owns the limit table -- that is
    the whole point of this module's docstring -- so a Python-side fallback
    table would be a second declaration site and would drift, which is exactly
    what `test_limits.py` and `_assembly/_units.py` forbid. Raising something
    that names the wheel is the only honest answer, and it is strictly better
    than what this module did before degraded mode existed, which was to make
    `import wardex_sdk` itself raise.
    """
    return RuntimeError(
        f"wardex native extension unavailable, so resource limits cannot be "
        f"resolved ({unavailable_reason()})"
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class LimitsConfig:
    """Resource limit overrides. None means the core default is used."""

    max_headers: int | None = None
    max_body_bytes: int | None = None
    max_opaque_body_bytes: int | None = None
    max_stream_buffer_bytes: int | None = None
    max_decoded_bytes: int | None = None
    max_streams: int | None = None
    max_ws_frame_bytes: int | None = None
    ws_sample_bytes: int | None = None
    max_connections: int | None = None
    max_sessions: int | None = None
    max_session_entries: int | None = None
    max_units: int | None = None
    max_entries_per_unit: int | None = None
    mcp_sniff_bytes: int | None = None
    max_extra_keys: int | None = None
    max_parse_backlog: int | None = None
    max_parse_backlog_bytes: int | None = None
    max_buffer_spans: int | None = None
    max_buffer_bytes: int | None = None
    max_otel_bridge_body_bytes: int | None = None
    max_otel_bridge_spans_per_session: int | None = None
    max_otlp_attribute_bytes: int | None = None
    max_otlp_request_bytes: int | None = None
    max_link_targets: int | None = None

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None and v < 1:
                raise ValueError(f"{f.name} must be >= 1, got {v}")

    def to_native(self) -> Any:
        """Build the native Limits object, applying only the overrides set here."""
        if not NATIVE_OK:
            raise _no_core()
        kwargs = {f.name: getattr(self, f.name) for f in fields(self)}
        return native.Limits(**{k: v for k, v in kwargs.items() if v is not None})

    def resolved(self) -> dict[str, int]:
        """Effective values (overrides merged onto core defaults) for host-side use."""
        if not NATIVE_OK:
            raise _no_core()
        out = dict(native.limits_defaults())
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None:
                out[f.name] = v
        return out


class LimitsConsumer(Enum):
    """A component that takes a resource bound as a keyword argument.

    A plain `Enum`, deliberately not `class LimitsConsumer(str, Enum)`. The
    string mixin makes `f"{member}"` render the VALUE on 3.10 and the
    `Class.MEMBER` spelling on 3.12+, so a failure message asserted on the
    development interpreter reads differently in CI's floor job — the exact
    shape of drift `scripts/check-py310.sh` exists to catch. Every message
    below spells `.value` instead, which reads the same everywhere.
    """

    UNIT_REGISTRY = "unit_registry"
    SESSION_ASSEMBLER = "session_assembler"
    MCP_TOOL_CATALOG = "mcp_tool_catalog"
    OTEL_BRIDGE_RECEIVER = "otel_bridge_receiver"
    WS_TRACKER = "ws_tracker"
    CONN_TIMING = "conn_timing"
    MCP_PROC_STATE = "mcp_proc_state"
    FINALIZE_QUEUE = "finalize_queue"


class _Delivery(NamedTuple):
    """How one consumer's constructor is split, parameter by parameter.

    `target` is a dotted `"module:Symbol"` or `"module:Symbol.method"` string
    rather than the object itself, and that is a layering decision, not a
    convenience. This module is a LEAF — `_assembly/_units.py` imports it, so
    an import in the other direction is an immediate cycle, and
    `test_import_graph.py` counts an import inside a function body as an edge
    too. A string carries no edge; the guards in `test_limits_wiring.py`
    resolve it with `importlib` at collection time, where a typo is a loud
    failure rather than a silent one.

    The three sets partition the parameters completely, which is the point:
    a consumer that grows a parameter is forced to classify it, so a NEW bound
    cannot arrive as an unclassified keyword nobody passes.
    """

    target: str
    #: constructor keyword -> `LimitsConfig` field name. Not always the same
    #: spelling: four of the seven consumers name their parameter after what
    #: they do with it (`sample_cap`, `cap`, `sniff_limit`, `max_entries`).
    delivers: Mapping[str, str]
    #: parameters that take the `to_native()` object, not a resolved int.
    native: frozenset[str]
    #: every other parameter — sinks, clients, debug flags, modes.
    passthrough: frozenset[str]


#: THE delivery table. One row per consumer; `test_limits_wiring.py` checks
#: every row against the real signature and every call against this table.
#:
#: Two rows spell one keyword two ways on purpose. `UNIT_REGISTRY` takes
#: `max_body_bytes` FROM the field of that name, while `OTEL_BRIDGE_RECEIVER`
#: takes a parameter also called `max_body_bytes` from
#: `max_otel_bridge_body_bytes`. Same spelling, different quantity: one bounds
#: what a logical unit accumulates, the other what the bridge accepts in one
#: request. Merging the two rows because "the kwarg is the same" would silently
#: swap the two numbers, and no signature check would notice. They stay apart.
_LIMIT_DELIVERY: dict[LimitsConsumer, _Delivery] = {
    LimitsConsumer.UNIT_REGISTRY: _Delivery(
        target="wardex_sdk._assembly._units:UnitRegistry",
        delivers={
            "max_units": "max_units",
            "max_entries_per_unit": "max_entries_per_unit",
            "max_link_targets": "max_link_targets",
            "max_body_bytes": "max_body_bytes",
        },
        native=frozenset(),
        passthrough=frozenset({"sink", "debug"}),
    ),
    LimitsConsumer.SESSION_ASSEMBLER: _Delivery(
        target="wardex_sdk._adapters._assembler:SessionAssembler",
        delivers={
            "max_sessions": "max_sessions",
            "max_session_entries": "max_session_entries",
        },
        native=frozenset(),
        passthrough=frozenset({"client", "units", "names", "bridge"}),
    ),
    LimitsConsumer.MCP_TOOL_CATALOG: _Delivery(
        target="wardex_sdk._adapters._anthropic_names:McpToolCatalog.apply_bound",
        delivers={"max_entries": "max_entries_per_unit"},
        native=frozenset(),
        passthrough=frozenset(),
    ),
    LimitsConsumer.OTEL_BRIDGE_RECEIVER: _Delivery(
        target="wardex_sdk._adapters._otel_receiver:_OtelBridgeReceiver",
        delivers={
            "max_body_bytes": "max_otel_bridge_body_bytes",
            "max_spans_per_session": "max_otel_bridge_spans_per_session",
            "max_sessions": "max_sessions",
        },
        native=frozenset(),
        passthrough=frozenset(),
    ),
    LimitsConsumer.WS_TRACKER: _Delivery(
        target="wardex_sdk._interceptors._trackers:_WebSocketTracker",
        delivers={"sample_cap": "ws_sample_bytes"},
        native=frozenset({"limits"}),
        # `llm_upgrade` is the endpoint table's answer about the upgrade
        # path, not a bound: a decision the seam passes through.
        passthrough=frozenset(
            {"path", "deflate", "parent", "parent_closed", "start_ns", "llm_upgrade"}
        ),
    ),
    LimitsConsumer.CONN_TIMING: _Delivery(
        target="wardex_sdk._interceptors._conn_timing:install_shared_timing",
        delivers={"cap": "max_connections"},
        native=frozenset(),
        passthrough=frozenset(),
    ),
    LimitsConsumer.MCP_PROC_STATE: _Delivery(
        target="wardex_sdk._interceptors._mcp_stdio:_ProcState",
        delivers={"sniff_limit": "mcp_sniff_bytes"},
        native=frozenset({"limits"}),
        passthrough=frozenset({"mode", "debug"}),
    ),
    LimitsConsumer.FINALIZE_QUEUE: _Delivery(
        target="wardex_sdk._finalize:FinalizeQueue",
        delivers={
            "max_jobs": "max_parse_backlog",
            "max_bytes": "max_parse_backlog_bytes",
        },
        native=frozenset(),
        passthrough=frozenset({"admit", "debug"}),
    ),
}


def limits_kwargs(consumer: LimitsConsumer, resolved: Mapping[str, int]) -> dict[str, int]:
    """The keyword arguments `consumer` must be built with, out of `resolved`.

    THE one place a resolved limit becomes a constructor argument. A caller
    that spells the keywords by hand can forget one — silently, because the
    consumer has a working default for every bound — and three of them did:
    a registry that never heard about the body cap, a tool catalog that
    resolved the process default for itself, and a timing store that latched
    the first init's number for the life of the process. What each cost was
    invisible from the outside, because the config object kept reporting the
    value the user asked for.

    `resolved` is always the output of `LimitsConfig.resolved()`, which carries
    every core field, so a `KeyError` here means a typo in the table above —
    and `test_limits_wiring.py` raises that at collection time, not in a host.
    """
    return {kw: resolved[field] for kw, field in _LIMIT_DELIVERY[consumer].delivers.items()}
