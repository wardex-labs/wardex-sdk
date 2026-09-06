from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from .._assembly import counters, report_once
from .._native import NATIVE_OK, native, unavailable_reason
from .._types import Envelope

if TYPE_CHECKING:
    from .._config import PIIConfig

DEFAULT_TIMEOUT = 5.0
"""The one 5.0 in this SDK's timeout story, and the single place to change it.

Every default budget on the export path is this number, and they were four
separate literals across two modules until one of them needed to move and only
some of them did. `_client._DEFAULT_TIMEOUT` is this constant, `Transport.flush`
and `Transport.close` default to it, and `Client`'s bound for the periodic
`transport.flush()` is it as well -- that last one because the number it wants
IS `Transport.flush`'s own default, not a coincidence that happens to match.

Deliberately NOT the default for a transport's own configured export timeout:
`OtlpHttpTransport(timeout=10.0)` chose ten seconds for its own reasons, and
sharing a constant between "how long one POST may take" and "how long a
shutdown step may take" would tie two numbers that answer different questions.
The tie is between the DEFAULTS, which is why `Transport.export_timeout` names
this constant as a starting point a subclass is expected to override.
"""


class Undelivered:
    """The type of `UNDELIVERED`. A singleton, so `is` is the whole test.

    Public so that `Transport.export`'s return annotation can be spelled by a
    third-party implementer -- the VALUE to return is still the one
    `UNDELIVERED` instance, never a fresh `Undelivered()`.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNDELIVERED"


UNDELIVERED = Undelivered()
"""What `Transport.export` returns to say "I did not send this one".

The client cannot see inside a transport, and every attempt to work out from the
outside whether a send would happen has been wrong: the check runs at a moment
that is not the moment of the send, and the work in between (a
`before_send_envelope` that outlives the budget, most of all) invalidates it. So
the transport says so itself, at the one moment it knows.

Precise meaning, because the client acts on it: *this envelope was not put on
the wire, nothing about it was consumed, and an identical attempt later with a
fresh budget could succeed.* That is what makes it safe for a non-final
`flush()` to keep the spans and re-send them -- a decline that had already
half-sent something would be a duplicate, and one that can never succeed would
be an infinite retry. A transport that fails mid-POST, or that cannot encode at
all, must NOT return this; it has its own reporting and its own reasons.

Anything else -- above all `None`, which is what every transport written before
this existed returns -- means the transport took the envelope. That default is
the safe direction: delivering unconditionally is a legal transport, and a
client guessing "not delivered" from the outside would announce losses that
never happened and burn the one-line-per-process report budget doing it.
"""


class CallerBudget(float):
    """A `timeout` a CALLER named, as opposed to one wardex derived for them.

    PYTHON-ONLY, and deliberately EXCLUDED from the cross-language transport
    SPI: this is a diagnostic refinement, not part of the contract Node or
    Java inherit. A float subclass carrying a field cannot exist in every
    language -- there is nothing to subclass in Node and nowhere for the field
    to ride in Java -- so other SDKs receive a plain timeout and may not be
    able to distinguish caller-named budgets from derived ones. A transport
    ported across languages must therefore never REQUIRE this distinction;
    here it sharpens one stderr diagnostic and nothing else.

    A transport receiving `timeout=2.0` cannot tell those two apart, and the
    difference is the whole of whether a cut-off POST is news:

      * `wardex.flush(2.0)` -- the caller named two seconds and did not wait
        long enough to learn the outcome. Their number, their fix, worth a line.
      * a bare `wardex.flush()` -- there is no caller number at all. wardex
        derives one from the transport's own configured timeout, and by the time
        the export runs it has already shrunk by the acquire and the encode, so
        the transport sees something like 9.97 under a 10s configuration. That
        is wardex telling the transport about its OWN timeout, and blaming the
        caller for it is a false report.

    That second case is a report the caller cannot act on, and because the
    channel is one line per key per process, a false line BURNS the key: the
    genuine cut-short report is silenced for the rest of the process. So the
    fact travels with the number, and it travels in the direction that fails
    safe.

    THE DEFAULT IS SILENCE, which is what stops this drifting back. A plain
    `float` -- what arithmetic produces, what a third-party client passes, what
    any future code path that forgets about this class will hand over -- means
    "wardex's own number, say nothing". Only an explicit `CallerBudget` speaks.
    Re-introducing the bug therefore takes an affirmative wrap on the derived
    path, not an omission on the caller's; and an omission on the caller's path
    costs a diagnostic, never a false accusation.

    `float` subclass rather than a second parameter because `Transport.export`
    is PUBLIC: third-party transports written against `(envelope, *, timeout)`
    must keep working, and they do -- to a transport that ignores this class,
    a `CallerBudget` is simply the float it already expected.

    `requested` is what the caller actually passed, kept alongside the (smaller)
    remaining budget so a report can name the number the caller would recognize:
    someone who wrote `flush(2.0)` should read "2.0s", not "1.97s".
    """

    __slots__ = ("requested",)

    requested: float

    def __new__(cls, remaining: float, requested: float) -> CallerBudget:
        budget = super().__new__(cls, remaining)
        budget.requested = float(requested)
        return budget

    def __reduce__(self) -> tuple[type[CallerBudget], tuple[float, float]]:
        """Survive `copy`, `deepcopy` and `pickle` as a `CallerBudget`.

        `float.__reduce_ex__` reconstructs through `cls(value)` -- one argument,
        because that is all a float needs -- and this subclass requires two, so
        every one of the three raised `TypeError: __new__() missing 1 required
        positional argument`. That matters here and not on an ordinary value
        type: this object is handed to a THIRD-PARTY `Transport.export`, and a
        transport that stores its arguments for a retry queue, hands them to a
        `ProcessPoolExecutor`, or merely deepcopies its inputs for a log would
        have raised out of wardex's own export path.

        Rebuilding by type rather than by identity is the right shape for this
        class: `CallerBudget` has no singletons, and both facts it carries --
        the remaining budget and the number the caller actually named -- are in
        the tuple, so a copy answers `isinstance` and reports the same number.
        """
        return (type(self), (float(self), self.requested))

    def __repr__(self) -> str:
        return f"CallerBudget({float(self)!r}, requested={self.requested!r})"


class Transport(abc.ABC):
    """The one advertised extension point: where finished envelopes go.

    THE PII CONTRACT, stated here because a subclass is host code the SDK
    cannot audit: export() receives PRE-masking data; if you serialize the
    envelope yourself, you own PII masking -- encode() is the sanctioned,
    masked path to bytes. The `Envelope` is opaque (its guaranteed surface is
    `span_count` and `encode()`), so a transport that stays on the sanctioned
    path gets the configured masking and limits without ever reading a field.

    THE ASYNC CONTRACT, and it is the cross-language one: `export`, `flush`
    and `close` are invoked from wardex's OWN worker thread -- never from the
    host's event loop or request threads -- and each may BLOCK up to its
    budget; blocking I/O here stalls no host code. That behavioral contract is
    what every wardex SDK keeps, each in its platform's idiom: Node binds
    `export` as async (`Promise`) and its pipeline awaits it, Java stays
    blocking on its exporter thread. `before_send_envelope` is SYNCHRONOUS in
    Python. (`CallerBudget` below is a Python-only diagnostic refinement and
    is excluded from this cross-language SPI -- see its docstring.)

    THE FORK EXTENSION POINT, optional: a transport may define
    `at_fork_child()` (no arguments), and wardex's `os.register_at_fork`
    child hook will call it -- under the SDK's guard, so a raise costs a
    counter and nothing else. It exists because wardex resets its OWN state
    at fork but cannot reset a transport's internals: an implementation
    holding a pooled `requests.Session`/`httpx.Client` shares live TCP
    sockets with the parent after a fork -- that library's classic fork
    hazard -- and only the transport knows how to rebuild its pool. wardex's
    built-in transports hold no per-request state and do not implement it.
    """

    # PII policy applied by the native encoders on wire paths (design §4.2).
    # Class-level defaults are secure-by-default: a transport used without
    # init() still masks.
    _pii_mode: str = "mask"
    _pii_disabled: tuple[str, ...] = ()

    export_timeout: float = DEFAULT_TIMEOUT
    """How long this transport may spend on ONE export, in seconds.

    Declared here because the client READS it: a `wardex.flush()` with no
    argument means "send what you have, I will wait", so it takes its budget
    from this attribute rather than capping the export at a default of its own.
    An `OtlpHttpTransport(timeout=30.0)` therefore gets its thirty seconds out
    of a bare `flush()`.

    NAMED `export_timeout` AND NOT `timeout`, because the collision already
    happened once. This attribute is read reflectively off the instance, which
    makes its name contract in the least visible way there is -- and under the
    bare spelling a third-party transport that happened to keep a
    `self.timeout` for its own bookkeeping silently redefined how long a bare
    `flush()` waited, while one that did not have the attribute silently got 5
    seconds instead of the thirty it was built for. Neither is a crash.
    `timeout` is a name every second transport keeps for itself;
    `export_timeout` is one nobody picks by accident. (It also spent a release
    undeclared entirely and read with a `try/except` -- the read is still
    guarded, see `_client._configured_transport_timeout`, because a
    declaration is not a guarantee.)

    Overriding it is expected, and a plain instance attribute
    (`self.export_timeout = ...` in `__init__`) or a property both work. The
    default is `DEFAULT_TIMEOUT` so that a transport with nothing to say
    behaves as it did before this attribute existed.

    Non-binding on I/O: this is what the transport ADVERTISES, and the client
    uses it to size its own wait. Actually bounding the socket is the
    transport's job, in `export`, using the `timeout` it is passed there.
    """

    _limits: object | None = None
    """The resolved native `Limits`, or None for the core's own defaults.

    Read by `encode()`: the OTLP encoder caps attribute values at
    `max_otlp_attribute_bytes` and splits its request bodies at
    `max_otlp_request_bytes`, and both of those are user-configurable.
    Class-level None keeps a transport constructed by hand — without `init()` —
    working on the core defaults rather than raising for a knob it was never
    given.
    """

    def encode(self, envelope: Envelope, *, compress: bool = True) -> tuple[bytes, ...]:
        """The sanctioned path from an envelope to wire bytes. DO NOT OVERRIDE.

        Returns one OTLP/HTTP request body per element — protobuf, gzipped
        when `compress` is true — split at `max_otlp_request_bytes`. An OTLP
        request is accepted or rejected whole, so a batch over the receiver's
        body limit leaves as several bodies rather than arriving short: POST
        every element, in order.

        PII MASKING AND LIMITS ARE APPLIED HERE, because this method hands the
        transport's stored policy — installed by `init()`, secure by default
        without it — to the native encoder. That is why the method is final: a
        transport that reaches wire bytes through `encode()` cannot ship
        unmasked data by accident, and one that serializes the envelope itself
        owns that risk (see the class docstring).

        A span too large to fit a request even with its payload removed is
        dropped and reported — one stderr line per process, on the first
        occurrence, its count covering that batch — because it never reaches
        the wire to carry a marker. The report lives here, concrete and
        shared, so every transport that encodes gets it.
        """
        if not NATIVE_OK:
            # Reachable from a hand-constructed transport that never went
            # through init(), so name the missing wheel instead of raising
            # AttributeError off a None module.
            raise RuntimeError(
                f"wardex native extension unavailable, so envelopes cannot be "
                f"encoded ({unavailable_reason()})"
            )
        # Three answers, not one: the bodies, the count of spans too large
        # for a request, and the reasons for spans the marshaller could not
        # read. The encoder does NOT raise for a bad typed-block value any
        # more -- it skips that one span and names it here, where it is
        # counted and reported. A raise here is still a raise: a corrupt
        # envelope or an absent native module is not a per-span loss.
        bodies, dropped, unmarshalled = native.codec.encode_otlp_requests(
            envelope,
            self._pii_mode,
            list(self._pii_disabled),
            self._limits,
            compress,
        )
        if unmarshalled:
            # A span the marshaller could not read -- a typed block holding a
            # value of the wrong Python type. It used to raise out of the
            # encoder, and the client's drain then dropped the WHOLE batch:
            # every good span around it, silently off-debug. Now the one span
            # is skipped, counted, and named once per process; the rest of
            # the batch ships. The first reason is quoted because it is the
            # encoder's own words about a wardex type, never host content.
            for _ in unmarshalled:
                counters.bump("transport.otlp.span_unmarshalled")
            report_once(
                f"{len(unmarshalled)} span(s) could not be marshalled for export and "
                f"were dropped; the rest of the batch shipped. First: {unmarshalled[0]}",
                key="transport.otlp.span_unmarshalled",
            )
        if dropped:
            # A span so large it would not fit a request even with its payload
            # removed. `report_once` rather than a debug print: the marker
            # mechanism cannot reach this loss -- the span is not on the wire
            # to carry one -- so off-debug it would be byte-identical to those
            # spans never having been captured. Bounded to one line per
            # process, which is what makes it affordable on a per-call path,
            # and keyed apart from the budget reports so "one span is too big
            # for your collector" stays separately actionable.
            report_once(
                f"{dropped} span(s) exceeded max_otlp_request_bytes even with "
                f"their payload removed and were not exported. Raise "
                f"max_otlp_request_bytes if your collector accepts more.",
                key="transport.otlp.span_over_request_cap",
            )
        return tuple(bodies)

    def _set_pii_policy(self, policy: PIIConfig) -> None:
        """Install the PII policy resolved from `WardexConfig`. Called by
        `init()` — plumbing, not a subclass hook.

        Stores the shape the native encoder takes (`encode()` is the reader),
        so a transport constructed by hand keeps the secure class-level
        defaults and one installed by `init()` carries exactly what was
        configured.
        """
        self._pii_mode = policy.mode.value
        self._pii_disabled = tuple(sorted(c.value for c in policy.disabled_categories))

    def _set_limits(self, limits: object | None) -> None:
        """Install the resolved resource limits. Called by `init()` — plumbing,
        not a subclass hook.

        Separate from `_set_pii_policy` because the two answer to different
        config. The object stored is the same native `Limits` the encoders
        take, and `encode()` is its reader.
        """
        self._limits = limits

    @abc.abstractmethod
    def export(self, envelope: Envelope, *, timeout: float | None = None) -> Undelivered | None:
        """Ship one envelope. `timeout` is the remaining budget for this export,
        in seconds, or None when no deadline was imposed at all.

        `envelope` arrives PRE-masking (see the class docstring): masking runs
        inside `encode()`, which is the sanctioned way to turn the envelope
        into bytes.

        Three values, not two: `None` means "no deadline"; a plain `float` means
        "a budget wardex derived -- from your own configured timeout, or from
        its shutdown default"; a `CallerBudget` means "a number the application
        named". Only the last one is the caller's to answer for, and a transport
        that diagnoses a cut-off POST must test for it rather than infer it from
        the number being small. See `CallerBudget`.

        Keyword-only and defaulted on purpose: `Transport` is public, and
        subclasses written before this parameter existed still declare
        `export(self, envelope)`. The client probes the signature and withholds
        the keyword from those, so an old subclass keeps working -- but it also
        keeps stalling, because only the transport can bound its own I/O. A
        transport that performs blocking I/O should honour this by *narrowing*
        its configured timeout, never widening it: a caller asking for 99s must
        not get more than the transport was configured for.

        Return `UNDELIVERED` when the envelope was NOT put on the wire and an
        identical attempt later could still succeed -- a spent `timeout` is the
        case wardex's own transport uses it for. Returning anything else, `None`
        included, means "taken": the client then neither retries nor reports it.
        A transport that declines silently is indistinguishable from one that
        delivered, and is treated as the latter; see `UNDELIVERED` for why that
        is the safe default and not a shrug.
        """

    def flush(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        return None

    def close(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        return None
