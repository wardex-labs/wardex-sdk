from __future__ import annotations

import abc

from .._types import InternalEnvelope

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
The tie is between the DEFAULTS, which is why `Transport.timeout` names this
constant as a starting point a subclass is expected to override.
"""


class _Undelivered:
    """The type of `UNDELIVERED`. A singleton, so `is` is the whole test."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNDELIVERED"


UNDELIVERED = _Undelivered()
"""What `Transport.export` returns to say "I did not send this one".

The client cannot see inside a transport, and every attempt to work out from the
outside whether a send would happen has been wrong: the check runs at a moment
that is not the moment of the send, and the work in between (a `before_send`
that outlives the budget, most of all) invalidates it. So the transport says so
itself, at the one moment it knows.

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
    # PII policy applied by the native encoders on wire paths (design §4.2).
    # Class-level defaults are secure-by-default: a transport used without
    # init() still masks.
    _pii_mode: str = "mask"
    _pii_disabled: tuple[str, ...] = ()

    timeout: float = DEFAULT_TIMEOUT
    """How long this transport may spend on ONE export, in seconds.

    Declared here because the client READS it: a `wardex.flush()` with no
    argument means "send what you have, I will wait", so it takes its budget
    from this attribute rather than capping the export at a default of its own.
    An `OtlpHttpTransport(timeout=30.0)` therefore gets its thirty seconds out
    of a bare `flush()`.

    It was undeclared for a release, and read off the instance with a
    `try/except`. Nothing crashed -- the read is still guarded, see
    `_client._configured_transport_timeout` -- but with no declaration there was
    no contract, and it failed quietly in both directions. A third-party
    transport that happened to keep a `self.timeout` for its own bookkeeping
    silently redefined how long a bare `flush()` waited; one that did not have
    the attribute silently got 5 seconds instead of the thirty it was built for,
    and nothing anywhere said why. Neither is a crash, which is exactly what
    made them the kind of defect this SDK keeps finding late.

    Overriding it is expected, and a plain instance attribute
    (`self.timeout = ...` in `__init__`) or a property both work. The default is
    `DEFAULT_TIMEOUT` so that a transport with nothing to say behaves as it did
    before this attribute existed.

    Non-binding on I/O: this is what the transport ADVERTISES, and the client
    uses it to size its own wait. Actually bounding the socket is the
    transport's job, in `export`, using the `timeout` it is passed there.
    """

    def set_pii_policy(self, mode: str, disabled: tuple[str, ...]) -> None:
        """Install the PII policy resolved from WardexConfig (called by init)."""
        self._pii_mode = mode
        self._pii_disabled = disabled

    @abc.abstractmethod
    def export(self, envelope: InternalEnvelope, *, timeout: float | None = None) -> object | None:
        """Ship one envelope. `timeout` is the remaining budget for this export,
        in seconds, or None when no deadline was imposed at all.

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
