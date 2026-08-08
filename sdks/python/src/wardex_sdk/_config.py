"""Configuration object — spec §2.3.

WardexConfig is the immutable configuration for SDK initialization. All fields have
sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ._enums import (
    AdapterName,
    CaptureMode,
    CaptureTrigger,
    InterceptorName,
    PIICategory,
    PIIMode,
    RetentionClass,
)
from ._limits import CaptureLimits
from ._types import BeforeSendCallback

# Fields moved into limits=CaptureLimits(...) in 0.2.0b1. Guarded in __new__
# below so callers get a message naming the new home instead of a bare
# unknown-keyword TypeError from the generated dataclass __init__.
_MOVED_TO_LIMITS = ("max_buffer_spans", "replay_buffer_size")


@dataclass(frozen=True, slots=True)
class WardexConfig:
    """Wardex SDK configuration. An immutable dataclass."""

    api_key: str | None = None
    endpoint: str | None = None

    default_retention: RetentionClass = RetentionClass.SUMMARY_ONLY
    retention_triggers: frozenset[CaptureTrigger] = frozenset(
        {CaptureTrigger.ERROR, CaptureTrigger.MANUAL_MARK}
    )

    pii_mode: PIIMode = PIIMode.MASK
    pii_disabled_categories: frozenset[PIICategory] = frozenset()

    flush_interval: float = 5.0
    flush_on_signals: bool = True
    limits: CaptureLimits = field(default_factory=CaptureLimits)

    adapters: tuple[AdapterName, ...] | None = None

    interceptors: tuple[InterceptorName, ...] | None = None
    """Which byte seams `intercept=True` installs. `None` means all of them.

    `intercept` is the SWITCH and this is the refinement: with `intercept=False`
    a selection installs nothing and says so under `debug`, because a refinement
    of a switch that is off is not an error but silence about it is how a user
    concludes it was honoured. `()` is a choice — install none — and `None` is
    the absence of one; they may not collapse into each other, the same
    distinction `adapters` already draws.

    ORDER IS NOT A CALLER'S TO SET. `interceptors._INTERCEPTORS` is walked and
    filtered by this tuple rather than the other way round, so a reordered
    selection installs in the same order as an unordered one: SSL patches
    `ssl.SSLSocket` and the raw socket seam patches `socket.socket` underneath
    it, which is a fact about the stack rather than a preference.

    EVERY MEMBER OF `InterceptorName` IS INSTALLABLE, which is why nothing here
    can be selected and quietly do nothing. `GRPC`, `WEBSOCKET` and `SSE` were
    members until selection went live and named no unit at all — they are
    protocols the byte seams parse, and `Protocol` is their home — so they were
    removed rather than left to be rejected: a name that cannot be spelled needs
    no validation. What IS rejected, in `__post_init__`, is a value that is not
    a member: `interceptors=("ssl",)` is the mistake a user actually makes, and
    matched against nothing it would install nothing in silence.
    """

    debug: bool = False
    before_send: BeforeSendCallback | None = None
    intercept: bool = False
    intercept_hosts: tuple[str, ...] | None = None

    propagate_trace: bool = False
    propagate_targets: tuple[str, ...] | None = None
    capture_mode: CaptureMode = CaptureMode.AGENT

    release: str | None = None
    environment: str | None = None
    tags: tuple[tuple[str, str], ...] = ()

    def __new__(cls, *args: object, **kwargs: object) -> WardexConfig:
        moved = [name for name in _MOVED_TO_LIMITS if name in kwargs]
        if moved:
            names = ", ".join(moved)
            raise TypeError(
                f"{names} moved into limits= in 0.2.0b1. "
                f"Use limits=CaptureLimits({moved[0]}=...) instead."
            )
        # object.__new__, not super().__new__: @dataclass(slots=True) rebuilds
        # the class to attach __slots__, which invalidates the zero-arg
        # super() closure cell on some interpreters (observed on CPython
        # 3.13; not 3.14). object.__new__(cls) sidesteps that entirely.
        return object.__new__(cls)

    def __post_init__(self) -> None:
        """Validate field invariants (intervals, PII mode support). Limit values
        (buffer sizes, etc.) are validated by CaptureLimits.__post_init__."""
        if self.pii_mode in (PIIMode.REDACT, PIIMode.HASH):
            raise NotImplementedError(
                f"PIIMode.{self.pii_mode.name} is not implemented yet "
                "(v1 supports MASK/OFF; see the PII masking design doc)"
            )
        if self.flush_interval <= 0:
            raise ValueError(f"flush_interval must be > 0, got {self.flush_interval}")
        if self.propagate_targets is not None:
            for pattern in self.propagate_targets:
                if not isinstance(pattern, str) or not pattern:
                    raise ValueError(
                        f"propagate_targets entries must be non-empty glob strings, got {pattern!r}"
                    )
        if self.interceptors is not None:
            # Refused HERE, where the mistake was made, rather than skipped at
            # install time. `install_configured_interceptors` walks its own
            # table and keeps what was asked for, so a value it does not
            # recognize — `interceptors=("ssl",)`, the string, is the one a user
            # actually writes — matches nothing and installs nothing, in
            # silence, which is indistinguishable from `intercept=False`.
            for name in self.interceptors:
                if not isinstance(name, InterceptorName):
                    raise ValueError(
                        f"interceptors entries must be InterceptorName members, got {name!r}"
                    )

    @property
    def effective_retention(self) -> RetentionClass:
        """Return the effective Retention class based on the environment.

        local/staging/development environments are upgraded to REPLAYABLE.
        """
        if self.environment in ("local", "staging", "development"):
            return RetentionClass.REPLAYABLE
        return self.default_retention

    @classmethod
    def from_env(cls, **overrides: object) -> WardexConfig:
        """Read configuration from environment variables. Can be overridden via overrides.

        Environment variables read:
        - WARDEX_API_KEY
        - WARDEX_ENDPOINT
        - WARDEX_ENVIRONMENT
        - WARDEX_DEBUG (true/false, case-insensitive)
        """

        def pick(key: str, env: str) -> object | None:
            """Select a value in overrides → env order."""
            if overrides.get(key) is not None:
                return overrides.get(key)
            return os.environ.get(env)

        return cls(
            api_key=pick("api_key", "WARDEX_API_KEY"),  # type: ignore[arg-type]
            endpoint=pick("endpoint", "WARDEX_ENDPOINT"),  # type: ignore[arg-type]
            environment=pick("environment", "WARDEX_ENVIRONMENT"),  # type: ignore[arg-type]
            debug=bool(overrides["debug"])
            if "debug" in overrides
            else os.environ.get("WARDEX_DEBUG", "").lower() == "true",
        )
