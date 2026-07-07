"""Configuration object — spec §2.3.

WardexConfig is the immutable configuration for SDK initialization. All fields have
sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ._enums import (
    AdapterName,
    CaptureTrigger,
    InterceptorName,
    PIICategory,
    PIIMode,
    RetentionClass,
)
from ._types import BeforeSendCallback


@dataclass(frozen=True, slots=True)
class WardexConfig:
    """Wardex SDK configuration. An immutable dataclass."""

    api_key: str | None = None
    endpoint: str | None = None

    default_retention: RetentionClass = RetentionClass.SUMMARY_ONLY
    replay_buffer_size: int = 100
    retention_triggers: frozenset[CaptureTrigger] = frozenset(
        {CaptureTrigger.ERROR, CaptureTrigger.MANUAL_MARK}
    )

    pii_mode: PIIMode = PIIMode.MASK
    pii_disabled_categories: frozenset[PIICategory] = frozenset()

    adapters: tuple[AdapterName, ...] | None = None
    interceptors: tuple[InterceptorName, ...] | None = None

    debug: bool = False
    before_send: BeforeSendCallback | None = None
    intercept: bool = False
    intercept_hosts: tuple[str, ...] | None = None

    release: str | None = None
    environment: str | None = None
    tags: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Validation: replay_buffer_size >= 1."""
        if self.replay_buffer_size < 1:
            raise ValueError(f"replay_buffer_size must be >= 1, got {self.replay_buffer_size}")
        if self.pii_mode in (PIIMode.REDACT, PIIMode.HASH):
            raise NotImplementedError(
                f"PIIMode.{self.pii_mode.name} is not implemented yet "
                "(v1 supports MASK/OFF; see the PII masking design doc)"
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
