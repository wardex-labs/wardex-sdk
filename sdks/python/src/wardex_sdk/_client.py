from __future__ import annotations

import platform
import sys
import time
import uuid

from ._config import WardexConfig
from ._types import (
    EnvelopeHeader,
    InternalEnvelope,
    InternalSpan,
    InternalStateSnapshot,
    SdkInfo,
)
from .transport._base import Transport


def build_sdk_info() -> SdkInfo:
    return SdkInfo(
        name="wardex.python",
        version="0.1.0",
        python_version=platform.python_version(),
        os=sys.platform,
        arch=platform.machine(),
    )


class Client:
    def __init__(self, config: WardexConfig, transport: Transport) -> None:
        self._config = config
        self._transport = transport
        self._sdk_info = build_sdk_info()
        self._spans: list[InternalSpan] = []
        self._snapshots: list[InternalStateSnapshot] = []
        self._closed = False

    @property
    def config(self) -> WardexConfig:
        return self._config

    def capture_span(self, span: InternalSpan) -> None:
        if self._closed:
            return
        self._spans.append(span)

    def capture_snapshot(self, snapshot: InternalStateSnapshot) -> None:
        if self._closed:
            return
        self._snapshots.append(snapshot)

    def flush(self, timeout: float = 5.0) -> None:
        if not self._spans and not self._snapshots:
            self._transport.flush(timeout)
            return
        header = EnvelopeHeader(
            event_id=str(uuid.uuid4()),
            api_key=self._config.api_key or "",
            sdk=self._sdk_info,
            sent_at_ns=time.time_ns(),
        )
        envelope = InternalEnvelope(
            header=header,
            spans=tuple(self._spans),
            state_snapshots=tuple(self._snapshots),
        )
        if self._config.before_send is not None:
            maybe = self._config.before_send(envelope)
            if maybe is None:
                self._spans.clear()
                self._snapshots.clear()
                return
            envelope = maybe
        self._transport.export(envelope)
        self._transport.flush(timeout)
        self._spans.clear()
        self._snapshots.clear()

    def close(self, timeout: float = 5.0) -> None:
        if self._closed:
            return
        self.flush(timeout)
        self._transport.close(timeout)
        self._closed = True
