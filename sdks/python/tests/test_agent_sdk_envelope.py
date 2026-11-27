"""What the Agent SDK adapter said about a conversation survives the real encoder.

A scripted CLI stream through the real session assembler, encoded, decoded
and held to the contract in `_adapter_envelopes.py`.
"""

from __future__ import annotations

from _adapter_envelopes import deliver
from test_agent_sdk_assembler import ASSISTANT, INIT, RESULT, FakeClient, _outbound, _submit
from test_codec import _header
from wardex_sdk._adapters._assembler import SessionAssembler
from wardex_sdk._types import Envelope
from wardex_sdk.transport import _codec


def test_fresh_agent_sdk_envelope():
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1, text="first question")
    asm.on_inbound(1, INIT)
    _submit(asm, "first question")
    asm.on_inbound(1, ASSISTANT)
    asm.on_inbound(1, RESULT)
    asm.on_close(1, None)
    deliver("agent_sdk", _codec.encode(Envelope(header=_header(), spans=tuple(client.spans))))
