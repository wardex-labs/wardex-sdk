from wardex_sdk._protocol import WsParser


def _frame(fin: bool, opcode: int, payload: bytes, mask: bytes | None = None) -> bytes:
    out = bytearray()
    out.append((0x80 if fin else 0) | opcode)
    masked = 0x80 if mask else 0
    n = len(payload)
    if n < 126:
        out.append(masked | n)
    elif n <= 0xFFFF:
        out.append(masked | 126)
        out += n.to_bytes(2, "big")
    else:
        out.append(masked | 127)
        out += n.to_bytes(8, "big")
    if mask:
        out += mask
        out += bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
    else:
        out += payload
    return bytes(out)


def test_parses_text_message():
    p = WsParser()
    r = p.feed(_frame(True, 0x1, b"hello"))
    assert len(r.frames) == 1
    assert r.frames[0].opcode == "text"
    assert r.frames[0].fin is True
    assert r.frames[0].payload_len == 5
    assert list(r.messages) == [b"hello"]


def test_unmasks_client_frame():
    p = WsParser()
    r = p.feed(_frame(True, 0x1, b"hello", mask=b"\x01\x02\x03\x04"))
    assert list(r.messages) == [b"hello"]
    assert r.frames[0].masked is True


def test_close_code():
    p = WsParser()
    payload = (1000).to_bytes(2, "big") + b"bye"
    r = p.feed(_frame(True, 0x8, payload))
    assert r.frames[0].opcode == "close"
    assert r.frames[0].close_code == 1000
    assert list(r.messages) == []
