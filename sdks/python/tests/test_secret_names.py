"""What a masked span says about its masking, and the two knobs an application
has over the name rules.

A value that reads `[SECRET]` must be explainable from the span alone: how
many values were replaced, by which rule, under which argument names. And the
name rules must bend where an application knows better — `extra_secret_names`
adds names, `reveal_names` exempts them — without ever exempting a value whose
own shape says it is a credential.
"""

from __future__ import annotations

import http.client
import json
import threading
import warnings
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub, _wardex_native
from wardex_sdk._config import PIIConfig
from wardex_sdk._enums import CaptureMode, PIIMode
from wardex_sdk.transport import Transport


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _reply(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    do_GET = _reply
    do_POST = _reply

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture(scope="module")
def port() -> Iterator[int]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


#: Categories off in tests that count replacements exactly. The loopback
#: address is an IP, and a connection id is a process object id that passes
#: the card checksum about one run in ten — both are replaced, correctly or
#: not, independently of the rule under test.
_OWN_IDS_OFF = frozenset({wardex.PIICategory.IP_ADDRESS, wardex.PIICategory.CREDIT_CARD})


class _Capture(Transport):
    def __init__(self) -> None:
        self.otlp: list[bytes] = []
        self.envelopes: list[bytes] = []

    def export(self, envelope: Any, *, timeout: float | None = None) -> None:
        self.otlp.extend(self.encode(envelope, compress=False))
        self.envelopes.append(
            _wardex_native.codec.encode_envelope(
                envelope,
                self._pii_mode,
                list(self._pii_disabled),
                self._limits,
                **self._pii_names(),
            )
        )


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    _hub.reset_for_test()
    yield
    from wardex_sdk._interceptors._registry import get_registry

    get_registry().uninstall_all()
    _hub.reset_for_test()


def _call(port: int, method: str, target: str, body: dict | None = None) -> None:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        data = json.dumps(body).encode() if body is not None else None
        c.request(method, target, data, {"Content-Type": "application/json"})
        c.getresponse().read()
    finally:
        c.close()


def _run(port: int, calls: list[tuple], pii: PIIConfig | None = None):
    cap = _Capture()
    kw = {"pii": pii} if pii is not None else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wardex.init(transport=cap, intercept=True, capture_mode=CaptureMode.ALL, **kw)
    for call in calls:
        _call(port, *call)
    wardex.flush()
    env_spans = [
        it["span"]
        for b in cap.envelopes
        for it in _wardex_native.codec.decode_envelope(b)["items"]
        if "span" in it
    ]
    otlp_spans = [
        sp
        for b in cap.otlp
        for rs in _wardex_native.codec.decode_otlp_traces(b)["resource_spans"]
        for ss in rs["scope_spans"]
        for sp in ss["spans"]
    ]
    return env_spans, otlp_spans


def _by_path(spans: list[dict], path: str) -> dict:
    (sp,) = [s for s in spans if s["name"].endswith(" " + path)]
    return sp


def test_a_masked_span_says_how_many_by_which_rule_under_which_names(port):
    env, otlp = _run(
        port,
        [
            ("POST", "/search?q=seoul&appid=owm999key", {"query": "seoul", "api_key": "abc123"}),
            ("GET", "/clean?q=seoul&page=2", None),
        ],
        pii=PIIConfig(disabled_categories=_OWN_IDS_OFF),
    )
    ci = _by_path(env, "/search")["capture_integrity"]
    assert ci["redacted"] is True
    assert ci["redaction_count"] == 2
    assert ci["redaction_rules"] == ["secret_last_word", "secret_exact_name"]
    assert sorted(ci["redaction_names"]) == ["api_key", "appid"]
    url = _by_path(env, "/search")["transport"]["http"]["url"]
    assert url.endswith("/search?q=seoul&appid=[SECRET]"), url

    attrs = _by_path(otlp, "/search")["attributes"]
    assert attrs["url.full"].endswith("/search?q=seoul&appid=[SECRET]")
    assert attrs["wardex.redacted"] is True
    assert attrs["wardex.redaction.count"] == 2
    assert sorted(attrs["wardex.redaction.rules"]) == ["secret_exact_name", "secret_last_word"]
    assert sorted(attrs["wardex.redaction.names"]) == ["api_key", "appid"]
    assert '"api_key": "[SECRET]"' in attrs["wardex.input_data"]

    # Nothing replaced, nothing said.
    clean_ci = _by_path(env, "/clean")["capture_integrity"]
    assert clean_ci["redaction_count"] == 0
    assert clean_ci["redaction_rules"] == [] and clean_ci["redaction_names"] == []
    clean = _by_path(otlp, "/clean")["attributes"]
    assert not any(k.startswith("wardex.redact") for k in clean)
    assert clean["url.full"].endswith("/clean?q=seoul&page=2")


def test_a_name_that_is_itself_personal_data_is_masked_in_the_names_list(port):
    env, otlp = _run(
        port,
        [("POST", "/k", {"john.doe@example.com_token": "t0ken-value"})],
        pii=PIIConfig(disabled_categories=_OWN_IDS_OFF),
    )
    names = _by_path(env, "/k")["capture_integrity"]["redaction_names"]
    assert names == ["[EMAIL]_token"]
    assert _by_path(otlp, "/k")["attributes"]["wardex.redaction.names"] == ["[EMAIL]_token"]
    assert "john.doe" not in repr(env) + repr(otlp)


def test_extra_secret_names_masks_every_spelling_of_the_same_words(port):
    env, otlp = _run(
        port,
        [
            ("GET", "/a?x_corp_widget=w1dget&q=1", None),
            ("POST", "/b", {"xCorpWidget": "w2dget", "q": 1}),
        ],
        pii=PIIConfig(
            extra_secret_names={"x_corp_widget"},
            disabled_categories=_OWN_IDS_OFF,
        ),
    )
    text = repr(env) + repr(otlp)
    assert "w1dget" not in text and "w2dget" not in text
    assert _by_path(env, "/b")["capture_integrity"]["redaction_rules"] == ["secret_user_name"]


def test_reveal_names_exempts_the_name_rules_but_not_a_credential_shaped_value(port):
    env, otlp = _run(
        port,
        [
            ("GET", "/cb?code=abc123&state=s", None),
            ("GET", "/cb2?code=sk-abcdefghijklmnop1234", None),
            ("POST", "/list", {"pageToken": "cursor-42"}),
        ],
        pii=PIIConfig(
            reveal_names={"code", "page_token"},
            disabled_categories=_OWN_IDS_OFF,
        ),
    )
    otlp_text = repr(otlp)
    assert "code=abc123" in otlp_text
    assert "cursor-42" in otlp_text
    assert "sk-abcdefghijklmnop1234" not in otlp_text + repr(env)
    assert _by_path(env, "/cb2")["capture_integrity"]["redaction_rules"] == ["secret_value"]


def test_without_reveal_names_the_same_calls_are_masked(port):
    env, otlp = _run(
        port,
        [("GET", "/cb?code=abc123&state=s", None), ("POST", "/list", {"pageToken": "cursor-42"})],
    )
    text = repr(env) + repr(otlp)
    assert "abc123" not in text and "cursor-42" not in text


@pytest.mark.parametrize("mode", [PIIMode.MASK, PIIMode.OFF])
def test_url_credentials_are_replaced_in_every_mode(port, mode):
    """Semantic conventions require them out of a URL, and PIIMode.OFF opts out
    of losing debugging values — which `user:password@` never is."""
    env, otlp = _run(
        port, [("GET", f"http://user:S3CRETpw@127.0.0.1:{port}/p", None)], pii=PIIConfig(mode=mode)
    )
    text = repr(env) + repr(otlp)
    assert "S3CRETpw" not in text
    assert "REDACTED:REDACTED@" in text
    for sp in env + otlp:
        assert "REDACTED" not in sp["name"] and "@" not in sp["name"], sp["name"]
    (sp,) = [s for s in otlp if s["name"].startswith("HTTP GET")]
    # MASK also replaces the loopback address (`ip_address`); OFF runs only this rule.
    assert "url_userinfo" in sp["attributes"]["wardex.redaction.rules"]
    if mode is PIIMode.OFF:
        assert sp["attributes"]["wardex.redaction.rules"] == ["url_userinfo"]


def test_names_are_validated_where_the_masker_could_never_match_them():
    with pytest.raises(ValueError, match="collection of names"):
        PIIConfig(reveal_names="code")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be strings"):
        PIIConfig(extra_secret_names={1})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no letters or digits"):
        PIIConfig(extra_secret_names={"__"})
    with pytest.raises(ValueError, match="masked or revealed, not both"):
        PIIConfig(extra_secret_names={"apiKey"}, reveal_names={"api_key"})
    cfg = PIIConfig(extra_secret_names=(n for n in ["a_b"]), reveal_names=["c"])
    assert cfg.extra_secret_names == frozenset({"a_b"})
    assert cfg.reveal_names == frozenset({"c"})


def test_names_under_pii_off_warn_that_they_do_nothing():
    with pytest.warns(wardex.WardexConfigWarning, match="no effect when pii mode is OFF"):
        wardex.init(
            transport=_Capture(),
            pii=PIIConfig(mode=PIIMode.OFF, reveal_names={"code"}),
        )


def test_a_span_attribute_under_a_secret_name_is_masked(port):
    """`s.set_attribute("api_key", ...)` is the same argument as `?api_key=`."""
    cap = _Capture()
    wardex.init(transport=cap, pii=PIIConfig(disabled_categories=frozenset()))
    with wardex.span("rank") as s:
        s.set_attribute("db_password", "hunter2")
        s.set_attribute("candidates", 42)
    wardex.flush()
    text = repr([_wardex_native.codec.decode_envelope(b) for b in cap.envelopes])
    text += repr([_wardex_native.codec.decode_otlp_traces(b) for b in cap.otlp])
    assert "hunter2" not in text
    assert "db_password" in text and "42" in text


def _readme_rule_rows() -> dict[str, set[str]]:
    import re
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[3] / "README.md").read_text()
    rows: dict[str, set[str]] = {}
    for line in readme.splitlines():
        m = re.match(r"^\| `(secret_[a-z_]+)`(, `name=value` only)? \|[^|]*\| (.*) \|$", line)
        if m:
            key = m.group(1) + (":url_form" if m.group(2) else "")
            rows[key] = set(re.findall(r"`([^`]+)`", m.group(3)))
    return rows


def test_the_readme_name_table_is_the_list_the_masker_runs():
    """The documentation prints the rule verbatim; this is what keeps the two
    one fact. A name added to the code and not to the README (or the other
    way round) fails here."""
    rules = _wardex_native.codec.pii_name_rules()
    assert _readme_rule_rows() == {
        "secret_word": set(rules["strong_words"]),
        "secret_last_word": set(rules["last_words"]),
        "secret_exact_name": set(rules["exact_names"]),
        "secret_exact_name:url_form": set(rules["url_form_only_names"]),
    }


_MASKED_DIR = Path(__file__).parent / "fixtures" / "masked_envelopes"


def _check_masked_body(body: bytes) -> None:
    """The contract a receiver can rely on for a body the masker touched: the
    secrets are gone, the arguments are not, and the report says why."""
    spans = [
        it["span"] for it in _wardex_native.codec.decode_envelope(body)["items"] if "span" in it
    ]
    (http,) = [s for s in spans if s["name"] == "HTTP POST /search"]
    assert "S3CRET" not in repr(spans)
    assert http["transport"]["http"]["url"].endswith("/search?q=seoul&appid=[SECRET]")
    ci = http["capture_integrity"]
    assert ci["redacted"] is True and ci["redaction_count"] == 2
    assert sorted(ci["redaction_rules"]) == ["secret_exact_name", "secret_last_word"]
    assert sorted(ci["redaction_names"]) == ["api_key", "appid"]


def test_a_masked_envelope_body_for_receivers(port):
    """A body exactly as `WardexTransport` would POST it after masking, for a
    receiver's tests to read in place of a hand-built one. Written only under
    `WARDEX_REGEN_ENVELOPES=1`; the committed copy is checked on every run."""
    import os

    cap = _Capture()
    wardex.init(
        transport=cap,
        intercept=True,
        capture_mode=CaptureMode.ALL,
        pii=PIIConfig(disabled_categories=_OWN_IDS_OFF),
    )
    _call(port, "POST", "/search?q=seoul&appid=S3CRETowm", {"query": "seoul", "api_key": "S3CRETk"})
    wardex.flush()
    # The body the wardex wire carries: the envelope encoder with the policy
    # `init()` installed, zstd included.
    (body,) = cap.envelopes
    _check_masked_body(body)
    if os.environ.get("WARDEX_REGEN_ENVELOPES") == "1":
        _MASKED_DIR.mkdir(parents=True, exist_ok=True)
        (_MASKED_DIR / "masked_http.envelope.zst").write_bytes(body)
    _check_masked_body((_MASKED_DIR / "masked_http.envelope.zst").read_bytes())


class _GzipHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        import gzip

        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = gzip.compress(b'{"access_token": "GZSEC1", "token_type": "Bearer"}')
        self.send_response(200)
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


def test_a_gzipped_response_is_captured_inflated_and_masked():
    """A token endpoint answering in gzip used to ship its `access_token` as
    compressed bytes anyone could inflate; the captured body is now the text
    it carries, masked like any other."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _GzipHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        env, otlp = _run(srv.server_address[1], [("POST", "/oauth/token", {"grant_type": "x"})])
    finally:
        srv.shutdown()
    text = repr(env) + repr(otlp)
    assert "GZSEC1" not in text
    (sp,) = [s for s in otlp if s["name"] == "HTTP POST /oauth/token"]
    assert sp["attributes"]["wardex.output_data"] == (
        '{"access_token": "[SECRET]", "token_type": "Bearer"}'
    )


def test_a_multipart_field_is_masked_like_a_form_field(port):
    cap = _Capture()
    wardex.init(transport=cap, intercept=True, capture_mode=CaptureMode.ALL)
    body = (
        b'--xb\r\nContent-Disposition: form-data; name="password"\r\n\r\nMPSEC1\r\n'
        b'--xb\r\nContent-Disposition: form-data; name="q"\r\n\r\nseoul-mp\r\n--xb--\r\n'
    )
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request("POST", "/upload", body, {"Content-Type": "multipart/form-data; boundary=xb"})
    c.getresponse().read()
    c.close()
    wardex.flush()
    text = repr([_wardex_native.codec.decode_envelope(b) for b in cap.envelopes])
    assert "MPSEC1" not in text and "seoul-mp" in text


def test_inflation_is_bounded_marked_and_never_invents_a_body():
    """A body that merely starts like a zlib header stays as it was; one that
    inflates past the bound keeps its inflated prefix and says it was cut."""
    import gzip
    import zlib

    from wardex_sdk._interceptors._trackers import _inflated

    class _T:
        truncated = False

    class _L:
        max_decoded_bytes = 1024
        max_opaque_body_bytes = 4096

    plain = b"HK: hello this is a plain text body that only starts like zlib"
    assert _inflated(_T(), plain, _L()) is plain
    txn = _T()
    big = gzip.compress(b'{"access_token": "BIG"}' + b" " * 5000)
    out = _inflated(txn, big, _L())
    assert out.startswith(b'{"access_token": "BIG"}') and len(out) == 1024
    assert txn.truncated is True
    small = _T()
    assert _inflated(small, zlib.compress(b"q=seoul"), _L()) == b"q=seoul"
    assert small.truncated is False
