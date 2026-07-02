"""Guards the invariant that the core knows nothing about any specific domain — design §1."""

import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_TYPES = _ROOT / "sdks" / "python" / "src" / "wardex_sdk" / "_types.py"
_STATE_PROTO = _ROOT / "proto" / "wardex" / "v1" / "state.proto"

_FORBIDDEN = ("HarnessState", "git_head_sha", "git_branch", "context_files", "cwd")


def test_core_types_have_no_domain_strings():
    text = _TYPES.read_text(encoding="utf-8")
    for token in _FORBIDDEN:
        assert token not in text, f"domain token '{token}' remains in core _types.py"


def test_state_proto_has_no_domain_strings():
    text = _STATE_PROTO.read_text(encoding="utf-8")
    for token in _FORBIDDEN + ("ContextFile",):
        assert token not in text, f"domain token '{token}' remains in state.proto"
