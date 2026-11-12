"""What wardex tells a user about a loss or a marker is true and complete.

Three promises. Every limitation marker a backend can show and every loss reason
`wardex.diagnostics()` can name has an explanation (`wardex.explain`). And
every setting those sentences tell a user to change exists, under the name
written: a next step that names a knob nobody can find sends the reader
nowhere. And no line wardex prints names a private counter, which no user
can read.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib
import re

import pytest

import wardex_sdk
from wardex_sdk import BatchingConfig, CodexExecConfig, LimitsConfig
from wardex_sdk._assembly import Limitation
from wardex_sdk._assembly._explain import MARKERS, markdown
from wardex_sdk._assembly._ledger import REASONS, DropReason


def _fields(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


_CONFIG_CLASSES = {
    "LimitsConfig": _fields(LimitsConfig),
    "BatchingConfig": _fields(BatchingConfig),
    "CodexExecConfig": _fields(CodexExecConfig),
}


def _user_texts() -> dict[str, str]:
    texts = {}
    for reason, info in REASONS.items():
        texts[f"reason {reason.value} meaning"] = info.meaning
        texts[f"reason {reason.value} next step"] = info.next_step
    for marker, text in MARKERS.items():
        for part, sentence in text._asdict().items():
            texts[f"marker {marker.value} {part}"] = sentence
    return texts


def test_every_marker_and_every_reason_has_an_explanation():
    assert set(MARKERS) == set(Limitation)
    assert set(REASONS) == set(DropReason)
    for marker in Limitation:
        assert all(MARKERS[marker]), marker
    for name in [m.value for m in Limitation] + [r.value for r in DropReason]:
        said = wardex_sdk.explain(name)
        assert said.startswith(name), (name, said)
        assert "is not a wardex" not in said


def test_explain_answers_a_marker_seen_in_a_backend_with_what_to_do():
    said = wardex_sdk.explain("unit_inferred_sole")
    assert "exactly one run was live" in said
    assert "0.5" in said
    said = wardex_sdk.explain("otlp_attribute_truncated")
    assert "LimitsConfig(max_otlp_attribute_bytes=...)" in said


def test_explain_reads_a_loss_by_its_wire_key_and_a_shared_name_twice():
    assert wardex_sdk.explain("queue_overflow:span") == wardex_sdk.explain("queue_overflow")
    both = wardex_sdk.explain("connection_evicted")
    assert "limitation marker" in both and "loss reason" in both


def test_explain_says_so_for_a_name_it_does_not_know_and_never_raises():
    said = wardex_sdk.explain("no_such_marker")
    assert "is not a wardex limitation marker or loss reason" in said
    assert "python -m wardex_sdk explain --all" in said


def test_the_generated_list_names_every_marker_and_reason():
    text = markdown()
    for name in [m.value for m in Limitation] + [r.value for r in DropReason]:
        assert f"### `{name}`" in text, name


def test_the_published_list_is_what_explain_prints():
    """LIMITATIONS.md is `python -m wardex_sdk explain --all`: an explanation
    changed without regenerating it publishes the old sentence."""
    published = pathlib.Path(__file__).resolve().parents[3] / "LIMITATIONS.md"
    assert published.read_text(encoding="utf-8") == markdown(), (
        "regenerate it: python -m wardex_sdk explain --all > LIMITATIONS.md"
    )


_KNOB = re.compile(r"\b(max_[a-z0-9_]+|ws_sample_bytes|mcp_sniff_bytes)\b")
_CALL = re.compile(r"\b(LimitsConfig|BatchingConfig|CodexExecConfig)\((\w+)=")


@pytest.mark.parametrize("where", sorted(_user_texts()))
def test_every_setting_a_sentence_names_exists(where):
    text = _user_texts()[where]
    for cls, field in _CALL.findall(text):
        assert field in _CONFIG_CLASSES[cls], f"{where}: {cls} has no field {field!r}"
    for knob in _KNOB.findall(text):
        assert knob in _CONFIG_CLASSES["LimitsConfig"], f"{where}: no limit named {knob!r}"


#: A next step that points at a bound without naming it, or names a private
#: counter: the reader cannot act on either.
_VAGUE = re.compile(
    r"\bRaise the\b|'s bound\b|\bthe [a-z -]*(?:sample|frame|decoded|capture) limit\b|"
    r"\b(?:transport|client|adapters|interceptors|assembly|protocol|semantics)\.[a-z_]+\.[a-z_]+"
)


@pytest.mark.parametrize("where", sorted(_user_texts()))
def test_no_sentence_points_at_an_unnamed_bound_or_a_private_counter(where):
    text = _user_texts()[where]
    assert not _VAGUE.search(text), f"{where}: {text!r}"


# -- the lines wardex prints --------------------------------------------------

_SRC = pathlib.Path(wardex_sdk.__file__).parent

#: The calls whose text reaches stderr or a host's log handler.
_SAYING = frozenset({"report_once", "diag_warning", "diag_info", "report_export_failed", "lose"})

#: Arguments that are keys or structured fields, not text a reader sees.
_NOT_SAID = frozenset({"key", "where", "reason", "subject", "unprintable", "count"})

#: A private counter's name (dotted, or filled in at run time as `{}`), or a
#: setting spelled the way no host writes it. A loss reason (`queue_overflow`)
#: is public and passes.
_PRIVATE = re.compile(r"[Cc]ounted under `?(?:\{\}|\w+\.[\w{])|\bthe counter \b|\blimits\.max_")

#: Lines that still name a private counter, per module, owned by changes to
#: those modules in flight. The list only shrinks: a module whose lines are
#: fixed fails here until its entry is lowered or deleted.
_STILL_PRIVATE: dict[str, int] = {}


def _module_strings(tree: ast.Module) -> dict[str, list[ast.expr]]:
    """Module-level names bound to a string expression, for a line held in a constant."""
    found: dict[str, list[ast.expr]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                found[target.id] = [node.value]
    return found


def _said_text(call: ast.Call, constants: dict[str, list[ast.expr]]) -> str:
    parts = list(call.args) + [k.value for k in call.keywords if k.arg not in _NOT_SAID]
    for part in list(parts):
        if isinstance(part, ast.Name):
            parts += constants.get(part.id, [])
    return " ".join(_render(part) for part in parts)


def _render(node: ast.expr) -> str:
    """A string expression's text, with each run-time value as `{}`."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else ""
    if isinstance(node, ast.JoinedStr):
        return "".join(_render(v) if isinstance(v, ast.Constant) else "{}" for v in node.values)
    if isinstance(node, ast.BinOp):
        return _render(node.left) + _render(node.right)
    return " ".join(_render(child) for child in ast.iter_child_nodes(node))


def _private_lines() -> dict[str, int]:
    found: dict[str, int] = {}
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = _module_strings(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name in _SAYING and _PRIVATE.search(_said_text(node, constants)):
                module = path.relative_to(_SRC).as_posix()
                found[module] = found.get(module, 0) + 1
    return found


def test_no_printed_line_names_a_private_counter():
    """A user reads these lines; `counters` is not public. A line points at
    `wardex.diagnostics().losses` and at a setting as a host writes it."""
    found = _private_lines()
    grew = {m: n for m, n in found.items() if n > _STILL_PRIVATE.get(m, 0)}
    assert not grew, f"printed lines that name a private counter or limits.<x>: {grew}"
    stale = {m: n for m, n in _STILL_PRIVATE.items() if found.get(m, 0) < n}
    assert not stale, f"fixed; lower or delete their _STILL_PRIVATE entries: {stale}"


def test_the_printed_line_scan_sees_a_counter_and_ignores_a_key():
    tree = ast.parse(
        "MSG = 'gone (counted under x.y)'\n"
        "report_once('lost; counted under client.buffer.evicted', key='client.buffer.evicted')\n"
        "report_once(MSG, key='k')\n"
        "report_once(f'lost (counted under {counter})', key='k')\n"
        "report_once(f'lost (counted under transport.{wire}.full)', key='k')\n"
        "report_once('lost', key='counted under x.y')\n"
        "report_once('lost; diagnostics() has it counted under queue_overflow.', key='k')\n"
    )
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    said = [bool(_PRIVATE.search(_said_text(c, _module_strings(tree)))) for c in calls]
    assert said == [True, True, True, True, False, False]
