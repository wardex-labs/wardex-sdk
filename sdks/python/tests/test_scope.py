import threading

import pytest

from wardex_sdk._scope import Scope, UserInfo, merge_scopes, merged_trace_fields
from wardex_sdk._types import SpanContext, SpanId, TraceId


def _ctx() -> SpanContext:
    return SpanContext(trace_id=TraceId.generate(), span_id=SpanId.generate(), trace_flags=1)


def test_set_tag_and_clone_isolation():
    s = Scope()
    s.set_tag("a", "1")
    c = s.clone()
    c.set_tag("a", "2")
    assert s.tags["a"] == "1"
    assert c.tags["a"] == "2"


def test_merge_layers_current_overrides():
    g = Scope()
    g.set_tag("env", "prod")
    g.set_tag("team", "core")
    iso = Scope()
    iso.set_tag("req", "r1")
    cur = Scope()
    cur.set_tag("env", "canary")
    merged = merge_scopes(g, iso, cur)
    assert merged.tags == {"env": "canary", "team": "core", "req": "r1"}


def test_merge_user_current_wins():
    g = Scope()
    g.set_user(UserInfo(id="g"))
    cur = Scope()
    cur.set_user(UserInfo(id="c"))
    merged = merge_scopes(g, Scope(), cur)
    assert merged.user is not None and merged.user.id == "c"


@pytest.mark.parametrize(
    "seed",
    [
        ("global", "isolation", "current"),
        ("global", "isolation", None),
        ("global", None, None),
        (None, "isolation", "current"),
        (None, None, "current"),
        (None, None, None),
        ("global", None, "current"),
    ],
)
def test_merged_trace_fields_agrees_with_the_full_merge(seed):
    """The cheap read is only allowed to be cheap, not to be different.

    `merged_trace_fields` restates `merge_scopes`' precedence for the two
    fields the W3C header readers need, so the two are asked the same question
    over every arrangement of which layers carry a value.
    """
    layers = []
    for name in seed:
        s = Scope()
        if name is not None:
            s.active_span_context = _ctx()
            s.tracestate = f"{name}=1"
        layers.append(s)
    g, iso, cur = layers
    merged = merge_scopes(g, iso, cur)
    assert merged_trace_fields(g, iso, cur) == (merged.active_span_context, merged.tracestate)


def test_merged_trace_fields_ignores_contexts_it_cannot_copy():
    """The reason it exists: `merge_scopes` deep-copies, this must not.

    `set_context()` takes arbitrary host objects, and `deepcopy` raises on the
    ones that do not copy — a lock, a socket, an open file. The two propagation
    fields are immutable scalars and are readable regardless.
    """
    g = Scope()
    g.set_context("runtime", {"lock": threading.Lock()})
    g.active_span_context = _ctx()
    with pytest.raises(TypeError):
        merge_scopes(g, Scope(), Scope())
    assert merged_trace_fields(g, Scope(), Scope()) == (g.active_span_context, None)
