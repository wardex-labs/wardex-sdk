import threading

from wardex_sdk import _hub
from wardex_sdk._scope import Scope


def setup_function():
    _hub.reset_for_test()


def test_global_scope_singleton():
    assert _hub.get_global_scope() is _hub.get_global_scope()


def test_new_scope_is_forked_and_restored():
    _hub.get_current_scope().set_tag("base", "1")
    with _hub.new_scope() as scope:
        assert isinstance(scope, Scope)
        scope.set_tag("inner", "x")
        assert _hub.get_current_scope().tags.get("inner") == "x"
    assert _hub.get_current_scope().tags.get("inner") is None
    assert _hub.get_current_scope().tags.get("base") == "1"


def test_isolation_scope_leaves_the_trace_position_alone():
    """Isolating identity must not cut the tree: the current scope, which holds
    the active span and the conversation, is the same object inside the block."""
    outer = _hub.get_current_scope()
    with _hub.isolation_scope():
        assert _hub.get_current_scope() is outer
    assert _hub.get_current_scope() is outer


def test_continuation_restarts_the_trace_but_keeps_the_conversation():
    from wardex_sdk._types import ConversationContext

    outer = _hub.get_current_scope()
    outer.conversation = ConversationContext(conversation_id="c")
    with _hub.isolation_scope(restart_trace=True):
        inner = _hub.get_current_scope()
        assert inner is not outer
        assert inner.active_span_context is None
        assert inner.conversation is outer.conversation
    assert _hub.get_current_scope() is outer


def test_isolation_scope_forks_the_enclosing_isolation_scope():
    """Sentry 2.x semantics: ambient context inherited, mutations isolated.

    The block used to start from a BLANK isolation scope, so a tag set on the
    enclosing one — a tenant id, a user — vanished inside every
    `isolation_scope()` block. A fork inherits it."""
    _hub.get_isolation_scope().set_tag("tenant", "acme")
    with _hub.isolation_scope() as forked:
        assert forked.tags.get("tenant") == "acme"
        assert _hub.get_isolation_scope() is forked


def test_isolation_scope_mutations_do_not_leak_out():
    _hub.get_isolation_scope().set_tag("tenant", "acme")
    with _hub.isolation_scope():
        _hub.get_isolation_scope().set_tag("tenant", "inner")
        _hub.get_isolation_scope().set_tag("request", "r1")
    assert _hub.get_isolation_scope().tags.get("tenant") == "acme"
    assert _hub.get_isolation_scope().tags.get("request") is None


def test_isolation_scope_tolerates_uncopyable_context_values():
    """`set_context()` documents "arbitrary host objects", and the fork on
    block entry clones the enclosing isolation scope — so a lock stored there
    must ride through the clone instead of blowing up `with isolation_scope()`
    with a pickling TypeError (I6: never raise into host code)."""
    lock = threading.Lock()
    _hub.get_isolation_scope().set_context("runtime", {"lock": lock})
    with _hub.isolation_scope() as forked:
        assert forked.contexts["runtime"]["lock"] is lock
        forked.set_context("runtime", {"lock": "replaced"})
    assert _hub.get_isolation_scope().contexts["runtime"]["lock"] is lock
