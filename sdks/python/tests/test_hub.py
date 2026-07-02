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


def test_isolation_scope_resets_current():
    _hub.get_current_scope().set_tag("outer", "1")
    with _hub.isolation_scope():
        assert _hub.get_current_scope().tags.get("outer") is None
        _hub.get_current_scope().set_tag("req", "r1")
    assert _hub.get_current_scope().tags.get("req") is None
