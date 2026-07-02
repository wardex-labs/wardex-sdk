from wardex_sdk._scope import Scope, UserInfo, merge_scopes


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
