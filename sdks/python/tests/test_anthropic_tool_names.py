"""The shared tool-name space — design §5.4, and the trap it exists for.

One in-process tool call is seen by two observers under two different strings.
`Unit.claim()` arbitrates between observers of ONE key; hand it two keys and it
stops being an arbiter and becomes a duplicate generator. So every test here is
about the same question: do the handler's spelling and the hook's spelling land
on the same `UnitKey`?

Measured facts these tests encode, none of which are guessable from the Python
SDK alone:

  * `claude_agent_sdk` 0.2.122 does NOT namespace — the string `mcp__` appears
    zero times in it. The CLI adds the prefix.
  * The server token is the `mcp_servers` dict KEY, not
    `create_sdk_mcp_server(name=...)`, so it cannot be known when the handler is
    wrapped.
  * `CLAUDE_AGENT_SDK_MCP_NO_PREFIX` removes the prefix for `type: "sdk"`
    servers — precisely the ones wardex wraps.
  * The CLI's sanitizer is not injective, and one of its collapses defeats the
    CLI's own `split("__")` reverse parse.
"""

from __future__ import annotations

import pytest

from wardex_sdk.adapters._anthropic_names import (
    McpToolCatalog,
    builtin_tool_key,
    mcp_tool_key,
    prefix_disabled,
    sanitize,
)


def _catalog(*servers: tuple[str, tuple[str, ...]]) -> tuple[McpToolCatalog, list]:
    """A catalog with `servers` registered and their tokens resolved from a dict.

    Built the way `create_sdk_mcp_server` + `_prepare_options` build it: the
    handle is created with no token, and the token arrives later, by identity of
    the server object, from the KEY the user chose in `mcp_servers`.
    """
    catalog = McpToolCatalog()
    handles = []
    options = {}
    for key, tools in servers:
        handle = catalog.handle_for(f"{key}-server-name")
        handle.instance = object()
        handle.tools.update(tools)
        handles.append(handle)
        options[key] = {"type": "sdk", "instance": handle.instance}
    catalog.resolve_tokens(options)
    return catalog, handles


# ==========================================================================
# The two spellings meet
# ==========================================================================


def test_the_handler_and_the_hook_land_on_the_same_key():
    """THE assertion of this module. Everything else is a degradation of it.

    The handler knows `greet`; the CLI reports `mcp__tools__greet`. If these two
    produce different `UnitKey`s the arbitration never happens and the call is
    emitted twice — which is what the adapter used to do, because the skip list
    held the bare name and was compared against the namespaced one.
    """
    catalog, (handle,) = _catalog(("tools", ("greet",)))

    assert handle.key_for("greet") == catalog.key_for_hook("mcp__tools__greet")


def test_the_token_is_the_dict_key_not_the_server_name():
    """`create_sdk_mcp_server(name="my-tools")` + `mcp_servers={"tools": ...}`
    produces `mcp__tools__greet`, which the SDK's own README documents. Keying on
    the server's name instead would split the key space for every user who names
    the two differently — i.e. for the example in the SDK's README.
    """
    catalog = McpToolCatalog()
    handle = catalog.handle_for("my-tools")
    handle.instance = object()
    handle.tools.add("greet")

    catalog.resolve_tokens({"tools": {"type": "sdk", "instance": handle.instance}})

    assert handle.token == "tools"
    assert handle.key_for("greet") == mcp_tool_key("tools", "greet")


def test_an_unresolved_token_falls_back_to_the_server_name_and_says_so():
    """The fallback is a GUESS, and `token_resolved` is how the caller knows to
    mark the span: if the dict key differs from the server name the two observers
    are on different keys and the call ships twice.
    """
    catalog = McpToolCatalog()
    handle = catalog.handle_for("srv")
    handle.tools.add("greet")

    assert handle.token_resolved is False
    assert handle.key_for("greet") == mcp_tool_key("srv", "greet")


def test_resolution_is_by_instance_identity_not_by_name():
    """Two servers created from the same `name` are still two servers."""
    catalog = McpToolCatalog()
    first = catalog.handle_for("srv")
    second = catalog.handle_for("srv")
    first.instance, second.instance = object(), object()

    catalog.resolve_tokens(
        {
            "alpha": {"type": "sdk", "instance": first.instance},
            "beta": {"type": "sdk", "instance": second.instance},
        }
    )

    assert (first.token, second.token) == ("alpha", "beta")


def test_a_reregistration_reuses_the_handle_the_wrapper_already_holds():
    """ "Fresh options per query, reused @tool definitions" is a documented
    pattern, and the wrapper closes over the handle it was built with. Minting a
    second handle for the second `create_sdk_mcp_server` call would leave the
    wrapper holding a token that no later `_prepare_options` ever resolves.
    """
    catalog = McpToolCatalog()
    first = catalog.handle_for("srv")

    assert catalog.handle_for("srv", first) is first


# ==========================================================================
# The CLI's grammar, reproduced
# ==========================================================================


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a.b", "a_b"),
        ("a b", "a_b"),  # collapses onto the previous one — not injective
        ("a  b", "a__b"),  # produces `__` INSIDE the token
        ("keep-me_09", "keep-me_09"),
        ("emoji🙂", "emoji_"),
    ],
)
def test_the_sanitizer_is_the_clis(raw, expected):
    assert sanitize(raw) == expected


def test_a_sanitized_server_name_still_lands_on_one_key():
    """Idempotency is load-bearing, so assert what it is load-bearing FOR.

    A token recovered from a prefixed hook name has already been through the
    CLI's sanitizer, while the handler side arrives raw from the `mcp_servers`
    dict key, and `mcp_tool_key` sanitizes both. `sanitize(sanitize(x)) ==
    sanitize(x)` cannot detect a broken sanitizer — idempotency is structural
    for anything shaped like "replace unsafe characters with a safe one", and
    the identity function satisfies it too. What a broken sanitizer actually
    costs is this: the two observers land on different keys, `claim()` never
    meets itself, and the call ships twice.
    """
    catalog, (handle,) = _catalog(("a b", ("greet",)))

    # The CLI reports `mcp__a_b__greet` for a server whose dict key is `a b`.
    assert catalog.key_for_hook("mcp__a_b__greet") == mcp_tool_key("a b", "greet")
    assert catalog.key_for_hook("mcp__a_b__greet") == handle.key_for("greet")


@pytest.mark.parametrize("server_key", ["a__b", "a  b", "my__tools"])
def test_a_server_key_carrying_the_separator_still_lands_on_one_key(server_key):
    """The separator is not a separator, and splitting on it silently doubles.

    `mcp__` + token + `__` + tool is only parseable positionally while the token
    is free of `__` — which a dict key is not required to be, and which the
    sanitizer manufactures on its own from any two adjacent unsafe characters.
    Split on the first `__` and `mcp__a__b__greet` reads as server `a`, tool
    `b__greet`, while the handler holds server `a__b`, tool `greet`.

    That is the module's whole failure mode reached from the other side: not a
    dropped span but a duplicated one, and `ambiguous_bare` is False — correctly,
    since with the prefix on there is no bare-name ambiguity — so nothing marks
    it. Asserting the keys match is therefore asserting the call is billed,
    displayed and counted ONCE.
    """
    catalog, (handle,) = _catalog((server_key, ("greet",)))

    hook_name = f"mcp__{handle.effective_token}__greet"

    assert catalog.key_for_hook(hook_name) == handle.key_for("greet")
    assert catalog.ambiguous_bare("greet") is False


def test_a_tool_name_the_cli_sanitized_stays_anchored_to_its_server():
    """Both halves arrive differently seasoned, and only one side is sanitized.

    The hook's string has been through the CLI's sanitizer end to end; the
    handle's `tools` set holds the raw `@tool` names. Anchoring on a server token
    means recognizing the tail as one of that server's tools, so comparing the
    two raw would fail for every tool whose name contains a space or a dot — and
    fail SILENTLY, by falling back to the positional split this test's server key
    is built to defeat.
    """
    catalog, (handle,) = _catalog(("a  b", ("say hi",)))

    assert catalog.key_for_hook("mcp__a__b__say_hi") == handle.key_for("say hi")


def test_two_wrapped_servers_that_both_explain_one_string_refuse_to_guess():
    """`a`+`b__greet` and `a__b`+`greet` are the same wire string.

    Nothing recovers which one ran, so the hook stands down rather than pick.
    That costs nothing it was going to contribute: for a tool wardex wrapped, a
    resolved key only ever loses to the handler in `outranked()` and opens no
    span, so None arrives at the same place. Guessing would not be symmetric with
    it — the loser's handler would face a rival observer holding its own key.
    """
    from wardex_sdk.assembly import counters

    catalog, _ = _catalog(("a", ("b__greet",)), ("a__b", ("greet",)))
    before = counters.get("adapters.anthropic.hook_tool_name_ambiguous")

    assert catalog.key_for_hook("mcp__a__b__greet") is None
    assert counters.get("adapters.anthropic.hook_tool_name_ambiguous") == before + 1


def test_a_builtin_tool_gets_its_own_key_space(monkeypatch):
    """`Read` is not `mcp.tool/tools/Read`. Nothing in this process can be a
    second observer of a builtin, so it must not be able to collide with a
    wrapped tool that happens to share a name.

    The collision IS the property, so the catalog here exports the colliding
    name — asserting this against a catalog that exports only `greet` tests the
    branch where the bare-name index is empty, which was never the broken one.
    While the CLI prefixes, the wrapped `Read` arrives as `mcp__tools__Read` and
    a bare `Read` can only be the builtin. Handing the builtin the wrapped
    tool's key instead is not a cosmetic mislabel: the handler claims that key
    at its own rank the first time the wrapped tool runs, and from then on every
    builtin `Read` is outranked in `_emit_tool` and discarded.
    """
    monkeypatch.delenv("CLAUDE_AGENT_SDK_MCP_NO_PREFIX", raising=False)
    catalog, _ = _catalog(("tools", ("greet", "Read")))

    key = catalog.key_for_hook("Read")

    assert key.namespace == "tool.name"
    assert key != mcp_tool_key("tools", "Read")
    assert catalog.key_for_hook("Bash") != key


def test_a_shared_bare_name_is_not_dropped_while_the_prefix_is_on(monkeypatch):
    """The other half of the same gate, and the worse half.

    With TWO wrapped servers exporting `Read`, the bare-name index answers
    "ambiguous" and the hook stands down entirely — `key_for_hook` returns None,
    which stops the observation from being opened, closed, or rebuilt from the
    stream, from the very first call. And nothing marks that loss: `ambiguous_bare`
    is gated on the variable and correctly reports False, while a builtin has no
    in-process handler span to hang `TOOL_NAME_COLLISION` on anyway. While the CLI
    prefixes there is no ambiguity to find, so there is nothing to stand down from
    and nothing to count.
    """
    from wardex_sdk.assembly import counters

    monkeypatch.delenv("CLAUDE_AGENT_SDK_MCP_NO_PREFIX", raising=False)
    catalog, _ = _catalog(("alpha", ("Read",)), ("beta", ("Read",)))
    before = counters.get("adapters.anthropic.hook_tool_name_ambiguous")

    assert catalog.key_for_hook("Read") == builtin_tool_key("Read")
    assert catalog.ambiguous_bare("Read") is False
    assert counters.get("adapters.anthropic.hook_tool_name_ambiguous") == before


def test_an_external_mcp_server_reverse_parses_without_being_registered():
    """The prefix is the CLI's grammar, not wardex's bookkeeping: a tool from a
    server wardex never wrapped still resolves to a stable key, so the hook owns
    it unambiguously.
    """
    catalog, _ = _catalog(("tools", ("greet",)))

    assert catalog.key_for_hook("mcp__github__search") == mcp_tool_key("github", "search")


# ==========================================================================
# CLAUDE_AGENT_SDK_MCP_NO_PREFIX — the escape hatch that removes the prefix
# ==========================================================================


def test_a_bare_name_is_attributed_through_the_install_time_index(monkeypatch):
    """Under NO_PREFIX the hook sees `greet` with no server in it at all. The
    only thing that can put it back is the index of which server exports which
    bare name, built when the handlers were wrapped.
    """
    monkeypatch.setenv("CLAUDE_AGENT_SDK_MCP_NO_PREFIX", "1")
    catalog, (handle,) = _catalog(("tools", ("greet",)))

    assert catalog.key_for_hook("greet") == handle.key_for("greet")


def test_an_ambiguous_bare_name_refuses_to_guess(monkeypatch):
    """Two servers, one bare name, no prefix. Picking one fails in BOTH
    directions at once: for the other server's call the hook claims a key the
    handler never claimed (double emit), and for this one the hook claims first
    and the layer that actually wrapped the execution loses (§8.4 inverted). So
    the hook does not participate in the arbitration at all.
    """
    monkeypatch.setenv("CLAUDE_AGENT_SDK_MCP_NO_PREFIX", "1")
    catalog, _ = _catalog(("alpha", ("search",)), ("beta", ("search",)))

    assert catalog.key_for_hook("search") is None


def test_ambiguity_is_only_reported_while_the_prefix_is_off(monkeypatch):
    """The marker has to be false when the CLI is prefixing, because then the
    two servers ARE told apart. A span that claims an ambiguity it does not have
    is a claim it cannot back.
    """
    catalog, _ = _catalog(("alpha", ("search",)), ("beta", ("search",)))

    monkeypatch.delenv("CLAUDE_AGENT_SDK_MCP_NO_PREFIX", raising=False)
    assert catalog.ambiguous_bare("search") is False

    monkeypatch.setenv("CLAUDE_AGENT_SDK_MCP_NO_PREFIX", "1")
    assert catalog.ambiguous_bare("search") is True
    assert catalog.ambiguous_bare("greet") is False


@pytest.mark.parametrize(
    ("value", "disabled"),
    [("1", True), ("true", True), ("yes", True), ("", False), ("0", False), ("false", False)],
)
def test_the_env_var_is_read_at_call_time(value, disabled):
    """Read per call, not at install: the variable is consumed by a CLI
    subprocess that has not been spawned yet when the adapter installs.
    """
    assert prefix_disabled({"CLAUDE_AGENT_SDK_MCP_NO_PREFIX": value}) is disabled


def test_no_variable_at_all_means_the_prefix_is_on():
    assert prefix_disabled({}) is False


# ==========================================================================
# Bounds
# ==========================================================================


def test_the_server_table_is_bounded_and_drops_the_oldest():
    """A host that builds a server per query would otherwise accumulate handles
    for the process lifetime. What overflow costs is a bare-name LOOKUP, never a
    span: the tool falls back to the builtin key space, where at worst one call
    is observed twice.
    """
    from wardex_sdk.assembly import counters

    counters.reset()
    catalog = McpToolCatalog(max_entries=2)
    for _ in range(4):
        catalog.handle_for("srv")

    assert counters.get("adapters.anthropic.server_table_full") == 2
    counters.reset()


def test_the_bound_comes_from_the_core():
    from wardex_sdk import _wardex_native

    assert McpToolCatalog()._max == _wardex_native.limits_defaults()["max_entries_per_unit"]
