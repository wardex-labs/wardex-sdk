"""`config.interceptors` selects which seams `intercept=True` installs.

The field was declared on `WardexConfig` and never read: `init()` installed SSL,
MCP-stdio and the raw socket seam unconditionally, so choosing a subset meant
editing `init()`. These tests are the difference between a config field and a
config field that does something.
"""

from __future__ import annotations

import pytest

from wardex_sdk._config import WardexConfig
from wardex_sdk._enums import InterceptorName
from wardex_sdk._interceptors import _INTERCEPTORS, install_configured_interceptors
from wardex_sdk._interceptors._registry import get_registry


@pytest.fixture(autouse=True)
def _no_seam_outlives_its_test():
    """These install the REAL seams, which patch `ssl`, `socket` and anyio's
    process spawning for the whole interpreter. A seam left behind is not this
    file's problem but every later test's."""
    get_registry().uninstall_all()
    yield
    get_registry().uninstall_all()


def _installed() -> list[str]:
    """Installed seam names IN INSTALL ORDER.

    Read off the registry's own table rather than asked one name at a time
    through `is_installed`, because order is half of what is asserted below and
    a per-name probe cannot see it.
    """
    return list(get_registry()._installed)


@pytest.mark.parametrize("name", list(InterceptorName), ids=lambda n: n.value)
def test_every_member_names_a_seam_that_exists_and_calls_itself_that(name):
    """The claim `InterceptorName`'s docstring makes, enforced.

    Parametrized over the ENUM and not over the table, which is the direction
    that matters: a member with no row is a name a user can select that installs
    nothing at all, and that is precisely what `GRPC`, `WEBSOCKET` and `SSE`
    were. The registry keys its install table on `interceptor.name()`, so a
    value that drifts from it leaves selection and teardown disagreeing about
    which seam is which.
    """
    build = _INTERCEPTORS[name]

    assert build(WardexConfig()).name() == name.value


def test_the_default_selection_is_every_seam_there_is():
    """`interceptors=None` must keep doing exactly what `intercept=True` did."""
    install_configured_interceptors(None, WardexConfig(intercept=True))

    assert _installed() == [name.value for name in _INTERCEPTORS]


def test_a_selection_installs_only_what_it_names():
    """THE regression. Before this, all three went in whatever was asked for."""
    install_configured_interceptors(
        None, WardexConfig(intercept=True, interceptors=(InterceptorName.SSL,))
    )

    assert _installed() == ["ssl"]


def test_an_empty_selection_installs_nothing_even_with_intercept_on():
    """`()` is a choice and `None` is the absence of one — they cannot collapse
    into each other, the same distinction `config.adapters` already draws."""
    install_configured_interceptors(None, WardexConfig(intercept=True, interceptors=()))

    assert _installed() == []


def test_the_table_decides_the_order_and_not_the_caller():
    """SSL patches `ssl.SSLSocket` and the raw socket seam patches
    `socket.socket` underneath it. Which one wraps first is a fact about the
    stack, not a preference, so a reordered tuple must not reorder the install.
    """
    install_configured_interceptors(
        None,
        WardexConfig(intercept=True, interceptors=(InterceptorName.SOCKET, InterceptorName.SSL)),
    )

    assert _installed() == ["ssl", "socket"]


def test_a_selection_without_intercept_installs_nothing():
    """`intercept` is the switch; `interceptors` only refines it.

    `intercept=False` is explicit since the default flipped to True — the
    premise under test is precisely "the switch is off".
    """
    install_configured_interceptors(
        None, WardexConfig(intercept=False, interceptors=(InterceptorName.SSL,))
    )

    assert _installed() == []


def test_a_refinement_of_a_switch_that_is_off_is_announced():
    """Not an error, but silence about it is how a user concludes the selection
    was honoured. The announcement is `init()`'s — a `WardexConfigWarning`,
    unconditional where it used to hide behind `debug` — and the install path
    keeps only the behavior (nothing installs).
    """
    import wardex_sdk
    from wardex_sdk import WardexConfigWarning, _hub

    _hub.reset_for_test()
    try:
        with pytest.warns(WardexConfigWarning, match="no effect without intercept=True"):
            wardex_sdk.init(intercept=False, interceptors=(InterceptorName.SSL,))
        assert _installed() == []
    finally:
        wardex_sdk.close()


def test_a_bare_string_is_refused_where_the_mistake_was_made():
    """`interceptors=("ssl",)` is what a user actually writes. Matched against
    nothing, it would install nothing in silence — indistinguishable from
    `intercept=False`, which is the failure this whole wiring exists to end."""
    with pytest.raises(ValueError, match="InterceptorName"):
        WardexConfig(interceptors=("ssl",))
