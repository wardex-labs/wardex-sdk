"""The closed vocabulary an adapter states at a site, kept beside `AdapterContext`.

Every value here is a literal an adapter writes at a patch site and wardex never infers: where a
site sits in the tree, what it may fall back to, which observer of an event it is, and why an
adapter is or is not running. Its own module because `_context.py` is at its line budget, and
because the four read as one set: none of them carries behaviour, each is a word the adapter must
say out loud. `_context.py` re-exports all four, so `from ._context import Placement` stays the
spelling every adapter uses.
"""

from __future__ import annotations

from enum import Enum


class Placement(Enum):
    """Whether a site may legitimately begin a trace. CLOSED, and never inferred.

    Not a property of a call — a property of a PATCH SITE. The failure it
    catches is "the adapter did not wrap the framework's run entry", which is a
    bug in the adapter's shape rather than a condition of one invocation, and a
    computed flag would let every site paper over it independently. An AST rule
    requires the literal.
    """

    ROOT = "root"  # a framework run entry: nothing above it is expected
    NESTED = "nested"  # always inside something this adapter already entered


class Fallback(Enum):
    """What a NESTED site may fall back to when the live scope did not reach it.

    CLOSED, never inferred, and never a default — for the same reason
    `Placement` is not: a heuristic that turns itself on is one nobody can find
    later. Declaring it at the site is the adapter saying "this call is reached
    through a carrier the framework may not have propagated to, and one live run
    is a defensible guess there".

    It is NOT a way to name a parent. The candidate comes from the registry's
    own table, filtered to units this adapter owns, and only when there is
    EXACTLY one — so no identifier, argument or adapter state can influence
    which unit is picked. The edge it produces is `unit_sole` at 0.5 carrying
    `UNIT_INFERRED_SOLE`, which is the vocabulary's word for "interpreted, not
    read".

    Reachable only where the site would otherwise become a trace root, which is
    structural rather than a rule: a live scope or a host span always wins, so
    the fallback can never displace something wardex actually read.
    """

    NONE = "none"  # orphan instead of guessing; the honest default
    SOLE_LIVE_RUN = "sole_live_run"  # exactly one live SESSION this adapter owns


class Observer(Enum):
    """Which observer of one event this is, when a framework offers two.

    A framework that announces a tool call through a callback AND lets wardex
    wrap the executor produces two observations of one logical call. Ranks
    arbitrate: the executor wrapped the real work, so it wins however late it
    arrives — the callback usually fires first, and first-come would hand every
    span to the observer that did not run the code.
    """

    EXECUTOR = 10
    CALLBACK = 0


class InstallOutcome(Enum):
    """Why an adapter is or is not running, kept distinct from each other.

    `DECLINED` and `UNSUPPORTED` are the pair today's adapters conflate into a
    bare `return`: the framework being absent is nothing to look at, while the
    framework being present with a surface wardex does not recognize is
    something a human should see.
    """

    INSTALLED = "installed"
    DECLINED = "declined"  # the framework is not present; nothing to do
    UNSUPPORTED = "unsupported"  # present, but its surface is not one we know
