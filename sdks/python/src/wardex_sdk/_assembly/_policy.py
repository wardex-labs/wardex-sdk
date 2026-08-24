"""The capture-policy gate — design §4.4, §5.1.

"Do we capture this at all" is one question, and until this module it had three
answers. `ByteSeamInterceptor._should_capture` implemented the §5.1 policy.
`RawSocketInterceptor._should_capture` OVERRODE it with a different predicate
that happened to share the name. `_interceptors/_mcp_stdio.py` asked nothing at
all.

The three had drifted the way copies always drift, and the drift was not
theoretical: the override never learned about `CaptureMode.ALL`, so a user who
asked for everything got everything except plaintext HTTP; and because the
override replaced the local-parent clause instead of composing with it, a
plaintext request issued inside an `execute_tool` span was dropped while the
byte-identical request over TLS was kept. Neither is a decision anybody made.
Both are what "the same rule, written twice" looks like a year later.

So the policy is a pure function of three inputs, and a seam that has its own
opinion about a CONNECTION -- a link-local metadata endpoint it must never
touch, a host the user named by hand -- composes with it through `Prefilter`
rather than replacing it.

`should_capture` cannot raise on its declared inputs: it compares an enum and
reads one bool off a frozen dataclass. That is not incidental. The alternative
to answering is dropping the user's data, and §5.1's rule is that noise is
filterable while lost data is not -- so the function callers fail open around
gives them as little as possible to fail on.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from .._enums import CaptureMode

if TYPE_CHECKING:
    from .._types import SpanContext


class Prefilter(Enum):
    """A byte seam's opinion about the CONNECTION, evaluated before the policy.

    Three values and not two, and `DEFER` is the one that matters: it is what
    makes this a composition instead of an override. A boolean prefilter has no
    way to say "this seam has nothing to add", so the only way to write "I do
    not care about this connection" is to answer the whole question -- which is
    exactly how `RawSocketInterceptor` came to answer a question about
    `capture_mode` that it knew nothing about.
    """

    DENY = "deny"  # this seam must never capture this connection
    ALLOW = "allow"  # the user named this host by hand; a stronger opt-in than the mode
    DEFER = "defer"  # no opinion — the shared policy decides


def capture_mode_of(client: Any) -> CaptureMode:
    """The mode `client` is configured with. Two fallbacks, not one.

    Every caller of `should_capture` holds a client-or-None, and each one used
    to spell the no-client case its own way: the seam had
    `client is None or mode is ALL -> True`, the socket override had no such
    branch at all, and `_mcp_stdio` never looked. Deciding it here means it is
    decided once.

    But "no client" and "a client whose policy I could not read" are different
    questions with opposite safe answers, and collapsing them is how a fallback
    becomes an assertion:

    NO CONFIG -> ALL. There is no wardex configuration in play at all, so there
    is no policy to apply and nothing was asked for; an unconfigured wardex
    filters nothing. This is the old `client is None -> True` branch, unchanged.

    A CONFIG THAT CANNOT STATE A MODE -> AGENT, the field's declared default.
    Here somebody DID configure wardex and the value is unreadable -- a str
    where the enum was expected, a None, a fake in a test. Answering `ALL`
    would take a user who asked for filtering and hand them the pre-Phase-4
    firehose, shipping request and response BODIES they never asked to export,
    silently. §5.1's "noise is filterable, lost data is not" governs the
    direction wardex fails when wardex is at fault; it is not a licence to
    widen a policy the user set because the SDK could not parse it. So the
    unreadable case narrows to the documented default instead of widening past
    it, which is also what the code this replaced did: `mode is CaptureMode.ALL`
    is False for a str, so the AGENT clauses ran.

    Duck-typed deliberately. `_assembly/` is permitted to import `_client`, but
    nothing here needs the type, and every seam is exercised with a fake client
    somewhere in the suite. Neither branch is an error to raise into the host
    (I6).
    """
    config = getattr(client, "config", None)
    if config is None:
        return CaptureMode.ALL
    mode = getattr(config, "capture_mode", None)
    return mode if isinstance(mode, CaptureMode) else CaptureMode.AGENT


def should_capture(
    mode: CaptureMode,
    *,
    parent: SpanContext | None,
    agent_semantic: bool,
    degraded: bool = False,
    parent_closed: bool = False,
) -> bool:
    """Design §5.1, whole: the SDK's only answer to "capture this?".

    AGENT (the default): agent-semantic traffic always; anything else only when
    a LOCAL wardex span was ambient at the moment the work was ISSUED.

    A remote-only parent does not open the gate. Service meshes attach
    `traceparent` to every request in the fleet, so treating a joined trace as
    evidence of agent activity would resurrect the firehose this mode exists to
    suppress -- `is_remote` is the whole difference between "something in this
    process is doing agent work" and "somebody upstream had a trace id".

    `agent_semantic` is the caller's claim about the traffic, not a property
    this function can check: the byte seam earns it from a parsed LLM response,
    MCP stdio has it by construction (a JSON-RPC tool call over a subprocess
    pipe is agent traffic or it is nothing). Passing a constant `True` is
    therefore a statement a site makes about itself, and it is reviewable
    precisely because it is written at the call.

    `degraded` is "wardex is the reason one of this gate's INPUTS is
    missing". Three producers say it, and they are one fact seen at three
    moments: `assembly._parentage.in_degraded_run()` — a span wardex FAILED
    to open is what should have been ambient here; the byte seam's
    `_Txn.parent_evicted`, where a parent WAS latched and wardex's own
    per-connection bound dropped the record before the response claimed it;
    and the seam's `unparsed` — a finalization whose LLM-semantic parse
    wardex itself skipped (backlog eviction, shutdown budget, a raising
    parser), so the `agent_semantic` claim cannot honestly be answered.
    The premise "no semantic claim means not agent traffic" is exactly as
    wrong for a parse wardex skipped as "no parent means not agent work" is
    for a parent wardex lost — the same argument, one input over. It is a
    DECLARED input rather than a `ContextVar` read hidden inside this
    function, so the policy stays what its docstring says it is: a pure function
    of its arguments, testable by them, unable to raise on them.

    What it buys is the difference between an agent run wardex could not follow
    and one that never happened. The gate's whole premise is that an absent
    local parent means "this traffic is not agent work"; when wardex is what
    lost the parent, that inference is simply wrong, and acting on it turns one
    bug at the top of a run into total silence underneath it. §5.1's rule is
    that noise is filterable and lost data is not, and this is the one place
    that rule was being applied backwards — against the user, on wardex's own
    fault. What comes through is not passed off as ordinary: see
    `resolve_observed`.

    `parent_closed` is `assembly._units.parent_is_closed_unit(parent)` — "the
    local span this was latched off belongs to a unit that had ALREADY CLOSED".
    A DECLARED input for the same reason `degraded` is, plus a stronger one: the
    fact is not knowable here at all. It has to be read on the task that ISSUED
    the work, at the instant it was issued, and this function runs on the
    response side of a seam that shares neither.

    It exists because AGENT mode's whole premise is that a local ambient span
    means "something in this process is doing agent work RIGHT NOW", and a
    leftover activation fork breaks that premise without breaking the type: the
    span is local, is not remote, and is finished. Measured on LangGraph, a
    generator-scoped session whose fork outlived it kept the gate open for every
    later request on that carrier — traffic the user's AGENT mode had asked to
    drop, captured and then hung off an already-shipped span. Closing the gate
    here loses nothing the mode wanted: it restores the answer the mode would
    have given had the fork come down. Agent-semantic traffic is untouched — it
    returns True two lines earlier, on a claim the site makes about the BYTES,
    which a dead parent says nothing about.
    """
    if mode is CaptureMode.ALL:
        return True
    if agent_semantic:
        return True
    if parent is not None and not parent.is_remote and not parent_closed:
        return True
    return degraded
