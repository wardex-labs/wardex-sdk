"""The conversation a framework run states, and the host's right to overrule it.

One rule, shared by every adapter whose framework names a conversation of its
own — a group id, a thread id — so that the rule is written once and no two
adapters can drift on it.
"""

from __future__ import annotations

from .._assembly import ConversationContext, ambient_stated_conversation, latch_ambient
from ._context import AdapterContext


def framework_conversation(
    ctx: AdapterContext, framework_id: object, *, shadowed_counter: str
) -> tuple[ConversationContext | None, str | None]:
    """`(the conversation to open the run with, the framework id the host shadowed)`.

    HOST WINS. A run opened inside the host's own `wardex.conversation(...)`
    keeps that id: one trace, one conversation, and the host's word is the one
    its backend already groups by. The framework's id is then handed back as
    the second value, counted, for the adapter to record as an attribute of its
    own instead of replacing the ambient id on every span underneath. With
    nothing ambient the framework's id IS the conversation, handed to the
    registry at the open so that children inherit it.

    The host's word is any ambient conversation this adapter did not state
    itself. Asking who OWNS the ambient unit is not enough: a run opened inside
    the host's block is this adapter's unit carrying the host's id, and a run
    nested under it — a subgraph called from a node — has to yield to that id
    exactly as the outer run did. What this adapter DID state is a nested run's
    parent, or a leftover from a run closed by `wardex.close()` on a thread
    whose carrier outlived it; neither is what the host asked for, and the
    framework's id stays the conversation.

    An id the framework did not state — `None` or `""` — opens no conversation.
    wardex does not mint one in its place: an empty conversation id says
    "nobody said", and a minted one would say a run is a conversation. Anything
    else is the host's word and is carried as its text, the way
    `ConversationContext` itself takes a `uuid.UUID` or an integer key.
    """
    if framework_id is None or framework_id == "":
        return None, None
    stated = str(framework_id)
    ambient = latch_ambient().conversation
    if ambient is not None and not ambient_stated_conversation(ctx.name, ambient):
        ctx.count(shadowed_counter)
        return None, stated
    return ConversationContext(conversation_id=stated), None
