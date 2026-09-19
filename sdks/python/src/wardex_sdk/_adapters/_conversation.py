"""The conversation a framework run states, and the host's right to overrule it.

One rule, shared by every adapter whose framework names a conversation of its
own — a group id, a thread id — so that the rule is written once and no two
adapters can drift on it.
"""

from __future__ import annotations

from .._assembly import ConversationContext, ambient_owner, latch_ambient
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

    The host's word is an ambient conversation that the host set. One installed
    by THIS adapter's own unit — a nested run, or a leftover from a run closed
    by `wardex.close()` on a thread whose carrier outlived it — is not what the
    host asked for: the framework's id stays the conversation.

    An id the framework did not state — absent, empty, or not a string or an
    integer — opens no conversation. wardex does not mint one in its place: an
    empty conversation id says "nobody said", and a minted one would say a run
    is a conversation.
    """
    if not isinstance(framework_id, str | int) or framework_id == "":
        return None, None
    if latch_ambient().conversation is not None and ambient_owner() != ctx.name:
        ctx.count(shadowed_counter)
        return None, str(framework_id)
    return ConversationContext(conversation_id=str(framework_id)), None
