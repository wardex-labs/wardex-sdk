"""Drop ledger census: every place the SDK says it lost something, accounted for.

A loss the SDK knows about leaves one of a few signals in the source: a
`counters.bump`, an adapter's `count`, a `report_once` line, a `_report_lost`,
or a call into the drop ledger (`record`, `record_items`, `lose`). Only the
ledger reaches the receiver. So every one of those sites is recorded below with
what happens to the loss it signals:

  * `Ledger(reasons)` -- the loss is recorded in the drop ledger under one of
    these reasons. Checked: each reason is spelled `DropReason.<NAME>` in the
    site's own function, or in the module `at` names when the record lives
    elsewhere (the client records an eviction where it happens; the line about
    it is said later, in `_drops.py`).
  * `Marker(member)` -- the loss is carried by a limitation marker on a span
    that still ships. Checked the same way, against `Limitation.<NAME>`.
  * `NotADrop(why)` -- the signal is not a loss: a tally of something
    observed, a second count of a loss counted elsewhere, a deliberate policy.
  * `Structural(why)` -- the signal is plumbing whose callers are the entries:
    the counter behind an adapter's `count()`, `lose()`'s own two calls, a
    line said for a swallow that `guard()` has already recorded.
  * `Pending(area)` -- a loss not yet in the ledger, left to the change that
    owns that area. This list only shrinks: each change moves its sites to
    `Ledger` or `Marker`. The area is said in plain words.

A site is `(module, signal kind, key text)`, never a line number, so code moving
around does not break the census; the key is the call's key argument as source
text, f-strings included. The table and the source must match EXACTLY, in both
directions: a new signal anywhere fails until it is classified, and an entry
whose site is gone fails until it is removed.

Everything here reads source text; nothing imports the native extension.
"""

from __future__ import annotations

import ast
import pathlib
import re
import shutil
from dataclasses import dataclass

import pytest

from wardex_sdk._assembly import Limitation
from wardex_sdk._assembly._ledger import DropReason

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "wardex_sdk"


@dataclass(frozen=True)
class Ledger:
    reasons: tuple[str, ...]
    at: str | None = None


@dataclass(frozen=True)
class Marker:
    member: str
    at: str | None = None


@dataclass(frozen=True)
class NotADrop:
    why: str


@dataclass(frozen=True)
class Structural:
    why: str


@dataclass(frozen=True)
class Pending:
    area: str


def L(*reasons: str, at: str | None = None) -> Ledger:  # noqa: N802 — reads as a table column
    return Ledger(reasons, at)


# -- the plain-words areas a pending site belongs to ---------------------------

_BRIDGE = Pending("OTel bridge loss")
_CLAUDE = Pending("Claude Agent SDK adapter loss")
_CODEX = Pending("Codex CLI adapter loss")
_OPENAI = Pending("OpenAI Agents adapter loss")
_LANGGRAPH = Pending("LangGraph adapter loss")
_ADAPTER_LIFECYCLE = Pending("adapter install and lifecycle loss")
_SEAM = Pending("byte-seam capture loss")
_MODE = Pending("default-mode refusal")
_POLICY = Pending("policy exclusion, to be counted as not a loss")
_MCP = Pending("MCP stdio capture loss")
_MCP_PENDING_TABLE = Pending(
    "MCP stdio requests evicted from the bounded pending table, meant for table_full:request"
)
_MCP_LINE_CAP = Pending(
    "MCP stdio lines skipped unparsed past the size cap, meant for body_cap:message"
)
_USAGE = Pending("usage totals withheld")
_UNITS = Pending("assembly-core unit tables and patches")
_SIGNAL = Pending("signal-path shutdown tail")
_REQUEST_CAP = Pending("OTLP request-cap drop")
_NATIVE = Pending("native extension missing: no batch, the drop report included, can be encoded")

_GUARDED = Structural("the line said for a swallow guard() has already recorded")
_VERSION_REFUSED = NotADrop(
    "a peer's version that does not read as a revision is left off on purpose; the call ships"
)
_TOOL_CEDED = NotADrop(
    "the call still ships as an MCP tools/call span with its tool; only the execute_tool name "
    "is ceded to the installed adapter"
)

# -- the census ------------------------------------------------------------------

_CENSUS: dict[tuple[str, str, str], object] = {
    # -- adapters: registry and status
    ("_adapters/__init__.py", "bump", "f'adapters.{name}.distribution_absent'"): NotADrop(
        "the framework is not installed, so there is nothing of it to capture"
    ),
    ("_adapters/__init__.py", "bump", "f'adapters.{name}.distribution_absent_explicit'"): NotADrop(
        "the adapter installs anyway; only the shadow check was skipped"
    ),
    ("_adapters/__init__.py", "bump", "f'adapters.{name}.shadowed'"): _ADAPTER_LIFECYCLE,
    ("_adapters/__init__.py", "bump", "f'adapters.{name}.{key}'"): L(
        "INSTRUMENTATION_UNAVAILABLE", at="_adapters/__init__.py"
    ),
    ("_adapters/__init__.py", "record", "DropReason.INSTRUMENTATION_UNAVAILABLE"): L(
        "INSTRUMENTATION_UNAVAILABLE"
    ),
    (
        "_adapters/__init__.py",
        "report_once",
        "f'adapters.{name}.distribution_absent_explicit'",
    ): NotADrop("the adapter installs anyway; only the shadow check was skipped"),
    ("_adapters/__init__.py", "report_once", "f'adapters.{name}.shadowed'"): _ADAPTER_LIFECYCLE,
    ("_adapters/__init__.py", "report_once", "f'adapters.{name}.{key}'"): L(
        "INSTRUMENTATION_UNAVAILABLE", at="_adapters/__init__.py"
    ),
    # -- adapters: Claude Agent SDK
    (
        "_adapters/_anthropic_agent_sdk.py",
        "bump",
        "'adapters.anthropic.otel_bridge.fork_torn_down'",
    ): NotADrop("a fork child lets go of the parent's bridge; the parent ships its data"),
    (
        "_adapters/_anthropic_agent_sdk.py",
        "bump",
        "'adapters.anthropic.otel_bridge.readback_failed'",
    ): NotADrop("the bridge carries on as an unconfirmed binding; nothing is dropped"),
    (
        "_adapters/_anthropic_agent_sdk.py",
        "bump",
        "'adapters.anthropic.tool_input_unserializable'",
    ): _CLAUDE,
    (
        "_adapters/_anthropic_agent_sdk.py",
        "report_once",
        "'adapters.anthropic_agent_sdk.no_context'",
    ): _CLAUDE,
    (
        "_adapters/_anthropic_agent_sdk.py",
        "report_once",
        "'adapters.anthropic_agent_sdk.otel_bridge.no_hijack'",
    ): NotADrop("the host's own telemetry settings win, by design, and the line says so"),
    (
        "_adapters/_anthropic_agent_sdk.py",
        "report_once",
        "'adapters.anthropic_agent_sdk.otel_bridge.receiver_failed'",
    ): _BRIDGE,
    ("_adapters/_anthropic_names.py", "bump", "'adapters.anthropic.hook_tool_name_ambiguous'"): (
        Marker("TOOL_NAME_COLLISION", at="_adapters/_anthropic_agent_sdk.py")
    ),
    ("_adapters/_anthropic_names.py", "bump", "'adapters.anthropic.server_table_full'"): NotADrop(
        "an old handle stops being re-read; no span is dropped"
    ),
    ("_adapters/_assembler.py", "bump", "'adapters.anthropic.otel_bridge.llm_join_tolerant'"): (
        Marker("OTEL_BRIDGE_JOIN_TOLERANT")
    ),
    ("_adapters/_assembler.py", "bump", "'adapters.assembler.chat_line_after_evict'"): _CLAUDE,
    ("_adapters/_assembler.py", "bump", "'adapters.assembler.close_all_sessions_reentrant'"): (
        _SIGNAL
    ),
    ("_adapters/_assembler.py", "bump", "'adapters.assembler.hook_session_conflict'"): NotADrop(
        "two sessions claimed one hook; it is attributed to one and nothing is dropped"
    ),
    ("_adapters/_assembler.py", "bump", "'adapters.assembler.hook_session_inferred_sole'"): (
        NotADrop("the hook was attributed to the only live session; nothing is dropped")
    ),
    ("_adapters/_assembler.py", "bump", "'adapters.assembler.hook_session_unresolved'"): _CLAUDE,
    ("_adapters/_assembler.py", "bump", "'adapters.assembler.otel_bridge_pending_overflow'"): (
        _BRIDGE
    ),
    ("_adapters/_assembler.py", "bump", "'adapters.assembler.otel_bridge_retired'"): _BRIDGE,
    ("_adapters/_assembler.py", "bump", "'adapters.assembler.otel_bridge_unconfirmed_no_data'"): (
        _BRIDGE
    ),
    ("_adapters/_assembler.py", "bump", "'adapters.assembler.prompt_overwritten'"): _CLAUDE,
    ("_adapters/_assembler.py", "bump", "'adapters.assembler.session_key_recycled'"): Marker(
        "CORRELATION_CONFLICT"
    ),
    ("_adapters/_assembler.py", "bump", "'adapters.assembler.session_unit_evicted'"): _CLAUDE,
    ("_adapters/_assembler.py", "bump", "'adapters.assembler.stream_usage_overflowed'"): _USAGE,
    ("_adapters/_assembler.py", "bump", "'adapters.assembler.stream_usage_totals_unpaired'"): (
        _USAGE
    ),
    ("_adapters/_assembler.py", "bump", "'adapters.assembler.subagent_stop_after_evict'"): (
        Marker("SESSION_ENTRY_TABLE_FULL", at="_adapters/_assembler.py")
    ),
    ("_adapters/_assembler.py", "bump", "'adapters.assembler.tool_claim_lost'"): NotADrop(
        "another observer of the same tool call emits it"
    ),
    ("_adapters/_assembler.py", "bump", "'adapters.assembler.tool_close_without_open'"): _CLAUDE,
    ("_adapters/_assembler.py", "bump", "'adapters.assembler.tool_completion_after_evict'"): (
        Marker("SESSION_ENTRY_TABLE_FULL")
    ),
    (
        "_adapters/_assembler.py",
        "bump",
        "'adapters.assembler.tool_completion_after_evict_duplicate'",
    ): NotADrop("a repeat of a completion already handled"),
    ("_adapters/_assembler.py", "bump", "'adapters.assembler.tool_name_unattributable'"): (
        Marker("TOOL_NAME_COLLISION", at="_adapters/_anthropic_agent_sdk.py")
    ),
    ("_adapters/_assembler.py", "bump", "f'adapters.assembler.{where}_table_full'"): _CLAUDE,
    ("_adapters/_hook_reach.py", "bump", "'adapters.anthropic.handshake_unreadable'"): _CLAUDE,
    ("_adapters/_hook_reach.py", "bump", "'adapters.anthropic.hooks_absent'"): _CLAUDE,
    ("_adapters/_hook_reach.py", "bump", "'adapters.anthropic.hooks_absent_unmarked'"): _CLAUDE,
    ("_adapters/_hook_reach.py", "report_once", "'adapters.anthropic_agent_sdk.hooks_absent'"): (
        _CLAUDE
    ),
    ("_adapters/_session_end.py", "bump", "'adapters.anthropic.otel_bridge.session_ended_late'"): (
        _BRIDGE
    ),
    ("_adapters/_session_outcome.py", "bump", "'adapters.assembler.stream_line_unrecognized'"): (
        _CLAUDE
    ),
    # -- adapters: Codex CLI
    ("_adapters/_codex_exec.py", "bump", "'adapters.codex_exec.argv_iterator_unread'"): _CODEX,
    ("_adapters/_codex_exec.py", "bump", "'adapters.codex_exec.process_not_weakrefable'"): _CODEX,
    ("_adapters/_codex_exec.py", "count", "'otel_bridge.fork_torn_down'"): NotADrop(
        "a fork child lets go of the parent's bridge; the parent ships its data"
    ),
    ("_adapters/_codex_exec.py", "count", "'otel_bridge.injected'"): NotADrop(
        "a tally of the bridge being switched on"
    ),
    ("_adapters/_codex_exec.py", "count", "'otel_bridge.stood_down'"): Marker(
        "SUBPROCESS_MODEL_CALLS_UNOBSERVED", at="_adapters/_codex_exec.py"
    ),
    (
        "_adapters/_codex_exec.py",
        "report_once",
        "'adapters.codex_exec.otel_bridge.receiver_failed'",
    ): _BRIDGE,
    ("_adapters/_codex_exec.py", "report_once", "'adapters.codex_exec.otel_bridge.stood_down'"): (
        Marker("SUBPROCESS_MODEL_CALLS_UNOBSERVED", at="_adapters/_codex_exec.py")
    ),
    ("_adapters/_codex_exec.py", "report_once", "'adapters.codex_exec.stream_only'"): Marker(
        "SUBPROCESS_MODEL_CALLS_UNOBSERVED"
    ),
    ("_adapters/_codex_read.py", "bump", "'adapters.codex_exec.argv_unreadable'"): _CODEX,
    ("_adapters/_codex_read.py", "bump", "'adapters.codex_exec.item_unparsed'"): _CODEX,
    ("_adapters/_codex_read.py", "bump", "'adapters.codex_exec.program_unreadable'"): _CODEX,
    ("_adapters/_codex_read.py", "bump", "'adapters.codex_exec.user_config_unread'"): _CODEX,
    # -- adapters: shared context
    ("_adapters/_context.py", "bump", "f'adapters.{self.name}.{where}'"): Structural(
        "the counter behind every adapter's count(); each count() call is its own entry"
    ),
    ("_adapters/_context.py", "count", "'link_target_unresolved'"): _ADAPTER_LIFECYCLE,
    ("_adapters/_context.py", "count", "f'active.{site}'"): NotADrop(
        "an entry point that ran: evidence of coverage, not of a loss"
    ),
    ("_adapters/_context.py", "report_once", "f'{self.name}.{where}'"): _GUARDED,
    ("_adapters/_conversation.py", "count", "shadowed_counter"): NotADrop(
        "the host's conversation id wins over the framework's, which stays on the span"
    ),
    ("_adapters/_registry.py", "bump", "f'adapters.{name}.{where}_interrupted'"): (
        _ADAPTER_LIFECYCLE
    ),
    # -- adapters: LangGraph
    ("_adapters/_langgraph.py", "count", "'unsupported_remote_surface'"): _LANGGRAPH,
    ("_adapters/_langgraph.py", "count", "'unsupported_surface'"): _LANGGRAPH,
    ("_adapters/_langgraph.py", "count", "'unsupported_tool_surface'"): _LANGGRAPH,
    ("_adapters/_langgraph.py", "count", "finalized"): NotADrop(
        "a tally of a stream finalizing; nothing is dropped"
    ),
    ("_adapters/_langgraph.py", "count", "off_carrier"): NotADrop(
        "a tally of where a stream finalized; nothing is dropped"
    ),
    (
        "_adapters/_langgraph.py",
        "report_once",
        "'adapters.langgraph.unsupported_remote_surface'",
    ): _LANGGRAPH,
    ("_adapters/_langgraph.py", "report_once", "'adapters.langgraph.unsupported_surface'"): (
        _LANGGRAPH
    ),
    ("_adapters/_langgraph.py", "report_once", "'adapters.langgraph.unsupported_tool_surface'"): (
        _LANGGRAPH
    ),
    ("_adapters/_langgraph_links.py", "count", "'join_ambiguous'"): Marker("LINK_AMBIGUOUS"),
    ("_adapters/_langgraph_links.py", "count", "'unsupported_config_merge'"): _LANGGRAPH,
    (
        "_adapters/_langgraph_links.py",
        "report_once",
        "'adapters.langgraph.unsupported_config_merge'",
    ): _LANGGRAPH,
    # -- adapters: OpenAI Agents
    ("_adapters/_openai_agents.py", "count", "'agent_end_unmatched'"): NotADrop(
        "an end with no start; the loss, if any, was counted at the start"
    ),
    ("_adapters/_openai_agents.py", "count", "'force_flush'"): NotADrop(
        "a tally of a framework call"
    ),
    ("_adapters/_openai_agents.py", "count", "'guardrail_end_unmatched'"): NotADrop(
        "an end with no start; the loss, if any, was counted at the start"
    ),
    ("_adapters/_openai_agents.py", "count", "'guardrail_failed'"): NotADrop(
        "a tally of the run's own outcome, which its span records"
    ),
    ("_adapters/_openai_agents.py", "count", "'guardrail_interrupted'"): NotADrop(
        "a tally of the run's own outcome, which its span records"
    ),
    ("_adapters/_openai_agents.py", "count", "'handoff_receiver_never_started'"): _OPENAI,
    # The handoff span closes as an error and its causal edge is lost, with no marker on any
    # span to say so: a loss, left with this adapter's others.
    ("_adapters/_openai_agents.py", "count", "'handoff_unresolved'"): _OPENAI,
    ("_adapters/_openai_agents.py", "count", "'handoff_without_agent'"): _OPENAI,
    ("_adapters/_openai_agents.py", "count", "'pin_ok' if ok else 'pin_refused'"): Marker(
        "CORRELATION_CONFLICT", at="_adapters/_openai_agents_kinds.py"
    ),
    ("_adapters/_openai_agents.py", "count", "'pin_stranded'"): NotADrop(
        "a pin left behind is released; nothing is dropped"
    ),
    ("_adapters/_openai_agents.py", "count", "'processor_removed'"): _OPENAI,
    ("_adapters/_openai_agents.py", "count", "'processor_removed_after_runs'"): _OPENAI,
    ("_adapters/_openai_agents.py", "count", "'processors_changed_under_us'"): NotADrop(
        "the uninstall found the framework's list changed; nothing is dropped"
    ),
    ("_adapters/_openai_agents.py", "count", "'processors_read_failed'"): NotADrop(
        "the install goes ahead; only a check of the framework's list was skipped"
    ),
    ("_adapters/_openai_agents.py", "count", "'run_root_reattached'"): NotADrop(
        "the run's root is kept and re-attached; nothing is dropped"
    ),
    ("_adapters/_openai_agents.py", "count", "'run_root_reattached_closed'"): NotADrop(
        "the run's root is kept and re-attached; nothing is dropped"
    ),
    ("_adapters/_openai_agents.py", "count", "'shutdown'"): NotADrop("a tally of a framework call"),
    (
        "_adapters/_openai_agents.py",
        "count",
        "'tool_call_id_ambiguous' if len(ids) > 1 else 'tool_call_id_unmatched'",
    ): Marker("TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS"),
    ("_adapters/_openai_agents.py", "count", "'tool_call_id_tracing_disabled'"): _OPENAI,
    ("_adapters/_openai_agents.py", "count", "'tool_end_unmatched'"): NotADrop(
        "an end with no start; the loss, if any, was counted at the start"
    ),
    # A run that began before wardex.init() ends here with its root never recorded, and
    # nothing counted it at a start wardex did not see.
    ("_adapters/_openai_agents.py", "count", "'trace_end_unmatched'"): _OPENAI,
    ("_adapters/_openai_agents.py", "count", "'trace_lookup_miss'"): _OPENAI,
    ("_adapters/_openai_agents.py", "count", "'trace_start_twice'"): NotADrop(
        "a repeated start is ignored; the first one stands"
    ),
    ("_adapters/_openai_agents.py", "count", "'tracing_disabled_at_install'"): _OPENAI,
    ("_adapters/_openai_agents.py", "count", "'tracing_state_unknown'"): NotADrop(
        "the install goes ahead; only a check of the framework's state was skipped"
    ),
    ("_adapters/_openai_agents.py", "count", "'uninstall_processor_left_inert'"): NotADrop(
        "the uninstall leaves an inert processor behind; nothing is dropped"
    ),
    ("_adapters/_openai_agents.py", "count", "'unpin'"): NotADrop("a tally of unpinning"),
    ("_adapters/_openai_agents.py", "count", "'unsupported_surface'"): _OPENAI,
    ("_adapters/_openai_agents.py", "count", "f'{site}_missed'"): _OPENAI,
    ("_adapters/_openai_agents.py", "report_once", "'adapters.openai_agents.pin_refused'"): (
        Marker("CORRELATION_CONFLICT", at="_adapters/_openai_agents_kinds.py")
    ),
    ("_adapters/_openai_agents.py", "report_once", "'adapters.openai_agents.processor_removed'"): (
        _OPENAI
    ),
    (
        "_adapters/_openai_agents.py",
        "report_once",
        "'adapters.openai_agents.processors_read_failed'",
    ): NotADrop("the install goes ahead; only a check of the framework's list was skipped"),
    (
        "_adapters/_openai_agents.py",
        "report_once",
        "'adapters.openai_agents.run_root_reattached'",
    ): NotADrop("the run's root is kept and re-attached; nothing is dropped"),
    (
        "_adapters/_openai_agents.py",
        "report_once",
        "'adapters.openai_agents.tracing_state_unknown'",
    ): NotADrop("the install goes ahead; only a check of the framework's state was skipped"),
    (
        "_adapters/_openai_agents.py",
        "report_once",
        "'adapters.openai_agents.uninstall_processor_left_inert'",
    ): NotADrop("the uninstall leaves an inert processor behind; nothing is dropped"),
    (
        "_adapters/_openai_agents.py",
        "report_once",
        "'adapters.openai_agents.unsupported_surface'",
    ): _OPENAI,
    ("_adapters/_openai_agents.py", "report_once", "f'adapters.openai_agents.{where}'"): _GUARDED,
    (
        "_adapters/_openai_agents_entry.py",
        "count",
        "'conversation_id_shadowed_by_group_id'",
    ): NotADrop("the framework's group id wins by rule, and stays on the span"),
    ("_adapters/_openai_agents_entry.py", "count", "'root_failed_at_call_exit'"): NotADrop(
        "the run's own failure, which its root span records"
    ),
    ("_adapters/_openai_agents_entry.py", "count", "'run_raised_after_root_ok'"): _OPENAI,
    ("_adapters/_openai_agents_entry.py", "count", "'streamed_outcome_unreadable'"): _OPENAI,
    ("_adapters/_openai_agents_entry.py", "count", "'tracing_disabled_run'"): _OPENAI,
    ("_adapters/_openai_agents_entry.py", "count", "'unsupported_entry_surface'"): _OPENAI,
    ("_adapters/_openai_agents_entry.py", "count", "counter"): _OPENAI,
    (
        "_adapters/_openai_agents_entry.py",
        "report_once",
        "'adapters.openai_agents.run_raised_after_root_ok'",
    ): _OPENAI,
    (
        "_adapters/_openai_agents_entry.py",
        "report_once",
        "'adapters.openai_agents.unsupported_entry_surface'",
    ): _OPENAI,
    (
        "_adapters/_openai_agents_entry.py",
        "report_once",
        "f'adapters.openai_agents.tracing_off.{reason}'",
    ): _OPENAI,
    ("_adapters/_openai_agents_kinds.py", "count", "'error_message_unmapped'"): NotADrop(
        "the error ships under the generic error type"
    ),
    ("_adapters/_openai_agents_kinds.py", "count", "'model_output_withheld'"): _OPENAI,
    ("_adapters/_openai_agents_kinds.py", "count", "'response_id_unavailable'"): Marker(
        "TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS", at="_adapters/_openai_agents.py"
    ),
    ("_adapters/_openai_agents_kinds.py", "count", "'response_without_agent'"): Marker(
        "TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS", at="_adapters/_openai_agents.py"
    ),
    ("_adapters/_openai_agents_kinds.py", "count", "'started_at_unparsed'"): NotADrop(
        "the span ships with the time wardex observed instead"
    ),
    ("_adapters/_openai_agents_kinds.py", "count", "'step_end_unmatched'"): _OPENAI,
    ("_adapters/_openai_agents_kinds.py", "count", "reason"): _OPENAI,
    ("_adapters/_openai_agents_kinds.py", "report_once", "'adapters.openai_agents.unmapped'"): (
        NotADrop("the error ships under the generic error type")
    ),
    ("_adapters/_openai_agents_kinds.py", "report_once", "f'adapters.openai_agents.{reason}'"): (
        _OPENAI
    ),
    # -- adapters: the OTel bridge
    ("_adapters/_otel_merge.py", "bump", "'adapters.anthropic.otel_bridge.attr_dropped'"): (
        NotADrop("an attribute outside the bridge's allow-list, left out by policy")
    ),
    ("_adapters/_otel_merge.py", "bump", "'adapters.anthropic.otel_bridge.llm_join_ambiguous'"): (
        Marker("CORRELATION_CONFLICT", at="_adapters/_assembler.py")
    ),
    ("_adapters/_otel_merge.py", "bump", "'adapters.anthropic.otel_bridge.span_unrecognized'"): (
        _BRIDGE
    ),
    ("_adapters/_otel_merge.py", "bump", "'adapters.anthropic.otel_bridge.tool_without_id'"): (
        _BRIDGE
    ),
    ("_adapters/_otel_merge.py", "bump", "'adapters.anthropic.otel_bridge.usage_demoted'"): (
        NotADrop("the value is kept under a lesser name")
    ),
    (
        "_adapters/_otel_receiver.py",
        "bump",
        "f'adapters.{receiver._owner}.otel_bridge.handler_error'",
    ): _BRIDGE,
    (
        "_adapters/_otel_receiver.py",
        "bump",
        "f'adapters.{self._owner}.otel_bridge.body_rejected'",
    ): (_BRIDGE),
    ("_adapters/_otel_receiver.py", "bump", "f'adapters.{self._owner}.otel_bridge.decode_error'"): (
        _BRIDGE
    ),
    ("_adapters/_otel_receiver.py", "bump", "f'adapters.{self._owner}.otel_bridge.gzip_error'"): (
        _BRIDGE
    ),
    (
        "_adapters/_otel_receiver.py",
        "bump",
        "f'adapters.{self._owner}.otel_bridge.reservation_evicted'",
    ): _BRIDGE,
    (
        "_adapters/_otel_receiver.py",
        "bump",
        "f'adapters.{self._owner}.otel_bridge.span_after_close'",
    ): _BRIDGE,
    ("_adapters/_otel_receiver.py", "bump", "f'adapters.{self._owner}.otel_bridge.span_dropped'"): (
        _BRIDGE
    ),
    (
        "_adapters/_otel_receiver.py",
        "bump",
        "f'adapters.{self._owner}.otel_bridge.span_unroutable'",
    ): _BRIDGE,
    ("_adapters/_otel_receiver.py", "bump", "f'adapters.{self._owner}.otel_bridge.tick_failed'"): (
        _BRIDGE
    ),
    (
        "_adapters/_otel_receiver.py",
        "bump",
        "f'adapters.{self._owner}.otel_bridge.token_rejected'",
    ): NotADrop("a POST without this run's token is not this run's data, and is refused"),
    ("_adapters/_otel_receiver.py", "bump", "f'adapters.{self._owner}.otel_bridge.undecodable'"): (
        _BRIDGE
    ),
    (
        "_adapters/_otel_receiver.py",
        "report_once",
        "f'adapters.{label}.otel_bridge.undecodable'",
    ): _BRIDGE,
    # -- assembly core
    ("_assembly/_builder.py", "bump", "'assembly.builder.gen_ai_usage_not_inclusive'"): NotADrop(
        "an observation about upstream usage numbers; the span ships whole"
    ),
    ("_assembly/_diag.py", "bump", "'ffi.panic_converted'"): NotADrop(
        "a second tally of a swallow the guard has already recorded"
    ),
    ("_assembly/_diag.py", "bump", "LOG_FAILED"): L("DIAGNOSTIC_UNWRITABLE"),
    ("_assembly/_diag.py", "bump", "self._where"): L("INTERNAL_SDK_ERROR"),
    ("_assembly/_diag.py", "record", "DropReason.DIAGNOSTIC_UNWRITABLE"): L(
        "DIAGNOSTIC_UNWRITABLE"
    ),
    ("_assembly/_diag.py", "record", "DropReason.INTERNAL_SDK_ERROR"): L("INTERNAL_SDK_ERROR"),
    ("_assembly/_drops.py", "bump", "'_runtime.open_spans_full'"): _SIGNAL,
    ("_assembly/_drops.py", "bump", "'_runtime.open_spans_unshipped'"): L("SHUTDOWN_TAIL"),
    ("_assembly/_drops.py", "bump", "'_runtime.signal_flush_raised'"): _SIGNAL,
    ("_assembly/_drops.py", "bump", "'client.buffer.evicted'"): L(
        "QUEUE_OVERFLOW", at="_client.py"
    ),
    ("_assembly/_drops.py", "bump", "'interceptors.mcp_stdio.parser_disabled'"): L(
        "PARSER_DISABLED"
    ),
    # Every latch a native parser sets on a connection is counted here, the
    # HTTP/2 caps on the HPACK table and on header blocks awaiting CONTINUATION
    # included: the seam asks the tracker for its latch reason, whatever it is.
    ("_assembly/_drops.py", "bump", "'interceptors.seam.parser_disabled'"): L("PARSER_DISABLED"),
    ("_assembly/_drops.py", "bump", "'worker.drain_raised'"): L("INTERNAL_SDK_ERROR"),
    ("_assembly/_drops.py", "bump", "counter"): L("SEND_ERROR"),
    ("_assembly/_drops.py", "bump", "f'client.drain.snapshot_dropped.{cause}'"): L(
        "BEFORE_SEND_ERROR", "SEND_ERROR", at="_assembly/_drops.py"
    ),
    ("_assembly/_drops.py", "bump", "f'client.drain.span_dropped.{cause}'"): L(
        "BEFORE_SEND_ERROR", "SEND_ERROR", at="_assembly/_drops.py"
    ),
    ("_assembly/_drops.py", "record", "DropReason.INTERNAL_SDK_ERROR"): L("INTERNAL_SDK_ERROR"),
    ("_assembly/_drops.py", "record", "DropReason.PARSER_DISABLED"): L("PARSER_DISABLED"),
    ("_assembly/_drops.py", "record", "DropReason.SEND_ERROR"): L("SEND_ERROR"),
    ("_assembly/_drops.py", "record", "DropReason.SHUTDOWN_TAIL"): L("SHUTDOWN_TAIL"),
    ("_assembly/_drops.py", "record", "_DRAIN_DROP_REASONS.get(cause, DropReason.SEND_ERROR)"): L(
        "BEFORE_SEND_ERROR", "SEND_ERROR", at="_assembly/_drops.py"
    ),
    ("_assembly/_drops.py", "record", "reason"): Structural(
        "lose()'s own record; each lose() call is its own entry"
    ),
    ("_assembly/_drops.py", "report_once", "'_runtime.open_spans_full'"): _SIGNAL,
    ("_assembly/_drops.py", "report_once", "'_runtime.open_spans_unshipped'"): L("SHUTDOWN_TAIL"),
    ("_assembly/_drops.py", "report_once", "'_runtime.signal_flush_raised'"): _SIGNAL,
    ("_assembly/_drops.py", "report_once", "'client.buffer.evicted'"): L(
        "QUEUE_OVERFLOW", at="_client.py"
    ),
    ("_assembly/_drops.py", "report_once", "'interceptors.mcp_stdio.parser_disabled'"): L(
        "PARSER_DISABLED"
    ),
    ("_assembly/_drops.py", "report_once", "'interceptors.seam.parser_disabled'"): L(
        "PARSER_DISABLED"
    ),
    ("_assembly/_drops.py", "report_once", "'worker.drain_raised'"): L("INTERNAL_SDK_ERROR"),
    ("_assembly/_drops.py", "report_once", "counter"): L("SEND_ERROR"),
    ("_assembly/_drops.py", "report_once", "f'{counter}:credentials' if hint else counter"): L(
        "NETWORK_ERROR",
        "SEND_ERROR",
        "RATELIMIT_BACKOFF",
        "SEND_UNCONFIRMED",
        "REQUEST_TOO_LARGE",
        at="transport/_otlp_http.py",
    ),
    ("_assembly/_drops.py", "report_once", "key"): Structural(
        "lose()'s own line; each lose() call is its own entry"
    ),
    ("_assembly/_forgotten.py", "bump", "'assembly._units.alias_forgotten_consumed'"): _UNITS,
    ("_assembly/_forgotten.py", "bump", "'assembly._units.alias_forgotten_table_full'"): _UNITS,
    ("_assembly/_ledger.py", "record", "reason"): Structural(
        "record_items()' own records; each record_items() call is its own entry"
    ),
    ("_assembly/_patchset.py", "bump", "f'{self._owner}.patch_refused'"): _UNITS,
    ("_assembly/_patchset.py", "bump", "f'{self._owner}.strong_hold'"): _UNITS,
    ("_assembly/_patchset.py", "bump", "f'{self._owner}.{_SUPERSEDED_MARKER.value}'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.alias_after_close'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.alias_rebound'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.alias_table_full'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.alias_unknown_key'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.ambient_closed_at_issue'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.ambient_foreign_registry'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.ambient_foreign_retired'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.ambient_stale'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.claim_superseded'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.close_after_close'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.close_all_reentrant'"): _SIGNAL,
    ("_assembly/_units.py", "bump", "'assembly._units.close_span_after_emit'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.emit_duplicate'"): NotADrop(
        "a span already emitted, not emitted twice"
    ),
    ("_assembly/_units.py", "bump", "'assembly._units.link_memory_full'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.note_after_close'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.open_span_after_close'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.parent_closed'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.parent_foreign_registry'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.pin_foreign_task'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.pin_retired_from_afar'"): _UNITS,
    (
        "_assembly/_units.py",
        "bump",
        "'assembly._units.pin_stale' if same_task else 'assembly._units.pin_leaked'",
    ): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.record_truncated'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.stale_activation_ambient'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.stale_ambient_evicted'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.stale_pin_ambient'"): _UNITS,
    ("_assembly/_units.py", "bump", "'assembly._units.stale_pin_foreign_registry'"): _UNITS,
    ("_assembly/_units.py", "bump", "f'assembly._units.{where}_table_full'"): _UNITS,
    # -- the client
    ("_client.py", "bump", "'client.finalize.rejected_closed'"): L("AFTER_CLOSE"),
    ("_client.py", "record", "DropReason.AFTER_CLOSE"): L("AFTER_CLOSE"),
    ("_client.py", "record", "DropReason.BEFORE_SEND"): L("BEFORE_SEND"),
    ("_client.py", "record", "DropReason.QUEUE_OVERFLOW"): L("QUEUE_OVERFLOW"),
    ("_client.py", "record", "DropReason.SHUTDOWN_TAIL"): L("SHUTDOWN_TAIL"),
    ("_client.py", "report_lost", "'client.close.behind_an_export'"): L(
        "SHUTDOWN_TAIL", at="_client.py"
    ),
    ("_client.py", "report_lost", "'client.close.transport_declined'"): L(
        "SHUTDOWN_TAIL", at="_client.py"
    ),
    ("_client.py", "report_lost", "'client.closed.declined_after_close'"): L(
        "SHUTDOWN_TAIL", at="_client.py"
    ),
    ("_client.py", "report_lost", "'wardex.finalize.after_close'"): L(
        "SHUTDOWN_TAIL", at="_client.py"
    ),
    ("_client.py", "report_once", "'client.drain.span_dropped.export_raised'"): L(
        "SEND_ERROR", at="_assembly/_drops.py"
    ),
    ("_client.py", "report_once", "'wardex.before_send_envelope'"): L(
        "BEFORE_SEND_ERROR", at="_assembly/_drops.py"
    ),
    ("_client.py", "report_once", "key"): L("SHUTDOWN_TAIL"),
    ("_finalize.py", "bump", "'client.finalize.backlog_evicted'"): Marker("PARSE_BACKLOG_FULL"),
    ("_finalize.py", "bump", "'client.finalize.shutdown_fallback'"): Marker(
        "PARSE_SKIPPED_AT_SHUTDOWN"
    ),
    ("_finalize.py", "report_once", "'wardex.finalize.backlog'"): Marker(
        "PARSE_BACKLOG_FULL", at="_finalize.py"
    ),
    # -- the manual span API: a guard swallowed, and the line says what it cost
    ("_decorators.py", "report_once", "'wardex.span.generator_scope'"): _GUARDED,
    ("_tracing.py", "report_once", "'wardex.conversation.scope'"): _GUARDED,
    ("_tracing.py", "report_once", "'wardex.span.manual_fork'"): _GUARDED,
    ("_tracing.py", "report_once", "'wardex.span.manual_open'"): _GUARDED,
    ("_tracing.py", "bump", "'tracing.block_scope_retired'"): NotADrop(
        "a block taken down from another context; the consumer gets its own scope back"
    ),
    # -- the byte seams and protocols
    ("_interceptors/_http1_requests.py", "bump", "'protocol.http1.request_unfinished'"): _SEAM,
    ("_interceptors/_http1_requests.py", "bump", "'protocol.http1.request_unobserved'"): _SEAM,
    (
        "_interceptors/_mcp_result.py",
        "bump",
        "'interceptors.mcp_stdio.protocol_version_unreadable'",
    ): _VERSION_REFUSED,
    (
        "_interceptors/_mcp_result.py",
        "report_once",
        "'interceptors.mcp_stdio.protocol_version_unreadable'",
    ): _VERSION_REFUSED,
    ("_interceptors/_mcp_stdio.py", "bump", "'interceptors.mcp_stdio.anyio_unavailable'"): _MCP,
    ("_interceptors/_mcp_stdio.py", "bump", "'interceptors.mcp_stdio.execute_tool_ceded'"): (
        _TOOL_CEDED
    ),
    ("_interceptors/_mcp_stdio.py", "bump", "'interceptors.mcp_stdio.line_over_cap'"): (
        _MCP_LINE_CAP
    ),
    ("_interceptors/_mcp_stdio.py", "bump", "'interceptors.mcp_stdio.pending_evicted'"): (
        _MCP_PENDING_TABLE
    ),
    ("_interceptors/_mcp_stdio.py", "bump", "'interceptors.mcp_stdio.stranded_requests'"): _MCP,
    (
        "_interceptors/_mcp_stdio.py",
        "report_once",
        "'interceptors.mcp_stdio.execute_tool_ceded'",
    ): _TOOL_CEDED,
    ("_interceptors/_mcp_stdio.py", "report_once", "'interceptors.mcp_stdio.line_over_cap'"): (
        _MCP_LINE_CAP
    ),
    ("_interceptors/_mcp_stdio.py", "report_once", "'interceptors.mcp_stdio.no_anyio'"): _MCP,
    ("_interceptors/_mcp_stdio.py", "report_once", "'interceptors.mcp_stdio.pending_evicted'"): (
        _MCP_PENDING_TABLE
    ),
    ("_interceptors/_seam.py", "bump", "'interceptors.seam.path_excluded'"): _POLICY,
    ("_interceptors/_seam.py", "bump", "'interceptors.seam.peer_unresolved'"): Marker(
        "PEER_UNRESOLVED", at="_interceptors/_seam.py"
    ),
    ("_interceptors/_seam.py", "bump", "'interceptors.seam.provider_state_dropped'"): _MODE,
    ("_interceptors/_seam.py", "bump", "'interceptors.seam.stream_unterminated'"): _SEAM,
    ("_interceptors/_seam.py", "bump", "'interceptors.seam.tracking_reset_at_fork'"): Marker(
        "TRACKING_RESET_AT_FORK", at="_interceptors/_seam.py"
    ),
    ("_interceptors/_seam.py", "bump", "'interceptors.seam.usage_leaves_dropped'"): Marker(
        "EXTRA_KEYS_DROPPED"
    ),
    ("_interceptors/_seam.py", "bump", "'interceptors.seam.ws_llm_endpoint_unconfirmed'"): _SEAM,
    ("_interceptors/_seam.py", "bump", "'interceptors.seam.ws_llm_semantics_unread'"): Marker(
        "WS_LLM_SEMANTICS_UNREAD", at="_interceptors/_trackers.py"
    ),
    # An HTTP/2 body past its cap bumps nothing: the span ships with
    # `Limitation.BODY_CAP_EXCEEDED`, and the limitation census holds that site.
    ("_interceptors/_trackers.py", "bump", "'protocol.http2.evicted_parent_forgotten'"): _SEAM,
    ("_interceptors/_trackers.py", "bump", "'protocol.http2.request_unobserved'"): _SEAM,
    ("_interceptors/_trackers.py", "bump", "'protocol.http2.stream_evicted'"): Marker(
        "H2_REQUEST_EVICTED", at="_interceptors/_trackers.py"
    ),
    # Unreachable while the limitation census holds (each code the core emits
    # has a member), but a marker the span cannot carry when it is reached.
    ("_protocol/_http1.py", "bump", "'protocol.limitation_unresolved'"): L("MARKER_LOST"),
    ("_protocol/_http1.py", "record", "DropReason.MARKER_LOST"): L("MARKER_LOST"),
    ("_semantics/_genai.py", "bump", "'protocol.semantic.provider_inferred'"): Marker(
        "PROVIDER_INFERRED", at="_semantics/_genai.py"
    ),
    ("_semantics/_genai.py", "bump", "'protocol.semantic.sse_event_unparsed'"): Marker(
        "SSE_EVENT_UNPARSED", at="_semantics/_genai.py"
    ),
    ("_semantics/_genai.py", "bump", "'semantics.build_gen_ai.usage_overflowed'"): _USAGE,
    ("_semantics/_genai.py", "bump", "'semantics.build_gen_ai.usage_totals_unpaired'"): _USAGE,
    ("_semantics/_genai.py", "bump", "'semantics.request_conversation_shadowed'"): NotADrop(
        "the request's own conversation id wins by rule, and stays on the span"
    ),
    ("_semantics/_genai.py", "bump", "'semantics.request_conversation_withheld'"): NotADrop(
        "the id the request body stated stays on the span under its own key"
    ),
    ("_runtime.py", "bump", "'_runtime.fork_child_reinit'"): NotADrop(
        "a timing tally of the fork reset"
    ),
    # -- the transports
    ("transport/_base.py", "bump", "'transport.otlp.reserved_attribute_overwritten'"): L(
        "ATTRIBUTE_DROPPED"
    ),
    ("transport/_base.py", "bump", "'transport.otlp.span_payload_dropped'"): _REQUEST_CAP,
    ("transport/_base.py", "bump", "'transport.otlp.span_unmarshalled'"): L("UNMARSHALLABLE"),
    ("transport/_base.py", "bump", "key"): L("INTERNAL_SDK_ERROR"),
    ("transport/_base.py", "record", "DropReason.INTERNAL_SDK_ERROR"): L("INTERNAL_SDK_ERROR"),
    ("transport/_base.py", "record", "DropReason.ATTRIBUTE_DROPPED"): L("ATTRIBUTE_DROPPED"),
    ("transport/_base.py", "record", "DropReason.REQUEST_TOO_LARGE"): L("REQUEST_TOO_LARGE"),
    ("transport/_base.py", "record", "DropReason.UNMARSHALLABLE"): L("UNMARSHALLABLE"),
    ("transport/_base.py", "report_once", "'transport.otlp.reserved_attribute_overwritten'"): L(
        "ATTRIBUTE_DROPPED"
    ),
    ("transport/_base.py", "report_once", "'transport.otlp.span_over_request_cap'"): L(
        "REQUEST_TOO_LARGE"
    ),
    ("transport/_base.py", "report_once", "'transport.otlp.span_payload_dropped'"): _REQUEST_CAP,
    ("transport/_base.py", "report_once", "key"): L("INTERNAL_SDK_ERROR"),
    ("transport/_base.py", "report_once", "'transport.otlp.span_unmarshalled'"): L(
        "UNMARSHALLABLE"
    ),
    ("transport/_otlp_http.py", "bump", "'transport.otlp.export_failed'"): L(
        "NETWORK_ERROR",
        "SEND_ERROR",
        "RATELIMIT_BACKOFF",
        "SEND_UNCONFIRMED",
        "REQUEST_TOO_LARGE",
        at="transport/_otlp_http.py",
    ),
    ("transport/_otlp_http.py", "record", "DropReason.BUDGET_EXHAUSTED"): L("BUDGET_EXHAUSTED"),
    ("transport/_otlp_http.py", "record", "lost_as"): L(
        "NETWORK_ERROR",
        "SEND_ERROR",
        "RATELIMIT_BACKOFF",
        "SEND_UNCONFIRMED",
        "REQUEST_TOO_LARGE",
        at="transport/_otlp_http.py",
    ),
    ("transport/_otlp_http.py", "report_once", "'transport.otlp.caller_budget_cut_short'"): L(
        "SEND_UNCONFIRMED", at="transport/_otlp_http.py"
    ),
    ("transport/_otlp_http.py", "report_once", "'transport.otlp.native_unavailable'"): _NATIVE,
    ("transport/_otlp_http.py", "report_once", "'transport.otlp.split_export_abandoned'"): L(
        "NETWORK_ERROR",
        "SEND_ERROR",
        "RATELIMIT_BACKOFF",
        "SEND_UNCONFIRMED",
        "REQUEST_TOO_LARGE",
        at="transport/_otlp_http.py",
    ),
    ("transport/_otlp_http.py", "report_once", "'transport.otlp.split_export_out_of_budget'"): L(
        "BUDGET_EXHAUSTED"
    ),
    ("transport/_wardex.py", "bump", "'transport.wardex.export_failed'"): L(
        "NETWORK_ERROR",
        "SEND_ERROR",
        "RATELIMIT_BACKOFF",
        "SEND_UNCONFIRMED",
        "REQUEST_TOO_LARGE",
        at="transport/_otlp_http.py",
    ),
    ("transport/_wardex.py", "bump", "'transport.wardex.span_unmarshalled'"): L("UNMARSHALLABLE"),
    ("transport/_wardex.py", "record", "DropReason.UNMARSHALLABLE"): L("UNMARSHALLABLE"),
    ("transport/_wardex.py", "record", "lost_as"): L(
        "NETWORK_ERROR",
        "SEND_ERROR",
        "RATELIMIT_BACKOFF",
        "SEND_UNCONFIRMED",
        "REQUEST_TOO_LARGE",
        at="transport/_otlp_http.py",
    ),
    ("transport/_wardex.py", "report_once", "'transport.wardex.caller_budget_cut_short'"): L(
        "SEND_UNCONFIRMED", at="transport/_otlp_http.py"
    ),
    ("transport/_wardex.py", "report_once", "'transport.wardex.native_unavailable'"): _NATIVE,
    ("transport/_wardex.py", "report_once", "'transport.wardex.span_unmarshalled'"): L(
        "UNMARSHALLABLE"
    ),
}


# ==============================================================================
# The scanner
# ==============================================================================

#: The ledger's own calls, as a caller spells them: bare after `from ... import
#: record` (under any alias), or through the module (`ledger.record`,
#: `_ledger.record`, a dotted path ending in `_ledger`, or any alias the module
#: was imported under). A `registry.record(status)` is not one.
_LEDGER_CALLS = frozenset({"record", "record_items", "lose"})
_LEDGER_MODULES = frozenset({"ledger", "_ledger"})
#: The signal functions a module can import by name, and so under another one.
_IMPORTED_SIGNALS = _LEDGER_CALLS | {"report_once"}


@dataclass(frozen=True)
class _Bindings:
    """What a module's imports bind, under whatever alias: names that stand for
    a signal function (mapped to its own name), and names for the ledger module."""

    calls: dict[str, str]
    modules: frozenset[str]


def _bindings(tree: ast.AST) -> _Bindings:
    calls: dict[str, str] = {}
    modules = set(_LEDGER_MODULES)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import | ast.ImportFrom):
            continue
        for alias in node.names:
            last = alias.name.rsplit(".", 1)[-1]
            if last == "_ledger" and (alias.asname or isinstance(node, ast.ImportFrom)):
                modules.add(alias.asname or last)
            elif last in _IMPORTED_SIGNALS and isinstance(node, ast.ImportFrom):
                calls[alias.asname or last] = last
    return _Bindings(calls, frozenset(modules))


@dataclass(frozen=True)
class _Site:
    module: str
    kind: str
    key: str
    scope: ast.AST  # the innermost function around the call, or the module


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def _getattr_spelled(func: ast.expr) -> ast.expr:
    """`getattr(obj, "name")` read as the `obj.name` it is."""
    if (
        isinstance(func, ast.Call)
        and isinstance(func.func, ast.Name)
        and func.func.id == "getattr"
        and len(func.args) >= 2
        and isinstance(func.args[1], ast.Constant)
        and isinstance(func.args[1].value, str)
    ):
        return ast.Attribute(value=func.args[0], attr=func.args[1].value, ctx=ast.Load())
    return func


def _tail(node: ast.expr) -> str | None:
    """The last name in `a.b.c` (`c`), or the name itself."""
    return node.id if isinstance(node, ast.Name) else getattr(node, "attr", None)


def _signal(call: ast.Call, module: str, bindings: _Bindings) -> tuple[str, ast.expr | None] | None:
    """`(kind, key node)` when `call` is a drop signal, else None."""
    func = _getattr_spelled(call.func)
    if isinstance(func, ast.Name):
        name, base = bindings.calls.get(func.id, func.id), None
    elif isinstance(func, ast.Attribute):
        name, base = func.attr, _tail(func.value)
    else:
        return None
    first = call.args[0] if call.args else None
    if name == "bump" and isinstance(func, ast.Attribute):
        return "bump", first
    if name == "count" and isinstance(func, ast.Attribute) and module.startswith("_adapters/"):
        return "count", first
    if name == "report_once":
        return "report_once", _keyword(call, "key")
    if name == "_report_lost":
        return "report_lost", _keyword(call, "key")
    if name in _LEDGER_CALLS and (isinstance(func, ast.Name) or base in bindings.modules):
        return ("lose", _keyword(call, "key")) if name == "lose" else ("record", first)
    return None


def _scan(root: pathlib.Path) -> dict[tuple[str, str, str], list[_Site]]:
    sites: dict[tuple[str, str, str], list[_Site]] = {}
    for path in sorted(root.rglob("*.py")):
        module = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        bindings = _bindings(tree)

        def visit(
            node: ast.AST, scope: ast.AST, module: str = module, bindings: _Bindings = bindings
        ) -> None:
            for child in ast.iter_child_nodes(node):
                inner = (
                    child
                    if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda)
                    else scope
                )
                if isinstance(child, ast.Call):
                    found = _signal(child, module, bindings)
                    if found is not None:
                        kind, key = found
                        text = ast.unparse(key) if key is not None else "<no key>"
                        site = _Site(module, kind, text, scope)
                        sites.setdefault((module, kind, text), []).append(site)
                visit(child, inner)

        visit(tree, tree)
    return sites


@pytest.fixture(scope="module")
def scanned() -> dict[tuple[str, str, str], list[_Site]]:
    return _scan(_SRC)


def _names_in(node: ast.AST, owner: str) -> set[str]:
    """Every `<owner>.<NAME>` spelled inside `node`."""
    return {
        sub.attr
        for sub in ast.walk(node)
        if isinstance(sub, ast.Attribute)
        and isinstance(sub.value, ast.Name)
        and sub.value.id == owner
    }


def _module_tree(module: str) -> ast.Module:
    return ast.parse((_SRC / module).read_text(encoding="utf-8"))


# ==============================================================================
# The tests
# ==============================================================================


def test_every_drop_signal_is_in_the_census_and_every_entry_is_in_the_source(scanned):
    new = sorted(set(scanned) - set(_CENSUS))
    gone = sorted(set(_CENSUS) - set(scanned))
    assert not new, (
        "drop signals with no census entry -- decide what happens to each loss "
        "(record it in the drop ledger, mark the span, or say why it is not a "
        f"loss) and add the entry: {new}"
    )
    assert not gone, f"census entries whose site is gone from the source -- remove them: {gone}"


def test_a_ledger_entry_names_reasons_its_site_records(scanned):
    """A `Ledger` entry is a claim that the loss reaches the ledger under those
    reasons. Held where it can be held from source: each reason is a real
    member, and is spelled `DropReason.<NAME>` beside the signal -- in the same
    function, or in the module the entry points at."""
    wrong: list[str] = []
    for key, disposition in _CENSUS.items():
        if not isinstance(disposition, Ledger):
            continue
        unknown = [r for r in disposition.reasons if r not in DropReason.__members__]
        if unknown:
            wrong.append(f"{key}: no such reason {unknown}")
            continue
        if disposition.at is not None:
            spelled = _names_in(_module_tree(disposition.at), "DropReason")
        else:
            spelled = set().union(*(_names_in(s.scope, "DropReason") for s in scanned[key]))
        missing = [r for r in disposition.reasons if r not in spelled]
        if missing:
            where = f"in {disposition.at}" if disposition.at else "beside it"
            wrong.append(f"{key}: {missing} not recorded {where}")
    assert not wrong, "\n".join(wrong)


def test_a_marker_entry_names_a_member_its_site_attaches(scanned):
    wrong: list[str] = []
    for key, disposition in _CENSUS.items():
        if not isinstance(disposition, Marker):
            continue
        if disposition.member not in Limitation.__members__:
            wrong.append(f"{key}: no such Limitation {disposition.member}")
            continue
        if disposition.at is not None:
            spelled = _names_in(_module_tree(disposition.at), "Limitation")
        else:
            spelled = set().union(*(_names_in(s.scope, "Limitation") for s in scanned[key]))
        if disposition.member not in spelled:
            wrong.append(f"{key}: Limitation.{disposition.member} not attached beside it")
    assert not wrong, "\n".join(wrong)


#: What a public tree may not carry in its prose: a tracker key or a work-stream
#: number explains nothing to a reader who cannot open the tracker.
_INTERNAL_REFERENCE = re.compile(r"\b[A-Z]{2,}-\d+\b|\bWS-?\d", re.IGNORECASE)


def test_every_entry_says_why_in_plain_words():
    for key, disposition in _CENSUS.items():
        text = getattr(disposition, "why", None) or getattr(disposition, "area", None)
        if isinstance(disposition, Ledger | Marker):
            continue
        assert text and len(text) > 10, f"{key}: say why"
        assert not _INTERNAL_REFERENCE.search(text), f"{key}: {text!r} cites an internal key"


def test_a_loss_a_marker_carries_is_filed_as_a_marker():
    """Nothing checks a `NotADrop` reason against the source; a `Marker` entry
    is checked. So "a span carries a marker for it" is never a `NotADrop`
    reason: it is a `Marker`, with `at` naming the module that attaches the
    member when that is not the signal's own function."""
    wrong = [k for k, d in _CENSUS.items() if isinstance(d, NotADrop) and "marker" in d.why.lower()]
    assert not wrong, f"file these as Marker(member, at=...): {wrong}"


def test_the_ledger_vocabulary_is_used_through_its_members_only(scanned):
    """Every reason a ledger call names directly is a member: a typo in a
    `DropReason.<NAME>` would raise at import, but a string would not."""
    for (module, kind, key), _sites in scanned.items():
        if kind == "record" and key.startswith(("'", '"')):
            pytest.fail(f"{module}: record({key}) names a reason by string, not by member")


def test_the_ledger_never_waits_for_a_lock():
    """`record` runs under the client's locks and inside signal handlers, and the
    drain's `attach` may never wait either. So the ledger may take its lock
    only with `acquire(blocking=False)`, and may not enter a lock with `with`,
    wait on an Event or a Condition, or sleep."""
    for module in ("_assembly/_ledger.py", "_assembly/_drop_reasons.py"):
        tree = _module_tree(module)
        for node in ast.walk(tree):
            if isinstance(node, ast.With | ast.AsyncWith):
                pytest.fail(f"{module}:{node.lineno}: a `with` block -- a lock entered so waits")
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            name = node.func.attr
            if name == "acquire":
                nonblocking = any(
                    kw.arg == "blocking"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is False
                    for kw in node.keywords
                ) or (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value is False
                )
                assert nonblocking, f"{module}:{node.lineno}: a blocking acquire()"
            assert name not in {"wait", "wait_for", "sleep", "join"}, (
                f"{module}:{node.lineno}: {name}() waits"
            )


def test_the_census_sees_every_kind_of_signal(tmp_path):
    """The scanner against a synthetic module holding one of each signal, in the
    spellings the SDK uses -- and the non-signals it must ignore."""
    src = tmp_path / "src"
    (src / "_adapters").mkdir(parents=True)
    (src / "_adapters" / "mod.py").write_text(
        "def f(ctx, registry, counters, ledger):\n"
        "    counters.bump('a.b')\n"
        "    ctx.count('c')\n"
        "    report_once('line', key='k')\n"
        "    self._report_lost(1, why='w', key='lost')\n"
        "    record(DropReason.QUEUE_OVERFLOW)\n"
        "    ledger.record_items(DropReason.SEND_ERROR, 1, 0)\n"
        "    lose(DropReason.TABLE_FULL, line='l', key='lk')\n"
        "    registry.record(status)\n"
        "    text.count('x')\n",
        encoding="utf-8",
    )
    (src / "other.py").write_text("def g(text):\n    return text.count('x')\n", encoding="utf-8")
    found = set(_scan(src))
    assert found == {
        ("_adapters/mod.py", "bump", "'a.b'"),
        ("_adapters/mod.py", "count", "'c'"),
        ("_adapters/mod.py", "report_once", "'k'"),
        ("_adapters/mod.py", "report_lost", "'lost'"),
        ("_adapters/mod.py", "record", "DropReason.QUEUE_OVERFLOW"),
        ("_adapters/mod.py", "record", "DropReason.SEND_ERROR"),
        ("_adapters/mod.py", "lose", "'lk'"),
        ("_adapters/mod.py", "count", "'x'"),  # inside _adapters/ every .count is read
    }


def test_a_new_silent_bump_fails_the_census(tmp_path):
    """The negative control, on a copy of the real tree: one new counter bump
    nobody classified, and the census must see it."""
    copy = tmp_path / "wardex_sdk"
    shutil.copytree(_SRC, copy, ignore=shutil.ignore_patterns("__pycache__", "*.so"))
    target = copy / "_worker.py"
    target.write_text(
        target.read_text(encoding="utf-8")
        + "\n\ndef _unclassified(counters):\n    counters.bump('x.y')\n",
        encoding="utf-8",
    )
    new = set(_scan(copy)) - set(_CENSUS)
    assert new == {("_worker.py", "bump", "'x.y'")}


def test_the_census_sees_a_signal_under_any_name(tmp_path):
    """A signal does not hide behind an alias: an imported name, a module
    alias, a dotted path, or a `getattr` spelling of the call is still the
    signal, and the census sees each."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "mod.py").write_text(
        "from ._assembly import _ledger as led\n"
        "from ._assembly._ledger import record as _rec, record_items as _ri\n"
        "from ._assembly import report_once as say\n"
        "import wardex_sdk._assembly._ledger\n"
        "def f(counters):\n"
        "    led.record(DropReason.TABLE_FULL)\n"
        "    _rec(DropReason.QUEUE_OVERFLOW)\n"
        "    _ri(DropReason.SEND_ERROR, 1, 0)\n"
        "    wardex_sdk._assembly._ledger.record(DropReason.AFTER_CLOSE)\n"
        "    getattr(counters, 'bump')('x.alias')\n"
        "    say('line', key='k.alias')\n",
        encoding="utf-8",
    )
    assert set(_scan(src)) == {
        ("mod.py", "record", "DropReason.TABLE_FULL"),
        ("mod.py", "record", "DropReason.QUEUE_OVERFLOW"),
        ("mod.py", "record", "DropReason.SEND_ERROR"),
        ("mod.py", "record", "DropReason.AFTER_CLOSE"),
        ("mod.py", "bump", "'x.alias'"),
        ("mod.py", "report_once", "'k.alias'"),
    }


def _uncalled_signals(tree: ast.AST, module: str) -> list[str]:
    """Every place `tree` names a signal callable without calling it there."""
    bindings = _bindings(tree)
    called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    found: list[str] = []
    for node in ast.walk(tree):
        if id(node) in called:
            continue
        spelled = _getattr_spelled(node) if isinstance(node, ast.Call) else node
        if isinstance(spelled, ast.Name) and isinstance(spelled.ctx, ast.Load):
            hit = spelled.id in bindings.calls
        elif isinstance(spelled, ast.Attribute) and isinstance(spelled.ctx, ast.Load):
            name = spelled.attr
            hit = (
                name in {"bump", "report_once", "_report_lost"}
                or (name == "count" and module.startswith("_adapters/"))
                or (name in _LEDGER_CALLS and _tail(spelled.value) in bindings.modules)
            )
        else:
            hit = False
        if hit:
            found.append(f"{module}:{node.lineno}: {ast.unparse(node)}")
    return found


def test_a_signal_is_only_ever_called_never_passed_or_renamed():
    """The rule that keeps the census closed under aliasing: a signal callable
    is never assigned to another name, passed along, or stored -- `say =
    report_once`, `bump = counters.bump`, `map(ledger.record, ...)` -- because
    the call made through the new name is one no scanner can attribute."""
    offenders = [
        hit
        for path in sorted(_SRC.rglob("*.py"))
        for hit in _uncalled_signals(
            ast.parse(path.read_text(encoding="utf-8")), path.relative_to(_SRC).as_posix()
        )
    ]
    assert not offenders, offenders


def test_the_renaming_rule_sees_each_way_of_renaming():
    """The negative control for the rule above: every renaming it forbids is
    seen, and an ordinary local named like a signal is not."""
    tree = ast.parse(
        "from ._assembly import report_once\n"
        "from ._ledger import record as _rec\n"
        "from . import _ledger\n"
        "def f(counters, ctx, rows):\n"
        "    say = report_once\n"
        "    bump = counters.bump\n"
        "    later = _rec\n"
        "    hooks = [_ledger.record_items]\n"
        "    tally = getattr(counters, 'bump')\n"
        "    sink = ctx.count\n"
        "    for record in rows:\n"
        "        record.close()\n"
    )
    found = {hit.split(": ", 1)[1] for hit in _uncalled_signals(tree, "_adapters/mod.py")}
    assert found == {
        "report_once",
        "counters.bump",
        "_rec",
        "_ledger.record_items",
        "getattr(counters, 'bump')",
        "ctx.count",
    }
