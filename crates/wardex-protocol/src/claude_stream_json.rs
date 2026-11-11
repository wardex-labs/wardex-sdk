//! Parser for the Claude Code CLI stream-json wire protocol.
//! This is the message stream spoken between the Agent SDKs (Python/Node) and
//! the `claude` CLI subprocess. Stateless per line; unknown message types map
//! to None (forward compatibility). Timestamps are stamped by the caller.

use serde_json::Value;

use crate::json::{i64_field, str_field};
use crate::usage::{InputConvention, TokenUsage};

/// Which line of the stream an event came from; it decides which of the
/// event's fields can be set at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    /// `system`/`init`: the CLI announcing its session and model.
    SessionInit,
    /// A `user` line written by the host to the CLI: the prompt itself.
    UserPrompt,
    /// An `assistant` line: one model message, with its usage and any tool
    /// calls it made.
    AssistantTurn,
    /// A `user` line written by the CLI that carries a `tool_result` block.
    ToolResult,
    /// A `stream_event` line: a partial-message delta. Only its ids are read;
    /// the delta itself is not parsed.
    StreamDelta,
    /// A `system` line about a background task: `task_started`,
    /// `task_progress`, `task_notification` or `task_updated`.
    TaskLifecycle,
    /// The closing `result` line: the CLI's own totals for the session.
    SessionResult,
}

/// One `tool_use` block of an assistant message.
#[derive(Debug, Clone)]
pub struct ToolUse {
    /// The block's `id`; the later `tool_result` names it in
    /// `ClaudeStreamEvent::tool_result_id`.
    pub id: String,
    /// The tool's name as the model called it, such as `Bash` or `Task`.
    pub name: String,
    /// The block's `input` object re-serialized as JSON. Empty when the block
    /// had no `input`.
    pub input_json: Vec<u8>,
}

/// Flat event struct: one shape for all kinds keeps the FFI surface trivial.
///
/// Every `Option` field is `None` when the line did not carry that key (or
/// carried it with another JSON type); nothing is filled in by inference.
#[derive(Debug, Clone)]
pub struct ClaudeStreamEvent {
    /// Which line this was, and so which fields below can be set.
    pub kind: EventKind,
    /// The CLI's session id. On an `AssistantTurn` it falls back to the
    /// message's own `session_id` when the line has none.
    pub session_id: Option<String>,
    /// `SessionInit` and `AssistantTurn` only: the model the CLI named.
    pub model: Option<String>,
    /// `AssistantTurn` only: the API message id (`msg_...`).
    pub message_id: Option<String>,
    /// `AssistantTurn` only: the message's `stop_reason`, verbatim.
    pub stop_reason: Option<String>,
    /// Which sub-agent produced the line: the id of the `Task` call that
    /// spawned it. `None` on lines from the top-level agent.
    pub parent_tool_use_id: Option<String>,
    /// Which call a `tool_result` answers. Distinct from `parent_tool_use_id`,
    /// which says which sub-agent produced the line, because one CLI line
    /// carries both and folding them into one field answers the wrong question
    /// exactly when a sub-agent runs a tool: the result would be filed against
    /// the `Task` call that spawned the agent, carrying the inner tool's
    /// output, and the inner call would get no result at all.
    pub tool_result_id: Option<String>,
    /// A `tool_result` that answers its call by launching it, not by finishing
    /// it: the line's `tool_use_result.status` is `async_launched`, as for an
    /// `Agent` call run in the background. The call has returned; the
    /// sub-agent it started has not, and goes on writing lines under the
    /// call's id until its task's `task_notification`.
    pub launched_async: bool,
    /// JSON of what the line said, re-serialized: the whole `message` for a
    /// `UserPrompt` or `ToolResult`, the `content` array for an
    /// `AssistantTurn`. `None` on other kinds.
    pub content_json: Option<Vec<u8>>,
    /// `AssistantTurn` only: its `tool_use` blocks in message order. A block
    /// missing its `id` or `name` is left out.
    pub tool_uses: Vec<ToolUse>,
    /// Normalized (semconv-inclusive) — `parse_usage` names Anthropic's
    /// convention, so `input_tokens()` already contains both cache tiers.
    pub usage: Option<TokenUsage>,
    /// `TaskLifecycle` only: the background task's id.
    pub task_id: Option<String>,
    /// `TaskLifecycle` only: the task's `status`, verbatim.
    pub task_status: Option<String>,
    /// `TaskLifecycle` only: the id of the tool call that started the task.
    pub task_tool_use_id: Option<String>,
    /// `SessionResult` only: model turns the CLI counted for the session.
    pub num_turns: Option<i64>,
    /// `SessionResult` only: the CLI's own cost estimate in US dollars. It is
    /// the CLI's figure, not one computed here from token counts.
    pub total_cost_usd: Option<f64>,
    /// `SessionResult` only: wall-clock milliseconds the CLI reports for the
    /// session.
    pub duration_ms: Option<i64>,
    /// `SessionResult` only: milliseconds of that time the CLI spent waiting
    /// on the model API.
    pub duration_api_ms: Option<i64>,
    /// `ToolResult`: the result block's `is_error`, so a failed tool call is
    /// told apart from one that returned. `SessionResult`: the line's own
    /// `is_error`. `false` when the key is absent, and on every other kind.
    pub is_error: bool,
    /// The `system` line's `subtype` on a `TaskLifecycle` (which of the four
    /// task events), or the `result` line's `subtype` on a `SessionResult`
    /// (such as `success`).
    pub subtype: Option<String>,
}

impl ClaudeStreamEvent {
    fn new(kind: EventKind) -> Self {
        Self {
            kind,
            session_id: None,
            model: None,
            message_id: None,
            stop_reason: None,
            parent_tool_use_id: None,
            tool_result_id: None,
            launched_async: false,
            content_json: None,
            tool_uses: Vec::new(),
            usage: None,
            task_id: None,
            task_status: None,
            task_tool_use_id: None,
            num_turns: None,
            total_cost_usd: None,
            duration_ms: None,
            duration_api_ms: None,
            is_error: false,
            subtype: None,
        }
    }
}

fn raw(v: &Value) -> Option<Vec<u8>> {
    serde_json::to_vec(v).ok()
}

fn parse_usage(m: &Value) -> Option<TokenUsage> {
    let u = m.get("usage")?;
    // The CLI relays Anthropic's own usage object, cache tiers reported
    // outside `input_tokens` — the same `ExcludesCache` fact as the HTTP
    // body parser, declared with the same type so neither path can drift.
    Some(TokenUsage::new(
        InputConvention::ExcludesCache,
        i64_field(u, "input_tokens"),
        i64_field(u, "output_tokens"),
        i64_field(u, "cache_read_input_tokens"),
        i64_field(u, "cache_creation_input_tokens"),
        None,
    ))
}

/// Parses one line of the stream. `outbound` is true for a line the host wrote
/// to the CLI and false for one the CLI wrote back.
///
/// Returns `None` for a line that is not JSON, has no string `type`, or is of
/// a type this parser does not treat as semantic (control traffic, and types
/// newer than this parser); also for an `assistant` line without a `message`,
/// a `system` line without a string `subtype`, and an inbound `user` line that
/// carries no `tool_result` block. Timestamps are not on the wire; the caller
/// stamps them.
pub fn parse_stream_line(line: &[u8], outbound: bool) -> Option<ClaudeStreamEvent> {
    let v: Value = serde_json::from_slice(line).ok()?;
    let typ = v.get("type")?.as_str()?;
    if outbound {
        // Host -> CLI. Only plain user prompts are semantic; control traffic is not.
        if typ != "user" {
            return None;
        }
        let mut e = ClaudeStreamEvent::new(EventKind::UserPrompt);
        e.session_id = str_field(&v, "session_id");
        e.parent_tool_use_id = str_field(&v, "parent_tool_use_id");
        e.content_json = v.get("message").and_then(raw);
        return Some(e);
    }
    match typ {
        "system" => {
            let subtype = str_field(&v, "subtype")?;
            match subtype.as_str() {
                "init" => {
                    let mut e = ClaudeStreamEvent::new(EventKind::SessionInit);
                    e.session_id = str_field(&v, "session_id");
                    e.model = str_field(&v, "model");
                    Some(e)
                }
                "task_started" | "task_progress" | "task_notification" | "task_updated" => {
                    let mut e = ClaudeStreamEvent::new(EventKind::TaskLifecycle);
                    e.session_id = str_field(&v, "session_id");
                    e.task_id = str_field(&v, "task_id");
                    e.task_status = str_field(&v, "status");
                    e.task_tool_use_id = str_field(&v, "tool_use_id");
                    e.subtype = Some(subtype);
                    Some(e)
                }
                _ => None,
            }
        }
        "assistant" => {
            let m = v.get("message")?;
            let mut e = ClaudeStreamEvent::new(EventKind::AssistantTurn);
            e.session_id = str_field(&v, "session_id").or_else(|| str_field(m, "session_id"));
            e.parent_tool_use_id = str_field(&v, "parent_tool_use_id");
            e.model = str_field(m, "model");
            e.message_id = str_field(m, "id");
            e.stop_reason = str_field(m, "stop_reason");
            e.usage = parse_usage(m);
            e.content_json = m.get("content").and_then(raw);
            if let Some(blocks) = m.get("content").and_then(Value::as_array) {
                for b in blocks {
                    if b.get("type").and_then(Value::as_str) == Some("tool_use") {
                        if let (Some(id), Some(name)) = (str_field(b, "id"), str_field(b, "name")) {
                            e.tool_uses.push(ToolUse {
                                id,
                                name,
                                input_json: b.get("input").and_then(raw).unwrap_or_default(),
                            });
                        }
                    }
                }
            }
            Some(e)
        }
        "user" => {
            // Inbound user messages carry tool results back to the conversation.
            let mut e = ClaudeStreamEvent::new(EventKind::ToolResult);
            e.session_id = str_field(&v, "session_id");
            e.parent_tool_use_id = str_field(&v, "parent_tool_use_id");
            let block = v
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(Value::as_array)
                .and_then(|blocks| {
                    blocks
                        .iter()
                        .find(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"))
                });
            e.tool_result_id = block.and_then(|b| str_field(b, "tool_use_id"));
            // The block says whether the call failed. Unread, a reconstructed
            // tool span would be reported `ok` on the strength of the result
            // having arrived at all -- so a failing tool and a succeeding one
            // would ship the same status, which is the one field anyone filters
            // an agent run by.
            e.is_error = block
                .and_then(|b| b.get("is_error"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            // The tool_result block is what makes this line semantic here. Keying
            // the refusal on `parent_tool_use_id` would also admit a sub-agent's
            // ordinary user message, which carries that field and no result.
            e.tool_result_id.as_ref()?;
            // Outside the message, in the CLI's own record of the call's
            // outcome: the block itself reads like any other result.
            e.launched_async = v
                .get("tool_use_result")
                .and_then(|r| r.get("status"))
                .and_then(Value::as_str)
                == Some("async_launched");
            e.content_json = v.get("message").and_then(raw);
            Some(e)
        }
        "stream_event" => {
            let mut e = ClaudeStreamEvent::new(EventKind::StreamDelta);
            e.session_id = str_field(&v, "session_id");
            e.parent_tool_use_id = str_field(&v, "parent_tool_use_id");
            Some(e)
        }
        "result" => {
            let mut e = ClaudeStreamEvent::new(EventKind::SessionResult);
            e.session_id = str_field(&v, "session_id");
            e.subtype = str_field(&v, "subtype");
            e.is_error = v.get("is_error").and_then(Value::as_bool).unwrap_or(false);
            e.num_turns = i64_field(&v, "num_turns");
            e.total_cost_usd = v.get("total_cost_usd").and_then(Value::as_f64);
            e.duration_ms = i64_field(&v, "duration_ms");
            e.duration_api_ms = i64_field(&v, "duration_api_ms");
            Some(e)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INIT: &[u8] = br#"{"type":"system","subtype":"init","session_id":"s-1","model":"claude-sonnet-5","cwd":"/tmp"}"#;
    const ASSISTANT_TOOL: &[u8] = br#"{"type":"assistant","session_id":"s-1","message":{"id":"msg_01","model":"claude-sonnet-5","stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":25,"cache_read_input_tokens":3,"cache_creation_input_tokens":0},"content":[{"type":"text","text":"checking"},{"type":"tool_use","id":"toolu_01","name":"Bash","input":{"command":"ls"}}]}}"#;
    const TOOL_RESULT: &[u8] = br#"{"type":"user","session_id":"s-1","parent_tool_use_id":"toolu_01","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01","content":"file.txt"}]}}"#;
    const RESULT: &[u8] = br#"{"type":"result","subtype":"success","session_id":"s-1","is_error":false,"num_turns":2,"total_cost_usd":0.0123,"duration_ms":5000,"duration_api_ms":4200,"usage":{"input_tokens":10,"output_tokens":25}}"#;
    const STREAM_EVENT: &[u8] = br#"{"type":"stream_event","session_id":"s-1","uuid":"u1","event":{"type":"content_block_delta"}}"#;
    const TASK_STARTED: &[u8] = br#"{"type":"system","subtype":"task_started","session_id":"s-1","task_id":"t-1","description":"explore","tool_use_id":"toolu_02","uuid":"u2"}"#;
    const OUTBOUND_USER: &[u8] = br#"{"type":"user","message":{"role":"user","content":"find bugs"},"parent_tool_use_id":null,"session_id":"s-1"}"#;

    #[test]
    fn parses_session_init() {
        let e = parse_stream_line(INIT, false).unwrap();
        assert_eq!(e.kind, EventKind::SessionInit);
        assert_eq!(e.session_id.as_deref(), Some("s-1"));
        assert_eq!(e.model.as_deref(), Some("claude-sonnet-5"));
    }

    #[test]
    fn parses_assistant_turn_with_tool_use() {
        let e = parse_stream_line(ASSISTANT_TOOL, false).unwrap();
        assert_eq!(e.kind, EventKind::AssistantTurn);
        assert_eq!(e.message_id.as_deref(), Some("msg_01"));
        assert_eq!(e.stop_reason.as_deref(), Some("tool_use"));
        let u = e.usage.unwrap();
        // Inclusive: raw 10 + cache_read 3 + cache_creation 0.
        assert_eq!(u.input_tokens(), Some(13));
        assert_eq!(u.output_tokens(), Some(25));
        assert_eq!(u.cache_read_input_tokens(), Some(3));
        assert_eq!(e.tool_uses.len(), 1);
        assert_eq!(e.tool_uses[0].id, "toolu_01");
        assert_eq!(e.tool_uses[0].name, "Bash");
        assert!(!e.content_json.as_ref().unwrap().is_empty());
    }

    #[test]
    fn parses_tool_result() {
        let e = parse_stream_line(TOOL_RESULT, false).unwrap();
        assert_eq!(e.kind, EventKind::ToolResult);
        assert_eq!(e.tool_result_id.as_deref(), Some("toolu_01"));
        assert_eq!(e.parent_tool_use_id.as_deref(), Some("toolu_01"));
    }

    /// A result produced inside a sub-agent names the call it answers, not the
    /// `Task` that spawned the agent. The CLI puts both on one line and they are
    /// different questions; folding them would cost the inner call its result
    /// and give the outer one bytes that were never its own.
    #[test]
    fn a_result_inside_a_subagent_names_the_call_it_answers() {
        const NESTED: &[u8] = br#"{"type":"user","session_id":"s-1","parent_tool_use_id":"toolu_TASK","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_BASH","content":"out"}]}}"#;
        let e = parse_stream_line(NESTED, false).unwrap();
        assert_eq!(e.tool_result_id.as_deref(), Some("toolu_BASH"));
        assert_eq!(e.parent_tool_use_id.as_deref(), Some("toolu_TASK"));
    }

    /// A tool that failed says so in its own result block.
    #[test]
    fn a_failed_tool_result_carries_its_failure() {
        const FAILED: &[u8] = br#"{"type":"user","session_id":"s-1","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01","content":"boom","is_error":true}]}}"#;
        let e = parse_stream_line(FAILED, false).unwrap();
        assert_eq!(e.tool_result_id.as_deref(), Some("toolu_01"));
        assert!(e.is_error);
        // and the ordinary one still does not
        assert!(!parse_stream_line(TOOL_RESULT, false).unwrap().is_error);
    }

    /// A background `Agent` call's result is its launch, recorded by the CLI
    /// beside the block, and only that status says so: the block is ordinary.
    #[test]
    fn a_result_that_launched_its_call_says_so() {
        const LAUNCHED: &[u8] = br#"{"type":"user","session_id":"s-1","parent_tool_use_id":null,"message":{"role":"user","content":[{"tool_use_id":"toolu_AGENT","type":"tool_result","content":[{"type":"text","text":"Async agent launched successfully."}]}]},"tool_use_result":{"isAsync":true,"status":"async_launched","agentId":"a1"}}"#;
        const FINISHED: &[u8] = br#"{"type":"user","session_id":"s-1","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_AGENT","content":"done"}]},"tool_use_result":{"status":"completed"}}"#;
        const TEXT_OUTCOME: &[u8] = br#"{"type":"user","session_id":"s-1","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01","content":"boom","is_error":true}]},"tool_use_result":"Error: boom"}"#;
        let e = parse_stream_line(LAUNCHED, false).unwrap();
        assert_eq!(e.tool_result_id.as_deref(), Some("toolu_AGENT"));
        assert!(e.launched_async);
        assert!(!parse_stream_line(FINISHED, false).unwrap().launched_async);
        assert!(
            !parse_stream_line(TEXT_OUTCOME, false)
                .unwrap()
                .launched_async
        );
        assert!(
            !parse_stream_line(TOOL_RESULT, false)
                .unwrap()
                .launched_async
        );
    }

    /// A sub-agent's ordinary user message carries `parent_tool_use_id` and no
    /// result block. Keying the refusal on that field would admit it as a tool
    /// result whose content is the whole message.
    #[test]
    fn a_subagents_plain_message_is_not_a_tool_result() {
        const PLAIN: &[u8] = br#"{"type":"user","session_id":"s-1","parent_tool_use_id":"toolu_TASK","message":{"role":"user","content":"carry on"}}"#;
        assert!(parse_stream_line(PLAIN, false).is_none());
    }

    #[test]
    fn parses_session_result() {
        let e = parse_stream_line(RESULT, false).unwrap();
        assert_eq!(e.kind, EventKind::SessionResult);
        assert_eq!(e.num_turns, Some(2));
        assert_eq!(e.total_cost_usd, Some(0.0123));
        assert_eq!(e.duration_api_ms, Some(4200));
        assert!(!e.is_error);
    }

    #[test]
    fn parses_stream_delta_and_task_lifecycle() {
        assert_eq!(
            parse_stream_line(STREAM_EVENT, false).unwrap().kind,
            EventKind::StreamDelta
        );
        let t = parse_stream_line(TASK_STARTED, false).unwrap();
        assert_eq!(t.kind, EventKind::TaskLifecycle);
        assert_eq!(t.task_id.as_deref(), Some("t-1"));
        assert_eq!(t.task_tool_use_id.as_deref(), Some("toolu_02"));
    }

    #[test]
    fn parses_outbound_user_prompt() {
        let e = parse_stream_line(OUTBOUND_USER, true).unwrap();
        assert_eq!(e.kind, EventKind::UserPrompt);
        assert!(!e.content_json.as_ref().unwrap().is_empty());
    }

    #[test]
    fn unknown_type_and_garbage_return_none() {
        assert!(parse_stream_line(br#"{"type":"totally_new_thing","x":1}"#, false).is_none());
        assert!(parse_stream_line(b"not json at all", false).is_none());
        assert!(parse_stream_line(br#"{"type":"control_response","response":{}}"#, true).is_none());
    }
}
