//! Parser for the event stream `codex exec --json` writes to stdout.
//!
//! One JSON object per line: `thread.started`, `turn.started`,
//! `turn.completed` (with the turn's usage), `turn.failed`, `item.started` /
//! `item.updated` / `item.completed` (each carrying one thread item), and a
//! top-level `error`. Stateless per line; unknown event types map to None
//! (forward compatibility), and an item of a type this parser does not name is
//! still returned with its raw JSON, so a caller decides what it is worth.
//!
//! What the stream does NOT carry, so no field here pretends to: the model
//! name, per-model-call boundaries, timestamps. `turn.completed.usage` is the
//! whole turn's total across every model call in it.

use serde_json::Value;

use crate::usage::{InputConvention, TokenUsage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexEventKind {
    ThreadStarted,
    TurnStarted,
    TurnCompleted,
    TurnFailed,
    ItemStarted,
    ItemUpdated,
    ItemCompleted,
    Error,
}

/// Flat event struct: one shape for all kinds keeps the FFI surface trivial.
#[derive(Debug, Clone)]
pub struct CodexExecEvent {
    pub kind: CodexEventKind,
    pub thread_id: Option<String>,
    /// `turn.completed` only. Normalized (semconv-inclusive): Codex reports
    /// `input_tokens` with the cached part already inside it, the OpenAI
    /// convention, so nothing is added back in.
    pub usage: Option<TokenUsage>,
    /// `turn.failed`'s `error.message`, or a top-level `error`'s `message`.
    pub message: Option<String>,
    pub item_id: Option<String>,
    pub item_type: Option<String>,
    /// The item's text for the types that have one: `agent_message`,
    /// `reasoning` (a summary, never the raw chain of thought) and `error`.
    pub text: Option<String>,
    /// The whole item object as JSON, for the caller to shape into io.
    pub item_json: Option<Vec<u8>>,
}

impl CodexExecEvent {
    fn new(kind: CodexEventKind) -> Self {
        Self {
            kind,
            thread_id: None,
            usage: None,
            message: None,
            item_id: None,
            item_type: None,
            text: None,
            item_json: None,
        }
    }
}

fn s(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(str::to_owned)
}

fn i(v: &Value, key: &str) -> Option<i64> {
    v.get(key).and_then(Value::as_i64)
}

fn parse_usage(v: &Value) -> Option<TokenUsage> {
    let u = v.get("usage")?;
    Some(TokenUsage::new(
        InputConvention::Inclusive,
        i(u, "input_tokens"),
        i(u, "output_tokens"),
        i(u, "cached_input_tokens"),
        i(u, "cache_write_input_tokens"),
        i(u, "reasoning_output_tokens"),
    ))
}

fn item_event(kind: CodexEventKind, v: &Value) -> Option<CodexExecEvent> {
    let item = v.get("item")?;
    let mut e = CodexExecEvent::new(kind);
    e.item_id = s(item, "id");
    e.item_type = s(item, "type");
    e.item_type.as_ref()?;
    if matches!(
        e.item_type.as_deref(),
        Some("agent_message") | Some("reasoning")
    ) {
        e.text = s(item, "text");
    } else if e.item_type.as_deref() == Some("error") {
        e.text = s(item, "message");
    }
    e.item_json = serde_json::to_vec(item).ok();
    Some(e)
}

pub fn parse_exec_line(line: &[u8]) -> Option<CodexExecEvent> {
    let v: Value = serde_json::from_slice(line).ok()?;
    let typ = v.get("type")?.as_str()?;
    match typ {
        "thread.started" => {
            let mut e = CodexExecEvent::new(CodexEventKind::ThreadStarted);
            e.thread_id = s(&v, "thread_id");
            Some(e)
        }
        "turn.started" => Some(CodexExecEvent::new(CodexEventKind::TurnStarted)),
        "turn.completed" => {
            let mut e = CodexExecEvent::new(CodexEventKind::TurnCompleted);
            e.usage = parse_usage(&v);
            Some(e)
        }
        "turn.failed" => {
            let mut e = CodexExecEvent::new(CodexEventKind::TurnFailed);
            e.message = v.get("error").and_then(|err| s(err, "message"));
            Some(e)
        }
        "item.started" => item_event(CodexEventKind::ItemStarted, &v),
        "item.updated" => item_event(CodexEventKind::ItemUpdated, &v),
        "item.completed" => item_event(CodexEventKind::ItemCompleted, &v),
        "error" => {
            let mut e = CodexExecEvent::new(CodexEventKind::Error);
            e.message = s(&v, "message");
            Some(e)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Lines as `codex-cli 0.160.0` wrote them (`codex exec --json`), ids kept.
    const THREAD: &[u8] =
        br#"{"type":"thread.started","thread_id":"01a1206e-068c-7bb1-914f-002bac8a3211"}"#;
    const TURN: &[u8] = br#"{"type":"turn.started"}"#;
    const CMD_STARTED: &[u8] = br#"{"type":"item.started","item":{"id":"item_0","type":"command_execution","command":"/bin/zsh -lc 'echo wardex-fixture'","aggregated_output":"","exit_code":null,"status":"in_progress"}}"#;
    const CMD_DONE: &[u8] = br#"{"type":"item.completed","item":{"id":"item_0","type":"command_execution","command":"/bin/zsh -lc 'echo wardex-fixture'","aggregated_output":"wardex-fixture\n","exit_code":0,"status":"completed"}}"#;
    const MESSAGE: &[u8] = br#"{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"wardex-fixture"}}"#;
    const DONE: &[u8] = br#"{"type":"turn.completed","usage":{"input_tokens":23130,"cached_input_tokens":20992,"cache_write_input_tokens":0,"output_tokens":44,"reasoning_output_tokens":0}}"#;

    #[test]
    fn parses_thread_and_turn_start() {
        let e = parse_exec_line(THREAD).unwrap();
        assert_eq!(e.kind, CodexEventKind::ThreadStarted);
        assert_eq!(
            e.thread_id.as_deref(),
            Some("01a1206e-068c-7bb1-914f-002bac8a3211")
        );
        assert_eq!(
            parse_exec_line(TURN).unwrap().kind,
            CodexEventKind::TurnStarted
        );
    }

    /// Codex's `input_tokens` already contains the cached part (23130 of which
    /// 20992 cached): the OpenAI convention, so the total is NOT added to.
    #[test]
    fn turn_usage_is_inclusive_as_reported() {
        let e = parse_exec_line(DONE).unwrap();
        assert_eq!(e.kind, CodexEventKind::TurnCompleted);
        let u = e.usage.unwrap();
        assert_eq!(u.input_tokens(), Some(23130));
        assert_eq!(u.cache_read_input_tokens(), Some(20992));
        assert_eq!(u.cache_creation_input_tokens(), Some(0));
        assert_eq!(u.output_tokens(), Some(44));
        assert_eq!(u.reasoning_output_tokens(), Some(0));
        assert!(!u.totals_unpaired());
    }

    #[test]
    fn items_carry_type_id_and_raw_json() {
        let started = parse_exec_line(CMD_STARTED).unwrap();
        assert_eq!(started.kind, CodexEventKind::ItemStarted);
        assert_eq!(started.item_type.as_deref(), Some("command_execution"));
        let done = parse_exec_line(CMD_DONE).unwrap();
        assert_eq!(done.kind, CodexEventKind::ItemCompleted);
        assert_eq!(done.item_id.as_deref(), Some("item_0"));
        let raw: Value = serde_json::from_slice(done.item_json.as_ref().unwrap()).unwrap();
        assert_eq!(raw["exit_code"], 0);
        assert!(done.text.is_none());
    }

    #[test]
    fn an_agent_message_carries_its_text() {
        let e = parse_exec_line(MESSAGE).unwrap();
        assert_eq!(e.item_type.as_deref(), Some("agent_message"));
        assert_eq!(e.text.as_deref(), Some("wardex-fixture"));
    }

    #[test]
    fn failures_carry_their_message() {
        let failed =
            parse_exec_line(br#"{"type":"turn.failed","error":{"message":"usage limit"}}"#)
                .unwrap();
        assert_eq!(failed.kind, CodexEventKind::TurnFailed);
        assert_eq!(failed.message.as_deref(), Some("usage limit"));
        let err = parse_exec_line(br#"{"type":"error","message":"reconnecting"}"#).unwrap();
        assert_eq!(err.kind, CodexEventKind::Error);
        assert_eq!(err.message.as_deref(), Some("reconnecting"));
        let item =
            parse_exec_line(br#"{"type":"item.completed","item":{"id":"item_2","type":"error","message":"model rerouted: a -> b"}}"#)
                .unwrap();
        assert_eq!(item.text.as_deref(), Some("model rerouted: a -> b"));
    }

    /// Forward compatibility: what this parser does not know is not an error.
    #[test]
    fn unknown_lines_are_none_and_unknown_items_still_parse() {
        assert!(parse_exec_line(br#"{"type":"thread.archived"}"#).is_none());
        assert!(parse_exec_line(b"not json").is_none());
        assert!(parse_exec_line(br#"{"no_type":1}"#).is_none());
        assert!(parse_exec_line(br#"{"type":"item.completed"}"#).is_none());
        let e = parse_exec_line(
            br#"{"type":"item.completed","item":{"id":"item_9","type":"collab_tool_call"}}"#,
        )
        .unwrap();
        assert_eq!(e.item_type.as_deref(), Some("collab_tool_call"));
        assert!(e.item_json.is_some());
    }

    #[test]
    fn a_turn_without_usage_has_none() {
        let e = parse_exec_line(br#"{"type":"turn.completed"}"#).unwrap();
        assert!(e.usage.is_none());
    }
}
