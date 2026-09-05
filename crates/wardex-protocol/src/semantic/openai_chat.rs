//! OpenAI Chat Completions: request/response fill and SSE reassembly.

use super::parts::*;
use super::usage::{flatten_usage, UsageBounds, UsageView};
use super::{LlmSemantics, StringOrVec};
use crate::sse::SseEvent;
use crate::usage::{InputConvention, TokenUsage};
use serde::Deserialize;

// Normalization table: LlmSemantics usage field <- dotted path in the raw
// usage Value. The extraction below reads THROUGH these constants, so the
// table a test walks and the path the code reads are one string (U4).
const P_INPUT: &str = "prompt_tokens";
const P_OUTPUT: &str = "completion_tokens";
const P_CACHE_READ: &str = "prompt_tokens_details.cached_tokens";
const P_REASONING: &str = "completion_tokens_details.reasoning_tokens";
// Read by the fixture test (`every_normalized_usage_path_is_extracted_
// from_its_fixture`), not by the runtime path — the runtime reads the
// P_* consts the table is built from, which makes the two one string.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) const NORMALIZED_USAGE_PATHS: &[(&str, &str)] = &[
    ("input_tokens", P_INPUT),
    ("output_tokens", P_OUTPUT),
    ("cache_read_input_tokens", P_CACHE_READ),
    ("reasoning_output_tokens", P_REASONING),
];

// --- OpenAI structs ---

#[derive(Deserialize)]
struct OAFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}
#[derive(Deserialize)]
struct OAToolCall {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OAFunction>,
}
#[derive(Deserialize, Default)]
struct OAMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    refusal: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OAToolCall>>,
    #[serde(default)]
    audio: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct OAChoice {
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    message: Option<OAMessage>,
}
#[derive(Deserialize)]
struct OpenAIChatResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<OAChoice>,
    #[serde(default)]
    usage: Option<serde_json::Value>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    system_fingerprint: Option<String>,
}
#[derive(Deserialize)]
struct OpenAIChatRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    max_tokens: Option<i64>,
    #[serde(default)]
    max_completion_tokens: Option<i64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    frequency_penalty: Option<f64>,
    #[serde(default)]
    presence_penalty: Option<f64>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    n: Option<i64>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    stop: Option<StringOrVec>,
    #[serde(default)]
    messages: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    reasoning_effort: Option<String>,
    #[serde(default)]
    response_format: Option<serde_json::Value>,
}

/// OpenAI chat request messages[] → system_instructions + input.messages.
fn fill_openai_input(out: &mut LlmSemantics, messages: &[serde_json::Value]) {
    let mut sys_parts: Vec<serde_json::Value> = Vec::new();
    let mut msgs: Vec<InMsg> = Vec::new();
    for m in messages {
        let role = m.get("role").and_then(|x| x.as_str()).unwrap_or("");
        let content = m.get("content");
        match role {
            "system" | "developer" => {
                if let Some(c) = content.and_then(|x| x.as_str()) {
                    sys_parts.push(text_part(c.to_string()));
                }
            }
            "tool" => {
                let id = m
                    .get("tool_call_id")
                    .and_then(|x| x.as_str())
                    .map(str::to_string);
                let response = content.cloned().unwrap_or(serde_json::Value::Null);
                msgs.push(InMsg {
                    role: "tool".to_string(),
                    parts: vec![tool_call_response_part(id, response)],
                });
            }
            "user" | "assistant" => {
                let mut parts = openai_content_to_parts(content, out);
                // assistant past tool_calls → ToolCallRequestPart
                if let Some(tcs) = m.get("tool_calls").and_then(|x| x.as_array()) {
                    for tc in tcs {
                        let id = tc.get("id").and_then(|x| x.as_str()).map(str::to_string);
                        let f = tc.get("function");
                        let name = f
                            .and_then(|x| x.get("name"))
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string();
                        let raw = f
                            .and_then(|x| x.get("arguments"))
                            .and_then(|x| x.as_str())
                            .unwrap_or("");
                        let arguments = serde_json::from_str::<serde_json::Value>(raw)
                            .unwrap_or_else(|_| serde_json::Value::from(raw));
                        parts.push(tool_call_part(id, name, arguments));
                    }
                }
                msgs.push(InMsg {
                    role: role.to_string(),
                    parts,
                });
            }
            _ => {}
        }
    }
    out.system_instructions = build_system_instructions(sys_parts);
    out.input_messages = build_input_messages(msgs);
}

/// OpenAI message content (string | array) → part list. Handles the media blocks
/// alongside text: `image_url`, `input_audio` and `file`.
fn openai_content_to_parts(
    content: Option<&serde_json::Value>,
    out: &mut LlmSemantics,
) -> Vec<serde_json::Value> {
    let mut parts: Vec<serde_json::Value> = Vec::new();
    match content {
        Some(serde_json::Value::String(s)) if !s.is_empty() => {
            parts.push(text_part(s.clone()));
        }
        Some(serde_json::Value::Array(blocks)) => {
            for b in blocks {
                let bt = b.get("type").and_then(|x| x.as_str()).unwrap_or("");
                match bt {
                    "text" => {
                        let t = b.get("text").and_then(|x| x.as_str()).unwrap_or("");
                        parts.push(text_part(t.to_string()));
                    }
                    "image_url" => {
                        let url = b
                            .get("image_url")
                            .and_then(|x| x.get("url"))
                            .and_then(|x| x.as_str())
                            .unwrap_or("");
                        if let Some((mime, b64)) = parse_data_url(url) {
                            parts.push(blob_part(modality_from_mime(mime), Some(mime), b64));
                        } else if !url.is_empty() {
                            parts.push(uri_part("image", None, url));
                        }
                    }
                    "input_audio" => {
                        let audio = b.get("input_audio");
                        let data = audio
                            .and_then(|x| x.get("data"))
                            .and_then(|x| x.as_str())
                            .unwrap_or("");
                        let fmt = audio
                            .and_then(|x| x.get("format"))
                            .and_then(|x| x.as_str())
                            .unwrap_or("");
                        let mime = format!("audio/{fmt}");
                        parts.push(blob_part("audio", Some(&mime), data));
                    }
                    "file" => {
                        let file_id = b
                            .get("file")
                            .and_then(|x| x.get("file_id"))
                            .and_then(|x| x.as_str())
                            .unwrap_or("");
                        parts.push(file_part("document", None, file_id));
                    }
                    other => {
                        parts.push(generic_part(other));
                        out.input_messages_has_unmapped = true;
                    }
                }
            }
        }
        _ => {}
    }
    parts
}

pub(super) fn fill_openai_chat(
    out: &mut LlmSemantics,
    req: &[u8],
    resp: &[u8],
    bounds: UsageBounds,
) {
    out.api_type = Some("chat_completions");
    // The pre-split dispatcher stamped `output_type="text"` on every parse;
    // the request half below refines it from `response_format.type`.
    out.output_type = Some("text".to_string());
    if let Ok(r) = serde_json::from_slice::<OpenAIChatResponse>(resp) {
        out.response_id = r.id;
        out.response_model = r.model;
        let mut fr: Vec<String> = Vec::new();
        let mut msgs: Vec<OutMsg> = Vec::new();
        for c in r.choices {
            // One producer for both carriers: the normalized value goes into
            // `finish_reasons` AND onto the message (unknown raw passes
            // through as itself — total function, never a silent drop).
            let finish_reason = c
                .finish_reason
                .as_deref()
                .map(|f| normalize_finish_reason("openai", f));
            if let Some(f) = &finish_reason {
                fr.push(f.clone());
            }
            let mut parts: Vec<serde_json::Value> = Vec::new();
            if let Some(msg) = c.message {
                if let Some(content) = msg.content {
                    if !content.is_empty() {
                        parts.push(text_part(content));
                    }
                }
                if let Some(refusal) = msg.refusal {
                    if !refusal.is_empty() {
                        parts.push(text_part(refusal));
                    }
                }
                if let Some(tcs) = msg.tool_calls {
                    for tc in tcs {
                        if let Some(func) = tc.function {
                            let name = func.name.unwrap_or_default();
                            let raw = func.arguments.unwrap_or_default();
                            let arguments = serde_json::from_str::<serde_json::Value>(&raw)
                                .unwrap_or_else(|_| {
                                    out.tool_args_unparsed = true;
                                    serde_json::Value::from(raw)
                                });
                            parts.push(tool_call_part(tc.id, name, arguments));
                        }
                    }
                }
                if let Some(audio) = msg.audio {
                    if let Some(data) = audio.get("data").and_then(|x| x.as_str()) {
                        parts.push(blob_part("audio", None, data));
                    }
                }
            }
            msgs.push(OutMsg {
                role: None,
                parts,
                finish_reason,
            });
        }
        if !fr.is_empty() {
            out.finish_reasons = Some(fr);
        }
        if out.output_messages.is_none() {
            out.output_messages = build_output_messages(msgs);
        }
        if let Some(u) = r.usage {
            // OpenAI `prompt_tokens` already contains `cached_tokens`, and
            // `completion_tokens` already contains the reasoning tokens.
            let view = UsageView(&u);
            out.usage = TokenUsage::new(
                InputConvention::Inclusive,
                view.i64_at(P_INPUT),
                view.i64_at(P_OUTPUT),
                view.i64_at(P_CACHE_READ),
                None,
                view.i64_at(P_REASONING),
            );
            let flat = flatten_usage(&u, bounds);
            out.usage_leaves = flat.leaves;
            out.usage_dropped_count = flat.dropped;
        }
        if let Some(v) = r.service_tier {
            out.response_service_tier = Some(v);
        }
        if let Some(v) = r.system_fingerprint {
            out.system_fingerprint = Some(v);
        }
    }
    if let Ok(q) = serde_json::from_slice::<OpenAIChatRequest>(req) {
        out.request_model = q.model;
        out.temperature = q.temperature;
        out.max_tokens = q.max_tokens.or(q.max_completion_tokens);
        out.top_p = q.top_p;
        out.frequency_penalty = q.frequency_penalty;
        out.presence_penalty = q.presence_penalty;
        out.seed = q.seed;
        out.choice_count = q.n;
        out.stream = q.stream;
        out.stop_sequences = q.stop.map(|s| s.into_vec());
        if let Some(v) = q.service_tier {
            out.request_service_tier = Some(v);
        }
        if let Some(v) = q.reasoning_effort {
            out.reasoning_level = Some(v);
        }
        // `response_format.type`: json_schema/json_object -> "json", anything
        // else (or absent) -> "text" — the same rule the Responses parser
        // applies to `text.format.type`.
        out.output_type = Some(output_type_from_format(q.response_format.as_ref()).to_string());
        if let Some(messages) = q.messages {
            fill_openai_input(out, &messages);
        }
    }
}

/// `response_format.type` / `text.format.type` -> `gen_ai.output.type`.
pub(super) fn output_type_from_format(format: Option<&serde_json::Value>) -> &'static str {
    let ty = format
        .and_then(|f| f.get("type"))
        .and_then(|t| t.as_str())
        .unwrap_or("text");
    match ty {
        "json_schema" | "json_object" => "json",
        _ => "text",
    }
}

/// OpenAI chat stream chunks → synthetic non-streaming-shaped JSON.
/// Accumulates tool_call deltas into a BTreeMap keyed by index, concatenating the arguments string,
/// then emits them as the choices[0].message.tool_calls array → fill_openai_chat reuses it for extraction.
pub(super) fn reassemble_openai(events: &[SseEvent]) -> Reassembled {
    use std::collections::BTreeMap;
    let mut id: Option<String> = None;
    let mut model: Option<String> = None;
    let mut content = String::new();
    let mut finish: Option<String> = None;
    let mut usage: Option<serde_json::Value> = None;
    // index -> (id, name, arguments(accumulated))
    let mut tool_calls: BTreeMap<i64, (Option<String>, Option<String>, String)> = BTreeMap::new();
    // Terminal = `[DONE]` OR any chunk with a non-null finish_reason: an
    // OpenAI-compatible gateway that omits `[DONE]` must not read as an
    // unterminated stream when its last chunk said why it stopped.
    let mut terminated = false;
    for ev in events {
        if ev.data.trim() == "[DONE]" {
            terminated = true;
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(&ev.data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if id.is_none() {
            id = v.get("id").and_then(|x| x.as_str()).map(str::to_string);
        }
        if model.is_none() {
            model = v.get("model").and_then(|x| x.as_str()).map(str::to_string);
        }
        if let Some(c0) = v
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
        {
            if let Some(t) = c0
                .get("delta")
                .and_then(|d| d.get("content"))
                .and_then(|x| x.as_str())
            {
                content.push_str(t);
            }
            if let Some(tcs) = c0
                .get("delta")
                .and_then(|d| d.get("tool_calls"))
                .and_then(|x| x.as_array())
            {
                for tc in tcs {
                    let idx = tc.get("index").and_then(|x| x.as_i64()).unwrap_or(0);
                    let entry = tool_calls.entry(idx).or_default();
                    if entry.0.is_none() {
                        if let Some(i) = tc.get("id").and_then(|x| x.as_str()) {
                            entry.0 = Some(i.to_string());
                        }
                    }
                    if let Some(f) = tc.get("function") {
                        if entry.1.is_none() {
                            if let Some(n) = f.get("name").and_then(|x| x.as_str()) {
                                entry.1 = Some(n.to_string());
                            }
                        }
                        if let Some(a) = f.get("arguments").and_then(|x| x.as_str()) {
                            entry.2.push_str(a);
                        }
                    }
                }
            }
            if let Some(fr) = c0.get("finish_reason").and_then(|x| x.as_str()) {
                finish = Some(fr.to_string());
                terminated = true;
            }
        }
        if let Some(u) = v.get("usage") {
            if !u.is_null() {
                usage = Some(u.clone());
            }
        }
    }
    let mut message = serde_json::json!({"role": "assistant", "content": content});
    if !tool_calls.is_empty() {
        let arr: Vec<serde_json::Value> = tool_calls
            .into_iter()
            .map(|(_, (tc_id, name, args))| {
                serde_json::json!({
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                })
            })
            .collect();
        message["tool_calls"] = serde_json::Value::Array(arr);
    }
    let mut obj = serde_json::json!({
        "id": id,
        "model": model,
        "choices": [{
            "message": message,
            "finish_reason": finish,
        }],
    });
    if let Some(u) = usage {
        obj["usage"] = u;
    }
    Reassembled {
        body: serde_json::to_vec(&obj).unwrap_or_default(),
        terminated,
    }
}
