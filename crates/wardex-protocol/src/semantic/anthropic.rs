//! Anthropic Messages: request/response fill and SSE reassembly.

use super::parts::*;
use super::LlmSemantics;
use crate::sse::SseEvent;
use crate::usage::{InputConvention, TokenUsage};
use serde::Deserialize;

// --- Anthropic structs ---

#[derive(Deserialize, Default)]
struct AnthUsage {
    #[serde(default)]
    input_tokens: Option<i64>,
    #[serde(default)]
    output_tokens: Option<i64>,
    #[serde(default)]
    cache_read_input_tokens: Option<i64>,
    #[serde(default)]
    cache_creation_input_tokens: Option<i64>,
}
#[derive(Deserialize)]
struct AnthropicResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<AnthUsage>,
    #[serde(default)]
    content: Option<Vec<serde_json::Value>>,
}
#[derive(Deserialize)]
struct AnthropicRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    max_tokens: Option<i64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    top_k: Option<f64>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    system: Option<serde_json::Value>,
    #[serde(default)]
    messages: Option<Vec<serde_json::Value>>,
}

/// Anthropic request system + messages[] → system_instructions + input.messages.
pub(super) fn fill_anthropic_input(
    out: &mut LlmSemantics,
    system: Option<&serde_json::Value>,
    messages: &[serde_json::Value],
) {
    let mut sys_parts: Vec<serde_json::Value> = Vec::new();
    match system {
        Some(serde_json::Value::String(s)) => sys_parts.push(text_part(s.clone())),
        Some(serde_json::Value::Array(blocks)) => {
            for b in blocks {
                if let Some(t) = b.get("text").and_then(|x| x.as_str()) {
                    sys_parts.push(text_part(t.to_string()));
                }
            }
        }
        _ => {}
    }
    let mut msgs: Vec<InMsg> = Vec::new();
    for m in messages {
        let role = m.get("role").and_then(|x| x.as_str()).unwrap_or("");
        let parts = anthropic_content_to_parts(m.get("content"), out);
        msgs.push(InMsg {
            role: role.to_string(),
            parts,
        });
    }
    out.system_instructions = build_system_instructions(sys_parts);
    out.input_messages = build_input_messages(msgs);
}

/// Anthropic message content (string | array) → part list.
fn anthropic_content_to_parts(
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
                    "tool_use" => {
                        let id = b.get("id").and_then(|x| x.as_str()).map(str::to_string);
                        let name = b
                            .get("name")
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string();
                        let arguments = b.get("input").cloned().unwrap_or(serde_json::Value::Null);
                        parts.push(tool_call_part(id, name, arguments));
                    }
                    "tool_result" => {
                        let id = b
                            .get("tool_use_id")
                            .and_then(|x| x.as_str())
                            .map(str::to_string);
                        let response = b.get("content").cloned().unwrap_or(serde_json::Value::Null);
                        parts.push(tool_call_response_part(id, response));
                    }
                    "image" | "document" => {
                        let src = b.get("source");
                        let st = src
                            .and_then(|x| x.get("type"))
                            .and_then(|x| x.as_str())
                            .unwrap_or("");
                        if st == "base64" {
                            let mime = src
                                .and_then(|x| x.get("media_type"))
                                .and_then(|x| x.as_str())
                                .unwrap_or("");
                            let data = src
                                .and_then(|x| x.get("data"))
                                .and_then(|x| x.as_str())
                                .unwrap_or("");
                            parts.push(blob_part(modality_from_mime(mime), Some(mime), data));
                        } else if st == "url" {
                            let url = src
                                .and_then(|x| x.get("url"))
                                .and_then(|x| x.as_str())
                                .unwrap_or("");
                            let modality = if bt == "image" { "image" } else { "document" };
                            parts.push(uri_part(modality, None, url));
                        } else {
                            parts.push(generic_part(bt));
                            out.input_messages_has_unmapped = true;
                        }
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

pub(super) fn fill_anthropic(out: &mut LlmSemantics, req: &[u8], resp: &[u8]) {
    if let Ok(r) = serde_json::from_slice::<AnthropicResponse>(resp) {
        out.response_id = r.id;
        out.response_model = r.model;
        let finish_reason = r
            .stop_reason
            .as_deref()
            .and_then(|sr| finish_reason_to_otel("anthropic", sr));
        if let Some(sr) = r.stop_reason {
            out.finish_reasons = Some(vec![sr]);
        }
        if let Some(u) = r.usage {
            // Anthropic reports the cache tiers OUTSIDE `input_tokens`; the
            // semconv Anthropic provider doc requires the inclusive sum, and
            // `ExcludesCache` is that requirement made unskippable.
            out.usage = TokenUsage::new(
                InputConvention::ExcludesCache,
                u.input_tokens,
                u.output_tokens,
                u.cache_read_input_tokens,
                u.cache_creation_input_tokens,
                None,
            );
        }
        if out.output_messages.is_none() {
            let mut parts: Vec<serde_json::Value> = Vec::new();
            if let Some(blocks) = r.content {
                for b in blocks {
                    let bt = b.get("type").and_then(|x| x.as_str()).unwrap_or("");
                    match bt {
                        "text" => {
                            let t = b.get("text").and_then(|x| x.as_str()).unwrap_or("");
                            if !t.is_empty() {
                                parts.push(text_part(t.to_string()));
                            }
                        }
                        "thinking" => {
                            let t = b.get("thinking").and_then(|x| x.as_str()).unwrap_or("");
                            parts.push(reasoning_part(t.to_string()));
                        }
                        "redacted_thinking" => {
                            parts.push(generic_part("reasoning"));
                        }
                        "tool_use" => {
                            let id = b.get("id").and_then(|x| x.as_str()).map(str::to_string);
                            let name = b
                                .get("name")
                                .and_then(|x| x.as_str())
                                .unwrap_or("")
                                .to_string();
                            let arguments =
                                b.get("input").cloned().unwrap_or(serde_json::Value::Null);
                            parts.push(tool_call_part(id, name, arguments));
                        }
                        "server_tool_use" => {
                            let id = b.get("id").and_then(|x| x.as_str()).map(str::to_string);
                            let name = b
                                .get("name")
                                .and_then(|x| x.as_str())
                                .unwrap_or("")
                                .to_string();
                            let call = b.get("input").cloned().unwrap_or(serde_json::Value::Null);
                            parts.push(server_tool_call_part(id, name, call));
                        }
                        "web_search_tool_result" => {
                            let id = b
                                .get("tool_use_id")
                                .and_then(|x| x.as_str())
                                .map(str::to_string);
                            let response =
                                b.get("content").cloned().unwrap_or(serde_json::Value::Null);
                            parts.push(server_tool_call_response_part(id, response));
                        }
                        other => {
                            parts.push(generic_part(other));
                            out.output_messages_has_unmapped = true;
                        }
                    }
                }
            }
            out.output_messages = build_output_messages(vec![OutMsg {
                parts,
                finish_reason,
            }]);
        }
    }
    if let Ok(q) = serde_json::from_slice::<AnthropicRequest>(req) {
        out.request_model = q.model;
        out.temperature = q.temperature;
        out.max_tokens = q.max_tokens;
        out.top_p = q.top_p;
        out.top_k = q.top_k;
        out.stream = q.stream;
        out.stop_sequences = q.stop_sequences;
        if let Some(messages) = q.messages {
            fill_anthropic_input(out, q.system.as_ref(), &messages);
        } else if let Some(system) = q.system.as_ref() {
            fill_anthropic_input(out, Some(system), &[]);
        }
    }
}

/// Anthropic messages stream events → synthetic non-streaming-shaped JSON.
pub(super) fn reassemble_anthropic(events: &[SseEvent]) -> Vec<u8> {
    use std::collections::BTreeMap;
    let mut id: Option<String> = None;
    let mut model: Option<String> = None;
    let mut content = String::new();
    let mut thinking = String::new();
    let mut stop_reason: Option<String> = None;
    let mut input_tokens: Option<i64> = None;
    let mut output_tokens: Option<i64> = None;
    let mut cache_read: Option<i64> = None;
    let mut cache_creation: Option<i64> = None;
    // index -> (id, name, partial_json accumulated)
    let mut tool_uses: BTreeMap<i64, (Option<String>, Option<String>, String)> = BTreeMap::new();
    // server_tool_use: same shape as tool_use (input_json_delta accumulated)
    let mut server_tool_uses: BTreeMap<i64, (Option<String>, Option<String>, String)> =
        BTreeMap::new();
    // web_search_tool_result: received fully-formed in the start event, no deltas
    let mut web_search_results: BTreeMap<i64, serde_json::Value> = BTreeMap::new();
    for ev in events {
        let v: serde_json::Value = match serde_json::from_str(&ev.data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let typ = ev
            .event
            .as_deref()
            .or_else(|| v.get("type").and_then(|x| x.as_str()));
        match typ {
            Some("message_start") => {
                if let Some(m) = v.get("message") {
                    id = m.get("id").and_then(|x| x.as_str()).map(str::to_string);
                    model = m.get("model").and_then(|x| x.as_str()).map(str::to_string);
                    if let Some(u) = m.get("usage") {
                        input_tokens = u.get("input_tokens").and_then(|x| x.as_i64());
                        output_tokens = u.get("output_tokens").and_then(|x| x.as_i64());
                        cache_read = u.get("cache_read_input_tokens").and_then(|x| x.as_i64());
                        cache_creation = u
                            .get("cache_creation_input_tokens")
                            .and_then(|x| x.as_i64());
                    }
                }
            }
            Some("content_block_start") => {
                let idx = v.get("index").and_then(|x| x.as_i64()).unwrap_or(0);
                if let Some(cb) = v.get("content_block") {
                    match cb.get("type").and_then(|x| x.as_str()) {
                        Some("tool_use") => {
                            let e = tool_uses.entry(idx).or_default();
                            e.0 = cb.get("id").and_then(|x| x.as_str()).map(str::to_string);
                            e.1 = cb.get("name").and_then(|x| x.as_str()).map(str::to_string);
                        }
                        Some("server_tool_use") => {
                            // same shape as tool_use: register id+name, wait for input_json_delta accumulation
                            let e = server_tool_uses.entry(idx).or_default();
                            e.0 = cb.get("id").and_then(|x| x.as_str()).map(str::to_string);
                            e.1 = cb.get("name").and_then(|x| x.as_str()).map(str::to_string);
                        }
                        Some("web_search_tool_result") => {
                            // result is fully contained in the start event — no deltas
                            let result = serde_json::json!({
                                "tool_use_id": cb.get("tool_use_id"),
                                "content": cb.get("content"),
                            });
                            web_search_results.insert(idx, result);
                        }
                        _ => {}
                    }
                }
            }
            Some("content_block_delta") => {
                if let Some(t) = v
                    .get("delta")
                    .and_then(|d| d.get("text"))
                    .and_then(|x| x.as_str())
                {
                    content.push_str(t);
                }
                if let Some(tk) = v
                    .get("delta")
                    .and_then(|d| d.get("thinking"))
                    .and_then(|x| x.as_str())
                {
                    thinking.push_str(tk);
                }
                if let Some(pj) = v
                    .get("delta")
                    .and_then(|d| d.get("partial_json"))
                    .and_then(|x| x.as_str())
                {
                    let idx = v.get("index").and_then(|x| x.as_i64()).unwrap_or(0);
                    // Accumulate only into indices registered by content_block_start(tool_use/server_tool_use).
                    // Orphan deltas with no start event (e.g. stream truncation) are silently ignored →
                    // prevents creating a ghost entry.
                    if let Some(entry) = tool_uses.get_mut(&idx) {
                        entry.2.push_str(pj);
                    } else if let Some(entry) = server_tool_uses.get_mut(&idx) {
                        entry.2.push_str(pj);
                    }
                }
            }
            Some("message_delta") => {
                if let Some(sr) = v
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(|x| x.as_str())
                {
                    stop_reason = Some(sr.to_string());
                }
                if let Some(ot) = v
                    .get("usage")
                    .and_then(|u| u.get("output_tokens"))
                    .and_then(|x| x.as_i64())
                {
                    output_tokens = Some(ot);
                }
            }
            _ => {}
        }
    }
    let mut usage = serde_json::json!({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    });
    if let Some(c) = cache_read {
        usage["cache_read_input_tokens"] = c.into();
    }
    if let Some(c) = cache_creation {
        usage["cache_creation_input_tokens"] = c.into();
    }
    // content array: thinking (if present) + text (if non-empty) + tool_use (index order)
    // The current API streams only 1 thinking block; multiple blocks are merged into one and the order is fixed as thinking→text→tool (may differ from the source order of the non-streaming path) — multi-block support is a follow-up.
    let mut content_arr: Vec<serde_json::Value> = Vec::new();
    if !thinking.is_empty() {
        content_arr.push(serde_json::json!({"type": "thinking", "thinking": thinking}));
    }
    if !content.is_empty() {
        content_arr.push(serde_json::json!({"type": "text", "text": content}));
    }
    for (_, (tu_id, name, args)) in tool_uses {
        let input: serde_json::Value = serde_json::from_str(&args)
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        content_arr.push(serde_json::json!({
            "type": "tool_use",
            "id": tu_id,
            "name": name,
            "input": input,
        }));
    }
    // server_tool_use: emitted in the same shape as tool_use → reuses fill_anthropic's server_tool_use branch
    for (_, (stu_id, name, args)) in server_tool_uses {
        let input: serde_json::Value = serde_json::from_str(&args)
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        content_arr.push(serde_json::json!({
            "type": "server_tool_use",
            "id": stu_id,
            "name": name,
            "input": input,
        }));
    }
    // web_search_tool_result: emitted as-is from data collected in the start event
    // → reuses fill_anthropic's web_search_tool_result branch
    for (_, result) in web_search_results {
        content_arr.push(serde_json::json!({
            "type": "web_search_tool_result",
            "tool_use_id": result["tool_use_id"],
            "content": result["content"],
        }));
    }
    let obj = serde_json::json!({
        "id": id,
        "model": model,
        "stop_reason": stop_reason,
        "content": content_arr,
        "usage": usage,
    });
    serde_json::to_vec(&obj).unwrap_or_default()
}
