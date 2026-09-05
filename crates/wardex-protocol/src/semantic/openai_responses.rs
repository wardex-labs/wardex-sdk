//! OpenAI Responses API: request/response fill and SSE reassembly.
//!
//! The default path of the openai-agents SDK (`Runner.run` == one
//! `POST /v1/responses`), invisible before this module existed: non-streaming
//! calls yielded no semantics at all, and streams were re-shaped by the Chat
//! reassembler into a fabricated empty chat body.
//!
//! Streaming truth (design §3.1): a terminal event
//! (`response.completed|incomplete|failed`) carries the COMPLETE `Response`
//! object, so its snapshot is the body and deltas are only a fallback. An
//! in-stream `error` event is the third terminal form — the only bytes that
//! name the failure — and is preserved into the synthetic body rather than
//! dropped.

use super::parts::*;
use super::usage::{flatten_usage, UsageBounds, UsageView};
use super::LlmSemantics;
use crate::sse::SseEvent;
use crate::usage::{InputConvention, TokenUsage};
use serde::Deserialize;

// Normalization table: LlmSemantics usage field <- dotted path in the raw
// usage Value (semconv openai.md maps cache_read to
// `usage.input_tokens_details.cached_tokens` and reasoning to
// `usage.output_tokens_details.reasoning_tokens` for this API).
const P_INPUT: &str = "input_tokens";
const P_OUTPUT: &str = "output_tokens";
const P_CACHE_READ: &str = "input_tokens_details.cached_tokens";
const P_REASONING: &str = "output_tokens_details.reasoning_tokens";
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

/// Output item types the CLIENT executes — their presence is what turns a
/// `completed` status into `finish_reasons=["tool_call"]`.
const CLIENT_TOOL_TYPES: &[&str] = &[
    "function_call",
    "computer_call",
    "custom_tool_call",
    "local_shell_call",
];

/// Server-executed tool items: mapped to server_tool_call(+response) parts.
const SERVER_TOOL_TYPES: &[&str] = &[
    "web_search_call",
    "file_search_call",
    "code_interpreter_call",
    "image_generation_call",
    "mcp_call",
    "mcp_list_tools",
];

#[derive(Deserialize)]
struct ResponsesRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    input: Option<serde_json::Value>,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    reasoning: Option<serde_json::Value>,
    #[serde(default)]
    max_output_tokens: Option<i64>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    previous_response_id: Option<String>,
    #[serde(default)]
    text: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ResponsesResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    incomplete_details: Option<serde_json::Value>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    output: Vec<serde_json::Value>,
    #[serde(default)]
    usage: Option<serde_json::Value>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

pub(super) fn fill_openai_responses(
    out: &mut LlmSemantics,
    req: &[u8],
    resp: &[u8],
    bounds: UsageBounds,
) {
    if let Ok(r) = serde_json::from_slice::<ResponsesResponse>(resp) {
        // A Response object, or a bare error envelope? Every field here is
        // `#[serde(default)]`, so the plain HTTP-error body `{"error":{...}}`
        // — what every 4xx/5xx from `/v1/responses` carries — deserializes
        // too, with no id, no status and no output. A response half derived
        // from it would be fabricated: `status="failed"`,
        // `finish_reasons=["error"]` and an empty assistant output message
        // the provider never sent. The sibling endpoints ship an HTTP error
        // with the response half EMPTY (`test_llm_error_spans` pins it), and
        // this endpoint does the same. The SSE synthetic body and every real
        // terminal snapshot DO carry id/status/output, so the in-stream
        // `error` event still maps to `failed` below.
        let is_response_object = r.id.is_some() || r.status.is_some() || !r.output.is_empty();
        out.response_id = r.id;
        out.response_model = r.model;
        if let Some(v) = r.service_tier {
            out.response_service_tier = Some(v);
        }
        if let Some(u) = &r.usage {
            let view = UsageView(u);
            out.usage = TokenUsage::new(
                InputConvention::Inclusive,
                view.i64_at(P_INPUT),
                view.i64_at(P_OUTPUT),
                view.i64_at(P_CACHE_READ),
                None,
                view.i64_at(P_REASONING),
            );
            let flat = flatten_usage(u, bounds);
            out.usage_leaves = flat.leaves;
            out.usage_dropped_count = flat.dropped;
        }
        // Output items -> ONE assistant OutMsg, item order preserved. A
        // `message` item carrying another role (a compaction echoes the
        // caller's own messages) becomes its own OutMsg in `pre` instead:
        // the model never said those words.
        let mut parts: Vec<serde_json::Value> = Vec::new();
        let mut pre: Vec<OutMsg> = Vec::new();
        let mut has_client_tool_call = false;
        for item in &r.output {
            let ty = item.get("type").and_then(|x| x.as_str()).unwrap_or("");
            if CLIENT_TOOL_TYPES.contains(&ty) {
                has_client_tool_call = true;
            }
            output_item_parts(item, ty, out, &mut parts, &mut pre);
        }
        // status -> finish_reasons, all through the one normalizer. An
        // `error` object is the provider's own failure declaration and wins
        // over whatever raw status the (possibly synthetic) body carried —
        // but only on a body that IS a Response object (see above): a bare
        // error envelope gets no response half at all.
        if is_response_object {
            let finish: Option<String> = if r.error.is_some() {
                out.response_status = Some("failed".to_string());
                Some(normalize_finish_reason("openai", "failed"))
            } else {
                out.response_status = r.status.clone();
                match r.status.as_deref() {
                    Some("completed") => Some(if has_client_tool_call {
                        normalize_finish_reason("openai", "tool_calls")
                    } else {
                        normalize_finish_reason("openai", "stop")
                    }),
                    Some("incomplete") => r
                        .incomplete_details
                        .as_ref()
                        .and_then(|d| d.get("reason"))
                        .and_then(|x| x.as_str())
                        .map(|reason| normalize_finish_reason("openai", reason)),
                    Some("failed") | Some("cancelled") => {
                        Some(normalize_finish_reason("openai", "failed"))
                    }
                    _ => None, // queued | in_progress | absent: not finished
                }
            };
            if let Some(f) = &finish {
                out.finish_reasons = Some(vec![f.clone()]);
            }
            if out.output_messages.is_none() {
                // Non-assistant messages first, in item order, then the one
                // assistant message: on the only body that produces them
                // (a compaction) the wire puts the echoed messages before
                // the model's item.
                pre.push(OutMsg {
                    role: None,
                    parts,
                    finish_reason: finish,
                });
                out.output_messages = build_output_messages(pre);
            }
        }
    }
    if let Ok(q) = serde_json::from_slice::<ResponsesRequest>(req) {
        out.request_model = q.model;
        out.temperature = q.temperature;
        out.top_p = q.top_p;
        out.stream = q.stream;
        out.max_tokens = q.max_output_tokens;
        if let Some(v) = q.service_tier {
            out.request_service_tier = Some(v);
        }
        out.previous_response_id = q.previous_response_id;
        if let Some(effort) = q
            .reasoning
            .as_ref()
            .and_then(|c| c.get("effort"))
            .and_then(|e| e.as_str())
        {
            out.reasoning_level = Some(effort.to_string());
        }
        // `text.format.type`: the same rule as Chat's `response_format.type`.
        let format = q.text.as_ref().and_then(|t| t.get("format"));
        out.output_type = Some(super::openai_chat::output_type_from_format(format).to_string());
        fill_responses_input(out, q.instructions, q.input.as_ref());
    }
}

/// One output item -> OTel parts (design §4.4 table). Unknown types become a
/// generic part AND set the unmapped flag — never silently skipped. A
/// `message` item whose `role` is not `assistant` goes to `pre` as its own
/// message rather than into the assistant's `parts`.
fn output_item_parts(
    item: &serde_json::Value,
    ty: &str,
    out: &mut LlmSemantics,
    parts: &mut Vec<serde_json::Value>,
    pre: &mut Vec<OutMsg>,
) {
    let str_of = |field: &str| item.get(field).and_then(|x| x.as_str());
    let call_id = || str_of("call_id").map(str::to_string);
    match ty {
        "message" => {
            let role = str_of("role").unwrap_or("assistant");
            let mut local: Vec<serde_json::Value> = Vec::new();
            if let Some(content) = item.get("content").and_then(|c| c.as_array()) {
                for block in content {
                    match block.get("type").and_then(|x| x.as_str()).unwrap_or("") {
                        "output_text" | "input_text" => {
                            let text = block.get("text").and_then(|x| x.as_str()).unwrap_or("");
                            if !text.is_empty() {
                                local.push(text_part(text.to_string()));
                            }
                        }
                        "refusal" => {
                            let refusal =
                                block.get("refusal").and_then(|x| x.as_str()).unwrap_or("");
                            if !refusal.is_empty() {
                                local.push(text_part(refusal.to_string()));
                            }
                        }
                        other => {
                            local.push(generic_part(other));
                            out.output_messages_has_unmapped = true;
                        }
                    }
                }
            }
            if role == "assistant" {
                parts.extend(local);
            } else {
                pre.push(OutMsg {
                    role: Some(role.to_string()),
                    parts: local,
                    finish_reason: None,
                });
            }
        }
        "function_call" => {
            let name = str_of("name").unwrap_or("").to_string();
            let raw = str_of("arguments").unwrap_or("");
            let arguments = serde_json::from_str::<serde_json::Value>(raw).unwrap_or_else(|_| {
                out.tool_args_unparsed = true;
                serde_json::Value::from(raw)
            });
            parts.push(tool_call_part(call_id(), name, arguments));
        }
        "computer_call" => {
            let action = item
                .get("action")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            parts.push(tool_call_part(call_id(), "computer".to_string(), action));
        }
        "custom_tool_call" => {
            let name = str_of("name").unwrap_or("").to_string();
            // `input` is a free-form string by contract — kept verbatim.
            let input = item
                .get("input")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            parts.push(tool_call_part(call_id(), name, input));
        }
        "local_shell_call" => {
            let action = item
                .get("action")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            parts.push(tool_call_part(call_id(), "local_shell".to_string(), action));
        }
        "reasoning" => {
            let joined = |field: &str| -> String {
                item.get(field)
                    .and_then(|s| s.as_array())
                    .map(|blocks| {
                        blocks
                            .iter()
                            .filter_map(|b| b.get("text").and_then(|x| x.as_str()))
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default()
            };
            let text = {
                let s = joined("summary");
                if s.is_empty() {
                    joined("content")
                } else {
                    s
                }
            };
            if text.is_empty() {
                // Same shape as Anthropic's redacted_thinking: reasoning
                // happened, its text is withheld — a generic part, no flag.
                parts.push(generic_part("reasoning"));
            } else {
                parts.push(reasoning_part(text));
            }
        }
        t if SERVER_TOOL_TYPES.contains(&t) => {
            let id = str_of("id").map(str::to_string);
            let name = str_of("name")
                .map(str::to_string)
                .unwrap_or_else(|| t.strip_suffix("_call").unwrap_or(t).to_string());
            let call = ["action", "queries", "code", "arguments", "input"]
                .iter()
                .find_map(|f| item.get(*f))
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            parts.push(server_tool_call_part(id.clone(), name, call));
            if t == "image_generation_call" {
                if let Some(b64) = str_of("result") {
                    parts.push(blob_part("image", None, b64));
                }
            } else if let Some(result) = ["results", "outputs", "output", "result"]
                .iter()
                .find_map(|f| item.get(*f))
            {
                if !result.is_null() {
                    parts.push(server_tool_call_response_part(id, result.clone()));
                }
            }
        }
        other => {
            // mcp_approval_request, compaction, and whatever ships next.
            parts.push(generic_part(other));
            out.output_messages_has_unmapped = true;
        }
    }
}

/// Request `instructions` + `input` -> system_instructions and input.messages.
fn fill_responses_input(
    out: &mut LlmSemantics,
    instructions: Option<String>,
    input: Option<&serde_json::Value>,
) {
    let mut sys_parts: Vec<serde_json::Value> = Vec::new();
    if let Some(text) = instructions {
        if !text.is_empty() {
            sys_parts.push(text_part(text));
        }
    }
    let mut msgs: Vec<InMsg> = Vec::new();
    match input {
        Some(serde_json::Value::String(s)) if !s.is_empty() => {
            msgs.push(InMsg {
                role: "user".to_string(),
                parts: vec![text_part(s.clone())],
            });
        }
        Some(serde_json::Value::Array(items)) => {
            for item in items {
                input_item_to_msgs(item, out, &mut sys_parts, &mut msgs);
            }
        }
        _ => {}
    }
    out.system_instructions = build_system_instructions(sys_parts);
    out.input_messages = build_input_messages(msgs);
}

fn input_item_to_msgs(
    item: &serde_json::Value,
    out: &mut LlmSemantics,
    sys_parts: &mut Vec<serde_json::Value>,
    msgs: &mut Vec<InMsg>,
) {
    let role = item.get("role").and_then(|x| x.as_str());
    let ty = item.get("type").and_then(|x| x.as_str());
    match (ty, role) {
        // `type` may be omitted for messages ("message" is implied).
        (None | Some("message"), Some("system" | "developer")) => match item.get("content") {
            Some(serde_json::Value::String(s)) => sys_parts.push(text_part(s.clone())),
            Some(serde_json::Value::Array(blocks)) => {
                for b in blocks {
                    if let Some(text) = b.get("text").and_then(|x| x.as_str()) {
                        sys_parts.push(text_part(text.to_string()));
                    }
                }
            }
            _ => {}
        },
        (None | Some("message"), Some(r @ ("user" | "assistant"))) => {
            let parts = responses_content_to_parts(item.get("content"), out);
            msgs.push(InMsg {
                role: r.to_string(),
                parts,
            });
        }
        (Some("function_call"), _) => {
            let id = item
                .get("call_id")
                .and_then(|x| x.as_str())
                .map(str::to_string);
            let name = item
                .get("name")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let raw = item.get("arguments").and_then(|x| x.as_str()).unwrap_or("");
            let arguments = serde_json::from_str::<serde_json::Value>(raw)
                .unwrap_or_else(|_| serde_json::Value::from(raw));
            msgs.push(InMsg {
                role: "assistant".to_string(),
                parts: vec![tool_call_part(id, name, arguments)],
            });
        }
        (Some("function_call_output"), _) => {
            let id = item
                .get("call_id")
                .and_then(|x| x.as_str())
                .map(str::to_string);
            let response = item
                .get("output")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            msgs.push(InMsg {
                role: "tool".to_string(),
                parts: vec![tool_call_response_part(id, response)],
            });
        }
        (Some("reasoning"), _) => {
            // Prior-turn reasoning replayed as history: presence, no flag.
            msgs.push(InMsg {
                role: "assistant".to_string(),
                parts: vec![generic_part("reasoning")],
            });
        }
        (Some("item_reference"), _) => {
            // The content lives on the server; the prompt cannot be fully
            // reconstructed from these bytes — exactly what marker 30 means.
            out.input_messages_has_unmapped = true;
            msgs.push(InMsg {
                role: "assistant".to_string(),
                parts: vec![generic_part("item_reference")],
            });
        }
        (Some(other), _) => {
            out.input_messages_has_unmapped = true;
            msgs.push(InMsg {
                role: role.unwrap_or("assistant").to_string(),
                parts: vec![generic_part(other)],
            });
        }
        (None, _) => {}
    }
}

/// Responses message content (string | array) -> part list.
fn responses_content_to_parts(
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
                    "input_text" | "output_text" => {
                        let t = b.get("text").and_then(|x| x.as_str()).unwrap_or("");
                        parts.push(text_part(t.to_string()));
                    }
                    "refusal" => {
                        let t = b.get("refusal").and_then(|x| x.as_str()).unwrap_or("");
                        parts.push(text_part(t.to_string()));
                    }
                    "input_image" => {
                        let url = b.get("image_url").and_then(|x| x.as_str()).unwrap_or("");
                        if let Some((mime, b64)) = parse_data_url(url) {
                            parts.push(blob_part(modality_from_mime(mime), Some(mime), b64));
                        } else if !url.is_empty() {
                            parts.push(uri_part("image", None, url));
                        } else if let Some(file_id) = b.get("file_id").and_then(|x| x.as_str()) {
                            parts.push(file_part("image", None, file_id));
                        }
                    }
                    "input_file" => {
                        if let Some(file_id) = b.get("file_id").and_then(|x| x.as_str()) {
                            parts.push(file_part("document", None, file_id));
                        } else if let Some(data) = b.get("file_data").and_then(|x| x.as_str()) {
                            parts.push(blob_part("document", None, data));
                        } else if let Some(url) = b.get("file_url").and_then(|x| x.as_str()) {
                            parts.push(uri_part("document", None, url));
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

// --- SSE -------------------------------------------------------------------

pub(super) fn reassemble_responses(events: &[SseEvent]) -> Reassembled {
    use std::collections::BTreeMap;
    // Terminal snapshot: the LAST of completed|incomplete|failed wins.
    let mut terminal: Option<serde_json::Value> = None;
    let mut error: Option<serde_json::Value> = None;
    // Fallback state.
    let mut id: Option<String> = None;
    let mut model: Option<String> = None;
    let mut last_status: Option<String> = None;
    // output_index -> item; `done` beats `added`.
    let mut done_items: BTreeMap<i64, serde_json::Value> = BTreeMap::new();
    let mut added_items: BTreeMap<i64, serde_json::Value> = BTreeMap::new();
    // item_id -> accumulations.
    let mut text_deltas: BTreeMap<(String, i64), String> = BTreeMap::new();
    let mut refusal_deltas: BTreeMap<(String, i64), String> = BTreeMap::new();
    let mut args_deltas: BTreeMap<String, String> = BTreeMap::new();
    let mut summary_deltas: BTreeMap<(String, i64), String> = BTreeMap::new();

    for ev in events {
        let v: serde_json::Value = match serde_json::from_str(&ev.data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ty = ev
            .event
            .as_deref()
            .or_else(|| v.get("type").and_then(|x| x.as_str()))
            .unwrap_or("");
        match ty {
            "response.completed" | "response.incomplete" | "response.failed" => {
                if let Some(r) = v.get("response") {
                    terminal = Some(r.clone());
                }
            }
            "error" => {
                error = Some(v.clone());
            }
            "response.output_item.done" => {
                let idx = v.get("output_index").and_then(|x| x.as_i64()).unwrap_or(0);
                if let Some(item) = v.get("item") {
                    done_items.insert(idx, item.clone());
                }
            }
            "response.output_item.added" => {
                let idx = v.get("output_index").and_then(|x| x.as_i64()).unwrap_or(0);
                if let Some(item) = v.get("item") {
                    added_items.insert(idx, item.clone());
                }
            }
            "response.output_text.delta" => {
                if let (Some(item_id), Some(delta)) = (
                    v.get("item_id").and_then(|x| x.as_str()),
                    v.get("delta").and_then(|x| x.as_str()),
                ) {
                    let ci = v.get("content_index").and_then(|x| x.as_i64()).unwrap_or(0);
                    text_deltas
                        .entry((item_id.to_string(), ci))
                        .or_default()
                        .push_str(delta);
                }
            }
            "response.refusal.delta" => {
                if let (Some(item_id), Some(delta)) = (
                    v.get("item_id").and_then(|x| x.as_str()),
                    v.get("delta").and_then(|x| x.as_str()),
                ) {
                    let ci = v.get("content_index").and_then(|x| x.as_i64()).unwrap_or(0);
                    refusal_deltas
                        .entry((item_id.to_string(), ci))
                        .or_default()
                        .push_str(delta);
                }
            }
            "response.function_call_arguments.delta" => {
                if let (Some(item_id), Some(delta)) = (
                    v.get("item_id").and_then(|x| x.as_str()),
                    v.get("delta").and_then(|x| x.as_str()),
                ) {
                    args_deltas
                        .entry(item_id.to_string())
                        .or_default()
                        .push_str(delta);
                }
            }
            "response.reasoning_summary_text.delta" => {
                if let (Some(item_id), Some(delta)) = (
                    v.get("item_id").and_then(|x| x.as_str()),
                    v.get("delta").and_then(|x| x.as_str()),
                ) {
                    let si = v.get("summary_index").and_then(|x| x.as_i64()).unwrap_or(0);
                    summary_deltas
                        .entry((item_id.to_string(), si))
                        .or_default()
                        .push_str(delta);
                }
            }
            _ => {
                // Any event carrying a `response` object updates the fallback
                // identity and last observed status (`created`, `in_progress`,
                // `queued`, ...).
                if let Some(r) = v.get("response") {
                    if id.is_none() {
                        id = r.get("id").and_then(|x| x.as_str()).map(str::to_string);
                    }
                    if model.is_none() {
                        model = r.get("model").and_then(|x| x.as_str()).map(str::to_string);
                    }
                    if let Some(s) = r.get("status").and_then(|x| x.as_str()) {
                        last_status = Some(s.to_string());
                    }
                }
            }
        }
    }

    // 1. Terminal snapshot IS the body — same source as a non-streaming
    //    response, annotations, status and usage included.
    if let Some(snapshot) = terminal {
        return Reassembled {
            body: serde_json::to_vec(&snapshot).unwrap_or_default(),
            terminated: true,
        };
    }

    // 2/3. Fallback reconstruction, with or without an in-stream error:
    //    `done` items whole; `added`-only items restored from their deltas;
    //    orphan deltas (no `added`) ignored — no ghost items.
    let mut output: Vec<serde_json::Value> = Vec::new();
    let mut indices: Vec<i64> = done_items
        .keys()
        .chain(added_items.keys())
        .copied()
        .collect();
    indices.sort_unstable();
    indices.dedup();
    for idx in indices {
        if let Some(item) = done_items.get(&idx) {
            output.push(item.clone());
            continue;
        }
        let Some(added) = added_items.get(&idx) else {
            continue;
        };
        let mut item = added.clone();
        let item_id = item
            .get("id")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        match item.get("type").and_then(|x| x.as_str()).unwrap_or("") {
            "message" => {
                let mut content: Vec<serde_json::Value> = Vec::new();
                let mut blocks: Vec<(&(String, i64), &String, &str)> = text_deltas
                    .iter()
                    .filter(|((iid, _), _)| *iid == item_id)
                    .map(|(k, v)| (k, v, "output_text"))
                    .chain(
                        refusal_deltas
                            .iter()
                            .filter(|((iid, _), _)| *iid == item_id)
                            .map(|(k, v)| (k, v, "refusal")),
                    )
                    .collect();
                blocks.sort_by_key(|((_, ci), _, _)| *ci);
                for ((_, _), text, kind) in blocks {
                    content.push(if kind == "output_text" {
                        serde_json::json!({"type": "output_text", "text": text, "annotations": []})
                    } else {
                        serde_json::json!({"type": "refusal", "refusal": text})
                    });
                }
                item["content"] = serde_json::Value::Array(content);
            }
            "function_call" => {
                if let Some(args) = args_deltas.get(&item_id) {
                    item["arguments"] = serde_json::Value::from(args.clone());
                }
            }
            "reasoning" => {
                let mut summaries: Vec<(&(String, i64), &String)> = summary_deltas
                    .iter()
                    .filter(|((iid, _), _)| *iid == item_id)
                    .collect();
                summaries.sort_by_key(|((_, si), _)| *si);
                let arr: Vec<serde_json::Value> = summaries
                    .into_iter()
                    .map(|(_, text)| serde_json::json!({"type": "summary_text", "text": text}))
                    .collect();
                item["summary"] = serde_json::Value::Array(arr);
            }
            _ => {}
        }
        output.push(item);
    }

    let mut body = serde_json::json!({
        "id": id,
        "model": model,
        "status": last_status,
        "output": output,
        "usage": null,
    });
    let terminated = error.is_some();
    if let Some(e) = error {
        // The only bytes that name the failure — preserved, not dropped. The
        // fill maps them to `response_status="failed"` / `finish_reasons=
        // ["error"]`: the provider's own failure declaration, normalized.
        body["error"] = e;
    }
    Reassembled {
        body: serde_json::to_vec(&body).unwrap_or_default(),
        terminated,
    }
}
