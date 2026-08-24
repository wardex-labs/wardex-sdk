//! LLM request/response body → semantic (gen_ai) extraction. framework-agnostic body parser.
//! provider=host priority + body-shape fallback, operation=path. fail-safe (parse failure = partial/empty result).

use std::io::Read;

use crate::sse::{self, SseEvent};
use crate::usage::{InputConvention, TokenUsage};
use serde::Deserialize;
use wardex_limits::Limits;

/// Neutral semantic fields absorbing per-provider differences. All Option (None = not extracted).
#[derive(Debug, Default, Clone, PartialEq)]
pub struct LlmSemantics {
    pub provider: String,  // "openai" | "anthropic"
    pub operation: String, // "chat" | "embeddings"
    pub request_model: Option<String>,
    pub response_model: Option<String>,
    pub response_id: Option<String>,
    /// Always semconv-inclusive — the parser that filled it named its
    /// provider's `InputConvention`, and `TokenUsage::new` did the arithmetic.
    pub usage: TokenUsage,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub max_tokens: Option<i64>,
    pub seed: Option<i64>,
    pub choice_count: Option<i64>,
    pub stop_sequences: Option<Vec<String>>,
    pub stream: Option<bool>,
    pub finish_reasons: Option<Vec<String>>,
    pub output_type: Option<String>,
    pub decoded_response: Option<Vec<u8>>,
    pub reassembled_from_stream: bool,
    /// OTel gen_ai.output.messages (all parts of the response: text·tool_call etc., Some only when present) — JSON string.
    pub output_messages: Option<String>,
    /// Whether the OpenAI arguments string failed JSON parsing (for marker use).
    pub tool_args_unparsed: bool,
    /// Whether output.messages had unrecognized blocks (GenericPart fallback) (for marker use).
    pub output_messages_has_unmapped: bool,
    /// OTel gen_ai.input.messages (request conversation: user/assistant/tool, Some only when present) — JSON string.
    pub input_messages: Option<String>,
    /// OTel gen_ai.system_instructions (system/developer instructions, TextPart array) — JSON string.
    pub system_instructions: Option<String>,
    /// Whether input.messages had unrecognized blocks (GenericPart fallback) (for marker use).
    pub input_messages_has_unmapped: bool,
}

/// stop: allows both a string and an array of strings.
#[derive(Deserialize)]
#[serde(untagged)]
enum StringOrVec {
    One(String),
    Many(Vec<String>),
}

impl StringOrVec {
    fn into_vec(self) -> Vec<String> {
        match self {
            StringOrVec::One(s) => vec![s],
            StringOrVec::Many(v) => v,
        }
    }
}

fn operation_from_path(path: &str) -> Option<&'static str> {
    if path.contains("/chat/completions") || path.contains("/messages") {
        Some("chat")
    } else if path.contains("/embeddings") {
        Some("embeddings")
    } else {
        None
    }
}

fn provider_from_host(host: &str) -> Option<&'static str> {
    if host.contains("openai") {
        Some("openai")
    } else if host.contains("anthropic") {
        Some("anthropic")
    } else {
        None
    }
}

// --- OpenAI structs ---

#[derive(Deserialize, Default)]
struct OAPromptDetails {
    #[serde(default)]
    cached_tokens: Option<i64>,
}
#[derive(Deserialize, Default)]
struct OACompletionDetails {
    #[serde(default)]
    reasoning_tokens: Option<i64>,
}
#[derive(Deserialize, Default)]
struct OAUsage {
    #[serde(default)]
    prompt_tokens: Option<i64>,
    #[serde(default)]
    completion_tokens: Option<i64>,
    #[serde(default)]
    prompt_tokens_details: Option<OAPromptDetails>,
    #[serde(default)]
    completion_tokens_details: Option<OACompletionDetails>,
}
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

/// Builds a single OTel ToolCallRequestPart (id included only when Some).
fn tool_call_part(
    id: Option<String>,
    name: String,
    arguments: serde_json::Value,
) -> serde_json::Value {
    let mut m = serde_json::Map::new();
    m.insert("type".to_string(), serde_json::Value::from("tool_call"));
    if let Some(i) = id {
        m.insert("id".to_string(), serde_json::Value::from(i));
    }
    m.insert("name".to_string(), serde_json::Value::from(name));
    m.insert("arguments".to_string(), arguments);
    serde_json::Value::Object(m)
}

/// OTel ToolCallResponsePart (id included only when Some).
fn tool_call_response_part(id: Option<String>, response: serde_json::Value) -> serde_json::Value {
    let mut m = serde_json::Map::new();
    m.insert(
        "type".to_string(),
        serde_json::Value::from("tool_call_response"),
    );
    if let Some(i) = id {
        m.insert("id".to_string(), serde_json::Value::from(i));
    }
    m.insert("response".to_string(), response);
    serde_json::Value::Object(m)
}

/// mime → OTel Modality enum (image/video/audio/document). Unknown falls back to document.
fn modality_from_mime(mime: &str) -> &'static str {
    if mime.starts_with("image/") {
        "image"
    } else if mime.starts_with("audio/") {
        "audio"
    } else if mime.starts_with("video/") {
        "video"
    } else {
        "document"
    }
}

/// OTel BlobPart (inline base64). modality·content required.
fn blob_part(modality: &str, mime_type: Option<&str>, content_b64: &str) -> serde_json::Value {
    let mut m = serde_json::Map::new();
    m.insert("type".to_string(), serde_json::Value::from("blob"));
    m.insert("modality".to_string(), serde_json::Value::from(modality));
    if let Some(mt) = mime_type {
        m.insert("mime_type".to_string(), serde_json::Value::from(mt));
    }
    m.insert("content".to_string(), serde_json::Value::from(content_b64));
    serde_json::Value::Object(m)
}

/// OTel UriPart (URL reference). modality·uri required. base64 data URLs are not used here (use BlobPart).
fn uri_part(modality: &str, mime_type: Option<&str>, uri: &str) -> serde_json::Value {
    let mut m = serde_json::Map::new();
    m.insert("type".to_string(), serde_json::Value::from("uri"));
    m.insert("modality".to_string(), serde_json::Value::from(modality));
    if let Some(mt) = mime_type {
        m.insert("mime_type".to_string(), serde_json::Value::from(mt));
    }
    m.insert("uri".to_string(), serde_json::Value::from(uri));
    serde_json::Value::Object(m)
}

/// OTel FilePart (pre-uploaded file id reference). modality·file_id required.
fn file_part(modality: &str, mime_type: Option<&str>, file_id: &str) -> serde_json::Value {
    let mut m = serde_json::Map::new();
    m.insert("type".to_string(), serde_json::Value::from("file"));
    m.insert("modality".to_string(), serde_json::Value::from(modality));
    if let Some(mt) = mime_type {
        m.insert("mime_type".to_string(), serde_json::Value::from(mt));
    }
    m.insert("file_id".to_string(), serde_json::Value::from(file_id));
    serde_json::Value::Object(m)
}

/// Parses a data URL (`data:<mime>;base64,<b64>`) → (mime, b64). Otherwise None.
fn parse_data_url(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, b64) = rest.split_once(',')?;
    let mime = meta.strip_suffix(";base64")?;
    Some((mime, b64))
}

/// Intermediate representation of a single input message (no finish_reason).
struct InMsg {
    role: String,
    parts: Vec<serde_json::Value>,
}

/// InMsg[] → OTel ChatMessage[] JSON. Messages with no parts are excluded; None if all are excluded.
fn build_input_messages(msgs: Vec<InMsg>) -> Option<String> {
    let arr: Vec<serde_json::Value> = msgs
        .into_iter()
        .filter(|m| !m.parts.is_empty())
        .map(|m| serde_json::json!({"role": m.role, "parts": m.parts}))
        .collect();
    if arr.is_empty() {
        return None;
    }
    serde_json::to_string(&serde_json::Value::Array(arr)).ok()
}

/// TextPart[] → OTel SystemInstructions JSON. None if empty.
fn build_system_instructions(parts: Vec<serde_json::Value>) -> Option<String> {
    if parts.is_empty() {
        return None;
    }
    serde_json::to_string(&serde_json::Value::Array(parts)).ok()
}

/// Intermediate representation of a single OutputMessage.
struct OutMsg {
    parts: Vec<serde_json::Value>,
    finish_reason: Option<String>,
}

/// OutMsg list → OTel OutputMessage[] JSON string. Messages with neither parts nor finish are excluded; None if all are excluded.
fn build_output_messages(msgs: Vec<OutMsg>) -> Option<String> {
    let arr: Vec<serde_json::Value> = msgs
        .into_iter()
        .filter(|m| !m.parts.is_empty() || m.finish_reason.is_some())
        .map(|m| {
            let mut obj = serde_json::Map::new();
            obj.insert("role".to_string(), serde_json::Value::from("assistant"));
            obj.insert("parts".to_string(), serde_json::Value::Array(m.parts));
            if let Some(fr) = m.finish_reason {
                obj.insert("finish_reason".to_string(), serde_json::Value::from(fr));
            }
            serde_json::Value::Object(obj)
        })
        .collect();
    if arr.is_empty() {
        return None;
    }
    serde_json::to_string(&serde_json::Value::Array(arr)).ok()
}

fn text_part(content: String) -> serde_json::Value {
    serde_json::json!({"type": "text", "content": content})
}

fn reasoning_part(content: String) -> serde_json::Value {
    serde_json::json!({"type": "reasoning", "content": content})
}

fn generic_part(type_str: &str) -> serde_json::Value {
    serde_json::json!({"type": type_str})
}

fn server_tool_call_part(
    id: Option<String>,
    name: String,
    call: serde_json::Value,
) -> serde_json::Value {
    let mut m = serde_json::Map::new();
    m.insert(
        "type".to_string(),
        serde_json::Value::from("server_tool_call"),
    );
    if let Some(i) = id {
        m.insert("id".to_string(), serde_json::Value::from(i));
    }
    m.insert("name".to_string(), serde_json::Value::from(name));
    m.insert("server_tool_call".to_string(), call);
    serde_json::Value::Object(m)
}

fn server_tool_call_response_part(
    id: Option<String>,
    response: serde_json::Value,
) -> serde_json::Value {
    let mut m = serde_json::Map::new();
    m.insert(
        "type".to_string(),
        serde_json::Value::from("server_tool_call_response"),
    );
    if let Some(i) = id {
        m.insert("id".to_string(), serde_json::Value::from(i));
    }
    m.insert("server_tool_call_response".to_string(), response);
    serde_json::Value::Object(m)
}

/// Per-provider finish_reason → OTel enum. Unmapped is None (field omitted).
fn finish_reason_to_otel(provider: &str, raw: &str) -> Option<String> {
    let v = match (provider, raw) {
        (_, "stop") | ("anthropic", "end_turn") | ("anthropic", "stop_sequence") => "stop",
        (_, "length") | ("anthropic", "max_tokens") => "length",
        (_, "tool_calls") | (_, "function_call") | ("anthropic", "tool_use") => "tool_call",
        (_, "content_filter") | ("anthropic", "refusal") => "content_filter",
        _ => return None,
    };
    Some(v.to_string())
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
    usage: Option<OAUsage>,
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

fn fill_openai_chat(out: &mut LlmSemantics, req: &[u8], resp: &[u8]) {
    if let Ok(r) = serde_json::from_slice::<OpenAIChatResponse>(resp) {
        out.response_id = r.id;
        out.response_model = r.model;
        let mut fr: Vec<String> = Vec::new();
        let mut msgs: Vec<OutMsg> = Vec::new();
        for c in r.choices {
            let finish_reason = c
                .finish_reason
                .as_deref()
                .and_then(|f| finish_reason_to_otel("openai", f));
            if let Some(f) = c.finish_reason {
                fr.push(f);
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
            out.usage = TokenUsage::new(
                InputConvention::Inclusive,
                u.prompt_tokens,
                u.completion_tokens,
                u.prompt_tokens_details.and_then(|d| d.cached_tokens),
                None,
                u.completion_tokens_details.and_then(|d| d.reasoning_tokens),
            );
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
        if let Some(messages) = q.messages {
            fill_openai_input(out, &messages);
        }
    }
}

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

// --- OpenAI embeddings ---

#[derive(Deserialize)]
struct OpenAIEmbeddingsResponse {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<OAUsage>,
}

/// Anthropic request system + messages[] → system_instructions + input.messages.
fn fill_anthropic_input(
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

fn fill_anthropic(out: &mut LlmSemantics, req: &[u8], resp: &[u8]) {
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

fn fill_openai_embeddings(out: &mut LlmSemantics, resp: &[u8]) {
    if let Ok(r) = serde_json::from_slice::<OpenAIEmbeddingsResponse>(resp) {
        out.response_model = r.model;
        if let Some(u) = r.usage {
            out.usage = TokenUsage::new(
                InputConvention::Inclusive,
                u.prompt_tokens,
                None,
                None,
                None,
                None,
            );
        }
    }
}

/// Infers the provider from the response JSON shape (fallback when host doesn't match).
fn provider_from_body(resp: &[u8]) -> Option<&'static str> {
    let v: serde_json::Value = serde_json::from_slice(resp).ok()?;
    let obj = v.as_object()?;
    // anthropic: stop_reason or usage.input_tokens
    if obj.contains_key("stop_reason")
        || obj
            .get("usage")
            .and_then(|u| u.get("input_tokens"))
            .is_some()
    {
        return Some("anthropic");
    }
    // openai: choices or usage.prompt_tokens or embeddings data+usage
    if obj.contains_key("choices")
        || obj
            .get("usage")
            .and_then(|u| u.get("prompt_tokens"))
            .is_some()
    {
        return Some("openai");
    }
    None
}

/// Infers the provider from the SSE event shape.
fn provider_from_sse(events: &[SseEvent]) -> Option<&'static str> {
    for ev in events {
        if ev.event.as_deref() == Some("message_start") {
            return Some("anthropic");
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&ev.data) {
            if v.get("type").and_then(|x| x.as_str()) == Some("message_start") {
                return Some("anthropic");
            }
            if v.get("choices").is_some() {
                return Some("openai");
            }
        }
    }
    None
}

/// OpenAI chat stream chunks → synthetic non-streaming-shaped JSON.
/// Accumulates tool_call deltas into a BTreeMap keyed by index, concatenating the arguments string,
/// then emits them as the choices[0].message.tool_calls array → fill_openai_chat reuses it for extraction.
fn reassemble_openai(events: &[SseEvent]) -> Vec<u8> {
    use std::collections::BTreeMap;
    let mut id: Option<String> = None;
    let mut model: Option<String> = None;
    let mut content = String::new();
    let mut finish: Option<String> = None;
    let mut usage: Option<serde_json::Value> = None;
    // index -> (id, name, arguments(accumulated))
    let mut tool_calls: BTreeMap<i64, (Option<String>, Option<String>, String)> = BTreeMap::new();
    for ev in events {
        if ev.data.trim() == "[DONE]" {
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
    serde_json::to_vec(&obj).unwrap_or_default()
}

/// Anthropic messages stream events → synthetic non-streaming-shaped JSON.
fn reassemble_anthropic(events: &[SseEvent]) -> Vec<u8> {
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

/// If the body is SSE, reassemble it for semantic extraction. Otherwise None (falls through to the existing path).
fn try_parse_sse(host: &str, path: &str, req: &[u8], decoded: &[u8]) -> Option<LlmSemantics> {
    if !sse::looks_like_sse(decoded) {
        return None;
    }
    let events = sse::parse(decoded);
    let operation = operation_from_path(path).unwrap_or("chat");
    let provider = provider_from_host(host).or_else(|| provider_from_sse(&events));
    match provider {
        Some("openai") => {
            let synthetic = reassemble_openai(&events);
            let mut out = LlmSemantics {
                provider: "openai".to_string(),
                operation: operation.to_string(),
                output_type: Some("text".to_string()),
                reassembled_from_stream: true,
                decoded_response: Some(synthetic.clone()),
                ..Default::default()
            };
            fill_openai_chat(&mut out, req, &synthetic);
            Some(out)
        }
        Some("anthropic") => {
            let synthetic = reassemble_anthropic(&events);
            let mut out = LlmSemantics {
                provider: "anthropic".to_string(),
                operation: operation.to_string(),
                output_type: Some("text".to_string()),
                reassembled_from_stream: true,
                decoded_response: Some(synthetic.clone()),
                ..Default::default()
            };
            fill_anthropic(&mut out, req, &synthetic);
            Some(out)
        }
        _ => {
            // unidentified: concat the data payloads
            let raw = events
                .iter()
                .map(|e| e.data.as_str())
                .collect::<Vec<_>>()
                .join("\n")
                .into_bytes();
            Some(LlmSemantics {
                operation: operation.to_string(),
                output_type: Some("text".to_string()),
                reassembled_from_stream: true,
                decoded_response: Some(raw),
                ..Default::default()
            })
        }
    }
}

/// Decompresses if gzip/zlib/deflate, otherwise copies the original. Returns the original on failure.
/// `limits.max_decoded_bytes` bounds the decompressed size — defense against decompression bombs.
fn decode_body(resp: &[u8], limits: Limits) -> Vec<u8> {
    let max_decoded = limits.max_decoded_bytes;
    if resp.len() >= 2 && resp[0] == 0x1f && resp[1] == 0x8b {
        let mut out = Vec::new();
        let r = flate2::read::GzDecoder::new(resp)
            .take((max_decoded + 1) as u64)
            .read_to_end(&mut out);
        if r.is_ok() && out.len() <= max_decoded {
            return out;
        }
    } else if resp.len() >= 2 && resp[0] == 0x78 {
        // zlib (deflate) header
        let mut out = Vec::new();
        let r = flate2::read::ZlibDecoder::new(resp)
            .take((max_decoded + 1) as u64)
            .read_to_end(&mut out);
        if r.is_ok() && out.len() <= max_decoded {
            return out;
        }
    }
    resp.to_vec()
}

pub fn parse_llm(
    host: &str,
    path: &str,
    req: &[u8],
    resp: &[u8],
    limits: Limits,
) -> Option<LlmSemantics> {
    let decoded = decode_body(resp, limits);
    if let Some(s) = try_parse_sse(host, path, req, &decoded) {
        return Some(s);
    }
    let operation = operation_from_path(path)?;
    let provider = provider_from_host(host).or_else(|| provider_from_body(&decoded))?;
    let mut out = LlmSemantics {
        provider: provider.to_string(),
        operation: operation.to_string(),
        output_type: Some("text".to_string()),
        decoded_response: Some(decoded.clone()),
        ..Default::default()
    };
    match (provider, operation) {
        ("openai", "chat") => fill_openai_chat(&mut out, req, &decoded),
        ("openai", "embeddings") => fill_openai_embeddings(&mut out, &decoded),
        ("anthropic", "chat") => fill_anthropic(&mut out, req, &decoded),
        _ => {} // unsupported combination (e.g. anthropic embeddings) is left as empty semantics
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPENAI_CHAT: &[u8] = br#"{
        "id":"chatcmpl-abc","model":"gpt-4o-mini-2024-07-18",
        "choices":[{"finish_reason":"stop"}],
        "usage":{"prompt_tokens":12,"completion_tokens":3,
                 "completion_tokens_details":{"reasoning_tokens":1}}
    }"#;
    const OPENAI_REQ: &[u8] = br#"{"model":"gpt-4o-mini","temperature":0.5,"max_tokens":64,"top_p":1.0,"n":2,"stream":false,"stop":"END"}"#;

    #[test]
    fn openai_chat_response_extracts_tokens_and_model() {
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            OPENAI_REQ,
            OPENAI_CHAT,
            Limits::default(),
        )
        .expect("supported host");
        assert_eq!(s.provider, "openai");
        assert_eq!(s.operation, "chat");
        assert_eq!(s.response_model.as_deref(), Some("gpt-4o-mini-2024-07-18"));
        assert_eq!(s.response_id.as_deref(), Some("chatcmpl-abc"));
        assert_eq!(s.usage.input_tokens(), Some(12));
        assert_eq!(s.usage.output_tokens(), Some(3));
        assert_eq!(s.usage.reasoning_output_tokens(), Some(1));
        assert_eq!(s.finish_reasons.as_deref(), Some(&["stop".to_string()][..]));
        // request params
        assert_eq!(s.request_model.as_deref(), Some("gpt-4o-mini"));
        assert_eq!(s.temperature, Some(0.5));
        assert_eq!(s.max_tokens, Some(64));
        assert_eq!(s.choice_count, Some(2));
        assert_eq!(s.stop_sequences.as_deref(), Some(&["END".to_string()][..]));
        // this body is not compressed, so `decode_body` passes the original through
        assert_eq!(s.decoded_response.as_deref(), Some(OPENAI_CHAT));
    }

    #[test]
    fn decoded_cap_comes_from_limits() {
        // The cap only bites on the decompression path (plain bodies pass through uncapped),
        // so gzip-compress the body to exercise it — mirrors `gzip_response_is_decompressed_and_parsed`.
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;
        let limits = Limits {
            max_decoded_bytes: 1,
            ..Default::default()
        };
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        enc.write_all(OPENAI_CHAT).unwrap();
        let gz = enc.finish().unwrap();

        let sem = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            OPENAI_REQ,
            &gz,
            limits,
        );
        // The decompressed response exceeds the tiny decode cap, so decompression is
        // rejected and no semantics can be extracted from the (still-compressed) bytes.
        assert!(sem.is_none() || sem.unwrap().response_model.is_none());
    }

    #[test]
    fn unsupported_host_returns_none() {
        assert!(parse_llm(
            "example.com",
            "/v1/foo",
            b"{}",
            b"{\"ok\":true}",
            Limits::default()
        )
        .is_none());
    }

    const ANTHROPIC_MSG: &[u8] = br#"{
        "id":"msg_1","model":"claude-opus-4-8","stop_reason":"end_turn",
        "usage":{"input_tokens":12,"output_tokens":3,"cache_read_input_tokens":4}
    }"#;
    const OPENAI_EMB: &[u8] =
        br#"{"model":"text-embedding-3-small","data":[{}],"usage":{"prompt_tokens":8}}"#;

    #[test]
    fn anthropic_messages_extracts_tokens() {
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            b"{}",
            ANTHROPIC_MSG,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(s.provider, "anthropic");
        assert_eq!(s.operation, "chat");
        assert_eq!(s.response_model.as_deref(), Some("claude-opus-4-8"));
        // Inclusive total: raw 12 + cache_read 4. That this asserted 12 —
        // and passed — was the defect's evidence (R2).
        assert_eq!(s.usage.input_tokens(), Some(16));
        assert_eq!(s.usage.output_tokens(), Some(3));
        assert_eq!(s.usage.cache_read_input_tokens(), Some(4));
        assert_eq!(
            s.finish_reasons.as_deref(),
            Some(&["end_turn".to_string()][..])
        );
    }

    #[test]
    fn openai_embeddings_extracts_input_tokens() {
        let s = parse_llm(
            "api.openai.com",
            "/v1/embeddings",
            b"{}",
            OPENAI_EMB,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(s.operation, "embeddings");
        assert_eq!(s.usage.input_tokens(), Some(8));
        assert_eq!(s.usage.output_tokens(), None);
    }

    #[test]
    fn body_shape_fallback_detects_provider_on_localhost() {
        // even when the host is unsupported (127.0.0.1), infer the provider from the response shape
        let oa = parse_llm(
            "127.0.0.1",
            "/v1/chat/completions",
            OPENAI_REQ,
            OPENAI_CHAT,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(oa.provider, "openai");
        assert_eq!(oa.usage.input_tokens(), Some(12));
        let an = parse_llm(
            "127.0.0.1",
            "/v1/messages",
            b"{}",
            ANTHROPIC_MSG,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(an.provider, "anthropic");
        // ANTHROPIC_MSG reports raw 12 + cache_read 4 — inclusive total 16.
        assert_eq!(an.usage.input_tokens(), Some(16));
    }

    #[test]
    fn non_llm_json_unknown_host_returns_none() {
        assert!(parse_llm(
            "127.0.0.1",
            "/v1/chat/completions",
            b"{}",
            br#"{"ok":true}"#,
            Limits::default(),
        )
        .is_none());
    }

    #[test]
    fn gzip_response_is_decompressed_and_parsed() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        enc.write_all(OPENAI_CHAT).unwrap();
        let gz = enc.finish().unwrap();
        assert_eq!(&gz[..2], &[0x1f, 0x8b]); // gzip magic

        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            OPENAI_REQ,
            &gz,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(s.usage.input_tokens(), Some(12)); // parsed after decompression
        assert_eq!(s.decoded_response.as_deref(), Some(OPENAI_CHAT)); // decoded body stored
    }

    #[test]
    fn decode_body_caps_decompression_bomb() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;
        let big = vec![0u8; 9 * 1024 * 1024]; // 9 MiB > 8 MiB cap
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&big).unwrap();
        let gz = enc.finish().unwrap();
        assert!(gz.len() < big.len()); // compressed
                                       // exceeds cap → not decompressed, original (gz) returned
        assert_eq!(decode_body(&gz, Limits::default()), gz);
    }

    #[test]
    fn decode_body_under_cap_decompresses() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;
        let small = b"hello world";
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        enc.write_all(small).unwrap();
        let gz = enc.finish().unwrap();
        assert_eq!(decode_body(&gz, Limits::default()), small);
    }

    #[test]
    fn anthropic_cache_creation_input_tokens_extracted() {
        let resp = br#"{
            "id":"msg_2","model":"claude-opus-4-8","stop_reason":"end_turn",
            "usage":{"input_tokens":5,"output_tokens":2,"cache_creation_input_tokens":7}
        }"#;
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            b"{}",
            resp,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(s.usage.cache_creation_input_tokens(), Some(7));
        // Inclusive total: raw 5 + cache_creation 7 (R2).
        assert_eq!(s.usage.input_tokens(), Some(12));
        assert_eq!(s.usage.output_tokens(), Some(2));
    }

    #[test]
    fn openai_stop_array_maps_to_string_or_vec_many() {
        // request where stop is an array (["A","B"]) → verifies the Many path into stop_sequences
        let req = br#"{"model":"gpt-4o","stop":["A","B"]}"#;
        let resp = br#"{
            "id":"chatcmpl-y","model":"gpt-4o",
            "choices":[{"finish_reason":"stop"}],
            "usage":{"prompt_tokens":1,"completion_tokens":1}
        }"#;
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            req,
            resp,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(
            s.stop_sequences.as_deref(),
            Some(&["A".to_string(), "B".to_string()][..])
        );
    }

    const OPENAI_SSE: &[u8] = b"data: {\"id\":\"chatcmpl-s\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-s\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

    #[test]
    fn openai_sse_stream_reassembles_text_and_finish_no_usage() {
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            b"{}",
            OPENAI_SSE,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(s.provider, "openai");
        assert!(s.reassembled_from_stream);
        assert_eq!(s.response_model.as_deref(), Some("gpt-4o-mini"));
        assert_eq!(s.response_id.as_deref(), Some("chatcmpl-s"));
        assert_eq!(s.finish_reasons.as_deref(), Some(&["stop".to_string()][..]));
        assert_eq!(s.usage.output_tokens(), None); // no usage
        let body = String::from_utf8(s.decoded_response.clone().unwrap()).unwrap();
        assert!(body.contains("\"content\":\"Hi!\"")); // text reassembled into the synthetic JSON
    }

    #[test]
    fn openai_sse_with_usage_chunk_extracts_tokens() {
        let sse = b"data: {\"id\":\"c\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n";
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            b"{}",
            sse,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(s.usage.input_tokens(), Some(11));
        assert_eq!(s.usage.output_tokens(), Some(2));
    }

    #[test]
    fn anthropic_sse_stream_extracts_tokens_text_stop() {
        let sse = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s\",\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":9,\"output_tokens\":1}}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            b"{}",
            sse,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(s.provider, "anthropic");
        assert!(s.reassembled_from_stream);
        assert_eq!(s.response_model.as_deref(), Some("claude-opus-4-8"));
        assert_eq!(s.usage.input_tokens(), Some(9));
        assert_eq!(s.usage.output_tokens(), Some(5)); // message_delta provides the final output_tokens
        assert_eq!(
            s.finish_reasons.as_deref(),
            Some(&["end_turn".to_string()][..])
        );
        let body = String::from_utf8(s.decoded_response.clone().unwrap()).unwrap();
        assert!(body.contains("\"text\":\"Hello\""));
    }

    /// P2 double-add guard: `reassemble_anthropic` keeps the RAW usage in its
    /// synthetic JSON and `fill_anthropic` does the one and only inclusive
    /// sum. If the reassembler ever pre-adds the cache tiers, this total
    /// doubles and the test names the reason.
    #[test]
    fn test_reassembled_stream_is_not_double_added() {
        let sse = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s\",\"model\":\"claude-sonnet-4-6\",\"usage\":{\"input_tokens\":1000,\"output_tokens\":1,\"cache_read_input_tokens\":8000,\"cache_creation_input_tokens\":2000}}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":500}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            b"{}",
            sse,
            Limits::default(),
        )
        .unwrap();
        assert!(s.reassembled_from_stream);
        // Exactly once: 1000 + 8000 + 2000, not 11000 + 10000.
        assert_eq!(s.usage.input_tokens(), Some(11000));
        assert_eq!(s.usage.cache_read_input_tokens(), Some(8000));
        assert_eq!(s.usage.cache_creation_input_tokens(), Some(2000));
        assert_eq!(s.usage.output_tokens(), Some(500));
        // The synthetic body still carries Anthropic's RAW value — the wire
        // truth is preserved and the sum happens in exactly one place.
        let body = String::from_utf8(s.decoded_response.clone().unwrap()).unwrap();
        assert!(body.contains("\"input_tokens\":1000"));
    }

    #[test]
    fn unknown_provider_sse_returns_raw_concat_no_semantics() {
        let sse = b"data: {\"foo\":1}\n\ndata: {\"bar\":2}\n\n";
        let s = parse_llm("127.0.0.1", "/v1/stream", b"{}", sse, Limits::default()).unwrap();
        assert!(s.reassembled_from_stream);
        assert_eq!(s.provider, ""); // unidentified
        assert_eq!(s.response_model, None);
        let body = String::from_utf8(s.decoded_response.clone().unwrap()).unwrap();
        assert_eq!(body, "{\"foo\":1}\n{\"bar\":2}");
    }

    #[test]
    fn non_sse_json_still_uses_existing_path() {
        // non-SSE keeps reassembled_from_stream=false (regression guard)
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            OPENAI_REQ,
            OPENAI_CHAT,
            Limits::default(),
        )
        .unwrap();
        assert!(!s.reassembled_from_stream);
        assert_eq!(s.usage.input_tokens(), Some(12));
    }

    #[test]
    fn gzip_sse_is_decompressed_then_reassembled() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        enc.write_all(OPENAI_SSE).unwrap();
        let gz = enc.finish().unwrap();
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            b"{}",
            &gz,
            Limits::default(),
        )
        .unwrap();
        assert!(s.reassembled_from_stream);
        assert_eq!(s.response_model.as_deref(), Some("gpt-4o-mini"));
    }

    #[test]
    fn openai_chat_extracts_tool_call() {
        let req = br#"{"model":"gpt-4o-mini"}"#;
        let resp = br#"{"id":"chatcmpl-x","model":"gpt-4o-mini","choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"add","arguments":"{\"a\":17,\"b\":25}"}}]},"finish_reason":"tool_calls"}]}"#;
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            req,
            resp,
            Limits::default(),
        )
        .expect("some");
        let om = s.output_messages.expect("output_messages");
        let v: serde_json::Value = serde_json::from_str(&om).unwrap();
        let part = &v[0]["parts"][0];
        assert_eq!(part["type"], "tool_call");
        assert_eq!(part["name"], "add");
        assert_eq!(part["id"], "call_1");
        assert_eq!(part["arguments"]["a"], 17);
        assert_eq!(part["arguments"]["b"], 25);
        assert!(!s.tool_args_unparsed);
    }

    #[test]
    fn openai_chat_multiple_parallel_tool_calls() {
        let req = br#"{"model":"gpt-4o-mini"}"#;
        let resp = br#"{"model":"gpt-4o-mini","choices":[{"message":{"tool_calls":[{"id":"c1","function":{"name":"a","arguments":"{}"}},{"id":"c2","function":{"name":"b","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#;
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            req,
            resp,
            Limits::default(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
        let parts = v[0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["name"], "a");
        assert_eq!(parts[1]["name"], "b");
    }

    #[test]
    fn text_only_response_emits_text_part() {
        let req = br#"{"model":"gpt-4o-mini"}"#;
        let resp = br#"{"model":"gpt-4o-mini","choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#;
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            req,
            resp,
            Limits::default(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
        assert_eq!(v[0]["parts"][0]["type"], "text");
        assert_eq!(v[0]["parts"][0]["content"], "hi");
        assert_eq!(v[0]["finish_reason"], "stop");
    }

    #[test]
    fn openai_tool_args_unparsable_keeps_raw_and_marks() {
        let req = br#"{"model":"gpt-4o-mini"}"#;
        let resp = br#"{"model":"gpt-4o-mini","choices":[{"message":{"tool_calls":[{"id":"c1","function":{"name":"a","arguments":"{not json"}}]},"finish_reason":"tool_calls"}]}"#;
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            req,
            resp,
            Limits::default(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
        assert_eq!(v[0]["parts"][0]["arguments"], "{not json");
        assert!(s.tool_args_unparsed);
    }

    #[test]
    fn anthropic_extracts_tool_use() {
        let req = br#"{"model":"claude-3"}"#;
        let resp = br#"{"id":"msg_1","model":"claude-3","stop_reason":"tool_use","content":[{"type":"text","text":"let me calculate"},{"type":"tool_use","id":"toolu_1","name":"add","input":{"a":17,"b":25}}],"usage":{"input_tokens":5,"output_tokens":2}}"#;
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            req,
            resp,
            Limits::default(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
        let parts = v[0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "tool_call");
        assert_eq!(parts[1]["name"], "add");
        assert_eq!(parts[1]["id"], "toolu_1");
        assert_eq!(parts[1]["arguments"]["a"], 17);
        assert_eq!(v[0]["finish_reason"], "tool_call");
    }

    #[test]
    fn anthropic_thinking_text_tooluse_ordered() {
        let resp = r#"{"model":"c","stop_reason":"tool_use","content":[{"type":"thinking","thinking":"hmm"},{"type":"text","text":"answer"},{"type":"tool_use","id":"t1","name":"add","input":{}}]}"#.as_bytes();
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            br#"{"model":"c"}"#,
            resp,
            Limits::default(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
        let parts = v[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "reasoning");
        assert_eq!(parts[0]["content"], "hmm");
        assert_eq!(parts[1]["type"], "text");
        assert_eq!(parts[1]["content"], "answer");
        assert_eq!(parts[2]["type"], "tool_call");
        assert!(!s.output_messages_has_unmapped);
    }

    #[test]
    fn anthropic_redacted_thinking_to_generic_no_flag() {
        let resp = br#"{"model":"c","stop_reason":"end_turn","content":[{"type":"redacted_thinking","data":"enc"}]}"#;
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            br#"{"model":"c"}"#,
            resp,
            Limits::default(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
        assert_eq!(v[0]["parts"][0]["type"], "reasoning");
        assert_eq!(v[0]["parts"][0].as_object().unwrap().len(), 1); // type only
        assert!(!s.output_messages_has_unmapped);
    }

    #[test]
    fn anthropic_server_tool_use_and_result() {
        let resp = br#"{"model":"c","stop_reason":"end_turn","content":[{"type":"server_tool_use","id":"s1","name":"web_search","input":{"query":"x"}},{"type":"web_search_tool_result","tool_use_id":"s1","content":[{"title":"r"}]}]}"#;
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            br#"{"model":"c"}"#,
            resp,
            Limits::default(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
        let parts = v[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "server_tool_call");
        assert_eq!(parts[0]["name"], "web_search");
        assert_eq!(parts[0]["id"], "s1");
        assert_eq!(parts[0]["server_tool_call"]["query"], "x");
        assert_eq!(parts[1]["type"], "server_tool_call_response");
        assert_eq!(parts[1]["id"], "s1");
        assert_eq!(parts[1]["server_tool_call_response"][0]["title"], "r");
    }

    #[test]
    fn anthropic_unknown_block_to_generic_sets_flag() {
        let resp = br#"{"model":"c","stop_reason":"end_turn","content":[{"type":"some_future_block","foo":1}]}"#;
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            br#"{"model":"c"}"#,
            resp,
            Limits::default(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
        assert_eq!(v[0]["parts"][0]["type"], "some_future_block");
        assert!(s.output_messages_has_unmapped);
    }

    #[test]
    fn anthropic_finish_reason_normalized() {
        let resp =
            br#"{"model":"c","stop_reason":"max_tokens","content":[{"type":"text","text":"x"}]}"#;
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            br#"{"model":"c"}"#,
            resp,
            Limits::default(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
        assert_eq!(v[0]["finish_reason"], "length");
    }

    #[test]
    fn openai_and_anthropic_tool_call_same_shape() {
        let oa = parse_llm("api.openai.com", "/v1/chat/completions",
            br#"{"model":"m"}"#,
            br#"{"model":"m","choices":[{"message":{"tool_calls":[{"id":"x","function":{"name":"add","arguments":"{\"a\":1}"}}]}}]}"#, Limits::default()).unwrap();
        let an = parse_llm("api.anthropic.com", "/v1/messages",
            br#"{"model":"m"}"#,
            br#"{"model":"m","content":[{"type":"tool_use","id":"x","name":"add","input":{"a":1}}]}"#, Limits::default()).unwrap();
        let ov: serde_json::Value = serde_json::from_str(&oa.output_messages.unwrap()).unwrap();
        let av: serde_json::Value = serde_json::from_str(&an.output_messages.unwrap()).unwrap();
        assert_eq!(ov[0]["parts"][0], av[0]["parts"][0]);
    }

    // --- SSE tool_call delta accumulation ---

    fn sse_bytes(lines: &[&str]) -> Vec<u8> {
        lines
            .iter()
            .map(|l| format!("data: {}\n\n", l))
            .collect::<String>()
            .into_bytes()
    }

    #[test]
    fn openai_sse_reassembles_tool_call_deltas() {
        let resp = sse_bytes(&[
            r#"{"id":"chatcmpl-s","model":"gpt-4o-mini","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"add","arguments":"{\"a\":1"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"7}"}}]},"finish_reason":"tool_calls"}]}"#,
            "[DONE]",
        ]);
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            br#"{"model":"gpt-4o-mini","stream":true}"#,
            &resp,
            Limits::default(),
        )
        .unwrap();
        assert!(s.reassembled_from_stream);
        let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
        assert_eq!(v[0]["parts"][0]["name"], "add");
        assert_eq!(v[0]["parts"][0]["arguments"]["a"], 17);
    }

    #[test]
    fn openai_sse_parallel_tool_calls_by_index() {
        let resp = sse_bytes(&[
            r#"{"model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","function":{"name":"a","arguments":"{}"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c1","function":{"name":"b","arguments":"{}"}}]}}]}"#,
            "[DONE]",
        ]);
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            br#"{"stream":true}"#,
            &resp,
            Limits::default(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
        let parts = v[0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["name"], "a");
        assert_eq!(parts[1]["name"], "b");
    }

    #[test]
    fn sse_text_plus_tool_call() {
        let resp = sse_bytes(&[
            r#"{"model":"m","choices":[{"delta":{"content":"wait"}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","function":{"name":"a","arguments":"{}"}}]}}]}"#,
            "[DONE]",
        ]);
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            br#"{"stream":true}"#,
            &resp,
            Limits::default(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
        let parts = v[0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["content"], "wait");
        assert_eq!(parts[1]["name"], "a");
    }

    #[test]
    fn finish_reason_maps_to_otel_enum() {
        assert_eq!(
            finish_reason_to_otel("openai", "tool_calls").as_deref(),
            Some("tool_call")
        );
        assert_eq!(
            finish_reason_to_otel("openai", "length").as_deref(),
            Some("length")
        );
        assert_eq!(
            finish_reason_to_otel("openai", "stop").as_deref(),
            Some("stop")
        );
        assert_eq!(
            finish_reason_to_otel("anthropic", "end_turn").as_deref(),
            Some("stop")
        );
        assert_eq!(
            finish_reason_to_otel("anthropic", "max_tokens").as_deref(),
            Some("length")
        );
        assert_eq!(
            finish_reason_to_otel("anthropic", "tool_use").as_deref(),
            Some("tool_call")
        );
        assert_eq!(
            finish_reason_to_otel("anthropic", "refusal").as_deref(),
            Some("content_filter")
        );
        assert_eq!(finish_reason_to_otel("anthropic", "pause_turn"), None);
        assert_eq!(
            finish_reason_to_otel("openai", "content_filter").as_deref(),
            Some("content_filter")
        );
        assert_eq!(
            finish_reason_to_otel("openai", "function_call").as_deref(),
            Some("tool_call")
        );
    }

    #[test]
    fn openai_text_then_tool_call_order() {
        let req = br#"{"model":"m"}"#;
        let resp = br#"{"model":"m","choices":[{"message":{"content":"calc","tool_calls":[{"id":"c1","function":{"name":"add","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#;
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            req,
            resp,
            Limits::default(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
        let parts = v[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "tool_call");
        assert_eq!(v[0]["finish_reason"], "tool_call");
    }

    #[test]
    fn openai_multiple_choices_emit_multiple_messages() {
        let req = br#"{"model":"m"}"#;
        let resp = br#"{"model":"m","choices":[{"message":{"content":"a"},"finish_reason":"stop"},{"message":{"content":"b"},"finish_reason":"tool_calls"}]}"#;
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            req,
            resp,
            Limits::default(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
        assert_eq!(v.as_array().unwrap().len(), 2);
        assert_eq!(v[0]["parts"][0]["content"], "a");
        assert_eq!(v[1]["finish_reason"], "tool_call");
    }

    // --- Anthropic SSE tool_use reassembly ---

    #[test]
    fn anthropic_sse_reassembles_tool_use() {
        let raw = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-3\",\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"add\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":1\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"7}\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":3}}\n\n",
        );
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            br#"{"model":"claude-3","stream":true}"#,
            raw.as_bytes(),
            Limits::default(),
        )
        .unwrap();
        assert!(s.reassembled_from_stream);
        let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
        assert_eq!(v[0]["parts"][0]["name"], "add");
        assert_eq!(v[0]["parts"][0]["id"], "toolu_1");
        assert_eq!(v[0]["parts"][0]["arguments"]["a"], 17);
    }

    /// thinking_delta events accumulate to assemble a ReasoningPart, and
    /// if text is also present, both parts are returned in the correct order.
    #[test]
    fn anthropic_sse_thinking_reassembled() {
        let raw = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"c\"}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"thi\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"nking\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"answer\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        );
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            br#"{"model":"c","stream":true}"#,
            raw.as_bytes(),
            Limits::default(),
        )
        .unwrap();
        assert!(s.reassembled_from_stream);
        let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
        let parts = v[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "reasoning");
        assert_eq!(parts[0]["content"], "thinking");
        assert_eq!(parts[1]["type"], "text");
        assert_eq!(parts[1]["content"], "answer");
    }

    // --- OpenAI input message parsing ---

    const OPENAI_REQ_MESSAGES: &[u8] = r#"{
        "model":"gpt-4o",
        "messages":[
            {"role":"system","content":"You are a weather assistant."},
            {"role":"user","content":"Weather in Seoul?"},
            {"role":"assistant","content":null,
             "tool_calls":[{"id":"call_1","type":"function",
                "function":{"name":"get_weather","arguments":"{\"city\":\"Seoul\"}"}}]},
            {"role":"tool","tool_call_id":"call_1","content":"18 degrees, clear"}
        ]
    }"#
    .as_bytes();
    const OPENAI_RESP_MIN: &[u8] =
        br#"{"id":"c1","model":"gpt-4o","choices":[{"finish_reason":"stop"}]}"#;

    #[test]
    fn openai_system_message_to_system_instructions() {
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            OPENAI_REQ_MESSAGES,
            OPENAI_RESP_MIN,
            Limits::default(),
        )
        .expect("supported");
        let si: serde_json::Value = serde_json::from_str(
            s.system_instructions
                .as_deref()
                .expect("system_instructions"),
        )
        .unwrap();
        assert_eq!(si.as_array().unwrap().len(), 1);
        assert_eq!(si[0]["type"], "text");
        assert_eq!(si[0]["content"], "You are a weather assistant.");
    }

    #[test]
    fn openai_input_messages_roles_and_tool_result() {
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            OPENAI_REQ_MESSAGES,
            OPENAI_RESP_MIN,
            Limits::default(),
        )
        .expect("supported");
        let im: serde_json::Value =
            serde_json::from_str(s.input_messages.as_deref().expect("input_messages")).unwrap();
        let arr = im.as_array().unwrap();
        // system is not in input.messages → 3 entries for user/assistant/tool
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0]["role"], "user");
        assert_eq!(arr[0]["parts"][0]["type"], "text");
        assert_eq!(arr[0]["parts"][0]["content"], "Weather in Seoul?");
        // assistant past tool_calls → ToolCallRequestPart
        assert_eq!(arr[1]["role"], "assistant");
        assert_eq!(arr[1]["parts"][0]["type"], "tool_call");
        assert_eq!(arr[1]["parts"][0]["id"], "call_1");
        assert_eq!(arr[1]["parts"][0]["name"], "get_weather");
        assert_eq!(arr[1]["parts"][0]["arguments"]["city"], "Seoul");
        // tool role → ToolCallResponsePart (A-3)
        assert_eq!(arr[2]["role"], "tool");
        assert_eq!(arr[2]["parts"][0]["type"], "tool_call_response");
        assert_eq!(arr[2]["parts"][0]["id"], "call_1");
        assert_eq!(arr[2]["parts"][0]["response"], "18 degrees, clear");
    }

    #[test]
    fn openai_developer_role_to_system_instructions() {
        let req = r#"{"model":"gpt-4o","messages":[{"role":"developer","content":"instruction"},{"role":"user","content":"hi"}]}"#.as_bytes();
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            req,
            OPENAI_RESP_MIN,
            Limits::default(),
        )
        .expect("supported");
        let si: serde_json::Value = serde_json::from_str(
            s.system_instructions
                .as_deref()
                .expect("system_instructions"),
        )
        .unwrap();
        assert_eq!(si[0]["content"], "instruction");
    }

    #[test]
    fn openai_unknown_content_block_sets_unmapped_flag() {
        let req = br#"{"model":"gpt-4o","messages":[{"role":"user","content":[{"type":"future_thing","x":1}]}]}"#;
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            req,
            OPENAI_RESP_MIN,
            Limits::default(),
        )
        .expect("supported");
        assert!(s.input_messages_has_unmapped);
        let im: serde_json::Value =
            serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
        assert_eq!(im[0]["parts"][0]["type"], "future_thing");
    }

    #[test]
    fn openai_input_messages_none_when_absent() {
        // request with no messages field → both input_messages/system_instructions are None
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            OPENAI_REQ,
            OPENAI_CHAT,
            Limits::default(),
        )
        .expect("supported");
        assert!(s.input_messages.is_none());
        assert!(s.system_instructions.is_none());
        assert!(!s.input_messages_has_unmapped);
    }

    // --- OpenAI input media parsing ---

    #[test]
    fn openai_image_url_data_uri_to_blob_part() {
        let req = r#"{"model":"gpt-4o","messages":[{"role":"user","content":[
            {"type":"text","text":"desc"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}
        ]}]}"#
            .as_bytes();
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            req,
            OPENAI_RESP_MIN,
            Limits::default(),
        )
        .expect("supported");
        let im: serde_json::Value =
            serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
        let parts = im[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "blob");
        assert_eq!(parts[1]["modality"], "image");
        assert_eq!(parts[1]["mime_type"], "image/png");
        assert_eq!(parts[1]["content"], "iVBORw0KGgo=");
        assert!(!s.input_messages_has_unmapped);
    }

    #[test]
    fn openai_image_url_http_to_uri_part() {
        let req = br#"{"model":"gpt-4o","messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"https://example.com/a.png"}}
        ]}]}"#;
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            req,
            OPENAI_RESP_MIN,
            Limits::default(),
        )
        .expect("supported");
        let im: serde_json::Value =
            serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
        let p = &im[0]["parts"][0];
        assert_eq!(p["type"], "uri");
        assert_eq!(p["modality"], "image");
        assert_eq!(p["uri"], "https://example.com/a.png");
    }

    #[test]
    fn openai_input_audio_to_blob_part() {
        let req = br#"{"model":"gpt-4o-audio","messages":[{"role":"user","content":[
            {"type":"input_audio","input_audio":{"data":"UklGRg==","format":"wav"}}
        ]}]}"#;
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            req,
            OPENAI_RESP_MIN,
            Limits::default(),
        )
        .expect("supported");
        let im: serde_json::Value =
            serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
        let p = &im[0]["parts"][0];
        assert_eq!(p["type"], "blob");
        assert_eq!(p["modality"], "audio");
        assert_eq!(p["mime_type"], "audio/wav");
        assert_eq!(p["content"], "UklGRg==");
    }

    #[test]
    fn openai_file_id_to_file_part() {
        let req = br#"{"model":"gpt-4o","messages":[{"role":"user","content":[
            {"type":"file","file":{"file_id":"file-abc"}}
        ]}]}"#;
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            req,
            OPENAI_RESP_MIN,
            Limits::default(),
        )
        .expect("supported");
        let im: serde_json::Value =
            serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
        let p = &im[0]["parts"][0];
        assert_eq!(p["type"], "file");
        assert_eq!(p["modality"], "document");
        assert_eq!(p["file_id"], "file-abc");
    }

    // --- Anthropic input message parsing ---

    const ANTHROPIC_RESP_MIN: &[u8] =
        br#"{"id":"msg_1","model":"claude-3-5-sonnet","stop_reason":"end_turn","content":[]}"#;

    #[test]
    fn anthropic_top_level_system_string_to_system_instructions() {
        let req = r#"{"model":"claude-3-5-sonnet","system":"You are an assistant.","messages":[{"role":"user","content":"hi"}]}"#.as_bytes();
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            req,
            ANTHROPIC_RESP_MIN,
            Limits::default(),
        )
        .expect("supported");
        let si: serde_json::Value =
            serde_json::from_str(s.system_instructions.as_deref().unwrap()).unwrap();
        assert_eq!(si[0]["type"], "text");
        assert_eq!(si[0]["content"], "You are an assistant.");
        // system is not in input.messages
        let im: serde_json::Value =
            serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
        assert_eq!(im.as_array().unwrap().len(), 1);
        assert_eq!(im[0]["role"], "user");
    }

    #[test]
    fn anthropic_system_array_to_system_instructions() {
        let req = br#"{"model":"claude-3-5-sonnet","system":[{"type":"text","text":"A"},{"type":"text","text":"B"}],"messages":[{"role":"user","content":"hi"}]}"#;
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            req,
            ANTHROPIC_RESP_MIN,
            Limits::default(),
        )
        .expect("supported");
        let si: serde_json::Value =
            serde_json::from_str(s.system_instructions.as_deref().unwrap()).unwrap();
        assert_eq!(si.as_array().unwrap().len(), 2);
        assert_eq!(si[1]["content"], "B");
    }

    #[test]
    fn anthropic_tool_result_block_to_tool_call_response_part() {
        let req = r#"{"model":"claude-3-5-sonnet","messages":[
            {"role":"assistant","content":[{"type":"tool_use","id":"tu_1","name":"get_weather","input":{"city":"Seoul"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":"18 degrees"}]}
        ]}"#.as_bytes();
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            req,
            ANTHROPIC_RESP_MIN,
            Limits::default(),
        )
        .expect("supported");
        let im: serde_json::Value =
            serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
        assert_eq!(im[0]["parts"][0]["type"], "tool_call");
        assert_eq!(im[0]["parts"][0]["id"], "tu_1");
        assert_eq!(im[1]["role"], "user");
        assert_eq!(im[1]["parts"][0]["type"], "tool_call_response");
        assert_eq!(im[1]["parts"][0]["id"], "tu_1");
        assert_eq!(im[1]["parts"][0]["response"], "18 degrees");
    }

    #[test]
    fn anthropic_image_base64_source_to_blob_part() {
        let req = br#"{"model":"claude-3-5-sonnet","messages":[{"role":"user","content":[
            {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"/9j/4AAQ="}}
        ]}]}"#;
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            req,
            ANTHROPIC_RESP_MIN,
            Limits::default(),
        )
        .expect("supported");
        let im: serde_json::Value =
            serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
        let p = &im[0]["parts"][0];
        assert_eq!(p["type"], "blob");
        assert_eq!(p["modality"], "image");
        assert_eq!(p["mime_type"], "image/jpeg");
        assert_eq!(p["content"], "/9j/4AAQ=");
    }

    #[test]
    fn anthropic_image_url_source_to_uri_part() {
        let req = br#"{"model":"claude-3-5-sonnet","messages":[{"role":"user","content":[
            {"type":"image","source":{"type":"url","url":"https://example.com/a.jpg"}}
        ]}]}"#;
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            req,
            ANTHROPIC_RESP_MIN,
            Limits::default(),
        )
        .expect("supported");
        let im: serde_json::Value =
            serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
        let p = &im[0]["parts"][0];
        assert_eq!(p["type"], "uri");
        assert_eq!(p["uri"], "https://example.com/a.jpg");
    }

    // --- OpenAI output audio BlobPart ---

    #[test]
    fn openai_output_audio_to_blob_part() {
        let resp = r#"{"id":"c1","model":"gpt-4o-audio","choices":[{"finish_reason":"stop",
            "message":{"role":"assistant","content":"hi",
                "audio":{"id":"a1","data":"UklGRg==","transcript":"hi"}}}]}"#
            .as_bytes();
        let s = parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            OPENAI_REQ,
            resp,
            Limits::default(),
        )
        .expect("supported");
        let om: serde_json::Value =
            serde_json::from_str(s.output_messages.as_deref().unwrap()).unwrap();
        let parts = om[0]["parts"].as_array().unwrap();
        // text + blob coexist
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "blob");
        assert_eq!(parts[1]["modality"], "audio");
        assert_eq!(parts[1]["content"], "UklGRg==");
    }

    /// Even if an orphan partial_json delta arrives without a content_block_start, no ghost tool call must be created.
    /// The Anthropic API guarantees content_block_start arrives before its deltas, so
    /// when a start event is missing (e.g. due to stream truncation) the corresponding delta must be silently ignored.
    #[test]
    fn anthropic_sse_orphan_partial_json_no_ghost_tool_call() {
        let raw = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"c\"}}\n\n",
            // partial_json delta arrives directly without content_block_start(tool_use) → orphan delta
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":1}\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        );
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            br#"{"model":"c","stream":true}"#,
            raw.as_bytes(),
            Limits::default(),
        )
        .unwrap();
        // An orphan partial_json must not create a tool call (parts must be empty).
        // Since stop_reason="end_turn" is present, output_messages must be Some.
        // Wrapping with if let Some would silently let a None regression pass, so a hard unwrap is used.
        let om = s
            .output_messages
            .as_ref()
            .expect("even the orphan case should be Some when finish_reason is present");
        let v: serde_json::Value = serde_json::from_str(om).unwrap();
        let parts = v[0]["parts"].as_array().unwrap();
        assert!(
            parts.is_empty(),
            "an orphan partial_json must not create a ghost tool call: {:?}",
            om
        );
    }

    // --- Anthropic server_tool SSE reassembly ---

    // b"..." byte literals are ASCII-only → since this contains Korean text (weather), str::as_bytes() is used instead.
    const ANTHROPIC_SSE_SERVER_TOOL: &[u8] = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-3-5-sonnet\",\"content\":[]}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"srv_1\",\"name\":\"web_search\",\"input\":{}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"query\\\":\\\"weather\\\"}\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"web_search_tool_result\",\"tool_use_id\":\"srv_1\",\"content\":[{\"type\":\"web_search_result\",\"title\":\"T\"}]}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    ).as_bytes();

    #[test]
    fn anthropic_sse_server_tool_use_reassembled() {
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            b"{}",
            ANTHROPIC_SSE_SERVER_TOOL,
            Limits::default(),
        )
        .expect("supported");
        assert!(s.reassembled_from_stream);
        let om: serde_json::Value =
            serde_json::from_str(s.output_messages.as_deref().unwrap()).unwrap();
        let parts = om[0]["parts"].as_array().unwrap();
        let stc = parts
            .iter()
            .find(|p| p["type"] == "server_tool_call")
            .expect("server_tool_call");
        assert_eq!(stc["name"], "web_search");
        assert_eq!(stc["server_tool_call"]["query"], "weather");
    }

    #[test]
    fn anthropic_sse_web_search_result_reassembled() {
        let s = parse_llm(
            "api.anthropic.com",
            "/v1/messages",
            b"{}",
            ANTHROPIC_SSE_SERVER_TOOL,
            Limits::default(),
        )
        .expect("supported");
        let om: serde_json::Value =
            serde_json::from_str(s.output_messages.as_deref().unwrap()).unwrap();
        let parts = om[0]["parts"].as_array().unwrap();
        let res = parts
            .iter()
            .find(|p| p["type"] == "server_tool_call_response")
            .expect("response part");
        assert_eq!(res["server_tool_call_response"][0]["title"], "T");
    }
}
