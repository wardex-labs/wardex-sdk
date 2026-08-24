//! Shared OTel message-part builders: text/blob/uri/file/tool-call parts,
//! the `InMsg`/`OutMsg` intermediate shapes, and finish-reason mapping.
//! Provider modules (`openai_chat`, `anthropic`, …) build with these so every
//! endpoint emits the same part vocabulary.

/// Builds a single OTel ToolCallRequestPart (id included only when Some).
pub(super) fn tool_call_part(
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
pub(super) fn tool_call_response_part(
    id: Option<String>,
    response: serde_json::Value,
) -> serde_json::Value {
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
pub(super) fn modality_from_mime(mime: &str) -> &'static str {
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
pub(super) fn blob_part(
    modality: &str,
    mime_type: Option<&str>,
    content_b64: &str,
) -> serde_json::Value {
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
pub(super) fn uri_part(modality: &str, mime_type: Option<&str>, uri: &str) -> serde_json::Value {
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
pub(super) fn file_part(
    modality: &str,
    mime_type: Option<&str>,
    file_id: &str,
) -> serde_json::Value {
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
pub(super) fn parse_data_url(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, b64) = rest.split_once(',')?;
    let mime = meta.strip_suffix(";base64")?;
    Some((mime, b64))
}

/// Intermediate representation of a single input message (no finish_reason).
pub(super) struct InMsg {
    pub(super) role: String,
    pub(super) parts: Vec<serde_json::Value>,
}

/// InMsg[] → OTel ChatMessage[] JSON. Messages with no parts are excluded; None if all are excluded.
pub(super) fn build_input_messages(msgs: Vec<InMsg>) -> Option<String> {
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
pub(super) fn build_system_instructions(parts: Vec<serde_json::Value>) -> Option<String> {
    if parts.is_empty() {
        return None;
    }
    serde_json::to_string(&serde_json::Value::Array(parts)).ok()
}

/// Intermediate representation of a single OutputMessage.
pub(super) struct OutMsg {
    pub(super) parts: Vec<serde_json::Value>,
    pub(super) finish_reason: Option<String>,
}

/// OutMsg list → OTel OutputMessage[] JSON string. Messages with neither parts nor finish are excluded; None if all are excluded.
pub(super) fn build_output_messages(msgs: Vec<OutMsg>) -> Option<String> {
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

pub(super) fn text_part(content: String) -> serde_json::Value {
    serde_json::json!({"type": "text", "content": content})
}

pub(super) fn reasoning_part(content: String) -> serde_json::Value {
    serde_json::json!({"type": "reasoning", "content": content})
}

pub(super) fn generic_part(type_str: &str) -> serde_json::Value {
    serde_json::json!({"type": type_str})
}

pub(super) fn server_tool_call_part(
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

pub(super) fn server_tool_call_response_part(
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

/// The closed set every KNOWN provider finish reason maps into. An UNKNOWN
/// raw value passes through in the provider's own spelling instead of being
/// dropped: a new finish reason the provider invents shows up under its own
/// name rather than silently vanishing.
pub const FINISH_REASONS: &[&str] = &["stop", "length", "tool_call", "content_filter", "error"];

/// The ONE producer of `gen_ai.response.finish_reasons` and of
/// `OutMsg.finish_reason`, total over its input. Three endpoints and the
/// Agent SDK assembler (through the PyO3 export) all spell a stop through
/// this function, so one fact cannot reach the wire in two spellings —
/// `["tool_calls"]` from Chat next to `["tool_call"]` from Responses was the
/// exact split this replaces, on the one attribute dashboards group by.
pub fn normalize_finish_reason(provider: &str, raw: &str) -> String {
    let v = match (provider, raw) {
        (_, "stop") | ("anthropic", "end_turn") | ("anthropic", "stop_sequence") => "stop",
        (_, "length") | ("anthropic", "max_tokens") | ("openai", "max_output_tokens") => "length",
        (_, "tool_calls") | (_, "function_call") | ("anthropic", "tool_use") => "tool_call",
        (_, "content_filter") | ("anthropic", "refusal") => "content_filter",
        ("openai", "failed") | ("openai", "cancelled") => "error",
        _ => return raw.to_string(), // total: unknown passes through as itself
    };
    v.to_string()
}

/// One reassembled SSE stream: the synthetic body plus whether the stream
/// carried its provider's terminal event. `terminated == false` is the
/// volume-evidence condition the seam counts
/// (`interceptors.seam.stream_unterminated`) — no marker is minted for it
/// until that counter shows real volume.
pub(super) struct Reassembled {
    pub(super) body: Vec<u8>,
    pub(super) terminated: bool,
}
