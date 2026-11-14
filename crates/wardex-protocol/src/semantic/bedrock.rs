//! Amazon Bedrock Converse and ConverseStream: request/response fill, the
//! model id from the path, and the event stream reassembly.
//!
//! Converse carries no model id in either body (the path names it) and no
//! response id or response model at all, so those stay None rather than
//! being made up. ConverseStream answers in the AWS event stream encoding
//! (`crate::aws_eventstream`), not SSE; `reassemble_converse_stream` turns
//! its events into one synthetic Converse response, as the SSE reassemblers
//! do for their providers.

use serde::Deserialize;

use super::event_json::parse_event;
use super::parts::{
    blob_part, build_input_messages, build_output_messages, build_system_instructions,
    declared_error_type, generic_part, normalize_finish_reason, reasoning_part, text_part,
    tool_call_part, tool_call_response_part, uri_part, InMsg, OutMsg, Reassembled, FINISH_ERROR,
};
use super::usage::{deep_merge, flatten_usage, UsageBounds, UsageView};
use super::LlmSemantics;
use crate::aws_eventstream;
use crate::usage::{InputConvention, TokenUsage};

/// The provider label, spelled as the semconv `gen_ai.provider.name` value.
pub(super) const PROVIDER: &str = "aws.bedrock";

// Normalization table, same contract as the other providers' (`anthropic.rs`):
// the extraction reads through these constants. `inputTokens` excludes the
// cache tiers reported beside it, as on Anthropic's own API; Converse reports
// no reasoning-token count, so the table has no row for one.
const P_INPUT: &str = "inputTokens";
const P_OUTPUT: &str = "outputTokens";
const P_CACHE_READ: &str = "cacheReadInputTokens";
const P_CACHE_CREATION: &str = "cacheWriteInputTokens";
// Read only by the fixture test (`every_normalized_usage_path_is_extracted_
// from_its_fixture`); the runtime reads the P_* consts the table is built
// from, which makes the two one string.
#[cfg(test)]
pub(super) const NORMALIZED_USAGE_PATHS: &[(&str, &str)] = &[
    ("input_tokens", P_INPUT),
    ("output_tokens", P_OUTPUT),
    ("cache_read_input_tokens", P_CACHE_READ),
    ("cache_creation_input_tokens", P_CACHE_CREATION),
];

#[derive(Deserialize)]
struct ConverseResponse {
    #[serde(default)]
    output: Option<serde_json::Value>,
    #[serde(default, rename = "stopReason")]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<serde_json::Value>,
    /// Only on the synthetic body of a stream that carried an exception
    /// message (`reassemble_converse_stream`): `{"type", "message"}`.
    #[serde(default)]
    error: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ConverseRequest {
    #[serde(default)]
    messages: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    system: Option<Vec<serde_json::Value>>,
    #[serde(default, rename = "inferenceConfig")]
    inference_config: Option<InferenceConfig>,
}

#[derive(Deserialize)]
struct InferenceConfig {
    #[serde(default, rename = "maxTokens")]
    max_tokens: Option<i64>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default, rename = "topP")]
    top_p: Option<f64>,
    #[serde(default, rename = "stopSequences")]
    stop_sequences: Option<Vec<String>>,
}

/// The model id the path names (the segment before `converse` or
/// `converse-stream`), percent-decoded: the SDKs escape the `:` of a version
/// suffix (`...-v2%3A0`) and the `/` of an ARN. A foundation model id, an
/// inference profile id (`us.anthropic...`) and an ARN all pass through as
/// the caller wrote them.
pub(super) fn model_id_from_path(path: &str) -> Option<String> {
    let segments = super::endpoint::segments_of(path);
    let [.., model, action] = segments.as_slice() else {
        return None;
    };
    if *action != "converse" && *action != "converse-stream" {
        return None;
    }
    let decoded = percent_decode(model);
    (!decoded.is_empty()).then_some(decoded)
}

/// `%XX` decoding; a malformed escape stays as written, and bytes that do not
/// decode to UTF-8 leave the whole segment as written.
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            // A hex digit is below 16, so two of them always fit one byte.
            let hex = |c: u8| {
                char::from(c)
                    .to_digit(16)
                    .and_then(|d| u8::try_from(d).ok())
            };
            if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                out.push(h * 16 + l);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| s.to_string())
}

/// Converse message content blocks → OTel parts. A `cachePoint` is a cache
/// boundary the caller placed, not content, so it yields no part.
fn content_to_parts(
    content: Option<&serde_json::Value>,
    out: &mut LlmSemantics,
) -> Vec<serde_json::Value> {
    let mut parts = Vec::new();
    let Some(blocks) = content.and_then(|c| c.as_array()) else {
        return parts;
    };
    for b in blocks {
        if let Some(t) = b.get("text").and_then(|x| x.as_str()) {
            parts.push(text_part(t.to_string()));
        } else if let Some(tu) = b.get("toolUse") {
            parts.push(tool_use_part(tu, out));
        } else if let Some(tr) = b.get("toolResult") {
            let id = str_field(tr, "toolUseId");
            let response = tr.get("content").cloned().unwrap_or_default();
            parts.push(tool_call_response_part(id, response));
        } else if let Some(rc) = b.get("reasoningContent") {
            parts.push(reasoning_content_part(rc));
        } else if let Some((kind, media)) = ["image", "document", "video"]
            .iter()
            .find_map(|k| b.get(*k).map(|m| (*k, m)))
        {
            parts.push(media_part(kind, media, out));
        } else if b.get("cachePoint").is_none() {
            parts.push(generic_part(first_key(b)));
            out.input_messages_has_unmapped = true;
        }
    }
    parts
}

/// The block's own name (its single key), for the GenericPart fallback.
fn first_key(b: &serde_json::Value) -> &str {
    b.as_object()
        .and_then(|o| o.keys().next())
        .map_or("unknown", String::as_str)
}

fn str_field(v: &serde_json::Value, key: &str) -> Option<String> {
    v.get(key).and_then(|x| x.as_str()).map(str::to_string)
}

/// A `toolUse` block. Its `input` is a JSON document on the wire; a stream
/// whose accumulated input string did not parse leaves that string in the
/// synthetic body, which is carried as-is and flagged as unparsed arguments.
fn tool_use_part(tu: &serde_json::Value, out: &mut LlmSemantics) -> serde_json::Value {
    let id = str_field(tu, "toolUseId");
    let name = str_field(tu, "name").unwrap_or_default();
    let arguments = tu.get("input").cloned().unwrap_or_default();
    if arguments.is_string() {
        out.tool_args_unparsed = true;
    }
    tool_call_part(id, name, arguments)
}

/// `reasoningContent`: readable reasoning text, or a redacted block that
/// carries none (kept as a bare reasoning part, as Anthropic's
/// `redacted_thinking` is).
fn reasoning_content_part(rc: &serde_json::Value) -> serde_json::Value {
    match rc
        .get("reasoningText")
        .and_then(|r| r.get("text"))
        .and_then(|x| x.as_str())
    {
        Some(t) => reasoning_part(t.to_string()),
        None => generic_part("reasoning"),
    }
}

/// An `image`/`document`/`video` block: inline bytes (base64 on the JSON
/// wire) become a BlobPart, an S3 location a UriPart.
fn media_part(kind: &str, media: &serde_json::Value, out: &mut LlmSemantics) -> serde_json::Value {
    let format = media.get("format").and_then(|x| x.as_str()).unwrap_or("");
    let mime = mime_of(kind, format);
    let source = media.get("source");
    if let Some(data) = source.and_then(|s| s.get("bytes")).and_then(|x| x.as_str()) {
        return blob_part(kind, mime.as_deref(), data);
    }
    if let Some(uri) = source
        .and_then(|s| s.get("s3Location"))
        .and_then(|l| l.get("uri"))
        .and_then(|x| x.as_str())
    {
        return uri_part(kind, mime.as_deref(), uri);
    }
    out.input_messages_has_unmapped = true;
    generic_part(kind)
}

/// Converse names a format, not a MIME type. Only the spellings whose MIME
/// type is unambiguous are translated; any other leaves the part without one.
fn mime_of(kind: &str, format: &str) -> Option<String> {
    let mime = match (kind, format) {
        ("image", "jpeg") => "image/jpeg",
        ("image", "png") => "image/png",
        ("image", "gif") => "image/gif",
        ("image", "webp") => "image/webp",
        ("document", "pdf") => "application/pdf",
        ("document", "csv") => "text/csv",
        ("document", "html") => "text/html",
        ("document", "txt") => "text/plain",
        ("document", "md") => "text/markdown",
        ("video", "mp4") => "video/mp4",
        ("video", "webm") => "video/webm",
        ("video", "mov") => "video/quicktime",
        _ => return None,
    };
    Some(mime.to_string())
}

pub(super) fn fill_bedrock_converse(
    out: &mut LlmSemantics,
    req: &[u8],
    resp: &[u8],
    bounds: UsageBounds,
) {
    out.output_type = Some("text".to_string());
    if let Ok(r) = serde_json::from_slice::<ConverseResponse>(resp) {
        // Only a body with `output` is a response object; a bare HTTP error
        // envelope (`{"message": "..."}`, its class in a response header) is
        // not, and its span keeps taking its error from the HTTP status.
        let message = r.output.as_ref().and_then(|o| o.get("message"));
        let declared = r
            .error
            .as_ref()
            .filter(|e| r.output.is_some() && !e.is_null())
            .map(declared_error_type);
        let finish_reason = if declared.is_some() {
            Some(FINISH_ERROR.to_string())
        } else {
            r.stop_reason
                .as_deref()
                .map(|sr| normalize_finish_reason(PROVIDER, sr))
        };
        if let Some(f) = &finish_reason {
            out.finish_reasons = Some(vec![f.clone()]);
        }
        out.error_type = declared;
        if let Some(u) = r.usage.filter(|u| !u.is_null()) {
            let view = UsageView(&u);
            out.usage = TokenUsage::new(
                InputConvention::ExcludesCache,
                view.i64_at(P_INPUT),
                view.i64_at(P_OUTPUT),
                view.i64_at(P_CACHE_READ),
                view.i64_at(P_CACHE_CREATION),
                None,
            );
            let flat = flatten_usage(&u, bounds);
            out.usage_leaves = flat.leaves;
            out.usage_dropped_count = flat.dropped;
        }
        if r.output.is_some() {
            let mut parts = Vec::new();
            for b in message
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_array())
                .into_iter()
                .flatten()
            {
                if let Some(t) = b.get("text").and_then(|x| x.as_str()) {
                    if !t.is_empty() {
                        parts.push(text_part(t.to_string()));
                    }
                } else if let Some(tu) = b.get("toolUse") {
                    parts.push(tool_use_part(tu, out));
                } else if let Some(rc) = b.get("reasoningContent") {
                    parts.push(reasoning_content_part(rc));
                } else {
                    parts.push(generic_part(first_key(b)));
                    out.output_messages_has_unmapped = true;
                }
            }
            out.output_messages = build_output_messages(vec![OutMsg {
                role: None,
                parts,
                finish_reason,
            }]);
        }
    }
    if let Ok(q) = serde_json::from_slice::<ConverseRequest>(req) {
        if let Some(c) = q.inference_config {
            out.max_tokens = c.max_tokens;
            out.temperature = c.temperature;
            out.top_p = c.top_p;
            out.stop_sequences = c.stop_sequences;
        }
        let sys_parts = q
            .system
            .iter()
            .flatten()
            .filter_map(|b| b.get("text").and_then(|x| x.as_str()))
            .map(|t| text_part(t.to_string()))
            .collect();
        out.system_instructions = build_system_instructions(sys_parts);
        let mut msgs = Vec::new();
        for m in q.messages.iter().flatten() {
            let role = m.get("role").and_then(|x| x.as_str()).unwrap_or("");
            let parts = content_to_parts(m.get("content"), out);
            msgs.push(InMsg {
                role: role.to_string(),
                parts,
            });
        }
        out.input_messages = build_input_messages(msgs);
    }
}

/// One content block of the stream, by `contentBlockIndex`.
enum Block {
    Text(String),
    Reasoning {
        text: String,
        redacted: bool,
    },
    ToolUse {
        id: Option<String>,
        name: Option<String>,
        input: String,
    },
}

/// ConverseStream event messages → one synthetic Converse response body.
///
/// Terminal = `messageStop`, or an exception message (the provider ended the
/// stream with a declaration). `metadata`, which carries the usage, follows
/// `messageStop`, so a stream cut between the two is terminated and has no
/// usage, which is what the bytes say. Unparsed counts frames the framing
/// decoder could not read plus event payloads that are not JSON; either may
/// have carried text, a stop reason or the usage.
pub(super) fn reassemble_converse_stream(body: &[u8]) -> Reassembled {
    use std::collections::BTreeMap;
    let decoded = aws_eventstream::decode(body);
    let mut unparsed = decoded.unparsed;
    let mut role: Option<String> = None;
    let mut blocks: BTreeMap<i64, Block> = BTreeMap::new();
    let mut stop_reason: Option<String> = None;
    let mut usage: Option<serde_json::Value> = None;
    let mut metrics: Option<serde_json::Value> = None;
    let mut error: Option<serde_json::Value> = None;
    let mut terminated = false;
    for msg in &decoded.messages {
        let payload = std::str::from_utf8(msg.payload).ok();
        let value = payload.and_then(parse_event);
        match msg.header_str(":message-type") {
            Some("exception") => {
                let message = value
                    .as_ref()
                    .and_then(|v| v.get("message"))
                    .cloned()
                    .unwrap_or_default();
                error = Some(serde_json::json!({
                    "type": msg.header_str(":exception-type").unwrap_or_default(),
                    "message": message,
                }));
                terminated = true;
                continue;
            }
            // A service error: its class and text travel in headers, the
            // payload is empty.
            Some("error") => {
                error = Some(serde_json::json!({
                    "type": msg.header_str(":error-code").unwrap_or_default(),
                    "message": msg.header_str(":error-message").unwrap_or_default(),
                }));
                terminated = true;
                continue;
            }
            _ => {}
        }
        let Some(v) = value else {
            let empty = payload.is_some_and(|p| p.trim().is_empty());
            unparsed = unparsed.saturating_add(u32::from(!empty));
            continue;
        };
        let index = v
            .get("contentBlockIndex")
            .and_then(|x| x.as_i64())
            .unwrap_or(0);
        match msg.header_str(":event-type") {
            Some("messageStart") => role = str_field(&v, "role"),
            Some("contentBlockStart") => {
                if let Some(tu) = v.get("start").and_then(|s| s.get("toolUse")) {
                    blocks.insert(
                        index,
                        Block::ToolUse {
                            id: str_field(tu, "toolUseId"),
                            name: str_field(tu, "name"),
                            input: String::new(),
                        },
                    );
                }
            }
            Some("contentBlockDelta") => {
                let Some(delta) = v.get("delta") else {
                    continue;
                };
                if let Some(t) = delta.get("text").and_then(|x| x.as_str()) {
                    if let Block::Text(acc) = blocks
                        .entry(index)
                        .or_insert_with(|| Block::Text(String::new()))
                    {
                        acc.push_str(t);
                    }
                } else if let Some(rc) = delta.get("reasoningContent") {
                    if let Block::Reasoning { text, redacted } =
                        blocks.entry(index).or_insert_with(|| Block::Reasoning {
                            text: String::new(),
                            redacted: false,
                        })
                    {
                        if let Some(t) = rc.get("text").and_then(|x| x.as_str()) {
                            text.push_str(t);
                        }
                        if rc.get("redactedContent").is_some() {
                            *redacted = true;
                        }
                    }
                } else if let Some(pj) = delta
                    .get("toolUse")
                    .and_then(|t| t.get("input"))
                    .and_then(|x| x.as_str())
                {
                    // Only into a block a `contentBlockStart` registered: an
                    // orphan delta (a stream cut before its start) has no
                    // name to hang it on.
                    if let Some(Block::ToolUse { input, .. }) = blocks.get_mut(&index) {
                        input.push_str(pj);
                    }
                }
            }
            Some("messageStop") => {
                stop_reason = str_field(&v, "stopReason");
                terminated = true;
            }
            Some("metadata") => {
                if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
                    match usage.as_mut() {
                        Some(base) => deep_merge(base, u),
                        None => usage = Some(u.clone()),
                    }
                }
                if let Some(m) = v.get("metrics").filter(|m| !m.is_null()) {
                    metrics = Some(m.clone());
                }
            }
            _ => {}
        }
    }
    let content: Vec<serde_json::Value> = blocks
        .into_values()
        .map(|b| match b {
            Block::Text(t) => serde_json::json!({"text": t}),
            Block::Reasoning { text, redacted } => {
                if redacted && text.is_empty() {
                    serde_json::json!({"reasoningContent": {"redactedContent": ""}})
                } else {
                    serde_json::json!({"reasoningContent": {"reasoningText": {"text": text}}})
                }
            }
            Block::ToolUse { id, name, input } => {
                // An input that does not parse stays the string it was: the
                // fill carries it as-is and flags the arguments unparsed.
                let input = if input.is_empty() {
                    serde_json::json!({})
                } else {
                    serde_json::from_str(&input).unwrap_or(serde_json::Value::String(input))
                };
                serde_json::json!({"toolUse": {"toolUseId": id, "name": name, "input": input}})
            }
        })
        .collect();
    let mut obj = serde_json::json!({
        "output": {"message": {"role": role.unwrap_or_else(|| "assistant".to_string()),
                               "content": content}},
        "stopReason": stop_reason,
    });
    if let Some(u) = usage {
        obj["usage"] = u;
    }
    if let Some(m) = metrics {
        obj["metrics"] = m;
    }
    if let Some(e) = error {
        obj["error"] = e;
    }
    Reassembled {
        body: serde_json::to_vec(&obj).unwrap_or_default(),
        terminated,
        unparsed,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{normalize_finish_reason, parse_llm, LlmSemantics, FINISH_REASONS};
    use wardex_limits::Limits;

    macro_rules! fixture {
        ($case:literal, $file:literal) => {
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/llm/",
                $case,
                "/",
                $file
            )) as &[u8]
        };
    }

    fn parsed(host: &str, path: &str, req: &[u8], resp: &[u8]) -> LlmSemantics {
        match parse_llm(host, path, req, resp, Limits::default()) {
            Some(s) => s,
            None => panic!("{host}{path}: no semantics"),
        }
    }

    const BEDROCK_HOST: &str = "bedrock-runtime.us-east-1.amazonaws.com";
    const BEDROCK_CONVERSE_PATH: &str =
        "/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/converse";
    const BEDROCK_STREAM_PATH: &str =
        "/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/converse-stream";

    fn converse_event(name: &str, payload: serde_json::Value) -> Vec<u8> {
        crate::aws_eventstream::encode_message(
            &[
                (":event-type", name),
                (":content-type", "application/json"),
                (":message-type", "event"),
            ],
            payload.to_string().as_bytes(),
        )
    }

    fn converse_exception(kind: &str, message: &str) -> Vec<u8> {
        crate::aws_eventstream::encode_message(
            &[
                (":exception-type", kind),
                (":content-type", "application/json"),
                (":message-type", "exception"),
            ],
            serde_json::json!({ "message": message })
                .to_string()
                .as_bytes(),
        )
    }

    /// A ConverseStream answer: reasoning, text in two deltas, one tool call
    /// whose input arrives in two pieces, the stop, then the usage.
    fn converse_stream_frames() -> Vec<Vec<u8>> {
        use serde_json::json;
        vec![
            converse_event("messageStart", json!({"role": "assistant"})),
            converse_event(
                "contentBlockDelta",
                json!({"contentBlockIndex": 0,
                   "delta": {"reasoningContent": {"text": "Weather tool."}}}),
            ),
            converse_event(
                "contentBlockDelta",
                json!({"contentBlockIndex": 0,
                   "delta": {"reasoningContent": {"signature": "sig"}}}),
            ),
            converse_event("contentBlockStop", json!({"contentBlockIndex": 0})),
            converse_event(
                "contentBlockDelta",
                json!({"contentBlockIndex": 1, "delta": {"text": "Let me "}}),
            ),
            converse_event(
                "contentBlockDelta",
                json!({"contentBlockIndex": 1, "delta": {"text": "check."}}),
            ),
            converse_event("contentBlockStop", json!({"contentBlockIndex": 1})),
            converse_event(
                "contentBlockStart",
                json!({"contentBlockIndex": 2,
                   "start": {"toolUse": {"toolUseId": "tooluse_1", "name": "get_weather"}}}),
            ),
            converse_event(
                "contentBlockDelta",
                json!({"contentBlockIndex": 2, "delta": {"toolUse": {"input": "{\"city\":"}}}),
            ),
            converse_event(
                "contentBlockDelta",
                json!({"contentBlockIndex": 2, "delta": {"toolUse": {"input": "\"Seoul\"}"}}}),
            ),
            converse_event("contentBlockStop", json!({"contentBlockIndex": 2})),
            converse_event("messageStop", json!({"stopReason": "tool_use"})),
            converse_event(
                "metadata",
                json!({"usage": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 170,
                             "cacheReadInputTokens": 40, "cacheWriteInputTokens": 10},
                   "metrics": {"latencyMs": 900}}),
            ),
        ]
    }

    fn output_messages(s: &LlmSemantics) -> serde_json::Value {
        serde_json::from_str(s.output_messages.as_deref().expect("output messages")).unwrap()
    }

    #[test]
    fn bedrock_converse_fixture_reads_request_response_and_usage() {
        let s = parsed(
            BEDROCK_HOST,
            BEDROCK_CONVERSE_PATH,
            fixture!("bedrock_converse", "request.json"),
            fixture!("bedrock_converse", "response.json"),
        );
        assert_eq!(s.provider, "aws.bedrock");
        assert!(
            !s.provider_inferred,
            "the regional runtime host is Bedrock's own"
        );
        assert_eq!(s.operation, "chat");
        assert_eq!(s.api_type, None);
        // The path names the model, percent-decoded; the body names none.
        assert_eq!(
            s.request_model.as_deref(),
            Some("anthropic.claude-3-5-sonnet-20241022-v2:0")
        );
        // Converse has neither a response id nor a response model: none invented.
        assert_eq!(s.response_model, None);
        assert_eq!(s.response_id, None);
        // inputTokens excludes the cache tiers; the normalized total adds them.
        assert_eq!(s.usage.input_tokens(), Some(11000));
        assert_eq!(s.usage.output_tokens(), Some(500));
        assert_eq!(s.usage.cache_read_input_tokens(), Some(8000));
        assert_eq!(s.usage.cache_creation_input_tokens(), Some(2000));
        assert_eq!(s.usage.reasoning_output_tokens(), None);
        assert_eq!(s.finish_reasons, Some(vec!["stop".to_string()]));
        assert_eq!(s.max_tokens, Some(1024));
        assert_eq!(s.temperature, Some(0.5));
        assert_eq!(s.top_p, Some(0.9));
        assert_eq!(s.stop_sequences, Some(vec!["END".to_string()]));
        assert_eq!(s.stream, Some(false));
        assert!(!s.reassembled_from_stream);
        assert_eq!(s.stream_terminated, None);
        let system: serde_json::Value =
            serde_json::from_str(s.system_instructions.as_deref().unwrap()).unwrap();
        assert_eq!(
            system,
            serde_json::json!([{"type": "text", "content": "You are a helpful assistant."}]),
            "a cachePoint is a cache boundary, not an instruction"
        );
        let input: serde_json::Value =
            serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["parts"][0]["content"], "Say hello.");
        let out = output_messages(&s);
        assert_eq!(out[0]["role"], "assistant");
        assert_eq!(out[0]["parts"][0]["content"], "Hello!");
        assert_eq!(out[0]["finish_reason"], "stop");
    }

    #[test]
    fn bedrock_converse_tool_use_becomes_a_tool_call_and_profile_ids_pass_through() {
        let s = parsed(
            BEDROCK_HOST,
            "/model/us.anthropic.claude-sonnet-4-20250514-v1%3A0/converse",
            fixture!("bedrock_converse_tools", "request.json"),
            fixture!("bedrock_converse_tools", "response.json"),
        );
        assert_eq!(
            s.request_model.as_deref(),
            Some("us.anthropic.claude-sonnet-4-20250514-v1:0")
        );
        assert_eq!(s.finish_reasons, Some(vec!["tool_call".to_string()]));
        let out = output_messages(&s);
        let parts = out[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["content"], "Let me check.");
        assert_eq!(parts[1]["type"], "tool_call");
        assert_eq!(parts[1]["id"], "tooluse_kZJMlvQmRJ6eAyJE5GIl7Q");
        assert_eq!(parts[1]["name"], "get_weather");
        assert_eq!(parts[1]["arguments"], serde_json::json!({"city": "Seoul"}));
        assert!(!s.tool_args_unparsed);
        assert!(!s.output_messages_has_unmapped);
    }

    /// An ARN model id arrives with its `/` and `:` escaped; it decodes whole.
    #[test]
    fn bedrock_model_arn_in_the_path_is_percent_decoded() {
        let s = parsed(
        BEDROCK_HOST,
        "/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Ainference-profile%2Fus.amazon.nova-pro-v1%3A0/converse",
        b"{}",
        br#"{"output":{"message":{"role":"assistant","content":[]}},"stopReason":"end_turn"}"#,
    );
        assert_eq!(
            s.request_model.as_deref(),
            Some(
                "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-pro-v1:0"
            )
        );
    }

    #[test]
    fn bedrock_stop_reasons_normalize_to_the_closed_set() {
        let rows = [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("max_tokens", "length"),
            ("tool_use", "tool_call"),
            ("guardrail_intervened", "content_filter"),
            ("content_filtered", "content_filter"),
        ];
        for (raw, want) in rows {
            assert_eq!(normalize_finish_reason("aws.bedrock", raw), want, "{raw}");
            assert!(FINISH_REASONS.contains(&want));
            let body = serde_json::json!({
                "output": {"message": {"role": "assistant", "content": [{"text": "x"}]}},
                "stopReason": raw,
            });
            let s = parsed(
                BEDROCK_HOST,
                BEDROCK_CONVERSE_PATH,
                b"{}",
                body.to_string().as_bytes(),
            );
            assert_eq!(s.finish_reasons, Some(vec![want.to_string()]), "{raw}");
        }
        // Unknown spellings pass through as themselves, as for every provider.
        assert_eq!(
            normalize_finish_reason("aws.bedrock", "model_context_window_exceeded"),
            "model_context_window_exceeded"
        );
    }

    /// Every request block kind: text, toolUse, toolResult, inline and S3 media,
    /// a cache point (no part), and an unknown block (generic part, flagged).
    #[test]
    fn bedrock_request_blocks_map_to_parts() {
        let req = serde_json::json!({
            "messages": [
                {"role": "user", "content": [
                    {"text": "Look."},
                    {"image": {"format": "png", "source": {"bytes": "iVBORw0K"}}},
                    {"document": {"format": "pdf", "name": "spec",
                                  "source": {"s3Location": {"uri": "s3://b/spec.pdf"}}}},
                    {"cachePoint": {"type": "default"}}
                ]},
                {"role": "assistant", "content": [
                    {"reasoningContent": {"reasoningText": {"text": "Use the tool.", "signature": "s"}}},
                    {"toolUse": {"toolUseId": "t1", "name": "lookup", "input": {"q": 1}}}
                ]},
                {"role": "user", "content": [
                    {"toolResult": {"toolUseId": "t1", "content": [{"json": {"a": 1}}],
                                    "status": "success"}},
                    {"guardContent": {"text": {"text": "x"}}}
                ]}
            ]
        });
        let s = parsed(
        BEDROCK_HOST,
        BEDROCK_CONVERSE_PATH,
        req.to_string().as_bytes(),
        br#"{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn"}"#,
    );
        let input: serde_json::Value =
            serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
        let user = input[0]["parts"].as_array().unwrap();
        assert_eq!(user.len(), 3, "the cache point yields no part: {user:?}");
        assert_eq!(
            user[1],
            serde_json::json!({"type": "blob", "modality": "image",
                           "mime_type": "image/png", "content": "iVBORw0K"})
        );
        assert_eq!(
            user[2],
            serde_json::json!({"type": "uri", "modality": "document",
                           "mime_type": "application/pdf", "uri": "s3://b/spec.pdf"})
        );
        let assistant = input[1]["parts"].as_array().unwrap();
        assert_eq!(assistant[0]["type"], "reasoning");
        assert_eq!(assistant[1]["type"], "tool_call");
        assert_eq!(assistant[1]["arguments"], serde_json::json!({"q": 1}));
        let tool = input[2]["parts"].as_array().unwrap();
        assert_eq!(tool[0]["type"], "tool_call_response");
        assert_eq!(tool[0]["id"], "t1");
        assert_eq!(tool[0]["response"], serde_json::json!([{"json": {"a": 1}}]));
        assert_eq!(tool[1]["type"], "guardContent");
        assert!(s.input_messages_has_unmapped);
    }

    #[test]
    fn bedrock_converse_stream_reassembles_into_one_response() {
        let body: Vec<u8> = converse_stream_frames().concat();
        let s = parsed(
            BEDROCK_HOST,
            BEDROCK_STREAM_PATH,
            b"{\"messages\":[]}",
            &body,
        );
        assert_eq!(s.provider, "aws.bedrock");
        assert!(!s.provider_inferred);
        assert_eq!(
            s.request_model.as_deref(),
            Some("anthropic.claude-3-5-sonnet-20241022-v2:0")
        );
        assert!(s.reassembled_from_stream);
        assert_eq!(s.stream, Some(true));
        assert_eq!(s.stream_terminated, Some(true));
        assert_eq!(s.stream_events_unparsed, 0);
        assert_eq!(s.finish_reasons, Some(vec!["tool_call".to_string()]));
        assert_eq!(
            s.usage.input_tokens(),
            Some(150),
            "100 + 40 read + 10 write"
        );
        assert_eq!(s.usage.output_tokens(), Some(20));
        assert_eq!(s.usage.cache_read_input_tokens(), Some(40));
        assert_eq!(s.usage.cache_creation_input_tokens(), Some(10));
        assert_eq!(s.error_type, None);
        let out = output_messages(&s);
        let parts = out[0]["parts"].as_array().unwrap();
        assert_eq!(
            parts[0],
            serde_json::json!({"type": "reasoning", "content": "Weather tool."})
        );
        assert_eq!(
            parts[1],
            serde_json::json!({"type": "text", "content": "Let me check."})
        );
        assert_eq!(parts[2]["type"], "tool_call");
        assert_eq!(parts[2]["id"], "tooluse_1");
        assert_eq!(parts[2]["arguments"], serde_json::json!({"city": "Seoul"}));
        assert!(!s.tool_args_unparsed);
        // The synthetic body is Converse-shaped, so it reads as Converse again.
        let body: serde_json::Value =
            serde_json::from_slice(s.decoded_response.as_deref().unwrap()).unwrap();
        assert_eq!(
            super::super::endpoint::from_body_shape(&body).map(|e| e.api),
            Some(super::super::endpoint::Api::BedrockConverse)
        );
    }

    /// An exception message mid-stream (`throttlingException` and friends) is
    /// the provider's failure declaration: the text before it stays, the finish
    /// is `error`, and its exception type is the declared error type.
    #[test]
    fn bedrock_stream_exception_declares_the_failure() {
        let frames = converse_stream_frames();
        let mut body: Vec<u8> = frames[..5].concat();
        body.extend(converse_exception(
            "throttlingException",
            "Too many requests",
        ));
        let s = parsed(BEDROCK_HOST, BEDROCK_STREAM_PATH, b"{}", &body);
        assert_eq!(
            s.stream_terminated,
            Some(true),
            "the exception ends the stream"
        );
        assert_eq!(s.finish_reasons, Some(vec!["error".to_string()]));
        assert_eq!(s.error_type.as_deref(), Some("throttlingException"));
        let decoded: serde_json::Value =
            serde_json::from_slice(s.decoded_response.as_deref().unwrap()).unwrap();
        assert_eq!(decoded["error"]["message"], "Too many requests");
        let out = output_messages(&s);
        assert_eq!(out[0]["finish_reason"], "error");
        assert_eq!(out[0]["parts"][1]["content"], "Let me ");
        // Alone, before anything else, it still declares the failure.
        let s = parsed(
            BEDROCK_HOST,
            BEDROCK_STREAM_PATH,
            b"{}",
            &converse_exception("validationException", "bad input"),
        );
        assert_eq!(s.error_type.as_deref(), Some("validationException"));
        assert_eq!(s.finish_reasons, Some(vec!["error".to_string()]));
    }

    /// A stream cut short: frames before the cut are kept and the cut one is
    /// counted. Cut before `messageStop`, it is unterminated; cut after it (in
    /// the trailing `metadata`), it is terminated but carries no usage.
    #[test]
    fn bedrock_truncated_stream_is_counted_and_unterminated_before_the_stop() {
        let frames = converse_stream_frames();
        let mut body: Vec<u8> = frames[..7].concat();
        body.extend_from_slice(&frames[7][..frames[7].len() / 2]);
        let s = parsed(BEDROCK_HOST, BEDROCK_STREAM_PATH, b"{}", &body);
        assert_eq!(s.stream_terminated, Some(false));
        assert_eq!(s.stream_events_unparsed, 1);
        assert_eq!(s.finish_reasons, None);
        assert_eq!(
            output_messages(&s)[0]["parts"][1]["content"],
            "Let me check."
        );

        let last = frames.len() - 1;
        let mut body: Vec<u8> = frames[..last].concat();
        body.extend_from_slice(&frames[last][..10]);
        let s = parsed(BEDROCK_HOST, BEDROCK_STREAM_PATH, b"{}", &body);
        assert_eq!(s.stream_terminated, Some(true));
        assert_eq!(s.stream_events_unparsed, 1);
        assert_eq!(
            s.usage.input_tokens(),
            None,
            "the usage was in the cut frame"
        );
    }

    /// A frame whose payload is not JSON is skipped and counted; one whose CRC
    /// fails is skipped and counted by the framing decoder.
    #[test]
    fn bedrock_stream_counts_unreadable_events() {
        let frames = converse_stream_frames();
        let mut body: Vec<u8> = frames[..4].concat();
        body.extend(crate::aws_eventstream::encode_message(
            &[
                (":event-type", "contentBlockDelta"),
                (":message-type", "event"),
            ],
            b"{\"contentBlockIndex\":1,\"delta\":{\"te",
        ));
        let mut corrupted = frames[5].clone();
        let at = corrupted.len() - 8;
        corrupted[at] ^= 0x20;
        body.extend(corrupted);
        body.extend(frames[6..].concat());
        let s = parsed(BEDROCK_HOST, BEDROCK_STREAM_PATH, b"{}", &body);
        assert_eq!(s.stream_events_unparsed, 2);
        assert_eq!(s.stream_terminated, Some(true));
        assert_eq!(s.finish_reasons, Some(vec!["tool_call".to_string()]));
    }

    /// A tool input whose accumulated pieces are not JSON is carried as the
    /// string it was, and flagged, rather than replaced with `{}`.
    #[test]
    fn bedrock_stream_unparseable_tool_input_is_kept_and_flagged() {
        let frames = converse_stream_frames();
        let mut body: Vec<u8> = frames[..9].concat();
        body.extend(frames[10..].concat());
        let s = parsed(BEDROCK_HOST, BEDROCK_STREAM_PATH, b"{}", &body);
        assert!(s.tool_args_unparsed);
        assert_eq!(
            output_messages(&s)[0]["parts"][2]["arguments"],
            "{\"city\":"
        );
    }

    /// A stream read from the plain `/converse` path (or through a gateway that
    /// rewrote it) is told by its CRC-valid first message.
    #[test]
    fn bedrock_stream_body_is_sniffed_on_the_plain_path() {
        let body: Vec<u8> = converse_stream_frames().concat();
        let s = parsed(BEDROCK_HOST, BEDROCK_CONVERSE_PATH, b"{}", &body);
        assert!(s.reassembled_from_stream);
        assert_eq!(s.stream, Some(true), "the body proved it streamed");
        assert_eq!(s.stream_terminated, Some(true));
        assert_eq!(s.usage.output_tokens(), Some(20));
    }

    #[test]
    fn bedrock_gzip_response_is_decompressed() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        enc.write_all(fixture!("bedrock_converse", "response.json"))
            .unwrap();
        let gz = enc.finish().unwrap();
        let s = parsed(BEDROCK_HOST, BEDROCK_CONVERSE_PATH, b"{}", &gz);
        assert_eq!(s.usage.output_tokens(), Some(500));
        assert_eq!(s.finish_reasons, Some(vec!["stop".to_string()]));
    }

    /// The label comes from Bedrock's own runtime hosts alone. A gateway or a
    /// private alias on the Converse path is still read as Converse, and ships
    /// no provider name, marked as unproven.
    #[test]
    fn bedrock_label_is_left_empty_off_the_runtime_host() {
        let resp = fixture!("bedrock_converse", "response.json");
        for host in ["127.0.0.1", "llm-gateway.corp", "my-bedrock-proxy.corp"] {
            let s = parsed(host, BEDROCK_CONVERSE_PATH, b"{}", resp);
            assert_eq!(s.provider, "", "{host}");
            assert!(s.provider_inferred, "{host}");
            assert_eq!(s.usage.output_tokens(), Some(500), "{host}: still read");
        }
        let body: Vec<u8> = converse_stream_frames().concat();
        let s = parsed("127.0.0.1", BEDROCK_STREAM_PATH, b"{}", &body);
        assert_eq!(s.provider, "");
        assert!(s.provider_inferred);
        assert!(s.reassembled_from_stream);
        let s = parsed(
            "vpce-0abc-1def.bedrock-runtime.eu-west-1.vpce.amazonaws.com",
            BEDROCK_CONVERSE_PATH,
            b"{}",
            resp,
        );
        assert_eq!(s.provider, "aws.bedrock");
        assert!(!s.provider_inferred);
    }

    /// The HTTP error envelope (`{"message": ...}`, its class in a header) is not
    /// a response object: no finish, no output, no declared error type.
    #[test]
    fn bedrock_bare_error_envelope_claims_nothing_about_the_response() {
        let s = parsed(
            BEDROCK_HOST,
            BEDROCK_CONVERSE_PATH,
            b"{}",
            br#"{"message":"Too many requests, please wait before trying again."}"#,
        );
        assert_eq!(s.finish_reasons, None);
        assert_eq!(s.output_messages, None);
        assert_eq!(s.error_type, None);
        assert_eq!(s.usage.input_tokens(), None);
    }

    /// The same envelope answering a ConverseStream request is an HTTP error,
    /// not a stream whose event could not be read: nothing is reassembled and
    /// nothing is said about stream termination. The request still streamed.
    #[test]
    fn bedrock_error_envelope_on_the_stream_path_is_not_read_as_a_stream() {
        for body in [
            &br#"{"message":"The security token included in the request is invalid."}"#[..],
            &b"  \n{\"message\":\"Too many requests\"}"[..],
        ] {
            let s = parsed(BEDROCK_HOST, BEDROCK_STREAM_PATH, b"{}", body);
            assert!(!s.reassembled_from_stream);
            assert_eq!(s.stream_terminated, None);
            assert_eq!(s.stream_events_unparsed, 0);
            assert_eq!(s.finish_reasons, None);
            assert_eq!(s.error_type, None);
            assert_eq!(s.stream, Some(true));
        }
    }

    /// Neither the legacy InvokeModel API nor anything below a Converse action
    /// is read: no semantics, even with a Converse-shaped body.
    #[test]
    fn bedrock_invoke_and_subpaths_are_not_read() {
        let resp = fixture!("bedrock_converse", "response.json");
        for path in [
            "/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/invoke",
            "/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/invoke-with-response-stream",
            "/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/converse/extra",
        ] {
            assert!(
                parse_llm(BEDROCK_HOST, path, b"{}", resp, Limits::default()).is_none(),
                "{path}"
            );
        }
    }
}
