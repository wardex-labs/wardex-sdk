//! LLM request/response body → semantic (gen_ai) extraction. framework-agnostic body parser.
//! endpoint = the path's last segments (`endpoint::ENDPOINTS`); provider = host
//! priority + body/SSE shape fallback. fail-safe (parse failure = partial/empty result).

use std::io::Read;

mod anthropic;
pub mod endpoint;
mod openai_chat;
mod openai_embeddings;
mod parts;
#[cfg(test)]
mod tests;
pub mod usage;

use anthropic::{fill_anthropic, reassemble_anthropic};
use endpoint::{Api, Endpoint};
use openai_chat::{fill_openai_chat, reassemble_openai};
use openai_embeddings::fill_openai_embeddings;
pub use parts::{normalize_finish_reason, FINISH_REASONS};
use usage::UsageBounds;
pub use usage::UsageLeaf;

use crate::sse;
use crate::usage::TokenUsage;
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
    /// semconv `openai.api.type`: which OpenAI API shape this call used
    /// (`chat_completions` | `responses`). None for other providers.
    pub api_type: Option<&'static str>,
    /// `openai.request.service_tier`, provider spelling (`auto` included — MAY).
    pub request_service_tier: Option<String>,
    /// `openai.response.service_tier`, as the response reported it.
    pub response_service_tier: Option<String>,
    /// `openai.response.system_fingerprint` (Chat Completions only).
    pub system_fingerprint: Option<String>,
    /// `gen_ai.request.reasoning.level`: the exact string the caller sent
    /// (Responses `reasoning.effort`, Chat `reasoning_effort`, Anthropic
    /// `output_config.effort`).
    pub reasoning_level: Option<String>,
    /// `gen_ai.request.previous_response.id` (Responses chaining).
    pub previous_response_id: Option<String>,
    /// `gen_ai.response.status` (Responses only: queued/in_progress/completed/
    /// incomplete/failed/cancelled).
    pub response_status: Option<String>,
    /// `gen_ai.request.encoding_formats` (embeddings).
    pub encoding_formats: Option<Vec<String>>,
    /// `gen_ai.embeddings.dimension.count` (embeddings request `dimensions`).
    pub embedding_dimensions: Option<i64>,
    /// Every scalar leaf of the provider's usage tree, spelling preserved —
    /// the `wardex.usage.*` mirror (see `semantic/usage.rs`). Bounded by
    /// `max_extra_keys`; what the bound dropped is in `usage_dropped_count`.
    pub usage_leaves: Vec<(String, UsageLeaf)>,
    /// Leaves the bounds dropped from `usage_leaves` (cap + the two
    /// structural sanity bounds — one counter, so leaves + dropped equals the
    /// tree's scalar-leaf count).
    pub usage_dropped_count: u32,
    /// SSE only (None otherwise): whether the stream carried its provider's
    /// terminal event. `Some(false)` feeds a diagnostics counter, not a marker.
    pub stream_terminated: Option<bool>,
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

fn provider_from_host(host: &str) -> Option<&'static str> {
    if host.contains("openai") {
        Some("openai")
    } else if host.contains("anthropic") {
        Some("anthropic")
    } else {
        None
    }
}

/// If the body is SSE, reassemble it for semantic extraction. Otherwise None (falls through to the existing path).
fn try_parse_sse(
    host: &str,
    path: &str,
    req: &[u8],
    decoded: &[u8],
    bounds: UsageBounds,
) -> Option<LlmSemantics> {
    if !sse::looks_like_sse(decoded) {
        return None;
    }
    let events = sse::parse(decoded);
    // The reassembler is selected by Api — path first, SSE grammar second.
    // The old dispatch keyed on the HOST and defaulted to the Chat
    // reassembler, which is exactly what fabricated an empty chat body out
    // of a Responses stream on api.openai.com.
    let matched = endpoint::from_path(path).or_else(|| endpoint::from_sse_shape(&events));
    let _ = host; // provider is implied by the Api; the host adds nothing here
    match matched.map(|e| e.api) {
        Some(Api::OpenAiChatCompletions) => {
            let reassembled = reassemble_openai(&events);
            let body = reassembled.body.clone();
            let mut out = sse_semantics(matched.unwrap(), reassembled);
            fill_openai_chat(&mut out, req, &body, bounds);
            Some(out)
        }
        Some(Api::AnthropicMessages) => {
            let reassembled = reassemble_anthropic(&events);
            let body = reassembled.body.clone();
            let mut out = sse_semantics(matched.unwrap(), reassembled);
            fill_anthropic(&mut out, req, &body, bounds);
            Some(out)
        }
        // No reassembler understands this stream (a Responses stream until
        // the Responses reassembler lands; embeddings never stream). The
        // unidentified branch below is the honest fallback: raw payloads,
        // and the seam marks `sse_unknown_provider` — whose meaning is
        // exactly "no SSE reassembler recognized this stream".
        _ => {
            let raw = events
                .iter()
                .map(|e| e.data.as_str())
                .collect::<Vec<_>>()
                .join("\n")
                .into_bytes();
            Some(LlmSemantics {
                operation: matched.map(|e| e.operation).unwrap_or("chat").to_string(),
                output_type: Some("text".to_string()),
                reassembled_from_stream: true,
                decoded_response: Some(raw),
                ..Default::default()
            })
        }
    }
}

/// The shared SSE scaffold: provider/operation/api_type from the endpoint,
/// the synthetic body, and the terminal verdict.
fn sse_semantics(endpoint: Endpoint, reassembled: parts::Reassembled) -> LlmSemantics {
    LlmSemantics {
        provider: endpoint.api.provider().to_string(),
        operation: endpoint.operation.to_string(),
        api_type: endpoint.api_type,
        reassembled_from_stream: true,
        stream_terminated: Some(reassembled.terminated),
        decoded_response: Some(reassembled.body),
        ..Default::default()
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
    let bounds = UsageBounds::from_limits(&limits);
    if let Some(s) = try_parse_sse(host, path, req, &decoded, bounds) {
        return Some(s);
    }
    // Non-streaming: the ENDPOINT is decided by the path alone (conservative
    // — same reach as before, minus the substring false positives). The body
    // shape decides only the PROVIDER, when the host names none.
    let matched = endpoint::from_path(path)?;
    let provider = provider_from_host(host).or_else(|| {
        serde_json::from_slice::<serde_json::Value>(&decoded)
            .ok()
            .and_then(|v| endpoint::from_body_shape(&v))
            .map(|e| e.api.provider())
    })?;
    let mut out = LlmSemantics {
        provider: provider.to_string(),
        operation: matched.operation.to_string(),
        api_type: matched.api_type,
        decoded_response: Some(decoded.clone()),
        ..Default::default()
    };
    match (provider, matched.api) {
        ("openai", Api::OpenAiChatCompletions) => fill_openai_chat(&mut out, req, &decoded, bounds),
        ("openai", Api::OpenAiEmbeddings) => {
            fill_openai_embeddings(&mut out, req, &decoded, bounds)
        }
        ("anthropic", Api::AnthropicMessages) => fill_anthropic(&mut out, req, &decoded, bounds),
        // A provider on another provider's API (and, until its parser lands,
        // the Responses API) is left as empty semantics — same fail-safe
        // posture as before.
        _ => {}
    }
    Some(out)
}

#[cfg(test)]
mod bench {
    //! `#[ignore]` timing probes for the semantic parser (design gate: the
    //! usage-model additions must cost <= 10% on the existing paths, and the
    //! Responses path must land within +-20% of the Chat path). Run with:
    //! `cargo test --release -p wardex-protocol -- --ignored bench_parse_llm --nocapture`
    use super::*;

    const ITERS: u32 = 10_000;

    fn bench(name: &str, host: &str, path: &str, req: &[u8], resp: &[u8]) {
        use std::hint::black_box;
        use std::time::Instant;
        let mut medians = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            for _ in 0..ITERS {
                let s = parse_llm(
                    black_box(host),
                    black_box(path),
                    black_box(req),
                    black_box(resp),
                    Limits::default(),
                );
                black_box(&s);
            }
            medians.push(start.elapsed().as_nanos() as f64 / f64::from(ITERS));
        }
        medians.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{name}: {:.0} ns/op (median of 3 x {ITERS})", medians[1]);
    }

    fn chat_request() -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Summarize the following. ".repeat(20)}
            ]
        }))
        .unwrap()
    }

    #[test]
    #[ignore = "timing probe: run explicitly with --ignored --nocapture"]
    fn bench_parse_llm_openai_chat_2k() {
        let content = "The quick brown fox jumps over the lazy dog. ".repeat(40);
        let resp = serde_json::to_vec(&serde_json::json!({
            "id": "chatcmpl-bench", "model": "gpt-4o-mini-2024-07-18",
            "choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 812, "completion_tokens": 460,
                      "prompt_tokens_details": {"cached_tokens": 512},
                      "completion_tokens_details": {"reasoning_tokens": 32}}
        }))
        .unwrap();
        assert!(resp.len() >= 2000, "body under target size: {}", resp.len());
        bench(
            "openai_chat_2k",
            "api.openai.com",
            "/v1/chat/completions",
            &chat_request(),
            &resp,
        );
    }

    #[test]
    #[ignore = "timing probe: run explicitly with --ignored --nocapture"]
    fn bench_parse_llm_openai_responses_3k() {
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(56);
        let req = serde_json::to_vec(&serde_json::json!({
            "model": "gpt-4.1", "input": "Summarize the following.",
            "instructions": "You are a helpful assistant.",
            "reasoning": {"effort": "medium"}, "max_output_tokens": 1024
        }))
        .unwrap();
        let resp = serde_json::to_vec(&serde_json::json!({
            "id": "resp_bench", "object": "response", "status": "completed",
            "model": "gpt-4.1-2025-04-14",
            "output": [
                {"id": "rs_1", "type": "reasoning", "summary": [
                    {"type": "summary_text", "text": "Thinking through the summary."}]},
                {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": text, "annotations": []}]}
            ],
            "usage": {"input_tokens": 812, "output_tokens": 460,
                      "input_tokens_details": {"cached_tokens": 512},
                      "output_tokens_details": {"reasoning_tokens": 32},
                      "total_tokens": 1272}
        }))
        .unwrap();
        assert!(resp.len() >= 3000, "body under target size: {}", resp.len());
        bench(
            "openai_responses_3k",
            "api.openai.com",
            "/v1/responses",
            &req,
            &resp,
        );
    }

    #[test]
    #[ignore = "timing probe: run explicitly with --ignored --nocapture"]
    fn bench_parse_llm_anthropic_2k() {
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(40);
        let req = serde_json::to_vec(&serde_json::json!({
            "model": "claude-sonnet-4-6", "max_tokens": 1024,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Summarize the following."}]
        }))
        .unwrap();
        let resp = serde_json::to_vec(&serde_json::json!({
            "id": "msg_bench", "model": "claude-sonnet-4-6", "type": "message",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 812, "output_tokens": 460,
                      "cache_read_input_tokens": 512,
                      "cache_creation_input_tokens": 128,
                      "cache_creation": {"ephemeral_5m_input_tokens": 128,
                                          "ephemeral_1h_input_tokens": 0},
                      "server_tool_use": {"web_search_requests": 0},
                      "service_tier": "standard"}
        }))
        .unwrap();
        assert!(resp.len() >= 2000, "body under target size: {}", resp.len());
        bench(
            "anthropic_2k",
            "api.anthropic.com",
            "/v1/messages",
            &req,
            &resp,
        );
    }

    #[test]
    #[ignore = "timing probe: run explicitly with --ignored --nocapture"]
    fn bench_parse_llm_openai_chat_sse_50ev() {
        let mut sse = String::new();
        for i in 0..48 {
            sse.push_str(&format!(
                "data: {{\"id\":\"chatcmpl-s\",\"model\":\"gpt-4o-mini\",\"choices\":[{{\"delta\":{{\"content\":\"chunk {i} of the answer \"}},\"finish_reason\":null}}]}}\n\n"
            ));
        }
        sse.push_str("data: {\"id\":\"chatcmpl-s\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":48}}\n\n");
        sse.push_str("data: [DONE]\n\n");
        bench(
            "openai_chat_sse_50ev",
            "api.openai.com",
            "/v1/chat/completions",
            &chat_request(),
            sse.as_bytes(),
        );
    }

    #[test]
    #[ignore = "timing probe: run explicitly with --ignored --nocapture"]
    fn bench_parse_llm_openai_responses_sse_50ev() {
        let mut sse = String::new();
        sse.push_str("event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_s\",\"object\":\"response\",\"status\":\"in_progress\",\"model\":\"gpt-4.1\",\"output\":[],\"usage\":null}}\n\n");
        sse.push_str("event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}\n\n");
        let mut text = String::new();
        for i in 0..46 {
            let piece = format!("chunk {i} of the answer ");
            text.push_str(&piece);
            sse.push_str(&format!(
                "event: response.output_text.delta\ndata: {{\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"{piece}\"}}\n\n"
            ));
        }
        let done = serde_json::json!({
            "type": "response.output_item.done", "output_index": 0,
            "item": {"id": "msg_1", "type": "message", "role": "assistant",
                     "status": "completed",
                     "content": [{"type": "output_text", "text": text, "annotations": []}]}
        });
        sse.push_str(&format!(
            "event: response.output_item.done\ndata: {done}\n\n"
        ));
        let completed = serde_json::json!({
            "type": "response.completed", "sequence_number": 49,
            "response": {"id": "resp_s", "object": "response", "status": "completed",
                         "model": "gpt-4.1",
                         "output": [{"id": "msg_1", "type": "message", "role": "assistant",
                                     "status": "completed",
                                     "content": [{"type": "output_text", "text": text,
                                                   "annotations": []}]}],
                         "usage": {"input_tokens": 10, "output_tokens": 46,
                                    "input_tokens_details": {"cached_tokens": 0},
                                    "output_tokens_details": {"reasoning_tokens": 0},
                                    "total_tokens": 56}}
        });
        sse.push_str(&format!("event: response.completed\ndata: {completed}\n\n"));
        bench(
            "openai_responses_sse_50ev",
            "api.openai.com",
            "/v1/responses",
            b"{\"model\":\"gpt-4.1\",\"input\":\"hi\",\"stream\":true}",
            sse.as_bytes(),
        );
    }
}
