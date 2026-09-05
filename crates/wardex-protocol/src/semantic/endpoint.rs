//! Endpoint recognition: which provider API a transaction spoke, from the
//! path's LAST segments, from the response body's shape, or from the SSE
//! event grammar. Replaces two substring checks whose false positives were
//! real defects: `path.contains("/messages")` classified
//! `/v1/messages/count_tokens` and `/v1/messages/batches` as chat calls (a
//! false `semantic_parse_failed` on every one), and the missing `/v1/responses`
//! row sent Responses SSE through the Chat reassembler, which fabricated an
//! empty chat body.
//!
//! Two wider questions live beside the LLM table: `treatment` classifies a
//! provider path that is NOT an LLM call (provider-owned state to keep as
//! plain HTTP, or a telemetry upload to exclude), and `ws_upgrade` says
//! whether a WebSocket upgrade on an LLM path may carry LLM calls.

/// Which provider API shape the transaction used. Not the same axis as the
/// provider label: a gateway can serve an OpenAI-shaped API from any host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    OpenAiChatCompletions,
    OpenAiResponses,
    OpenAiEmbeddings,
    AnthropicMessages,
}

impl Api {
    /// The provider label this API shape implies (used when the host says
    /// nothing — the shapes are provider-specific).
    pub(super) fn provider(self) -> &'static str {
        match self {
            Api::OpenAiChatCompletions | Api::OpenAiResponses | Api::OpenAiEmbeddings => "openai",
            Api::AnthropicMessages => "anthropic",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub api: Api,
    /// wardex operation vocabulary: `"chat"` | `"embeddings"`.
    pub operation: &'static str,
    /// semconv `openai.api.type`; None for non-OpenAI APIs.
    pub api_type: Option<&'static str>,
    /// This API has a WebSocket transport on the same path; the seam may
    /// treat a WS upgrade here as an LLM connection — only after
    /// corroboration, see `ws_upgrade`.
    pub ws_transport: bool,
}

/// Matched against the path's LAST segments — query, fragment and trailing
/// `/` stripped, then split on `/`. Gateway prefixes (`/openai/v1/...`,
/// `/proxy/...`) pass through; sub-resources (`/v1/responses/{id}`,
/// `/v1/messages/count_tokens`, `/v1/messages/batches`) do not match.
const ENDPOINTS: &[(&[&str], Endpoint)] = &[
    (
        &["chat", "completions"],
        Endpoint {
            api: Api::OpenAiChatCompletions,
            operation: "chat",
            api_type: Some("chat_completions"),
            ws_transport: false,
        },
    ),
    (
        &["responses"],
        Endpoint {
            api: Api::OpenAiResponses,
            operation: "chat",
            api_type: Some("responses"),
            ws_transport: true,
        },
    ),
    // A compaction is billed: model + input in, usage + output items out.
    // Same parser and label as /responses; a `CompactedResponse` carries no
    // `status`, so finish_reasons/response_status stay empty, and its echoed
    // user messages keep their role (parts.rs `OutMsg.role`) rather than
    // becoming the model's words. Listed AFTER the /responses row so
    // `lookup(Api::OpenAiResponses)` keeps answering with /responses.
    (
        &["responses", "compact"],
        Endpoint {
            api: Api::OpenAiResponses,
            operation: "chat",
            api_type: Some("responses"),
            ws_transport: false,
        },
    ),
    (
        &["embeddings"],
        Endpoint {
            api: Api::OpenAiEmbeddings,
            operation: "embeddings",
            api_type: None,
            ws_transport: false,
        },
    ),
    (
        &["messages"],
        Endpoint {
            api: Api::AnthropicMessages,
            operation: "chat",
            api_type: None,
            ws_transport: false,
        },
    ),
];

/// The path's segments: query, fragment and trailing `/` stripped, split on
/// `/`, empties dropped.
fn segments_of(path: &str) -> Vec<&str> {
    let path = path.split(['?', '#']).next().unwrap_or("");
    let path = path.trim_end_matches('/');
    path.split('/').filter(|s| !s.is_empty()).collect()
}

/// The path-based decision. Conservative by design: a non-matching path is
/// None, never a guess (body-shape capture on arbitrary paths is the
/// compat-gateway follow-up, not this table's job).
pub fn from_path(path: &str) -> Option<Endpoint> {
    let segments = segments_of(path);
    for (suffix, endpoint) in ENDPOINTS {
        if segments.len() >= suffix.len() && segments[segments.len() - suffix.len()..] == **suffix {
            return Some(*endpoint);
        }
    }
    None
}

/// One segment of a non-LLM path pattern: a literal, or any single segment
/// (a resource id).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Seg {
    Lit(&'static str),
    Any,
}

use Seg::{Any, Lit};

/// What the seam does with a provider path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Treatment {
    /// An LLM endpoint (`from_path` is Some): parsed for gen_ai semantics.
    LlmCall,
    /// Provider-owned agent state (the Conversations API family). Plain
    /// HTTP: no model, no usage, no assistant output anywhere in the
    /// family, so there is nothing an LLM span would carry. Follows the
    /// non-LLM capture rule, and a refusal is counted.
    ProviderState,
    /// A telemetry upload (`/v1/traces/ingest`): the caller's own run
    /// record on its way to a tracing backend. Telemetry ABOUT the agent,
    /// not agent activity — never captured, in any mode, above any
    /// allowlist; counted.
    Excluded,
}

/// Anchored on `v1`: `/conversations` and `/traces` are shared API
/// vocabulary, unlike the LLM rows' last segments. A gateway prefix still
/// passes (suffix match); a root-level `/conversations` (Intercom, Front)
/// does not.
const NON_LLM_PATHS: &[(&[Seg], Treatment)] = &[
    (&[Lit("v1"), Lit("conversations")], Treatment::ProviderState),
    (
        &[Lit("v1"), Lit("conversations"), Any],
        Treatment::ProviderState,
    ),
    (
        &[Lit("v1"), Lit("conversations"), Any, Lit("items")],
        Treatment::ProviderState,
    ),
    (
        &[Lit("v1"), Lit("conversations"), Any, Lit("items"), Any],
        Treatment::ProviderState,
    ),
    (
        &[Lit("v1"), Lit("traces"), Lit("ingest")],
        Treatment::Excluded,
    ),
];

/// The path classification: `LlmCall` when `from_path` matches, else the
/// first `NON_LLM_PATHS` suffix match, else None (an unrecognised path —
/// ordinary HTTP, no claim).
pub fn treatment(path: &str) -> Option<Treatment> {
    if from_path(path).is_some() {
        return Some(Treatment::LlmCall);
    }
    let segments = segments_of(path);
    for (pattern, treatment) in NON_LLM_PATHS {
        if segments.len() < pattern.len() {
            continue;
        }
        let tail = &segments[segments.len() - pattern.len()..];
        let matches = pattern.iter().zip(tail).all(|(seg, got)| match seg {
            Lit(want) => want == got,
            Any => true,
        });
        if matches {
            return Some(*treatment);
        }
    }
    None
}

/// The WebSocket-transport question's answer: the upgrade path is a
/// WebSocket-capable LLM row, and the host either names that row's provider
/// or does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WsUpgrade {
    /// The host names the row's provider: the connection carries LLM calls.
    KnownProvider,
    /// The host names no provider: the seam must corroborate from the first
    /// client message before claiming anything.
    UnknownHost,
}

/// None unless the upgrade path matches a `ws_transport` row (the one
/// provider API with a WebSocket transport today is Responses).
pub fn ws_upgrade(host: &str, path: &str) -> Option<WsUpgrade> {
    let e = from_path(path)?;
    if !e.ws_transport {
        return None;
    }
    if super::provider_from_host(host) == Some(e.api.provider()) {
        Some(WsUpgrade::KnownProvider)
    } else {
        Some(WsUpgrade::UnknownHost)
    }
}

/// The body-shape decision, for PROVIDER inference when the host says
/// nothing. Order is load-bearing: a Responses body carries
/// `usage.input_tokens` too, so the Responses check must run before the
/// Anthropic one.
pub fn from_body_shape(body: &serde_json::Value) -> Option<Endpoint> {
    let obj = body.as_object()?;
    let is = |name: &str| obj.get("object").and_then(|o| o.as_str()) == Some(name);
    // 1. Responses: object == "response" (or a compaction's
    //    "response.compaction"), or an output array beside a status.
    if is("response")
        || is("response.compaction")
        || (obj.get("output").is_some_and(serde_json::Value::is_array)
            && obj.contains_key("status"))
    {
        return lookup(Api::OpenAiResponses);
    }
    // 2. Chat Completions: choices, or the usage spelling only it uses.
    if obj.contains_key("choices")
        || obj
            .get("usage")
            .and_then(|u| u.get("prompt_tokens"))
            .is_some()
    {
        return lookup(Api::OpenAiChatCompletions);
    }
    // 3. Embeddings: a list whose first datum is an embedding vector.
    if is("list")
        && obj
            .get("data")
            .and_then(|d| d.as_array())
            .and_then(|a| a.first())
            .is_some_and(|first| first.get("embedding").is_some())
    {
        return lookup(Api::OpenAiEmbeddings);
    }
    // 4. Anthropic Messages — last, because `usage.input_tokens` alone is
    //    shared with Responses (step 1 already claimed those).
    if obj.contains_key("stop_reason")
        || (is_type(obj, "message") && obj.contains_key("content"))
        || obj
            .get("usage")
            .and_then(|u| u.get("input_tokens"))
            .is_some()
    {
        return lookup(Api::AnthropicMessages);
    }
    None
}

fn is_type(obj: &serde_json::Map<String, serde_json::Value>, name: &str) -> bool {
    obj.get("type").and_then(|t| t.as_str()) == Some(name)
}

/// The SSE-grammar decision: `response.*` event types are the Responses API,
/// `message_start` is Anthropic, a `choices` payload is Chat Completions.
pub fn from_sse_shape(events: &[crate::sse::SseEvent]) -> Option<Endpoint> {
    for ev in events {
        if let Some(name) = ev.event.as_deref() {
            if name.starts_with("response.") {
                return lookup(Api::OpenAiResponses);
            }
            if name == "message_start" {
                return lookup(Api::AnthropicMessages);
            }
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&ev.data) {
            if let Some(ty) = v.get("type").and_then(|x| x.as_str()) {
                if ty.starts_with("response.") {
                    return lookup(Api::OpenAiResponses);
                }
                if ty == "message_start" {
                    return lookup(Api::AnthropicMessages);
                }
            }
            if v.get("choices").is_some() {
                return lookup(Api::OpenAiChatCompletions);
            }
        }
    }
    None
}

fn lookup(api: Api) -> Option<Endpoint> {
    ENDPOINTS.iter().map(|(_, e)| *e).find(|e| e.api == api)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-R12 — the last-segment table: gateway prefixes pass, sub-resources
    /// and sibling operations do not.
    #[test]
    fn path_table_accepts_endpoints_and_rejects_subresources() {
        let positive = [
            ("/v1/responses", Api::OpenAiResponses),
            ("/openai/v1/responses?api-version=x", Api::OpenAiResponses),
            ("/v1/responses/", Api::OpenAiResponses),
            ("/v1/responses/compact", Api::OpenAiResponses),
            ("/v1/chat/completions", Api::OpenAiChatCompletions),
            (
                "/openai/deployments/d/chat/completions",
                Api::OpenAiChatCompletions,
            ),
            ("/v1/embeddings", Api::OpenAiEmbeddings),
            ("/v1/messages", Api::AnthropicMessages),
            ("/proxy/v1/messages", Api::AnthropicMessages),
        ];
        for (path, want) in positive {
            assert_eq!(from_path(path).map(|e| e.api), Some(want), "{path}");
        }
        let negative = [
            "/v1/responses/resp_1",
            "/v1/responses/resp_1/input_items",
            "/v1/messages/count_tokens",
            "/v1/messages/batches",
            "/v1/models",
            "/",
            "",
        ];
        for path in negative {
            assert_eq!(from_path(path), None, "{path}");
        }
    }

    /// The wider classification: LLM rows, the Conversations family
    /// (provider state, anchored on `v1`), the telemetry upload (excluded),
    /// and everything else None.
    #[test]
    fn non_llm_provider_paths_get_a_treatment() {
        use Treatment::*;
        let rows = [
            ("/v1/responses", Some(LlmCall)),
            ("/v1/responses/compact", Some(LlmCall)),
            ("/v1/conversations", Some(ProviderState)),
            ("/v1/conversations/conv_1", Some(ProviderState)),
            (
                "/v1/conversations/conv_1/items?limit=20",
                Some(ProviderState),
            ),
            ("/v1/conversations/conv_1/items/msg_1", Some(ProviderState)),
            ("/openai/v1/conversations/conv_1/items", Some(ProviderState)),
            ("/v1/traces/ingest", Some(Excluded)),
            ("/proxy/v1/traces/ingest", Some(Excluded)),
            ("/conversations/conv_1", None),
            ("/api/traces/ingest", None),
            ("/v1/responses/resp_1", None),
            ("/v1/models", None),
            ("", None),
        ];
        for (path, want) in rows {
            assert_eq!(treatment(path), want, "{path}");
        }
    }

    /// Only a `ws_transport` row opens the WebSocket question, and the host
    /// decides between a known provider and one still to corroborate.
    #[test]
    fn ws_upgrade_needs_a_ws_capable_row() {
        assert_eq!(
            ws_upgrade("api.openai.com", "/v1/responses"),
            Some(WsUpgrade::KnownProvider)
        );
        assert_eq!(
            ws_upgrade("127.0.0.1", "/v1/responses"),
            Some(WsUpgrade::UnknownHost)
        );
        for (host, path) in [
            ("api.openai.com", "/v1/chat/completions"),
            ("chat.example.com", "/messages"),
            ("api.openai.com", "/v1/responses/compact"),
            ("x", "/realtime"),
        ] {
            assert_eq!(ws_upgrade(host, path), None, "{host}{path}");
        }
    }

    #[test]
    fn endpoint_rows_carry_operation_and_api_type() {
        let e = from_path("/v1/responses").unwrap();
        assert_eq!(e.operation, "chat");
        assert_eq!(e.api_type, Some("responses"));
        let e = from_path("/v1/chat/completions").unwrap();
        assert_eq!(e.operation, "chat");
        assert_eq!(e.api_type, Some("chat_completions"));
        let e = from_path("/v1/messages").unwrap();
        assert_eq!(e.api_type, None);
        let e = from_path("/v1/embeddings").unwrap();
        assert_eq!(e.operation, "embeddings");
    }

    /// 4.2.2 order — a Responses body carries `usage.input_tokens` too and
    /// must not read as Anthropic.
    #[test]
    fn responses_shaped_body_is_not_anthropic() {
        let body = serde_json::json!({
            "id": "resp_1", "object": "response", "status": "completed",
            "output": [], "usage": {"input_tokens": 5, "output_tokens": 1}
        });
        assert_eq!(
            from_body_shape(&body).map(|e| e.api),
            Some(Api::OpenAiResponses)
        );
        // and without the `object` spelling, output[] + status still decides:
        let body = serde_json::json!({
            "output": [], "status": "completed",
            "usage": {"input_tokens": 5}
        });
        assert_eq!(
            from_body_shape(&body).map(|e| e.api),
            Some(Api::OpenAiResponses)
        );
        // a compaction has no status; its `object` spelling decides:
        let body = serde_json::json!({
            "object": "response.compaction", "output": [], "usage": {}
        });
        assert_eq!(
            from_body_shape(&body).map(|e| e.api),
            Some(Api::OpenAiResponses)
        );
    }

    #[test]
    fn body_shapes_map_to_their_apis() {
        let chat = serde_json::json!({"choices": [], "usage": {"prompt_tokens": 1}});
        assert_eq!(
            from_body_shape(&chat).map(|e| e.api),
            Some(Api::OpenAiChatCompletions)
        );
        let anth = serde_json::json!({"stop_reason": "end_turn", "content": []});
        assert_eq!(
            from_body_shape(&anth).map(|e| e.api),
            Some(Api::AnthropicMessages)
        );
        let emb = serde_json::json!({"object": "list", "data": [{"embedding": [0.1]}]});
        assert_eq!(
            from_body_shape(&emb).map(|e| e.api),
            Some(Api::OpenAiEmbeddings)
        );
        assert_eq!(from_body_shape(&serde_json::json!({"hello": 1})), None);
        assert_eq!(from_body_shape(&serde_json::json!("scalar")), None);
    }

    #[test]
    fn sse_shapes_map_to_their_apis() {
        use crate::sse;
        let responses =
            sse::parse(b"event: response.created\ndata: {\"type\":\"response.created\"}\n\n");
        assert_eq!(
            from_sse_shape(&responses).map(|e| e.api),
            Some(Api::OpenAiResponses)
        );
        let anth = sse::parse(b"data: {\"type\":\"message_start\",\"message\":{}}\n\n");
        assert_eq!(
            from_sse_shape(&anth).map(|e| e.api),
            Some(Api::AnthropicMessages)
        );
        let chat = sse::parse(b"data: {\"choices\":[{\"delta\":{}}]}\n\n");
        assert_eq!(
            from_sse_shape(&chat).map(|e| e.api),
            Some(Api::OpenAiChatCompletions)
        );
        let unknown = sse::parse(b"data: {\"kind\":\"other\"}\n\n");
        assert_eq!(from_sse_shape(&unknown), None);
    }
}
