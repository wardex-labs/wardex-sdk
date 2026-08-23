//! Endpoint recognition: which provider API a transaction spoke, from the
//! path's LAST segments, from the response body's shape, or from the SSE
//! event grammar. Replaces two substring checks whose false positives were
//! real defects: `path.contains("/messages")` classified
//! `/v1/messages/count_tokens` and `/v1/messages/batches` as chat calls (a
//! false `semantic_parse_failed` on every one), and the missing `/v1/responses`
//! row sent Responses SSE through the Chat reassembler, which fabricated an
//! empty chat body.

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
        },
    ),
    (
        &["responses"],
        Endpoint {
            api: Api::OpenAiResponses,
            operation: "chat",
            api_type: Some("responses"),
        },
    ),
    (
        &["embeddings"],
        Endpoint {
            api: Api::OpenAiEmbeddings,
            operation: "embeddings",
            api_type: None,
        },
    ),
    (
        &["messages"],
        Endpoint {
            api: Api::AnthropicMessages,
            operation: "chat",
            api_type: None,
        },
    ),
];

/// The path-based decision. Conservative by design: a non-matching path is
/// None, never a guess (body-shape capture on arbitrary paths is the
/// compat-gateway follow-up, not this table's job).
pub fn from_path(path: &str) -> Option<Endpoint> {
    let path = path.split(['?', '#']).next().unwrap_or("");
    let path = path.trim_end_matches('/');
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    for (suffix, endpoint) in ENDPOINTS {
        if segments.len() >= suffix.len() && segments[segments.len() - suffix.len()..] == **suffix {
            return Some(*endpoint);
        }
    }
    None
}

/// The body-shape decision, for PROVIDER inference when the host says
/// nothing. Order is load-bearing: a Responses body carries
/// `usage.input_tokens` too, so the Responses check must run before the
/// Anthropic one.
pub fn from_body_shape(body: &serde_json::Value) -> Option<Endpoint> {
    let obj = body.as_object()?;
    let is = |name: &str| obj.get("object").and_then(|o| o.as_str()) == Some(name);
    // 1. Responses: object == "response", or an output array beside a status.
    if is("response")
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
            "/v1/responses/compact",
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
