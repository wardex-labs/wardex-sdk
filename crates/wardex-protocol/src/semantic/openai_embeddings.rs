//! OpenAI Embeddings: request/response fill.

use super::usage::{flatten_usage, UsageBounds, UsageView};
use super::LlmSemantics;
use crate::usage::{InputConvention, TokenUsage};
use serde::Deserialize;

// Normalization table: LlmSemantics usage field <- dotted path in the raw
// usage Value (see `openai_chat.rs` — same contract, one entry).
const P_INPUT: &str = "prompt_tokens";
// Read by the fixture test (`every_normalized_usage_path_is_extracted_
// from_its_fixture`), not by the runtime path — the runtime reads the
// P_* consts the table is built from, which makes the two one string.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) const NORMALIZED_USAGE_PATHS: &[(&str, &str)] = &[("input_tokens", P_INPUT)];

/// `encoding_format`: a single string or an array of strings.
#[derive(Deserialize)]
#[serde(untagged)]
enum StringOrStrings {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct OpenAIEmbeddingsRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    encoding_format: Option<StringOrStrings>,
    #[serde(default)]
    dimensions: Option<i64>,
}

#[derive(Deserialize)]
struct OpenAIEmbeddingsResponse {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<serde_json::Value>,
}

pub(super) fn fill_openai_embeddings(
    out: &mut LlmSemantics,
    req: &[u8],
    resp: &[u8],
    bounds: UsageBounds,
) {
    // `output_type` stays None: an embeddings response is vectors, and the
    // old unconditional `"text"` stamp was a false claim on every one.
    if let Ok(r) = serde_json::from_slice::<OpenAIEmbeddingsResponse>(resp) {
        out.response_model = r.model;
        if let Some(u) = r.usage {
            let view = UsageView(&u);
            out.usage = TokenUsage::new(
                InputConvention::Inclusive,
                view.i64_at(P_INPUT),
                None,
                None,
                None,
                None,
            );
            let flat = flatten_usage(&u, bounds);
            out.usage_leaves = flat.leaves;
            out.usage_dropped_count = flat.dropped;
        }
    }
    if let Ok(q) = serde_json::from_slice::<OpenAIEmbeddingsRequest>(req) {
        out.request_model = q.model;
        out.encoding_formats = q.encoding_format.map(|f| match f {
            StringOrStrings::One(s) => vec![s],
            StringOrStrings::Many(v) => v,
        });
        out.embedding_dimensions = q.dimensions;
    }
}
