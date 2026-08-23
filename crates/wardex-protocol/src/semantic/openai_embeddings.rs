//! OpenAI Embeddings: response fill.

use super::openai_chat::OAUsage;
use super::LlmSemantics;
use crate::usage::{InputConvention, TokenUsage};
use serde::Deserialize;

// --- OpenAI embeddings ---

#[derive(Deserialize)]
struct OpenAIEmbeddingsResponse {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<OAUsage>,
}

pub(super) fn fill_openai_embeddings(out: &mut LlmSemantics, resp: &[u8]) {
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
