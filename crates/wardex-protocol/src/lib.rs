//! Protocol parsers for Wardex SDK.
//!
//! One module per wire format the SDK can read off a byte seam: framing parsers
//! (`http1`, `http2`, `websocket`, `sse`, `grpc`, `json_rpc`,
//! `claude_stream_json`) and, above them, `semantic`, which turns a parsed
//! request/response pair into provider-neutral message parts.

pub mod claude_stream_json;
pub mod grpc;
pub mod http1;
pub mod http2;
pub mod json_rpc;
pub mod semantic;
pub mod sse;
pub mod usage;
pub mod websocket;

pub use usage::{InputConvention, TokenUsage};

#[cfg(test)]
mod tests {
    #[test]
    fn it_compiles() {}
}
