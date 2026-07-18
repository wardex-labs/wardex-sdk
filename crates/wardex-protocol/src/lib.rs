//! Protocol parsers for Wardex SDK.
//!
//! Phase 0 scaffold — modules are empty placeholders.

pub mod claude_stream_json;
pub mod grpc;
pub mod http1;
pub mod http2;
pub mod json_rpc;
pub mod semantic;
pub mod sse;
pub mod websocket;

#[cfg(test)]
mod tests {
    #[test]
    fn it_compiles() {}
}
