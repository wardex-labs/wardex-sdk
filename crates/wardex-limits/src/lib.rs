//! Resource limits for the Wardex core.
//!
//! This crate is the single source of truth for every tunable resource bound in
//! the SDK: the struct defines the schema, `Default` defines the values, and
//! every language SDK reads both from here rather than re-declaring them.
//!
//! Enforcement is distributed — some limits are applied by the Rust parsers,
//! others by the host SDK (span buffer, connection maps, adapter session maps).
//! Ownership of the *values* stays here regardless, so a limit can never drift
//! between two declaration sites.

/// Tunable resource bounds. Every field is a hard ceiling except `zstd_level`,
/// which is a compression tuning knob included under the same rule: it is a
/// hardcoded, workload-dependent value that callers must be able to override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    // --- Enforced by the Rust parsers ---
    /// Maximum header count in a single HTTP/1 message. HTTP/1 needs it
    /// because httparse fills a caller-allocated header array, so the array
    /// has to be sized up front. HTTP/2 has no equivalent bound and does not
    /// need one: HPACK is stateful, so every header block on the connection
    /// must be decoded in full to keep the dynamic table in sync — refusing
    /// one after a count check would desync the table and corrupt every later
    /// message rather than protect anything.
    pub max_headers: usize,
    /// Body cap for content types carrying extractable meaning (JSON, text,
    /// SSE, form-encoded, gRPC). Defaults above the Anthropic Messages API
    /// request ceiling so LLM traffic can never be truncated.
    pub max_body_bytes: usize,
    /// Body cap for every other content type. Opaque bytes yield no semantics,
    /// so a small sample is kept purely for diagnostics.
    pub max_opaque_body_bytes: usize,
    /// Maximum size of the incomplete-message buffer held by a stream parser.
    /// Exceeding it means the peer is not speaking the expected protocol.
    ///
    /// It bounds a different quantity than `max_body_bytes` and deliberately
    /// need not be larger than it. A parser measures this against the residue
    /// left *after* a parse pass — a header block with no terminator, a torn
    /// chunk-size line, an unterminated JSON-RPC line, or bytes that are not
    /// the expected protocol at all. Body bytes never appear in that residue:
    /// the parser moves them into the message as it consumes them, and how
    /// many are stored is `max_body_bytes`'s decision alone. A body of any
    /// declared size therefore passes through whatever this value is, so the
    /// two are not comparable quantities and no ordering between them is
    /// validated.
    pub max_stream_buffer_bytes: usize,
    /// Maximum size of a decompressed body during semantic extraction.
    pub max_decoded_bytes: usize,
    /// Maximum concurrently tracked HTTP/2 streams per connection.
    pub max_streams: usize,
    /// Maximum accepted WebSocket frame payload length.
    pub max_ws_frame_bytes: usize,
    /// Per-direction WebSocket content sample retained for a span.
    pub ws_sample_bytes: usize,

    // --- Enforced by the host SDK ---
    /// Maximum tracked connections, applied to both the byte-seam connection
    /// map and the shared connection-timing store.
    pub max_connections: usize,
    /// Maximum concurrently tracked adapter sessions.
    pub max_sessions: usize,
    /// Maximum entries in each per-session tracking map (open tools, streamed
    /// tool metadata, subagents).
    pub max_session_entries: usize,
    /// Bytes read before giving up on detecting JSON-RPC over a stdio stream.
    pub mcp_sniff_bytes: usize,
    /// Maximum spans buffered before the oldest are dropped.
    pub max_buffer_spans: usize,
    /// Maximum approximate bytes buffered across pending spans. The final
    /// backstop: resident memory stays bounded even if every parser cap fails.
    pub max_buffer_bytes: usize,
    /// Replay ring-buffer depth. Reserved: no SDK reads this value today, so
    /// setting it changes nothing. It stays in the schema because it is part
    /// of the published configuration surface, and callers are told it is
    /// inert wherever the knob is documented rather than left to discover it.
    pub replay_buffer_size: usize,

    // --- Enforced by the codec ---
    /// Zstd compression level. Read by `encode_envelope`, which nothing in the
    /// live export path calls today: the OTLP exporter uses
    /// `encode_otlp_traces`, which neither takes limits nor compresses. So the
    /// wiring is real but the knob is currently unobservable to a user, and it
    /// is documented as inert alongside `replay_buffer_size` until an envelope
    /// transport ships.
    pub zstd_level: i32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_headers: 96,
            max_body_bytes: 32 * 1024 * 1024,
            max_opaque_body_bytes: 256 * 1024,
            max_stream_buffer_bytes: 16 * 1024 * 1024,
            max_decoded_bytes: 8 * 1024 * 1024,
            max_streams: 1024,
            max_ws_frame_bytes: 1024 * 1024,
            ws_sample_bytes: 64 * 1024,
            max_connections: 4096,
            max_sessions: 512,
            max_session_entries: 256,
            mcp_sniff_bytes: 8192,
            max_buffer_spans: 2048,
            max_buffer_bytes: 64 * 1024 * 1024,
            replay_buffer_size: 100,
            zstd_level: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_catalog() {
        let l = Limits::default();
        assert_eq!(l.max_headers, 96);
        assert_eq!(l.max_body_bytes, 32 * 1024 * 1024);
        assert_eq!(l.max_opaque_body_bytes, 256 * 1024);
        assert_eq!(l.max_stream_buffer_bytes, 16 * 1024 * 1024);
        assert_eq!(l.max_decoded_bytes, 8 * 1024 * 1024);
        assert_eq!(l.max_streams, 1024);
        assert_eq!(l.max_ws_frame_bytes, 1024 * 1024);
        assert_eq!(l.ws_sample_bytes, 64 * 1024);
        assert_eq!(l.max_connections, 4096);
        assert_eq!(l.max_sessions, 512);
        assert_eq!(l.max_session_entries, 256);
        assert_eq!(l.mcp_sniff_bytes, 8192);
        assert_eq!(l.max_buffer_spans, 2048);
        assert_eq!(l.max_buffer_bytes, 64 * 1024 * 1024);
        assert_eq!(l.replay_buffer_size, 100);
        assert_eq!(l.zstd_level, 3);
    }

    #[test]
    fn is_copy() {
        let a = Limits::default();
        let b = a; // moves only if !Copy
        assert_eq!(a, b);
    }
}
