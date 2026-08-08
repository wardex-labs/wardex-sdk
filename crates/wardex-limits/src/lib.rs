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
    /// Maximum concurrently tracked *root* logical units.
    ///
    /// Deliberately not a flat cap over units of every kind. Children are
    /// bounded per-unit by `max_entries_per_unit`, and a flat ceiling at N
    /// would make "evict the oldest" pick a long-lived session root nearly
    /// every time, so one chatty session would evict *other* sessions' roots
    /// almost immediately. Scoping it to roots is what makes N read as a
    /// concurrency ceiling: N concurrent runs.
    ///
    /// It is not, however, the ONLY thing that evicts a root. A registry also
    /// holds a derived ceiling over live units of every kind — the product of
    /// this bound and `max_entries_per_unit` — and reaching that evicts roots
    /// too. A run nested deeply enough inside itself can therefore still cost
    /// another run its root. The scoping raises the price of that enormously;
    /// it does not make it unreachable.
    ///
    /// Read by the logical-unit registry. Crossing it CLOSES the oldest root
    /// so its span is emitted carrying `unit_evicted`: a ceiling that dropped
    /// state silently would be a worse failure than an unenforced one.
    pub max_units: usize,
    /// Maximum entries in each per-unit table (child units, lookup aliases,
    /// de-duplication keys, open span drafts). Generalizes
    /// `max_session_entries` to units of every kind; same order, same
    /// semantics. Host SDKs also reuse it for adapter-side bookkeeping of the
    /// same order — in the Python SDK, the table of wrapped in-process MCP
    /// servers — so lowering it shrinks more than the four tables named above.
    ///
    /// Only two of those four tables hold entries that OWN a span, and only
    /// those two evictions reach the wire. Crossing the bound on child units or
    /// on open span drafts force-closes the oldest entry and emits its span
    /// with `child_span_unclosed`, for the same reason as `max_units`. Crossing
    /// it on lookup aliases or de-duplication keys emits nothing — there is no
    /// span to mark — and is recorded only in the host SDK's internal counters
    /// (`assembly._units.alias_table_full`, `assembly._units.claim_table_full`,
    /// `adapters.anthropic.server_table_full` in the Python SDK).
    ///
    /// What surfaces from a silent eviction is a consequence, and the alias one
    /// is worth stating precisely because it can look like an IMPROVEMENT. An
    /// alias edge is `unit_alias` at `0.9`; losing the alias does not simply
    /// lower that. It sends the resolver back down its ladder, and if the task
    /// carries an ambient span the ladder stops at `contextvar` — confidence
    /// `1.0`, no marker — which for a sub-agent means its subtree silently
    /// flattens into the enclosing session. Only when there is no ambient span
    /// at all does it reach `unit_inferred_sole` (`0.5`) or `parent_unresolved`
    /// (`0.0`). An evicted de-duplication key can let one logical call be
    /// observed twice, and an evicted server handle degrades a hook's tool-name
    /// lookup to the builtin key space, which can do the same.
    pub max_entries_per_unit: usize,
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
    /// live export path calls today: the OTLP exporter compresses with gzip,
    /// because that is the encoding OTLP receivers accept. So the wiring is
    /// real but the knob is currently unobservable to a user, and it is
    /// documented as inert alongside `replay_buffer_size` until an envelope
    /// transport ships.
    pub zstd_level: i32,
    /// Maximum size of ONE attribute value on the OTLP surface, measured as it
    /// will appear on the wire — i.e. AFTER the base64 rewrite a binary payload
    /// goes through, not before it. A value over the bound is truncated and the
    /// span says so with an `otlp_attribute_truncated` marker.
    ///
    /// It is not a second `max_body_bytes` and does not overlap it. That one
    /// caps what a parser KEEPS, in raw bytes, before anything is encoded; this
    /// one caps what one attribute may COST on a wire whose receiver has its
    /// own opinion about request size. The two differ by up to 4/3 for binary
    /// payloads, which is exactly the gap that made a batch inside every raw
    /// cap encode to something no collector would accept.
    ///
    /// Sized against `max_otlp_request_bytes`: two payload attributes at this
    /// bound, plus semantics, still fit one request, so a single span alone
    /// crossing the request cap is the pathological case rather than the
    /// ordinary one. Raising this without raising that moves work onto the
    /// split path instead of onto the wire.
    pub max_otlp_attribute_bytes: usize,
    /// Maximum size of ONE OTLP/HTTP request, measured on BOTH numbers a
    /// receiver checks: the encoded, compressed body that goes on the wire and
    /// the message it decompresses to. A batch over either is split across
    /// several POSTs rather than sent whole and rejected.
    ///
    /// Both, because a receiver enforces both. gRPC refuses a frame over
    /// `max_receive_message_length` and then refuses what it decompresses to
    /// under the same number; the collector's HTTP receiver applies its body
    /// limit to the decompressed stream, which is how it refuses a
    /// decompression bomb. Checking only the compressed size would leave the
    /// all-or-nothing rejection reachable exactly where compression helps most
    /// — OTLP payload attributes are base64 text and gzip several-fold.
    ///
    /// The default is gRPC's own `max_receive_message_length`, which the
    /// OTLP/gRPC receiver inherits and collector HTTP deployments commonly
    /// mirror; a backend that accepts more will simply never see a split.
    /// Rejection here is all-or-nothing at the request level, which is what
    /// makes an unbounded body worse than a truncated one: the receiver drops
    /// the whole batch, so every span in it disappears together.
    pub max_otlp_request_bytes: usize,
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
            max_units: 512,
            max_entries_per_unit: 256,
            mcp_sniff_bytes: 8192,
            max_buffer_spans: 2048,
            max_buffer_bytes: 64 * 1024 * 1024,
            replay_buffer_size: 100,
            zstd_level: 3,
            max_otlp_attribute_bytes: 1024 * 1024,
            max_otlp_request_bytes: 4 * 1024 * 1024,
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
        assert_eq!(l.max_units, 512);
        assert_eq!(l.max_entries_per_unit, 256);
        assert_eq!(l.mcp_sniff_bytes, 8192);
        assert_eq!(l.max_buffer_spans, 2048);
        assert_eq!(l.max_buffer_bytes, 64 * 1024 * 1024);
        assert_eq!(l.replay_buffer_size, 100);
        assert_eq!(l.zstd_level, 3);
        assert_eq!(l.max_otlp_attribute_bytes, 1024 * 1024);
        assert_eq!(l.max_otlp_request_bytes, 4 * 1024 * 1024);
    }

    /// The relationship the two OTLP bounds are sized against, asserted rather
    /// than described: a span carrying both payload attributes at the attribute
    /// ceiling must still fit one request, so splitting is about BATCHES.
    #[test]
    fn two_capped_payloads_fit_one_request() {
        let l = Limits::default();
        assert!(2 * l.max_otlp_attribute_bytes < l.max_otlp_request_bytes);
    }

    #[test]
    fn is_copy() {
        let a = Limits::default();
        let b = a; // moves only if !Copy
        assert_eq!(a, b);
    }
}
