//! Wardex codec — Protobuf encoding + compression.
//!
//! `proto` re-exports the generated wire types; `encode_envelope`/`decode_envelope`
//! are the round trip over them, and `otlp` maps an envelope onto OTLP.
//!
//! TWO compressors, and which one applies is decided by the wire, not by
//! preference. The wardex envelope travels zstd because both ends are ours.
//! OTLP travels gzip because the OTLP/HTTP specification names gzip as the
//! encoding a receiver must accept, and a collector handed
//! `Content-Encoding: zstd` answers 415 — a whole batch lost to a header.

pub mod proto {
    pub mod wardex {
        pub mod v1 {
            #[allow(clippy::large_enum_variant)]
            pub mod inner {
                include!(concat!(env!("OUT_DIR"), "/wardex.v1.rs"));
            }
            pub use inner::*;
        }
    }
}

pub mod otlp;
pub mod vocab;

use prost::Message;
use wardex_limits::Limits;

use crate::proto::wardex::v1 as pb;

#[derive(Debug)]
pub enum CodecError {
    Compress(std::io::Error),
    Decompress(std::io::Error),
    Decode(prost::DecodeError),
}

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodecError::Compress(e) => write!(f, "zstd compress failed: {e}"),
            CodecError::Decompress(e) => write!(f, "zstd decompress failed: {e}"),
            CodecError::Decode(e) => write!(f, "protobuf decode failed: {e}"),
        }
    }
}

impl std::error::Error for CodecError {}

/// Envelope → protobuf serialize → zstd compress.
pub fn encode_envelope(env: &pb::Envelope, limits: Limits) -> Result<Vec<u8>, CodecError> {
    let proto_bytes = env.encode_to_vec();
    zstd::stream::encode_all(&proto_bytes[..], limits.zstd_level).map_err(CodecError::Compress)
}

/// zstd decompress → protobuf deserialize → Envelope.
pub fn decode_envelope(data: &[u8]) -> Result<pb::Envelope, CodecError> {
    let proto_bytes = zstd::stream::decode_all(data).map_err(CodecError::Decompress)?;
    pb::Envelope::decode(&proto_bytes[..]).map_err(CodecError::Decode)
}

/// The deflate level every OTLP request is compressed at.
///
/// Deliberately not a `Limits` field, unlike `zstd_level`. That knob exists
/// because nothing else in the SDK can trade the envelope's CPU against its
/// bytes; here the trade is already owned by `max_otlp_request_bytes`, which
/// decides what "small enough" means and splits the batch when compression
/// cannot get there. Level 6 is flate2's default and the level the OTLP
/// ecosystem's own exporters use for this wire, so a request wardex compresses
/// costs a receiver what every other exporter's does.
const GZIP_LEVEL: u32 = 6;

/// Raw bytes → gzip stream (RFC 1952), the encoding OTLP/HTTP receivers accept.
///
/// Separate from the zstd pair above rather than a mode of it: they serve
/// different wires and only one of them is negotiable. This one's output has to
/// match a `Content-Encoding: gzip` header exactly, so it emits a gzip frame —
/// not a bare deflate stream, which is a different encoding token and is
/// rejected under this one.
pub fn gzip(data: &[u8]) -> Result<Vec<u8>, CodecError> {
    use flate2::write::GzEncoder;
    use std::io::Write as _;

    let mut encoder = GzEncoder::new(Vec::new(), flate2::Compression::new(GZIP_LEVEL));
    encoder.write_all(data).map_err(CodecError::Compress)?;
    encoder.finish().map_err(CodecError::Compress)
}

/// gzip stream → raw bytes. The receiver's half of [`gzip`], for tests and for
/// any host that wants to read back what it sent.
pub fn gunzip(data: &[u8]) -> Result<Vec<u8>, CodecError> {
    use flate2::read::GzDecoder;
    use std::io::Read as _;

    let mut out = Vec::new();
    GzDecoder::new(data)
        .read_to_end(&mut out)
        .map_err(CodecError::Decompress)?;
    Ok(out)
}

// Every enum mapping now lives in `vocab`, derived from the schema rather than
// transcribed from it. The whole table used to sit here as a hand-written
// `match` per enum; `Limitation` arriving with 37 members made that untenable,
// and what replaced it removes the class of bug rather than one instance of it.
//
// These mappings live in this crate rather than in the PyO3 binding for one
// practical reason: this crate owns the generated enums and `cargo test` can
// build a harness for it, while `bindings/python` is `crate-type = ["cdylib"]`
// with pyo3's `extension-module` and gets no test harness at all. A mapping
// whose unknown-value branch is the entire point needs a test that reaches it.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::wardex::v1 as pb;
    // Moved to `vocab` (schema-derived, no longer a hand table). The two tests
    // below stay here unchanged: they pin the BEHAVIOUR the move had to
    // preserve, so keeping them where they were is what makes the move
    // reviewable.
    use crate::vocab::link_reason_name;

    fn sample() -> pb::Envelope {
        pb::Envelope {
            header: Some(pb::EnvelopeHeader {
                event_id: "evt-1".into(),
                api_key: "k".into(),
                sdk: Some(pb::SdkInfo {
                    name: "wardex.python".into(),
                    version: "0.1.0".into(),
                    ..Default::default()
                }),
                sent_at_unix_nano: 42,
                ..Default::default()
            }),
            items: vec![pb::EnvelopeItem {
                header: Some(pb::EnvelopeItemHeader {
                    r#type: "span".into(),
                    length: 0,
                }),
                payload: Some(pb::envelope_item::Payload::Span(pb::Span {
                    trace_id: vec![1u8; 16],
                    span_id: vec![2u8; 8],
                    name: "GET /".into(),
                    kind: pb::SpanKind::Client as i32,
                    input_data: vec![0u8; 5000], // repetitive bytes that compress well
                    capture_sources: vec![pb::CaptureSource::Socket as i32],
                    ..Default::default()
                })),
            }],
        }
    }

    #[test]
    fn roundtrip_preserves_envelope() {
        let env = sample();
        let back = decode_envelope(&encode_envelope(&env, Limits::default()).unwrap()).unwrap();
        assert_eq!(env, back);
    }

    #[test]
    fn encode_is_deterministic() {
        let env = sample();
        assert_eq!(
            encode_envelope(&env, Limits::default()).unwrap(),
            encode_envelope(&env, Limits::default()).unwrap()
        );
    }

    #[test]
    fn zstd_compresses_repetitive_bodies() {
        let bytes = encode_envelope(&sample(), Limits::default()).unwrap();
        assert!(
            bytes.len() < 2000,
            "expected compression, got {} bytes",
            bytes.len()
        );
    }

    #[test]
    fn decode_rejects_corrupt_input() {
        assert!(decode_envelope(b"not a zstd frame").is_err());
    }

    #[test]
    fn gzip_round_trips() {
        let payload = b"POST /v1/traces".repeat(500);
        let compressed = gzip(&payload).unwrap();
        assert!(compressed.len() < payload.len());
        assert_eq!(gunzip(&compressed).unwrap(), payload);
    }

    #[test]
    fn gzip_emits_a_gzip_frame_not_a_bare_deflate_stream() {
        // The magic a receiver checks before it will honour
        // `Content-Encoding: gzip`. A bare deflate stream decompresses fine
        // with the wrong decoder and is rejected by the right one, so the
        // frame header is the assertion, not the round trip above.
        let compressed = gzip(b"x").unwrap();
        assert_eq!(&compressed[..2], &[0x1f, 0x8b]);
    }

    #[test]
    fn gzip_round_trips_the_empty_input() {
        // An empty OTLP request is legal (zero resource_spans), and an encoder
        // that only flushes on write would emit nothing at all for it.
        assert_eq!(gunzip(&gzip(b"").unwrap()).unwrap(), b"");
    }

    #[test]
    fn gunzip_rejects_input_that_is_not_gzip() {
        assert!(gunzip(b"not a gzip frame").is_err());
    }

    #[test]
    fn link_reason_unset_and_unknown_are_not_the_same_string() {
        // The whole point of the mapping. A reader must be able to tell "no
        // reason was set" from "a reason this build does not know" — the second
        // is what arrives from a newer SDK, and reading it as the first turns a
        // handoff sibling back into a nested child (design §6.3).
        assert_eq!(link_reason_name(0), "");
        assert_eq!(link_reason_name(99), "link_reason_unrecognized_99");
        assert_ne!(link_reason_name(99), link_reason_name(0));
    }

    #[test]
    fn every_declared_link_reason_maps_to_its_wardex_string() {
        assert_eq!(
            link_reason_name(pb::LinkReason::TriggeredBy as i32),
            "triggered_by"
        );
        assert_eq!(
            link_reason_name(pb::LinkReason::HandoffFrom as i32),
            "handoff_from"
        );
        assert_eq!(
            link_reason_name(pb::LinkReason::ResumedFrom as i32),
            "resumed_from"
        );
        assert_eq!(
            link_reason_name(pb::LinkReason::RetriedFrom as i32),
            "retried_from"
        );
        assert_eq!(
            link_reason_name(pb::LinkReason::CacheSource as i32),
            "cache_source"
        );
    }
}
