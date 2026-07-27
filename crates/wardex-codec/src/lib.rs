//! Wardex codec — Protobuf encoding + Zstd compression.
//!
//! Phase 0 scaffold: expose proto types only. Actual encode/compress functions land in Phase 7.

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::wardex::v1 as pb;

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
}
