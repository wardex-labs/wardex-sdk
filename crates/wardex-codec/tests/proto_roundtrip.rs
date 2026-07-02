//! Phase 0 smoke: verify build.rs generates the proto and prost round-trips it.

use prost::Message;
use wardex_codec::proto::wardex::v1::{Envelope, TransportAttributes};

#[test]
fn envelope_default_roundtrip() {
    let original = Envelope::default();
    let bytes = original.encode_to_vec();
    let decoded = Envelope::decode(&bytes[..]).expect("decode failed");
    assert_eq!(original, decoded);
}

#[test]
fn transport_attributes_connection_reused_roundtrip() {
    let original = TransportAttributes {
        connection_reused: true,
        ..Default::default()
    };
    let bytes = original.encode_to_vec();
    let decoded = TransportAttributes::decode(&bytes[..]).expect("decode failed");
    assert_eq!(original, decoded);
    assert!(decoded.connection_reused);
}
