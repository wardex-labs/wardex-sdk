//! gRPC message framing parser (library/transport-layer agnostic).
//! Wire format: `[1B compressed flag][4B big-endian length][payload]` repeated.
//! Payload bytes are not copied (the raw bytes are already preserved in the caller's body).

/// Metadata for a single gRPC message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrpcMessage {
    pub compressed: bool,
    pub length: u32,
}

/// Framing result for one direction's (request or response) body.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GrpcFrames {
    pub messages: Vec<GrpcMessage>,
    pub truncated: bool,
}

/// Decomposes the body into gRPC length-prefixed frames.
/// If the tail doesn't have enough bytes left for the 5-byte prefix or the declared length, marks truncated=true and stops.
pub fn parse_grpc_frames(body: &[u8]) -> GrpcFrames {
    let mut messages = Vec::new();
    let n = body.len();
    let mut off = 0usize;
    loop {
        if off == n {
            return GrpcFrames {
                messages,
                truncated: false,
            };
        }
        if n - off < 5 {
            return GrpcFrames {
                messages,
                truncated: true,
            };
        }
        let compressed = body[off] != 0;
        let length =
            u32::from_be_bytes([body[off + 1], body[off + 2], body[off + 3], body[off + 4]]);
        let end = match off
            .checked_add(5)
            .and_then(|x| x.checked_add(length as usize))
        {
            Some(e) => e,
            None => {
                return GrpcFrames {
                    messages,
                    truncated: true,
                }
            }
        };
        if end > n {
            return GrpcFrames {
                messages,
                truncated: true,
            };
        }
        messages.push(GrpcMessage { compressed, length });
        off = end;
    }
}

/// Maps a standard gRPC status code to its standard name. Out-of-range codes map to "UNKNOWN_CODE".
pub fn grpc_status_name(code: i32) -> &'static str {
    match code {
        0 => "OK",
        1 => "CANCELLED",
        2 => "UNKNOWN",
        3 => "INVALID_ARGUMENT",
        4 => "DEADLINE_EXCEEDED",
        5 => "NOT_FOUND",
        6 => "ALREADY_EXISTS",
        7 => "PERMISSION_DENIED",
        8 => "RESOURCE_EXHAUSTED",
        9 => "FAILED_PRECONDITION",
        10 => "ABORTED",
        11 => "OUT_OF_RANGE",
        12 => "UNIMPLEMENTED",
        13 => "INTERNAL",
        14 => "UNAVAILABLE",
        15 => "DATA_LOSS",
        16 => "UNAUTHENTICATED",
        _ => "UNKNOWN_CODE",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes a single gRPC message for testing.
    fn msg(compressed: u8, payload: &[u8]) -> Vec<u8> {
        let mut v = vec![compressed];
        v.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        v.extend_from_slice(payload);
        v
    }

    #[test]
    fn parses_single_message() {
        let f = parse_grpc_frames(&msg(0, b"hello"));
        assert_eq!(
            f.messages,
            vec![GrpcMessage {
                compressed: false,
                length: 5
            }]
        );
        assert!(!f.truncated);
    }

    #[test]
    fn parses_multiple_messages_with_compression_flag() {
        let mut body = msg(0, b"aa");
        body.extend(msg(1, b"bbbb"));
        let f = parse_grpc_frames(&body);
        assert_eq!(
            f.messages,
            vec![
                GrpcMessage {
                    compressed: false,
                    length: 2
                },
                GrpcMessage {
                    compressed: true,
                    length: 4
                },
            ]
        );
        assert!(!f.truncated);
    }

    #[test]
    fn marks_truncated_on_short_prefix() {
        let f = parse_grpc_frames(&[0u8, 0, 0]); // fewer than 5 bytes
        assert!(f.messages.is_empty());
        assert!(f.truncated);
    }

    #[test]
    fn marks_truncated_on_short_payload() {
        let mut body = msg(0, b"hello");
        body.truncate(7); // prefix (5) + only 2 bytes of payload
        let f = parse_grpc_frames(&body);
        assert!(f.messages.is_empty());
        assert!(f.truncated);
    }

    #[test]
    fn empty_body_is_not_truncated() {
        let f = parse_grpc_frames(&[]);
        assert!(f.messages.is_empty());
        assert!(!f.truncated);
    }

    #[test]
    fn status_name_maps_known_and_unknown() {
        assert_eq!(grpc_status_name(0), "OK");
        assert_eq!(grpc_status_name(5), "NOT_FOUND");
        assert_eq!(grpc_status_name(16), "UNAUTHENTICATED");
        assert_eq!(grpc_status_name(99), "UNKNOWN_CODE");
    }
}
