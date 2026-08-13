//! Machine conversion between a wardex value string and its proto enum number.
//!
//! Every closed vocabulary crosses the PyO3 boundary as its *wardex value*
//! (`"execute_tool"`, `"body_cap_exceeded"`, `"contextvar"`) rather than its
//! proto value name, because that string is what a Python enum member holds and
//! what already travels in `Span.extra`. Something has to turn it into a number.
//!
//! Doing that with a hand-written `match` table is how a vocabulary drifts. The
//! table is a SECOND declaration of a list the `.proto` already declares, and
//! nothing makes the compiler compare the two — a member added to one and
//! forgotten in the other compiles, ships, and silently flattens to
//! `UNSPECIFIED` on the wire. At 40 members (`Limitation`) that is not a
//! hypothetical.
//!
//! prost generates `from_str_name` / `as_str_name` from the schema itself, so
//! deriving the proto value name mechanically — `PREFIX` + `_` + the uppercased
//! wardex value — leaves exactly one declaration of the vocabulary: the
//! `.proto` file. That is what design §6.6 means by "proto is the single source
//! of truth"; a second SDK in another language generates from the same file and
//! gets the same table for free.
//!
//! The naming convention this relies on is not incidental — proto3 file-level
//! enum values share the package namespace, so the prefix is mandatory anyway.
//! `tests::every_python_value_shape_round_trips` pins the round trip.

use crate::proto::wardex::v1 as pb;

/// wardex value → proto value name. `"in_process"` → `"TOOL_EXECUTION_TYPE_IN_PROCESS"`.
fn proto_name(prefix: &str, value: &str) -> String {
    let mut out = String::with_capacity(prefix.len() + 1 + value.len());
    out.push_str(prefix);
    out.push('_');
    out.push_str(&value.to_uppercase());
    out
}

/// proto value name → wardex value. The inverse of [`proto_name`].
///
/// A name that does not carry the expected prefix is lowercased whole rather
/// than silently emptied: the caller asked what this number means, and a
/// truthful odd answer beats a tidy wrong one.
fn wardex_value(prefix: &str, name: &str) -> String {
    name.strip_prefix(prefix)
        .and_then(|rest| rest.strip_prefix('_'))
        .unwrap_or(name)
        .to_lowercase()
}

/// Defines the two directions for one closed vocabulary.
///
/// The encode direction returns `Option` on purpose. Collapsing "I do not know
/// this value" into `UNSPECIFIED` is the silent coercion this module exists to
/// prevent (design I4); who fills that hole, and with what, is the caller's
/// decision because it differs per vocabulary — `Limitation` has a dedicated
/// `VOCABULARY_UNMAPPED` value to carry the fact, most others have nothing
/// better than `UNSPECIFIED` and should say so at the call site.
macro_rules! closed_vocabulary {
    ($enum_ty:ty, $prefix:literal, $to_proto:ident, $to_wardex:ident, $doc:literal) => {
        #[doc = $doc]
        #[doc = ""]
        #[doc = "wardex value → proto number. `None` means the schema has no such value."]
        pub fn $to_proto(value: &str) -> Option<i32> {
            <$enum_ty>::from_str_name(&proto_name($prefix, value)).map(|v| v as i32)
        }

        #[doc = $doc]
        #[doc = ""]
        #[doc = "proto number → wardex value. Three outcomes stay distinguishable:"]
        #[doc = "`0` → `\"\"` (unset), a known number → its wardex value, and an"]
        #[doc = "unknown number → a string that declares its own ignorance rather"]
        #[doc = "than passing for \"unset\"."]
        pub fn $to_wardex(v: i32) -> String {
            if v == 0 {
                return String::new();
            }
            match <$enum_ty>::try_from(v) {
                Ok(e) => wardex_value($prefix, e.as_str_name()),
                Err(_) => format!("{}_unrecognized_{}", $prefix.to_lowercase(), v),
            }
        }
    };
}

closed_vocabulary!(
    pb::SpanKind,
    "SPAN_KIND",
    span_kind_to_proto,
    span_kind_name,
    "`Span.kind`."
);
closed_vocabulary!(
    pb::StatusCode,
    "STATUS_CODE",
    status_code_to_proto,
    status_code_name,
    "`Status.code`."
);
closed_vocabulary!(
    pb::CaptureSource,
    "CAPTURE_SOURCE",
    capture_source_to_proto,
    capture_source_name,
    "`Span.capture_sources`."
);
closed_vocabulary!(
    pb::Protocol,
    "PROTOCOL",
    protocol_to_proto,
    protocol_name,
    "`TransportAttributes.protocol`."
);
closed_vocabulary!(
    pb::Direction,
    "DIRECTION",
    direction_to_proto,
    direction_name,
    "`TransportAttributes.direction`."
);
closed_vocabulary!(
    pb::Modality,
    "MODALITY",
    modality_to_proto,
    modality_name,
    "`TransportAttributes.request_modality` / `response_modality`."
);
closed_vocabulary!(
    pb::SnapshotType,
    "SNAPSHOT_TYPE",
    snapshot_type_to_proto,
    snapshot_type_name,
    "`StateSnapshot.snapshot_type`."
);
closed_vocabulary!(
    pb::OperationName,
    "OPERATION_NAME",
    operation_name_to_proto,
    operation_name_name,
    "The twelve span intents. Rides as `extra[\"gen_ai.operation.name\"]`."
);
closed_vocabulary!(
    pb::ToolExecutionType,
    "TOOL_EXECUTION_TYPE",
    tool_execution_type_to_proto,
    tool_execution_type_name,
    "How a tool body ran. Rides as `extra[\"wardex.tool.execution_type\"]`."
);
closed_vocabulary!(
    pb::LinkReason,
    "LINK_REASON",
    link_reason_to_proto,
    link_reason_name,
    "`SpanLink.reason` — why one span points at another (causality, not containment)."
);
closed_vocabulary!(
    pb::Limitation,
    "LIMITATION",
    limitation_to_proto,
    limitation_name,
    "`CaptureIntegrity.limitation_codes` — what wardex failed to capture."
);
closed_vocabulary!(
    pb::ParentSource,
    "PARENT_SOURCE",
    parent_source_to_proto,
    parent_source_name,
    "`CorrelationInfo.parent_source` — how the parent edge was derived."
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values_round_trip() {
        for v in [
            "chat",
            "execute_tool",
            "invoke_workflow",
            "generate_content",
            "execute_step",
        ] {
            let n = operation_name_to_proto(v).expect(v);
            assert_eq!(operation_name_name(n), v);
        }
        for v in ["in_process", "ipc", "unknown"] {
            let n = tool_execution_type_to_proto(v).expect(v);
            assert_eq!(tool_execution_type_name(n), v);
        }
        for v in ["triggered_by", "handoff_from", "cache_source"] {
            let n = link_reason_to_proto(v).expect(v);
            assert_eq!(link_reason_name(n), v);
        }
        for v in ["mcp_stdio", "websocket", "sse"] {
            let n = protocol_to_proto(v).expect(v);
            assert_eq!(protocol_name(n), v);
        }
    }

    /// The whole 40-member vocabulary, round-tripped by number. A member added
    /// to the schema with a name that breaks the convention fails here rather
    /// than flattening to UNSPECIFIED on a user's wire.
    #[test]
    fn every_limitation_round_trips() {
        let mut seen = 0;
        for n in 1..=40 {
            let value = limitation_name(n);
            assert!(!value.is_empty(), "no name for limitation {n}");
            assert!(
                !value.contains("unrecognized"),
                "limitation {n} is not declared: {value}"
            );
            assert_eq!(
                limitation_to_proto(&value),
                Some(n),
                "round trip for {value}"
            );
            seen += 1;
        }
        assert_eq!(seen, 40);
    }

    #[test]
    fn every_parent_source_round_trips() {
        for n in 1..=7 {
            let value = parent_source_name(n);
            assert!(!value.is_empty(), "no name for parent source {n}");
            assert_eq!(parent_source_to_proto(&value), Some(n));
        }
        // The Python enum has exactly seven members; an eighth appearing on the
        // wire means the two drifted.
        assert!(parent_source_name(8).contains("unrecognized"));
    }

    /// `VOCABULARY_UNMAPPED` is a META value, not vocabulary. It must be
    /// reachable for decode (a newer SDK can send it) and must NOT sit inside
    /// the 1..=40 band a consumer iterates as "the vocabulary".
    #[test]
    fn the_meta_value_is_outside_the_vocabulary_band() {
        assert_eq!(pb::Limitation::VocabularyUnmapped as i32, 9001);
        assert_eq!(limitation_name(9001), "vocabulary_unmapped");
    }

    /// Zero is "unset", not a value. Collapsing it into the vocabulary would
    /// make "no limitations" indistinguishable from "one unnamed limitation".
    #[test]
    fn zero_is_empty_not_unspecified() {
        assert_eq!(limitation_name(0), "");
        assert_eq!(link_reason_name(0), "");
        assert_eq!(parent_source_name(0), "");
    }

    /// The failure this module exists to prevent, asserted directly: an
    /// unknown value is reported, never coerced.
    #[test]
    fn unknown_values_are_reported_not_coerced() {
        assert_eq!(limitation_to_proto("no_such_marker"), None);
        assert_eq!(parent_source_to_proto("adapter_hook"), None);
        assert_eq!(operation_name_to_proto("banana"), None);
        assert_eq!(link_reason_name(777), "link_reason_unrecognized_777");
        assert_eq!(limitation_name(888), "limitation_unrecognized_888");
    }
}
