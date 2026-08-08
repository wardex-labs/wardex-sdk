//! One export → the requests a receiver will actually accept.
//!
//! [`super::map`] decides what a span MEANS and [`super::encode_traces`]
//! serializes it; this decides how many POSTs the result has to become. It is
//! the third thing an export needs and the only one whose answer depends on a
//! number the backend owns rather than on anything in the span.
//!
//! Why it cannot be left to the receiver: an OTLP request is rejected WHOLE. A
//! collector over its body limit answers 413 (or its gRPC front end answers
//! RESOURCE_EXHAUSTED) and nothing in the request is stored, so one oversized
//! batch costs every span in it — including the small ones that were never the
//! problem. Splitting turns a total loss into several accepted requests, and
//! the only case that still loses anything is a single span that cannot be made
//! to fit, which is counted rather than swallowed.

use wardex_limits::Limits;

use super::map;
use super::otlp_pb;
use crate::{gzip, CodecError};

/// One export, as the request bodies that will be POSTed.
#[derive(Debug, Default)]
pub struct Requests {
    /// Bodies in span order, each already compressed if compression was asked
    /// for. Empty when the export carried no spans — a caller can therefore
    /// tell "nothing to send" from "one empty request" without decoding.
    pub bodies: Vec<Vec<u8>>,
    /// Spans that could not be made to fit even alone, after their payload was
    /// dropped. Returned rather than logged here: the core has no channel to a
    /// user, and a loss that only the core knows about is the silent kind.
    pub dropped_spans: usize,
}

/// Encode an export, splitting it into as many bodies as
/// `limits.max_otlp_request_bytes` requires.
///
/// BOTH sizes are measured against the cap — the body as it goes on the wire
/// and the message it decompresses to — because a receiver checks both. gRPC
/// rejects a frame over `max_receive_message_length` and then rejects what it
/// decompresses to for the same reason; the collector's HTTP receiver applies
/// its body limit to the decompressed stream, which is how it refuses a
/// decompression bomb. Measuring only the compressed body would leave the
/// failure this module exists to prevent fully reachable in the case
/// compression makes likely: OTLP payload attributes are base64 text and gzip
/// several-fold, so a 40 MiB export that compresses under 4 MiB would go out
/// whole and be rejected whole.
///
/// Never an ESTIMATE, in either direction: both numbers come from actually
/// encoding and actually compressing, so nothing here can disagree with what
/// the receiver measures.
///
/// The cost of that honesty is an encode per attempt, so the shape of the
/// algorithm matters. Two things keep it near one pass: the whole export is
/// tried first, so a batch that fits — nearly every batch — costs one encode
/// and no split at all; and a node already over the cap UNCOMPRESSED is never
/// compressed, because gzip cannot rescue a message whose decompressed size is
/// the thing being rejected. Compression therefore runs only on bodies that
/// will be sent, once each, whatever depth the split reached.
pub fn encode_requests(
    req: otlp_pb::trace_service::ExportTraceServiceRequest,
    limits: Limits,
    compress: bool,
) -> Result<Requests, CodecError> {
    let mut out = Requests::default();
    for rs in req.resource_spans {
        let resource = rs.resource;
        let resource_schema_url = rs.schema_url;
        for ss in rs.scope_spans {
            let template = Template {
                resource: resource.clone(),
                resource_schema_url: resource_schema_url.clone(),
                scope: ss.scope,
                scope_schema_url: ss.schema_url,
            };
            emit(
                &template,
                ss.spans,
                limits.max_otlp_request_bytes,
                compress,
                &mut out,
            )?;
        }
    }
    Ok(out)
}

/// Everything about a request except which spans are in it.
///
/// Rebuilt around every chunk rather than sliced out of one request, because
/// the resource and the scope are not divisible: each body has to name the
/// service and the instrumentation library for itself or the spans in it arrive
/// unattributed.
struct Template {
    resource: Option<otlp_pb::resource::Resource>,
    resource_schema_url: String,
    scope: Option<otlp_pb::common::InstrumentationScope>,
    scope_schema_url: String,
}

impl Template {
    fn request(
        &self,
        spans: Vec<otlp_pb::trace::Span>,
    ) -> otlp_pb::trace_service::ExportTraceServiceRequest {
        otlp_pb::trace_service::ExportTraceServiceRequest {
            resource_spans: vec![otlp_pb::trace::ResourceSpans {
                resource: self.resource.clone(),
                schema_url: self.resource_schema_url.clone(),
                scope_spans: vec![otlp_pb::trace::ScopeSpans {
                    scope: self.scope.clone(),
                    schema_url: self.scope_schema_url.clone(),
                    spans,
                }],
            }],
        }
    }

    /// The spans back out of a request that did not fit, so the smaller
    /// requests it becomes cost no copy of the payloads.
    fn reclaim(
        req: otlp_pb::trace_service::ExportTraceServiceRequest,
    ) -> Vec<otlp_pb::trace::Span> {
        req.resource_spans
            .into_iter()
            .flat_map(|rs| rs.scope_spans)
            .flat_map(|ss| ss.spans)
            .collect()
    }
}

fn emit(
    template: &Template,
    spans: Vec<otlp_pb::trace::Span>,
    cap: usize,
    compress: bool,
    out: &mut Requests,
) -> Result<(), CodecError> {
    if spans.is_empty() {
        return Ok(());
    }
    let req = template.request(spans);
    let over = match fit(&req, cap, compress)? {
        Fit::Fits(body) => {
            out.bodies.push(body);
            return Ok(());
        }
        Fit::Over(size) => size,
    };
    let mut spans = Template::reclaim(req);
    if spans.len() == 1 {
        let mut span = spans.pop().expect("length was just checked");
        // The payload is what makes a span large, and a span is worth more than
        // its payload: dropping it keeps the node in the trace, its parent edge,
        // its timing and its semantics, and says on the span itself what was
        // removed.
        map::drop_payload_attributes(&mut span);
        match fit(&template.request(vec![span]), cap, compress)? {
            Fit::Fits(body) => out.bodies.push(body),
            // A span whose METADATA alone will not fit — a pathological name or
            // an event list past the cap. Counted, so the caller can say one
            // span was lost rather than let a batch quietly arrive short.
            Fit::Over(_) => out.dropped_spans += 1,
        }
        return Ok(());
    }
    // Fan out in ONE step from the measurement already paid for, rather than
    // bisecting. Halving looks cheaper and is not: every level re-encodes the
    // same N bytes, so depth d costs d+1 full passes over the whole batch, and
    // the depth is reachable — a full 64 MiB buffer against the 4 MiB default
    // is five of them. Dividing the measured size by the cap lands on the right
    // number of chunks immediately, and a chunk that is still over (because one
    // span in it dominates) simply recurses on its own measurement.
    //
    // `cap.max(1)` guards the division alone: a cap of 0 is not reachable from
    // the SDK's own validation, and the core must not panic for a limit a host
    // constructed by hand.
    let parts = over.div_ceil(cap.max(1)).clamp(2, spans.len());
    let chunk = spans.len().div_ceil(parts);
    let mut rest = spans;
    while !rest.is_empty() {
        let tail = rest.split_off(chunk.min(rest.len()));
        emit(template, rest, cap, compress, out)?;
        rest = tail;
    }
    Ok(())
}

/// One request measured against the cap: the body to send, or how large the
/// attempt turned out to be.
enum Fit {
    Fits(Vec<u8>),
    Over(usize),
}

/// Encode, compress if asked, and answer whether a receiver would take it.
///
/// The uncompressed size is checked FIRST and the compression pass is skipped
/// when it already fails, which is both halves of the point. A receiver
/// enforces its ceiling on the decompressed message as well as on the body, so
/// a batch that gzips under the cap from far above it is rejected anyway;
/// and a node the split is about to discard must not have been compressed,
/// because that pass over the whole batch is the expensive one.
///
/// Both buffers are owned here, so a node that does not fit frees them on the
/// way out — the recursion holds the span tree, not a discarded encoding of it
/// at every level.
fn fit(
    req: &otlp_pb::trace_service::ExportTraceServiceRequest,
    cap: usize,
    compress: bool,
) -> Result<Fit, CodecError> {
    let bytes = super::encode_traces(req)?;
    if bytes.len() > cap {
        return Ok(Fit::Over(bytes.len()));
    }
    let body = if compress { gzip(&bytes)? } else { bytes };
    if body.len() > cap {
        return Ok(Fit::Over(body.len()));
    }
    Ok(Fit::Fits(body))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gunzip;
    use otlp_pb::trace::{ResourceSpans, ScopeSpans, Span};
    use otlp_pb::trace_service::ExportTraceServiceRequest;

    fn payload_span(id: u8, payload: usize) -> Span {
        Span {
            trace_id: vec![1u8; 16],
            span_id: vec![id; 8],
            name: format!("chat model-{id}"),
            attributes: vec![otlp_pb::common::KeyValue {
                key: "wardex.input_data".into(),
                value: Some(otlp_pb::common::AnyValue {
                    value: Some(otlp_pb::common::any_value::Value::StringValue(
                        // Random-ish so gzip cannot collapse the batch to
                        // nothing and make the split untestable.
                        (0..payload)
                            .map(|i| char::from(b'a' + ((i * 7 + id as usize) % 26) as u8))
                            .collect(),
                    )),
                }),
            }],
            ..Default::default()
        }
    }

    fn request(spans: Vec<Span>) -> ExportTraceServiceRequest {
        ExportTraceServiceRequest {
            resource_spans: vec![ResourceSpans {
                resource: Some(otlp_pb::resource::Resource {
                    attributes: vec![otlp_pb::common::KeyValue {
                        key: "service.name".into(),
                        value: Some(otlp_pb::common::AnyValue {
                            value: Some(otlp_pb::common::any_value::Value::StringValue(
                                "wardex.python".into(),
                            )),
                        }),
                    }],
                    ..Default::default()
                }),
                scope_spans: vec![ScopeSpans {
                    scope: Some(otlp_pb::common::InstrumentationScope {
                        name: "wardex.python".into(),
                        ..Default::default()
                    }),
                    spans,
                    ..Default::default()
                }],
                ..Default::default()
            }],
        }
    }

    fn limits_with_request_cap(cap: usize) -> Limits {
        Limits {
            max_otlp_request_bytes: cap,
            ..Limits::default()
        }
    }

    fn decode(body: &[u8], compressed: bool) -> ExportTraceServiceRequest {
        let raw = if compressed {
            gunzip(body).unwrap()
        } else {
            body.to_vec()
        };
        super::super::decode_traces(&raw).unwrap()
    }

    fn span_ids(out: &Requests, compressed: bool) -> Vec<u8> {
        out.bodies
            .iter()
            .flat_map(|b| {
                decode(b, compressed)
                    .resource_spans
                    .into_iter()
                    .flat_map(|rs| rs.scope_spans)
                    .flat_map(|ss| ss.spans)
                    .map(|sp| sp.span_id[0])
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn a_batch_that_fits_is_one_request() {
        let out =
            encode_requests(request(vec![payload_span(1, 10)]), Limits::default(), false).unwrap();
        assert_eq!(out.bodies.len(), 1);
        assert_eq!(out.dropped_spans, 0);
    }

    #[test]
    fn an_export_with_no_spans_produces_no_request() {
        // "Nothing to send" must be distinguishable from "one empty request"
        // without decoding: a POST of an empty batch is a round trip nobody
        // asked for and a 200 that means nothing.
        let out = encode_requests(
            ExportTraceServiceRequest::default(),
            Limits::default(),
            true,
        )
        .unwrap();
        assert!(out.bodies.is_empty());
        assert_eq!(out.dropped_spans, 0);
    }

    #[test]
    fn a_batch_over_the_cap_splits_and_every_span_survives_in_order() {
        let spans: Vec<Span> = (1..=8).map(|i| payload_span(i, 4000)).collect();
        let out = encode_requests(request(spans), limits_with_request_cap(8_000), false).unwrap();
        assert!(out.bodies.len() > 1, "expected a split");
        assert_eq!(out.dropped_spans, 0);
        assert_eq!(span_ids(&out, false), (1..=8).collect::<Vec<u8>>());
        for body in &out.bodies {
            assert!(body.len() <= 8_000, "chunk of {} bytes", body.len());
        }
    }

    #[test]
    fn a_batch_many_times_the_cap_does_not_split_into_many_times_too_many() {
        // The fan-out is derived from the measured size, so it has to land near
        // the number of requests the batch actually needs. A splitter that
        // over-shoots turns one export into a burst of POSTs, and one that
        // under-shoots pays another full encode of the whole batch per level —
        // which is the cost that made bisection the wrong shape here.
        let spans: Vec<Span> = (1..=32).map(|i| payload_span(i, 4000)).collect();
        let total = super::super::encode_traces(&request(spans.clone()))
            .unwrap()
            .len();
        let cap = total / 8;
        let out = encode_requests(request(spans), limits_with_request_cap(cap), false).unwrap();
        assert!(
            (8..=20).contains(&out.bodies.len()),
            "8 requests' worth of spans became {}",
            out.bodies.len()
        );
        assert_eq!(span_ids(&out, false), (1..=32).collect::<Vec<u8>>());
    }

    #[test]
    fn every_chunk_carries_the_resource_and_the_scope() {
        // A split body that lost them would arrive unattributed: the spans in
        // it would name no service and no instrumentation library.
        let spans: Vec<Span> = (1..=8).map(|i| payload_span(i, 4000)).collect();
        let out = encode_requests(request(spans), limits_with_request_cap(8_000), false).unwrap();
        for body in &out.bodies {
            let rs = &decode(body, false).resource_spans[0];
            assert_eq!(rs.resource.as_ref().unwrap().attributes.len(), 1);
            assert_eq!(
                rs.scope_spans[0].scope.as_ref().unwrap().name,
                "wardex.python"
            );
        }
    }

    #[test]
    fn the_cap_holds_on_the_body_and_on_what_it_decompresses_to() {
        // A receiver checks both, so both are measured. Enforcing only the
        // compressed size is the dangerous half: OTLP payload attributes are
        // base64 text and gzip several-fold, so a batch far over a collector's
        // decompressed ceiling fits under it compressed, goes out as one
        // request, and is rejected WHOLE — every span in it lost, which is the
        // outcome splitting exists to avoid.
        let spans: Vec<Span> = (1..=8).map(|i| payload_span(i, 4000)).collect();
        let out = encode_requests(request(spans), limits_with_request_cap(8_000), true).unwrap();
        assert_eq!(out.dropped_spans, 0);
        assert_eq!(span_ids(&out, true), (1..=8).collect::<Vec<u8>>());
        for body in &out.bodies {
            assert!(body.len() <= 8_000, "body of {} bytes", body.len());
            let raw = gunzip(body).unwrap();
            assert!(raw.len() <= 8_000, "decompresses to {} bytes", raw.len());
        }
    }

    #[test]
    fn a_batch_that_only_fits_once_gzipped_is_still_split() {
        // The regression the test above describes, as a single assertion: this
        // payload is repetitive enough that gzip crushes the whole batch to
        // well under the cap, so a splitter measuring only the compressed body
        // would emit exactly one request.
        let spans: Vec<Span> = (1..=8).map(|i| payload_span(i, 4000)).collect();
        let whole = super::super::encode_traces(&request(spans.clone())).unwrap();
        assert!(
            crate::gzip(&whole).unwrap().len() <= 8_000,
            "not the case under test"
        );
        let out = encode_requests(request(spans), limits_with_request_cap(8_000), true).unwrap();
        assert!(
            out.bodies.len() > 1,
            "an oversized batch went out as one request"
        );
    }

    #[test]
    fn a_batch_exactly_at_the_cap_is_not_split() {
        // The boundary is inclusive on both sides of one comparison, so an
        // off-by-one here is a request split for no reason or a request one
        // byte over the limit.
        let req = request(vec![payload_span(1, 500)]);
        let exact = super::super::encode_traces(&req).unwrap().len();
        let at = encode_requests(
            request(vec![payload_span(1, 500)]),
            limits_with_request_cap(exact),
            false,
        )
        .unwrap();
        assert_eq!(at.bodies.len(), 1);
        let under = encode_requests(
            request(vec![payload_span(1, 500)]),
            limits_with_request_cap(exact - 1),
            false,
        )
        .unwrap();
        // One span, so there is nothing to split: it takes the last-resort path
        // instead and ships without its payload.
        assert_eq!(under.bodies.len(), 1);
        assert_eq!(under.dropped_spans, 0);
        let sp = &decode(&under.bodies[0], false).resource_spans[0].scope_spans[0].spans[0];
        assert!(!sp.attributes.iter().any(|kv| kv.key == "wardex.input_data"));
    }

    #[test]
    fn a_single_giant_span_loses_its_payload_and_keeps_its_place() {
        let out = encode_requests(
            request(vec![payload_span(1, 200_000), payload_span(2, 10)]),
            limits_with_request_cap(4_000),
            false,
        )
        .unwrap();
        assert_eq!(out.dropped_spans, 0);
        assert_eq!(span_ids(&out, false), vec![1, 2], "the small span went too");
        let giant = out
            .bodies
            .iter()
            .flat_map(|b| {
                decode(b, false).resource_spans[0].scope_spans[0]
                    .spans
                    .clone()
            })
            .find(|sp| sp.span_id[0] == 1)
            .expect("the oversized span");
        assert!(!giant
            .attributes
            .iter()
            .any(|kv| kv.key == "wardex.input_data"));
        assert_eq!(giant.name, "chat model-1", "the span itself survived");
        let marked = giant
            .attributes
            .iter()
            .any(|kv| kv.key == "wardex.limitations");
        assert!(marked, "the loss reached the wire unmarked");
    }

    #[test]
    fn a_span_that_cannot_fit_even_bare_is_dropped_alone() {
        // The cap is below what the span's NAME costs, so nothing survives it.
        // Every other span in the batch still has to reach the wire — the whole
        // point of splitting is that one bad span is not a lost batch.
        let mut giant = payload_span(1, 10);
        giant.name = "x".repeat(4000);
        let out = encode_requests(
            request(vec![giant, payload_span(2, 10)]),
            limits_with_request_cap(300),
            false,
        )
        .unwrap();
        assert_eq!(out.dropped_spans, 1);
        assert_eq!(span_ids(&out, false), vec![2]);
    }
}
