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
/// The measurement is the FINAL body — encoded and, when `compress` is set,
/// compressed — because that is the number the receiver measures. Estimating
/// from the uncompressed size would split batches that would have fitted, and
/// estimating from a compression ratio would occasionally not split one that
/// does not.
///
/// The cost of that honesty is one encode per attempt, so the shape of the
/// algorithm matters: the whole export is tried first and the common case is
/// therefore ONE encode with no splitting at all. Only a batch that does not
/// fit pays for halving, and the halves it pays for shrink geometrically.
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

    /// The spans back out of a request that did not fit, so a retry at half the
    /// size costs no copy of the payloads.
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
    let body = finish(&req, compress)?;
    if body.len() <= cap {
        out.bodies.push(body);
        return Ok(());
    }
    let mut spans = Template::reclaim(req);
    if spans.len() == 1 {
        let mut span = spans.pop().expect("length was just checked");
        // The payload is what makes a span large, and a span is worth more than
        // its payload: dropping it keeps the node in the trace, its parent edge,
        // its timing and its semantics, and says on the span itself what was
        // removed.
        map::drop_payload_attributes(&mut span);
        let body = finish(&template.request(vec![span]), compress)?;
        if body.len() <= cap {
            out.bodies.push(body);
        } else {
            // A span whose METADATA alone will not fit — a pathological name or
            // an event list past the cap. Counted, so the caller can say one
            // span was lost rather than let a batch quietly arrive short.
            out.dropped_spans += 1;
        }
        return Ok(());
    }
    // Halving rather than greedy packing by measured span size. Both produce
    // requests under the cap; this one costs a logarithmic number of encodes in
    // the depth it reaches and needs no per-span size model to keep in step
    // with what the encoder actually writes.
    let right = spans.split_off(spans.len() / 2);
    emit(template, spans, cap, compress, out)?;
    emit(template, right, cap, compress, out)
}

fn finish(
    req: &otlp_pb::trace_service::ExportTraceServiceRequest,
    compress: bool,
) -> Result<Vec<u8>, CodecError> {
    let bytes = super::encode_traces(req)?;
    if compress {
        gzip(&bytes)
    } else {
        Ok(bytes)
    }
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
    fn the_cap_is_measured_on_the_compressed_body() {
        // Compression is what decides whether a batch fits, so measuring
        // before it would split batches a receiver would have accepted.
        let spans: Vec<Span> = (1..=8).map(|i| payload_span(i, 4000)).collect();
        let uncompressed = encode_requests(
            request(spans.clone()),
            limits_with_request_cap(8_000),
            false,
        )
        .unwrap();
        let compressed =
            encode_requests(request(spans), limits_with_request_cap(8_000), true).unwrap();
        assert!(
            compressed.bodies.len() < uncompressed.bodies.len(),
            "compressed into {} requests, uncompressed into {}",
            compressed.bodies.len(),
            uncompressed.bodies.len()
        );
        assert_eq!(span_ids(&compressed, true), (1..=8).collect::<Vec<u8>>());
    }

    #[test]
    fn a_batch_exactly_at_the_cap_is_not_split() {
        // The boundary is inclusive on both sides of one comparison, so an
        // off-by-one here is a request split for no reason or a request one
        // byte over the limit.
        let req = request(vec![payload_span(1, 500)]);
        let exact = super::finish(&req, false).unwrap().len();
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
