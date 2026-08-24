//! Python-visible resource limits: the `Limits` pyclass and `limits_defaults()`.
//!
//! Kept in its own module (rather than `lib.rs`) so the `useless_conversion`
//! workaround below stays scoped to the one function that needs it instead of
//! silencing the lint for the whole crate — mirrors the `codec` module's use
//! of the same attribute for the same reason.

// In the trampoline code generated when the pyo3 #[pyfunction] macro wraps a function
// returning `PyResult<T>`, clippy mistakes the `?`'s `From<PyErr> for PyErr` (identity)
// conversion for a useless conversion
// (a pre-existing pyo3 0.22 issue; a function-level #[allow] can't cover macro-generated sibling items).
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use wardex_limits::Limits;

/// Python-visible resource limits. Unspecified fields keep the core default.
#[pyclass(name = "Limits")]
#[derive(Clone, Copy)]
pub(crate) struct PyLimits {
    pub(crate) inner: Limits,
}

#[pymethods]
impl PyLimits {
    #[new]
    #[pyo3(signature = (
        max_headers=None, max_body_bytes=None, max_opaque_body_bytes=None,
        max_stream_buffer_bytes=None, max_decoded_bytes=None, max_streams=None,
        max_ws_frame_bytes=None, ws_sample_bytes=None, max_connections=None,
        max_sessions=None, max_session_entries=None, max_units=None,
        max_entries_per_unit=None, mcp_sniff_bytes=None, max_extra_keys=None,
        max_parse_backlog=None, max_parse_backlog_bytes=None,
        max_buffer_spans=None, max_buffer_bytes=None, replay_buffer_size=None,
        max_otel_bridge_body_bytes=None, max_otel_bridge_spans_per_session=None,
        zstd_level=None, max_otlp_attribute_bytes=None, max_otlp_request_bytes=None,
        max_link_targets=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_headers: Option<usize>,
        max_body_bytes: Option<usize>,
        max_opaque_body_bytes: Option<usize>,
        max_stream_buffer_bytes: Option<usize>,
        max_decoded_bytes: Option<usize>,
        max_streams: Option<usize>,
        max_ws_frame_bytes: Option<usize>,
        ws_sample_bytes: Option<usize>,
        max_connections: Option<usize>,
        max_sessions: Option<usize>,
        max_session_entries: Option<usize>,
        max_units: Option<usize>,
        max_entries_per_unit: Option<usize>,
        mcp_sniff_bytes: Option<usize>,
        max_extra_keys: Option<usize>,
        max_parse_backlog: Option<usize>,
        max_parse_backlog_bytes: Option<usize>,
        max_buffer_spans: Option<usize>,
        max_buffer_bytes: Option<usize>,
        replay_buffer_size: Option<usize>,
        max_otel_bridge_body_bytes: Option<usize>,
        max_otel_bridge_spans_per_session: Option<usize>,
        zstd_level: Option<i32>,
        max_otlp_attribute_bytes: Option<usize>,
        max_otlp_request_bytes: Option<usize>,
        max_link_targets: Option<usize>,
    ) -> Self {
        let d = Limits::default();
        Self {
            inner: Limits {
                max_headers: max_headers.unwrap_or(d.max_headers),
                max_body_bytes: max_body_bytes.unwrap_or(d.max_body_bytes),
                max_opaque_body_bytes: max_opaque_body_bytes.unwrap_or(d.max_opaque_body_bytes),
                max_stream_buffer_bytes: max_stream_buffer_bytes
                    .unwrap_or(d.max_stream_buffer_bytes),
                max_decoded_bytes: max_decoded_bytes.unwrap_or(d.max_decoded_bytes),
                max_streams: max_streams.unwrap_or(d.max_streams),
                max_ws_frame_bytes: max_ws_frame_bytes.unwrap_or(d.max_ws_frame_bytes),
                ws_sample_bytes: ws_sample_bytes.unwrap_or(d.ws_sample_bytes),
                max_connections: max_connections.unwrap_or(d.max_connections),
                max_sessions: max_sessions.unwrap_or(d.max_sessions),
                max_session_entries: max_session_entries.unwrap_or(d.max_session_entries),
                max_units: max_units.unwrap_or(d.max_units),
                max_entries_per_unit: max_entries_per_unit.unwrap_or(d.max_entries_per_unit),
                mcp_sniff_bytes: mcp_sniff_bytes.unwrap_or(d.mcp_sniff_bytes),
                max_extra_keys: max_extra_keys.unwrap_or(d.max_extra_keys),
                max_parse_backlog: max_parse_backlog.unwrap_or(d.max_parse_backlog),
                max_parse_backlog_bytes: max_parse_backlog_bytes
                    .unwrap_or(d.max_parse_backlog_bytes),
                max_buffer_spans: max_buffer_spans.unwrap_or(d.max_buffer_spans),
                max_buffer_bytes: max_buffer_bytes.unwrap_or(d.max_buffer_bytes),
                replay_buffer_size: replay_buffer_size.unwrap_or(d.replay_buffer_size),
                max_otel_bridge_body_bytes: max_otel_bridge_body_bytes
                    .unwrap_or(d.max_otel_bridge_body_bytes),
                max_otel_bridge_spans_per_session: max_otel_bridge_spans_per_session
                    .unwrap_or(d.max_otel_bridge_spans_per_session),
                zstd_level: zstd_level.unwrap_or(d.zstd_level),
                max_otlp_attribute_bytes: max_otlp_attribute_bytes
                    .unwrap_or(d.max_otlp_attribute_bytes),
                max_otlp_request_bytes: max_otlp_request_bytes.unwrap_or(d.max_otlp_request_bytes),
                max_link_targets: max_link_targets.unwrap_or(d.max_link_targets),
            },
        }
    }

    #[getter]
    fn max_headers(&self) -> usize {
        self.inner.max_headers
    }
    #[getter]
    fn max_body_bytes(&self) -> usize {
        self.inner.max_body_bytes
    }
    #[getter]
    fn max_opaque_body_bytes(&self) -> usize {
        self.inner.max_opaque_body_bytes
    }
    #[getter]
    fn max_stream_buffer_bytes(&self) -> usize {
        self.inner.max_stream_buffer_bytes
    }
    #[getter]
    fn max_decoded_bytes(&self) -> usize {
        self.inner.max_decoded_bytes
    }
    #[getter]
    fn max_streams(&self) -> usize {
        self.inner.max_streams
    }
    #[getter]
    fn max_ws_frame_bytes(&self) -> usize {
        self.inner.max_ws_frame_bytes
    }
    #[getter]
    fn ws_sample_bytes(&self) -> usize {
        self.inner.ws_sample_bytes
    }
    #[getter]
    fn max_connections(&self) -> usize {
        self.inner.max_connections
    }
    #[getter]
    fn max_sessions(&self) -> usize {
        self.inner.max_sessions
    }
    #[getter]
    fn max_session_entries(&self) -> usize {
        self.inner.max_session_entries
    }
    #[getter]
    fn max_units(&self) -> usize {
        self.inner.max_units
    }
    #[getter]
    fn max_entries_per_unit(&self) -> usize {
        self.inner.max_entries_per_unit
    }
    #[getter]
    fn mcp_sniff_bytes(&self) -> usize {
        self.inner.mcp_sniff_bytes
    }
    #[getter]
    fn max_extra_keys(&self) -> usize {
        self.inner.max_extra_keys
    }
    #[getter]
    fn max_parse_backlog(&self) -> usize {
        self.inner.max_parse_backlog
    }
    #[getter]
    fn max_parse_backlog_bytes(&self) -> usize {
        self.inner.max_parse_backlog_bytes
    }
    #[getter]
    fn max_buffer_spans(&self) -> usize {
        self.inner.max_buffer_spans
    }
    #[getter]
    fn max_buffer_bytes(&self) -> usize {
        self.inner.max_buffer_bytes
    }
    #[getter]
    fn replay_buffer_size(&self) -> usize {
        self.inner.replay_buffer_size
    }
    #[getter]
    fn max_otel_bridge_body_bytes(&self) -> usize {
        self.inner.max_otel_bridge_body_bytes
    }
    #[getter]
    fn max_otel_bridge_spans_per_session(&self) -> usize {
        self.inner.max_otel_bridge_spans_per_session
    }
    #[getter]
    fn zstd_level(&self) -> i32 {
        self.inner.zstd_level
    }
    #[getter]
    fn max_otlp_attribute_bytes(&self) -> usize {
        self.inner.max_otlp_attribute_bytes
    }
    #[getter]
    fn max_otlp_request_bytes(&self) -> usize {
        self.inner.max_otlp_request_bytes
    }
    #[getter]
    fn max_link_targets(&self) -> usize {
        self.inner.max_link_targets
    }
}

/// The core's default limits as a plain dict. The Python mirror asserts key
/// parity against this so a new limit cannot be added without mirroring it.
#[pyfunction]
fn limits_defaults(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = Limits::default();
    let out = PyDict::new_bound(py);
    out.set_item("max_headers", d.max_headers)?;
    out.set_item("max_body_bytes", d.max_body_bytes)?;
    out.set_item("max_opaque_body_bytes", d.max_opaque_body_bytes)?;
    out.set_item("max_stream_buffer_bytes", d.max_stream_buffer_bytes)?;
    out.set_item("max_decoded_bytes", d.max_decoded_bytes)?;
    out.set_item("max_streams", d.max_streams)?;
    out.set_item("max_ws_frame_bytes", d.max_ws_frame_bytes)?;
    out.set_item("ws_sample_bytes", d.ws_sample_bytes)?;
    out.set_item("max_connections", d.max_connections)?;
    out.set_item("max_sessions", d.max_sessions)?;
    out.set_item("max_session_entries", d.max_session_entries)?;
    out.set_item("max_units", d.max_units)?;
    out.set_item("max_entries_per_unit", d.max_entries_per_unit)?;
    out.set_item("mcp_sniff_bytes", d.mcp_sniff_bytes)?;
    out.set_item("max_extra_keys", d.max_extra_keys)?;
    out.set_item("max_parse_backlog", d.max_parse_backlog)?;
    out.set_item("max_parse_backlog_bytes", d.max_parse_backlog_bytes)?;
    out.set_item("max_buffer_spans", d.max_buffer_spans)?;
    out.set_item("max_buffer_bytes", d.max_buffer_bytes)?;
    out.set_item("replay_buffer_size", d.replay_buffer_size)?;
    out.set_item("max_otel_bridge_body_bytes", d.max_otel_bridge_body_bytes)?;
    out.set_item(
        "max_otel_bridge_spans_per_session",
        d.max_otel_bridge_spans_per_session,
    )?;
    out.set_item("zstd_level", d.zstd_level)?;
    out.set_item("max_otlp_attribute_bytes", d.max_otlp_attribute_bytes)?;
    out.set_item("max_otlp_request_bytes", d.max_otlp_request_bytes)?;
    out.set_item("max_link_targets", d.max_link_targets)?;
    Ok(out)
}

/// Registers `Limits` and `limits_defaults()` on the top-level `_wardex_native`
/// module — same Python-facing surface as if they were declared in `lib.rs`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLimits>()?;
    m.add_function(wrap_pyfunction!(limits_defaults, m)?)?;
    Ok(())
}
