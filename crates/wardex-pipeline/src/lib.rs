//! Wardex pipeline — PII masking over the wire types.

// The per-span PII scrub is fail-closed through `catch_unwind` (see `pii::walk`);
// under `panic = "abort"` a masking panic would end the host process instead of
// dropping one span. The release profile sets `panic = "unwind"`.
#[cfg(not(panic = "unwind"))]
compile_error!("wardex must be built with panic = \"unwind\"; the fail-closed PII scrub relies on catch_unwind");

pub mod pii;
