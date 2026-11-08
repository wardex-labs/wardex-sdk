//! Wardex core — unified facade for the Rust core libraries.
//!
//! Re-exports the core crates so language bindings depend on a single crate.

pub use wardex_codec as codec;
pub use wardex_pipeline as pipeline;
pub use wardex_protocol as protocol;
