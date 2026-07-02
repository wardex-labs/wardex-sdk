//! Wardex core — unified facade for the Rust core libraries.
//!
//! Re-exports four core crates so language bindings depend on a single crate.

pub use wardex_codec as codec;
pub use wardex_pipeline as pipeline;
pub use wardex_protocol as protocol;
pub use wardex_replay as replay;

#[cfg(test)]
mod tests {
    #[test]
    fn it_compiles() {}
}
