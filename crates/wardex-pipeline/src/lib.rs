//! Wardex pipeline — Normalizer, Sampler, PII regex.
//!
//! Only `pii` carries an implementation; `normalizer` and `sampler` are named
//! placeholders that reserve the module path.

pub mod normalizer;
pub mod pii;
pub mod sampler;

#[cfg(test)]
mod tests {
    #[test]
    fn it_compiles() {}
}
