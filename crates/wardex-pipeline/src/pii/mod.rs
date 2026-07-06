//! PII masking engine — design doc 2026-07-07-phase3-pii-masking.
//!
//! Detection is fully deterministic: regex proposes, checksum validators
//! confirm (design §5.1). No ML/entropy heuristics.

mod patterns;
