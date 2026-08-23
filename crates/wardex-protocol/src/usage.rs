//! Normalized token usage: the one place a provider's input-counting
//! convention becomes the semconv contract (`gen_ai.usage.input_tokens`
//! includes every input token, cached ones included). Constructed by the
//! provider parsers in `semantic.rs` and `claude_stream_json.rs`; consumed
//! through getters only, so a raw (exclusive) value cannot leak back out.
//!
//! Why a type and not a convention: the neutral structs used to carry the
//! provider's numbers with the inclusivity contract written in a comment,
//! and the Anthropic parsers silently violated it — every backend that
//! subtracts the cache tiers back out (Langfuse, Phoenix) then under-billed
//! by the cache volume. `TokenUsage::new` cannot be called without naming
//! the provider's [`InputConvention`], so the next parser faces a compile
//! error instead of a coin flip.

/// How the provider counts `input_tokens`. Choosing the constructor's
/// convention IS the declaration — there is no default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputConvention {
    /// `input_tokens` already includes cached tokens (OpenAI, Gemini, Cohere).
    Inclusive,
    /// `input_tokens` excludes the cache tiers, which are reported beside it
    /// (Anthropic, Bedrock Claude). The constructor adds them back in.
    ExcludesCache,
}

/// Token usage, always semconv-inclusive. All fields private: the getters are
/// the contract, and the only way in is [`TokenUsage::new`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenUsage {
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
    cache_read_input_tokens: Option<i64>,
    cache_creation_input_tokens: Option<i64>,
    reasoning_output_tokens: Option<i64>,
    totals_unpaired: bool,
    overflowed: bool,
}

impl TokenUsage {
    /// Normalize one provider usage report.
    ///
    /// Addition only, and only for [`InputConvention::ExcludesCache`]: no
    /// input can make a value shrink, so no negative and no `max(…, 0)` can
    /// ever be needed. `output_tokens` is never adjusted — every supported
    /// provider reports it inclusive of reasoning already.
    ///
    /// Error paths (never a panic — capture must not kill the host):
    /// * a sub-counter arrives without the total it belongs to → the total
    ///   stays `None` and [`totals_unpaired`](Self::totals_unpaired) is set;
    ///   inventing a total out of the cache tiers would be a fabrication.
    /// * `checked_add` fails (unreachable at provider magnitudes, but `i64`
    ///   arithmetic says so, not us) → the total is withheld as `None` and
    ///   [`overflowed`](Self::overflowed) is set.
    pub fn new(
        convention: InputConvention,
        input_tokens: Option<i64>,
        output_tokens: Option<i64>,
        cache_read_input_tokens: Option<i64>,
        cache_creation_input_tokens: Option<i64>,
        reasoning_output_tokens: Option<i64>,
    ) -> Self {
        let has_cache = cache_read_input_tokens.is_some() || cache_creation_input_tokens.is_some();
        let mut overflowed = false;
        let normalized_input = match (convention, input_tokens) {
            (InputConvention::Inclusive, raw) => raw,
            (InputConvention::ExcludesCache, Some(raw)) => {
                let summed = raw
                    .checked_add(cache_read_input_tokens.unwrap_or(0))
                    .and_then(|s| s.checked_add(cache_creation_input_tokens.unwrap_or(0)));
                if summed.is_none() {
                    overflowed = true;
                }
                summed
            }
            (InputConvention::ExcludesCache, None) => None,
        };
        let totals_unpaired = (input_tokens.is_none() && has_cache)
            || (output_tokens.is_none() && reasoning_output_tokens.is_some());
        Self {
            input_tokens: normalized_input,
            output_tokens,
            cache_read_input_tokens,
            cache_creation_input_tokens,
            reasoning_output_tokens,
            totals_unpaired,
            overflowed,
        }
    }

    /// The semconv-inclusive input total, whatever the provider's convention.
    pub fn input_tokens(&self) -> Option<i64> {
        self.input_tokens
    }

    /// Output total, inclusive of reasoning tokens.
    pub fn output_tokens(&self) -> Option<i64> {
        self.output_tokens
    }

    /// Cache-read tier, exactly as the provider reported it.
    pub fn cache_read_input_tokens(&self) -> Option<i64> {
        self.cache_read_input_tokens
    }

    /// Cache-write tier, exactly as the provider reported it.
    pub fn cache_creation_input_tokens(&self) -> Option<i64> {
        self.cache_creation_input_tokens
    }

    /// Reasoning tier, exactly as the provider reported it.
    pub fn reasoning_output_tokens(&self) -> Option<i64> {
        self.reasoning_output_tokens
    }

    /// A sub-counter was present but the total it belongs to was `None`, so
    /// the normative total could not be formed and was not invented.
    pub fn totals_unpaired(&self) -> bool {
        self.totals_unpaired
    }

    /// `checked_add` failed and the input total was withheld.
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// G4 — the two conventions converge on the same inclusive total.
    #[test]
    fn both_conventions_yield_the_same_inclusive_total() {
        let table: &[(InputConvention, i64, i64, i64)] = &[
            (InputConvention::Inclusive, 11000, 8000, 2000),
            (InputConvention::ExcludesCache, 1000, 8000, 2000),
        ];
        for &(conv, input, read, write) in table {
            let u = TokenUsage::new(conv, Some(input), Some(500), Some(read), Some(write), None);
            assert_eq!(u.input_tokens(), Some(11000), "{conv:?}");
            assert_eq!(u.output_tokens(), Some(500));
            assert_eq!(u.cache_read_input_tokens(), Some(read));
            assert_eq!(u.cache_creation_input_tokens(), Some(write));
            assert!(!u.totals_unpaired());
            assert!(!u.overflowed());
        }
    }

    /// The invariant the whole module exists for, spelled as arithmetic.
    #[test]
    fn input_total_is_never_below_the_sum_of_its_cache_tiers() {
        for conv in [InputConvention::Inclusive, InputConvention::ExcludesCache] {
            let raw_input = match conv {
                InputConvention::Inclusive => 11000,
                InputConvention::ExcludesCache => 1000,
            };
            let u = TokenUsage::new(conv, Some(raw_input), Some(9), Some(8000), Some(2000), None);
            let total = u.input_tokens().unwrap();
            let tiers = u.cache_read_input_tokens().unwrap_or(0)
                + u.cache_creation_input_tokens().unwrap_or(0);
            assert!(total >= tiers, "{conv:?}: {total} < {tiers}");
        }
    }

    #[test]
    fn missing_cache_fields_count_as_zero() {
        let u = TokenUsage::new(
            InputConvention::ExcludesCache,
            Some(1000),
            Some(500),
            None,
            None,
            None,
        );
        assert_eq!(u.input_tokens(), Some(1000));
        assert!(!u.totals_unpaired());
    }

    #[test]
    fn a_cache_tier_without_its_total_is_unpaired_not_invented() {
        for conv in [InputConvention::Inclusive, InputConvention::ExcludesCache] {
            let u = TokenUsage::new(conv, None, Some(500), Some(8000), None, None);
            assert_eq!(
                u.input_tokens(),
                None,
                "{conv:?}: a total must not be invented"
            );
            assert!(u.totals_unpaired());
            assert!(!u.overflowed());
        }
    }

    #[test]
    fn reasoning_without_an_output_total_is_unpaired() {
        let u = TokenUsage::new(
            InputConvention::Inclusive,
            Some(10),
            None,
            None,
            None,
            Some(3),
        );
        assert_eq!(u.output_tokens(), None);
        assert!(u.totals_unpaired());
    }

    /// §4.3 — overflow withholds the total instead of panicking.
    #[test]
    fn overflow_withholds_the_total_and_says_so() {
        let u = TokenUsage::new(
            InputConvention::ExcludesCache,
            Some(i64::MAX),
            Some(500),
            Some(1),
            None,
            None,
        );
        assert_eq!(u.input_tokens(), None);
        assert!(u.overflowed());
        assert!(!u.totals_unpaired());
        // The tiers themselves survive — only the sum was impossible.
        assert_eq!(u.cache_read_input_tokens(), Some(1));
    }

    #[test]
    fn inclusive_input_is_passed_through_untouched() {
        // Over-correction guard: OpenAI's prompt_tokens already contains the
        // cached tokens; adding them again would double-count.
        let u = TokenUsage::new(
            InputConvention::Inclusive,
            Some(11000),
            Some(500),
            Some(8000),
            None,
            Some(100),
        );
        assert_eq!(u.input_tokens(), Some(11000));
        assert_eq!(u.output_tokens(), Some(500));
        assert_eq!(u.reasoning_output_tokens(), Some(100));
    }

    /// §4.4 — constructor cost, absolute value. Not a regression gate (the
    /// type is new, there is no "before"); a report in whole-digit ns means
    /// the `Option` plumbing inlined. Run with:
    /// `cargo test --release -p wardex-protocol -- --ignored bench_token_usage_new --nocapture`
    #[test]
    #[ignore = "microbench: run explicitly with --ignored --nocapture"]
    fn bench_token_usage_new() {
        use std::hint::black_box;
        use std::time::Instant;
        const ITERS: u64 = 10_000_000;
        let mut medians = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            for i in 0..ITERS {
                let u = TokenUsage::new(
                    black_box(InputConvention::ExcludesCache),
                    black_box(Some(i as i64)),
                    black_box(Some(500)),
                    black_box(Some(8000)),
                    black_box(Some(2000)),
                    black_box(None),
                );
                black_box(u.input_tokens());
            }
            let elapsed = start.elapsed();
            medians.push(elapsed.as_nanos() as f64 / ITERS as f64);
        }
        medians.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "TokenUsage::new: {:.2} ns/call (median of 5 x {ITERS})",
            medians[2]
        );
    }
}
