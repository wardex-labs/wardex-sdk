//! The process-wide bound on body bytes the stream parsers hold for messages still arriving.
//!
//! Each parser stores a message's body as it arrives, up to the cap its Content-Type selects. With
//! many messages in flight at once that is up to `max_body_bytes` per message and nothing summed
//! it. This module sums it: a byte a parser stores for a message in flight is reserved here first,
//! against `Limits::max_in_flight_body_bytes`, and given back when the message leaves the parser
//! (completed and handed to the caller, ended by `finish`, or dropped with the parser). A body
//! the budget has no room for stops being stored where it stands, exactly as at its own cap, and
//! the message says so (`body_cap_exceeded`). Bodies grow side by side as their bytes arrive, so
//! with many large ones in flight each is cut wherever the budget filled, not only the last to
//! start.
//!
//! The budget is not first come, first served for every body: one that cannot be a model call's
//! by its exchange's target (`ceiling`) may take only half of it, so downloads, uploads and
//! long-lived feeds can never take the room a model call needs.
//!
//! A completed message is the caller's to bound: once handed back, its body is no longer held
//! here.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Bytes reserved by every message in flight in this process.
static HELD: AtomicUsize = AtomicUsize::new(0);

/// Body bytes the parsers in this process hold for messages still arriving.
pub fn held() -> usize {
    HELD.load(Ordering::Relaxed)
}

/// The level up to which a body may take from the budget (`Reservation::grow`'s `ceiling`): all of
/// `max_in_flight_body_bytes` for a body of an exchange with an LLM endpoint (`model_call`), half
/// of it for any other. The other half is therefore always there for model calls, however many
/// other bodies are arriving.
pub(crate) fn ceiling(max_in_flight_body_bytes: usize, model_call: bool) -> usize {
    if model_call {
        max_in_flight_body_bytes
    } else {
        max_in_flight_body_bytes / 2
    }
}

/// One message's share of the budget. Dropping it gives the share back.
#[derive(Debug)]
pub(crate) struct Reservation {
    /// The process-wide counter, or a private one in this crate's tests: the tests run in
    /// parallel, and a shared counter would let one test's message take another's budget.
    pool: &'static AtomicUsize,
    /// Bytes this reservation holds in `pool`.
    bytes: usize,
}

impl Default for Reservation {
    fn default() -> Self {
        Self::in_pool(process_pool())
    }
}

/// The counter every parser outside this crate's tests reserves from.
pub(crate) fn process_pool() -> &'static AtomicUsize {
    &HELD
}

impl Reservation {
    /// An empty reservation against `pool`.
    pub(crate) fn in_pool(pool: &'static AtomicUsize) -> Self {
        Self { pool, bytes: 0 }
    }

    /// Reserves up to `want` more bytes without taking the pool past `ceiling`, and returns how
    /// many it reserved: `want` while there is room, less once the pool is near `ceiling`, 0 at it.
    pub(crate) fn grow(&mut self, want: usize, ceiling: usize) -> usize {
        let mut current = self.pool.load(Ordering::Relaxed);
        loop {
            let take = want.min(ceiling.saturating_sub(current));
            if take == 0 {
                return 0;
            }
            // `take <= ceiling - current`, so the sum cannot pass `ceiling`.
            match self.pool.compare_exchange_weak(
                current,
                current + take,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.bytes += take;
                    return take;
                }
                Err(now) => current = now,
            }
        }
    }

    /// Gives back everything this reservation holds.
    pub(crate) fn release(&mut self) {
        if self.bytes > 0 {
            self.pool.fetch_sub(self.bytes, Ordering::Relaxed);
            self.bytes = 0;
        }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
pub(crate) fn private_pool() -> &'static AtomicUsize {
    Box::leak(Box::new(AtomicUsize::new(0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grows_up_to_the_ceiling_and_no_further() {
        let pool = private_pool();
        let mut first = Reservation::in_pool(pool);
        let mut second = Reservation::in_pool(pool);
        assert_eq!(first.grow(6, 10), 6);
        assert_eq!(
            second.grow(6, 10),
            4,
            "only the room left under the ceiling"
        );
        assert_eq!(second.grow(1, 10), 0, "the pool is at the ceiling");
        assert_eq!(pool.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn other_bodies_leave_half_the_budget_to_model_calls() {
        let pool = private_pool();
        let mut download = Reservation::in_pool(pool);
        let mut model_call = Reservation::in_pool(pool);
        assert_eq!(download.grow(100, ceiling(100, false)), 50);
        assert_eq!(download.grow(1, ceiling(100, false)), 0);
        assert_eq!(model_call.grow(60, ceiling(100, true)), 50);
    }

    #[test]
    fn release_and_drop_give_the_bytes_back() {
        let pool = private_pool();
        let mut first = Reservation::in_pool(pool);
        assert_eq!(first.grow(8, 10), 8);
        first.release();
        assert_eq!(pool.load(Ordering::Relaxed), 0);
        {
            let mut second = Reservation::in_pool(pool);
            assert_eq!(second.grow(10, 10), 10);
        }
        assert_eq!(
            pool.load(Ordering::Relaxed),
            0,
            "a dropped reservation holds nothing"
        );
    }
}
