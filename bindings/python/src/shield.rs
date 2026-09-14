//! The panic boundary between the Rust core and the host interpreter.
//!
//! PyO3 already catches a panic that unwinds out of a `#[pyfunction]` or a
//! `#[pymethods]` entry — but it re-raises it as `pyo3_runtime.PanicException`,
//! which inherits from `BaseException`, not `Exception`. Every guard the SDK
//! puts around its own work on the host's call path catches `Exception`
//! (`_assembly/_diag.py` says why: `KeyboardInterrupt` and `CancelledError`
//! must reach the host untouched). So a panic under that default walks through
//! every guard and surfaces inside the host's own `recv()`, after the socket
//! has already consumed the bytes. `shielded` closes that gap: it catches the
//! panic here, on the Rust side of the boundary, and raises `NativePanic`
//! instead — a `RuntimeError`, which the guards do catch, count, and swallow.
//!
//! Every entry the module exposes runs its body inside `shielded`. A test
//! reads these sources and fails when one does not
//! (`sdks/python/tests/test_ffi_panic.py`). Getters that hand back an owned
//! field are the one exemption: nothing from the core runs in them.
//!
//! What this does NOT do: silence the panic hook. The default hook prints
//! `thread '<unnamed>' panicked at ...` to stderr before the unwind begins,
//! and the only way to stop it is `std::panic::set_hook`, which is
//! process-global — wardex replacing the host's panic hook would be exactly
//! the kind of implicit behavior the SDK refuses to have. So a converted panic
//! leaves one line on stderr and nothing else.

use std::panic::{catch_unwind, AssertUnwindSafe};

use pyo3::prelude::*;

pyo3::create_exception!(
    _wardex_native,
    NativePanic,
    pyo3::exceptions::PyRuntimeError,
    "A panic in wardex's Rust core, converted at the FFI boundary. A RuntimeError, \
     so the SDK's own guards swallow and count it (`ffi.panic_converted`) instead \
     of letting it reach the host's call."
);

/// Run `f`, turning a panic into `NativePanic` rather than letting it unwind
/// into PyO3's `BaseException`-derived `PanicException`.
///
/// `AssertUnwindSafe` because every caller holds `&mut self` on a parser or a
/// borrowed Python object, neither of which is `UnwindSafe` by the type
/// system's conservative rule. The state a panic leaves behind is accepted as
/// is: a parser that panicked mid-feed is a parser its caller stops feeding
/// (the seam guard counts the failure and the connection's tracker latches
/// off), and nothing here is shared across threads without the GIL.
pub(crate) fn shielded<T>(f: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => Err(NativePanic::new_err(describe(payload.as_ref()))),
    }
}

/// The panic's message, for the exception text. `panic!("literal")` carries a
/// `&str`; `panic!("{x}")` carries a `String`; anything else is opaque.
fn describe(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic with a non-string payload".to_owned()
    }
}

/// Panics on purpose, through `shielded`, so a Python test can prove the
/// conversion end to end without depending on a real bug in the core.
///
/// Compiled only with the `panic-injection` cargo feature, which the
/// development build turns on (`[tool.uv]` in the workspace `pyproject.toml`)
/// and the release wheel does not. A shipped wheel has no function whose
/// purpose is to fail.
#[cfg(feature = "panic-injection")]
#[pyfunction]
#[allow(clippy::panic)] // the whole point of the function
fn _panic_for_test() -> PyResult<()> {
    shielded(|| -> PyResult<()> { panic!("injected by _panic_for_test") })
}

/// Registers `NativePanic` (and, under the feature, `_panic_for_test`) on the
/// top-level module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("NativePanic", m.py().get_type_bound::<NativePanic>())?;
    #[cfg(feature = "panic-injection")]
    m.add_function(pyo3::wrap_pyfunction!(_panic_for_test, m)?)?;
    Ok(())
}
