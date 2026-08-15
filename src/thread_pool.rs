/// Flush-to-zero + denormals-are-zero. Denormal floats in the stencil kernels
/// trigger microcode assists on x86; this is thread-local MXCSR state.
#[cfg(target_arch = "x86_64")]
#[allow(deprecated)] // _mm_setcsr is the concise form; the deprecation nudge is asm-only
fn enable_ftz_daz() {
    use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
    unsafe {
        // MXCSR: bit 15 = flush-to-zero, bit 6 = denormals-are-zero
        _mm_setcsr(_mm_getcsr() | 0x8000 | 0x0040);
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn enable_ftz_daz() {}

/// A rayon pool whose workers flush denormals to zero.
pub struct Pool {
    inner: rayon::ThreadPool,
}

impl Pool {
    pub fn new(num_threads: usize) -> Self {
        Self {
            inner: rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .start_handler(|_| enable_ftz_daz())
                .build()
                .expect("failed to build rayon pool"),
        }
    }

    /// Run `op` on this pool's workers.
    pub fn install<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        self.inner.install(op)
    }
}
