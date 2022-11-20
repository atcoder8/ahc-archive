//! This module provides the ability to measure execution time.

use std::time::Instant;

/// This structure provides the ability to measure execution time.
#[derive(Debug, Hash, Clone, Copy)]
pub struct StopWatch(Instant);

impl Default for StopWatch {
    fn default() -> Self {
        Self::new()
    }
}

impl StopWatch {
    /// Instantiate this structure and start the measurement.
    pub fn new() -> Self {
        Self(Instant::now())
    }

    /// Returns the time elapsed since this structure was instantiated (in seconds).
    pub fn elapsed_time(&self) -> f64 {
        let duration = self.0.elapsed();
        duration.as_secs() as f64 + duration.subsec_nanos() as f64 * 1e-9
    }
}
