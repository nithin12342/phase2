//! h5-kernel: shared pure-math primitives. No I/O, no external crates.
//!
//! This crate is the innermost dependency. Every domain crate may use it.
//! It must never depend on a domain, application, or infrastructure crate.
pub mod ho;
pub mod quality;
pub mod scan;
pub mod stats;