//! Module for custom error-handling of recoverable errors in Rustdb crates.
mod error;
mod macros;

pub use error::{Error, Result};
#[allow(unused_imports)]
pub use macros::*;
