//! Exposes a system catalog that provides an interface to work with representations of SQL tables.
//!
//! Also provides an API (via [`crate::serde`]) for converting between serialized and deserialized
//! representations of table rows / tuples:
pub mod catalog;
pub mod column;
pub mod field;
pub mod schema;
pub mod serde;
pub mod tuple;
pub mod types;
