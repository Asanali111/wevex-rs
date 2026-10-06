//! Storage, schema migrations, scopes, entities and hybrid recall.

pub mod error;
pub mod lock;
pub mod paths;

pub use error::{Error, Result};

/// Crate version, shared by every workspace member.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
