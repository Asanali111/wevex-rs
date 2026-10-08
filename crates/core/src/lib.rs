//! Storage, schema migrations, scopes, entities and hybrid recall.

pub mod error;
pub mod lock;
pub mod paths;
pub mod storage;
pub mod value;

pub use error::{Error, Result};

/// Crate version, shared by every workspace member.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
