use std::path::PathBuf;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(
        "database schema version {found} is newer than this build supports ({supported}); upgrade wevex"
    )]
    SchemaTooNew { found: i64, supported: i64 },

    #[error("another wevex daemon already holds {0}")]
    AlreadyRunning(PathBuf),

    #[error("storage writer thread is gone")]
    WriterGone,

    #[error("a database write panicked and was rolled back")]
    WritePanicked,

    #[error("Store::write called from inside another write")]
    ReentrantWrite,

    #[error("database failed its integrity check: {}", .0.join("; "))]
    Corrupt(Vec<String>),

    #[error(transparent)]
    Embed(#[from] wevex_embed::Error),

    #[error("could not determine the home directory")]
    NoHome,
}
