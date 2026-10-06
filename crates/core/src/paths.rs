//! Per-user state directory. Matches Python v0.2.2 so both can share a database:
//!
//! * macOS / Linux: `~/.config/wevex/`
//! * Windows: `%APPDATA%\wevex\` (fallback `~\AppData\Roaming\wevex\`)
//!
//! `WEVEX_HOME` overrides both, for tests and side-by-side runs.

use std::path::PathBuf;

use crate::{Error, Result};

pub fn state_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("WEVEX_HOME") {
        return Ok(PathBuf::from(dir));
    }
    if cfg!(windows) {
        if let Some(appdata) = std::env::var_os("APPDATA") {
            return Ok(PathBuf::from(appdata).join("wevex"));
        }
        return Ok(home()?.join("AppData").join("Roaming").join("wevex"));
    }
    Ok(home()?.join(".config").join("wevex"))
}

pub fn db_path() -> Result<PathBuf> {
    Ok(state_dir()?.join("wevex.db"))
}

pub fn lock_path() -> Result<PathBuf> {
    Ok(state_dir()?.join("daemon.lock"))
}

fn home() -> Result<PathBuf> {
    let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    std::env::var_os(var)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .ok_or(Error::NoHome)
}
