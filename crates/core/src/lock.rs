//! Single-instance lock for the daemon.
//!
//! An exclusive OS lock on `daemon.lock`. The kernel releases it when the
//! process exits, crashed or not, so there is no stale-PID cleanup. Uses
//! `flock` on Unix and `LockFileEx` on Windows, which conflict with the
//! `flock` / `msvcrt.locking` locks taken by Python v0.2.2: a Rust and a
//! Python daemon cannot both run against the same state directory.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use fs4::fs_std::FileExt;

use crate::{Error, Result};

#[derive(Debug)]
pub struct InstanceLock {
    file: File,
    path: PathBuf,
}

impl InstanceLock {
    /// Take the lock or fail with [`Error::AlreadyRunning`]. Never blocks.
    pub fn acquire(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        if !file.try_lock_exclusive()? {
            return Err(Error::AlreadyRunning(path.to_path_buf()));
        }
        // Informational only; the lock itself is the source of truth.
        file.set_len(0)?;
        writeln!(file, "{}", std::process::id())?;
        Ok(Self {
            file,
            path: path.to_path_buf(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for InstanceLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_acquire_fails_until_first_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("daemon.lock");

        let first = InstanceLock::acquire(&path).unwrap();
        assert!(matches!(
            InstanceLock::acquire(&path),
            Err(Error::AlreadyRunning(_))
        ));

        drop(first);
        InstanceLock::acquire(&path).unwrap();
    }
}
