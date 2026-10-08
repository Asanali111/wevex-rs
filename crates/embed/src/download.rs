//! Fetch pinned model files over HTTPS and verify them by SHA-256.
//!
//! Files are pinned to one Hugging Face revision with known hashes, so a
//! truncated, corrupted or tampered download is rejected instead of loaded.
//! Downloads go to a temp file and are renamed into place only after the
//! hash matches, so an interrupted download never leaves a bad file behind.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

use crate::{Error, Result};

pub struct ModelFile {
    pub name: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

/// Make sure every file is present in `dir`; download the missing ones.
/// A present file is trusted if its size matches (hashing 130 MB on every
/// start would cost about half a second); downloads are always hashed.
pub fn ensure(dir: &Path, base_url: &str, files: &[ModelFile]) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for f in files {
        let path = dir.join(f.name);
        if std::fs::metadata(&path).is_ok_and(|m| m.len() == f.size) {
            continue;
        }
        fetch(&format!("{base_url}/{}", f.name), &path, f)?;
    }
    Ok(())
}

fn fetch(url: &str, dest: &Path, f: &ModelFile) -> Result<()> {
    // Unique per download, so two processes (daemon and CLI on first run)
    // fetching at once never write to or rename each other's temp file.
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let tmp = PathBuf::from(format!(
        "{}.{}-{}.part",
        dest.display(),
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let mut response = ureq::get(url)
        .call()
        .map_err(|e| Error::Download(format!("{url}: {e}")))?;
    let mut body = response.body_mut().with_config().limit(f.size + 1).reader();

    let mut out = File::create(&tmp)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut total = 0u64;
    loop {
        let n = body.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        out.write_all(&buf[..n])?;
        total += n as u64;
    }
    out.sync_all()?;
    drop(out);

    let digest = hex(&hasher.finalize());
    if total != f.size || digest != f.sha256 {
        let _ = std::fs::remove_file(&tmp);
        return Err(Error::Download(format!(
            "{}: got {total} bytes sha256 {digest}, expected {} bytes sha256 {}",
            f.name, f.size, f.sha256
        )));
    }
    if let Err(e) = std::fs::rename(&tmp, dest) {
        let _ = std::fs::remove_file(&tmp);
        // Another process finished the same file first: that is fine.
        if !std::fs::metadata(dest).is_ok_and(|m| m.len() == f.size) {
            return Err(e.into());
        }
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
