//! One writer thread, N reader connections.
//!
//! SQLite in WAL mode allows many concurrent readers but a single writer.
//! Instead of letting writers contend on the file lock, every write is sent
//! to one thread that owns the only read-write connection, so writes queue
//! in order and never hit `SQLITE_BUSY`. Reads take a `query_only`
//! connection from a small pool and never wait on writes.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{self, Sender};
use std::sync::{Condvar, Mutex};
use std::thread::{JoinHandle, ThreadId};

use rusqlite::Connection;

use crate::{Error, Result};

type Job = Box<dyn FnOnce(&mut Connection) + Send>;

pub struct Writer {
    tx: Option<Sender<Job>>,
    handle: Option<JoinHandle<()>>,
    thread: ThreadId,
}

impl Writer {
    pub fn spawn(mut conn: Connection) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<Job>();
        let handle = std::thread::Builder::new()
            .name("wevex-db-writer".into())
            .spawn(move || {
                for job in rx {
                    // A panicking job must not take the thread down with it:
                    // every later write would fail while the daemon looked
                    // healthy. The job's caller gets `WritePanicked` instead.
                    if catch_unwind(AssertUnwindSafe(|| job(&mut conn))).is_err()
                        && !conn.is_autocommit()
                    {
                        // A transaction left open by the panic is undone.
                        let _ = conn.execute_batch("ROLLBACK");
                    }
                }
            })?;
        Ok(Self {
            thread: handle.thread().id(),
            tx: Some(tx),
            handle: Some(handle),
        })
    }

    /// Run `f` on the writer thread and wait for its result.
    ///
    /// Must not be called from inside another write: the writer would wait
    /// on itself. That case returns [`Error::ReentrantWrite`] instead.
    pub fn run<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection) -> Result<T> + Send + 'static,
    {
        if std::thread::current().id() == self.thread {
            return Err(Error::ReentrantWrite);
        }
        let (rtx, rrx) = mpsc::sync_channel(1);
        let job: Job = Box::new(move |conn| {
            let _ = rtx.send(f(conn));
        });
        self.tx
            .as_ref()
            .ok_or(Error::WriterGone)?
            .send(job)
            .map_err(|_| Error::WriterGone)?;
        // The reply sender is dropped without sending only if `f` panicked.
        rrx.recv().map_err(|_| Error::WritePanicked)?
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        // Closing the channel ends the thread's loop after queued jobs finish.
        drop(self.tx.take());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

pub struct Readers {
    idle: Mutex<Vec<Connection>>,
    available: Condvar,
}

impl Readers {
    pub fn new(conns: Vec<Connection>) -> Self {
        Self {
            idle: Mutex::new(conns),
            available: Condvar::new(),
        }
    }

    /// Borrow a read-only connection for the duration of `f`. Do not nest
    /// reads inside `f`: with every connection leased, the inner call would
    /// wait forever.
    pub fn run<T>(&self, f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
        let lease = self.checkout();
        f(lease.conn())
    }

    fn checkout(&self) -> Lease<'_> {
        let mut idle = self.idle.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(conn) = idle.pop() {
                return Lease {
                    pool: self,
                    conn: Some(conn),
                };
            }
            idle = self.available.wait(idle).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// Returns its connection to the pool on drop, including on panic.
struct Lease<'a> {
    pool: &'a Readers,
    conn: Option<Connection>,
}

impl Lease<'_> {
    fn conn(&self) -> &Connection {
        self.conn
            .as_ref()
            .expect("lease holds a connection until drop")
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            self.pool
                .idle
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(conn);
            self.pool.available.notify_one();
        }
    }
}
