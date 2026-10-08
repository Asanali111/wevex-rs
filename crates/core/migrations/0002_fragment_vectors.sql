-- 0002: embeddings move out of fragments.content_embedding into their own
-- table, keyed by the model that produced them.
--
-- `model` is the embedder id (model + pinned weights). A vector whose model
-- or content_hash no longer matches is re-embedded, so vectors from
-- different models are never compared. The legacy content_embedding column
-- is left untouched for Python v0.2.2 running side by side; the Rust core
-- never reads it.

CREATE TABLE IF NOT EXISTS fragment_vectors (
  fragment_id  TEXT PRIMARY KEY REFERENCES fragments(id) ON DELETE CASCADE,
  model        TEXT NOT NULL,
  content_hash TEXT NOT NULL,   -- sha256 of the embedded text, hex
  vector       BLOB NOT NULL,   -- f32 little-endian, unit length
  created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_fragment_vectors_model ON fragment_vectors(model);

-- v0.2.2's trigger fired on *every* UPDATE, so recall bookkeeping
-- (recall_hits, last_recalled_at, value) stamped fragments as just edited.
-- Only changes to what a fragment says or how it is classified count now.
DROP TRIGGER IF EXISTS fragments_updated_at;
CREATE TRIGGER fragments_updated_at
  AFTER UPDATE OF content, type, tags, territory, metadata, confidence,
                  is_stale, stale_reason, expires_at, permanent
  ON fragments BEGIN
    UPDATE fragments SET updated_at = datetime('now') WHERE id = new.id;
  END;
