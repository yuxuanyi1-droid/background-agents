-- Public model-catalog cache (OpenRouter /api/v1/models) used to prefill
-- metadata when importing custom-provider models. The catalog is read and
-- written wholesale as one JSON blob, never queried per model, so a
-- single-row table keeps the schema and upsert trivial.
CREATE TABLE model_catalog_cache (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  source TEXT NOT NULL,
  payload TEXT NOT NULL,
  fetched_at INTEGER NOT NULL
);
