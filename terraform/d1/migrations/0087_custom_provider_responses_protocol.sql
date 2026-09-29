-- Add the OpenAI Responses wire protocol to custom providers. SQLite cannot
-- alter a CHECK in place, so custom_providers is rebuilt following the 0031
-- pattern: stage and drop its two children (releasing their foreign-key
-- claims), rebuild the parent with the widened protocol CHECK, then recreate
-- and repopulate the children and indexes verbatim. Every row is preserved,
-- so all foreign keys hold at commit. Rollback is fix-forward; this
-- migration is never reversed.
PRAGMA defer_foreign_keys = TRUE;

-- === stage and drop the children ===
CREATE TABLE _bak_custom_provider_secrets AS SELECT * FROM custom_provider_secrets;
DROP TABLE custom_provider_secrets;
CREATE TABLE _bak_custom_provider_models AS SELECT * FROM custom_provider_models;
DROP TABLE custom_provider_models;

-- === rebuild the parent with the widened protocol CHECK ===
CREATE TABLE _bak_custom_providers AS SELECT * FROM custom_providers;
DROP TABLE custom_providers;

CREATE TABLE custom_providers (
  id TEXT PRIMARY KEY CHECK (length(id) = 32 AND id NOT GLOB '*[^0-9a-f]*'),
  name TEXT NOT NULL CHECK (length(name) BETWEEN 1 AND 100),
  protocol TEXT NOT NULL CHECK (protocol IN ('anthropic', 'openai_compatible', 'openai_responses')),
  base_url TEXT NOT NULL CHECK (length(base_url) BETWEEN 1 AND 2048),
  custom_headers TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
  created_by TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

INSERT INTO custom_providers (id, name, protocol, base_url, custom_headers, status, created_by, created_at, updated_at)
SELECT id, name, protocol, base_url, custom_headers, status, created_by, created_at, updated_at FROM _bak_custom_providers;

-- === recreate the children verbatim ===
CREATE TABLE custom_provider_secrets (
  provider_id TEXT PRIMARY KEY
    REFERENCES custom_providers(id) ON DELETE CASCADE,
  encrypted_payload TEXT NOT NULL,
  schema_version INTEGER NOT NULL CHECK (schema_version > 0),
  updated_at INTEGER NOT NULL
);

INSERT INTO custom_provider_secrets (provider_id, encrypted_payload, schema_version, updated_at)
SELECT provider_id, encrypted_payload, schema_version, updated_at FROM _bak_custom_provider_secrets;

CREATE TABLE custom_provider_models (
  provider_id TEXT NOT NULL
    REFERENCES custom_providers(id) ON DELETE CASCADE,
  model_id TEXT NOT NULL CHECK (length(model_id) BETWEEN 1 AND 200),
  display_name TEXT NOT NULL CHECK (length(display_name) BETWEEN 1 AND 200),
  modalities TEXT NOT NULL DEFAULT '["text"]',
  reasoning_efforts TEXT NOT NULL DEFAULT '[]',
  context_window_tokens INTEGER NOT NULL CHECK (context_window_tokens > 0),
  max_output_tokens INTEGER NOT NULL CHECK (max_output_tokens > 0),
  enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (provider_id, model_id)
);

INSERT INTO custom_provider_models (provider_id, model_id, display_name, modalities, reasoning_efforts, context_window_tokens, max_output_tokens, enabled, created_at, updated_at)
SELECT provider_id, model_id, display_name, modalities, reasoning_efforts, context_window_tokens, max_output_tokens, enabled, created_at, updated_at FROM _bak_custom_provider_models;

-- === indexes, verbatim ===
CREATE INDEX idx_custom_provider_models_enabled
  ON custom_provider_models(provider_id, enabled);

-- Model IDs route through `cp[ao]-` + the first 8 hex chars of the provider
-- ID, so that prefix pair must stay unique across providers.
CREATE UNIQUE INDEX idx_custom_provider_key_prefix
  ON custom_providers(substr(id, 1, 8), protocol);

DROP TABLE _bak_custom_provider_secrets;
DROP TABLE _bak_custom_provider_models;
DROP TABLE _bak_custom_providers;
