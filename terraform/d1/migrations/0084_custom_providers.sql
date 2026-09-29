-- Custom model providers: administrator-registered gateways (base URL, API
-- key, protocol, custom headers) and the models imported from them. Model IDs
-- route through a 12-char provider key (`cpa-xxxxxxxx` / `cpo-xxxxxxxx`)
-- derived from the first 8 hex chars of the provider ID, so the static model
-- catalog stays compile-time and custom models stay deployment data.

CREATE TABLE custom_providers (
  id TEXT PRIMARY KEY CHECK (length(id) = 32 AND id NOT GLOB '*[^0-9a-f]*'),
  name TEXT NOT NULL CHECK (length(name) BETWEEN 1 AND 100),
  protocol TEXT NOT NULL CHECK (protocol IN ('anthropic', 'openai_compatible')),
  base_url TEXT NOT NULL CHECK (length(base_url) BETWEEN 1 AND 2048),
  custom_headers TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
  created_by TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE custom_provider_secrets (
  provider_id TEXT PRIMARY KEY
    REFERENCES custom_providers(id) ON DELETE CASCADE,
  encrypted_payload TEXT NOT NULL,
  schema_version INTEGER NOT NULL CHECK (schema_version > 0),
  updated_at INTEGER NOT NULL
);

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

CREATE INDEX idx_custom_provider_models_enabled
  ON custom_provider_models(provider_id, enabled);

-- Model IDs route through `cp[ao]-` + the first 8 hex chars of the provider
-- ID, so that prefix pair must stay unique across providers.
CREATE UNIQUE INDEX idx_custom_provider_key_prefix
  ON custom_providers(substr(id, 1, 8), protocol);
