-- Sandbox backend chosen at session create; NULL means the deployment
-- default (SANDBOX_PROVIDER), which covers rows created before per-session
-- provider selection existed.
ALTER TABLE sessions ADD COLUMN sandbox_provider TEXT;
