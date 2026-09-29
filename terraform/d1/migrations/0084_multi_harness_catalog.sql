-- Widen the agent-harness catalog. A harness id is validated by the
-- shared catalog and the sandbox runtime, but D1 also pins the set with a
-- CHECK on sessions.harness and automations.harness. SQLite cannot alter a
-- CHECK in place and both tables are referenced by child rows, so each
-- parent is rebuilt following the 0031 pattern: stage and drop its
-- children (releasing their foreign-key claims), rebuild the parent, then
-- recreate and repopulate the children, indexes and triggers verbatim.
-- Every id is preserved, so all foreign keys hold at commit. Rollback is
-- fix-forward; this migration is never reversed.
PRAGMA defer_foreign_keys = TRUE;

-- === sessions ===
CREATE TABLE _bak_session_repositories AS SELECT * FROM session_repositories;
DROP TABLE session_repositories;
CREATE TABLE _bak_session_pull_requests AS SELECT * FROM session_pull_requests;
DROP TABLE session_pull_requests;
CREATE TABLE _bak_session_read_states AS SELECT * FROM session_read_states;
DROP TABLE session_read_states;
CREATE TABLE _bak_child_admission_leases AS SELECT * FROM child_admission_leases;
DROP TABLE child_admission_leases;
CREATE TABLE _bak_session_skill_manifests AS SELECT * FROM session_skill_manifests;
DROP TABLE session_skill_manifests;
CREATE TABLE _bak_session_model_provider_auth AS SELECT * FROM session_model_provider_auth;
DROP TABLE session_model_provider_auth;
CREATE TABLE _bak_pr_autofix_feedback AS SELECT * FROM pr_autofix_feedback;
DROP TABLE pr_autofix_feedback;
CREATE TABLE _bak_session_collaborators AS SELECT * FROM session_collaborators;
DROP TABLE session_collaborators;

CREATE TABLE "sessions_new" (
  id          TEXT    PRIMARY KEY,
  title       TEXT,
  repo_owner  TEXT,
  repo_name   TEXT,
  model       TEXT    NOT NULL DEFAULT 'claude-haiku-4-5',
  status      TEXT    NOT NULL DEFAULT 'created',
  created_at  INTEGER NOT NULL,
  updated_at  INTEGER NOT NULL,
  reasoning_effort TEXT,
  base_branch TEXT,
  parent_session_id TEXT,
  spawn_source TEXT NOT NULL DEFAULT 'user',
  spawn_depth INTEGER NOT NULL DEFAULT 0,
  automation_id TEXT,
  automation_run_id TEXT,
  scm_login TEXT,
  total_cost REAL NOT NULL DEFAULT 0,
  active_duration_ms INTEGER NOT NULL DEFAULT 0,
  message_count INTEGER NOT NULL DEFAULT 0,
  pr_count INTEGER NOT NULL DEFAULT 0,
  user_id TEXT, environment_id TEXT, latest_terminal_message_id TEXT, latest_terminal_message_created_at INTEGER, latest_terminal_message_completed_at INTEGER
  CHECK (
    (
      latest_terminal_message_id IS NULL
      AND latest_terminal_message_created_at IS NULL
      AND latest_terminal_message_completed_at IS NULL
    ) OR (
      latest_terminal_message_id IS NOT NULL
      AND latest_terminal_message_created_at IS NOT NULL
      AND latest_terminal_message_completed_at IS NOT NULL
      AND latest_terminal_message_completed_at >= latest_terminal_message_created_at
    )
  ), root_session_id TEXT, harness TEXT NOT NULL DEFAULT 'opencode'
  CHECK (harness IN ('opencode', 'claude', 'codex', 'pi', 'dsh', 'zcode')), status_revision INTEGER NOT NULL DEFAULT 0, input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0, reasoning_tokens INTEGER NOT NULL DEFAULT 0, cache_read_tokens INTEGER NOT NULL DEFAULT 0, cache_write_tokens INTEGER NOT NULL DEFAULT 0, owner_team_id TEXT REFERENCES teams(id) ON DELETE RESTRICT, visibility TEXT NOT NULL DEFAULT 'workspace' CHECK (visibility IN ('team', 'workspace', 'private') AND (visibility != 'team' OR owner_team_id IS NOT NULL)), project_id TEXT,
  CHECK ((repo_owner IS NULL) = (repo_name IS NULL)),
  CHECK (repo_owner IS NOT NULL OR base_branch IS NULL)
);
INSERT INTO "sessions_new" SELECT * FROM "sessions";
DROP TABLE "sessions";
ALTER TABLE "sessions_new" RENAME TO "sessions";

CREATE TABLE session_repositories (
  session_id  TEXT    NOT NULL,
  position    INTEGER NOT NULL,
  repo_owner  TEXT    NOT NULL,
  repo_name   TEXT    NOT NULL,
  repo_id     INTEGER,
  base_branch TEXT    NOT NULL,
  PRIMARY KEY (session_id, repo_owner, repo_name),
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
INSERT INTO session_repositories SELECT * FROM _bak_session_repositories;
DROP TABLE _bak_session_repositories;
CREATE INDEX idx_session_repositories_repo
  ON session_repositories (repo_owner, repo_name, session_id);
CREATE TABLE session_pull_requests (
  artifact_id            TEXT PRIMARY KEY,  -- matches the DO artifact id
  session_id             TEXT NOT NULL,
  repository_external_id TEXT,              -- stable provider repo id (canonical identity)
  repo_owner             TEXT NOT NULL,     -- mutable lookup/display (refreshed on rename)
  repo_name              TEXT NOT NULL,
  pr_number              INTEGER NOT NULL CHECK (pr_number > 0),
  url                    TEXT NOT NULL,
  lifecycle_state        TEXT NOT NULL CHECK (lifecycle_state IN ('open', 'closed', 'merged')),
  is_draft               INTEGER NOT NULL CHECK (is_draft IN (0, 1)),
  head_branch            TEXT NOT NULL,
  base_branch            TEXT NOT NULL,
  head_sha               TEXT,
  provider_updated_at    INTEGER,           -- provider's updated_at (epoch ms); monotonic guard source
  created_at             INTEGER NOT NULL,
  updated_at             INTEGER NOT NULL, provider_created_at INTEGER, merged_at INTEGER, closed_at INTEGER,
  -- Shared-contract invariant: draft is only meaningful while open. Enforced
  -- at the authority boundary so no writer can persist a terminal draft.
  CONSTRAINT chk_spr_draft_only_while_open CHECK (lifecycle_state = 'open' OR is_draft = 0),
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
INSERT INTO session_pull_requests SELECT * FROM _bak_session_pull_requests;
DROP TABLE _bak_session_pull_requests;
CREATE UNIQUE INDEX idx_spr_external_identity
  ON session_pull_requests (repository_external_id, pr_number)
  WHERE repository_external_id IS NOT NULL;
CREATE UNIQUE INDEX idx_spr_legacy_identity
  ON session_pull_requests (repo_owner, repo_name, pr_number)
  WHERE repository_external_id IS NULL;
CREATE INDEX idx_spr_session
  ON session_pull_requests (session_id);
CREATE INDEX idx_spr_analytics_created
  ON session_pull_requests (COALESCE(provider_created_at, created_at));
CREATE INDEX idx_spr_merged_at
  ON session_pull_requests (merged_at)
  WHERE merged_at IS NOT NULL;
CREATE TABLE session_read_states (
  user_id              TEXT NOT NULL,
  session_id           TEXT NOT NULL,
  last_read_message_id TEXT NOT NULL,
  updated_at           INTEGER NOT NULL,
  PRIMARY KEY (user_id, session_id),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
INSERT INTO session_read_states SELECT * FROM _bak_session_read_states;
DROP TABLE _bak_session_read_states;
CREATE INDEX idx_session_read_states_session
  ON session_read_states(session_id, user_id);
CREATE TABLE child_admission_leases (
  lease_token TEXT PRIMARY KEY,
  parent_session_id TEXT NOT NULL,
  child_session_id TEXT NOT NULL UNIQUE,
  expires_at INTEGER NOT NULL,
  FOREIGN KEY (parent_session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
INSERT INTO child_admission_leases SELECT * FROM _bak_child_admission_leases;
DROP TABLE _bak_child_admission_leases;
CREATE INDEX idx_child_admission_leases_parent
  ON child_admission_leases(parent_session_id, expires_at);
CREATE TABLE session_skill_manifests (
  session_id             TEXT PRIMARY KEY,
  selection_mode         TEXT NOT NULL CHECK (selection_mode IN ('all', 'none', 'profile')),
  profile_id             TEXT,
  profile_name           TEXT,
  resolver_version       INTEGER NOT NULL,
  manifest_sha256        TEXT NOT NULL,
  resolved_at            INTEGER NOT NULL,
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
  CHECK (
    (selection_mode = 'profile' AND profile_id IS NOT NULL AND profile_name IS NOT NULL)
    OR (selection_mode IN ('all', 'none') AND profile_id IS NULL AND profile_name IS NULL)
  )
);
INSERT INTO session_skill_manifests SELECT * FROM _bak_session_skill_manifests;
DROP TABLE _bak_session_skill_manifests;
CREATE TABLE session_model_provider_auth (
  session_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  auth_mode TEXT NOT NULL,
  provider_account_id TEXT,
  selection_source TEXT NOT NULL,
  inherited_from_session_id TEXT,
  created_at INTEGER NOT NULL,
  PRIMARY KEY (session_id, provider),
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
  FOREIGN KEY (provider_account_id, provider)
    REFERENCES model_provider_accounts(id, provider),
  CHECK (auth_mode IN ('provider_account', 'api_key', 'legacy_scoped_oauth')),
  CHECK (
    (auth_mode = 'provider_account' AND provider_account_id IS NOT NULL)
    OR (auth_mode IN ('api_key', 'legacy_scoped_oauth') AND provider_account_id IS NULL)
  )
);
INSERT INTO session_model_provider_auth SELECT * FROM _bak_session_model_provider_auth;
DROP TABLE _bak_session_model_provider_auth;
CREATE INDEX idx_session_model_provider_auth_account
  ON session_model_provider_auth(provider_account_id, created_at)
  WHERE provider_account_id IS NOT NULL;
CREATE TABLE pr_autofix_feedback (
  feedback_key           TEXT PRIMARY KEY,
  provider_object_kind   TEXT NOT NULL CHECK (provider_object_kind IN ('pr_comment', 'review')),
  provider_object_id     TEXT NOT NULL,
  delivery_id            TEXT NOT NULL,
  repository_external_id TEXT NOT NULL,
  repo_owner             TEXT NOT NULL,
  repo_name              TEXT NOT NULL,
  pr_number              INTEGER NOT NULL CHECK (pr_number > 0),
  artifact_id            TEXT,
  session_id             TEXT,
  author_id               TEXT,
  author_login            TEXT,
  author_type             TEXT,
  feedback_url            TEXT,
  decision                TEXT NOT NULL CHECK (decision IN ('received', 'queued', 'skipped', 'failed')),
  reason                  TEXT,
  message_id              TEXT,
  dispatch_attempted_at   INTEGER,
  delivery_count          INTEGER NOT NULL DEFAULT 1 CHECK (delivery_count > 0),
  last_error              TEXT,
  first_received_at       INTEGER NOT NULL,
  last_received_at        INTEGER NOT NULL,
  decided_at              INTEGER,
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL
);
INSERT INTO pr_autofix_feedback SELECT * FROM _bak_pr_autofix_feedback;
DROP TABLE _bak_pr_autofix_feedback;
CREATE INDEX idx_pr_autofix_feedback_activity
  ON pr_autofix_feedback (last_received_at DESC, feedback_key DESC);
CREATE INDEX idx_pr_autofix_feedback_session
  ON pr_autofix_feedback (session_id, last_received_at DESC)
  WHERE session_id IS NOT NULL;
CREATE TABLE session_collaborators (
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  added_by TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  PRIMARY KEY (session_id, user_id)
);
INSERT INTO session_collaborators SELECT * FROM _bak_session_collaborators;
DROP TABLE _bak_session_collaborators;
CREATE INDEX idx_session_collaborators_user ON session_collaborators(user_id, session_id);

CREATE INDEX idx_sessions_status_updated
  ON sessions (status, updated_at DESC);
CREATE INDEX idx_sessions_repo
  ON sessions (repo_owner, repo_name, updated_at DESC);
CREATE INDEX idx_sessions_parent_session_id
  ON sessions(parent_session_id)
  WHERE parent_session_id IS NOT NULL;
CREATE INDEX idx_sessions_automation
  ON sessions (automation_id)
  WHERE automation_id IS NOT NULL;
CREATE INDEX idx_sessions_scm_login
  ON sessions(scm_login, created_at DESC);
CREATE INDEX idx_sessions_created_at
  ON sessions(created_at DESC);
CREATE INDEX idx_sessions_user_id
  ON sessions(user_id, created_at DESC);
CREATE INDEX idx_sessions_updated_at
  ON sessions(updated_at DESC);
CREATE INDEX idx_sessions_user_updated_at
  ON sessions(user_id, updated_at DESC);
CREATE INDEX idx_sessions_user_non_automation_updated
  ON sessions(user_id, updated_at DESC)
  WHERE automation_id IS NULL;
CREATE INDEX idx_sessions_root_updated
  ON sessions(root_session_id, updated_at);
CREATE INDEX idx_sessions_export_roots
  ON sessions(created_at DESC, id ASC);
CREATE INDEX idx_sessions_export_members
  ON sessions(root_session_id, spawn_depth, created_at, id);
CREATE INDEX idx_sessions_owner_team ON sessions(owner_team_id, status, updated_at DESC);
CREATE INDEX idx_sessions_owner_team_visibility ON sessions(owner_team_id, visibility, updated_at DESC);
CREATE TRIGGER sessions_root_after_insert
AFTER INSERT ON sessions
BEGIN
  UPDATE sessions
  SET root_session_id = COALESCE(
    (SELECT root_session_id FROM sessions WHERE id = NEW.parent_session_id),
    NEW.id
  )
  WHERE id = NEW.id;
  -- A newly inserted row may satisfy dangling parent links from old data.
  UPDATE sessions
  SET root_session_id = (
    WITH RECURSIVE
      ancestors(id, parent_session_id) AS (
        SELECT sessions.id, sessions.parent_session_id
        UNION
        SELECT parent.id, parent.parent_session_id
        FROM ancestors
        JOIN sessions parent ON parent.id = ancestors.parent_session_id
      ),
      parent_reachability(start_id, id) AS (
        SELECT ancestor.id, parent.id
        FROM ancestors ancestor
        JOIN sessions parent ON parent.id = ancestor.parent_session_id
        UNION
        SELECT parent_reachability.start_id, parent.id
        FROM parent_reachability
        JOIN sessions current ON current.id = parent_reachability.id
        JOIN sessions parent ON parent.id = current.parent_session_id
      ),
      cycle_members AS (
        SELECT start_id AS id
        FROM parent_reachability
        WHERE start_id = id
      )
    SELECT COALESCE(
      (
        SELECT ancestor.id
        FROM ancestors ancestor
        LEFT JOIN sessions parent ON parent.id = ancestor.parent_session_id
        WHERE ancestor.parent_session_id IS NULL OR parent.id IS NULL
        LIMIT 1
      ),
      (
        SELECT MIN(ancestor.id)
        FROM ancestors ancestor
        JOIN cycle_members ON cycle_members.id = ancestor.id
      ),
      sessions.id
    )
  )
  WHERE id IN (
    WITH RECURSIVE affected(id) AS (
      SELECT NEW.id
      UNION
      SELECT child.id
      FROM sessions child
      JOIN affected ON child.parent_session_id = affected.id
    )
    SELECT id FROM affected
  );
  SELECT CASE
    WHEN (SELECT root_session_id FROM sessions WHERE id = NEW.id) IS NULL
    THEN RAISE(ABORT, 'session root could not be resolved')
  END;
END;
CREATE TRIGGER sessions_parent_root_after_update
AFTER UPDATE OF parent_session_id ON sessions
BEGIN
  UPDATE sessions
  SET root_session_id = (
    WITH RECURSIVE
      ancestors(id, parent_session_id) AS (
        SELECT sessions.id, sessions.parent_session_id
        UNION
        SELECT parent.id, parent.parent_session_id
        FROM ancestors
        JOIN sessions parent ON parent.id = ancestors.parent_session_id
      ),
      parent_reachability(start_id, id) AS (
        SELECT ancestor.id, parent.id
        FROM ancestors ancestor
        JOIN sessions parent ON parent.id = ancestor.parent_session_id
        UNION
        SELECT parent_reachability.start_id, parent.id
        FROM parent_reachability
        JOIN sessions current ON current.id = parent_reachability.id
        JOIN sessions parent ON parent.id = current.parent_session_id
      ),
      cycle_members AS (
        SELECT start_id AS id
        FROM parent_reachability
        WHERE start_id = id
      )
    SELECT COALESCE(
      (
        SELECT ancestor.id
        FROM ancestors ancestor
        LEFT JOIN sessions parent ON parent.id = ancestor.parent_session_id
        WHERE ancestor.parent_session_id IS NULL OR parent.id IS NULL
        LIMIT 1
      ),
      (
        SELECT MIN(ancestor.id)
        FROM ancestors ancestor
        JOIN cycle_members ON cycle_members.id = ancestor.id
      ),
      sessions.id
    )
  )
  WHERE id IN (
    WITH RECURSIVE affected(id) AS (
      SELECT NEW.id
      UNION
      SELECT child.id
      FROM sessions child
      JOIN affected ON child.parent_session_id = affected.id
    )
    SELECT id FROM affected
  );
END;
CREATE TRIGGER sessions_root_before_delete
BEFORE DELETE ON sessions
BEGIN
  UPDATE sessions
  SET parent_session_id = NULL
  WHERE parent_session_id = OLD.id;
END;
CREATE TRIGGER sessions_seed_legacy_provider_auth
AFTER INSERT ON sessions
BEGIN
  INSERT INTO session_model_provider_auth
    (session_id, provider, auth_mode, selection_source, created_at)
  VALUES (NEW.id, 'openai', 'legacy_scoped_oauth', 'legacy_migration', NEW.created_at);
  INSERT INTO session_model_provider_auth
    (session_id, provider, auth_mode, selection_source, created_at)
  VALUES (NEW.id, 'xai', 'legacy_scoped_oauth', 'legacy_migration', NEW.created_at);
  INSERT INTO session_model_provider_auth
    (session_id, provider, auth_mode, selection_source, created_at)
  VALUES (NEW.id, 'anthropic', 'api_key', 'legacy_migration', NEW.created_at);
END;

-- === automations ===
CREATE TABLE _bak_automation_invocations AS SELECT * FROM automation_invocations;
DROP TABLE automation_invocations;
CREATE TABLE _bak_automation_repositories AS SELECT * FROM automation_repositories;
DROP TABLE automation_repositories;
CREATE TABLE _bak_automation_environments AS SELECT * FROM automation_environments;
DROP TABLE automation_environments;
CREATE TABLE _bak_automation_runs AS SELECT * FROM automation_runs;
DROP TABLE automation_runs;
CREATE TABLE _bak_automation_model_provider_auth AS SELECT * FROM automation_model_provider_auth;
DROP TABLE automation_model_provider_auth;

CREATE TABLE "automations_new" (
  id                   TEXT    PRIMARY KEY,
  name                 TEXT    NOT NULL,
  instructions         TEXT    NOT NULL,
  trigger_type         TEXT    NOT NULL DEFAULT 'schedule',
  schedule_cron        TEXT,
  schedule_tz          TEXT    NOT NULL DEFAULT 'UTC',
  model                TEXT    NOT NULL,
  enabled              INTEGER NOT NULL DEFAULT 1,
  next_run_at          INTEGER,
  consecutive_failures INTEGER NOT NULL DEFAULT 0,
  created_by           TEXT    NOT NULL,
  created_at           INTEGER NOT NULL,
  updated_at           INTEGER NOT NULL,
  deleted_at           INTEGER,
  reasoning_effort     TEXT,
  event_type           TEXT,
  trigger_config       TEXT,
  trigger_auth_data    TEXT,
  user_id              TEXT
, harness TEXT NOT NULL DEFAULT 'opencode'
  CHECK (harness IN ('opencode', 'claude', 'codex', 'pi', 'dsh', 'zcode')), owner_team_id TEXT REFERENCES teams(id) ON DELETE RESTRICT);
INSERT INTO "automations_new" SELECT * FROM "automations";
DROP TABLE "automations";
ALTER TABLE "automations_new" RENAME TO "automations";

CREATE TABLE automation_invocations (
  id                 TEXT    PRIMARY KEY,
  automation_id      TEXT    NOT NULL,
  source             TEXT    NOT NULL,
  scheduled_at       INTEGER,
  trigger_key        TEXT,
  concurrency_key    TEXT,
  trigger_metadata   TEXT,
  skip_reason        TEXT,
  failure_counted_at INTEGER,
  created_at         INTEGER NOT NULL,
  updated_at         INTEGER NOT NULL,
  CHECK (source <> 'schedule' OR scheduled_at IS NOT NULL),
  FOREIGN KEY (automation_id) REFERENCES automations(id)
);
INSERT INTO automation_invocations SELECT * FROM _bak_automation_invocations;
DROP TABLE _bak_automation_invocations;
CREATE UNIQUE INDEX idx_invocations_idempotency
  ON automation_invocations (automation_id, scheduled_at)
  WHERE source = 'schedule';
CREATE UNIQUE INDEX idx_invocations_trigger_key
  ON automation_invocations (automation_id, trigger_key)
  WHERE trigger_key IS NOT NULL;
CREATE INDEX idx_invocations_concurrency
  ON automation_invocations (automation_id, concurrency_key)
  WHERE concurrency_key IS NOT NULL;
CREATE INDEX idx_invocations_automation_created
  ON automation_invocations (automation_id, created_at DESC);
CREATE INDEX idx_invocations_created
  ON automation_invocations (created_at DESC);
CREATE TABLE automation_repositories (
  automation_id TEXT    NOT NULL,
  repo_owner    TEXT    NOT NULL,
  repo_name     TEXT    NOT NULL,
  repo_id       INTEGER,
  base_branch   TEXT,
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL,
  PRIMARY KEY (automation_id, repo_owner, repo_name),
  FOREIGN KEY (automation_id) REFERENCES automations(id)
);
INSERT INTO automation_repositories SELECT * FROM _bak_automation_repositories;
DROP TABLE _bak_automation_repositories;
CREATE INDEX idx_automation_repositories_repo
  ON automation_repositories (repo_owner, repo_name);
CREATE TABLE automation_environments (
  automation_id  TEXT    NOT NULL,
  environment_id TEXT    NOT NULL,
  created_at     INTEGER NOT NULL,
  updated_at     INTEGER NOT NULL,
  PRIMARY KEY (automation_id, environment_id),
  FOREIGN KEY (automation_id) REFERENCES automations(id)
);
INSERT INTO automation_environments SELECT * FROM _bak_automation_environments;
DROP TABLE _bak_automation_environments;
CREATE INDEX idx_automation_environments_environment
  ON automation_environments (environment_id);
CREATE TABLE "automation_runs" (
  id              TEXT    PRIMARY KEY,
  automation_id   TEXT    NOT NULL,
  session_id      TEXT,
  status          TEXT    NOT NULL DEFAULT 'starting',
  skip_reason     TEXT,
  failure_reason  TEXT,
  scheduled_at    INTEGER NOT NULL,
  started_at      INTEGER,
  completed_at    INTEGER,
  created_at      INTEGER NOT NULL,
  invocation_id   TEXT    NOT NULL,
  repo_owner      TEXT,
  repo_name       TEXT,
  repo_id         INTEGER,
  base_branch     TEXT,
  environment_id  TEXT, execution_deadline_at INTEGER,
  FOREIGN KEY (automation_id) REFERENCES automations(id)
);
INSERT INTO automation_runs SELECT * FROM _bak_automation_runs;
DROP TABLE _bak_automation_runs;
CREATE INDEX idx_runs_active_lookup
  ON automation_runs (automation_id, created_at DESC)
  WHERE status IN ('starting', 'running');
CREATE INDEX idx_runs_automation_created
  ON automation_runs (automation_id, created_at DESC);
CREATE INDEX idx_runs_invocation
  ON automation_runs (invocation_id, created_at);
CREATE UNIQUE INDEX idx_runs_invocation_repo
  ON automation_runs (invocation_id, repo_owner, repo_name)
  WHERE repo_owner IS NOT NULL;
CREATE INDEX idx_runs_orphan_sweep
  ON automation_runs (created_at)
  WHERE status = 'starting';
CREATE INDEX idx_runs_session
  ON automation_runs (session_id)
  WHERE session_id IS NOT NULL;
CREATE UNIQUE INDEX idx_runs_invocation_environment
  ON automation_runs (invocation_id, environment_id)
  WHERE environment_id IS NOT NULL;
CREATE INDEX idx_runs_timeout_sweep
  ON automation_runs (execution_deadline_at)
  WHERE status = 'running';
CREATE TABLE automation_model_provider_auth (
  automation_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  auth_mode TEXT NOT NULL,
  provider_account_id TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (automation_id, provider),
  FOREIGN KEY (automation_id) REFERENCES automations(id) ON DELETE CASCADE,
  FOREIGN KEY (provider_account_id, provider)
    REFERENCES model_provider_accounts(id, provider),
  CHECK (auth_mode IN ('provider_account', 'api_key')),
  CHECK (
    (auth_mode = 'provider_account' AND provider_account_id IS NOT NULL)
    OR (auth_mode = 'api_key' AND provider_account_id IS NULL)
  )
);
INSERT INTO automation_model_provider_auth SELECT * FROM _bak_automation_model_provider_auth;
DROP TABLE _bak_automation_model_provider_auth;

CREATE INDEX idx_automations_schedule_due
  ON automations (enabled, trigger_type, next_run_at)
  WHERE enabled = 1 AND deleted_at IS NULL AND trigger_type = 'schedule';
CREATE INDEX idx_automations_sentry_match
  ON automations (trigger_type, event_type)
  WHERE enabled = 1 AND deleted_at IS NULL AND trigger_type = 'sentry';
CREATE INDEX idx_automations_active_created_id
  ON automations(created_at DESC, id DESC)
  WHERE deleted_at IS NULL;
