/**
 * Resolves the user-defined environment a session's sandbox receives: decrypts
 * and folds global/repo/environment secrets, derives the managed-provider env
 * from the session's provider auth modes, and answers whether a model's
 * provider has usable authentication in that environment. Legacy session rows
 * that predate `repo_id` resolve it through the injected `resolveRepoId`
 * capability when the repo-scoped secrets lookup needs it.
 */

import { SessionIndexStore } from "../db/session-index";
import { GlobalSecretsStore } from "../db/global-secrets";
import { RepoSecretsStore } from "../db/repo-secrets";
import { EnvironmentSecretsStore } from "../db/environment-secrets";
import { CustomProviderStore } from "../db/custom-providers";
import {
  auditSecretsMerge,
  mergeSecretSources,
  parseSecretsCapMode,
} from "../db/secrets-validation";
import {
  getProviderAuthenticationError as resolveProviderAuthenticationError,
  prepareManagedProviderEnv,
} from "../sandbox/managed-provider-env";
import {
  SUBSCRIPTION_PROVIDER_DISPLAY_METADATA,
  type SessionProviderAuthMode,
  type SubscriptionProviderId,
} from "@open-inspect/shared/types/provider-accounts";
import { isCustomModelId } from "@open-inspect/shared/types/custom-providers";
import { ModelProviderAccountStore } from "../db/model-provider-accounts";
import type { SqlDatabase } from "../db/sql-database";
import type { Logger } from "../logger";
import { resolvePublicSessionId } from "./public-session-id";
import { buildSessionTargetSecretSources } from "./session-target-secrets";
import type { SessionRepositoryEntry } from "./repository-target";
import type { SessionCoreRepository } from "./session-core-repository";
import type { SessionRow } from "./types";

/** Ceiling on the serialized custom-provider manifest handed to a sandbox. */
const CUSTOM_PROVIDER_MANIFEST_MAX_BYTES = 128 * 1024;

/**
 * Dependencies injected into UserEnvResolver.
 */
export interface UserEnvResolverDeps {
  db: SqlDatabase;
  sessionCoreRepository: SessionCoreRepository;
  /**
   * Resolves (and persists) the session's primary repo id for legacy rows
   * that predate `repo_id` — see `resolveSessionRepoId`. Injected as a
   * capability so this class carries no SCM-provider dependency.
   */
  resolveRepoId: (session: SessionRow) => Promise<number>;
  /** The owning Durable Object's id; the resolvePublicSessionId fallback. */
  durableObjectId: string;
  repoSecretsEncryptionKey: string;
  secretsCapEnforcement: string | undefined;
  /** Key for provider-account credentials; custom provider keys share it. */
  providerAccountsEncryptionKey: string | undefined;
  /** The session-scoped logger; the composition root creates it before this class. */
  log: Logger;
}

interface UserEnvContext {
  sandboxEnv: Record<string, string>;
  providerAuthModes: Record<SubscriptionProviderId, SessionProviderAuthMode>;
  /** Bound account per provider in provider_account mode. */
  providerAccountIds: Partial<Record<SubscriptionProviderId, string>>;
}

export class UserEnvResolver {
  private readonly db: SqlDatabase;
  private readonly sessionCoreRepository: SessionCoreRepository;
  private readonly resolveRepoId: (session: SessionRow) => Promise<number>;
  private readonly durableObjectId: string;
  private readonly repoSecretsEncryptionKey: string;
  private readonly providerAccountsEncryptionKey: string | undefined;
  private readonly secretsCapEnforcement: string | undefined;
  private readonly log: Logger;

  constructor(deps: UserEnvResolverDeps) {
    this.db = deps.db;
    this.sessionCoreRepository = deps.sessionCoreRepository;
    this.resolveRepoId = deps.resolveRepoId;
    this.durableObjectId = deps.durableObjectId;
    this.repoSecretsEncryptionKey = deps.repoSecretsEncryptionKey;
    this.providerAccountsEncryptionKey = deps.providerAccountsEncryptionKey;
    this.secretsCapEnforcement = deps.secretsCapEnforcement;
    this.log = deps.log;
  }

  /**
   * The user-defined environment for the sandbox, or undefined when the
   * session is missing or the assembled environment is empty.
   */
  async getUserEnvVars(): Promise<Record<string, string> | undefined> {
    const context = await this.loadUserEnvContext();
    if (!context) return undefined;
    return Object.keys(context.sandboxEnv).length === 0 ? undefined : context.sandboxEnv;
  }

  /**
   * A user-facing error message when the model's provider has no usable
   * authentication in the assembled environment, or null when authenticated.
   */
  async getProviderAuthenticationError(model: string): Promise<string | null> {
    if (isCustomModelId(model)) return this.checkCustomModel(model);
    const context = await this.loadUserEnvContext();
    if (!context) return null;
    const accountIssue = await this.checkBoundAccount(model, context);
    if (accountIssue) return accountIssue;
    const issue = resolveProviderAuthenticationError(
      model,
      context.sandboxEnv,
      context.providerAuthModes
    );
    if (!issue) return null;
    this.log.error("provider_auth.unavailable", {
      event: "provider_auth.unavailable",
      provider: issue.provider,
      auth_mode: context.providerAuthModes[issue.provider],
    });
    return issue.message;
  }

  /**
   * Pre-spawn check: a session bound to a connected account that has since
   * been disabled, archived, or fenced fails the prompt in the queue, before
   * a sandbox is spawned into a credential denial.
   */
  private async checkBoundAccount(model: string, context: UserEnvContext): Promise<string | null> {
    const provider = model.split("/", 1)[0] as SubscriptionProviderId;
    const accountId = context.providerAccountIds[provider];
    if (!accountId || context.providerAuthModes[provider] !== "provider_account") return null;
    const account = await new ModelProviderAccountStore(this.db).getById(accountId);
    const subscription = SUBSCRIPTION_PROVIDER_DISPLAY_METADATA[provider].subscriptionName;
    if (!account || account.archivedAt !== null) {
      return `The connected ${subscription} account for this session was removed. Start a new session with another account or an API key.`;
    }
    if (account.status !== "active") {
      const state = account.status === "disabled" ? "disabled" : "needs to be reconnected";
      this.log.error("provider_auth.account_unusable", {
        event: "provider_auth.account_unusable",
        provider,
        provider_account_id: accountId,
        status: account.status,
      });
      return `The connected ${subscription} account for this session ${state}. Reconnect it in Settings, then start a new session.`;
    }
    return null;
  }

  /**
   * A custom-provider model must still resolve against the registry at
   * prompt time: the provider must exist and be active with a stored key,
   * and the model must be imported and enabled.
   */
  private async checkCustomModel(model: string): Promise<string | null> {
    const store = this.customProviderStore();
    const resolved = store ? await store.resolveCustomModel(model) : null;
    if (!resolved) {
      this.log.error("custom_model.unresolvable", {
        event: "custom_model.unresolvable",
        model,
      });
      return `The custom provider model "${model}" is no longer available. Ask an administrator to restore it, then start a new session.`;
    }
    return null;
  }

  private customProviderStore(): CustomProviderStore | null {
    if (!this.providerAccountsEncryptionKey) return null;
    return new CustomProviderStore(this.db, this.providerAccountsEncryptionKey);
  }

  private async loadUserEnvContext(): Promise<UserEnvContext | null> {
    const session = this.sessionCoreRepository.getSession();
    if (!session) {
      this.log.warn("Cannot load secrets: no session");
      return null;
    }

    const db = this.db;
    const providerAuth = await new SessionIndexStore(db).getCompleteProviderAuth(
      resolvePublicSessionId(session, this.durableObjectId)
    );
    const providerAuthModes = Object.fromEntries(
      providerAuth.map(({ provider, authMode }) => [provider, authMode])
    ) as Record<SubscriptionProviderId, SessionProviderAuthMode>;
    const providerAccountIds = Object.fromEntries(
      providerAuth.flatMap((auth) =>
        "providerAccountId" in auth && auth.providerAccountId
          ? [[auth.provider, auth.providerAccountId]]
          : []
      )
    ) as Partial<Record<SubscriptionProviderId, string>>;

    // Fail hard on secret loading — sandboxes must not silently lose secrets
    const encryptionKey = this.repoSecretsEncryptionKey;
    const globalStore = new GlobalSecretsStore(db, encryptionKey);
    const globalSecrets = await globalStore.getDecryptedSecrets();

    const repoStore = new RepoSecretsStore(db, encryptionKey);
    const environmentSecretsStore = new EnvironmentSecretsStore(db, encryptionKey);
    const members = this.sessionCoreRepository.getSessionRepositories();
    const sources = await buildSessionTargetSecretSources({
      environmentId: session.environment_id,
      globalSecrets,
      members,
      loadMemberSecrets: (member) => this.loadMemberRepoSecrets(session, member, repoStore),
      loadEnvironmentSecrets: (environmentId) =>
        environmentSecretsStore.getDecryptedSecrets(environmentId),
    });

    const merge = mergeSecretSources(sources);
    auditSecretsMerge({
      merge,
      mode: parseSecretsCapMode(this.secretsCapEnforcement),
      log: this.log,
      context: { session_id: session.id },
    });

    const mergedCount = Object.keys(merge.merged).length;
    if (mergedCount > 0) {
      this.log.info("Secrets merged for sandbox", {
        source_count: sources.length,
        merged_count: mergedCount,
        payload_bytes: merge.totalBytes,
        exceeds_limit: merge.exceedsLimit,
      });
    }

    const primary = members.find((member) => member.isPrimary);
    const managedSources = session.environment_id
      ? sources
      : sources.filter(
          (source) =>
            source.label === "global" ||
            (primary && source.label === `${primary.repoOwner}/${primary.repoName}`)
        );
    const managedSecrets = mergeSecretSources(managedSources).merged;
    const sandboxEnv = prepareManagedProviderEnv({
      exposedSecrets: merge.merged,
      brokerSecrets: managedSecrets,
      providerAuthModes,
    });
    await this.injectCustomProviderEnv(sandboxEnv);
    return { sandboxEnv, providerAuthModes, providerAccountIds };
  }

  /**
   * Fold the custom-provider manifest and decrypted keys into the sandbox
   * env: `OI_CUSTOM_PROVIDERS` carries every active provider's routing and
   * model metadata (never a key), and one `CP_XXXXXXXX_API_KEY` var carries
   * each provider's credential. Both harnesses read these; see
   * sandbox-runtime's opencode_server and claude harness.
   */
  private async injectCustomProviderEnv(sandboxEnv: Record<string, string>): Promise<void> {
    const store = this.customProviderStore();
    if (!store) return;
    const entries = await store.getSandboxEntries();
    if (entries.length === 0) return;
    const manifest = JSON.stringify(entries);
    if (manifest.length > CUSTOM_PROVIDER_MANIFEST_MAX_BYTES) {
      this.log.error("custom_provider.manifest_too_large", {
        event: "custom_provider.manifest_too_large",
        bytes: manifest.length,
        providers: entries.length,
      });
      return;
    }
    for (const entry of entries) {
      const apiKey = await store.readApiKey(entry.id);
      if (apiKey === null) continue;
      sandboxEnv[entry.apiKeyEnv] = apiKey;
    }
    sandboxEnv.CUSTOM_MODEL_PROVIDERS = manifest;
  }

  /**
   * Decrypt one member repo's secrets — the injected leaf loader for
   * buildSessionTargetSecretSources. The member row carries the repo id; a
   * synthesized primary (legacy scalar row) resolves it lazily via the
   * injected resolveRepoId. A member without a resolvable id (a secondary
   * with a null row id) can't be keyed, so it contributes nothing.
   */
  private async loadMemberRepoSecrets(
    session: SessionRow,
    member: SessionRepositoryEntry,
    repoStore: RepoSecretsStore
  ): Promise<Record<string, string>> {
    const repoId =
      member.row?.repo_id ?? (member.isPrimary ? await this.resolveRepoId(session) : null);
    if (repoId === null) {
      return {};
    }
    return repoStore.getDecryptedSecrets(repoId);
  }
}
