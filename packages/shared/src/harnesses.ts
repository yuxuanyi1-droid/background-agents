/**
 * Agent harness catalog.
 *
 * A session runs on exactly one harness — the agent that sits behind the
 * sandbox runtime's `AgentHarness` seam. The catalog is the declared
 * capability record for each harness, and `checkHarnessCompatibility` is the
 * one rule applied wherever a model or a provider-auth selection enters a
 * session.
 *
 * An id enters the catalog together with the runtime that boots it: the
 * catalog is exactly the set a session can be created on, never a wider set
 * of recognized names.
 */

import { z } from "zod";
import { extractProviderAndModel } from "./models";
import { isCustomProviderKey, type CustomProviderProtocol } from "./types/custom-providers";
import type {
  ModelProviderSelections,
  ProviderAuthMode,
  SessionProviderAuthMode,
} from "./types/provider-accounts";

export const HARNESS_IDS = ["opencode", "claude", "codex", "pi", "dsh", "zcode"] as const;
export type HarnessId = (typeof HARNESS_IDS)[number];
export const DEFAULT_HARNESS: HarnessId = "opencode";
export const harnessIdSchema = z.enum(HARNESS_IDS);

/**
 * The model family a provider segment belongs to. Custom-provider keys map
 * to protocol-scoped families (`cpa-*` → custom-anthropic, `cpo-*` →
 * custom-openai) so harness capability checks stay pure without a registry
 * lookup. Both OpenAI wire protocols (chat completions and responses) share
 * `custom-openai`; the protocol field distinguishes them at routing time.
 */
export function modelFamilyForProvider(provider: string): string {
  if (isCustomProviderKey(provider)) {
    return provider.startsWith("cpa-") ? "custom-anthropic" : "custom-openai";
  }
  return provider;
}

export interface HarnessCapabilities {
  /** User-facing name. Never "Claude Code" (branding rule). */
  readonly label: string;
  /** Model providers (catalog id prefixes) the harness can run, or any. */
  readonly modelFamilies: "any" | readonly string[];
  /** Provider id → auth modes the harness can *select* for that provider. */
  readonly providerAuth: Readonly<Partial<Record<string, readonly ProviderAuthMode[]>>>;
  /**
   * Custom-provider wire protocols the harness can run. Undefined means every
   * protocol: the family check alone decides. The model ID does not carry the
   * protocol (both OpenAI wire protocols share the `cpo-` prefix), so callers
   * holding registry metadata pass it alongside the ID.
   */
  readonly customProviderProtocols?: readonly CustomProviderProtocol[];
  /** How a sandbox restore resumes the conversation. */
  readonly resume: "session_id";
}

export const HARNESS_CATALOG = {
  opencode: {
    label: "OpenCode",
    modelFamilies: "any",
    providerAuth: {
      anthropic: ["api_key"],
      openai: ["api_key", "provider_account"],
      xai: ["api_key", "provider_account"],
    },
    resume: "session_id",
  },
  claude: {
    label: "Claude Agent",
    modelFamilies: ["anthropic", "custom-anthropic"],
    providerAuth: {
      anthropic: ["api_key", "provider_account"],
    },
    resume: "session_id",
  },
  codex: {
    label: "Codex",
    modelFamilies: ["openai", "custom-openai"],
    providerAuth: {
      openai: ["api_key"],
    },
    // Codex runs the OpenAI Responses wire API only on custom gateways; the
    // chat-completions protocol routes to the pi and dsh harnesses instead.
    customProviderProtocols: ["openai_responses"],
    resume: "session_id",
  },
  pi: {
    label: "Pi",
    modelFamilies: "any",
    providerAuth: {
      anthropic: ["api_key"],
      openai: ["api_key"],
      xai: ["api_key"],
      deepseek: ["api_key"],
      "zai-coding-plan": ["api_key"],
    },
    resume: "session_id",
  },
  dsh: {
    label: "DeepSeek Harness",
    modelFamilies: ["deepseek", "custom-anthropic", "custom-openai"],
    providerAuth: {
      deepseek: ["api_key"],
    },
    resume: "session_id",
  },
  zcode: {
    label: "ZCode",
    modelFamilies: ["zai-coding-plan"],
    providerAuth: {
      "zai-coding-plan": ["api_key"],
    },
    resume: "session_id",
  },
} as const satisfies Record<HarnessId, HarnessCapabilities>;

export function isValidHarness(value: unknown): value is HarnessId {
  return typeof value === "string" && (HARNESS_IDS as readonly string[]).includes(value);
}

/** Resolve a harness from an optional wire value; absent means the built-in harness. */
export function getValidHarnessOrDefault(value: string | null | undefined): HarnessId {
  return isValidHarness(value) ? value : DEFAULT_HARNESS;
}

export function getHarnessCapabilities(harness: HarnessId): HarnessCapabilities {
  // Widen the literal catalog entry to the interface so callers index by string.
  return HARNESS_CATALOG[harness] as HarnessCapabilities;
}

export function getHarnessLabel(harness: HarnessId): string {
  return HARNESS_CATALOG[harness].label;
}

/** Whether the harness can run a model (by its catalog provider prefix). */
export function harnessSupportsModel(harness: HarnessId, model: string): boolean {
  const families = getHarnessCapabilities(harness).modelFamilies;
  if (families === "any") return true;
  const { provider } = extractProviderAndModel(model);
  return families.includes(modelFamilyForProvider(provider));
}

/**
 * Whether a custom-provider model runs on the harness, given the wire protocol
 * the registry carries for it. The ID alone cannot answer this: both OpenAI
 * wire protocols share the `cpo-` prefix, so protocol-restricted harnesses
 * (Codex runs Responses only) need the registry metadata callers already hold.
 */
export function harnessSupportsCustomModel(
  harness: HarnessId,
  model: string,
  protocol?: CustomProviderProtocol
): boolean {
  if (!harnessSupportsModel(harness, model)) return false;
  if (protocol === undefined) return true;
  const allowed = getHarnessCapabilities(harness).customProviderProtocols;
  return allowed === undefined || allowed.includes(protocol);
}

/**
 * Whether the harness can run a provider under an auth mode.
 * `legacy_scoped_oauth` is resolver-assigned, not selectable, and passes through.
 */
export function harnessSupportsProviderAuth(
  harness: HarnessId,
  provider: string,
  mode: SessionProviderAuthMode
): boolean {
  if (mode === "legacy_scoped_oauth") return true;
  const modes = getHarnessCapabilities(harness).providerAuth[provider];
  // No row means the harness does not run the provider at all, so it cannot
  // select any auth mode for it either. The model check reports that to the
  // user; the auth resolver falls back to the API key.
  return modes !== undefined && modes.includes(mode);
}

/** Models from a list that the harness can run. */
export function filterModelsForHarness<T extends string>(
  harness: HarnessId,
  models: readonly T[]
): T[] {
  return models.filter((model) => harnessSupportsModel(harness, model));
}

/**
 * The auth mode each explicit provider selection asks for, in the shape
 * `checkHarnessCompatibility` takes. Providers without a selection are left
 * out: they resolve later, against the harness, in the auth resolver.
 */
export function selectedProviderAuthModes(
  selections: ModelProviderSelections
): Partial<Record<string, SessionProviderAuthMode>> {
  return Object.fromEntries(
    Object.entries(selections).flatMap(([provider, selection]) =>
      selection ? [[provider, selection.mode]] : []
    )
  );
}

/**
 * Explicit selections the harness can honour. A selection is dropped only
 * when the harness runs that provider but not in the selected mode (an
 * Anthropic account on OpenCode): kept, it would fail every session or run
 * created with it. A provider the harness does not run at all cannot be
 * reached through it, so that selection survives a switch back untouched.
 * Returns the same object when nothing changes.
 */
export function reconcileProviderSelectionsForHarness(
  harness: HarnessId,
  selections: ModelProviderSelections
): ModelProviderSelections {
  const providerAuth = getHarnessCapabilities(harness).providerAuth;
  const next: ModelProviderSelections = {};
  let changed = false;
  for (const [provider, selection] of Object.entries(selections)) {
    if (!selection) continue;
    const modes = providerAuth[provider];
    if (modes !== undefined && !modes.includes(selection.mode)) {
      changed = true;
      continue;
    }
    next[provider as keyof ModelProviderSelections] = selection;
  }
  return changed ? next : selections;
}

export interface HarnessCompatibilityError {
  readonly code: "model" | "provider_auth";
  readonly message: string;
}

/**
 * The one compatibility rule: can `harness` run `model` under the session's
 * resolved provider-auth modes? Applied at session create, prompt admission,
 * queued-message dispatch, child spawn, automation save, and installation
 * default resolution. Returns null when compatible.
 */
export function checkHarnessCompatibility(
  harness: HarnessId,
  model: string,
  providerAuthModes?: Readonly<Partial<Record<string, SessionProviderAuthMode>>>
): HarnessCompatibilityError | null {
  const label = getHarnessLabel(harness);
  if (!harnessSupportsModel(harness, model)) {
    return {
      code: "model",
      message: `Model "${model}" cannot run on the ${label} harness.`,
    };
  }
  const { provider } = extractProviderAndModel(model);
  const mode = providerAuthModes?.[provider];
  if (mode && !harnessSupportsProviderAuth(harness, provider, mode)) {
    return {
      code: "provider_auth",
      message:
        mode === "provider_account"
          ? `The ${label} harness cannot use a connected ${provider} account; select an API key for this session.`
          : `The ${label} harness cannot run ${provider} in ${mode} mode.`,
    };
  }
  return null;
}
