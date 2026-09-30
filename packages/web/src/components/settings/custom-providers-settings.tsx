"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import {
  CUSTOM_MODEL_MODALITIES,
  CUSTOM_MODEL_REASONING_EFFORTS,
  CUSTOM_PROVIDER_PROTOCOL_LABELS,
  type CustomModelRecord,
  type CustomProviderConnectionTestResult,
  type CustomProviderHeader,
  type CustomProviderProtocol,
  type CustomProviderRecord,
  type SyncedCustomProviderModel,
} from "@open-inspect/shared/types/custom-providers";
import { browserApiFetch } from "@/lib/browser-api-fetch";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { ChevronRightIcon, CustomProviderIcon } from "@/components/ui/icons";

interface ImportedModelDraft {
  modelId: string;
  displayName: string;
  modalities: string[];
  reasoningEfforts: string[];
  contextWindowTokens: number;
  maxOutputTokens: number;
}

/** Import defaults for one synced model; catalog metadata fills what it has. */
function draftForModel(model: SyncedCustomProviderModel): ImportedModelDraft {
  const catalog = model.catalog;
  const modalities = catalog
    ? [...new Set([...catalog.inputModalities, ...catalog.outputModalities])]
    : [];
  return {
    modelId: model.modelId,
    displayName: model.displayName,
    modalities: modalities.length > 0 ? modalities : ["text"],
    reasoningEfforts: [],
    contextWindowTokens: catalog?.contextWindowTokens ?? 128_000,
    maxOutputTokens: catalog?.maxOutputTokens ?? 8_192,
  };
}

async function api<T>(path: `/api/${string}`, init?: RequestInit): Promise<T> {
  const response = await browserApiFetch(path, init);
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      typeof body === "object" && body !== null && "error" in body && typeof body.error === "string"
        ? body.error
        : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return body as T;
}

/**
 * The two protocol families a provider key encodes: switching family re-keys
 * the provider (`cpa-…` ↔ `cpo-…`) and with it every imported model ID.
 */
function protocolFamily(protocol: CustomProviderProtocol): "anthropic" | "openai" {
  return protocol === "anthropic" ? "anthropic" : "openai";
}

interface ConnectionTestState {
  running: boolean;
  result: CustomProviderConnectionTestResult | null;
  error: string | null;
}

const IDLE_TEST: ConnectionTestState = { running: false, result: null, error: null };

function TestResultLine({ state }: { state: ConnectionTestState }) {
  if (state.running) {
    return <span className="text-xs text-muted-foreground">Testing connection…</span>;
  }
  if (state.error) {
    return (
      <span className="text-xs text-destructive" role="alert">
        {state.error}
      </span>
    );
  }
  if (!state.result) return null;
  const { result } = state;
  return (
    <span className={cn("text-xs", result.ok ? "text-success" : "text-destructive")} role="status">
      {result.ok ? "Connected" : "Failed"} — {result.detail} ({result.latencyMs} ms)
    </span>
  );
}

function HeaderRows({
  headers,
  onChange,
  disabled,
}: {
  headers: CustomProviderHeader[];
  onChange: (headers: CustomProviderHeader[]) => void;
  disabled?: boolean;
}) {
  return (
    <div className="space-y-2">
      {headers.map((header, index) => (
        <div key={index} className="flex items-center gap-2">
          <Input
            aria-label={`Header name ${index + 1}`}
            placeholder="Header name"
            value={header.name}
            disabled={disabled}
            onChange={(event) => {
              const next = [...headers];
              next[index] = { ...header, name: event.target.value };
              onChange(next);
            }}
            className="flex-1"
          />
          <Input
            aria-label={`Header value ${index + 1}`}
            placeholder="Value"
            value={header.value}
            disabled={disabled}
            onChange={(event) => {
              const next = [...headers];
              next[index] = { ...header, value: event.target.value };
              onChange(next);
            }}
            className="flex-1"
          />
          <Button
            type="button"
            variant="subtle"
            size="xs"
            disabled={disabled}
            onClick={() => onChange(headers.filter((_, position) => position !== index))}
          >
            Remove
          </Button>
        </div>
      ))}
      <Button
        type="button"
        variant="subtle"
        size="xs"
        disabled={disabled}
        onClick={() => onChange([...headers, { name: "", value: "" }])}
      >
        Add header
      </Button>
    </div>
  );
}

function MultiSelect({
  options,
  selected,
  onChange,
  label,
}: {
  options: readonly string[];
  selected: string[];
  onChange: (next: string[]) => void;
  label: string;
}) {
  return (
    <div className="flex flex-wrap gap-3" role="group" aria-label={label}>
      {options.map((option) => {
        const active = selected.includes(option);
        return (
          <label key={option} className="flex items-center gap-1.5 text-sm text-foreground">
            <Checkbox
              checked={active}
              onCheckedChange={() =>
                onChange(
                  active ? selected.filter((value) => value !== option) : [...selected, option]
                )
              }
              aria-label={`${label}: ${option}`}
            />
            {option}
          </label>
        );
      })}
    </div>
  );
}

function ProviderDialog({
  open,
  onOpenChange,
  onSaved,
  existing,
  modelCount,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSaved: () => void;
  existing: CustomProviderRecord | null;
  /** Imported-model count, so a protocol family switch can warn about re-keying. */
  modelCount: number;
}) {
  const [name, setName] = useState(existing?.name ?? "");
  const [protocol, setProtocol] = useState<CustomProviderProtocol>(
    existing?.protocol ?? "anthropic"
  );
  const [baseUrl, setBaseUrl] = useState(existing?.baseUrl ?? "");
  const [apiKey, setApiKey] = useState("");
  const [headers, setHeaders] = useState<CustomProviderHeader[]>(existing?.headers ?? []);
  const [saving, setSaving] = useState(false);

  const protocolChanged = existing !== null && protocol !== existing.protocol;
  const rekeysModels =
    protocolChanged &&
    protocolFamily(protocol) !== protocolFamily(existing.protocol) &&
    modelCount > 0;

  const save = async () => {
    setSaving(true);
    try {
      if (existing) {
        await api(`/api/custom-providers/${existing.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name,
            ...(protocolChanged ? { protocol } : {}),
            baseUrl,
            headers: headers.filter((header) => header.name && header.value),
            ...(apiKey ? { apiKey } : {}),
          }),
        });
      } else {
        await api("/api/custom-providers", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name,
            protocol,
            baseUrl,
            apiKey,
            headers: headers.filter((header) => header.name && header.value),
          }),
        });
      }
      toast.success(existing ? "Provider updated" : "Provider created");
      onSaved();
      onOpenChange(false);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to save provider");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <div className="mb-4">
          <DialogTitle>{existing ? "Edit custom provider" : "Add custom provider"}</DialogTitle>
          <DialogDescription>
            Point Open-Inspect at a gateway. The base URL is the API root including the version
            segment (for example <code>https://api.example.com/v1</code> or an Anthropic-compatible
            root); <code>/models</code> is appended for model-list sync.
          </DialogDescription>
        </div>
        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="custom-provider-name">Name</Label>
            <Input
              id="custom-provider-name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="My gateway"
            />
          </div>
          <div className="space-y-1.5">
            <Label>Protocol</Label>
            <Select
              value={protocol}
              onValueChange={(value) => setProtocol(value as CustomProviderProtocol)}
            >
              <SelectTrigger aria-label="Protocol">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {(Object.keys(CUSTOM_PROVIDER_PROTOCOL_LABELS) as CustomProviderProtocol[]).map(
                  (value) => (
                    <SelectItem key={value} value={value}>
                      {CUSTOM_PROVIDER_PROTOCOL_LABELS[value]}
                    </SelectItem>
                  )
                )}
              </SelectContent>
            </Select>
            {protocolChanged && (
              <p
                className={cn(
                  "text-xs",
                  rekeysModels ? "text-destructive" : "text-muted-foreground"
                )}
              >
                {rekeysModels
                  ? `Switching protocol family re-keys this provider's ${modelCount} imported model${
                      modelCount === 1 ? "" : "s"
                    } (cpa-… ↔ cpo-…): their IDs change, and sessions pinned to the old IDs must re-select the model.`
                  : "Switching between the two OpenAI protocols keeps model IDs unchanged."}
              </p>
            )}
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="custom-provider-base-url">Base URL</Label>
            <Input
              id="custom-provider-base-url"
              value={baseUrl}
              onChange={(event) => setBaseUrl(event.target.value)}
              placeholder="https://gateway.example/v1"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="custom-provider-api-key">
              API key{existing ? " (leave blank to keep)" : ""}
            </Label>
            <Input
              id="custom-provider-api-key"
              type="password"
              value={apiKey}
              onChange={(event) => setApiKey(event.target.value)}
              placeholder="sk-..."
              autoComplete="new-password"
            />
          </div>
          <div className="space-y-1.5">
            <Label>Custom headers</Label>
            <HeaderRows headers={headers} onChange={setHeaders} disabled={saving} />
          </div>
        </div>
        <div className="mt-4 flex flex-col gap-3 border-t border-border-muted pt-4 sm:flex-row sm:items-center sm:justify-between">
          <Button variant="subtle" onClick={() => onOpenChange(false)} disabled={saving}>
            Cancel
          </Button>
          <Button onClick={save} disabled={saving || !name || !baseUrl || (!existing && !apiKey)}>
            {saving ? "Saving..." : existing ? "Save" : "Create"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function ImportDialog({
  provider,
  open,
  onOpenChange,
  onImported,
}: {
  provider: CustomProviderRecord | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onImported: () => void;
}) {
  const [syncing, setSyncing] = useState(false);
  const [synced, setSynced] = useState<SyncedCustomProviderModel[]>([]);
  const [selected, setSelected] = useState<Record<string, ImportedModelDraft>>({});
  const [importing, setImporting] = useState(false);

  const sync = useCallback(async () => {
    if (!provider) return;
    setSyncing(true);
    try {
      const body = await api<{ models: SyncedCustomProviderModel[] }>(
        `/api/custom-providers/${provider.id}/sync-models`,
        { method: "POST" }
      );
      setSynced(body.models);
      if (body.models.length === 0) {
        toast("All models on this gateway are already imported.");
      }
      setSelected(
        Object.fromEntries(body.models.map((model) => [model.modelId, draftForModel(model)]))
      );
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Model sync failed");
    } finally {
      setSyncing(false);
    }
  }, [provider]);

  const chosen = useMemo(() => Object.values(selected), [selected]);

  const runImport = async () => {
    if (!provider || chosen.length === 0) return;
    setImporting(true);
    try {
      await api(`/api/custom-providers/${provider.id}/models`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ models: chosen }),
      });
      toast.success(`Imported ${chosen.length} model${chosen.length === 1 ? "" : "s"}`);
      onImported();
      onOpenChange(false);
      setSynced([]);
      setSelected({});
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Import failed");
    } finally {
      setImporting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <div className="mb-4">
          <DialogTitle>Import models — {provider?.name}</DialogTitle>
          <DialogDescription>
            Sync the gateway&apos;s model list, pick the models to import, and adjust each
            model&apos;s metadata. Already-imported models are hidden from the list; importing adds
            new models and leaves existing ones and their edits untouched.
          </DialogDescription>
        </div>
        <div className="flex items-center gap-2">
          <Button type="button" variant="subtle" size="sm" onClick={sync} disabled={syncing}>
            {syncing ? "Syncing..." : synced.length > 0 ? "Re-sync" : "Sync models"}
          </Button>
          {synced.length > 0 && (
            <span className="text-sm text-muted-foreground">
              {chosen.length} of {synced.length} selected
            </span>
          )}
        </div>
        {synced.length > 0 && (
          <div className="max-h-[50vh] space-y-3 overflow-y-auto">
            {synced.map((model) => {
              const draft = selected[model.modelId];
              const active = draft !== undefined;
              return (
                <div key={model.modelId} className="rounded border border-border p-3 space-y-2">
                  <label className="flex items-center gap-2 text-sm font-medium text-foreground">
                    <Checkbox
                      checked={active}
                      onCheckedChange={() =>
                        setSelected((current) => {
                          const next = { ...current };
                          if (active) delete next[model.modelId];
                          else next[model.modelId] = draftForModel(model);
                          return next;
                        })
                      }
                      aria-label={`Import ${model.modelId}`}
                    />
                    {model.modelId}
                    {model.catalog && (
                      <span className="text-xs font-normal text-muted-foreground">
                        metadata from OpenRouter
                      </span>
                    )}
                  </label>
                  {active && draft && (
                    <div className="grid gap-3 sm:grid-cols-2">
                      <div className="space-y-1">
                        <Label htmlFor={`display-${model.modelId}`}>Display name</Label>
                        <Input
                          id={`display-${model.modelId}`}
                          value={draft.displayName}
                          onChange={(event) =>
                            setSelected((current) => ({
                              ...current,
                              [model.modelId]: { ...draft, displayName: event.target.value },
                            }))
                          }
                        />
                      </div>
                      <div className="space-y-1">
                        <Label>Modalities</Label>
                        <MultiSelect
                          label={`Modalities for ${model.modelId}`}
                          options={CUSTOM_MODEL_MODALITIES}
                          selected={draft.modalities}
                          onChange={(next) =>
                            setSelected((current) => ({
                              ...current,
                              [model.modelId]: { ...draft, modalities: next },
                            }))
                          }
                        />
                      </div>
                      <div className="space-y-1">
                        <Label>Thinking efforts</Label>
                        <MultiSelect
                          label={`Thinking efforts for ${model.modelId}`}
                          options={CUSTOM_MODEL_REASONING_EFFORTS}
                          selected={draft.reasoningEfforts}
                          onChange={(next) =>
                            setSelected((current) => ({
                              ...current,
                              [model.modelId]: { ...draft, reasoningEfforts: next },
                            }))
                          }
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="space-y-1">
                          <Label htmlFor={`context-${model.modelId}`}>Context tokens</Label>
                          <Input
                            id={`context-${model.modelId}`}
                            type="number"
                            min={1}
                            value={draft.contextWindowTokens}
                            onChange={(event) =>
                              setSelected((current) => ({
                                ...current,
                                [model.modelId]: {
                                  ...draft,
                                  contextWindowTokens: Number(event.target.value) || 0,
                                },
                              }))
                            }
                          />
                        </div>
                        <div className="space-y-1">
                          <Label htmlFor={`output-${model.modelId}`}>Max output</Label>
                          <Input
                            id={`output-${model.modelId}`}
                            type="number"
                            min={1}
                            value={draft.maxOutputTokens}
                            onChange={(event) =>
                              setSelected((current) => ({
                                ...current,
                                [model.modelId]: {
                                  ...draft,
                                  maxOutputTokens: Number(event.target.value) || 0,
                                },
                              }))
                            }
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
        <div className="mt-4 flex flex-col gap-3 border-t border-border-muted pt-4 sm:flex-row sm:items-center sm:justify-between">
          <Button variant="subtle" onClick={() => onOpenChange(false)} disabled={importing}>
            Cancel
          </Button>
          <Button onClick={runImport} disabled={importing || chosen.length === 0}>
            {importing ? "Importing..." : `Import ${chosen.length || ""}`}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function CustomProvidersSettings() {
  const [providers, setProviders] = useState<CustomProviderRecord[] | null>(null);
  const [models, setModels] = useState<Record<string, CustomModelRecord[]>>({});
  const [editing, setEditing] = useState<CustomProviderRecord | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [importFor, setImportFor] = useState<CustomProviderRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Test state is keyed by provider ID and by full model ID (`cp?-…/model`),
  // which cannot collide.
  const [tests, setTests] = useState<Record<string, ConnectionTestState>>({});
  const [expandedModels, setExpandedModels] = useState<ReadonlySet<string>>(new Set());

  const load = useCallback(async () => {
    try {
      const body = await api<{ providers: CustomProviderRecord[] }>("/api/custom-providers");
      setProviders(body.providers);
      const perProvider = await Promise.all(
        body.providers.map(async (provider) => {
          const modelsBody = await api<{ models: CustomModelRecord[] }>(
            `/api/custom-providers/${provider.id}/models`
          );
          return [provider.id, modelsBody.models] as const;
        })
      );
      setModels(Object.fromEntries(perProvider));
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load custom providers");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const remove = async (provider: CustomProviderRecord) => {
    if (!window.confirm(`Delete provider "${provider.name}" and its imported models?`)) return;
    try {
      await api(`/api/custom-providers/${provider.id}`, { method: "DELETE" });
      toast.success("Provider deleted");
      await load();
    } catch (deleteError) {
      toast.error(deleteError instanceof Error ? deleteError.message : "Delete failed");
    }
  };

  const removeModel = async (provider: CustomProviderRecord, model: CustomModelRecord) => {
    if (!window.confirm(`Remove "${model.displayName}" from the imported models?`)) return;
    try {
      await api(
        `/api/custom-providers/${provider.id}/models/${encodeURIComponent(model.modelId)}`,
        { method: "DELETE" }
      );
      toast.success("Model removed");
      await load();
    } catch (deleteError) {
      toast.error(deleteError instanceof Error ? deleteError.message : "Delete failed");
    }
  };

  const toggleModel = async (provider: CustomProviderRecord, model: CustomModelRecord) => {
    try {
      await api(
        `/api/custom-providers/${provider.id}/models/${encodeURIComponent(model.modelId)}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: !model.enabled }),
        }
      );
      await load();
    } catch (toggleError) {
      toast.error(toggleError instanceof Error ? toggleError.message : "Update failed");
    }
  };

  const setStatus = async (provider: CustomProviderRecord, status: "active" | "disabled") => {
    try {
      await api(`/api/custom-providers/${provider.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status }),
      });
      await load();
    } catch (statusError) {
      toast.error(statusError instanceof Error ? statusError.message : "Update failed");
    }
  };

  /** Provider-level test lists models; a modelId adds a one-token generation. */
  const runConnectionTest = async (
    key: string,
    provider: CustomProviderRecord,
    modelId?: string
  ) => {
    setTests((current) => ({ ...current, [key]: { running: true, result: null, error: null } }));
    try {
      const result = await api<CustomProviderConnectionTestResult>(
        `/api/custom-providers/${provider.id}/test-connection`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(modelId ? { modelId } : {}),
        }
      );
      setTests((current) => ({ ...current, [key]: { running: false, result, error: null } }));
    } catch (testError) {
      setTests((current) => ({
        ...current,
        [key]: {
          running: false,
          result: null,
          error: testError instanceof Error ? testError.message : "Connection test failed",
        },
      }));
    }
  };

  const toggleExpanded = (modelId: string) => {
    setExpandedModels((current) => {
      const next = new Set(current);
      if (next.has(modelId)) next.delete(modelId);
      else next.add(modelId);
      return next;
    });
  };

  return (
    <div>
      <div className="flex items-start justify-between mb-6 gap-4">
        <div>
          <h2 className="text-xl font-semibold text-foreground mb-1">Custom Providers</h2>
          <p className="text-sm text-muted-foreground">
            Register Anthropic-, OpenAI-compatible, or OpenAI Responses gateways. Imported models
            appear in the model picker and run on both harnesses (Anthropic-protocol models only on
            Claude Agent; OpenAI-protocol models also on Codex).
          </p>
        </div>
        <Button
          size="sm"
          onClick={() => {
            setEditing(null);
            setDialogOpen(true);
          }}
        >
          Add provider
        </Button>
      </div>

      {error && (
        <p role="alert" className="text-sm text-destructive mb-4">
          {error}
        </p>
      )}
      {providers === null && !error && (
        <p className="text-sm text-muted-foreground">Loading custom providers...</p>
      )}
      {providers?.length === 0 && (
        <p className="text-sm text-muted-foreground">
          No custom providers configured. Add one to route sessions through your own gateway.
        </p>
      )}

      <div className="space-y-6">
        {(providers ?? []).map((provider) => {
          const providerModels = models[provider.id] ?? [];
          const providerTest = tests[provider.id] ?? IDLE_TEST;
          return (
            <div key={provider.id} className="rounded-lg border border-border p-4 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex min-w-0 items-start gap-2.5">
                  <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-md border border-border-muted text-foreground">
                    <CustomProviderIcon className="size-4" aria-hidden="true" />
                  </span>
                  <div className="min-w-0">
                    <h3 className="text-sm font-semibold text-foreground truncate">
                      {provider.name}{" "}
                      <span className="font-normal text-muted-foreground">
                        · {CUSTOM_PROVIDER_PROTOCOL_LABELS[provider.protocol]}
                      </span>
                    </h3>
                    <p className="text-xs text-muted-foreground truncate">
                      {provider.baseUrl} · {provider.providerKey}
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <label className="flex items-center gap-1.5 text-xs text-foreground">
                    <Switch
                      checked={provider.status === "active"}
                      onCheckedChange={(checked) =>
                        setStatus(provider, checked ? "active" : "disabled")
                      }
                      aria-label={`${provider.name} active`}
                    />
                    Active
                  </label>
                  <Button
                    variant="subtle"
                    size="xs"
                    onClick={() => runConnectionTest(provider.id, provider)}
                    disabled={providerTest.running}
                  >
                    {providerTest.running ? "Testing..." : "Test"}
                  </Button>
                  <Button variant="subtle" size="xs" onClick={() => setImportFor(provider)}>
                    Models
                  </Button>
                  <Button
                    variant="subtle"
                    size="xs"
                    onClick={() => {
                      setEditing(provider);
                      setDialogOpen(true);
                    }}
                  >
                    Edit
                  </Button>
                  <Button variant="subtle" size="xs" onClick={() => remove(provider)}>
                    Delete
                  </Button>
                </div>
              </div>
              <TestResultLine state={providerTest} />
              {providerModels.length > 0 && (
                <div className="space-y-1">
                  {providerModels.map((model) => {
                    const expanded = expandedModels.has(model.id);
                    const modelTest = tests[model.id] ?? IDLE_TEST;
                    return (
                      <div key={model.id} className="rounded border border-border-muted">
                        <div className="flex items-center gap-2 px-3 py-2">
                          <button
                            type="button"
                            onClick={() => toggleExpanded(model.id)}
                            aria-expanded={expanded}
                            className="flex min-w-0 flex-1 items-center gap-1.5 text-left"
                          >
                            <ChevronRightIcon
                              className={cn(
                                "size-3.5 shrink-0 text-muted-foreground transition-transform",
                                expanded && "rotate-90"
                              )}
                            />
                            <span className="text-sm text-foreground truncate">
                              {model.displayName}
                            </span>
                            <span className="text-xs text-muted-foreground truncate">
                              {model.modelId}
                            </span>
                          </button>
                          <Button
                            variant="subtle"
                            size="xs"
                            onClick={() => removeModel(provider, model)}
                            aria-label={`Delete ${model.displayName}`}
                          >
                            Delete
                          </Button>
                          <Switch
                            checked={model.enabled}
                            onCheckedChange={() => toggleModel(provider, model)}
                            aria-label={`${model.displayName} enabled`}
                          />
                        </div>
                        {expanded && (
                          <div className="space-y-2 border-t border-border-muted px-3 py-2">
                            <dl className="grid gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
                              <div className="flex min-w-0 gap-2">
                                <dt className="shrink-0 text-muted-foreground">Model ID</dt>
                                <dd className="truncate font-mono text-foreground">{model.id}</dd>
                              </div>
                              <div className="flex gap-2">
                                <dt className="shrink-0 text-muted-foreground">Modalities</dt>
                                <dd className="text-foreground">{model.modalities.join(", ")}</dd>
                              </div>
                              <div className="flex gap-2">
                                <dt className="shrink-0 text-muted-foreground">Context window</dt>
                                <dd className="text-foreground">
                                  {model.contextWindowTokens.toLocaleString()} tokens
                                </dd>
                              </div>
                              <div className="flex gap-2">
                                <dt className="shrink-0 text-muted-foreground">Max output</dt>
                                <dd className="text-foreground">
                                  {model.maxOutputTokens.toLocaleString()} tokens
                                </dd>
                              </div>
                              <div className="flex gap-2">
                                <dt className="shrink-0 text-muted-foreground">Thinking</dt>
                                <dd className="text-foreground">
                                  {model.reasoningEfforts.length > 0
                                    ? model.reasoningEfforts.join(", ")
                                    : "—"}
                                </dd>
                              </div>
                            </dl>
                            <div className="flex flex-wrap items-center gap-2">
                              <Button
                                variant="subtle"
                                size="xs"
                                onClick={() => runConnectionTest(model.id, provider, model.modelId)}
                                disabled={modelTest.running}
                              >
                                {modelTest.running ? "Testing..." : "Test model"}
                              </Button>
                              <TestResultLine state={modelTest} />
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {dialogOpen && (
        <ProviderDialog
          key={editing?.id ?? "new"}
          open={dialogOpen}
          onOpenChange={setDialogOpen}
          existing={editing}
          modelCount={editing ? (models[editing.id] ?? []).length : 0}
          onSaved={load}
        />
      )}
      <ImportDialog
        provider={importFor}
        open={importFor !== null}
        onOpenChange={(open) => {
          if (!open) setImportFor(null);
        }}
        onImported={load}
      />
    </div>
  );
}
