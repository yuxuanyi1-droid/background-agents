"use client";

import { useCallback, useMemo, useState } from "react";
import { toast } from "sonner";
import {
  CUSTOM_MODEL_MODALITIES,
  CUSTOM_MODEL_REASONING_EFFORTS,
  type CustomModelRecord,
  type CustomProviderHeader,
  type CustomProviderProtocol,
  type CustomProviderRecord,
  type SyncedCustomProviderModel,
} from "@open-inspect/shared/types/custom-providers";
import { browserApiFetch } from "@/lib/browser-api-fetch";
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

const PROTOCOL_LABELS: Record<CustomProviderProtocol, string> = {
  anthropic: "Anthropic Messages API",
  openai_compatible: "OpenAI-compatible",
};

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
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSaved: () => void;
  existing: CustomProviderRecord | null;
}) {
  const [name, setName] = useState(existing?.name ?? "");
  const [protocol, setProtocol] = useState<CustomProviderProtocol>(
    existing?.protocol ?? "anthropic"
  );
  const [baseUrl, setBaseUrl] = useState(existing?.baseUrl ?? "");
  const [apiKey, setApiKey] = useState("");
  const [headers, setHeaders] = useState<CustomProviderHeader[]>(existing?.headers ?? []);
  const [saving, setSaving] = useState(false);

  const save = async () => {
    setSaving(true);
    try {
      if (existing) {
        await api(`/api/custom-providers/${existing.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name,
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
          {!existing && (
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
                  {(Object.keys(PROTOCOL_LABELS) as CustomProviderProtocol[]).map((value) => (
                    <SelectItem key={value} value={value}>
                      {PROTOCOL_LABELS[value]}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
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

  if (providers === null && error === null) {
    void load();
  }

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

  return (
    <div>
      <div className="flex items-start justify-between mb-6 gap-4">
        <div>
          <h2 className="text-xl font-semibold text-foreground mb-1">Custom Providers</h2>
          <p className="text-sm text-muted-foreground">
            Register Anthropic- or OpenAI-compatible gateways. Imported models appear in the model
            picker and run on both harnesses (Anthropic-protocol models only on Claude Agent).
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
        {(providers ?? []).map((provider) => (
          <div key={provider.id} className="rounded-lg border border-border p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="text-sm font-semibold text-foreground">
                  {provider.name}{" "}
                  <span className="font-normal text-muted-foreground">
                    · {PROTOCOL_LABELS[provider.protocol]} · {provider.providerKey}
                  </span>
                </h3>
                <p className="text-xs text-muted-foreground">{provider.baseUrl}</p>
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
            {(models[provider.id] ?? []).length > 0 && (
              <div className="space-y-1">
                {(models[provider.id] ?? []).map((model) => (
                  <div
                    key={model.modelId}
                    className="flex items-center justify-between rounded border border-border-muted px-3 py-2"
                  >
                    <div className="min-w-0">
                      <span className="text-sm text-foreground">{model.displayName}</span>
                      <span className="text-xs text-muted-foreground ml-2 truncate">
                        {model.id} · {model.contextWindowTokens.toLocaleString()} ctx ·{" "}
                        {model.maxOutputTokens.toLocaleString()} out
                        {model.reasoningEfforts.length > 0 &&
                          ` · ${model.reasoningEfforts.join(", ")}`}
                      </span>
                    </div>
                    <Switch
                      checked={model.enabled}
                      onCheckedChange={() => toggleModel(provider, model)}
                      aria-label={`${model.displayName} enabled`}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {dialogOpen && (
        <ProviderDialog
          key={editing?.id ?? "new"}
          open={dialogOpen}
          onOpenChange={setDialogOpen}
          existing={editing}
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
