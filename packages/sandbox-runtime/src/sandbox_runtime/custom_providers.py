"""Custom model provider routing from the control-plane env manifest.

The control plane hands a sandbox one ``CUSTOM_MODEL_PROVIDERS`` JSON
manifest (every active provider's routing and model metadata) plus one
``CP_XXXXXXXX_API_KEY`` variable per provider carrying its decrypted key.
Keys never appear inside the manifest.

Three consumers:

- OpenCode server setup builds provider config blocks (``opencode_server``)
- The Claude harness resolves an Anthropic-protocol provider for a model ID
  and derives the child's ``ANTHROPIC_*`` credential environment
  (``harness.claude``)
- The CLI harnesses write vendor config files — Codex a ``[model_providers.*]``
  entry plus a model catalog in ``~/.codex``, Pi a ``models.json``, dsh a
  profile patch, ZCode a personal provider config (``harness.cli_vendors``)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

CUSTOM_PROVIDERS_ENV = "CUSTOM_MODEL_PROVIDERS"
_ANTHROPIC_PROTOCOL = "anthropic"
_OPENAI_COMPATIBLE_PROTOCOL = "openai_compatible"
_OPENAI_RESPONSES_PROTOCOL = "openai_responses"
_OPENAI_PROTOCOLS = (_OPENAI_COMPATIBLE_PROTOCOL, _OPENAI_RESPONSES_PROTOCOL)

# Env the Claude child receives in custom-provider mode, on top of the
# standard API-key family. ANTHROPIC_CUSTOM_HEADERS is the Claude CLI's
# native mechanism (comma-separated ``Name: value`` pairs).
CUSTOM_ANTHROPIC_HEADERS_ENV = "ANTHROPIC_CUSTOM_HEADERS"


@dataclass(frozen=True)
class CustomModelEntry:
    model_id: str
    display_name: str
    reasoning_efforts: tuple[str, ...]
    context_window_tokens: int
    max_output_tokens: int


@dataclass(frozen=True)
class CustomProvider:
    provider_key: str
    protocol: str
    base_url: str
    headers: tuple[tuple[str, str], ...]
    api_key: str
    api_key_env: str
    models: tuple[CustomModelEntry, ...]

    @property
    def is_anthropic_protocol(self) -> bool:
        return self.protocol == _ANTHROPIC_PROTOCOL

    @property
    def is_openai_protocol(self) -> bool:
        return self.protocol in _OPENAI_PROTOCOLS


def _parse_model(entry: Mapping[str, Any]) -> CustomModelEntry | None:
    model_id = entry.get("modelId")
    if not isinstance(model_id, str) or not model_id:
        return None
    efforts = entry.get("reasoningEfforts")
    context = entry.get("contextWindowTokens")
    output = entry.get("maxOutputTokens")
    if not isinstance(context, int) or context <= 0:
        return None
    if not isinstance(output, int) or output <= 0:
        return None
    display = entry.get("displayName")
    return CustomModelEntry(
        model_id=model_id,
        display_name=display if isinstance(display, str) and display else model_id,
        reasoning_efforts=tuple(
            effort
            for effort in (efforts if isinstance(efforts, list) else [])
            if isinstance(effort, str)
        ),
        context_window_tokens=context,
        max_output_tokens=output,
    )


def _parse_provider(entry: Mapping[str, Any], environ: Mapping[str, str]) -> CustomProvider | None:
    provider_key = entry.get("providerKey")
    protocol = entry.get("protocol")
    base_url = entry.get("baseUrl")
    api_key_env = entry.get("apiKeyEnv")
    if not isinstance(provider_key, str) or not provider_key:
        return None
    if protocol not in (_ANTHROPIC_PROTOCOL, *_OPENAI_PROTOCOLS):
        return None
    if not isinstance(base_url, str) or not base_url:
        return None
    api_key = environ.get(api_key_env) if isinstance(api_key_env, str) else None
    if not api_key or not isinstance(api_key_env, str):
        return None
    raw_headers = entry.get("headers")
    headers: tuple[tuple[str, str], ...] = tuple(
        (str(item.get("name")), str(item.get("value")))
        for item in (raw_headers if isinstance(raw_headers, list) else [])
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and isinstance(item.get("value"), str)
    )
    raw_models = entry.get("models")
    models = tuple(
        parsed
        for parsed in (
            _parse_model(item)
            for item in (raw_models if isinstance(raw_models, list) else [])
            if isinstance(item, dict)
        )
        if parsed is not None
    )
    if not models:
        return None
    return CustomProvider(
        provider_key=provider_key,
        protocol=str(protocol),
        base_url=base_url,
        headers=headers,
        api_key=api_key,
        api_key_env=api_key_env,
        models=models,
    )


def load_custom_providers(environ: Mapping[str, str] = os.environ) -> tuple[CustomProvider, ...]:
    """Parse the manifest, dropping providers whose key or models are unusable."""
    raw = environ.get(CUSTOM_PROVIDERS_ENV)
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    if not isinstance(parsed, list):
        return ()
    parsed_providers: list[CustomProvider] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        provider = _parse_provider(item, environ)
        if provider is not None:
            parsed_providers.append(provider)
    return tuple(parsed_providers)


def find_provider_for_model(
    model: str, providers: tuple[CustomProvider, ...]
) -> tuple[CustomProvider, CustomModelEntry] | None:
    """Resolve ``{provider_key}/{model_id}`` against the loaded providers."""
    slash = model.find("/")
    if slash <= 0:
        return None
    provider_key = model[:slash]
    model_id = model[slash + 1 :]
    for provider in providers:
        if provider.provider_key != provider_key:
            continue
        for entry in provider.models:
            if entry.model_id == model_id:
                return provider, entry
    return None


def opencode_provider_config(providers: tuple[CustomProvider, ...]) -> dict[str, dict[str, Any]]:
    """OpenCode ``provider`` config blocks for the loaded custom providers."""
    config: dict[str, dict[str, Any]] = {}
    for provider in providers:
        options: dict[str, Any] = {
            "baseURL": provider.base_url,
            "apiKey": provider.api_key,
        }
        if provider.headers:
            options["headers"] = dict(provider.headers)
        models: dict[str, dict[str, Any]] = {}
        for entry in provider.models:
            models[entry.model_id] = {
                "name": entry.display_name,
                "limit": {
                    "context": entry.context_window_tokens,
                    "output": entry.max_output_tokens,
                },
            }
        config[provider.provider_key] = {
            "npm": (
                "@ai-sdk/anthropic"
                if provider.is_anthropic_protocol
                else "@ai-sdk/openai-compatible"
            ),
            "name": provider.provider_key,
            "options": options,
            "models": models,
        }
    return config


def anthropic_root_base_url(base_url: str) -> str:
    """The base URL an Anthropic-protocol client wants: the API root.

    Anthropic clients (Claude Code, the SDK pi-ai embeds, hence both the Pi
    and dsh harnesses) append ``/v1/messages`` themselves, so a base URL
    registered with a trailing ``/v1`` — the SDK-style form gateways also
    document for their model-list endpoints — would be requested at
    ``/v1/v1/messages``. Strip the version segment so both registration
    styles reach the gateway.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")].rstrip("/")
    return root


def custom_anthropic_env(provider: CustomProvider) -> dict[str, str]:
    """The Anthropic credential env a Claude child needs for a custom gateway."""
    env = {
        "ANTHROPIC_BASE_URL": anthropic_root_base_url(provider.base_url),
        "ANTHROPIC_API_KEY": provider.api_key,
    }
    if provider.headers:
        env[CUSTOM_ANTHROPIC_HEADERS_ENV] = ", ".join(
            f"{name}: {value}" for name, value in provider.headers
        )
    return env


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


_MANAGED_CODEX_SECTION = re.compile(r"^\[model_providers\.(cp[ao]-[0-9a-f]{8})\]$")


def codex_model_provider_entries(providers: tuple[CustomProvider, ...]) -> str:
    """``[model_providers.*]`` TOML sections for the Responses-protocol providers.

    Codex appends the wire path itself (``/responses``) to ``base_url``, which
    gateways document WITH the version segment — the opposite convention of
    Claude Code — so the registered URL passes through unchanged. The key never
    rides the file: the CLI reads it from ``env_key``, a variable the
    supervisor's environment already carries.

    Chat-completions gateways are not registered: the CLI removed
    ``wire_api = "chat"`` support and refuses to load a config that still
    carries it, so those routes belong to the Pi and dsh harnesses.
    """
    sections: list[str] = []
    for provider in providers:
        if provider.protocol != _OPENAI_RESPONSES_PROTOCOL:
            continue
        sections.append(
            f"[model_providers.{provider.provider_key}]\n"
            f'name = "{_toml_escape(provider.provider_key)}"\n'
            f'base_url = "{_toml_escape(provider.base_url)}"\n'
            f'env_key = "{_toml_escape(provider.api_key_env)}"\n'
            f'wire_api = "responses"\n'
        )
    return "\n".join(sections)


def _strip_managed_codex_sections(existing: str) -> str:
    """Drop our ``[model_providers.cp[ao]-…]`` sections, keeping the rest.

    Custom provider keys are exclusively ours, so every section under one is
    managed: this both deduplicates re-runs and purges a ``wire_api = "chat"``
    section an earlier run wrote, which the CLI now refuses to load.
    """

    def is_section_header(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("[") and stripped.endswith("]")

    lines = existing.splitlines(keepends=True)
    kept: list[str] = []
    dropping = False
    for line in lines:
        if _MANAGED_CODEX_SECTION.match(line.strip()):
            dropping = True
            continue
        if dropping and is_section_header(line):
            dropping = False
        if not dropping:
            kept.append(line)
    # Collapse the blank lines a dropped section leaves behind.
    text = "".join(kept)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


def write_codex_model_providers(config_path: Path, providers: tuple[CustomProvider, ...]) -> bool:
    """Rewrite the custom-provider sections in the Codex CLI's ``config.toml``.

    Idempotent: our sections are replaced wholesale with the current set (a
    resumed session re-opening the harness, or purging a section a provider
    lost), while unknown pre-existing content is preserved verbatim. Returns
    whether anything was written.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    existing = config_path.read_text() if config_path.exists() else ""
    body = _strip_managed_codex_sections(existing).rstrip("\n")
    fresh = [
        section
        for section in codex_model_provider_entries(providers).split("\n\n")
        if section.strip()
    ]
    merged = body
    for section in fresh:
        if merged:
            merged += "\n\n"
        merged += section.rstrip("\n")
    if merged:
        merged += "\n"
    if merged == existing:
        return False
    config_path.write_text(merged)
    return True


# --- Codex model catalog -------------------------------------------------------

# The CLI resolves a model slug against its bundled catalog and, for slugs it
# does not know (every gateway model), degrades to fallback metadata — a
# hardcoded context window and a per-turn warning — unless a catalog supplies
# the entry. Catalog entries require instruction text, and the generic prompt
# the fallback itself uses is compiled into the CLI binary, so the same text
# is vendored here. Copied verbatim from codex-rs `models-manager/prompt.md`
# at the version sandbox-images' toolchain.json pins; refresh it alongside a
# codex bump.
_CODEX_BASE_PROMPT = Path(__file__).with_name("codex_base_prompt.md").read_text()

# Menu copy Codex shows next to each effort level; descriptions are ours.
_CODEX_EFFORT_DESCRIPTIONS = {
    "none": "Disable reasoning for the fastest responses",
    "low": "Fast responses with lighter reasoning",
    "medium": "Balanced reasoning depth",
    "high": "Deeper reasoning for complex problems",
    "xhigh": "Extra deep reasoning for hard problems",
    "max": "Maximum reasoning depth for the hardest problems",
}


def codex_model_catalog_path(home: Path | None = None) -> Path:
    """Location of the generated catalog inside the Codex home directory."""
    return (home if home is not None else Path.home()) / ".codex" / "custom-models.json"


def _codex_catalog_entry(model: CustomModelEntry) -> dict[str, Any]:
    """One ``ModelInfo`` record, mirroring the CLI's fallback defaults.

    Every field the registry knows (slug, context window, effort levels) is
    supplied; the rest matches what ``model_info_from_slug`` would build for
    an unknown slug, so routing a catalog hit changes the metadata, not the
    agent behavior.
    """
    efforts = list(model.reasoning_efforts)
    default = "high" if "high" in efforts else next((e for e in efforts if e != "none"), None)
    entry: dict[str, Any] = {
        "slug": model.model_id,
        "display_name": model.display_name,
        "supported_reasoning_levels": [
            {"effort": effort, "description": _CODEX_EFFORT_DESCRIPTIONS.get(effort, effort)}
            for effort in efforts
        ],
        "shell_type": "unified_exec",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 99,
        "support_verbosity": False,
        "truncation_policy": {"mode": "tokens", "limit": 10_000},
        "experimental_supported_tools": [],
        "context_window": model.context_window_tokens,
        "max_context_window": model.context_window_tokens,
        "model_messages": {"instructions_template": _CODEX_BASE_PROMPT},
    }
    if default is not None:
        entry["default_reasoning_level"] = default
    return entry


def codex_model_catalog(providers: tuple[CustomProvider, ...]) -> dict[str, Any] | None:
    """A ``ModelsResponse`` catalog covering the Responses-protocol models.

    Only those models route to Codex, so only they need catalog entries. The
    slug is the bare model id — the form the CLI is invoked with, since the
    provider rides ``-c model_provider`` instead.
    """
    models = [
        _codex_catalog_entry(model)
        for provider in providers
        if provider.protocol == _OPENAI_RESPONSES_PROTOCOL
        for model in provider.models
    ]
    return {"models": models} if models else None


def write_codex_model_catalog(catalog_path: Path, providers: tuple[CustomProvider, ...]) -> bool:
    """Write the generated catalog, replacing any earlier generated one.

    The file is ours alone (the CLI never writes it), so it regenerates
    wholesale. It takes effect only through the per-turn
    ``-c model_catalog_json`` override the harness passes while a custom
    provider is routed — official-model sessions keep the CLI's own catalog.
    """
    catalog = codex_model_catalog(providers)
    if catalog is None:
        return False
    body = json.dumps(catalog, indent=2) + "\n"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    if catalog_path.exists() and catalog_path.read_text() == body:
        return False
    catalog_path.write_text(body)
    return True


# --- Pi (models.json) -------------------------------------------------------


_PI_API_BY_PROTOCOL = {
    "anthropic": "anthropic-messages",
    "openai_compatible": "openai-completions",
    "openai_responses": "openai-responses",
}


def pi_models_document(providers: tuple[CustomProvider, ...]) -> dict[str, Any]:
    """The ``~/.pi/agent/models.json`` document registering every provider.

    Pi (via pi-ai) appends the wire path itself — the Anthropic implementation
    joins ``/v1/messages``, the OpenAI ones ``/chat/completions``/``/responses``
    — so Anthropic gateways are written at their API root (any registered
    version segment stripped) and OpenAI gateways keep it. The key interpolates
    from the provider's env var, which the supervisor's environment already
    carries.
    """
    document: dict[str, Any] = {"providers": {}}
    for provider in providers:
        api = _PI_API_BY_PROTOCOL.get(provider.protocol)
        if api is None:
            continue
        base_url = (
            anthropic_root_base_url(provider.base_url)
            if provider.is_anthropic_protocol
            else provider.base_url
        )
        document["providers"][provider.provider_key] = {
            "baseUrl": base_url,
            "api": api,
            "apiKey": f"${provider.api_key_env}",
            "models": [{"id": model.model_id} for model in provider.models],
        }
    return document


def write_pi_models_json(config_path: Path, providers: tuple[CustomProvider, ...]) -> bool:
    """Merge the custom providers into the Pi CLI's ``models.json``.

    Idempotent: provider keys are upserted, unknown pre-existing providers
    are preserved, and Pi reloads the file when a model is selected.
    """
    document = pi_models_document(providers)
    if not document["providers"]:
        return False
    config_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if config_path.exists():
        try:
            parsed = json.loads(config_path.read_text())
            if isinstance(parsed, dict):
                existing = parsed
        except ValueError:
            existing = {}
    merged_providers = existing.get("providers")
    providers_out = dict(merged_providers) if isinstance(merged_providers, dict) else {}
    changed = any(providers_out.get(key) != value for key, value in document["providers"].items())
    providers_out.update(document["providers"])
    existing["providers"] = providers_out
    if not changed:
        return False
    config_path.write_text(json.dumps(existing, indent=2) + "\n")
    return True


# --- dsh (cordis profile patch) ----------------------------------------------


_DSH_API_BY_PROTOCOL = _PI_API_BY_PROTOCOL


def _yaml_scalar(value: str) -> str:
    return json.dumps(value)


def dsh_profile_patch_entries(providers: tuple[CustomProvider, ...]) -> list[str]:
    """The ``cordis.patch.yml`` entry declaring every provider's route.

    The ``llm-pi-ai`` service's config is a dict keyed by provider route, so
    one patch entry carries all of them: endpoint, protocol, the credential
    ref (the provider's env var name, which the credential seam resolves from
    the environment), and the model catalog with the sizes the registry
    imported. Anthropic gateways are written at their API root — pi-ai's
    Anthropic client joins ``/v1/messages`` itself — while OpenAI gateways
    keep the registered version segment.
    """
    if not providers:
        return []
    lines = ["- id: llm-pi-ai", "  config:", "    providers:"]
    for provider in providers:
        api = _DSH_API_BY_PROTOCOL.get(provider.protocol)
        if api is None:
            continue
        base_url = (
            anthropic_root_base_url(provider.base_url)
            if provider.is_anthropic_protocol
            else provider.base_url
        )
        lines.append(f"      {_yaml_scalar(provider.provider_key)}:")
        lines.append(f"        displayName: {_yaml_scalar(provider.provider_key)}")
        lines.append(f"        api: {api}")
        lines.append(f"        baseURL: {_yaml_scalar(base_url)}")
        lines.append(f"        apiKeyEnv: {_yaml_scalar(provider.api_key_env)}")
        if provider.models:
            lines.append("        models:")
            for model in provider.models:
                lines.append(f"          - id: {_yaml_scalar(model.model_id)}")
                lines.append(f"            name: {_yaml_scalar(model.display_name)}")
                lines.append(f"            contextWindow: {int(model.context_window_tokens)}")
                lines.append(f"            maxTokens: {int(model.max_output_tokens)}")
                lines.append("            input: [text]")
                # dsh's schema is a map from thinking level to the wire value
                # dispatch should send, not a list; the identity mapping sends
                # the level name itself, which every supported protocol accepts.
                efforts = {effort: effort for effort in model.reasoning_efforts if effort != "none"}
                if efforts:
                    rendered = ", ".join(
                        f"{_yaml_scalar(level)}: {_yaml_scalar(wire)}"
                        for level, wire in efforts.items()
                    )
                    lines.append(f"            reasoningEfforts: {{{rendered}}}")
    return lines if len(lines) > 4 else []


def write_dsh_profile_patch(patch_path: Path, providers: tuple[CustomProvider, ...]) -> bool:
    """Rewrite the headless profile's user patch layer with the providers.

    The patch file belongs to the harness (a fresh sandbox carries only the
    shipped empty layer), so it is regenerated wholesale on every prepare —
    unlike the Codex and Pi writers, which merge into files a login flow may
    also write.
    """
    entries = dsh_profile_patch_entries(providers)
    if not entries:
        return False
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(entries) + "\n"
    if patch_path.exists() and patch_path.read_text() == body:
        return False
    patch_path.write_text(body)
    return True


def dsh_model_selection_patch(
    provider_key: str, model_id: str, reasoning_effort: str | None
) -> str:
    """The per-turn ``--patch`` overlay selecting the model on ``dsh``.

    ``agent-default-model`` is the headless profile's only model selector, so
    each turn's spawn overlays it with the routed provider and model. The
    effort ids are pi-ai's, where the registry's ``none`` maps to ``off``.
    """
    effort = "off" if reasoning_effort == "none" else reasoning_effort
    lines = [
        "- id: agent-default-model",
        "  config:",
        f"    provider: {_yaml_scalar(provider_key)}",
        f"    model: {_yaml_scalar(model_id)}",
    ]
    if effort:
        lines.append(f"    reasoningEffort: {_yaml_scalar(effort)}")
    return "\n".join(lines) + "\n"


# --- zcode (personal provider config) -----------------------------------------

_ZCODE_API_BY_PROTOCOL = {
    "anthropic": "anthropic-messages",
    "openai_compatible": "openai-chat-completions",
    "openai_responses": "openai-responses",
}

# The levels the CLI's own fallback model rule offers; a model without
# declared efforts selects from these instead of the registry's list.
_ZCODE_DEFAULT_REASONING_LEVELS = ("disabled", "enabled")


def zcode_provider_config_path(home: Path | None = None) -> Path:
    """Location of the generated personal provider config file."""
    return (home if home is not None else Path.home()) / ".zcode" / "v2" / "provider_config.json"


def _zcode_reasoning_levels(model: CustomModelEntry) -> list[str] | None:
    """The zcode level names a model's registry efforts become, if any.

    The registry's ``none`` marks a gateway that accepts disabling reasoning —
    zcode's own name for that state is ``disabled`` — while every other level
    passes through by name: the CLI's per-protocol wire maps forward unknown
    level names as the wire effort value.
    """
    levels = [effort for effort in model.reasoning_efforts if effort != "none"]
    return ["disabled", *levels] if levels else None


def zcode_provider_config_document(providers: tuple[CustomProvider, ...]) -> dict[str, Any]:
    """The ``~/.zcode/v2/provider_config.json`` document registering every provider.

    ZCode has no environment seam for endpoints or credentials — the api-key
    rides the file as a literal, the one place the CLI reads it from — and its
    base URL convention follows the client: Anthropic gateways at their API
    root (the client appends ``/v1/messages``), OpenAI gateways with whatever
    version segment they registered. Model sizes ride sparse "smart" rules so
    the CLI's bundled per-protocol wire mappings keep applying underneath.
    """
    provider_rules: list[dict[str, Any]] = []
    model_rules: list[dict[str, Any]] = []
    for provider in providers:
        api_type = _ZCODE_API_BY_PROTOCOL.get(provider.protocol)
        if api_type is None:
            continue
        api: dict[str, Any] = {
            "type": api_type,
            "baseUrl": (
                anthropic_root_base_url(provider.base_url)
                if provider.is_anthropic_protocol
                else provider.base_url
            ),
        }
        if provider.headers:
            api["headers"] = dict(provider.headers)
        provider_rules.append(
            {
                "providerId": provider.provider_key,
                "providerName": provider.provider_key,
                "enabled": True,
                "config": {
                    "group": "standard-personal",
                    "access": {"type": "api-key", "apiKey": provider.api_key},
                    "api": api,
                    "personalModelIds": [model.model_id for model in provider.models],
                    "visibility": "visible",
                },
            }
        )
        for model in provider.models:
            option_specs: dict[str, Any] = {"maxOutputTokens": {"max": model.max_output_tokens}}
            levels = _zcode_reasoning_levels(model)
            if levels is not None:
                option_specs["reasoningLevel"] = {"values": levels}
            model_rules.append(
                {
                    "providerId": provider.provider_key,
                    "modelId": model.model_id,
                    "config": {
                        "properties": {"contextWindow": model.context_window_tokens},
                        "optionSpecs": option_specs,
                    },
                }
            )
    return {
        "schemaVersion": 1,
        "config": {
            "providerConfigRules": {"providerRules": provider_rules},
            "modelConfigRules": {
                "providerModelRules": model_rules,
                "manualProviderModelRules": [],
            },
        },
    }


def write_zcode_provider_config(config_path: Path, providers: tuple[CustomProvider, ...]) -> bool:
    """Write the personal provider config, replacing any earlier generated one.

    The file belongs to the harness — the CLI never writes it, a fresh sandbox
    ships without one — so it regenerates wholesale on every prepare, unlike
    the Codex and Pi writers, which merge into files a login flow may touch.
    """
    document = zcode_provider_config_document(providers)
    if not document["config"]["providerConfigRules"]["providerRules"]:
        return False
    body = json.dumps(document, indent=2) + "\n"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and config_path.read_text() == body:
        return False
    config_path.write_text(body)
    return True


def zcode_model_selection(
    provider: CustomProvider, model: CustomModelEntry, reasoning_effort: str | None
) -> dict[str, Any]:
    """The ``defaultModelSelection`` entry routing one turn's model.

    The CLI reads its model from this field, so each turn rewrites it before
    the spawn. The level must be one the model declares: the registry's
    ``none`` maps to zcode's ``disabled``, an explicit effort passes through,
    and an unspecified one takes the highest declared level — the same choice
    the CLI itself makes for a fresh selection.
    """
    if reasoning_effort == "none":
        level = "disabled"
    elif reasoning_effort:
        level = reasoning_effort
    else:
        level = (_zcode_reasoning_levels(model) or list(_ZCODE_DEFAULT_REASONING_LEVELS))[-1]
    return {
        "providerId": provider.provider_key,
        "modelId": model.model_id,
        "options": {"reasoningLevel": level},
    }


def write_zcode_model_selection(
    config_path: Path,
    providers: tuple[CustomProvider, ...],
    model: str,
    reasoning_effort: str | None,
) -> bool:
    """Point the config file's ``defaultModelSelection`` at the routed model.

    Reads the document the prepare step wrote and swaps only the selection, so
    a missing or unreadable file regenerates from the manifest rather than
    sending the turn at the CLI's own default model.
    """
    resolved = find_provider_for_model(model, providers)
    if resolved is None:
        return False
    selection = zcode_model_selection(*resolved, reasoning_effort)
    document: dict[str, Any] | None = None
    if config_path.exists():
        try:
            parsed = json.loads(config_path.read_text())
            document = parsed if isinstance(parsed, dict) else None
        except ValueError:
            document = None
    if document is None:
        document = zcode_provider_config_document(providers)
    config = document.get("config")
    if not isinstance(config, dict):
        config = {}
        document["config"] = config
    if config.get("defaultModelSelection") == selection:
        return False
    config["defaultModelSelection"] = selection
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(document, indent=2) + "\n")
    return True
