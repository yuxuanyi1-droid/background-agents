"""Custom model provider routing from the control-plane env manifest.

The control plane hands a sandbox one ``CUSTOM_MODEL_PROVIDERS`` JSON
manifest (every active provider's routing and model metadata) plus one
``CP_XXXXXXXX_API_KEY`` variable per provider carrying its decrypted key.
Keys never appear inside the manifest.

Two consumers:

- OpenCode server setup builds provider config blocks (``opencode_server``)
- The Claude harness resolves an Anthropic-protocol provider for a model ID
  and derives the child's ``ANTHROPIC_*`` credential environment
  (``harness.claude``)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

CUSTOM_PROVIDERS_ENV = "CUSTOM_MODEL_PROVIDERS"
_ANTHROPIC_PROTOCOL = "anthropic"
_OPENAI_COMPATIBLE_PROTOCOL = "openai_compatible"

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
    models: tuple[CustomModelEntry, ...]

    @property
    def is_anthropic_protocol(self) -> bool:
        return self.protocol == _ANTHROPIC_PROTOCOL


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
    if protocol not in (_ANTHROPIC_PROTOCOL, _OPENAI_COMPATIBLE_PROTOCOL):
        return None
    if not isinstance(base_url, str) or not base_url:
        return None
    api_key = environ.get(api_key_env) if isinstance(api_key_env, str) else None
    if not api_key:
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


def custom_anthropic_env(provider: CustomProvider) -> dict[str, str]:
    """The Anthropic credential env a Claude child needs for a custom gateway."""
    env = {
        "ANTHROPIC_BASE_URL": provider.base_url,
        "ANTHROPIC_API_KEY": provider.api_key,
    }
    if provider.headers:
        env[CUSTOM_ANTHROPIC_HEADERS_ENV] = ", ".join(
            f"{name}: {value}" for name, value in provider.headers
        )
    return env
