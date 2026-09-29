"""Custom provider manifest parsing and routing."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from sandbox_runtime.custom_providers import (
    CUSTOM_PROVIDERS_ENV,
    codex_model_provider_entries,
    codex_wire_api,
    custom_anthropic_env,
    find_provider_for_model,
    load_custom_providers,
    opencode_provider_config,
    write_codex_model_providers,
)

if TYPE_CHECKING:
    from pathlib import Path


def manifest_env(manifest: dict, api_key: str = "sk-gateway") -> dict[str, str]:
    return {
        CUSTOM_PROVIDERS_ENV: json.dumps([manifest]),
        manifest.get("apiKeyEnv", "CP_00112233_API_KEY"): api_key,
    }


def anthropic_manifest() -> dict:
    return {
        "id": "0011223344556677889900aabbccddee",
        "providerKey": "cpa-00112233",
        "protocol": "anthropic",
        "baseUrl": "https://gateway.example/api/anthropic",
        "headers": [{"name": "X-Org", "value": "acme"}],
        "apiKeyEnv": "CP_00112233_API_KEY",
        "models": [
            {
                "modelId": "glm-4.7",
                "displayName": "GLM 4.7",
                "reasoningEfforts": ["high"],
                "contextWindowTokens": 200_000,
                "maxOutputTokens": 32_768,
            }
        ],
    }


def openai_manifest() -> dict:
    return {
        "id": "99887766554433221100ffeeddccbbaa",
        "providerKey": "cpo-99887766",
        "protocol": "openai_compatible",
        "baseUrl": "https://gateway.example/v1",
        "headers": [],
        "apiKeyEnv": "CP_99887766_API_KEY",
        "models": [
            {
                "modelId": "deepseek-v4-pro",
                "displayName": "DeepSeek V4 Pro",
                "reasoningEfforts": [],
                "contextWindowTokens": 128_000,
                "maxOutputTokens": 16_384,
            }
        ],
    }


def responses_manifest() -> dict:
    return {
        "id": "554433221100ffeeddccbbaa99887766",
        "providerKey": "cpo-55443322",
        "protocol": "openai_responses",
        "baseUrl": "https://responses-gateway.example/v1",
        "headers": [],
        "apiKeyEnv": "CP_55443322_API_KEY",
        "models": [
            {
                "modelId": "gpt-x",
                "displayName": "GPT X",
                "reasoningEfforts": ["high"],
                "contextWindowTokens": 400_000,
                "maxOutputTokens": 65_536,
            }
        ],
    }


def test_missing_or_malformed_manifest_yields_no_providers():
    assert load_custom_providers({}) == ()
    assert load_custom_providers({CUSTOM_PROVIDERS_ENV: "not json"}) == ()
    assert load_custom_providers({CUSTOM_PROVIDERS_ENV: json.dumps({"no": "list"})}) == ()


def test_provider_without_key_is_dropped():
    providers = load_custom_providers({CUSTOM_PROVIDERS_ENV: json.dumps([anthropic_manifest()])})
    assert providers == ()


def test_anthropic_provider_parses_and_routes():
    providers = load_custom_providers(manifest_env(anthropic_manifest()))
    assert len(providers) == 1
    provider = providers[0]
    assert provider.is_anthropic_protocol
    assert provider.models[0].model_id == "glm-4.7"

    resolved = find_provider_for_model("cpa-00112233/glm-4.7", providers)
    assert resolved is not None and resolved[0] is provider

    assert find_provider_for_model("cpa-00112233/missing", providers) is None
    assert find_provider_for_model("cpo-00112233/glm-4.7", providers) is None


def test_opencode_config_blocks_per_protocol():
    providers = load_custom_providers(
        {
            CUSTOM_PROVIDERS_ENV: json.dumps([anthropic_manifest(), openai_manifest()]),
            "CP_00112233_API_KEY": "sk-a",
            "CP_99887766_API_KEY": "sk-b",
        }
    )
    config = opencode_provider_config(providers)
    assert config["cpa-00112233"]["npm"] == "@ai-sdk/anthropic"
    assert config["cpo-99887766"]["npm"] == "@ai-sdk/openai-compatible"
    assert config["cpa-00112233"]["options"]["baseURL"] == "https://gateway.example/api/anthropic"
    assert config["cpa-00112233"]["options"]["headers"] == {"X-Org": "acme"}
    assert config["cpa-00112233"]["models"]["glm-4.7"]["limit"] == {
        "context": 200_000,
        "output": 32_768,
    }


def test_custom_anthropic_env_carries_gateway_credential():
    providers = load_custom_providers(manifest_env(anthropic_manifest()))
    env = custom_anthropic_env(providers[0])
    assert env["ANTHROPIC_BASE_URL"] == "https://gateway.example/api/anthropic"
    assert env["ANTHROPIC_API_KEY"] == "sk-gateway"
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "X-Org: acme"


@pytest.mark.parametrize(
    "registered,expected",
    [
        ("https://gateway.example/api/anthropic/", "https://gateway.example/api/anthropic"),
        ("https://gateway.example/api/anthropic/v1", "https://gateway.example/api/anthropic"),
        ("https://gateway.example/api/anthropic/v1/", "https://gateway.example/api/anthropic"),
        ("https://gateway.example/v1", "https://gateway.example"),
    ],
)
def test_custom_anthropic_env_strips_version_segment(registered: str, expected: str):
    manifest = anthropic_manifest()
    manifest["baseUrl"] = registered
    providers = load_custom_providers(manifest_env(manifest))
    assert custom_anthropic_env(providers[0])["ANTHROPIC_BASE_URL"] == expected


def test_responses_protocol_parses_and_routes():
    providers = load_custom_providers(manifest_env(responses_manifest()))
    assert len(providers) == 1
    provider = providers[0]
    assert provider.is_openai_protocol
    assert not provider.is_anthropic_protocol
    assert codex_wire_api(provider) == "responses"
    resolved = find_provider_for_model("cpo-55443322/gpt-x", providers)
    assert resolved is not None and resolved[0] is provider


def test_codex_wire_api_distinguishes_the_openai_protocols():
    providers = load_custom_providers(
        {
            CUSTOM_PROVIDERS_ENV: json.dumps([responses_manifest(), openai_manifest()]),
            "CP_55443322_API_KEY": "sk-r",
            "CP_99887766_API_KEY": "sk-c",
        }
    )
    by_key = {provider.provider_key: provider for provider in providers}
    assert codex_wire_api(by_key["cpo-55443322"]) == "responses"
    assert codex_wire_api(by_key["cpo-99887766"]) == "chat"
    with pytest.raises(ValueError):
        codex_wire_api(load_custom_providers(manifest_env(anthropic_manifest()))[0])


def test_codex_entries_cover_openai_protocols_only():
    providers = load_custom_providers(
        {
            CUSTOM_PROVIDERS_ENV: json.dumps(
                [responses_manifest(), openai_manifest(), anthropic_manifest()]
            ),
            "CP_55443322_API_KEY": "sk-r",
            "CP_99887766_API_KEY": "sk-c",
            "CP_00112233_API_KEY": "sk-a",
        }
    )
    entries = codex_model_provider_entries(providers)
    assert "[model_providers.cpo-55443322]" in entries
    assert "[model_providers.cpo-99887766]" in entries
    assert "cpa-00112233" not in entries
    assert 'wire_api = "responses"' in entries
    assert 'wire_api = "chat"' in entries
    assert "sk-r" not in entries and "sk-c" not in entries


def test_write_codex_model_providers_is_idempotent_and_preserves_content(tmp_path: Path):
    config = tmp_path / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text('model = "gpt-5"\n\n[model_providers.existing]\nname = "existing"\n')
    providers = load_custom_providers(
        {
            CUSTOM_PROVIDERS_ENV: json.dumps([responses_manifest()]),
            "CP_55443322_API_KEY": "sk-r",
        }
    )

    assert write_codex_model_providers(config, providers) is True
    merged = config.read_text()
    assert 'model = "gpt-5"' in merged
    assert "[model_providers.existing]" in merged
    assert "[model_providers.cpo-55443322]" in merged
    assert 'base_url = "https://responses-gateway.example/v1"' in merged
    assert 'env_key = "CP_55443322_API_KEY"' in merged

    assert write_codex_model_providers(config, providers) is False
    assert config.read_text() == merged


def test_write_codex_model_providers_without_openai_providers_writes_nothing(tmp_path: Path):
    config = tmp_path / "config.toml"
    providers = load_custom_providers(manifest_env(anthropic_manifest()))
    assert write_codex_model_providers(config, providers) is False
    assert not config.exists()
