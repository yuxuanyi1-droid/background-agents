"""Custom provider manifest parsing and routing."""

from __future__ import annotations

import json

import pytest

from sandbox_runtime.custom_providers import (
    CUSTOM_PROVIDERS_ENV,
    custom_anthropic_env,
    find_provider_for_model,
    load_custom_providers,
    opencode_provider_config,
)


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
