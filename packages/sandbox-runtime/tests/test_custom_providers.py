"""Custom provider manifest parsing and routing."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from sandbox_runtime.custom_providers import (
    CUSTOM_PROVIDERS_ENV,
    codex_model_provider_entries,
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
    assert provider.protocol == "openai_responses"
    resolved = find_provider_for_model("cpo-55443322/gpt-x", providers)
    assert resolved is not None and resolved[0] is provider


def test_codex_entries_cover_responses_protocol_only():
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
    # Chat-completions gateways are not registered: the CLI removed
    # wire_api = "chat" and refuses to load a config carrying it.
    assert "cpo-99887766" not in entries
    assert "cpa-00112233" not in entries
    assert 'wire_api = "responses"' in entries
    assert 'wire_api = "chat"' not in entries
    assert "sk-r" not in entries


def test_write_codex_model_providers_is_idempotent_and_preserves_content(tmp_path: Path):
    config = tmp_path / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        'model = "gpt-5"\n\n'
        "[model_providers.existing]\n"
        'name = "existing"\n\n'
        "[model_providers.cpo-99887766]\n"
        'name = "cpo-99887766"\n'
        'base_url = "https://stale.example/v1"\n'
        'env_key = "CP_99887766_API_KEY"\n'
        'wire_api = "chat"\n'
    )
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
    # The stale chat section an earlier run wrote is purged, not preserved:
    # the CLI refuses to load any config still carrying wire_api = "chat".
    assert "cpo-99887766" not in merged
    assert 'wire_api = "chat"' not in merged
    assert 'base_url = "https://responses-gateway.example/v1"' in merged
    assert 'env_key = "CP_55443322_API_KEY"' in merged

    assert write_codex_model_providers(config, providers) is False
    assert config.read_text() == merged


def test_write_codex_model_providers_without_openai_providers_writes_nothing(tmp_path: Path):
    config = tmp_path / "config.toml"
    providers = load_custom_providers(manifest_env(anthropic_manifest()))
    assert write_codex_model_providers(config, providers) is False
    assert not config.exists()


def test_codex_catalog_covers_responses_models_only():
    from sandbox_runtime.custom_providers import codex_model_catalog

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
    catalog = codex_model_catalog(providers)
    assert [model["slug"] for model in catalog["models"]] == ["gpt-x"]
    entry = catalog["models"][0]
    assert entry["display_name"] == "GPT X"
    assert entry["context_window"] == 400_000
    assert entry["max_context_window"] == 400_000
    # The CLI refuses to load an entry without instruction text.
    assert entry["model_messages"]["instructions_template"].startswith("You are a coding agent")
    assert entry["supported_reasoning_levels"] == [
        {"effort": "high", "description": "Deeper reasoning for complex problems"}
    ]
    assert entry["default_reasoning_level"] == "high"


def test_codex_catalog_defaults_away_from_none_and_fills_effort_copy():
    from sandbox_runtime.custom_providers import codex_model_catalog

    manifest = responses_manifest()
    manifest["models"][0]["reasoningEfforts"] = ["none", "xhigh"]
    providers = load_custom_providers(manifest_env(manifest))

    entry = codex_model_catalog(providers)["models"][0]
    assert [level["effort"] for level in entry["supported_reasoning_levels"]] == [
        "none",
        "xhigh",
    ]
    # "none" only marks a gateway that accepts disabling reasoning; it is not
    # a sensible default, so the last real level wins.
    assert entry["default_reasoning_level"] == "xhigh"


def test_write_codex_model_catalog_is_idempotent(tmp_path: Path):
    from sandbox_runtime.custom_providers import write_codex_model_catalog

    catalog_path = tmp_path / ".codex" / "custom-models.json"
    providers = load_custom_providers(manifest_env(responses_manifest()))
    assert write_codex_model_catalog(catalog_path, providers) is True
    written = catalog_path.read_text()
    assert json.loads(written)["models"][0]["slug"] == "gpt-x"
    assert write_codex_model_catalog(catalog_path, providers) is False
    assert catalog_path.read_text() == written

    # Without Responses-protocol providers nothing is written at all.
    assert write_codex_model_catalog(tmp_path / "other.json", ()) is False
    assert not (tmp_path / "other.json").exists()


# --- Pi models.json and dsh profile patch ---


def _load_both() -> tuple:
    env = {}
    for factory in (anthropic_manifest, openai_manifest):
        manifest = factory()
        env[CUSTOM_PROVIDERS_ENV] = json.dumps(
            [*(json.loads(env.get(CUSTOM_PROVIDERS_ENV, "[]"))), manifest]
        )
        env[manifest["apiKeyEnv"]] = "sk-gateway"
    return load_custom_providers(env)


def test_pi_document_covers_every_protocol():
    from sandbox_runtime.custom_providers import pi_models_document

    document = pi_models_document(_load_both())
    anthropic = document["providers"]["cpa-00112233"]
    assert anthropic["api"] == "anthropic-messages"
    assert anthropic["baseUrl"] == "https://gateway.example/api/anthropic"
    assert anthropic["apiKey"] == "$CP_00112233_API_KEY"
    assert [model["id"] for model in anthropic["models"]] == ["glm-4.7"]
    openai = document["providers"]["cpo-99887766"]
    assert openai["api"] == "openai-completions"


def test_write_pi_models_json_merges_and_is_idempotent(tmp_path: Path):
    from sandbox_runtime.custom_providers import write_pi_models_json

    config = tmp_path / "models.json"
    config.write_text(json.dumps({"providers": {"other": {"api": "anthropic-messages"}}}))
    assert write_pi_models_json(config, _load_both()) is True
    written = json.loads(config.read_text())
    assert set(written["providers"]) == {"other", "cpa-00112233", "cpo-99887766"}
    assert write_pi_models_json(config, _load_both()) is False
    assert json.loads(config.read_text()) == written


def test_dsh_patch_declares_routes_and_selection():
    from sandbox_runtime.custom_providers import (
        dsh_model_selection_patch,
        dsh_profile_patch_entries,
    )

    entries = dsh_profile_patch_entries(_load_both())
    text = "\n".join(entries)
    assert "- id: llm-pi-ai" in text
    assert '      "cpo-99887766":' in text
    assert "        api: openai-completions" in text
    assert "        api: anthropic-messages" in text
    assert '        apiKeyEnv: "CP_00112233_API_KEY"' in text
    assert "            contextWindow: 200000" in text
    assert '            reasoningEfforts: {"high": "high"}' in text

    selection = dsh_model_selection_patch("cpo-99887766", "glm-4.7", "none")
    assert 'provider: "cpo-99887766"' in selection
    assert 'model: "glm-4.7"' in selection
    assert 'reasoningEffort: "off"' in selection
    assert dsh_model_selection_patch("cpo-99887766", "glm-4.7", None).count("\n") == 4


@pytest.mark.parametrize(
    "registered,expected",
    [
        ("https://gateway.example/api/anthropic/", "https://gateway.example/api/anthropic"),
        ("https://gateway.example/api/anthropic/v1", "https://gateway.example/api/anthropic"),
        ("https://gateway.example/api/anthropic/v1/", "https://gateway.example/api/anthropic"),
    ],
)
def test_pi_and_dsh_write_anthropic_gateways_at_the_api_root(registered: str, expected: str):
    """pi-ai's Anthropic client joins ``/v1/messages`` itself, so both CLIs'
    configs carry the API root — a registered version segment would be
    requested twice. OpenAI gateways keep the segment (their clients append
    only the wire path)."""
    from sandbox_runtime.custom_providers import dsh_profile_patch_entries, pi_models_document

    manifest = anthropic_manifest()
    manifest["baseUrl"] = registered
    providers = load_custom_providers(manifest_env(manifest))

    pi_url = pi_models_document(providers)["providers"]["cpa-00112233"]["baseUrl"]
    assert pi_url == expected

    dsh_text = "\n".join(dsh_profile_patch_entries(providers))
    assert f'baseURL: "{expected}"' in dsh_text

    openai_providers = load_custom_providers(manifest_env(openai_manifest()))
    assert (
        pi_models_document(openai_providers)["providers"]["cpo-99887766"]["baseUrl"]
        == "https://gateway.example/v1"
    )
    assert 'baseURL: "https://gateway.example/v1"' in "\n".join(
        dsh_profile_patch_entries(openai_providers)
    )


def test_write_dsh_profile_patch_regenerates(tmp_path: Path):
    from sandbox_runtime.custom_providers import write_dsh_profile_patch

    patch = tmp_path / "cordis.patch.yml"
    assert write_dsh_profile_patch(patch, _load_both()) is True
    body = patch.read_text()
    assert body.startswith("- id: llm-pi-ai\n")
    assert write_dsh_profile_patch(patch, _load_both()) is False
    patch.write_text(body + "# trailing edit\n")
    assert write_dsh_profile_patch(patch, _load_both()) is True
