"""Vendor argv and stdout-record translation for the CLI harnesses."""

import json
from pathlib import Path

from sandbox_runtime.custom_providers import codex_model_catalog_path, load_custom_providers
from sandbox_runtime.harness.base import HarnessId, TurnOutcome
from sandbox_runtime.harness.cli_harness import CliTurnState
from sandbox_runtime.harness.cli_vendors import (
    CodexVendor,
    DshVendor,
    PiVendor,
    ZcodeVendor,
    get_cli_vendor,
)

WORKDIR = Path("/workspace")


def _state() -> CliTurnState:
    return CliTurnState(message_id="m1", step_id="s1")


def _types(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


def test_registry_lists_only_cli_harnesses() -> None:
    assert isinstance(get_cli_vendor(HarnessId.CODEX), CodexVendor)
    assert isinstance(get_cli_vendor(HarnessId.PI), PiVendor)
    assert isinstance(get_cli_vendor(HarnessId.DSH), DshVendor)
    assert isinstance(get_cli_vendor(HarnessId.ZCODE), ZcodeVendor)
    assert get_cli_vendor(HarnessId.OPENCODE) is None
    assert get_cli_vendor(HarnessId.CLAUDE) is None


class TestCodex:
    def test_fresh_and_resume_argv(self) -> None:
        vendor = CodexVendor()
        fresh = vendor.build_argv(
            session_id=None,
            prompt_text="do it",
            model="openai/gpt-5.5",
            reasoning_effort="high",
            workdir=WORKDIR,
        )
        assert "resume" not in fresh
        assert fresh[:1] == ["exec"]
        assert "--json" in fresh
        assert "--model" in fresh and fresh[fresh.index("--model") + 1] == "gpt-5.5"
        assert fresh[-1] == "do it"
        assert fresh[-2] == "--"

        resume = vendor.build_argv(
            session_id="thread-1",
            prompt_text="again",
            model=None,
            reasoning_effort=None,
            workdir=WORKDIR,
        )
        assert resume[:3] == ["exec", "resume", "thread-1"]

    def test_model_provider_selects_the_custom_gateway(self) -> None:
        vendor = CodexVendor()
        argv = vendor.build_argv(
            session_id=None,
            prompt_text="do it",
            model="cpo-55443322/gpt-x",
            reasoning_effort=None,
            workdir=WORKDIR,
            model_provider="cpo-55443322",
        )
        assert argv[argv.index("-c") + 1] == "model_provider=cpo-55443322"
        assert argv[argv.index("--model") + 1] == "gpt-x"
        # The routed turn also points at the generated model catalog, so the
        # CLI resolves the gateway model's metadata instead of falling back.
        assert f"model_catalog_json={codex_model_catalog_path()}" in argv

    def test_official_model_argv_carries_no_catalog_override(self) -> None:
        argv = CodexVendor().build_argv(
            session_id=None,
            prompt_text="do it",
            model="openai/gpt-5.5",
            reasoning_effort=None,
            workdir=WORKDIR,
        )
        assert not any(argument.startswith("model_catalog_json=") for argument in argv)

    def test_prepare_writes_openai_protocol_gates_into_codex_config(
        self, tmp_path, monkeypatch
    ) -> None:
        manifest = {
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
                    "reasoningEfforts": [],
                    "contextWindowTokens": 400_000,
                    "maxOutputTokens": 65_536,
                }
            ],
        }
        monkeypatch.setenv("CUSTOM_MODEL_PROVIDERS", json.dumps([manifest]))
        monkeypatch.setenv("CP_55443322_API_KEY", "sk-r")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)

        CodexVendor().prepare(load_custom_providers())

        config = (tmp_path / ".codex" / "config.toml").read_text()
        assert "[model_providers.cpo-55443322]" in config
        assert 'wire_api = "responses"' in config
        assert "sk-r" not in config
        catalog = json.loads((tmp_path / ".codex" / "custom-models.json").read_text())
        assert [model["slug"] for model in catalog["models"]] == ["gpt-x"]

    def test_translates_thread_items_and_completion(self) -> None:
        vendor = CodexVendor()
        state = _state()
        assert vendor.parse_record({"type": "thread.started", "thread_id": "t-1"}, state) == []
        assert state.session_id == "t-1"

        events = vendor.parse_record(
            {
                "type": "item.completed",
                "item": {
                    "id": "i1",
                    "type": "command_execution",
                    "command": "ls",
                    "status": "completed",
                    "aggregated_output": "ok",
                },
            },
            state,
        )
        assert "tool_call" in _types(events)

        events = vendor.parse_record(
            {
                "type": "item.completed",
                "item": {"id": "i2", "type": "agent_message", "text": "done"},
            },
            state,
        )
        assert events[-1]["type"] == "token" and events[-1]["content"] == "done"

        events = vendor.parse_record(
            {"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 5}}, state
        )
        assert state.completed
        assert state.tokens == {"input_tokens": 3, "output_tokens": 5}

    def test_turn_failed_is_an_error(self) -> None:
        vendor = CodexVendor()
        events = vendor.parse_record(
            {"type": "turn.failed", "error": {"message": "boom"}}, _state()
        )
        assert events == [{"type": "error", "error": "boom", "messageId": "m1"}]

    def test_transient_error_records_are_not_fatal(self) -> None:
        vendor = CodexVendor()
        state = _state()
        reconnect = vendor.parse_record(
            {"type": "error", "message": "Reconnecting... 2/5 (401)"}, state
        )
        assert reconnect == []
        assert state.error is None
        events = vendor.parse_record({"type": "error", "message": "disk on fire"}, state)
        assert events == [{"type": "warning", "scope": "provider", "message": "disk on fire"}]
        assert state.error is None

    def test_exit_outcome(self) -> None:
        assert CodexVendor().exit_outcome(_state(), 0, "") == TurnOutcome.failed(
            "codex completed without emitting assistant output."
        )
        state = _state()
        state.text = "hi"
        assert CodexVendor().exit_outcome(state, 0, "").success


class TestPi:
    def test_session_id_is_chosen_up_front(self) -> None:
        session_id = PiVendor().initial_session_id()
        assert session_id

    def test_argv_uses_session_model_and_thinking(self) -> None:
        argv = PiVendor().build_argv(
            session_id="abc-123",
            prompt_text="hello",
            model="deepseek/deepseek-v4-pro",
            reasoning_effort="high",
            workdir=WORKDIR,
        )
        assert argv[:1] == ["--mode"]
        assert "--session-id" in argv and argv[argv.index("--session-id") + 1] == "abc-123"
        assert argv[argv.index("--model") + 1] == "deepseek/deepseek-v4-pro"
        assert argv[argv.index("--thinking") + 1] == "high"
        assert argv[-2:] == ["--", "hello"]

    def test_translates_session_delta_and_tools(self) -> None:
        vendor = PiVendor()
        state = _state()
        assert vendor.parse_record({"type": "session", "id": "p-1"}, state) == []
        assert state.session_id == "p-1"

        events = vendor.parse_record(
            {
                "type": "message_update",
                "usage": {"cost": {"total": 0.02}},
                "assistantMessageEvent": {"type": "text_delta", "delta": "Hel"},
            },
            state,
        )
        assert events[-1] == {"type": "token", "content": "Hel", "messageId": "m1"}
        events = vendor.parse_record(
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "text_delta", "delta": "lo"},
            },
            state,
        )
        assert events[-1]["content"] == "Hello"
        assert state.cost_usd == 0.02

        events = vendor.parse_record(
            {
                "type": "tool_execution_end",
                "toolCallId": "c1",
                "toolName": "bash",
                "result": {"content": [{"type": "text", "text": "out"}]},
                "isError": False,
            },
            state,
        )
        assert events[-1]["type"] == "tool_call"
        assert events[-1]["status"] == "completed"
        assert events[-1]["output"] == "out"

    def test_compaction_and_title(self) -> None:
        vendor = PiVendor()
        state = _state()
        assert vendor.parse_record({"type": "compaction_end"}, state) == [
            {"type": "context_compacted", "messageId": "m1"}
        ]
        assert vendor.parse_record({"type": "session_info_changed", "name": "Fix"}, state) == [
            {"type": "session_title", "title": "Fix"}
        ]

    def test_assistant_message_end_carries_the_failure_signal(self) -> None:
        # Pi exits 0 even when the provider rejects the turn; the assistant
        # message's stopReason is the only fatal marker on the wire.
        vendor = PiVendor()
        state = _state()
        events = vendor.parse_record(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [],
                    "stopReason": "error",
                    "errorMessage": "401 invalid x-api-key",
                    "usage": {"cost": {"total": 0}},
                },
            },
            state,
        )
        assert events == [{"type": "error", "error": "401 invalid x-api-key", "messageId": "m1"}]
        assert not vendor.exit_outcome(state, 0, "").success
        assert "401 invalid x-api-key" in (vendor.exit_outcome(state, 0, "").error or "")

    def test_non_assistant_message_ends_are_ignored(self) -> None:
        vendor = PiVendor()
        state = _state()
        for role in ("system", "user"):
            events = vendor.parse_record(
                {"type": "message_end", "message": {"role": role, "content": "say hi"}}, state
            )
            assert events == []
        assert state.error is None

    def test_final_assistant_text_extends_the_deltas(self) -> None:
        vendor = PiVendor()
        state = _state()
        vendor.parse_record(
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "text_delta", "delta": "Hel"},
            },
            state,
        )
        events = vendor.parse_record(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello"}],
                    "stopReason": "stop",
                },
            },
            state,
        )
        assert events == [{"type": "token", "content": "Hello", "messageId": "m1"}]


class TestDsh:
    def test_resume_argv_and_extra_env(self) -> None:
        vendor = DshVendor()
        argv = vendor.build_argv(
            session_id="sess-9",
            prompt_text="task",
            model="deepseek/deepseek-v4-pro",
            reasoning_effort=None,
            workdir=WORKDIR,
        )
        assert argv[:3] == ["--profile", "headless", "--json"]
        assert argv[argv.index("--session-id") + 1] == "sess-9"
        assert argv[-1] == "task"
        assert vendor.extra_env(model="deepseek/deepseek-v4-pro") == {
            "DSH_MODEL": "deepseek-v4-pro"
        }
        assert vendor.extra_env(model=None) == {}

    def test_translates_session_text_tools_and_final(self) -> None:
        vendor = DshVendor()
        state = _state()
        vendor.parse_record({"type": "session", "sessionId": "d-1", "cwd": "/workspace"}, state)
        assert state.session_id == "d-1"

        events = vendor.parse_record({"type": "text", "text": "wor"}, state)
        assert events[-1]["type"] == "token"
        # Record shapes per dsh-headless's projection (0.1.7-rc.2): the name
        # rides `tool`, the arguments `input`, and result records carry no
        # name at all — the learned one must survive the completing event.
        events = vendor.parse_record(
            {"type": "tool_call", "callId": "c", "tool": "bash", "input": {"command": "ls"}},
            state,
        )
        assert events[-1]["type"] == "tool_call"
        assert events[-1]["tool"] == "bash"
        assert events[-1]["args"] == {"command": "ls"}
        events = vendor.parse_record(
            {"type": "tool_result", "callId": "c", "status": "completed", "result": "ok"}, state
        )
        assert events[-1]["tool"] == "bash"
        assert events[-1]["status"] == "completed"

        events = vendor.parse_record({"type": "final", "text": "world"}, state)
        assert events[-1] == {"type": "token", "content": "world", "messageId": "m1"}
        assert state.completed

    def test_tool_result_error_status_is_preserved(self) -> None:
        vendor = DshVendor()
        state = _state()
        vendor.parse_record(
            {"type": "tool_call", "callId": "c", "tool": "bash", "input": {"command": "ls"}},
            state,
        )
        events = vendor.parse_record(
            {"type": "tool_result", "callId": "c", "status": "error", "result": "denied"}, state
        )
        assert events[-1]["tool"] == "bash"
        assert events[-1]["status"] == "error"
        assert events[-1]["output"] == "denied"

    def test_error_record(self) -> None:
        events = DshVendor().parse_record({"type": "error", "message": "nope"}, _state())
        assert events[0]["type"] == "error" and events[0]["error"] == "nope"


class TestZcode:
    def test_argv_is_headless_prompt_with_optional_resume(self) -> None:
        vendor = ZcodeVendor()
        fresh = vendor.build_argv(
            session_id=None,
            prompt_text="build it",
            model="zai-coding-plan/glm-5.3",
            reasoning_effort=None,
            workdir=WORKDIR,
        )
        assert fresh == ["--prompt", "build it"]
        resumed = vendor.build_argv(
            session_id="sess_1",
            prompt_text="continue",
            model=None,
            reasoning_effort=None,
            workdir=WORKDIR,
        )
        assert resumed == ["--prompt", "continue", "--resume", "sess_1"]
        assert vendor.initial_session_id() is None
