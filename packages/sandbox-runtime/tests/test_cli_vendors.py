"""Vendor argv and stdout-record translation for the CLI harnesses."""

import json
from pathlib import Path

import pytest

from sandbox_runtime.custom_providers import codex_model_catalog_path, load_custom_providers
from sandbox_runtime.harness.base import HarnessId, TurnOutcome
from sandbox_runtime.harness.cli_harness import CliTurnSettled, CliTurnState
from sandbox_runtime.harness.cli_vendors import (
    CodexVendor,
    PiVendor,
    ZcodeVendor,
    get_cli_vendor,
)

WORKDIR = Path("/workspace")


def _state() -> CliTurnState:
    return CliTurnState(message_id="m1", step_id="s1")


def _types(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


class TestCodexAppServer:
    """Resident app-server mode: request building and envelope translation."""

    def test_setup_starts_a_thread_with_the_routed_gateway(self) -> None:
        vendor = CodexVendor()
        message = vendor.next_setup_message(
            session_id=None,
            model="cpo-55443322/gpt-x",
            reasoning_effort=None,
            workdir=WORKDIR,
            model_provider="cpo-55443322",
        )
        assert message is not None
        assert message["method"] == "thread/start"
        params = message["params"]
        assert params["modelProvider"] == "cpo-55443322"
        assert params["model"] == "gpt-x"
        assert params["config"]["model_catalog_json"].endswith("custom-models.json")

    def test_setup_resumes_an_adopted_thread_only_once(self) -> None:
        vendor = CodexVendor()
        message = vendor.next_setup_message(
            session_id="t-9",
            model=None,
            reasoning_effort=None,
            workdir=WORKDIR,
            model_provider="cpo-55443322",
        )
        assert message is not None
        assert message["method"] == "thread/resume"
        assert message["params"]["threadId"] == "t-9"
        vendor.adopt_response_id({"result": {"thread": {"id": "t-9"}}})
        # The thread this process created is live: no setup on later turns.
        assert (
            vendor.next_setup_message(
                session_id="t-9",
                model=None,
                reasoning_effort=None,
                workdir=WORKDIR,
                model_provider="cpo-55443322",
            )
            is None
        )
        # A restarted server process knows no thread: the id resumes again.
        vendor.reset()
        message = vendor.next_setup_message(
            session_id="t-9",
            model=None,
            reasoning_effort=None,
            workdir=WORKDIR,
            model_provider=None,
        )
        assert message is not None
        assert message["method"] == "thread/resume"

    def test_turn_start_carries_prompt_model_and_effort(self) -> None:
        vendor = CodexVendor()
        message = vendor.turn_start_message(
            session_id="t-1",
            prompt_text="hi",
            model="cpo-55443322/gpt-x",
            reasoning_effort="high",
        )
        assert message is not None
        params = message["params"]
        assert params["threadId"] == "t-1"
        assert params["input"] == [{"type": "text", "text": "hi"}]
        assert params["model"] == "gpt-x"
        assert params["effort"] == "high"
        assert (
            vendor.turn_start_message(
                session_id=None, prompt_text="hi", model=None, reasoning_effort=None
            )
            is None
        )

    def test_notifications_translate_through_the_exec_shapes(self) -> None:
        vendor = CodexVendor()
        state = _state()
        assert (
            vendor.parse_server_message(
                {
                    "method": "turn/started",
                    "params": {"threadId": "t-1", "turn": {"id": "turn-7"}},
                },
                state,
            )
            == []
        )
        assert state.server_turn_id == "turn-7"

        events = vendor.parse_server_message(
            {
                "method": "item/completed",
                "params": {"item": {"id": "i2", "type": "agentMessage", "text": "done"}},
            },
            state,
        )
        assert events[-1]["type"] == "token" and events[-1]["content"] == "done"

        with pytest.raises(CliTurnSettled) as settled:
            vendor.parse_server_message(
                {
                    "method": "turn/completed",
                    "params": {"threadId": "t-1", "turn": {"usage": {"input_tokens": 3}}},
                },
                state,
            )
        assert state.completed
        assert state.tokens == {"input_tokens": 3}
        # The step already fired at the first output; the sentinel may carry
        # no further events — the harness emits whatever rides it either way.
        assert isinstance(settled.value.events, list)

    def test_compaction_item_emits_context_compacted(self) -> None:
        vendor = CodexVendor()
        state = _state()
        assert (
            vendor.parse_server_message(
                {
                    "method": "item/started",
                    "params": {"item": {"id": "i3", "type": "contextCompaction"}},
                },
                state,
            )
            == []
        )
        assert vendor.parse_server_message(
            {
                "method": "item/completed",
                "params": {"item": {"id": "i3", "type": "contextCompaction"}},
            },
            state,
        ) == [{"type": "context_compacted", "messageId": "m1"}]

    def test_turn_failed_settles_with_the_error_event(self) -> None:
        vendor = CodexVendor()
        state = _state()
        with pytest.raises(CliTurnSettled) as settled:
            vendor.parse_server_message(
                {
                    "method": "turn/failed",
                    "params": {"threadId": "t-1", "error": {"message": "boom"}},
                },
                state,
            )
        assert settled.value.events[0]["type"] == "error"

    def test_failed_turn_surfaces_the_provider_error(self) -> None:
        # Codex rides a failed turn on turn/completed with turn.error and no
        # items; the provider's reason must reach the user, not "no output".
        vendor = CodexVendor()
        state = _state()
        with pytest.raises(CliTurnSettled) as settled:
            vendor.parse_server_message(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "t-1",
                        "turn": {
                            "status": "failed",
                            "error": {"message": "Model does not support this protocol."},
                        },
                    },
                },
                state,
            )
        assert settled.value.events[-1]["type"] == "error"
        assert "protocol" in settled.value.events[-1]["error"]

    def test_interrupt_uses_the_running_turn_id(self) -> None:
        vendor = CodexVendor()
        state = _state()
        assert vendor.interrupt_messages(session_id="t-1", state=state) == []
        state.server_turn_id = "turn-7"
        (message,) = vendor.interrupt_messages(session_id="t-1", state=state)
        assert message["method"] == "turn/interrupt"
        assert message["params"] == {"threadId": "t-1", "turnId": "turn-7"}

    def test_server_requests_get_an_error_reply(self) -> None:
        vendor = CodexVendor()
        (reply,) = vendor.server_request_messages({"id": "server-1", "method": "agent/ask"})
        assert reply["id"] == "server-1"
        assert reply["error"]["code"] == -32601
        assert vendor.server_request_messages({"method": "turn/started"}) == []


class TestPiRpc:
    """Resident rpc mode: spawn argv, setup chain, and event translation."""

    def test_server_argv_carries_session_model_and_thinking(self) -> None:
        vendor = PiVendor()
        assert vendor.server_argv(
            session_id="s-1", model="cpo-55443322/gpt-x", reasoning_effort="none"
        ) == [
            "--mode",
            "rpc",
            "--approve",
            "--session-id",
            "s-1",
            "--model",
            "cpo-55443322/gpt-x",
            "--thinking",
            "off",
        ]

    def test_setup_probes_then_applies_model_and_thinking_once(self) -> None:
        vendor = PiVendor()
        assert vendor.next_setup_message(
            session_id="s-1",
            model="cpo-55443322/gpt-x",
            reasoning_effort="high",
            workdir=WORKDIR,
        ) == {"type": "get_state"}
        assert vendor.next_setup_message(
            session_id="s-1",
            model="cpo-55443322/gpt-x",
            reasoning_effort="high",
            workdir=WORKDIR,
        ) == {"type": "set_model", "provider": "cpo-55443322", "modelId": "gpt-x"}
        assert vendor.next_setup_message(
            session_id="s-1",
            model="cpo-55443322/gpt-x",
            reasoning_effort="high",
            workdir=WORKDIR,
        ) == {"type": "set_thinking_level", "level": "high"}
        # Converged: nothing more to send this turn, and nothing on the next.
        assert (
            vendor.next_setup_message(
                session_id="s-1",
                model="cpo-55443322/gpt-x",
                reasoning_effort="high",
                workdir=WORKDIR,
            )
            is None
        )
        # A model switch mid-conversation goes through set_model again.
        assert vendor.next_setup_message(
            session_id="s-1", model="anthropic/claude-x", reasoning_effort=None, workdir=WORKDIR
        ) == {"type": "set_model", "provider": "anthropic", "modelId": "claude-x"}

    def test_reset_reprobes_after_a_restart(self) -> None:
        vendor = PiVendor()
        vendor.next_setup_message(
            session_id=None, model=None, reasoning_effort=None, workdir=WORKDIR
        )
        vendor.reset()
        assert vendor.next_setup_message(
            session_id="s-1", model=None, reasoning_effort=None, workdir=WORKDIR
        ) == {"type": "get_state"}

    def test_adopt_and_turn_start(self) -> None:
        vendor = PiVendor()
        assert (
            vendor.adopt_response_id(
                {"type": "response", "command": "get_state", "data": {"sessionId": "s-7"}}
            )
            == "s-7"
        )
        assert vendor.adopt_response_id({"type": "response", "command": "prompt"}) is None
        assert vendor.turn_start_message(
            session_id="s-7", prompt_text="hi", model=None, reasoning_effort=None
        ) == {"type": "prompt", "message": "hi"}
        assert (
            vendor.turn_start_message(
                session_id=None, prompt_text="hi", model=None, reasoning_effort=None
            )
            is None
        )

    def test_events_translate_and_agent_settled_ends_the_turn(self) -> None:
        vendor = PiVendor()
        state = _state()
        assert vendor.parse_server_message({"type": "response", "command": "prompt"}, state) == []
        events = vendor.parse_server_message(
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "text_delta", "delta": "Hi"},
            },
            state,
        )
        assert events[-1]["type"] == "token" and events[-1]["content"] == "Hi"
        with pytest.raises(CliTurnSettled):
            vendor.parse_server_message({"type": "agent_settled"}, state)
        assert state.completed

    def test_interrupt_and_extension_ui_reply(self) -> None:
        vendor = PiVendor()
        assert vendor.interrupt_messages(session_id="s-1", state=_state()) == [{"type": "abort"}]
        assert vendor.server_request_messages(
            {"type": "extension_ui_request", "id": "ext-1", "method": "confirm"}
        ) == [{"type": "extension_ui_response", "id": "ext-1", "cancelled": True}]
        assert vendor.server_request_messages({"type": "message_update"}) == []


class TestZcodeAppServer:
    """Resident app-server mode: setup chain, events, interrupts, replies."""

    def _session_event(self, event_type: str, payload: dict | None = None) -> dict:
        return {"method": "session/event", "params": {"type": event_type, "payload": payload or {}}}

    def test_setup_creates_then_subscribes_with_the_adopted_id(self) -> None:
        vendor = ZcodeVendor()
        create = vendor.next_setup_message(
            session_id=None, model=None, reasoning_effort=None, workdir=WORKDIR
        )
        assert create is not None
        assert create["method"] == "session/create"
        assert create["params"]["workspace"] == {
            "workspacePath": str(WORKDIR),
            "workspaceKey": str(WORKDIR),
        }
        assert create["params"]["mode"] == "yolo"
        assert vendor.adopt_response_id({"result": {"session": {"sessionId": "z-1"}}}) == "z-1"
        subscribe = vendor.next_setup_message(
            session_id="z-1", model=None, reasoning_effort=None, workdir=WORKDIR
        )
        assert subscribe is not None
        assert subscribe["method"] == "session/subscribe"
        assert subscribe["params"]["sessionId"] == "z-1"
        assert subscribe["params"]["deliveryKind"] == "desktop-continuous"
        # Converged for this process.
        assert (
            vendor.next_setup_message(
                session_id="z-1", model=None, reasoning_effort=None, workdir=WORKDIR
            )
            is None
        )

    def test_setup_resumes_after_a_restart(self) -> None:
        vendor = ZcodeVendor()
        vendor.reset()
        resume = vendor.next_setup_message(
            session_id="z-1", model=None, reasoning_effort=None, workdir=WORKDIR
        )
        assert resume is not None
        assert resume["method"] == "session/resume"
        assert resume["params"] == {"sessionId": "z-1"}
        vendor.adopt_response_id({"result": {"session": {"sessionId": "z-1"}}})
        subscribe = vendor.next_setup_message(
            session_id="z-1", model=None, reasoning_effort=None, workdir=WORKDIR
        )
        assert subscribe is not None
        assert subscribe["method"] == "session/subscribe"

    def test_turn_start_sends_content_with_the_routed_selection(
        self, tmp_path, monkeypatch
    ) -> None:
        TestZcode()._stage_custom_provider(tmp_path, monkeypatch)
        vendor = ZcodeVendor()
        message = vendor.turn_start_message(
            session_id="z-1",
            prompt_text="task",
            model="cpa-00112233/glm-4.7",
            reasoning_effort=None,
            model_provider="cpa-00112233",
        )
        assert message is not None
        assert message["method"] == "session/send"
        assert message["params"]["content"] == "task"
        selection = message["params"]["modelSelection"]
        assert selection["providerId"] == "cpa-00112233"
        assert selection["modelId"] == "glm-4.7"

    def test_turn_start_without_a_route_has_no_selection(self) -> None:
        vendor = ZcodeVendor()
        message = vendor.turn_start_message(
            session_id="z-1",
            prompt_text="task",
            model="zai-coding-plan/glm-5.3",
            reasoning_effort=None,
        )
        assert message is not None
        assert "modelSelection" not in message["params"]
        assert (
            vendor.turn_start_message(
                session_id=None, prompt_text="task", model=None, reasoning_effort=None
            )
            is None
        )

    def test_events_translate_text_tools_and_terminal(self) -> None:
        vendor = ZcodeVendor()
        state = _state()
        events = vendor.parse_server_message(
            self._session_event("part.delta", {"field": "text", "delta": "Hel"}), state
        )
        assert events[-1]["type"] == "token" and events[-1]["content"] == "Hel"
        # Reasoning deltas are not surfaced yet.
        assert (
            vendor.parse_server_message(
                self._session_event("part.delta", {"field": "reasoning", "delta": "hm"}), state
            )
            == []
        )

        events = vendor.parse_server_message(
            self._session_event(
                "tool.updated",
                {
                    "kind": "started",
                    "toolCallId": "c1",
                    "toolName": "bash",
                    "input": {"command": "ls"},
                },
            ),
            state,
        )
        assert events[-1]["type"] == "tool_call" and events[-1]["status"] == "running"
        events = vendor.parse_server_message(
            self._session_event(
                "tool.updated", {"kind": "result", "toolCallId": "c1", "result": {"text": "ok"}}
            ),
            state,
        )
        assert events[-1]["status"] == "completed" and events[-1]["output"] == "ok"
        events = vendor.parse_server_message(
            self._session_event(
                "tool.updated",
                {"kind": "error", "toolCallId": "c1", "error": {"message": "denied"}},
            ),
            state,
        )
        assert events[-1]["status"] == "error" and events[-1]["output"] == "denied"

        with pytest.raises(CliTurnSettled) as settled:
            vendor.parse_server_message(
                self._session_event(
                    "turn.completed",
                    {"response": "Hello", "usage": {"input_tokens": 5, "junk": True}},
                ),
                state,
            )
        assert state.completed
        assert state.tokens == {"input_tokens": 5}
        assert settled.value.events[-1]["type"] == "token"
        assert settled.value.events[-1]["content"] == "Hello"

    def test_compaction_part_emits_context_compacted_once(self) -> None:
        vendor = ZcodeVendor()
        state = _state()
        boundary = {
            "messageId": "msg-9",
            "partId": "p-9",
            "part": {"type": "compaction", "auto": True},
        }
        assert (
            vendor.parse_server_message(self._session_event("part.started", boundary), state) == []
        )
        assert vendor.parse_server_message(
            self._session_event("part.upserted", boundary), state
        ) == [{"type": "context_compacted", "messageId": "m1"}]
        # Timeline separator parts are display-only; the boundary already
        # reported itself.
        assert (
            vendor.parse_server_message(
                self._session_event(
                    "part.upserted",
                    {"messageId": "msg-9", "partId": "p-10", "part": {"type": "timeline"}},
                ),
                state,
            )
            == []
        )

    def test_turn_failed_settles_with_the_error_event(self) -> None:
        vendor = ZcodeVendor()
        state = _state()
        with pytest.raises(CliTurnSettled) as settled:
            vendor.parse_server_message(
                self._session_event("turn.failed", {"error": {"message": "boom"}}), state
            )
        assert settled.value.events[0]["type"] == "error"

    def test_interrupt_stops_the_session(self) -> None:
        vendor = ZcodeVendor()
        assert vendor.interrupt_messages(session_id=None, state=_state()) == []
        (message,) = vendor.interrupt_messages(session_id="z-1", state=_state())
        assert message["method"] == "session/stop"
        assert message["params"] == {"sessionId": "z-1"}

    def test_permission_requests_are_denied_and_others_errored(self) -> None:
        vendor = ZcodeVendor()
        assert vendor.server_request_messages(
            {"id": "server-1", "method": "interaction/requestPermission"}
        ) == [{"id": "server-1", "result": {"decision": "deny"}}]
        # A user-input question is declined, not errored: the broker maps a
        # cancel to a graceful deny instead of a rejected promise.
        assert vendor.server_request_messages(
            {"id": "server-3", "method": "interaction/requestUserInput"}
        ) == [{"id": "server-3", "result": {"action": "cancel"}}]
        (reply,) = vendor.server_request_messages(
            {"id": "server-2", "method": "interaction/requestProviderRuntimeHeaders"}
        )
        assert reply["id"] == "server-2"
        assert reply["error"]["code"] == -32601
        assert vendor.server_request_messages({"method": "session/event"}) == []


def test_registry_lists_only_cli_harnesses() -> None:
    assert isinstance(get_cli_vendor(HarnessId.CODEX), CodexVendor)
    assert isinstance(get_cli_vendor(HarnessId.PI), PiVendor)
    assert isinstance(get_cli_vendor(HarnessId.ZCODE), ZcodeVendor)
    assert get_cli_vendor(HarnessId.OPENCODE) is None
    assert get_cli_vendor(HarnessId.CLAUDE) is None


def test_resident_envelopes_match_each_wire_schema() -> None:
    # Codex speaks JSON-RPC 2.0; Pi and ZCode reject or do not use the
    # envelope field, so the resident server must not inject it for them.
    assert CodexVendor.jsonrpc is True
    assert PiVendor.jsonrpc is False
    assert ZcodeVendor.jsonrpc is False


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

    def test_compaction_item_emits_context_compacted(self) -> None:
        vendor = CodexVendor()
        state = _state()
        assert (
            vendor.parse_record(
                {"type": "item.started", "item": {"id": "i3", "type": "context_compaction"}},
                state,
            )
            == []
        )
        assert vendor.parse_record(
            {"type": "item.completed", "item": {"id": "i3", "type": "context_compaction"}},
            state,
        ) == [{"type": "context_compacted", "messageId": "m1"}]

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

    def _stage_custom_provider(self, tmp_path, monkeypatch) -> None:
        manifest = {
            "id": "0011223344556677889900aabbccddee",
            "providerKey": "cpa-00112233",
            "protocol": "anthropic",
            "baseUrl": "https://gateway.example/api/anthropic",
            "headers": [],
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
        monkeypatch.setenv("CUSTOM_MODEL_PROVIDERS", json.dumps([manifest]))
        monkeypatch.setenv("CP_00112233_API_KEY", "sk-a")
        monkeypatch.setenv("HOME", str(tmp_path))

    def test_prepare_writes_personal_provider_config(self, tmp_path, monkeypatch) -> None:
        self._stage_custom_provider(tmp_path, monkeypatch)

        ZcodeVendor().prepare(load_custom_providers())

        config = tmp_path / ".zcode" / "v2" / "provider_config.json"
        document = json.loads(config.read_text())
        rule = document["config"]["providerConfigRules"]["providerRules"][0]
        assert rule["providerId"] == "cpa-00112233"
        assert rule["config"]["api"]["baseUrl"] == "https://gateway.example/api/anthropic"
        # ZCode has no environment seam for credentials, so the key rides the
        # file itself.
        assert rule["config"]["access"]["apiKey"] == "sk-a"
        assert "defaultModelSelection" not in document["config"]

    def test_custom_model_turn_routes_via_default_model_selection(
        self, tmp_path, monkeypatch
    ) -> None:
        self._stage_custom_provider(tmp_path, monkeypatch)
        vendor = ZcodeVendor()
        vendor.prepare(load_custom_providers())
        config = tmp_path / ".zcode" / "v2" / "provider_config.json"

        argv = vendor.build_argv(
            session_id=None,
            prompt_text="build it",
            model="cpa-00112233/glm-4.7",
            reasoning_effort="high",
            workdir=WORKDIR,
            model_provider="cpa-00112233",
        )
        assert argv == ["--prompt", "build it"]
        selection = json.loads(config.read_text())["config"]["defaultModelSelection"]
        assert selection == {
            "providerId": "cpa-00112233",
            "modelId": "glm-4.7",
            "options": {"reasoningLevel": "high"},
        }

        # An official-model turn carries no routing and leaves the file alone.
        assert vendor.build_argv(
            session_id=None,
            prompt_text="again",
            model="zai-coding-plan/glm-5.3",
            reasoning_effort=None,
            workdir=WORKDIR,
        ) == ["--prompt", "again"]
        assert json.loads(config.read_text())["config"]["defaultModelSelection"] == selection
