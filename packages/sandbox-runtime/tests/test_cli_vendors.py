"""Vendor argv and stdout-record translation for the CLI harnesses."""

import json
from pathlib import Path

import pytest

from sandbox_runtime.custom_providers import codex_model_catalog_path, load_custom_providers
from sandbox_runtime.harness.base import HarnessId, TurnOutcome
from sandbox_runtime.harness.cli_harness import CliTurnSettled, CliTurnState, step_finish_event
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
        assert state.tokens == {"input": 3}
        # The step already fired at the first output; the sentinel may carry
        # no further events — the harness emits whatever rides it either way.
        assert isinstance(settled.value.events, list)

    def test_reasoning_summary_deltas_stream_as_thinking(self) -> None:
        # The summary is Codex's visible thinking trail; the raw
        # item/reasoning/textDelta counterpart stays provider-internal and is
        # not read. Deltas accumulate per summary part, and the completed
        # item's snapshot completes anything that streamed short.
        vendor = CodexVendor()
        state = _state()
        events = vendor.parse_server_message(
            {
                "method": "item/reasoning/summaryTextDelta",
                "params": {"itemId": "r1", "summaryIndex": 0, "delta": "Let me"},
            },
            state,
        )
        assert events == [
            {"type": "step_start", "messageId": "m1", "stepId": state.step_id},
            {"type": "thinking", "content": "Let me", "messageId": "m1"},
        ]
        events = vendor.parse_server_message(
            {
                "method": "item/reasoning/summaryTextDelta",
                "params": {"itemId": "r1", "summaryIndex": 0, "delta": " check"},
            },
            state,
        )
        assert events == [{"type": "thinking", "content": "Let me check", "messageId": "m1"}]
        # Part 1 streamed no deltas: the item.completed snapshot seeds it and
        # joins it after part 0, in summary order.
        snapshot = {
            "method": "item/completed",
            "params": {
                "item": {"id": "r1", "type": "reasoning", "summary": ["Let me check", "Then act"]}
            },
        }
        assert vendor.parse_server_message(snapshot, state) == [
            {"type": "thinking", "content": "Let me check\n\nThen act", "messageId": "m1"}
        ]
        # A repeated snapshot is a no-op — the deltas and the snapshot must
        # not double the trail.
        assert vendor.parse_server_message(snapshot, state) == []

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

    def test_integer_request_ids_are_answered(self) -> None:
        # The app-server numbers its server→client requests with integers
        # (0, 1, ...); an approval request no client answers hangs the turn.
        vendor = CodexVendor()
        (reply,) = vendor.server_request_messages(
            {"id": 0, "method": "item/commandExecution/requestApproval", "params": {}}
        )
        assert reply["id"] == 0
        assert reply["error"]["code"] == -32601
        assert vendor.server_request_messages({"id": None, "method": "agent/ask"}) == []

    def test_thread_setup_pins_unattended_sandbox_and_approval(self) -> None:
        # The exec path passes --dangerously-bypass-approvals-and-sandbox;
        # the app-server path must request the same handling per thread or
        # every edit blocks on an approval nobody answers.
        vendor = CodexVendor()
        start = vendor.next_setup_message(
            session_id=None, model=None, reasoning_effort=None, workdir=WORKDIR
        )
        assert start is not None
        assert start["params"]["config"] == {
            "sandbox_mode": "danger-full-access",
            "approval_policy": "never",
        }
        resume = vendor.next_setup_message(
            session_id="t-9", model=None, reasoning_effort=None, workdir=WORKDIR
        )
        assert resume is not None
        assert resume["params"]["config"] == {
            "sandbox_mode": "danger-full-access",
            "approval_policy": "never",
        }

    def test_writer_lock_rejection_is_recognized(self) -> None:
        # rollout's writer lock is held for the thread's live lifetime by one
        # app-server process; a resume reaching another process's lock is the
        # recoverable contention the harness sweeps for.
        vendor = CodexVendor()
        assert vendor.setup_lock_contention(
            "thread 01a10210-3e88-7b10-9997-11ba39470c2d already has an active writer"
        )
        assert not vendor.setup_lock_contention("no rollout found for thread t-1")
        assert not vendor.turn_start_busy_failure("turn already in progress")
        assert not vendor.previous_run_settled({"method": "turn/completed"})

    def test_command_items_reach_the_translator_with_their_output(self) -> None:
        # App-server command items carry camelCase fields (aggregatedOutput);
        # the exec-shaped parser reads snake_case, so without the adapter the
        # command's output is silently dropped from the tool_call event.
        vendor = CodexVendor()
        state = _state()
        events = vendor.parse_server_message(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "t-1",
                    "item": {
                        "id": "c1",
                        "type": "commandExecution",
                        "command": "ls",
                        "status": "completed",
                        "aggregatedOutput": "a.txt",
                    },
                },
            },
            state,
        )
        tool = next(event for event in events if event["type"] == "tool_call")
        assert tool["tool"] == "bash"
        assert tool["status"] == "completed"
        assert tool["output"] == "a.txt"

    def test_declined_command_is_an_error_not_running_work(self) -> None:
        # A declined approval means the command never ran; reporting it as
        # still running leaves a permanently pending tool call in the UI.
        vendor = CodexVendor()
        state = _state()
        events = vendor.parse_server_message(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "t-1",
                    "item": {
                        "id": "c1",
                        "type": "commandExecution",
                        "command": "rm -rf /",
                        "status": "declined",
                    },
                },
            },
            state,
        )
        tool = next(event for event in events if event["type"] == "tool_call")
        assert tool["status"] == "error"

    def test_interrupted_turn_settles_as_cancelled(self) -> None:
        # Every terminal turn rides turn/completed; an interrupted one must
        # not settle as success (or as a misleading "no output" failure).
        vendor = CodexVendor()
        state = _state()
        with pytest.raises(CliTurnSettled):
            vendor.parse_server_message(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "t-1",
                        "turn": {"id": "turn-7", "status": "interrupted"},
                    },
                },
                state,
            )
        assert state.cancelled
        outcome = vendor.exit_outcome(state, 0, "")
        assert outcome.cancelled and not outcome.success
        assert outcome.error == "Task was cancelled"

    def test_failed_turn_without_a_message_still_fails(self) -> None:
        # turn.error is optional on a failed turn; the settle must not fall
        # through to success when the provider gave no reason.
        vendor = CodexVendor()
        state = _state()
        with pytest.raises(CliTurnSettled) as settled:
            vendor.parse_server_message(
                {
                    "method": "turn/completed",
                    "params": {"threadId": "t-1", "turn": {"id": "turn-7", "status": "failed"}},
                },
                state,
            )
        assert settled.value.events[-1]["type"] == "error"
        assert state.error == "Codex turn failed"

    def test_token_usage_notification_feeds_the_settled_tokens(self) -> None:
        # The v2 turn carries no usage; the thread's tokenUsage notification
        # is the only source, and its breakdown is camelCase.
        vendor = CodexVendor()
        state = _state()
        assert (
            vendor.parse_server_message(
                {
                    "method": "thread/tokenUsage/updated",
                    "params": {
                        "threadId": "t-1",
                        "turnId": "turn-7",
                        "tokenUsage": {
                            "total": {
                                "totalTokens": 120,
                                "inputTokens": 100,
                                "cachedInputTokens": 40,
                                "cacheWriteInputTokens": 10,
                                "outputTokens": 20,
                                "reasoningOutputTokens": 5,
                            },
                            "last": {"totalTokens": 120},
                            "modelContextWindow": 400000,
                        },
                    },
                },
                state,
            )
            == []
        )
        assert state.tokens == {
            "input": 100,
            "output": 20,
            "reasoning": 5,
            "cache": {"read": 40, "write": 10},
            "total": 120,
        }

    def test_agent_message_deltas_stream_the_live_answer(self) -> None:
        # The app-server streams the answer per delta while the exec stream
        # only prints the finished item; cumulative emission reuses one token
        # card, and the completed item's snapshot reconciles it.
        vendor = CodexVendor()
        state = _state()
        events = vendor.parse_server_message(
            {
                "method": "item/agentMessage/delta",
                "params": {"threadId": "t-1", "turnId": "turn-7", "itemId": "i2", "delta": "Hel"},
            },
            state,
        )
        assert events == [
            {"type": "step_start", "messageId": "m1", "stepId": state.step_id},
            {"type": "token", "content": "Hel", "messageId": "m1"},
        ]
        events = vendor.parse_server_message(
            {
                "method": "item/agentMessage/delta",
                "params": {"threadId": "t-1", "turnId": "turn-7", "itemId": "i2", "delta": "lo"},
            },
            state,
        )
        assert events == [{"type": "token", "content": "Hello", "messageId": "m1"}]
        events = vendor.parse_server_message(
            {
                "method": "item/completed",
                "params": {"item": {"id": "i2", "type": "agentMessage", "text": "Hello"}},
            },
            state,
        )
        assert events == []
        assert state.text == "Hello"

    def test_command_output_deltas_stream_into_the_running_card(self) -> None:
        # App-server command output rides item/commandExecution/outputDelta
        # (the exec stream only reports the final aggregatedOutput); the card
        # is re-emitted cumulatively with the cached name and args, and the
        # completed item reuses the streamed text when it carries none.
        vendor = CodexVendor()
        state = _state()
        vendor.parse_server_message(
            {
                "method": "item/started",
                "params": {
                    "item": {
                        "id": "c1",
                        "type": "commandExecution",
                        "command": "ls",
                        "status": "inProgress",
                    }
                },
            },
            state,
        )
        events = vendor.parse_server_message(
            {
                "method": "item/commandExecution/outputDelta",
                "params": {
                    "threadId": "t-1",
                    "turnId": "turn-7",
                    "itemId": "c1",
                    "delta": "a.txt\n",
                },
            },
            state,
        )
        tool = next(event for event in events if event["type"] == "tool_call")
        assert tool == {
            "type": "tool_call",
            "tool": "bash",
            "args": {"command": "ls"},
            "callId": "c1",
            "status": "running",
            "output": "a.txt\n",
            "messageId": "m1",
        }
        events = vendor.parse_server_message(
            {
                "method": "item/commandExecution/outputDelta",
                "params": {
                    "threadId": "t-1",
                    "turnId": "turn-7",
                    "itemId": "c1",
                    "delta": "b.txt\n",
                },
            },
            state,
        )
        tool = next(event for event in events if event["type"] == "tool_call")
        assert tool["output"] == "a.txt\nb.txt\n"

        events = vendor.parse_server_message(
            {
                "method": "item/completed",
                "params": {
                    "item": {
                        "id": "c1",
                        "type": "commandExecution",
                        "command": "ls",
                        "status": "completed",
                    }
                },
            },
            state,
        )
        tool = next(event for event in events if event["type"] == "tool_call")
        assert tool["status"] == "completed"
        assert tool["output"] == "a.txt\nb.txt\n"

    def test_patch_updated_streams_the_running_edit_with_its_diff(self) -> None:
        # item/fileChange/patchUpdated carries the full replacement change
        # set with no item body; it becomes a running edit whose output shows
        # the per-file diff, and later sets replace it in place.
        vendor = CodexVendor()
        state = _state()
        changes = [
            {"path": "/workspace/a.ts", "kind": {"type": "update"}, "diff": "@@\n-old\n+new"},
            {"path": "/workspace/b.ts", "kind": "add", "diff": "+b"},
        ]
        events = vendor.parse_server_message(
            {
                "method": "item/fileChange/patchUpdated",
                "params": {
                    "threadId": "t-1",
                    "turnId": "turn-7",
                    "itemId": "f1",
                    "changes": changes,
                },
            },
            state,
        )
        tool = next(event for event in events if event["type"] == "tool_call")
        assert tool["tool"] == "edit"
        assert tool["callId"] == "f1"
        assert tool["status"] == "running"
        assert tool["args"] == {"changes": changes}
        assert tool["output"] == "/workspace/a.ts:\n@@\n-old\n+new\n/workspace/b.ts:\n+b"

        fuller = [*changes, {"path": "/workspace/c.ts", "kind": "add", "diff": "+c"}]
        events = vendor.parse_server_message(
            {
                "method": "item/fileChange/patchUpdated",
                "params": {
                    "threadId": "t-1",
                    "turnId": "turn-7",
                    "itemId": "f1",
                    "changes": fuller,
                },
            },
            state,
        )
        tool = next(event for event in events if event["type"] == "tool_call")
        assert tool["args"] == {"changes": fuller}
        assert tool["output"].endswith("/workspace/c.ts:\n+c")

        events = vendor.parse_server_message(
            {
                "method": "item/completed",
                "params": {
                    "item": {
                        "id": "f1",
                        "type": "fileChange",
                        "status": "completed",
                        "changes": fuller,
                    }
                },
            },
            state,
        )
        tool = next(event for event in events if event["type"] == "tool_call")
        assert tool["status"] == "completed"
        assert tool["output"].endswith("/workspace/c.ts:\n+c")

    def test_web_search_items_become_tool_calls(self) -> None:
        # WebSearchItem has no status field; the started/completed item pair
        # carries the progress instead, and the query (plus an action when
        # present) is forwarded.
        vendor = CodexVendor()
        state = _state()
        events = vendor.parse_server_message(
            {
                "method": "item/started",
                "params": {"item": {"id": "w1", "type": "webSearch", "query": "vitest config"}},
            },
            state,
        )
        tool = next(event for event in events if event["type"] == "tool_call")
        assert tool["tool"] == "WebSearch"
        assert tool["args"] == {"query": "vitest config"}
        assert tool["status"] == "running"
        events = vendor.parse_server_message(
            {
                "method": "item/completed",
                "params": {
                    "item": {
                        "id": "w1",
                        "type": "webSearch",
                        "query": "vitest config",
                        "action": {"type": "search", "query": "vitest config"},
                        "results": [{"title": "Vitest"}],
                    }
                },
            },
            state,
        )
        tool = next(event for event in events if event["type"] == "tool_call")
        assert tool["status"] == "completed"
        assert tool["args"] == {
            "query": "vitest config",
            "action": {"type": "search", "query": "vitest config"},
        }

    def test_turn_plan_notifications_become_todowrite(self) -> None:
        # The plan rides turn/plan/updated; the task panel reads TodoWrite
        # tool calls, so each update is forwarded as one with the camelCase
        # step statuses mapped to the todo vocabulary.
        vendor = CodexVendor()
        state = _state()
        events = vendor.parse_server_message(
            {
                "method": "turn/plan/updated",
                "params": {
                    "threadId": "t-1",
                    "turnId": "turn-7",
                    "explanation": "working",
                    "plan": [
                        {"step": "Inspect", "status": "completed"},
                        {"step": "Patch", "status": "inProgress"},
                        {"step": "Verify", "status": "pending"},
                    ],
                },
            },
            state,
        )
        tool = next(event for event in events if event["type"] == "tool_call")
        assert tool["tool"] == "TodoWrite"
        assert tool["callId"] == "plan"
        assert tool["status"] == "running"
        assert tool["args"] == {
            "todos": [
                {"content": "Inspect", "status": "completed"},
                {"content": "Patch", "status": "in_progress"},
                {"content": "Verify", "status": "pending"},
            ]
        }

        events = vendor.parse_server_message(
            {
                "method": "turn/plan/updated",
                "params": {
                    "threadId": "t-1",
                    "turnId": "turn-7",
                    "plan": [
                        {"step": "Inspect", "status": "completed"},
                        {"step": "Patch", "status": "completed"},
                        {"step": "Verify", "status": "completed"},
                    ],
                },
            },
            state,
        )
        tool = next(event for event in events if event["type"] == "tool_call")
        assert tool["status"] == "completed"

    def test_turn_plan_edges(self) -> None:
        vendor = CodexVendor()
        state = _state()
        # An empty plan is a deliberate clear: emit it so the panel empties.
        events = vendor.parse_server_message(
            {"method": "turn/plan/updated", "params": {"plan": []}}, state
        )
        tool = next(event for event in events if event["type"] == "tool_call")
        assert tool["args"] == {"todos": []}
        assert tool["status"] == "running"
        # An unknown status is simply not carried; the step still shows.
        events = vendor.parse_server_message(
            {
                "method": "turn/plan/updated",
                "params": {"plan": [{"step": "Odd", "status": "weird"}]},
            },
            state,
        )
        tool = next(event for event in events if event["type"] == "tool_call")
        assert tool["args"] == {"todos": [{"content": "Odd"}]}
        # A populated plan with nothing readable must not clobber the panel.
        events = vendor.parse_server_message(
            {"method": "turn/plan/updated", "params": {"plan": [{"status": "pending"}, 3]}},
            state,
        )
        assert [event for event in events if event["type"] == "tool_call"] == []
        # A plan that is not a list at all is noise.
        assert (
            vendor.parse_server_message(
                {"method": "turn/plan/updated", "params": {"plan": "nope"}}, state
            )
            == []
        )


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

    def test_thinking_deltas_stream_and_the_end_never_shrinks_the_trail(self) -> None:
        vendor = PiVendor()
        state = _state()
        events = vendor.parse_server_message(
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "thinking_delta", "delta": "Hmm"},
            },
            state,
        )
        assert events == [
            {"type": "step_start", "messageId": "m1", "stepId": state.step_id},
            {"type": "thinking", "content": "Hmm", "messageId": "m1"},
        ]
        events = vendor.parse_server_message(
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "thinking_delta", "delta": " ok"},
            },
            state,
        )
        assert events == [{"type": "thinking", "content": "Hmm ok", "messageId": "m1"}]
        # A shorter end-of-block snapshot must not shrink the streamed trail.
        assert (
            vendor.parse_server_message(
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "thinking_end", "content": "Hm"},
                },
                state,
            )
            == []
        )
        # A block whose deltas never arrived (a redacted block is complete at
        # start) is adopted whole when its end exceeds the trail.
        events = vendor.parse_server_message(
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "thinking_end", "content": "Hmm ok, plus"},
            },
            state,
        )
        assert events == [{"type": "thinking", "content": "Hmm ok, plus", "messageId": "m1"}]

    def test_interrupt_and_extension_ui_reply(self) -> None:
        vendor = PiVendor()
        assert vendor.interrupt_messages(session_id="s-1", state=_state()) == [{"type": "abort"}]
        assert vendor.server_request_messages(
            {"type": "extension_ui_request", "id": "ext-1", "method": "confirm"}
        ) == [{"type": "extension_ui_response", "id": "ext-1", "cancelled": True}]
        assert vendor.server_request_messages({"type": "message_update"}) == []

    def test_busy_turn_start_rejections_are_recognized(self) -> None:
        # Both are raised while the previous run has not truly ended; waiting
        # for its agent_settled and resubmitting recovers them.
        vendor = PiVendor()
        assert vendor.turn_start_busy_failure(
            "Agent is already processing. Specify streamingBehavior ('steer' or 'followUp') "
            "to queue the message."
        )
        assert vendor.turn_start_busy_failure(
            "Cannot submit a prompt while compaction is in progress. "
            "Wait for compaction to finish and retry."
        )
        assert not vendor.turn_start_busy_failure('No API key found for provider "cpo-x".')
        assert not vendor.setup_lock_contention("Agent is already processing.")

    def test_only_agent_settled_ends_the_previous_run(self) -> None:
        vendor = PiVendor()
        assert vendor.previous_run_settled({"type": "agent_settled"})
        assert not vendor.previous_run_settled(
            {"type": "message_end", "message": {"role": "assistant", "stopReason": "stop"}}
        )


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
        # The session-store part.delta projection is not the reasoning carrier
        # (model.streaming's reasoning_delta is, see the streaming test);
        # consuming both would double the trail.
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
        assert state.tokens == {"input": 5}
        assert settled.value.events[-1]["type"] == "token"
        assert settled.value.events[-1]["content"] == "Hello"

    def test_compact_boundary_emits_context_compacted_once(self) -> None:
        # Compact lifecycle events have no dedicated protocol type: the raw
        # payloads ride "session.updated". Only the boundary payload (with
        # boundaryId + summarizedMessageCount) marks the successful
        # compaction; the started/completed timeline payloads bracket it.
        vendor = ZcodeVendor()
        state = _state()
        started = {
            "operationId": "op-1",
            "messageId": "msg-9",
            "status": "started",
            "trigger": "auto",
            "display": "separator",
        }
        assert (
            vendor.parse_server_message(self._session_event("session.updated", started), state)
            == []
        )
        boundary = {
            "boundaryId": "b-1",
            "trigger": "auto",
            "preCompactTokenCount": 90000,
            "summarizedMessageCount": 24,
            "summaryMessageIds": ["msg-20"],
            "traceId": "tr-1",
        }
        assert vendor.parse_server_message(
            self._session_event("session.updated", boundary), state
        ) == [{"type": "context_compacted", "messageId": "m1"}]
        completed = {
            "operationId": "op-1",
            "messageId": "msg-9",
            "status": "completed",
            "trigger": "auto",
            "boundaryId": "b-1",
            "summaryMessageId": "msg-20",
            "replace": True,
            "postCompactTokenCount": 12000,
        }
        assert (
            vendor.parse_server_message(self._session_event("session.updated", completed), state)
            == []
        )
        # Microcompaction only clears tool results; it is not a compaction.
        microcompact = {
            "trigger": "token_pressure",
            "strategy": "local_tool_result_clear",
            "preMicrocompactTokenCount": 80000,
            "postMicrocompactTokenCount": 70000,
            "tokensSaved": 10000,
            "clearedToolCallIds": ["tc-1"],
            "keptToolCallIds": [],
            "clearedMessageCount": 1,
            "traceId": "tr-2",
        }
        assert (
            vendor.parse_server_message(self._session_event("session.updated", microcompact), state)
            == []
        )

    def test_turn_completed_usage_uses_model_usage_summary_fields(self) -> None:
        # turn.completed.usage is zcode's ModelUsageSummary: camelCase counts,
        # including cacheReadTokens/cacheWriteTokens.
        vendor = ZcodeVendor()
        state = _state()
        with pytest.raises(CliTurnSettled):
            vendor.parse_server_message(
                self._session_event(
                    "turn.completed",
                    {
                        "response": "done",
                        "usage": {
                            "source": "provider",
                            "modelRequestCount": 3,
                            "inputTokens": 120,
                            "outputTokens": 40,
                            "totalTokens": 160,
                            "cacheReadTokens": 90,
                            "cacheWriteTokens": 10,
                            "reasoningTokens": 5,
                            "webSearchRequests": 0,
                            "webFetchRequests": 0,
                        },
                    },
                ),
                state,
            )
        assert state.tokens == {
            "input": 120,
            "output": 40,
            "reasoning": 5,
            "cache": {"read": 90, "write": 10},
            "total": 160,
        }

    def test_streaming_text_and_tool_input_merge_into_the_turn(self) -> None:
        # model.streaming is the only carrier of incremental assistant text;
        # a tool's complete input arrives on its tool_call frame and is then
        # omitted from the lifecycle frame (inputOmitted/inputRef).
        vendor = ZcodeVendor()
        state = _state()
        assert vendor.parse_server_message(
            self._session_event("model.streaming", {"kind": "text_delta", "delta": "Hel"}),
            state,
        ) == [
            {"type": "step_start", "messageId": "m1", "stepId": state.step_id},
            {"type": "token", "content": "Hel", "messageId": "m1"},
        ]
        tokens = vendor.parse_server_message(
            self._session_event("model.streaming", {"kind": "text_delta", "delta": "lo"}),
            state,
        )
        assert [e["content"] for e in tokens if e["type"] == "token"] == ["Hello"]
        assert vendor.parse_server_message(
            self._session_event("model.streaming", {"kind": "reasoning_delta", "delta": "hmm"}),
            state,
        ) == [{"type": "thinking", "content": "hmm", "messageId": "m1"}]
        assert (
            vendor.parse_server_message(
                self._session_event(
                    "model.streaming",
                    {
                        "kind": "tool_call",
                        "toolCallId": "tc-1",
                        "toolName": "Write",
                        "input": {"path": "a.py"},
                    },
                ),
                state,
            )
            == []
        )
        # The scheduled frame omits the streamed input; the cached one merges.
        scheduled = vendor.parse_server_message(
            self._session_event(
                "tool.updated",
                {
                    "kind": "scheduled",
                    "toolCallId": "tc-1",
                    "toolName": "Write",
                    "inputOmitted": True,
                    "inputRef": "model_stream",
                },
            ),
            state,
        )
        tool_call = next(e for e in scheduled if e["type"] == "tool_call")
        assert tool_call["tool"] == "Write"
        assert tool_call["args"] == {"path": "a.py"}

    def test_reasoning_streams_as_thinking_across_the_turn(self) -> None:
        # model.streaming is the live reasoning carrier; the store-projection
        # part.delta channel duplicates it and is deliberately not consumed.
        vendor = ZcodeVendor()
        state = _state()
        events = vendor.parse_server_message(
            self._session_event("model.streaming", {"kind": "reasoning_delta", "delta": "Hmm"}),
            state,
        )
        assert events == [
            {"type": "step_start", "messageId": "m1", "stepId": state.step_id},
            {"type": "thinking", "content": "Hmm", "messageId": "m1"},
        ]
        events = vendor.parse_server_message(
            self._session_event(
                "model.streaming", {"kind": "reasoning_delta", "delta": ", let me"}
            ),
            state,
        )
        assert events == [{"type": "thinking", "content": "Hmm, let me", "messageId": "m1"}]
        # reasoning_start/end and empty deltas carry no text.
        assert (
            vendor.parse_server_message(
                self._session_event("model.streaming", {"kind": "reasoning_delta", "delta": ""}),
                state,
            )
            == []
        )

    def test_completed_response_does_not_duplicate_streamed_text(self) -> None:
        vendor = ZcodeVendor()
        state = _state()
        vendor.parse_server_message(
            self._session_event("model.streaming", {"kind": "text_delta", "delta": "Hello"}),
            state,
        )
        with pytest.raises(CliTurnSettled) as settled:
            vendor.parse_server_message(
                self._session_event("turn.completed", {"response": "Hello"}), state
            )
        assert [e for e in settled.value.events if e["type"] == "token"] == []
        assert state.text == "Hello"

    def test_prompt_already_running_is_recognized_as_busy(self) -> None:
        vendor = ZcodeVendor()
        assert vendor.turn_start_busy_failure("A prompt is already running for this session")
        assert not vendor.turn_start_busy_failure("Subagent sessions are read-only")
        assert not vendor.setup_lock_contention("A prompt is already running for this session")
        assert vendor.previous_run_settled(
            {"method": "state.updated", "params": {"reason": "prompt_completed"}}
        )
        assert vendor.previous_run_settled(
            {"method": "state.updated", "params": {"reason": "prompt_failed"}}
        )
        assert not vendor.previous_run_settled(
            {"method": "state.updated", "params": {"reason": "accepted"}}
        )
        assert not vendor.previous_run_settled({"method": "session/event"})

    def test_turn_failed_settles_with_the_error_event(self) -> None:
        vendor = ZcodeVendor()
        state = _state()
        with pytest.raises(CliTurnSettled) as settled:
            vendor.parse_server_message(
                self._session_event("turn.failed", {"error": {"message": "boom"}}), state
            )
        assert settled.value.events[0]["type"] == "error"

    def test_cancelled_turn_settles_as_cancelled_not_empty_output(self) -> None:
        # A user stop rides turn.completed with resultType "cancelled" and an
        # empty response (TurnError is reserved for real errors); it must not
        # fail as a turn that "completed without emitting assistant output",
        # and the stopped turn's usage still counts.
        vendor = ZcodeVendor()
        state = _state()
        with pytest.raises(CliTurnSettled) as settled:
            vendor.parse_server_message(
                self._session_event(
                    "turn.completed",
                    {
                        "response": "",
                        "resultType": "cancelled",
                        "usage": {"inputTokens": 7, "outputTokens": 0},
                    },
                ),
                state,
            )
        assert state.cancelled
        assert settled.value.events == []
        assert state.tokens == {"input": 7, "output": 0}
        outcome = vendor.exit_outcome(state, 0, "")
        assert outcome.cancelled and not outcome.success
        assert outcome.error == "Task was cancelled"

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

    def test_session_title_update_announces_the_title(self) -> None:
        # The server names the session once its first turn produced content;
        # the title also rides the session record, but only this event
        # announces it live.
        vendor = ZcodeVendor()
        state = _state()
        assert vendor.parse_server_message(
            self._session_event(
                "session.titleUpdated",
                {"title": "Fix the flaky test", "previousTitle": None, "source": "auto"},
            ),
            state,
        ) == [{"type": "session_title", "title": "Fix the flaky test"}]
        # An empty or missing title is not a title.
        assert (
            vendor.parse_server_message(
                self._session_event("session.titleUpdated", {"title": ""}), state
            )
            == []
        )
        assert vendor.parse_server_message(self._session_event("session.titleUpdated"), state) == []


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
        # --cd is an exec option, not one of the global ones: after the resume
        # subcommand clap rejects it as an unexpected argument, so every shared
        # flag precedes the subcommand and the prompt stays last.
        assert resume == [
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "--cd",
            str(WORKDIR),
            "resume",
            "thread-1",
            "--",
            "again",
        ]

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
        assert state.tokens == {"input": 3, "output": 5}

    def test_reasoning_item_adopts_its_summary_without_deltas(self) -> None:
        # The one-shot exec stream emits reasoning items without the live
        # summary deltas; the completed item's summary seeds the trail there.
        vendor = CodexVendor()
        state = _state()
        item = {"id": "r1", "type": "reasoning", "summary": ["Look", "Leap"]}
        events = vendor.parse_record({"type": "item.completed", "item": item}, state)
        assert events == [
            {"type": "step_start", "messageId": "m1", "stepId": state.step_id},
            {"type": "thinking", "content": "Look\n\nLeap", "messageId": "m1"},
        ]
        # Re-adoption must not double a trail the deltas already delivered.
        assert vendor.parse_record({"type": "item.completed", "item": item}, state) == []

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
        assert vendor.parse_record(
            {"type": "compaction_end", "result": {"summary": "trimmed"}, "aborted": False},
            state,
        ) == [{"type": "context_compacted", "messageId": "m1"}]
        assert vendor.parse_record({"type": "session_info_changed", "name": "Fix"}, state) == [
            {"type": "session_title", "title": "Fix"}
        ]

    def test_failed_or_aborted_compactions_do_not_report(self) -> None:
        # Pi emits compaction_end for failed and cancelled attempts too, with
        # `result` undefined plus an errorMessage; nothing was compacted.
        vendor = PiVendor()
        state = _state()
        assert (
            vendor.parse_record(
                {"type": "compaction_end", "aborted": True, "errorMessage": "cancelled"},
                state,
            )
            == []
        )
        assert (
            vendor.parse_record(
                {"type": "compaction_end", "reason": "overflow", "errorMessage": "server error"},
                state,
            )
            == []
        )

    def test_argv_keeps_an_at_leading_prompt_as_text(self) -> None:
        # Pi expands a positional starting with "@" into a file attachment
        # even after "--", which would turn the whole prompt into a file
        # reference; the leading space keeps it as prompt text.
        vendor = PiVendor()
        argv = vendor.build_argv(
            session_id="abc-123",
            prompt_text="@agent review this",
            model=None,
            reasoning_effort=None,
            workdir=WORKDIR,
        )
        assert argv[-2:] == ["--", " @agent review this"]
        plain = vendor.build_argv(
            session_id=None,
            prompt_text="hello",
            model=None,
            reasoning_effort=None,
            workdir=WORKDIR,
        )
        assert plain[-2:] == ["--", "hello"]

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

    def test_toolcall_message_is_not_the_final_answer(self) -> None:
        # pi keeps the run going while tools execute, so a "toolUse" message
        # must not arm the settle grace — a tool run longer than the grace
        # used to settle the turn mid-step, as "pi completed without emitting
        # assistant output" when the message carried no text.
        vendor = PiVendor()
        state = _state()
        vendor.parse_record(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "c1",
                            "name": "bash",
                            "arguments": {"command": "make"},
                        }
                    ],
                    "stopReason": "toolUse",
                },
            },
            state,
        )
        assert state.final_message_seen_at is None

        vendor.parse_record(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Done."}],
                    "stopReason": "stop",
                },
            },
            state,
        )
        assert state.final_message_seen_at is not None

    def test_a_recovered_provider_error_does_not_fail_the_turn(self) -> None:
        # pi retries a retryable provider error internally; the completed
        # response that follows proves the run continued, so the failure was
        # transient and must not fail a turn that delivered an answer.
        vendor = PiVendor()
        state = _state()
        vendor.parse_record(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [],
                    "stopReason": "error",
                    "errorMessage": "429 rate limited",
                },
            },
            state,
        )
        assert state.error == "429 rate limited"

        vendor.parse_record(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Recovered."}],
                    "stopReason": "stop",
                },
            },
            state,
        )
        assert state.error is None
        assert vendor.exit_outcome(state, 0, "").success

        # A tool-call response proves the same recovery: pi is mid-run, so
        # the earlier failure is equally stale.
        state = _state()
        vendor.parse_record(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [],
                    "stopReason": "error",
                    "errorMessage": "429 rate limited",
                },
            },
            state,
        )
        vendor.parse_record(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "id": "c1", "name": "bash", "arguments": {}}],
                    "stopReason": "toolUse",
                },
            },
            state,
        )
        assert state.error is None

    def test_length_stop_does_not_arm_the_settle_grace(self) -> None:
        # A "length" stop is recovered by pi's own overflow compaction, which
        # continues the same run (agent.continue) before it settles, so the
        # settle grace must not start there — only a plain "stop" is final.
        vendor = PiVendor()
        state = _state()
        vendor.parse_record(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Truncated."}],
                    "stopReason": "length",
                },
            },
            state,
        )
        assert state.final_message_seen_at is None

        vendor.parse_record(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Truncated, then done."}],
                    "stopReason": "stop",
                },
            },
            state,
        )
        assert state.final_message_seen_at is not None

    def test_cost_and_tokens_accumulate_across_messages(self) -> None:
        # Pi reports usage per assistant message and one turn spans several of
        # them (the tool loop), so each finished message's cost and tokens add
        # onto the turn total; an in-flight message's partial usage may only
        # be a running total on top of what is already committed.
        vendor = PiVendor()
        state = _state()
        vendor.parse_record(
            {
                "type": "message_update",
                "usage": {"cost": {"total": 0.02}},
                "assistantMessageEvent": {"type": "text_delta", "delta": "one"},
            },
            state,
        )
        assert state.cost_usd == 0.02

        vendor.parse_record(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "one"}],
                    "stopReason": "toolUse",
                    "usage": {
                        "input": 100,
                        "output": 30,
                        "cacheRead": 40,
                        "cacheWrite": 10,
                        "reasoning": 12,
                        "totalTokens": 180,
                        "cost": {
                            "input": 0.01,
                            "output": 0.005,
                            "cacheRead": 0.004,
                            "cacheWrite": 0.001,
                            "total": 0.02,
                        },
                    },
                },
            },
            state,
        )
        assert state.cost_usd == 0.02
        assert state.tokens == {
            "input": 100,
            "output": 30,
            "reasoning": 12,
            "cache": {"read": 40, "write": 10},
            "total": 180,
        }

        # The next message's partial builds on the committed total...
        vendor.parse_record(
            {
                "type": "message_update",
                "usage": {"cost": {"total": 0.015}},
                "assistantMessageEvent": {"type": "text_delta", "delta": "two"},
            },
            state,
        )
        assert state.cost_usd == pytest.approx(0.035)
        # ...and its end replaces the partial with the authoritative figures.
        vendor.parse_record(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "one two"}],
                    "stopReason": "stop",
                    "usage": {
                        "input": 50,
                        "output": 10,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "reasoning": 3,
                        "totalTokens": 60,
                        "cost": {
                            "input": 0.02,
                            "output": 0.008,
                            "cacheRead": 0.0,
                            "cacheWrite": 0.0,
                            "total": 0.03,
                        },
                    },
                },
            },
            state,
        )
        assert state.cost_usd == pytest.approx(0.05)
        assert state.tokens == {
            "input": 150,
            "output": 40,
            "reasoning": 15,
            "cache": {"read": 40, "write": 10},
            "total": 240,
        }
        # The settled step carries the whole turn in the canonical shape.
        finish = step_finish_event(state, reason="completed")
        assert finish["tokens"] == state.tokens
        assert finish["cost"] == pytest.approx(0.05)

    def test_exit_outcome(self) -> None:
        assert PiVendor().exit_outcome(_state(), 0, "") == TurnOutcome.failed(
            "pi completed without emitting assistant output."
        )
        state = _state()
        state.text = "hi"
        assert PiVendor().exit_outcome(state, 0, "").success


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
