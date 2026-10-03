"""Vendor descriptions for the generic CLI harness (see ``cli_harness.py``).

Each class encodes one vendor's non-interactive contract: the argv for a turn,
the stdout record shapes, and how the process exit maps to a turn outcome —
plus, for vendors with a resident protocol server, the request/notification
dialect that contract's resident half speaks. They are deliberately small and
side-effect free so the translation can be unit-tested against synthetic
records without the vendor installed.

Session continuity differs by vendor:

- One-shot turns: Codex creates the conversation itself (the id rides the
  first stdout record), Pi accepts a client-chosen ``--session-id``, and
  ZCode's CLI exposes no resume switch (every turn is a fresh conversation).
- Resident servers: Codex threads and ZCode sessions are created server-side
  and resumed by id after a restart; Pi's session id is chosen up front at
  spawn (``--session-id``) and confirmed by ``get_state``.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

from ..custom_providers import (
    codex_model_catalog_path,
    find_provider_for_model,
    load_custom_providers,
    write_codex_model_catalog,
    write_codex_model_providers,
    write_pi_models_json,
    write_zcode_model_selection,
    write_zcode_provider_config,
    zcode_model_selection,
    zcode_provider_config_path,
)
from .base import HarnessId, TurnOutcome
from .cli_harness import (
    CliTurnSettled,
    CliTurnState,
    append_text_events,
    error_event,
    step_start_events,
    text_events,
    tool_events,
)

if TYPE_CHECKING:
    from ..custom_providers import CustomProvider


def _bare_model(model: str | None) -> str | None:
    if not model:
        return None
    return model.split("/", 1)[1] if "/" in model else model


def _pi_thinking_level(effort: str) -> str:
    # Pi's level for disabling thinking is "off", not the registry's "none".
    return "off" if effort == "none" else effort


def _jsonrpc_unattended_reply_error(request_id: str | int, method: str) -> dict[str, Any]:
    # Answering with a protocol error releases the server's pending request
    # instead of hanging the turn on input no interactive client will give.
    return {
        "id": request_id,
        "error": {"code": -32601, "message": f"No client available for {method}"},
    }


def _is_request_id(value: Any) -> TypeGuard[str | int]:
    # JSON-RPC ids are strings or integers; codex's app-server numbers
    # server→client requests with integers (0, 1, ...).
    return (isinstance(value, str) and value != "") or (
        isinstance(value, int) and not isinstance(value, bool)
    )


def _jsonrpc_unattended_reply(message: dict[str, Any]) -> list[dict[str, Any]]:
    request_id = message.get("id")
    method = message.get("method")
    if not (_is_request_id(request_id) and isinstance(method, str)):
        return []
    return [_jsonrpc_unattended_reply_error(request_id, method)]


def _result_text(result: Any) -> str:
    """Best-effort text from a tool result object of unknown shape."""
    if isinstance(result, dict):
        for key in ("content", "text", "output", "summary"):
            value = result.get(key)
            if isinstance(value, str):
                return value
        return ""
    return _as_text(result)


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return ""


# App-server item types are camelCase; the exec stream and the translator
# below speak snake_case.
_CODEX_ITEM_TYPES = {
    "userMessage": "user_message",
    "agentMessage": "agent_message",
    "reasoning": "reasoning",
    "webSearch": "web_search",
    "commandExecution": "command_execution",
    "fileChange": "file_change",
    "mcpToolCall": "mcp_tool_call",
    "error": "error",
    # Emitted when Codex compacts the thread's history (auto or thread/compact).
    "contextCompaction": "context_compaction",
}

# App-server item payload fields are camelCase where the exec records the
# translator reads are snake_case. The command item is the only type whose
# fields differ; MCP ``arguments``/``result`` hold arbitrary user JSON whose
# nested keys must survive untouched, so no generic key conversion applies.
_CODEX_ITEM_FIELD_RENAMES = {
    "command_execution": {
        "aggregatedOutput": "aggregated_output",
        "exitCode": "exit_code",
        "commandActions": "command_actions",
        "processId": "process_id",
        "durationMs": "duration_ms",
    },
}


class CodexVendor:
    """OpenAI Codex CLI driven through its resident app-server.

    ``codex app-server`` speaks JSON-RPC over stdio: one ``initialize``, a
    ``thread/start`` (or ``thread/resume`` after a restart) carrying the
    routed model provider, and a ``turn/start`` per turn whose
    ``turn/completed``/``turn/failed`` notification settles it. The item and
    turn notifications carry the same payloads the ``codex exec --json``
    records did, so translation reuses the one-shot parser behind a thin
    envelope adapter. The one-shot argv path stays for the
    ``OI_CLI_ONE_SHOT=codex`` escape hatch.
    """

    id = HarnessId.CODEX
    binary = "codex"
    json_stream = True
    resident = True
    jsonrpc = True  # the app-server speaks JSON-RPC 2.0 proper

    def __init__(self) -> None:
        # The thread id this server process created; a persisted id adopted
        # from a previous process is resumed lazily on its first turn.
        self._live_thread_id: str | None = None

    def initial_session_id(self) -> str | None:
        return None

    def build_argv(
        self,
        *,
        session_id: str | None,
        prompt_text: str,
        model: str | None,
        reasoning_effort: str | None,
        workdir: Path,
        model_provider: str | None = None,
    ) -> list[str]:
        # ``--cd`` belongs to the exec options, not the global set: after the
        # ``resume`` subcommand clap rejects it as an unexpected argument, so
        # every shared flag stays before the subcommand.
        argv = [
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "--cd",
            str(workdir),
        ]
        if session_id:
            argv += ["resume", session_id]
        if model_provider:
            argv += ["-c", f"model_provider={model_provider}"]
            # The catalog carries the routed model's metadata (context window,
            # effort levels); without it the CLI falls back to hardcoded
            # defaults and warns every turn. Passed per turn so sessions on
            # official models keep the CLI's own bundled catalog.
            argv += ["-c", f"model_catalog_json={codex_model_catalog_path()}"]
        bare = _bare_model(model)
        if bare:
            argv += ["--model", bare]
        if reasoning_effort:
            argv += ["-c", f"model_reasoning_effort={reasoning_effort}"]
        argv += ["--", prompt_text]
        return argv

    def extra_env(self, *, model: str | None) -> dict[str, str]:
        return {}

    def prepare(self, custom_providers: tuple[CustomProvider, ...]) -> None:
        """Register OpenAI-protocol gateways in the CLI's ``config.toml``."""
        write_codex_model_providers(Path.home() / ".codex" / "config.toml", custom_providers)
        write_codex_model_catalog(codex_model_catalog_path(), custom_providers)

    # --- Resident app-server protocol -------------------------------------

    def server_argv(
        self,
        *,
        session_id: str | None,
        model: str | None,
        reasoning_effort: str | None,
    ) -> list[str]:
        return ["app-server"]

    def reset(self) -> None:
        # A fresh app-server knows no thread: a persisted id resumes lazily.
        self._live_thread_id = None

    def handshake_requests(self) -> list[dict[str, Any]]:
        return [
            {
                "method": "initialize",
                "params": {"clientInfo": {"name": "open-inspect", "version": "0"}},
            }
        ]

    def next_setup_message(
        self,
        *,
        session_id: str | None,
        model: str | None,
        reasoning_effort: str | None,
        workdir: Path,
        model_provider: str | None = None,
    ) -> dict[str, Any] | None:
        """``thread/start`` for a new conversation, ``thread/resume`` for one
        adopted from a previous server process, nothing while the thread this
        process created is still live."""
        if session_id and session_id == self._live_thread_id:
            return None
        bare = _bare_model(model)
        # Unattended, the config twin of the exec path's
        # --dangerously-bypass-approvals-and-sandbox: without it the
        # app-server builds the thread with its defaults (read-only sandbox,
        # approvals on request), so edits block on an approval nobody answers
        # and escalated commands are denied.
        config: dict[str, Any] = {
            "sandbox_mode": "danger-full-access",
            "approval_policy": "never",
        }
        params: dict[str, Any] = {"cwd": str(workdir)}
        if model_provider:
            params["modelProvider"] = model_provider
            # Same per-turn catalog the exec path passes as -c: the routed
            # model's metadata, so the CLI does not fall back and warn.
            config["model_catalog_json"] = str(codex_model_catalog_path())
        params["config"] = config
        if bare:
            params["model"] = bare
        if session_id:
            return {"method": "thread/resume", "params": params | {"threadId": session_id}}
        return {"method": "thread/start", "params": params}

    def turn_start_message(
        self,
        *,
        session_id: str | None,
        prompt_text: str,
        model: str | None,
        reasoning_effort: str | None,
        model_provider: str | None = None,
    ) -> dict[str, Any] | None:
        if not session_id:
            return None
        params: dict[str, Any] = {
            "threadId": session_id,
            "input": [{"type": "text", "text": prompt_text}],
        }
        bare = _bare_model(model)
        if bare:
            params["model"] = bare
        if reasoning_effort:
            params["effort"] = reasoning_effort
        return {"method": "turn/start", "params": params}

    def adopt_response_id(self, response: dict[str, Any]) -> str | None:
        result = response.get("result")
        thread = result.get("thread") if isinstance(result, dict) else None
        thread_id = thread.get("id") if isinstance(thread, dict) else None
        if isinstance(thread_id, str) and thread_id:
            self._live_thread_id = thread_id
            return thread_id
        return None

    def parse_server_message(self, message: dict[str, Any], state: CliTurnState) -> list[Any]:
        method = message.get("method")
        if not isinstance(method, str):
            return []
        if method == "thread/tokenUsage/updated":
            # The app-server carries usage only here (the v2 turn has no usage
            # field); keep the thread total the exec stream reports at its
            # turn.completed so both paths record the same figure.
            usage = message.get("params", {}).get("tokenUsage")
            total = usage.get("total") if isinstance(usage, dict) else None
            if isinstance(total, dict):
                state.tokens = _usage_tokens(total)
            return []
        record = self._exec_record(message)
        if method == "turn/started":
            turn = message.get("params", {}).get("turn", {})
            if isinstance(turn, dict) and isinstance(turn.get("id"), str):
                state.server_turn_id = turn["id"]
            return self.parse_record(record, state)
        if method == "turn/completed":
            events = self.parse_record(record, state)
            raise CliTurnSettled(events)
        if method == "turn/failed":
            events = self.parse_record(record, state)
            raise CliTurnSettled(events)
        return self.parse_record(record, state)

    def interrupt_messages(
        self, *, session_id: str | None, state: CliTurnState | None
    ) -> list[dict[str, Any]]:
        thread_id = session_id or self._live_thread_id
        if not thread_id or state is None or not state.server_turn_id:
            return []
        return [
            {
                "method": "turn/interrupt",
                "params": {"threadId": thread_id, "turnId": state.server_turn_id},
            }
        ]

    def server_request_messages(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        return _jsonrpc_unattended_reply(message)

    # --- Failure recovery ---------------------------------------------------

    def setup_lock_contention(self, failure: str) -> bool:
        # A per-thread rollout writer lock is held for the thread's live
        # lifetime in one process; this rejection means another app-server
        # (an orphan this bridge never reaped) still owns the thread.
        return "already has an active writer" in failure

    def turn_start_busy_failure(self, failure: str) -> bool:
        return False

    def previous_run_settled(self, message: dict[str, Any]) -> bool:
        return False

    def _exec_record(self, message: dict[str, Any]) -> dict[str, Any]:
        """Adapt one JSON-RPC notification to the ``exec --json`` record shape
        the one-shot parser consumes."""
        method = message.get("method", "")
        params = message.get("params", {})
        record: dict[str, Any] = {"type": method.replace("/", ".")}
        item = params.get("item")
        if isinstance(item, dict):
            adapted = dict(item)
            item_type = adapted.get("type")
            if isinstance(item_type, str):
                item_type = _CODEX_ITEM_TYPES.get(item_type, item_type)
                adapted["type"] = item_type
                for field, renamed in _CODEX_ITEM_FIELD_RENAMES.get(item_type, {}).items():
                    if field in adapted:
                        adapted[renamed] = adapted.pop(field)
            record["item"] = adapted
        turn = params.get("turn")
        if isinstance(turn, dict):
            record["turn"] = turn
            usage = turn.get("usage")
            if isinstance(usage, dict):
                record["usage"] = usage
            error = turn.get("error")
            if isinstance(error, dict) and error:
                record["error"] = error
        error = params.get("error")
        if isinstance(error, dict) and error:
            record["error"] = error
        return record

    def parse_record(self, record: dict[str, Any], state: CliTurnState) -> list[Any]:
        kind = record.get("type")
        if kind == "thread.started":
            thread_id = record.get("thread_id") or record.get("threadId")
            if isinstance(thread_id, str) and thread_id:
                state.session_id = thread_id
            return []
        if kind in ("item.started", "item.completed"):
            item = record.get("item")
            if not isinstance(item, dict):
                return []
            events = self._tool_events(item, state, running=kind == "item.started")
            if kind == "item.completed" and item.get("type") == "agent_message":
                key = str(item.get("id") or len(state.agent_texts))
                state.agent_texts[key] = _as_text(item.get("text"))
                events = [
                    *text_events(state, "".join(state.agent_texts.values())),
                    *events,
                ]
            if kind == "item.completed" and item.get("type") == "error":
                message = _error_text(item)
                if message:
                    events.append({"type": "warning", "scope": "provider", "message": message})
            if kind == "item.completed" and item.get("type") == "context_compaction":
                # Completed only: started would double every boundary.
                events.append({"type": "context_compacted", "messageId": state.message_id})
            return events
        if kind == "turn.completed":
            # The app-server reports every terminal turn through this one
            # notification, distinguished by the turn's status.
            turn = record.get("turn")
            if isinstance(turn, dict):
                if turn.get("status") == "interrupted":
                    # The turn was aborted (a stop the client asked for, or
                    # codex's own cancellation); it settles as cancelled so
                    # the stop is not reported as a completed turn.
                    state.cancelled = True
                    return step_start_events(state)
                error = turn.get("error")
                if (
                    isinstance(error, dict)
                    and isinstance(error.get("message"), str)
                    and error["message"]
                ):
                    return [
                        *step_start_events(state),
                        *error_event(state, error["message"]),
                    ]
                if turn.get("status") == "failed":
                    # Failed with no message of its own; still a failure.
                    return [
                        *step_start_events(state),
                        *error_event(state, "Codex turn failed"),
                    ]
            usage = record.get("usage")
            if isinstance(usage, dict):
                state.tokens = _usage_tokens(usage)
            state.completed = True
            return step_start_events(state)
        if kind == "turn.failed":
            return error_event(state, _error_text(record.get("error")) or "Codex turn failed")
        if kind == "error":
            # Codex streams transient `error` records while it reconnects; only
            # `turn.failed` (or a non-zero exit) ends the turn. Reconnect noise
            # is dropped, anything else surfaces as a provider warning.
            message = _error_text(record)
            if not message or message.startswith("Reconnecting"):
                return []
            return [{"type": "warning", "scope": "provider", "message": message}]
        return []

    def _tool_events(
        self, item: dict[str, Any], state: CliTurnState, *, running: bool
    ) -> list[Any]:
        item_type = item.get("type")
        call_id = str(item.get("id") or item.get("call_id") or "tool")
        status = "running" if running else _codex_status(item.get("status"))
        if item_type == "command_execution":
            output = _as_text(item.get("aggregated_output") or item.get("output"))
            return tool_events(
                state,
                call_id=call_id,
                name="bash",
                args={"command": item.get("command")},
                status=status,
                output=output,
            )
        if item_type == "file_change":
            return tool_events(
                state,
                call_id=call_id,
                name="edit",
                args={"changes": item.get("changes")},
                status=status,
                output=_as_text(item.get("output")),
            )
        if item_type == "mcp_tool_call":
            return tool_events(
                state,
                call_id=call_id,
                name=str(item.get("tool") or "mcp"),
                args=item.get("arguments") if isinstance(item.get("arguments"), dict) else {},
                status=status,
                output=_as_text(item.get("result")),
            )
        return []

    def exit_outcome(
        self, state: CliTurnState, returncode: int | None, stderr_tail: str
    ) -> TurnOutcome:
        return _default_exit_outcome("codex", state, returncode, stderr_tail)


class PiVendor:
    """Pi coding agent: one-shot ``pi --mode json`` turns, or the resident
    ``pi --mode rpc`` server.

    The rpc mode runs Pi as a long-lived subprocess speaking JSON records on
    stdin/stdout — the same ``toJsonEvent`` stream the json mode prints, so
    record translation is shared. Commands are ``{type, ...}`` frames with an
    optional id (no JSON-RPC envelope), a ``prompt`` command only acknowledges
    (the turn settles on the ``agent_settled`` event), and the model rides the
    spawn argv plus ``set_model``/``set_thinking_level`` commands.
    """

    id = HarnessId.PI
    binary = "pi"
    json_stream = True
    resident = True
    jsonrpc = False
    # A terminal assistant message is authoritative on its own; agent_settled
    # only confirms pi stopped its post-turn bookkeeping, which has been
    # observed to hang. Settle shortly after the answer instead of spending
    # the whole inactivity budget waiting for the confirmation.
    settle_after_final_message = 60.0

    def __init__(self) -> None:
        # The rpc server is probed once per process; the model and thinking
        # level are re-applied only on change, so setup converges in a pass.
        self._probed = False
        self._applied_model: str | None = None
        self._applied_thinking: str | None = None

    def initial_session_id(self) -> str | None:
        # Pi accepts ``--session-id`` as an exact id, creating it when absent,
        # so both one-shot children and resident spawns carry a chosen id.
        return str(uuid.uuid4())

    def build_argv(
        self,
        *,
        session_id: str | None,
        prompt_text: str,
        model: str | None,
        reasoning_effort: str | None,
        workdir: Path,
        model_provider: str | None = None,
    ) -> list[str]:
        argv = ["--mode", "json", "--approve"]
        if session_id:
            argv += ["--session-id", session_id]
        if model:
            argv += ["--model", model]
        if reasoning_effort:
            argv += ["--thinking", _pi_thinking_level(reasoning_effort)]
        argv += ["--", prompt_text]
        return argv

    def extra_env(self, *, model: str | None) -> dict[str, str]:
        return {}

    def prepare(self, custom_providers: tuple[CustomProvider, ...]) -> None:
        """Register the providers in the Pi CLI's ``models.json``.

        Pi supports every wire protocol a provider can carry, and its model
        ids are ``{provider}/{model}`` — exactly the selection string the
        bridge already passes, so no argv routing is needed.
        """
        write_pi_models_json(Path.home() / ".pi" / "agent" / "models.json", custom_providers)

    # --- Resident rpc protocol ----------------------------------------------

    def server_argv(
        self,
        *,
        session_id: str | None,
        model: str | None,
        reasoning_effort: str | None,
    ) -> list[str]:
        # A non-interactive spawn must resolve a model or the CLI exits at
        # startup, so the first turn's selection rides the argv; later
        # changes go through set_model in the setup chain.
        argv = ["--mode", "rpc", "--approve"]
        if session_id:
            argv += ["--session-id", session_id]
        if model:
            argv += ["--model", model]
        if reasoning_effort:
            argv += ["--thinking", _pi_thinking_level(reasoning_effort)]
        return argv

    def reset(self) -> None:
        self._probed = False
        self._applied_model = None
        self._applied_thinking = None

    def handshake_requests(self) -> list[dict[str, Any]]:
        # No initialize in this protocol; get_state on the first turn doubles
        # as the readiness probe.
        return []

    def next_setup_message(
        self,
        *,
        session_id: str | None,
        model: str | None,
        reasoning_effort: str | None,
        workdir: Path,
        model_provider: str | None = None,
    ) -> dict[str, Any] | None:
        """``get_state`` to adopt the session id, then the model and thinking
        level applied on change — the protocol seam for what argv routing does
        in one-shot mode."""
        if not self._probed:
            self._probed = True
            return {"type": "get_state"}
        if model and model != self._applied_model:
            self._applied_model = model
            provider, _, model_id = model.partition("/")
            if provider and model_id:
                return {"type": "set_model", "provider": provider, "modelId": model_id}
        if reasoning_effort and reasoning_effort != self._applied_thinking:
            self._applied_thinking = reasoning_effort
            return {"type": "set_thinking_level", "level": _pi_thinking_level(reasoning_effort)}
        return None

    def turn_start_message(
        self,
        *,
        session_id: str | None,
        prompt_text: str,
        model: str | None,
        reasoning_effort: str | None,
        model_provider: str | None = None,
    ) -> dict[str, Any] | None:
        if not session_id:
            return None
        # The response only acknowledges the prompt; the turn settles on
        # agent_settled.
        return {"type": "prompt", "message": prompt_text}

    def adopt_response_id(self, response: dict[str, Any]) -> str | None:
        if response.get("type") != "response" or response.get("command") != "get_state":
            return None
        data = response.get("data")
        session_id = data.get("sessionId") if isinstance(data, dict) else None
        return session_id if isinstance(session_id, str) and session_id else None

    def parse_server_message(self, message: dict[str, Any], state: CliTurnState) -> list[Any]:
        # Command responses ride the same stream; only events translate.
        if message.get("type") == "response":
            return []
        events = self.parse_record(message, state)
        if message.get("type") == "agent_settled":
            # Settled is the terminal: Pi has stopped retrying, compacting,
            # and draining queued messages, not just ended one agent run.
            raise CliTurnSettled(events)
        return events

    def interrupt_messages(
        self, *, session_id: str | None, state: CliTurnState | None
    ) -> list[dict[str, Any]]:
        return [{"type": "abort"}]

    def server_request_messages(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        # Extension UI prompts are answered as cancelled: nobody is watching.
        if message.get("type") == "extension_ui_request" and isinstance(message.get("id"), str):
            return [{"type": "extension_ui_response", "id": message["id"], "cancelled": True}]
        return []

    # --- Failure recovery ---------------------------------------------------

    def setup_lock_contention(self, failure: str) -> bool:
        return False

    def turn_start_busy_failure(self, failure: str) -> bool:
        # Raised while the previous run has not truly ended: it is still
        # streaming, or compacting inside the run.
        return "already processing" in failure or "compaction is in progress" in failure

    def previous_run_settled(self, message: dict[str, Any]) -> bool:
        # agent_settled is pi's true end of run: emitted only once the run's
        # retries, compactions, and queued messages have all drained.
        return message.get("type") == "agent_settled"

    def parse_record(self, record: dict[str, Any], state: CliTurnState) -> list[Any]:
        kind = record.get("type")
        if kind == "session":
            session_id = record.get("id")
            if isinstance(session_id, str) and session_id:
                state.session_id = session_id
            return []
        if kind == "message_update":
            return self._message_update(record, state)
        if kind == "message_end":
            return self._message_end(record, state)
        if kind == "tool_execution_start":
            return tool_events(
                state,
                call_id=str(record.get("toolCallId") or "tool"),
                name=str(record.get("toolName") or "tool"),
                args=record.get("args") if isinstance(record.get("args"), dict) else {},
                status="running",
            )
        if kind == "tool_execution_end":
            return tool_events(
                state,
                call_id=str(record.get("toolCallId") or "tool"),
                name=str(record.get("toolName") or "tool"),
                args=None,
                status="error" if record.get("isError") else "completed",
                output=_pi_result_text(record.get("result")),
            )
        if kind == "compaction_end":
            return [{"type": "context_compacted", "messageId": state.message_id}]
        if kind == "session_info_changed":
            name = record.get("name")
            if isinstance(name, str) and name:
                return [{"type": "session_title", "title": name}]
            return []
        if kind == "auto_retry_end" and record.get("success") is False:
            return error_event(state, str(record.get("finalError") or "Pi retries exhausted"))
        if kind == "agent_settled":
            state.completed = True
            return step_start_events(state)
        return []

    def _message_update(self, record: dict[str, Any], state: CliTurnState) -> list[Any]:
        usage = record.get("usage")
        if isinstance(usage, dict):
            cost = _pi_message_cost(usage)
            if cost is not None:
                # Pi's usage is per message, so the running total builds on the
                # messages that already finished; the commit happens on the
                # message's end, which is authoritative for it.
                state.cost_usd = state.cost_committed_usd + cost
        event = record.get("assistantMessageEvent")
        if not isinstance(event, dict):
            return []
        event_type = event.get("type")
        if event_type == "text_delta":
            delta = event.get("delta")
            return append_text_events(state, delta) if isinstance(delta, str) else []
        if event_type == "text_end":
            content = event.get("content")
            return text_events(state, content) if isinstance(content, str) else []
        if event_type == "toolcall_end":
            call = event.get("toolCall")
            if not isinstance(call, dict):
                return []
            return tool_events(
                state,
                call_id=str(call.get("id") or "tool"),
                name=str(call.get("name") or "tool"),
                args=call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
                status="running",
            )
        if event_type == "error":
            return error_event(state, str(event.get("error") or "Pi reported an error"))
        return []

    def _message_end(self, record: dict[str, Any], state: CliTurnState) -> list[Any]:
        """The authoritative assistant message; also the only failure signal.

        Pi exits 0 even when the provider rejected the turn, so an assistant
        message with ``stopReason: "error"`` is what fails it here — but only
        if no later response shows pi got past it. The final text replaces the
        accumulated deltas only when it is longer, so an earlier assistant
        message's text is never lost.
        """
        message = record.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            return []
        events: list[Any] = []
        stop_reason = message.get("stopReason")
        if stop_reason == "error":
            detail = str(message.get("errorMessage") or "Pi reported a provider error")
            events += error_event(state, detail)
        elif stop_reason not in (None, "aborted", "pending"):
            # Only a plain "stop" is the final answer. A "length" stop is
            # recovered by pi's own overflow compaction, which continues the
            # same run before it settles, and a "toolUse" message leaves tools
            # to execute — neither may start the settle grace, or the turn
            # settles before the answer exists.
            if stop_reason == "stop" and not _pi_has_tool_calls(message):
                state.final_message_seen_at = time.monotonic()
            if state.error:
                # Any completed response after a failed one means pi recovered
                # on its own — an internal retry or overflow compaction
                # continued the run — so the transient failure must not fail
                # a turn that delivered an answer.
                state.error = None
                state.emitted_error = False
        final_text = _as_text(message.get("content"))
        if final_text and len(final_text) > len(state.text):
            events += text_events(state, final_text)
        usage = message.get("usage")
        if isinstance(usage, dict):
            # Pi reports usage per assistant message and one turn spans several
            # of them (the tool loop), so this message's cost and tokens are
            # added onto the turn total rather than replacing it.
            cost = _pi_message_cost(usage)
            if cost is not None:
                state.cost_committed_usd += cost
                state.cost_usd = state.cost_committed_usd
            tokens = _usage_tokens(usage)
            if tokens:
                state.tokens = _merged_usage_tokens(state.tokens, tokens)
        return events

    def exit_outcome(
        self, state: CliTurnState, returncode: int | None, stderr_tail: str
    ) -> TurnOutcome:
        return _default_exit_outcome("pi", state, returncode, stderr_tail)


class ZcodeVendor:
    """ZCode CLI: one-shot ``zcode --prompt`` turns, or the resident
    ``app-server``.

    The resident path is the same channel the desktop app drives the CLI
    with: ``zcode app-server --stdio`` speaking NDJSON frames —
    ``{id, method, params}`` requests with no JSON-RPC envelope field (the
    wire schema rejects unknown keys) — a session created per conversation
    (server-chosen id; a client-chosen one is refused outside imported
    history), event delivery turned on by ``session/subscribe``, turns
    submitted with ``session/send`` and observed through ``session/event``
    notifications until ``turn.completed``/``turn.failed``.

    One-shot turns remain plain text: ``--prompt`` runs one headless turn
    (permission mode defaults to ``yolo`` under it, so no approvals) and
    carries no session id on the stream, so every one-shot turn is a fresh
    conversation. Custom providers ride the CLI's personal provider config —
    ZCode offers no environment seam for endpoints or keys, so unlike the
    other vendors the api-key rides the file as a literal; one-shot turns
    rewrite its ``defaultModelSelection`` before each spawn while resident
    turns pass the selection per ``session/send``.
    """

    id = HarnessId.ZCODE
    binary = "zcode"
    json_stream = False
    resident = True
    # The wire schema is strict: a jsonrpc envelope field would see every
    # request rejected with -32600 and no correlatable response.
    jsonrpc = False

    def __init__(self) -> None:
        self._live_session_id: str | None = None
        self._subscribed_session: str | None = None

    def initial_session_id(self) -> str | None:
        return None

    def build_argv(
        self,
        *,
        session_id: str | None,
        prompt_text: str,
        model: str | None,
        reasoning_effort: str | None,
        workdir: Path,
        model_provider: str | None = None,
    ) -> list[str]:
        argv = ["--prompt", prompt_text]
        if model_provider and model:
            write_zcode_model_selection(
                zcode_provider_config_path(), load_custom_providers(), model, reasoning_effort
            )
        if session_id:
            argv += ["--resume", session_id]
        return argv

    def extra_env(self, *, model: str | None) -> dict[str, str]:
        return {}

    def prepare(self, custom_providers: tuple[CustomProvider, ...]) -> None:
        """Register the providers in the CLI's personal provider config."""
        write_zcode_provider_config(zcode_provider_config_path(), custom_providers)

    # --- Resident app-server protocol -------------------------------------

    def server_argv(
        self,
        *,
        session_id: str | None,
        model: str | None,
        reasoning_effort: str | None,
    ) -> list[str]:
        return ["app-server", "--stdio"]

    def reset(self) -> None:
        self._live_session_id = None
        self._subscribed_session = None

    def handshake_requests(self) -> list[dict[str, Any]]:
        return []

    def next_setup_message(
        self,
        *,
        session_id: str | None,
        model: str | None,
        reasoning_effort: str | None,
        workdir: Path,
        model_provider: str | None = None,
    ) -> dict[str, Any] | None:
        """``session/create`` for a new conversation, ``session/resume`` for
        one adopted from a previous server process, then
        ``session/subscribe`` — delivery only flows to subscribed sessions,
        and the id to subscribe to is only known after the create/resume
        response, hence the pull-based chain."""
        if session_id is None and self._live_session_id is None:
            return {
                "method": "session/create",
                "params": {
                    "workspace": {"workspacePath": str(workdir), "workspaceKey": str(workdir)},
                    "mode": "yolo",
                },
            }
        if session_id is not None and session_id != self._live_session_id:
            return {"method": "session/resume", "params": {"sessionId": session_id}}
        if session_id is not None and self._subscribed_session != session_id:
            # The delivery kind the desktop host uses: live, no replay.
            self._subscribed_session = session_id
            return {
                "method": "session/subscribe",
                "params": {
                    "sessionId": session_id,
                    "deliveryKind": "desktop-continuous",
                    "includeSnapshot": False,
                },
            }
        return None

    def turn_start_message(
        self,
        *,
        session_id: str | None,
        prompt_text: str,
        model: str | None,
        reasoning_effort: str | None,
        model_provider: str | None = None,
    ) -> dict[str, Any] | None:
        if not session_id:
            return None
        params: dict[str, Any] = {"sessionId": session_id, "content": prompt_text}
        if model_provider:
            # The routed gateway rides the per-turn selection — the protocol
            # seam for what the one-shot path writes into defaultModelSelection.
            resolved = find_provider_for_model(model or "", load_custom_providers())
            if resolved is not None:
                params["modelSelection"] = zcode_model_selection(*resolved, reasoning_effort)
        return {"method": "session/send", "params": params}

    def adopt_response_id(self, response: dict[str, Any]) -> str | None:
        # create and resume both answer with a state snapshot.
        result = response.get("result")
        snapshot = result.get("session") if isinstance(result, dict) else None
        session_id = snapshot.get("sessionId") if isinstance(snapshot, dict) else None
        if isinstance(session_id, str) and session_id:
            self._live_session_id = session_id
            return session_id
        return None

    def parse_server_message(self, message: dict[str, Any], state: CliTurnState) -> list[Any]:
        if message.get("method") != "session/event":
            return []
        event = message.get("params")
        if not isinstance(event, dict):
            return []
        kind = event.get("type")
        raw_payload = event.get("payload")
        payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
        if kind == "part.delta":
            if payload.get("field") in (None, "text") and isinstance(payload.get("delta"), str):
                return append_text_events(state, payload["delta"])
            return []
        if kind == "part.upserted":
            # The compact boundary rides the message stream as a "compaction"
            # part; upserted only, so a started+upserted pair emits once.
            part = payload.get("part")
            if isinstance(part, dict) and part.get("type") == "compaction":
                return [{"type": "context_compacted", "messageId": state.message_id}]
            return []
        if kind == "tool.updated":
            return self._tool_updated_events(payload, state)
        if kind == "model.streaming":
            return self._streaming_events(payload, state)
        if kind == "turn.completed":
            response = str(payload.get("response") or "")
            # The response is the last step's full text: the live stream
            # usually delivered it already (keep), it may extend a streamed
            # prefix (replace), and only when the stream missed it entirely
            # is it appended to earlier steps' text.
            events: list[Any] = []
            if response and not state.text.endswith(response):
                events = (
                    text_events(state, response)
                    if response.startswith(state.text)
                    else append_text_events(state, response)
                )
            usage = payload.get("usage")
            if isinstance(usage, dict):
                state.tokens = _usage_tokens(usage)
            state.completed = True
            raise CliTurnSettled(events)
        if kind == "turn.failed":
            detail = _error_text(payload.get("error")) or "ZCode turn failed"
            raise CliTurnSettled(error_event(state, detail))
        return []

    def _streaming_events(self, payload: dict[str, Any], state: CliTurnState) -> list[Any]:
        """The model's live stream: the only carrier of incremental assistant
        text, and of each tool's complete input.

        A tool's full input arrives once on its ``tool_call`` frame; the later
        lifecycle frame (``tool.updated`` scheduled) then omits it
        (``inputOmitted``/``inputRef == "model_stream"``), so the input is
        cached here for that merge.
        """
        stream_kind = payload.get("kind")
        if stream_kind == "text_delta":
            delta = payload.get("delta")
            if isinstance(delta, str) and delta:
                return append_text_events(state, delta)
            return []
        if stream_kind == "tool_call":
            call_id = payload.get("toolCallId")
            tool_input = payload.get("input")
            if isinstance(call_id, str) and call_id and isinstance(tool_input, dict):
                state.tool_args[call_id] = tool_input
        return []

    def _tool_updated_events(self, payload: dict[str, Any], state: CliTurnState) -> list[Any]:
        call_id = str(payload.get("toolCallId") or "tool")
        kind = payload.get("kind")
        if kind in ("scheduled", "started"):
            args = payload.get("input")
            if not isinstance(args, dict):
                # The frame omitted the input because the stream already
                # carried it; the cached value is the same one.
                args = state.tool_args.get(call_id)
            return tool_events(
                state,
                call_id=call_id,
                name=str(payload.get("toolName") or "tool"),
                args=args if isinstance(args, dict) else {},
                status="running",
            )
        if kind == "result":
            return tool_events(
                state,
                call_id=call_id,
                name="",
                args=None,
                status="completed",
                output=_result_text(payload.get("result")),
            )
        if kind == "error":
            return tool_events(
                state,
                call_id=call_id,
                name="",
                args=None,
                status="error",
                output=_error_text(payload.get("error")),
            )
        return []

    def interrupt_messages(
        self, *, session_id: str | None, state: CliTurnState | None
    ) -> list[dict[str, Any]]:
        if not session_id:
            return []
        return [{"method": "session/stop", "params": {"sessionId": session_id}}]

    def server_request_messages(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        request_id = message.get("id")
        method = message.get("method")
        if not (_is_request_id(request_id) and isinstance(method, str)):
            return []
        if method == "interaction/requestPermission":
            # Unattended: deny so the turn continues past the refused tool —
            # the same answer ZCode's own headless broker gives.
            return [{"id": request_id, "result": {"decision": "deny"}}]
        if method == "interaction/requestUserInput":
            # A schema-valid cancel declines the question gracefully (the
            # broker maps it to a deny the model sees and works around); an
            # error frame would instead reject the broker's promise.
            return [{"id": request_id, "result": {"action": "cancel"}}]
        return [_jsonrpc_unattended_reply_error(request_id, method)]

    # --- Failure recovery ---------------------------------------------------

    def setup_lock_contention(self, failure: str) -> bool:
        return False

    def turn_start_busy_failure(self, failure: str) -> bool:
        # -32010: the previous turn's worker still holds the session's active
        # controller, which it clears only after the terminal turn event; a
        # send inside that window is rejected until the run truly ends.
        return "already running" in failure

    def previous_run_settled(self, message: dict[str, Any]) -> bool:
        # prompt_completed/prompt_failed are the vendor's own "ready"
        # boundary: the active controller is released before they broadcast.
        if message.get("method") != "state.updated":
            return False
        params = message.get("params")
        reason = params.get("reason") if isinstance(params, dict) else None
        return reason in ("prompt_completed", "prompt_failed")

    def parse_record(self, record: dict[str, Any], state: CliTurnState) -> list[Any]:
        return []

    def exit_outcome(
        self, state: CliTurnState, returncode: int | None, stderr_tail: str
    ) -> TurnOutcome:
        return _default_exit_outcome("zcode", state, returncode, stderr_tail)


_VENDORS: dict[HarnessId, type[Any]] = {
    HarnessId.CODEX: CodexVendor,
    HarnessId.PI: PiVendor,
    HarnessId.ZCODE: ZcodeVendor,
}


def get_cli_vendor(harness_id: HarnessId) -> Any:
    """The vendor instance for a CLI harness, or ``None`` when it is not one."""
    vendor_type = _VENDORS.get(harness_id)
    return vendor_type() if vendor_type is not None else None


def _codex_status(status: Any) -> str:
    # Declined: the approval was refused, so the command never ran — an error
    # outcome, not work still in progress. The one-shot stream spells it
    # snake_case ("declined"), the app-server camelCase ("declined" too).
    if status in ("failed", "declined"):
        return "error"
    if status == "completed":
        return "completed"
    return "running"


def _pi_has_tool_calls(message: dict[str, Any]) -> bool:
    """Whether a pi assistant message leaves tool calls to execute.

    Pi's own loop continues while any tool call is pending, so such a message
    is never the final answer of the turn."""
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") == "toolCall" for block in content)


def _pi_result_text(result: Any) -> str:
    if isinstance(result, dict):
        return _as_text(result.get("content"))
    return _as_text(result)


def _pi_message_cost(usage: dict[str, Any] | None) -> float | None:
    """The USD cost pi attributes to one assistant message, if it reported one."""
    cost = usage.get("cost") if isinstance(usage, dict) else None
    total = cost.get("total") if isinstance(cost, dict) else None
    if isinstance(total, (int, float)) and not isinstance(total, bool):
        return float(total)
    return None


# Vendors report usage with different flat key styles (codex snake_case, pi
# camelCase); each canonical count maps to its known spellings in preference
# order. ``total`` is the vendor's own turn total where it reports one.
_USAGE_KEY_SPELLINGS: dict[str, tuple[str, ...]] = {
    "input": ("input_tokens", "inputTokens", "input"),
    "output": ("output_tokens", "outputTokens", "output"),
    "reasoning": (
        "reasoning_output_tokens",
        "reasoningOutputTokens",
        "reasoning_tokens",
        "reasoningTokens",
        "reasoning",
    ),
    "read": (
        "cached_input_tokens",
        "cachedInputTokens",
        "cache_read_input_tokens",
        "cacheReadInputTokens",
        "cacheRead",
    ),
    "write": (
        "cache_write_input_tokens",
        "cacheWriteInputTokens",
        "cache_creation_input_tokens",
        "cacheWrite",
    ),
    "total": ("total_tokens", "totalTokens", "total"),
}


def _usage_tokens(usage: dict[str, Any]) -> dict[str, Any]:
    """Canonical token usage (the bridge's nested shape) from a vendor's flat
    usage object, or ``{}`` when it carries no counts the bridge understands."""
    picks: dict[str, int] = {}
    for canonical, spellings in _USAGE_KEY_SPELLINGS.items():
        for spelling in spellings:
            value = usage.get(spelling)
            if isinstance(value, int) and not isinstance(value, bool):
                picks[canonical] = value
                break
    tokens: dict[str, Any] = {
        key: picks[key] for key in ("input", "output", "reasoning") if key in picks
    }
    cache = {key: picks[key] for key in ("read", "write") if key in picks}
    if cache:
        tokens["cache"] = cache
    if "total" in picks:
        tokens["total"] = picks["total"]
    return tokens


def _merged_usage_tokens(base: dict[str, Any] | None, addition: dict[str, Any]) -> dict[str, Any]:
    """Sum two canonical usage dicts (the shape ``_usage_tokens`` returns).

    Vendors whose stream reports usage per message need a turn total, and the
    counts are additive; a count either side carries is kept, even at zero.
    """

    def pick(source: dict[str, Any] | None, path: tuple[str, ...]) -> int | None:
        value: Any = source
        for step in path:
            if not isinstance(value, dict):
                return None
            value = value.get(step)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    merged: dict[str, Any] = {}
    for key in ("input", "output", "reasoning", "total"):
        parts = (pick(base, (key,)), pick(addition, (key,)))
        if any(part is not None for part in parts):
            merged[key] = sum(part or 0 for part in parts)
    cache: dict[str, int] = {}
    for key in ("read", "write"):
        parts = (pick(base, ("cache", key)), pick(addition, ("cache", key)))
        if any(part is not None for part in parts):
            cache[key] = sum(part or 0 for part in parts)
    if cache:
        merged["cache"] = cache
    return merged


def _error_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("message", "error", "detail"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return ""


def _default_exit_outcome(
    name: str, state: CliTurnState, returncode: int | None, stderr_tail: str
) -> TurnOutcome:
    if state.cancelled:
        return TurnOutcome(
            success=False,
            error="Task was cancelled",
            cancelled=True,
            message_cost_usd=state.cost_usd,
        )
    if state.error:
        return TurnOutcome.failed(state.error, message_cost_usd=state.cost_usd)
    if returncode != 0:
        detail = stderr_tail or f"{name} exited with code {returncode}"
        return TurnOutcome.failed(detail, message_cost_usd=state.cost_usd)
    if not state.text:
        return TurnOutcome.failed(
            f"{name} completed without emitting assistant output.",
            message_cost_usd=state.cost_usd,
        )
    return TurnOutcome.ok(message_cost_usd=state.cost_usd)


__all__ = [
    "CodexVendor",
    "PiVendor",
    "ZcodeVendor",
    "get_cli_vendor",
]
