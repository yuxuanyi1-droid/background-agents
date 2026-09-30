"""Vendor descriptions for the generic CLI harness (see ``cli_harness.py``).

Each class encodes one vendor's non-interactive contract: the argv for a turn,
the stdout record shapes, and how the process exit maps to a turn outcome.
They are deliberately small and side-effect free so the translation can be
unit-tested against synthetic records without the vendor installed.

Session continuity differs by vendor:

- Codex and DeepSeek Harness create the conversation themselves; the id is
  read from the first stdout record and passed back on the next turn.
- Pi accepts a client-chosen ``--session-id`` and creates it when absent, so
  the id is chosen up front.
- ZCode's CLI exposes no documented resume switch, so every turn is a fresh
  process and continuity is not available yet.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..custom_providers import (
    codex_model_catalog_path,
    dsh_model_selection_patch,
    load_custom_providers,
    write_codex_model_catalog,
    write_codex_model_providers,
    write_dsh_profile_patch,
    write_pi_models_json,
    write_zcode_model_selection,
    write_zcode_provider_config,
    zcode_provider_config_path,
)
from .base import HarnessId, TurnOutcome
from .cli_harness import (
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


class CodexVendor:
    """OpenAI Codex CLI (``codex exec --json``)."""

    id = HarnessId.CODEX
    binary = "codex"
    json_stream = True

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
        argv = ["exec"]
        if session_id:
            argv += ["resume", session_id]
        argv += [
            "--json",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "--cd",
            str(workdir),
        ]
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
            return events
        if kind == "turn.completed":
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
    """Pi coding agent (``pi --mode json``)."""

    id = HarnessId.PI
    binary = "pi"
    json_stream = True

    def initial_session_id(self) -> str | None:
        # Pi accepts ``--session-id`` as an exact id, creating it when absent.
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
            # Pi's level for disabling thinking is "off", not the registry's "none".
            argv += ["--thinking", "off" if reasoning_effort == "none" else reasoning_effort]
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
            cost = usage.get("cost")
            if isinstance(cost, dict) and isinstance(cost.get("total"), (int, float)):
                state.cost_usd = float(cost["total"])
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
        message with ``stopReason: "error"`` is what fails it here. The final
        text replaces the accumulated deltas only when it is longer, so an
        earlier assistant message's text is never lost.
        """
        message = record.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            return []
        events: list[Any] = []
        if message.get("stopReason") == "error":
            detail = str(message.get("errorMessage") or "Pi reported a provider error")
            events += error_event(state, detail)
        final_text = _as_text(message.get("content"))
        if final_text and len(final_text) > len(state.text):
            events += text_events(state, final_text)
        usage = message.get("usage")
        if isinstance(usage, dict):
            cost = usage.get("cost")
            if isinstance(cost, dict) and isinstance(cost.get("total"), (int, float)):
                state.cost_usd = float(cost["total"])
        return events

    def exit_outcome(
        self, state: CliTurnState, returncode: int | None, stderr_tail: str
    ) -> TurnOutcome:
        return _default_exit_outcome("pi", state, returncode, stderr_tail)


class DshVendor:
    """DeepSeek Harness (``dsh --profile headless --json``)."""

    id = HarnessId.DSH
    binary = "dsh"
    json_stream = True

    def initial_session_id(self) -> str | None:
        # An unknown ``--session-id`` is rejected, so the id is read from the
        # opening session record instead of chosen here.
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
        argv = ["--profile", "headless"]
        if model_provider:
            # The headless profile has no model flag: the routed model rides a
            # per-turn --patch overlay of the agent-default-model config, the
            # same overlay mechanism the provider declaration below uses.
            bare = _bare_model(model)
            if bare:
                selection_path = Path(tempfile.gettempdir()) / "oi-dsh-model-selection.yml"
                selection_path.write_text(
                    dsh_model_selection_patch(model_provider, bare, reasoning_effort)
                )
                argv += ["--patch", str(selection_path)]
        argv += ["--json"]
        if session_id:
            argv += ["--session-id", session_id]
        argv.append(prompt_text)
        return argv

    def extra_env(self, *, model: str | None) -> dict[str, str]:
        bare = _bare_model(model)
        return {"DSH_MODEL": bare} if bare else {}

    def prepare(self, custom_providers: tuple[CustomProvider, ...]) -> None:
        """Declare the providers in the headless profile's user patch layer.

        dsh's LLM stack is pi-ai behind a cordis config overlay, so a route
        pi-ai has never heard of is fully describable from ``cordis.patch.yml``;
        the credential ref resolves from the provider's env var.
        """
        home = Path(os.environ.get("DSH_HOME") or (Path.home() / ".dsh"))
        write_dsh_profile_patch(
            home / "profiles" / "headless" / "cordis.patch.yml", custom_providers
        )

    def parse_record(self, record: dict[str, Any], state: CliTurnState) -> list[Any]:
        session_id = record.get("sessionId")
        events: list[Any] = []
        if isinstance(session_id, str) and session_id:
            state.session_id = session_id
        kind = record.get("type")
        if kind == "text":
            delta = record.get("delta")
            if isinstance(delta, str):
                events += append_text_events(state, delta)
            else:
                text = record.get("text") or record.get("content")
                if isinstance(text, str):
                    events += append_text_events(state, text)
            return events
        if kind == "tool_call":
            # Record shape per dsh-headless's projection (0.1.7-rc.2): the
            # tool name rides `tool` and the arguments `input`.
            return tool_events(
                state,
                call_id=str(record.get("callId") or record.get("id") or "tool"),
                name=str(record.get("tool") or record.get("name") or "tool"),
                args=record.get("input") if isinstance(record.get("input"), dict) else {},
                status="running",
            )
        if kind == "tool_result":
            # Result records carry no tool name — tool_events keeps the one the
            # tool_call record learned — and report errors as status:"error".
            return tool_events(
                state,
                call_id=str(record.get("callId") or record.get("id") or "tool"),
                name="",
                args=None,
                status="error" if record.get("status") == "error" else "completed",
                output=_as_text(record.get("result")),
            )
        if kind == "final":
            text = record.get("text") or record.get("content")
            if isinstance(text, str):
                events += text_events(state, text)
            state.completed = True
            return events
        if kind == "error":
            return error_event(state, _error_text(record) or "DeepSeek Harness reported an error")
        return events

    def exit_outcome(
        self, state: CliTurnState, returncode: int | None, stderr_tail: str
    ) -> TurnOutcome:
        return _default_exit_outcome("dsh", state, returncode, stderr_tail)


class ZcodeVendor:
    """ZCode CLI (``zcode --prompt``), text output only.

    Verified against the v3.14.3 source build: ``--prompt`` runs one headless
    turn (permission mode defaults to ``yolo`` under it, so no approvals) and
    ``--json`` emits nothing for ``--prompt``. Custom providers ride the CLI's
    personal provider config — ZCode offers no environment seam for endpoints
    or keys, so unlike the other vendors the api-key rides the file as a
    literal. The CLI reads its model from the same file's
    ``defaultModelSelection``, so each turn rewrites that field before the
    spawn (the per-turn seam the dsh vendor solves with a patch overlay). A
    session id is never captured today (the plain-text stream does not carry
    one), so ``--resume`` stays unwired and every turn is a fresh conversation.
    """

    id = HarnessId.ZCODE
    binary = "zcode"
    json_stream = False

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

    def parse_record(self, record: dict[str, Any], state: CliTurnState) -> list[Any]:
        return []

    def exit_outcome(
        self, state: CliTurnState, returncode: int | None, stderr_tail: str
    ) -> TurnOutcome:
        return _default_exit_outcome("zcode", state, returncode, stderr_tail)


_VENDORS: dict[HarnessId, type[Any]] = {
    HarnessId.CODEX: CodexVendor,
    HarnessId.PI: PiVendor,
    HarnessId.DSH: DshVendor,
    HarnessId.ZCODE: ZcodeVendor,
}


def get_cli_vendor(harness_id: HarnessId) -> Any:
    """The vendor instance for a CLI harness, or ``None`` when it is not one."""
    vendor_type = _VENDORS.get(harness_id)
    return vendor_type() if vendor_type is not None else None


def _codex_status(status: Any) -> str:
    if status == "failed":
        return "error"
    if status == "completed":
        return "completed"
    return "running"


def _pi_result_text(result: Any) -> str:
    if isinstance(result, dict):
        return _as_text(result.get("content"))
    return _as_text(result)


def _usage_tokens(usage: dict[str, Any]) -> dict[str, Any]:
    tokens: dict[str, Any] = {}
    for key in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            tokens[key] = value
    return tokens


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
    "DshVendor",
    "PiVendor",
    "ZcodeVendor",
    "get_cli_vendor",
]
