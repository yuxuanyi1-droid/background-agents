"""``AgentHarness`` for vendor coding-agent CLIs driven as subprocesses.

Several vendors ship a non-interactive mode that runs one turn, writes
progress to stdout, and can continue a prior conversation by id: Codex
(``codex exec``), Pi (``pi --mode json``), DeepSeek Harness
(``dsh --profile headless``), and ZCode (``zcode --print``). They differ only
in argv shape and stdout records, so this module owns the process lifecycle,
the inactivity and prompt deadlines, cancellation, and the vendor-neutral
bridge-event vocabulary. A ``CliVendor`` supplies the argv builder, the record
translator, and the exit policy.

Unlike ``opencode serve``, these agents have no resident server: the
supervisor half (``CliStager``) only stages the filesystem, and every turn is
a fresh child process. Conversation continuity rides on the vendor's own
on-disk session store, which the sandbox snapshot carries.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..custom_providers import find_provider_for_model, load_custom_providers
from .base import (
    BridgeEvent,
    EventSink,
    HarnessId,
    HarnessPrompt,
    HarnessStartError,
    PromptLimits,
    TurnOutcome,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ..custom_providers import CustomProvider
    from ..log_config import StructuredLogger

STDERR_TAIL_CHARS = 2000
KILL_GRACE_SECONDS = 5.0


@dataclass
class CliTurnState:
    """Accumulated state while translating one vendor turn."""

    message_id: str
    step_id: str
    text: str = ""
    last_emitted_text: str = ""
    session_id: str | None = None
    error: str | None = None
    emitted_error: bool = False
    cost_usd: float | None = None
    tokens: dict[str, Any] | None = None
    step_started: bool = False
    completed: bool = False
    tool_names: dict[str, str] = field(default_factory=dict)
    tool_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    agent_texts: dict[str, str] = field(default_factory=dict)


def step_start_events(state: CliTurnState) -> list[BridgeEvent]:
    if state.step_started:
        return []
    state.step_started = True
    return [{"type": "step_start", "messageId": state.message_id, "stepId": state.step_id}]


def text_events(state: CliTurnState, text: str) -> list[BridgeEvent]:
    """Emit one cumulative token event when the assistant text grows."""
    if not text or text == state.text:
        return []
    state.text = text
    if text == state.last_emitted_text:
        return []
    state.last_emitted_text = text
    return [
        *step_start_events(state),
        {"type": "token", "content": text, "messageId": state.message_id},
    ]


def append_text_events(state: CliTurnState, delta: str) -> list[BridgeEvent]:
    return text_events(state, state.text + delta)


def tool_events(
    state: CliTurnState,
    *,
    call_id: str,
    name: str,
    args: dict[str, Any] | None,
    status: str,
    output: str = "",
) -> list[BridgeEvent]:
    if name:
        state.tool_names[call_id] = name
    if args is not None:
        state.tool_args[call_id] = args
    return [
        *step_start_events(state),
        {
            "type": "tool_call",
            "tool": state.tool_names.get(call_id, name or "tool"),
            "args": state.tool_args.get(call_id, {}),
            "callId": call_id,
            "status": status,
            "output": output,
            "messageId": state.message_id,
        },
    ]


def step_finish_event(state: CliTurnState, *, reason: str) -> BridgeEvent:
    finish: BridgeEvent = {
        "type": "step_finish",
        "messageId": state.message_id,
        "stepId": state.step_id,
        "cost": state.cost_usd or 0.0,
        "messageCostUsd": state.cost_usd or 0.0,
        "reason": reason,
    }
    if state.tokens:
        finish["tokens"] = state.tokens
    return finish


def error_event(state: CliTurnState, message: str) -> list[BridgeEvent]:
    if state.emitted_error:
        return []
    state.emitted_error = True
    state.error = message
    return [{"type": "error", "error": message, "messageId": state.message_id}]


@runtime_checkable
class CliVendor(Protocol):
    """One vendor CLI behind the generic subprocess harness."""

    id: HarnessId
    binary: str
    json_stream: bool

    def initial_session_id(self) -> str | None:
        """A client-chosen conversation id, or ``None`` to learn it from output."""
        ...

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
        """Arguments after the binary name for one turn. ``model_provider`` is
        the custom-provider key a ``{key}/{model}`` selection routed to, for
        vendors whose CLI selects a gateway by name (Codex)."""
        ...

    def parse_record(self, record: dict[str, Any], state: CliTurnState) -> list[BridgeEvent]:
        """Translate one stdout JSON record into zero or more bridge events."""
        ...

    def extra_env(self, *, model: str | None) -> dict[str, str]:
        """Environment overrides for the child (e.g. a vendor's model selector)."""
        ...

    def prepare(self, custom_providers: tuple[CustomProvider, ...]) -> None:
        """Register custom providers in the vendor's on-disk config, if it
        consumes them. Called once at harness open."""
        ...

    def exit_outcome(
        self, state: CliTurnState, returncode: int | None, stderr_tail: str
    ) -> TurnOutcome:
        """Settle the turn from the translated state and the process exit."""
        ...


class CliInactivityTimeout(Exception):
    """No stdout record arrived within the inactivity budget."""


class CliPromptTimeout(Exception):
    """The whole turn exceeded the prompt budget."""


class CliHarness:
    """Bridge half for a ``CliVendor``; owns one child process per turn."""

    def __init__(
        self,
        *,
        vendor: CliVendor,
        log: StructuredLogger,
        limits: PromptLimits,
        workdir: Path,
        has_repository: bool,
    ) -> None:
        self.vendor = vendor
        self.log = log
        self.limits = limits
        self.workdir = workdir
        self.has_repository = has_repository
        self.session_id: str | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._abort_requested = False
        self._custom_providers: tuple[CustomProvider, ...] = ()

    @property
    def id(self) -> HarnessId:
        return self.vendor.id

    async def open(self) -> None:
        if shutil.which(self.vendor.binary) is None:
            raise HarnessStartError(
                f"The {self.vendor.binary} CLI is not installed in this sandbox."
            )
        self._custom_providers = load_custom_providers(os.environ)
        self.vendor.prepare(self._custom_providers)

    async def close(self) -> None:
        if self._process is not None:
            await self._kill(self._process)

    async def resume_session(self, persisted_id: str) -> bool:
        # These CLIs resolve a prior conversation from their own on-disk store;
        # the id is adopted optimistically and the first turn reports an error
        # if the store did not survive a snapshot restore.
        self.session_id = persisted_id
        self.log.info(
            f"{self.vendor.id.value}.session.ensure",
            agent_session_id=persisted_id,
            action="loaded",
        )
        return True

    async def create_session(self) -> None:
        chosen = self.vendor.initial_session_id()
        self.session_id = chosen
        self.log.info(
            f"{self.vendor.id.value}.session.ensure",
            agent_session_id=chosen,
            action="created" if chosen else "deferred",
        )

    async def run_prompt(self, prompt: HarnessPrompt, emit: EventSink) -> TurnOutcome:
        state = CliTurnState(message_id=prompt.message_id, step_id=str(uuid.uuid4()))
        self._abort_requested = False
        budget = (
            prompt.max_duration_seconds
            if prompt.max_duration_seconds is not None
            else self.limits.prompt_max_duration_seconds
        )
        resolved = (
            find_provider_for_model(prompt.model, self._custom_providers) if prompt.model else None
        )
        argv = [
            self.vendor.binary,
            *self.vendor.build_argv(
                session_id=self.session_id,
                prompt_text=prompt.text,
                model=prompt.model,
                reasoning_effort=prompt.reasoning_effort,
                workdir=self.workdir,
                model_provider=resolved[0].provider_key if resolved else None,
            ),
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(self.workdir),
                env={**os.environ, **self.vendor.extra_env(model=prompt.model)},
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError:
            return TurnOutcome.failed(f"The {self.vendor.binary} CLI is not installed.")
        self._process = process
        assert process.stdout is not None and process.stderr is not None
        stderr_task = asyncio.create_task(process.stderr.read())
        timeout_error: Exception | None = None
        returncode: int | None = None
        try:
            async with asyncio.timeout(budget):
                if self.vendor.json_stream:
                    await self._consume_jsonl(process.stdout, state, emit)
                else:
                    await self._consume_text(process.stdout, state, emit)
            returncode = await process.wait()
        except (TimeoutError, CliPromptTimeout) as error:
            timeout_error = (
                error
                if isinstance(error, CliPromptTimeout)
                else CliPromptTimeout("The turn exceeded its time budget.")
            )
            await self._kill(process)
            returncode = process.returncode
        except CliInactivityTimeout as error:
            timeout_error = error
            await self._kill(process)
            returncode = process.returncode
        except asyncio.CancelledError:
            await self._kill(process)
            raise
        finally:
            self._process = None
            if not stderr_task.done():
                # The process is gone or being killed; bound the drain so a
                # wedged child cannot hold the turn open.
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(stderr_task), KILL_GRACE_SECONDS)
            if not stderr_task.done():
                stderr_task.cancel()
        stderr_tail = _decode_tail(stderr_task) if stderr_task.done() else ""
        if timeout_error is not None:
            return TurnOutcome.failed(str(timeout_error), message_cost_usd=state.cost_usd)
        await emit(step_finish_event(state, reason="completed"))
        return self.vendor.exit_outcome(state, returncode, stderr_tail)

    async def _consume_jsonl(
        self, stream: asyncio.StreamReader, state: CliTurnState, emit: EventSink
    ) -> None:
        while True:
            try:
                line = await asyncio.wait_for(
                    stream.readline(), self.limits.inactivity_timeout_seconds
                )
            except TimeoutError as error:
                raise CliInactivityTimeout(
                    f"No output for {self.limits.inactivity_timeout_seconds:.0f}s."
                ) from error
            if not line:
                return
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            for event in self.vendor.parse_record(record, state):
                await emit(event)

    async def _consume_text(
        self, stream: asyncio.StreamReader, state: CliTurnState, emit: EventSink
    ) -> None:
        try:
            raw = await asyncio.wait_for(stream.read(), self.limits.inactivity_timeout_seconds)
        except TimeoutError as error:
            raise CliInactivityTimeout(
                f"No output for {self.limits.inactivity_timeout_seconds:.0f}s."
            ) from error
        for event in text_events(state, raw.decode("utf-8", errors="replace").strip()):
            await emit(event)

    async def abort(self) -> bool:
        self._abort_requested = True
        if self._process is None:
            return False
        await self._kill(self._process)
        return True

    async def stop_execution(self, timeout_seconds: float) -> bool:
        process = self._process
        if process is None:
            return True
        try:
            async with asyncio.timeout(max(timeout_seconds, 0.0)):
                await self._kill(process)
        except TimeoutError:
            return False
        return process.returncode is not None

    async def _kill(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
        try:
            await asyncio.wait_for(process.wait(), KILL_GRACE_SECONDS)
        except TimeoutError:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), KILL_GRACE_SECONDS)


def _decode_tail(stderr_task: asyncio.Task[bytes]) -> str:
    if stderr_task.cancelled():
        return ""
    error = stderr_task.exception()
    data = None if error is not None else stderr_task.result()
    if not data:
        return ""
    return data.decode("utf-8", errors="replace").strip()[-STDERR_TAIL_CHARS:]
