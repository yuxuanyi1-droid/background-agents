"""``AgentHarness`` for vendor coding-agent CLIs driven as subprocesses.

Several vendors ship a non-interactive mode that runs one turn, writes
progress to stdout, and can continue a prior conversation by id: Codex
(``codex exec``), Pi (``pi --mode json``), and ZCode (``zcode --prompt``).
They differ only in argv shape and stdout records, so this module owns the
process lifecycle, the inactivity and prompt deadlines, cancellation, and the
vendor-neutral bridge-event vocabulary. A ``CliVendor`` supplies the argv
builder, the record translator, and the exit policy.

The same vendors also ship a resident protocol server — Codex
``app-server``, Pi ``--mode rpc``, ZCode ``app-server --stdio`` — speaking a
line protocol over stdio: one long-lived process, a conversation established
per turn, and a notification stream that settles each turn. A
``ResidentCliVendor`` supplies that dialect and the harness drives it through
:class:`_ResidentServer`; the one-shot argv path remains as the
``OI_CLI_ONE_SHOT`` escape hatch. Either way the supervisor half
(``CliStager``) only stages the filesystem, and conversation continuity rides
on the vendor's own on-disk session store, which the sandbox snapshot carries.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import uuid
from collections import deque
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
# Vendor setup chains are short — create/resume then subscribe, or state then
# model then thinking — so this cap only exists to turn a non-converging
# vendor into a failed turn instead of a loop.
SETUP_REQUEST_LIMIT = 4


@dataclass
class CliTurnState:
    """Accumulated state while translating one vendor turn."""

    message_id: str
    step_id: str
    text: str = ""
    last_emitted_text: str = ""
    session_id: str | None = None
    server_turn_id: str | None = None
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
class ResidentCliVendor(Protocol):
    """A vendor driven through a resident line-protocol server.

    The harness spawns the server lazily on the first turn (when the model is
    known — some vendors only accept it on the command line), replays
    :meth:`handshake_requests`, then per turn pulls :meth:`next_setup_message`
    until the conversation is ready — one request at a time, so each may
    depend on the previous response's adopted id — submits the turn, and
    translates the notification stream through :meth:`parse_server_message`
    until the vendor raises :class:`CliTurnSettled`.
    """

    resident = True

    jsonrpc = True
    """Whether requests carry the JSON-RPC 2.0 envelope field. Vendors whose
    wire schema rejects unknown keys (ZCode) or speaks command-shaped frames
    (Pi) clear it."""

    def server_argv(
        self,
        *,
        session_id: str | None,
        model: str | None,
        reasoning_effort: str | None,
    ) -> list[str]:
        """Arguments after the binary name that start the protocol server.
        The conversation id and model are the spawn-time seam for vendors that
        only accept them on the command line (Pi); Codex and ZCode route both
        per turn over the protocol."""
        ...

    def handshake_requests(self) -> list[dict[str, Any]]:
        """Requests to send (in order) right after the process starts."""
        ...

    def reset(self) -> None:
        """Drop any live-conversation tracking. Called whenever a server
        process starts, so a restarted one resumes a persisted id instead of
        assuming its own conversation is still live."""
        ...

    def next_setup_message(
        self,
        *,
        session_id: str | None,
        model: str | None,
        reasoning_effort: str | None,
        workdir: Path,
        model_provider: str | None = None,
    ) -> dict[str, Any] | None:
        """The next request establishing this turn's conversation, or ``None``
        when it is ready. Called repeatedly with the latest adopted session
        id, so one message may depend on the previous response — ZCode needs
        that to subscribe to a session id only the create response carries."""
        ...

    def turn_start_message(
        self,
        *,
        session_id: str | None,
        prompt_text: str,
        model: str | None,
        reasoning_effort: str | None,
        model_provider: str | None = None,
    ) -> dict[str, Any] | None:
        """The request submitting the turn, or ``None`` when no conversation
        id is known (the turn fails as a harness error)."""
        ...

    def adopt_response_id(self, response: dict[str, Any]) -> str | None:
        """The conversation id in a setup response, if any."""
        ...

    def parse_server_message(
        self, message: dict[str, Any], state: CliTurnState
    ) -> list[BridgeEvent]:
        """Translate one queued protocol message (notification, event, or a
        server→client request); raise :class:`CliTurnSettled` on the turn's
        terminal notification."""
        ...

    def interrupt_messages(
        self, *, session_id: str | None, state: CliTurnState | None
    ) -> list[dict[str, Any]]:
        """Requests that ask the server to stop the running turn."""
        ...

    def server_request_messages(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Raw frames answering a server→client request carried by
        ``message``. Unattended harnesses must deny or cancel them — a
        permission or input prompt nobody answers would hang the turn."""
        ...


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


class CliTurnSettled(Exception):
    """A resident vendor saw the turn's terminal notification.

    The protocol server keeps running after a turn, so the consume loop ends
    on this sentinel instead of the stdout EOF a one-shot child relies on. The
    terminal notification's own events ride the sentinel so the loop can emit
    them before ending the turn.
    """

    def __init__(self, events: list[Any] | None = None) -> None:
        super().__init__("turn settled")
        self.events = events or []


class _ResidentServer:
    """A resident vendor protocol server: one process, many turns.

    A background reader correlates responses to pending requests by id and
    queues every other message as a notification — protocol notifications,
    command-shaped events, and server→client requests all ride that queue.
    stderr is drained into a bounded tail, so a chatty server never blocks on
    a full pipe. ``run_prompt`` sends its turn requests through
    :meth:`request`, then translates queued notifications until the vendor
    raises :class:`CliTurnSettled`. The process is killed on close; a mid-turn
    death fails that turn and the next one restarts it (conversations resume
    by id).
    """

    def __init__(self, vendor: ResidentCliVendor, log: StructuredLogger) -> None:
        self._vendor = vendor
        self._log = log
        self._process: asyncio.subprocess.Process | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._notifications: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail = ""
        self._rpc_id = 0

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def stderr_tail(self) -> str:
        """The last stderr lines, for failure diagnostics after a death."""
        return self._stderr_tail

    async def start(
        self,
        workdir: Path,
        *,
        session_id: str | None,
        model: str | None,
        reasoning_effort: str | None,
    ) -> None:
        assert self._process is None
        argv = [
            self._vendor.binary,
            *self._vendor.server_argv(
                session_id=session_id, model=model, reasoning_effort=reasoning_effort
            ),
        ]
        self._process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(workdir),
            env=os.environ,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        assert self._process.stdout is not None and self._process.stderr is not None
        self._reader_task = asyncio.create_task(self._read_stdout(self._process.stdout))
        self._stderr_task = asyncio.create_task(self._drain_stderr(self._process.stderr))
        # A fresh process knows no conversation: the vendor must resume any
        # persisted id instead of assuming its own live one.
        self._vendor.reset()
        for message in self._vendor.handshake_requests():
            await self.request(message)

    async def _read_stdout(self, stream: asyncio.StreamReader) -> None:
        while True:
            line = await stream.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict):
                continue
            request_id = message.get("id")
            if isinstance(request_id, str) and request_id in self._pending:
                future = self._pending.pop(request_id)
                if not future.done():
                    future.set_result(message)
            else:
                self._notifications.put_nowait(message)

    async def _drain_stderr(self, stream: asyncio.StreamReader) -> None:
        tail: deque[str] = deque(maxlen=16)
        while True:
            line = await stream.readline()
            if not line:
                return
            tail.append(line.decode("utf-8", errors="replace").rstrip())
            self._stderr_tail = "\n".join(tail)[-4096:]

    async def request(self, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        """Send one request and await its correlated response."""
        process = self._process
        if process is None or process.stdin is None or not self.alive:
            raise RuntimeError("The vendor protocol server is not running.")
        self._rpc_id += 1
        request_id = f"oi-{self._rpc_id}"
        envelope: dict[str, Any] = {"id": request_id}
        if getattr(self._vendor, "jsonrpc", True):
            envelope["jsonrpc"] = "2.0"
        payload = envelope | payload
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            process.stdin.write((json.dumps(payload) + "\n").encode())
            await process.stdin.drain()
            return await asyncio.wait_for(future, timeout)
        except TimeoutError:
            # Name the request: an unanswered one must not read as the turn's
            # own budget running out.
            raise CliPromptTimeout(
                f"No response to {_request_label(payload)} within {timeout:.0f}s."
            ) from None
        finally:
            self._pending.pop(request_id, None)

    async def send(self, frame: dict[str, Any]) -> None:
        """Write one raw frame — no id assigned, no response awaited — the
        seam for answering server→client requests."""
        process = self._process
        if process is None or process.stdin is None or not self.alive:
            raise RuntimeError("The vendor protocol server is not running.")
        process.stdin.write((json.dumps(frame) + "\n").encode())
        await process.stdin.drain()

    async def next_notification(self, timeout: float) -> dict[str, Any] | None:
        try:
            return await asyncio.wait_for(self._notifications.get(), timeout)
        except TimeoutError:
            return None

    def drop_queued_notifications(self) -> None:
        while not self._notifications.empty():
            self._notifications.get_nowait()

    async def kill(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        for task in (self._reader_task, self._stderr_task):
            if task is not None:
                task.cancel()
        self._reader_task = None
        self._stderr_task = None
        if process.returncode is None:
            process.kill()
            with contextlib.suppress(ProcessLookupError):
                await process.wait()


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
        self._server: _ResidentServer | None = None
        self._active_state: CliTurnState | None = None

    @property
    def _resident_mode(self) -> bool:
        """Resident protocol mode, unless the one-shot fuse is pulled for this
        harness id (``OI_CLI_ONE_SHOT=codex,pi,zcode``) — an escape hatch when
        a vendor's server mode misbehaves."""
        if not getattr(self.vendor, "resident", False):
            return False
        one_shot = {
            name.strip()
            for name in os.environ.get("OI_CLI_ONE_SHOT", "").split(",")
            if name.strip()
        }
        return self.vendor.id.value not in one_shot

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
        # The resident protocol server starts lazily on the first turn, when
        # the model is known: some vendors only accept it on the command line.

    async def close(self) -> None:
        if self._server is not None:
            await self._server.kill()
            self._server = None
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
        if self._resident_mode:
            assert self.vendor is not None
            self._active_state = state
            try:
                return await self._run_prompt_resident(
                    prompt, state, resolved[0].provider_key if resolved else None, budget, emit
                )
            finally:
                self._active_state = None
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

    async def _run_prompt_resident(
        self,
        prompt: HarnessPrompt,
        state: CliTurnState,
        model_provider: str | None,
        budget: float,
        emit: EventSink,
    ) -> TurnOutcome:
        """Run one turn over the resident protocol server.

        The server starts lazily here and is restarted (with the conversation
        resumed by id) when a previous turn killed or lost it; a turn ending
        normally leaves it running for the next one.
        """
        server = self._server
        if server is None or not server.alive:
            if server is not None:
                await server.kill()
            server = _ResidentServer(self.vendor, self.log)  # type: ignore[arg-type]
            await server.start(
                self.workdir,
                session_id=self.session_id,
                model=prompt.model,
                reasoning_effort=prompt.reasoning_effort,
            )
            self._server = server
        server.drop_queued_notifications()
        self._abort_requested = False
        vendor: ResidentCliVendor = self.vendor  # type: ignore[assignment]
        timeout_error: Exception | None = None
        try:
            async with asyncio.timeout(budget):
                # One setup request at a time, each seeing the latest adopted
                # id; the cap turns a vendor that never converges into a turn
                # failure instead of a loop.
                for _ in range(SETUP_REQUEST_LIMIT):
                    message = vendor.next_setup_message(
                        session_id=self.session_id,
                        model=prompt.model,
                        reasoning_effort=prompt.reasoning_effort,
                        workdir=self.workdir,
                        model_provider=model_provider,
                    )
                    if message is None:
                        break
                    response = await server.request(message)
                    failure = _response_error(response)
                    if failure is not None:
                        return TurnOutcome.failed(
                            f"{_request_label(message)}: {failure}",
                            message_cost_usd=state.cost_usd,
                        )
                    adopted = vendor.adopt_response_id(response)
                    if adopted:
                        self.session_id = adopted
                        state.session_id = adopted
                        self.log.info(
                            f"{self.vendor.id.value}.session.ensure",
                            agent_session_id=adopted,
                            action="created",
                        )
                else:
                    return TurnOutcome.failed(
                        f"The {self.vendor.id.value} conversation setup did not converge.",
                        message_cost_usd=state.cost_usd,
                    )
                start = vendor.turn_start_message(
                    session_id=self.session_id,
                    prompt_text=prompt.text,
                    model=prompt.model,
                    reasoning_effort=prompt.reasoning_effort,
                    model_provider=model_provider,
                )
                if start is None:
                    return TurnOutcome.failed(
                        f"The {self.vendor.id.value} conversation could not be started."
                    )
                response = await server.request(start)
                failure = _response_error(response)
                if failure is not None:
                    return TurnOutcome.failed(
                        f"{_request_label(start)}: {failure}",
                        message_cost_usd=state.cost_usd,
                    )
                await self._consume_server_notifications(server, state, emit)
        except (TimeoutError, CliPromptTimeout) as error:
            timeout_error = (
                error
                if isinstance(error, CliPromptTimeout)
                else CliPromptTimeout("The turn exceeded its time budget.")
            )
        except CliInactivityTimeout as error:
            timeout_error = error
        except asyncio.CancelledError:
            await server.kill()
            raise
        if timeout_error is not None:
            # The server may still be mid-turn; the next turn restarts it.
            await server.kill()
            self._server = None
            return TurnOutcome.failed(str(timeout_error), message_cost_usd=state.cost_usd)
        await emit(step_finish_event(state, reason="completed"))
        return vendor.exit_outcome(state, 0, "")

    async def _consume_server_notifications(
        self, server: _ResidentServer, state: CliTurnState, emit: EventSink
    ) -> None:
        vendor: ResidentCliVendor = self.vendor  # type: ignore[assignment]
        while True:
            message = await server.next_notification(self.limits.inactivity_timeout_seconds)
            if message is None:
                raise CliInactivityTimeout(
                    f"No output for {self.limits.inactivity_timeout_seconds:.0f}s."
                )
            for reply in vendor.server_request_messages(message):
                # Server→client requests must be answered — a permission or
                # input prompt nobody replies to would hang the turn. Replies
                # are fire-and-forget: a dead server surfaces as a death
                # below or a failed turn request next time.
                with contextlib.suppress(Exception):
                    await server.send(reply)
            try:
                events = vendor.parse_server_message(message, state)
            except CliTurnSettled as settled:
                for event in settled.events:
                    await emit(event)
                return
            for event in events:
                await emit(event)

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
        server = self._server
        if server is not None and server.alive:
            vendor: ResidentCliVendor = self.vendor  # type: ignore[assignment]
            try:
                for message in vendor.interrupt_messages(
                    session_id=self.session_id, state=self._active_state
                ):
                    # Interrupts are fire-and-forget: the turn loop settles on
                    # the server's terminal notification, or its death below.
                    await server.request(message, timeout=max(timeout_seconds, 5.0))
                return True
            except Exception:
                # An interrupt that cannot be delivered falls back to killing
                # the server; the next turn restarts and resumes the thread.
                await server.kill()
                self._server = None
                return True
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


def _response_error(response: dict[str, Any]) -> str | None:
    """The failure text in a protocol response, whatever its envelope: a
    JSON-RPC error object or a command response carrying success/error."""
    error = response.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(error, str) and error:
        return error
    return None


def _request_label(message: dict[str, Any]) -> str:
    """How a request names itself in a failure, across envelope styles."""
    label = message.get("method") or message.get("type")
    return label if isinstance(label, str) and label else "request"


def _decode_tail(stderr_task: asyncio.Task[bytes]) -> str:
    if stderr_task.cancelled():
        return ""
    error = stderr_task.exception()
    data = None if error is not None else stderr_task.result()
    if not data:
        return ""
    return data.decode("utf-8", errors="replace").strip()[-STDERR_TAIL_CHARS:]
