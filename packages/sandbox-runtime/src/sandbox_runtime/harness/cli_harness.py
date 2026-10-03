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
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
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
    from ..custom_providers import CustomProvider
    from ..log_config import StructuredLogger

STDERR_TAIL_CHARS = 2000
KILL_GRACE_SECONDS = 5.0
# How long a sweep gives SIGKILLed processes to finish tearing down (which is
# when their conversation locks actually free) before the awaiting request
# retries anyway.
KILL_SETTLE_SECONDS = 2.0
KILL_SETTLE_POLL_SECONDS = 0.05
# Vendor setup chains are short — create/resume then subscribe, or state then
# model then thinking — so this cap only exists to turn a non-converging
# vendor into a failed turn instead of a loop.
SETUP_REQUEST_LIMIT = 4
# asyncio's subprocess streams cap a readline at 64 KiB by default; a protocol
# frame larger than that raises out of the reader and used to be reported as a
# server death ("exited mid-turn (code None)") while the process was alive.
PROTOCOL_STREAM_LIMIT_BYTES = 16 * 1024 * 1024
# Default bound on one protocol request (handshake, setup, turn start) waiting
# for its response: a server that stays silent must fail the turn, not wedge it.
PROTOCOL_REQUEST_TIMEOUT_SECONDS = 30.0
# A closed stdout almost always means the process just exited; this is how long
# the reader waits for the exit code to be reaped before reporting the death
# without one.
SERVER_EXIT_REAP_GRACE_SECONDS = 1.0


@dataclass
class CliTurnState:
    """Accumulated state while translating one vendor turn."""

    message_id: str
    step_id: str
    text: str = ""
    last_emitted_text: str = ""
    session_id: str | None = None
    server_turn_id: str | None = None
    final_message_seen_at: float | None = None
    error: str | None = None
    emitted_error: bool = False
    cost_usd: float | None = None
    # Cost of the assistant messages that already finished within this turn.
    # Vendors whose stream reports usage per message (pi) add each finished
    # message's share here, so the running total in ``cost_usd`` never double
    # counts the in-flight message's partial usage.
    cost_committed_usd: float = 0.0
    tokens: dict[str, Any] | None = None
    step_started: bool = False
    completed: bool = False
    # The vendor reported the turn as aborted (a stop the client asked for, or
    # the vendor's own interruption); the turn settles as cancelled, not failed.
    cancelled: bool = False
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

    id: HarnessId
    binary: str
    resident: bool = True

    jsonrpc: bool = True
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

    settle_after_final_message: float
    """Grace seconds after the vendor marks ``state.final_message_seen_at``
    before the turn settles without its own terminal notification — for
    vendors whose final assistant message is authoritative (Pi) but whose
    settle event can be late or lost in a post-turn hang. Zero (default)
    disables it: only the vendor's terminal notification settles."""

    def setup_lock_contention(self, failure: str) -> bool:
        """Whether a failed setup request means another live process holds
        the conversation's writer lock. The harness kills the holder and
        retries the request once; any other failure is final."""
        ...

    def turn_start_busy_failure(self, failure: str) -> bool:
        """Whether a rejected turn start means the vendor is still finishing
        the previous run. The harness waits for that run to end, then
        resubmits the start once; any other failure is final."""
        ...

    def previous_run_settled(self, message: dict[str, Any]) -> bool:
        """Whether a notification observed while recovering from a busy turn
        start marks the previous run's true end — the point a fresh prompt is
        accepted again. Its preceding tail belongs to the turn that already
        settled and is discarded."""
        ...

    def exit_outcome(
        self, state: CliTurnState, returncode: int | None, stderr_tail: str
    ) -> TurnOutcome:
        """The turn's result from its final state and exit code (the resident
        path settles through the protocol, so there is no exit code)."""
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


class CliServerDied(Exception):
    """A resident protocol server exited while its turn was still open."""


# Queued by the stdout reader when it stops — EOF, cancellation, or a crash —
# so the consume loop learns of a dead server instead of idling to the
# inactivity budget.
_SERVER_EXITED = object()


def _resident_server_args(vendor: ResidentCliVendor) -> list[str]:
    """The static server argv of this vendor: the arguments that start the
    protocol server, called with no session or model so vendors that take
    spawn-time routing (Pi) return their static form."""
    return vendor.server_argv(session_id=None, model=None, reasoning_effort=None)


def _starts_vendor_server(args: list[str], binary: str, server_args: list[str]) -> bool:
    """Whether a process's argv belongs to this vendor's protocol server.

    The binary reaches the process through whatever the install left in
    PATH: the sandbox installs the npm packages into a local tree, so the
    kernel execs the ``.bin`` shim — a symlink that keeps its own name — and
    the process shows ``node /opt/.../bin/codex app-server``; Codex's node
    launcher then spawns the native app-server at an absolute path showing
    ``/opt/.../codex app-server``. So the match is: some argument names the
    vendor CLI (the shim, the native binary, or the bin script directly) and
    the vendor's server arguments directly follow it.
    """
    if not server_args:
        return False
    name = Path(binary).name
    names = {name, f"{name}.js", f"{name}.cjs"}
    for index, argument in enumerate(args):
        if Path(argument).name not in names:
            continue
        if args[index + 1 : index + 1 + len(server_args)] == server_args:
            return True
    return False


def _process_reaped(pid: int) -> bool:
    """Whether a killed process has finished tearing down (gone or a zombie).

    File locks are released during teardown, before the zombie state, so both
    count as "the lock is free" — and a zombie still has a /proc entry until
    its parent reaps it, which must not hold up the wait.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_bytes()
    except OSError:
        return True
    close = stat.rfind(b")")
    return close != -1 and stat[close + 2 : close + 3] == b"Z"


async def kill_orphaned_resident_servers(
    binary: str, server_args: list[str], *, exclude_pgid: int | None = None
) -> list[int]:
    """SIGKILL resident server processes this harness does not own.

    A resident server is spawned with ``start_new_session`` so a bridge crash
    or restart does not kill it — which also means nothing ever does. The
    orphan keeps its vendor's per-conversation locks (Codex holds a
    single-writer lock per thread), and the replacement server's resume is
    rejected with "already has an active writer" until the orphan dies. One
    sandbox runs one session, so any matching process that is not the server
    this harness is about to start is an orphan.

    ``exclude_pgid`` spares one process group — the live server's own when
    the sweep runs mid-turn: the server that could not take a conversation
    lock is not the process holding it.
    """
    killed: list[int] = []
    mine = os.getpid()
    try:
        entries = list(os.scandir("/proc"))
    except OSError:
        return killed
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == mine:
            continue
        if exclude_pgid is not None:
            try:
                if os.getpgid(pid) == exclude_pgid:
                    continue
            except OSError:
                continue
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        args = [part.decode("utf-8", "replace") for part in argv if part]
        if not _starts_vendor_server(args, binary, server_args):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except OSError:
            continue
    if killed:
        # The resume that prompted the sweep must not race the kill: a
        # process's file locks are released only once it has torn down.
        deadline = time.monotonic() + KILL_SETTLE_SECONDS
        while not all(_process_reaped(pid) for pid in killed):
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(KILL_SETTLE_POLL_SECONDS)
    return killed


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    """Stop a spawned process and its whole session group.

    Every server is spawned with ``start_new_session`` so a bridge crash
    cannot take it down — which also means signalling the direct child is not
    enough. Codex's npm launcher spawns the native app-server as its child
    and forwards only SIGINT/SIGTERM/SIGHUP, so killing the launcher alone
    would leave the real server — the one holding the conversation's writer
    lock — running.
    """
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
        self._reader_error: str | None = None
        self._rpc_id = 0

    @property
    def alive(self) -> bool:
        # A server whose stdout reader has stopped is unusable even while its
        # process lingers: no response can ever be correlated again, so the
        # next turn must restart it instead of talking into the void.
        if self._process is None or self._process.returncode is not None:
            return False
        reader = self._reader_task
        return reader is not None and not reader.done()

    def death_detail(self) -> str:
        """The failure text for a server that stopped mid-turn: a reader
        failure (the process may still be running) outranks an exit, and the
        exit code is read live (None while not yet reaped)."""
        if self._reader_error:
            detail = (
                f"The {self._vendor.id.value} protocol server stream failed mid-turn "
                f"({self._reader_error}); the server process was still running."
            )
        else:
            code = self._process.returncode if self._process is not None else None
            detail = f"The {self._vendor.id.value} protocol server exited mid-turn (code {code})."
        if self._stderr_tail:
            detail += f" Stderr tail: {self._stderr_tail[-STDERR_TAIL_CHARS:]}"
        return detail

    def _fail_pending(self, error: CliServerDied) -> None:
        """Settle every in-flight request with the given failure, so an
        awaiting :meth:`request` never idles out its own timeout after the
        server is gone."""
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    @property
    def stderr_tail(self) -> str:
        """The last stderr lines, for failure diagnostics after a death."""
        return self._stderr_tail

    @property
    def reader_error(self) -> str | None:
        """Why the stdout reader stopped without a server exit, if it did."""
        return self._reader_error

    @property
    def pgid(self) -> int | None:
        """The server's process-group id (its own session), while it runs."""
        process = self._process
        if process is None:
            return None
        try:
            return os.getpgid(process.pid)
        except OSError:
            return None

    async def start(
        self,
        workdir: Path,
        *,
        session_id: str | None,
        model: str | None,
        reasoning_effort: str | None,
    ) -> None:
        assert self._process is None
        # A previous server this bridge never reaped may still hold the
        # conversation's locks; clear it before it can reject our resume.
        killed = await kill_orphaned_resident_servers(
            self._vendor.binary, _resident_server_args(self._vendor)
        )
        if killed:
            self._log.info(
                f"{self._vendor.id.value}.server.orphan_sweep",
                killed_pids=killed,
            )
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
            limit=PROTOCOL_STREAM_LIMIT_BYTES,
        )
        assert self._process.stdout is not None and self._process.stderr is not None
        self._reader_error = None
        self._reader_task = asyncio.create_task(self._read_stdout(self._process.stdout))
        self._stderr_task = asyncio.create_task(self._drain_stderr(self._process.stderr))
        # A fresh process knows no conversation: the vendor must resume any
        # persisted id instead of assuming its own live one.
        self._vendor.reset()
        for message in self._vendor.handshake_requests():
            await self.request(message)

    async def _read_stdout(self, stream: asyncio.StreamReader) -> None:
        try:
            while True:
                try:
                    line = await stream.readline()
                except ValueError as error:
                    # A frame larger than the stream limit. The server is still
                    # alive; record why the reader stopped so the turn failure
                    # names the real cause instead of a phantom exit.
                    self._reader_error = f"{type(error).__name__}: {error}"
                    return
                if not line:
                    # stdout closed: the server is done talking. Give the OS a
                    # moment to reap it so the death message can name the exit
                    # code; a lingering process must not stall the report.
                    await self._await_exit_code()
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
                    continue
                # Answer server→client requests here — the one point every
                # inbound message passes, so a reverse request sent while a
                # setup or turn request is still blocking gets its reply
                # within milliseconds instead of after that request dies.
                for reply in self._vendor.server_request_messages(message):
                    await self._write_frame(reply)
                self._notifications.put_nowait(message)
        finally:
            # Whatever ended the reader — an exit, a stream failure, a kill —
            # the server can answer nothing more: settle the in-flight requests
            # with the death before the turn loop sees the exit marker.
            self._fail_pending(CliServerDied(self.death_detail()))
            self._notifications.put_nowait(_SERVER_EXITED)

    async def _await_exit_code(self) -> None:
        """Wait briefly for a just-exited server to be reaped, so the death
        message can name its exit code instead of a phantom ``None``."""
        process = self._process
        if process is None or process.returncode is not None:
            return
        with contextlib.suppress(Exception):
            async with asyncio.timeout(SERVER_EXIT_REAP_GRACE_SECONDS):
                await process.wait()

    async def _write_frame(self, frame: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            return
        process.stdin.write((json.dumps(frame) + "\n").encode())
        await process.stdin.drain()

    async def _drain_stderr(self, stream: asyncio.StreamReader) -> None:
        tail: deque[str] = deque(maxlen=16)
        while True:
            try:
                line = await stream.readline()
            except ValueError:
                # An oversize stderr line; keep the tail collected so far.
                return
            if not line:
                return
            tail.append(line.decode("utf-8", errors="replace").rstrip())
            self._stderr_tail = "\n".join(tail)[-4096:]

    async def request(
        self, payload: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any]:
        """Send one request and await its correlated response."""
        process = self._process
        if process is None or process.stdin is None or not self.alive:
            # A turn failure with the real cause, not an exception out of
            # run_prompt; the caller kills whatever process remains.
            raise CliServerDied(f"The {self._vendor.id.value} protocol server is not running.")
        if timeout is None:
            timeout = PROTOCOL_REQUEST_TIMEOUT_SECONDS
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

    async def next_notification(self, timeout: float) -> dict[str, Any] | None:
        try:
            return await asyncio.wait_for(self._notifications.get(), timeout)
        except TimeoutError:
            return None

    async def exit_details(self) -> tuple[int | None, str]:
        """The server's exit code and stderr tail, once its stdout closed."""
        process = self._process
        if process is None:
            return None, self._stderr_tail
        if process.returncode is None:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(1.0):
                    await process.wait()
        return process.returncode, self._stderr_tail

    def drop_queued_notifications(self) -> None:
        while not self._notifications.empty():
            self._notifications.get_nowait()

    async def kill(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        self._fail_pending(
            CliServerDied(f"The {self._vendor.id.value} protocol server was stopped mid-turn.")
        )
        for task in (self._reader_task, self._stderr_task):
            if task is not None:
                task.cancel()
        self._reader_task = None
        self._stderr_task = None
        await _terminate_process(process)


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
            await _terminate_process(self._process)

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
                limit=PROTOCOL_STREAM_LIMIT_BYTES,
            )
        except FileNotFoundError:
            return TurnOutcome.failed(f"The {self.vendor.binary} CLI is not installed.")
        self._process = process
        assert process.stdout is not None and process.stderr is not None
        stderr_task = asyncio.create_task(process.stderr.read())
        failure: Exception | None = None
        returncode: int | None = None
        try:
            async with asyncio.timeout(budget):
                if self.vendor.json_stream:
                    await self._consume_jsonl(process.stdout, state, emit)
                else:
                    await self._consume_text(process.stdout, state, emit)
                # After stdout closes the process should exit at once; a
                # lingering child must still be bounded by the turn's budget
                # instead of parking the turn on wait() forever.
                returncode = await process.wait()
        except (TimeoutError, CliPromptTimeout) as error:
            failure = (
                error
                if isinstance(error, CliPromptTimeout)
                else CliPromptTimeout("The turn exceeded its time budget.")
            )
            await _terminate_process(process)
            returncode = process.returncode
        except CliInactivityTimeout as error:
            failure = error
            await _terminate_process(process)
            returncode = process.returncode
        except CliServerDied as error:
            failure = error
            await _terminate_process(process)
            returncode = process.returncode
        except asyncio.CancelledError:
            await _terminate_process(process)
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
        if failure is not None:
            return TurnOutcome.failed(str(failure), message_cost_usd=state.cost_usd)
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
        if server is not None and not server.alive:
            await server.kill()
            self._server = None
            server = None
        if server is None:
            server = _ResidentServer(self.vendor, self.log)  # type: ignore[arg-type]
            try:
                await server.start(
                    self.workdir,
                    session_id=self.session_id,
                    model=prompt.model,
                    reasoning_effort=prompt.reasoning_effort,
                )
            except FileNotFoundError:
                await server.kill()
                return TurnOutcome.failed(f"The {self.vendor.binary} CLI is not installed.")
            except (CliServerDied, CliPromptTimeout) as error:
                # A server that dies or never answers its handshake before the
                # turn even starts is a failed turn with the real reason — not
                # an exception out of run_prompt.
                await server.kill()
                return TurnOutcome.failed(str(error), message_cost_usd=state.cost_usd)
            except asyncio.CancelledError:
                await server.kill()
                raise
            self._server = server
        server.drop_queued_notifications()
        self._abort_requested = False
        vendor: ResidentCliVendor = self.vendor  # type: ignore[assignment]
        failure: Exception | None = None
        loop = asyncio.get_running_loop()
        turn_deadline = loop.time() + budget

        def turn_request_timeout() -> float:
            # A turn's own request may legitimately take a while — pi's prompt
            # acknowledgement only lands after its preflight, which can run a
            # full compaction model call — so it is bounded by the remaining
            # turn budget, not the fixed protocol default. The floor keeps the
            # outer budget timeout the one that fires at the deadline, so the
            # failure reads as the turn's budget rather than a missing reply.
            return max(turn_deadline - loop.time(), 1.0)

        try:
            async with asyncio.timeout_at(turn_deadline):
                # One setup request at a time, each seeing the latest adopted
                # id; the cap turns a vendor that never converges into a turn
                # failure instead of a loop.
                recovered_contention = False
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
                    response_failure = _response_error(response)
                    if (
                        response_failure is not None
                        and not recovered_contention
                        and vendor.setup_lock_contention(response_failure)
                    ):
                        # Another live process still owns the conversation
                        # (an orphan this bridge never reaped); the request
                        # can only succeed once it is dead.
                        recovered_contention = True
                        response = await self._sweep_lock_holder_and_retry(server, message)
                        response_failure = _response_error(response)
                    if response_failure is not None:
                        return TurnOutcome.failed(
                            f"{_request_label(message)}: {response_failure}",
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
                response = await server.request(start, timeout=turn_request_timeout())
                response_failure = _response_error(response)
                if response_failure is not None and vendor.turn_start_busy_failure(
                    response_failure
                ):
                    # The previous turn settled on its final answer while the
                    # vendor was still finishing run work (a retry, a
                    # compaction); it rejects a new prompt until that run
                    # truly ends. The run's remaining events belong to the
                    # settled turn, so they are dropped, and the start is
                    # resubmitted once it is over.
                    await self._drop_until_previous_run_settled(server, vendor)
                    response = await server.request(start, timeout=turn_request_timeout())
                    response_failure = _response_error(response)
                if response_failure is not None:
                    return TurnOutcome.failed(
                        f"{_request_label(start)}: {response_failure}",
                        message_cost_usd=state.cost_usd,
                    )
                await self._consume_server_notifications(server, state, emit)
        except (TimeoutError, CliPromptTimeout) as error:
            failure = (
                error
                if isinstance(error, CliPromptTimeout)
                else CliPromptTimeout("The turn exceeded its time budget.")
            )
        except CliInactivityTimeout as error:
            failure = error
        except CliServerDied as error:
            failure = error
        except asyncio.CancelledError:
            await server.kill()
            raise
        if failure is not None:
            # The server may still be mid-turn; the next turn restarts it.
            await server.kill()
            self._server = None
            return TurnOutcome.failed(str(failure), message_cost_usd=state.cost_usd)
        await emit(step_finish_event(state, reason="completed"))
        return vendor.exit_outcome(state, 0, "")

    async def _consume_server_notifications(
        self, server: _ResidentServer, state: CliTurnState, emit: EventSink
    ) -> None:
        vendor: ResidentCliVendor = self.vendor  # type: ignore[assignment]
        while True:
            wait = self.limits.inactivity_timeout_seconds
            if state.final_message_seen_at is not None:
                grace = getattr(vendor, "settle_after_final_message", 0.0) or 0.0
                remaining = grace - (time.monotonic() - state.final_message_seen_at)
                if remaining <= 0:
                    # The vendor's authoritative final answer landed; its own
                    # settle event is late or never coming (a post-turn hang),
                    # and the answer is not worth the inactivity budget.
                    return
                wait = min(wait, remaining)
            message = await server.next_notification(wait)
            if message is None:
                if state.final_message_seen_at is not None:
                    return
                raise CliInactivityTimeout(
                    f"No output for {self.limits.inactivity_timeout_seconds:.0f}s."
                )
            if message is _SERVER_EXITED:
                if state.final_message_seen_at is not None:
                    # The authoritative final answer already landed; a server
                    # that exits immediately after delivering it (observed on
                    # pi 0.87.1) must not fail a complete turn.
                    return
                error = await self._server_death_error(server)
                raise error
            # Reverse requests were already answered by the reader; here they
            # only feed the vendor's translation.
            try:
                events = vendor.parse_server_message(message, state)
            except CliTurnSettled as settled:
                for event in settled.events:
                    await emit(event)
                return
            for event in events:
                await emit(event)

    async def _drop_until_previous_run_settled(
        self, server: _ResidentServer, vendor: ResidentCliVendor
    ) -> None:
        """Discard a settled turn's trailing notifications until the vendor
        reports its run truly over.

        A busy turn start means the previous run was still live when that
        turn settled; everything it emits from then on belongs to the turn
        the client already has the answer for, and only the vendor's
        end-of-run notification means a fresh prompt will be accepted.
        """
        while True:
            message = await server.next_notification(self.limits.inactivity_timeout_seconds)
            if message is None:
                raise CliInactivityTimeout(
                    f"No output for {self.limits.inactivity_timeout_seconds:.0f}s while "
                    f"the {self.id.value} agent finished its previous run."
                )
            if message is _SERVER_EXITED:
                error = await self._server_death_error(server)
                raise error
            if vendor.previous_run_settled(message):
                return

    async def _sweep_lock_holder_and_retry(
        self, server: _ResidentServer, message: dict[str, Any]
    ) -> dict[str, Any]:
        """Kill the process holding the conversation's lock, then retry the
        setup request that the lock rejection failed.

        The sweep spares the live server's own process group: the server
        that could not take the lock is not the process holding it.
        """
        vendor: ResidentCliVendor = self.vendor  # type: ignore[assignment]
        killed = await kill_orphaned_resident_servers(
            self.vendor.binary,
            _resident_server_args(vendor),
            exclude_pgid=server.pgid,
        )
        if killed:
            self.log.info(
                f"{self.vendor.id.value}.server.lock_sweep",
                killed_pids=killed,
            )
        return await server.request(message)

    async def _server_death_error(self, server: _ResidentServer) -> CliServerDied:
        """The failure describing a server that died mid-turn, naming a
        reader failure (the process may still be running) over an exit."""
        # exit_details bounds its wait for a just-exited process, so the code
        # named below is the real one whenever the OS has it.
        await server.exit_details()
        return CliServerDied(server.death_detail())

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
            except ValueError as error:
                # A frame larger than the stream limit; the vendor is still
                # running but its output can no longer be read. Name the real
                # cause — the caller terminates the process.
                raise CliServerDied(
                    f"The {self.id.value} output stream failed mid-turn "
                    f"(frame exceeded {PROTOCOL_STREAM_LIMIT_BYTES} bytes: {error})."
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
        await _terminate_process(self._process)
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
                await _terminate_process(process)
        except TimeoutError:
            return False
        return process.returncode is not None


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
