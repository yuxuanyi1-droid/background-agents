"""The generic CliHarness: process lifecycle, streaming, timeout, cancellation."""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from sandbox_runtime.harness import cli_harness
from sandbox_runtime.harness.base import (
    HarnessId,
    HarnessPrompt,
    HarnessStartError,
    PromptLimits,
    TurnOutcome,
)
from sandbox_runtime.harness.cli_harness import (
    CliHarness,
    CliInactivityTimeout,
    CliTurnSettled,
    CliTurnState,
    append_text_events,
    step_start_events,
)
from sandbox_runtime.harness.cli_vendors import CodexVendor, PiVendor


class _ScriptedVendor:
    """A CLI vendor whose binary is the test interpreter running a script."""

    id = HarnessId.PI
    json_stream = True

    def __init__(self, script: str, binary: str = sys.executable) -> None:
        self.script = script
        self.binary = binary

    def initial_session_id(self) -> str | None:
        return "chosen-1"

    def build_argv(self, **_: Any) -> list[str]:
        return ["-c", self.script]

    def extra_env(self, *, model: str | None) -> dict[str, str]:
        return {"EXTRA_MARKER": model or ""}

    def prepare(self, custom_providers: tuple[Any, ...]) -> None:
        return None

    def parse_record(self, record: dict[str, Any], state: CliTurnState) -> list[dict[str, Any]]:
        if record.get("type") == "text":
            return append_text_events(state, str(record["delta"]))
        if record.get("type") == "end":
            return step_start_events(state)
        return []

    def exit_outcome(
        self, state: CliTurnState, returncode: int | None, stderr_tail: str
    ) -> TurnOutcome:
        if state.error:
            return TurnOutcome.failed(state.error, message_cost_usd=state.cost_usd)
        if returncode != 0:
            return TurnOutcome.failed(stderr_tail or f"exited {returncode}")
        if not state.text:
            return TurnOutcome.failed("no output")
        return TurnOutcome.ok(message_cost_usd=state.cost_usd)


LIMITS = PromptLimits(
    inactivity_timeout_seconds=5.0,
    prompt_max_duration_seconds=10.0,
    prompt_cleanup_timeout_seconds=1.0,
)


def _harness(vendor: _ScriptedVendor, tmp_path: Path) -> CliHarness:
    return CliHarness(
        vendor=vendor,  # type: ignore[arg-type]
        log=MagicMock(),
        limits=LIMITS,
        workdir=tmp_path,
        has_repository=True,
    )


async def _run(harness: CliHarness) -> tuple[Any, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        events.append(event)

    outcome = await harness.run_prompt(HarnessPrompt(message_id="m1", text="hi"), emit)
    return outcome, events


@pytest.mark.asyncio
async def test_streams_jsonl_into_cumulative_tokens(tmp_path: Path) -> None:
    script = (
        "import sys\n"
        'for line in [\'{"type":"text","delta":"Hel"}\','
        '\'{"type":"text","delta":"lo"}\',\'{"type":"end"}\']:\n'
        "    print(line, flush=True)\n"
    )
    harness = _harness(_ScriptedVendor(script), tmp_path)
    await harness.create_session()
    outcome, events = await _run(harness)
    assert outcome.success
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert tokens == ["Hel", "Hello"]
    assert events[-1]["type"] == "step_finish"
    assert harness.session_id == "chosen-1"


@pytest.mark.asyncio
async def test_nonzero_exit_fails_with_stderr_tail(tmp_path: Path) -> None:
    script = "import sys\nprint('boom', file=sys.stderr, flush=True)\nsys.exit(3)\n"
    outcome, _ = await _run(_harness(_ScriptedVendor(script), tmp_path))
    assert not outcome.success
    assert "boom" in (outcome.error or "")


@pytest.mark.asyncio
async def test_inactivity_timeout_kills_the_turn(tmp_path: Path) -> None:
    script = 'import time\ntime.sleep(5)\nprint(\'{"type":"end"}\', flush=True)\n'
    limits = PromptLimits(
        inactivity_timeout_seconds=0.2,
        prompt_max_duration_seconds=10.0,
        prompt_cleanup_timeout_seconds=1.0,
    )
    harness = CliHarness(
        vendor=_ScriptedVendor(script),  # type: ignore[arg-type]
        log=MagicMock(),
        limits=limits,
        workdir=tmp_path,
        has_repository=False,
    )
    outcome, _ = await _run(harness)
    assert not outcome.success
    assert "No output" in (outcome.error or "")


@pytest.mark.asyncio
async def test_one_shot_eof_wait_is_bounded_by_the_budget(tmp_path: Path) -> None:
    # A process that closes stdout but keeps running used to leave the turn
    # parked on process.wait() indefinitely — outside the turn's budget.
    script = "import os, time\nos.close(1)\ntime.sleep(30)\n"
    limits = PromptLimits(
        inactivity_timeout_seconds=5.0,
        prompt_max_duration_seconds=0.3,
        prompt_cleanup_timeout_seconds=1.0,
    )
    harness = CliHarness(
        vendor=_ScriptedVendor(script),  # type: ignore[arg-type]
        log=MagicMock(),
        limits=limits,
        workdir=tmp_path,
        has_repository=False,
    )
    async with asyncio.timeout(5):
        outcome, _ = await _run(harness)
    assert not outcome.success
    assert "time budget" in (outcome.error or "")


@pytest.mark.asyncio
async def test_one_shot_oversize_frame_names_the_stream_not_a_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A one-shot frame past the stream limit raises ValueError out of the
    # reader; it must become a failed turn naming the stream (and kill the
    # process), not an exception out of run_prompt.
    monkeypatch.setattr(cli_harness, "PROTOCOL_STREAM_LIMIT_BYTES", 1024)
    script = "import sys, time\nsys.stdout.write('x' * 4096 + chr(10))\nsys.stdout.flush()\ntime.sleep(30)\n"
    harness = _harness(_ScriptedVendor(script), tmp_path)
    async with asyncio.timeout(10):
        outcome, _ = await _run(harness)
    assert not outcome.success
    error = outcome.error or ""
    assert "stream failed mid-turn" in error and "1024" in error


@pytest.mark.asyncio
async def test_missing_binary_is_a_deterministic_start_error(tmp_path: Path) -> None:
    harness = _harness(_ScriptedVendor("", binary="definitely-not-a-real-binary-xyz"), tmp_path)
    with pytest.raises(HarnessStartError):
        await harness.open()


@pytest.mark.asyncio
async def test_abort_terminates_the_running_child(tmp_path: Path) -> None:
    script = "import time\ntime.sleep(30)\n"
    harness = _harness(_ScriptedVendor(script), tmp_path)
    task = asyncio.create_task(_run(harness))
    await asyncio.sleep(0.3)
    assert await harness.abort() is True
    outcome, _ = await task
    assert not outcome.success


def test_protocol_contract_holder() -> None:
    # Guard against accidental removal of the timeout type used by the reader.
    assert issubclass(CliInactivityTimeout, Exception)


# --- Resident protocol mode ---------------------------------------------------
#
# A scripted vendor speaks a synthetic line protocol through the test
# interpreter: `ensure` adopts a session id, `prompt` streams deltas, asks one
# reverse request (and waits for the reply), then settles with `done`.


class _ResidentScriptVendor(_ScriptedVendor):
    """A resident vendor whose protocol server is the test interpreter."""

    resident = True
    jsonrpc = False

    def __init__(self, script: str) -> None:
        super().__init__(script)
        self._live_session: str | None = None

    def server_argv(
        self,
        *,
        session_id: str | None,
        model: str | None,
        reasoning_effort: str | None,
    ) -> list[str]:
        return ["-c", self.script]

    def reset(self) -> None:
        self._live_session = None

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
        if session_id and session_id == self._live_session:
            return None
        return {"type": "ensure"}

    def adopt_response_id(self, response: dict[str, Any]) -> str | None:
        result = response.get("result")
        session = result.get("session") if isinstance(result, dict) else None
        if isinstance(session, str) and session:
            self._live_session = session
            return session
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
        return {"type": "prompt", "text": prompt_text}

    def parse_server_message(self, message: dict[str, Any], state: CliTurnState) -> list[Any]:
        if message.get("type") == "delta":
            return append_text_events(state, str(message.get("text") or ""))
        if message.get("type") == "done":
            raise CliTurnSettled(step_start_events(state))
        return []

    def interrupt_messages(
        self, *, session_id: str | None, state: CliTurnState | None
    ) -> list[dict[str, Any]]:
        return [{"type": "interrupt"}]

    def server_request_messages(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        request_id = message.get("id")
        if isinstance(request_id, str) and isinstance(message.get("method"), str):
            return [{"id": request_id, "result": {}}]
        return []

    # No vendor-specific failure recovery in the scripted dialect: a rejected
    # setup or turn-start request is final.
    def setup_lock_contention(self, failure: str) -> bool:
        return False

    def turn_start_busy_failure(self, failure: str) -> bool:
        return False

    def previous_run_settled(self, message: dict[str, Any]) -> bool:
        return False


_SERVER_HEAD = (
    "import json, sys\n"
    "def send(obj):\n"
    "    print(json.dumps(obj), flush=True)\n"
    "for line in sys.stdin:\n"
    "    message = json.loads(line)\n"
    "    kind = message.get('type')\n"
    "    if kind == 'ensure':\n"
    "        send({'id': message['id'], 'result': {'session': 's-1'}})\n"
)

_SERVER_INTERRUPT_ARM = (
    "    elif kind == 'interrupt':\n"
    "        send({'id': message['id'], 'result': {'stopped': True}})\n"
)

_SERVER_SCRIPT = (
    _SERVER_HEAD + "    elif kind == 'prompt':\n"
    "        send({'id': message['id'], 'result': {'accepted': True}})\n"
    "        send({'type': 'delta', 'text': 'Hel'})\n"
    "        send({'type': 'delta', 'text': 'lo'})\n"
    "        send({'id': 'server-1', 'method': 'ask'})\n"
    "        reply = json.loads(sys.stdin.readline())\n"
    "        if reply.get('id') == 'server-1':\n"
    "            send({'type': 'done'})\n" + _SERVER_INTERRUPT_ARM
)


@pytest.fixture(autouse=True)
def _no_one_shot_escape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OI_CLI_ONE_SHOT", raising=False)


@pytest.mark.asyncio
async def test_resident_turn_streams_settles_and_reuses_the_process(tmp_path: Path) -> None:
    harness = _harness(_ResidentScriptVendor(_SERVER_SCRIPT), tmp_path)
    await harness.create_session()
    outcome, events = await _run(harness)
    assert outcome.success
    assert [e["content"] for e in events if e["type"] == "token"] == ["Hel", "Hello"]
    assert events[-1]["type"] == "step_finish"
    assert harness.session_id == "s-1"
    server = harness._server
    assert server is not None and server.alive

    outcome, events = await _run(harness)
    assert outcome.success
    assert harness._server is server
    assert [e["content"] for e in events if e["type"] == "token"] == ["Hel", "Hello"]
    await harness.close()
    assert harness._server is None


@pytest.mark.asyncio
async def test_resident_server_restarts_after_death(tmp_path: Path) -> None:
    harness = _harness(_ResidentScriptVendor(_SERVER_SCRIPT), tmp_path)
    await harness.create_session()
    assert (await _run(harness))[0].success
    dead = harness._server
    assert dead is not None
    await dead.kill()

    outcome, _ = await _run(harness)
    assert outcome.success
    assert harness._server is not None and harness._server is not dead
    # The restarted process resumed the adopted session id.
    assert harness.session_id == "s-1"
    await harness.close()


@pytest.mark.asyncio
async def test_resident_turn_fails_on_an_error_response(tmp_path: Path) -> None:
    script = (
        _SERVER_HEAD + "    elif kind == 'prompt':\n"
        "        send({'id': message['id'], 'error': {'message': 'rejected by gateway'}})\n"
    )
    harness = _harness(_ResidentScriptVendor(script), tmp_path)
    await harness.create_session()
    outcome, _ = await _run(harness)
    assert not outcome.success
    assert "rejected by gateway" in (outcome.error or "")
    await harness.close()


@pytest.mark.asyncio
async def test_resident_setup_converging_on_the_last_request_still_runs(
    tmp_path: Path,
) -> None:
    # A converging vendor reports completion only on the call after its final
    # message; when that final message is also the limit-th request, setup
    # must still succeed.
    class _ExactSetupVendor(_ResidentScriptVendor):
        def __init__(self, script: str) -> None:
            super().__init__(script)
            self.remaining = cli_harness.SETUP_REQUEST_LIMIT
            self.calls = 0

        def next_setup_message(self, **_: Any) -> dict[str, Any] | None:
            self.calls += 1
            if self.remaining <= 0:
                return None
            self.remaining -= 1
            return {"type": "ensure"}

    script = (
        _SERVER_HEAD + "    elif kind == 'prompt':\n"
        "        send({'id': message['id'], 'result': {'accepted': True}})\n"
        "        send({'type': 'delta', 'text': 'ok'})\n"
        "        send({'type': 'done'})\n"
    )
    vendor = _ExactSetupVendor(script)
    harness = _harness(vendor, tmp_path)
    await harness.create_session()
    outcome, _ = await _run(harness)
    assert outcome.success
    assert vendor.calls == cli_harness.SETUP_REQUEST_LIMIT + 1
    await harness.close()


@pytest.mark.asyncio
async def test_resident_setup_past_the_request_limit_fails_without_extra_requests(
    tmp_path: Path,
) -> None:
    # The cap bounds requests, not observations: the request-free iteration
    # after the limit-th request is what proves the vendor never converges.
    class _NeverConvergingVendor(_ResidentScriptVendor):
        def __init__(self, script: str) -> None:
            super().__init__(script)
            self.calls = 0
            self.adopted = 0

        def next_setup_message(self, **_: Any) -> dict[str, Any] | None:
            self.calls += 1
            return {"type": "ensure"}

        def adopt_response_id(self, response: dict[str, Any]) -> str | None:
            self.adopted += 1
            return super().adopt_response_id(response)

    script = _SERVER_HEAD
    vendor = _NeverConvergingVendor(script)
    harness = _harness(vendor, tmp_path)
    await harness.create_session()
    outcome, _ = await _run(harness)
    assert not outcome.success
    assert "did not converge" in (outcome.error or "")
    assert vendor.adopted == cli_harness.SETUP_REQUEST_LIMIT
    assert vendor.calls == cli_harness.SETUP_REQUEST_LIMIT + 1
    await harness.close()


@pytest.mark.asyncio
async def test_resident_interrupt_settles_the_turn(tmp_path: Path) -> None:
    script = (
        _SERVER_HEAD + "    elif kind == 'prompt':\n"
        "        send({'id': message['id'], 'result': {'accepted': True}})\n"
        "        send({'type': 'delta', 'text': 'Hi'})\n"
        "        for parked in sys.stdin:\n"
        "            parked = json.loads(parked)\n"
        "            if parked.get('type') == 'interrupt':\n"
        "                send({'id': parked['id'], 'result': {'stopped': True}})\n"
        "                send({'type': 'done'})\n"
        "                break\n"
    )
    harness = _harness(_ResidentScriptVendor(script), tmp_path)
    await harness.create_session()
    events: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        events.append(event)

    task = asyncio.create_task(harness.run_prompt(HarnessPrompt(message_id="m1", text="hi"), emit))
    async with asyncio.timeout(5):
        while not any(event["type"] == "token" for event in events):
            await asyncio.sleep(0.05)
    assert await harness.stop_execution(5.0) is True
    outcome = await asyncio.wait_for(task, 5)
    assert outcome.success
    await harness.close()


@pytest.mark.asyncio
async def test_stop_execution_without_an_interrupt_channel_kills_the_server(
    tmp_path: Path,
) -> None:
    # A vendor with no interrupt request for this state (zcode before a
    # session id exists) cannot be asked to stop over the protocol; claiming
    # the stop succeeded would leave the turn running, so the fallback is
    # killing the server.
    script = (
        _SERVER_HEAD + "    elif kind == 'prompt':\n"
        "        send({'id': message['id'], 'result': {'accepted': True}})\n"
        "        send({'type': 'delta', 'text': 'Working'})\n"
        "        import time\n"
        "        time.sleep(600)\n"
    )

    class _NoInterruptVendor(_ResidentScriptVendor):
        def interrupt_messages(
            self, *, session_id: str | None, state: CliTurnState | None
        ) -> list[dict[str, Any]]:
            return []

    harness = _harness(_NoInterruptVendor(script), tmp_path)
    await harness.create_session()
    events: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        events.append(event)

    task = asyncio.create_task(harness.run_prompt(HarnessPrompt(message_id="m1", text="hi"), emit))
    async with asyncio.timeout(5):
        while not any(event["type"] == "token" for event in events):
            await asyncio.sleep(0.05)
    try:
        assert await harness.stop_execution(5.0) is True
        assert harness._server is None
        outcome = await asyncio.wait_for(task, 5)
        assert not outcome.success
    finally:
        await harness.close()
        if not task.done():
            task.cancel()


@pytest.mark.asyncio
async def test_resident_server_survives_a_chatty_stderr(tmp_path: Path) -> None:
    # A resident server that logs past the pipe buffer would deadlock without
    # the drain task.
    script = (
        _SERVER_HEAD + "    elif kind == 'prompt':\n"
        "        send({'id': message['id'], 'result': {'accepted': True}})\n"
        "        chunk = 'x' * 8192\n"
        "        for _ in range(48):\n"
        "            sys.stderr.write(chunk + chr(10))\n"
        "        sys.stderr.flush()\n"
        "        send({'type': 'delta', 'text': 'Drained'})\n"
        "        send({'type': 'done'})\n"
    )
    harness = _harness(_ResidentScriptVendor(script), tmp_path)
    await harness.create_session()
    outcome, events = await _run(harness)
    assert outcome.success
    assert events[-1]["type"] == "step_finish"
    await harness.close()


@pytest.mark.asyncio
async def test_resident_inactivity_timeout_kills_the_server(tmp_path: Path) -> None:
    script = (
        _SERVER_HEAD + "    elif kind == 'prompt':\n"
        "        send({'id': message['id'], 'result': {'accepted': True}})\n"
        "        send({'type': 'delta', 'text': 'Stuck'})\n"
        "        import time\n"
        "        time.sleep(600)\n"
    )
    limits = PromptLimits(
        inactivity_timeout_seconds=0.3,
        prompt_max_duration_seconds=10.0,
        prompt_cleanup_timeout_seconds=1.0,
    )
    harness = CliHarness(
        vendor=_ResidentScriptVendor(script),  # type: ignore[arg-type]
        log=MagicMock(),
        limits=limits,
        workdir=tmp_path,
        has_repository=False,
    )
    await harness.create_session()
    outcome, _ = await _run(harness)
    assert not outcome.success
    assert "No output" in (outcome.error or "")
    # The wedged server must not survive into the next turn.
    assert harness._server is None


@pytest.mark.asyncio
async def test_resident_server_death_fails_fast_with_exit_details(tmp_path: Path) -> None:
    # A server that dies mid-turn must fail the turn immediately with its exit
    # code and stderr tail, not idle to the inactivity budget.
    script = (
        _SERVER_HEAD + "    elif kind == 'prompt':\n"
        "        send({'id': message['id'], 'result': {'accepted': True}})\n"
        "        send({'type': 'delta', 'text': 'Half'})\n"
        "        import sys\n"
        "        print('fatal: heap exhausted', file=sys.stderr, flush=True)\n"
        "        sys.exit(3)\n"
    )
    harness = _harness(_ResidentScriptVendor(script), tmp_path)
    await harness.create_session()
    outcome, _ = await _run(harness)
    assert not outcome.success
    error = outcome.error or ""
    assert "exited mid-turn" in error and "code 3" in error
    assert "heap exhausted" in error


@pytest.mark.asyncio
async def test_resident_startup_exit_surfaces_the_stderr_reason(tmp_path: Path) -> None:
    # A server that dies before answering anything (pi with an unknown model
    # id exits at startup) leaves the reason only on stderr; the first setup
    # request must fail with that detail instead of a static "not running"
    # message or the request timeout.
    script = (
        "import sys\n"
        "print('Error: Model \"some/model\" not found. Use --list-models.', file=sys.stderr)\n"
        "sys.exit(1)\n"
    )
    harness = _harness(_ResidentScriptVendor(script), tmp_path)
    await harness.create_session()
    outcome, _ = await _run(harness)
    assert not outcome.success
    error = outcome.error or ""
    assert "code 1" in error and "Model" in error and "--list-models" in error


@pytest.mark.asyncio
async def test_request_on_a_dead_server_names_the_startup_failure(tmp_path: Path) -> None:
    # A server that already exited and was reaped before the request (reader
    # done) must still fail with the death details — exit code and stderr
    # reason — not the static "not running" message.
    script = (
        "import sys\n"
        "print('Error: Model \"some/model\" not found. Use --list-models.', file=sys.stderr)\n"
        "sys.exit(1)\n"
    )
    server = cli_harness._ResidentServer(_ResidentScriptVendor(script), MagicMock())
    await server.start(tmp_path, session_id=None, model="some/model", reasoning_effort=None)
    async with asyncio.timeout(5):
        while server.alive or not server.stderr_tail:
            await asyncio.sleep(0.02)
    with pytest.raises(cli_harness.CliServerDied) as excinfo:
        await server.request({"type": "ensure"})
    error = str(excinfo.value)
    assert "code 1" in error and "Model" in error and "--list-models" in error
    await server.kill()


@pytest.mark.asyncio
async def test_resident_server_survives_an_oversize_frame(tmp_path: Path) -> None:
    # asyncio's default 64 KiB readline cap used to turn one large protocol
    # frame into a phantom "server exited mid-turn (code None)".
    script = (
        _SERVER_HEAD + "    elif kind == 'prompt':\n"
        "        send({'id': message['id'], 'result': {'accepted': True}})\n"
        "        send({'type': 'delta', 'text': 'x' * (200 * 1024)})\n"
        "        send({'type': 'done'})\n"
    )
    harness = _harness(_ResidentScriptVendor(script), tmp_path)
    await harness.create_session()
    outcome, _ = await _run(harness)
    assert outcome.success, outcome.error


@pytest.mark.asyncio
async def test_resident_reader_overrun_names_the_stream_not_a_phantom_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If a frame ever does exceed the stream limit, the failure must say the
    # reader stopped while the server was still running — not claim an exit.
    script = (
        _SERVER_HEAD + "    elif kind == 'prompt':\n"
        "        send({'id': message['id'], 'result': {'accepted': True}})\n"
        "        send({'type': 'delta', 'text': 'y' * 2048})\n"
        "        import time\n"
        "        time.sleep(30)\n"
    )
    monkeypatch.setattr(cli_harness, "PROTOCOL_STREAM_LIMIT_BYTES", 1024)
    harness = _harness(_ResidentScriptVendor(script), tmp_path)
    await harness.create_session()
    outcome, _ = await _run(harness)
    assert not outcome.success
    error = outcome.error or ""
    assert "stream failed mid-turn" in error
    assert "still running" in error
    assert "exited mid-turn" not in error
    # The wedged server must not survive into the next turn.
    assert harness._server is None


@pytest.mark.asyncio
async def test_resident_in_flight_server_death_settles_the_request(tmp_path: Path) -> None:
    # A server dying while a request is in flight (here: it exits instead of
    # answering the turn start) must settle that request with the death right
    # away — not idle out the request timeout with "No response to prompt".
    script = _SERVER_HEAD + "    elif kind == 'prompt':\n        sys.exit(3)\n"
    harness = _harness(_ResidentScriptVendor(script), tmp_path)
    await harness.create_session()
    async with asyncio.timeout(5):
        outcome, _ = await _run(harness)
    assert not outcome.success
    error = outcome.error or ""
    assert "exited mid-turn" in error and "code 3" in error
    assert "No response" not in error


@pytest.mark.asyncio
async def test_resident_handshake_wedge_is_a_turn_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A server that never answers its handshake must fail the turn naming the
    # request, not raise out of run_prompt.
    monkeypatch.setattr(cli_harness, "PROTOCOL_REQUEST_TIMEOUT_SECONDS", 0.3)

    class _WedgedVendor(_ResidentScriptVendor):
        def handshake_requests(self) -> list[dict[str, Any]]:
            return [{"type": "hello"}]

    harness = _harness(_WedgedVendor(_SERVER_HEAD), tmp_path)
    await harness.create_session()
    async with asyncio.timeout(5):
        outcome, _ = await _run(harness)
    assert not outcome.success
    assert "No response to hello" in (outcome.error or "")
    assert harness._server is None


@pytest.mark.asyncio
async def test_resident_missing_binary_is_a_turn_failure(tmp_path: Path) -> None:
    # A resident harness whose CLI vanished between open() and the turn must
    # return a start failure, not raise out of run_prompt.
    vendor = _ResidentScriptVendor("")
    vendor.binary = "definitely-not-a-real-binary-xyz"
    harness = _harness(vendor, tmp_path)
    await harness.create_session()
    outcome, _ = await _run(harness)
    assert not outcome.success
    assert "not installed" in (outcome.error or "")
    assert harness._server is None


@pytest.mark.asyncio
async def test_resident_stale_reader_server_is_restarted_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A reader failure can outlive its process: the server settles a turn,
    # then its stdout overruns (reader dies) while the process lingers. The
    # next turn must restart the server — talking to the zombie would idle
    # out the request timeout — and succeed.
    monkeypatch.setattr(cli_harness, "PROTOCOL_STREAM_LIMIT_BYTES", 1024)
    script = (
        _SERVER_HEAD + "    elif kind == 'prompt':\n"
        "        send({'id': message['id'], 'result': {'accepted': True}})\n"
        "        send({'type': 'delta', 'text': 'ok'})\n"
        "        send({'type': 'done'})\n"
        "        send({'type': 'delta', 'text': 'z' * 4096})\n"
        "        import time\n"
        "        time.sleep(30)\n"
    )
    harness = _harness(_ResidentScriptVendor(script), tmp_path)
    await harness.create_session()
    outcome, _ = await _run(harness)
    assert outcome.success
    stale = harness._server
    assert stale is not None
    await asyncio.sleep(0.2)  # let the reader hit the overrun

    async with asyncio.timeout(10):
        outcome, _ = await _run(harness)
    assert outcome.success
    assert harness._server is not None and harness._server is not stale
    await harness.close()


@pytest.mark.asyncio
async def test_resident_settles_after_the_final_message_grace(tmp_path: Path) -> None:
    # Pi semantics: a terminal assistant message is authoritative, so a
    # settle event that never arrives costs a bounded grace, not the whole
    # inactivity budget.
    script = (
        _SERVER_HEAD + "    elif kind == 'prompt':\n"
        "        send({'id': message['id'], 'result': {'accepted': True}})\n"
        "        send({'type': 'delta', 'text': 'Answer'})\n"
        "        send({'type': 'final_message'})\n"
        "        import time\n"
        "        time.sleep(600)\n"
    )

    class _GraceVendor(_ResidentScriptVendor):
        settle_after_final_message = 0.5

        def parse_server_message(self, message: dict[str, Any], state: CliTurnState) -> list[Any]:
            if message.get("type") == "final_message":
                state.final_message_seen_at = __import__("time").monotonic()
                return []
            return super().parse_server_message(message, state)

    harness = _harness(_GraceVendor(script), tmp_path)
    await harness.create_session()
    outcome, events = await _run(harness)
    assert outcome.success
    assert events[-1]["type"] == "step_finish"
    await harness.close()


@pytest.mark.asyncio
async def test_final_message_without_a_settle_grace_waits_for_the_terminal(tmp_path: Path) -> None:
    # The grace is opt-in (pi's 60s); a vendor that leaves it unset settles on
    # its own terminal notification, so trailing notifications after the final
    # message must still be consumed — not dropped by an implicit zero grace.
    script = (
        _SERVER_HEAD + "    elif kind == 'prompt':\n"
        "        send({'id': message['id'], 'result': {'accepted': True}})\n"
        "        send({'type': 'final_message'})\n"
        "        send({'type': 'delta', 'text': ' tail'})\n"
        "        send({'type': 'done'})\n"
    )

    class _NoGraceVendor(_ResidentScriptVendor):
        def parse_server_message(self, message: dict[str, Any], state: CliTurnState) -> list[Any]:
            if message.get("type") == "final_message":
                state.final_message_seen_at = __import__("time").monotonic()
                return []
            return super().parse_server_message(message, state)

    harness = _harness(_NoGraceVendor(script), tmp_path)
    await harness.create_session()
    outcome, events = await _run(harness)
    assert outcome.success
    tokens = [event["content"] for event in events if event["type"] == "token"]
    assert tokens and tokens[-1].endswith("tail")
    await harness.close()


@pytest.mark.asyncio
async def test_resident_reverse_request_during_setup_is_answered(tmp_path: Path) -> None:
    # ZCode asks for runtime preferences WHILE session/create is still being
    # awaited; the reply must come from the reader (the one place every
    # inbound message passes), not from a consume loop that only runs later.
    script = (
        "import json, sys\n"
        "def send(obj):\n"
        "    print(json.dumps(obj), flush=True)\n"
        "for line in sys.stdin:\n"
        "    message = json.loads(line)\n"
        "    kind = message.get('type')\n"
        "    if kind == 'ensure':\n"
        "        send({'id': 'server-9', 'method': 'ask'})\n"
        "        reply = json.loads(sys.stdin.readline())\n"
        "        if reply.get('id') == 'server-9':\n"
        "            send({'id': message['id'], 'result': {'session': 's-1'}})\n"
        "    elif kind == 'prompt':\n"
        "        send({'id': message['id'], 'result': {'accepted': True}})\n"
        "        send({'type': 'delta', 'text': 'Hi'})\n"
        "        send({'type': 'done'})\n"
    )
    harness = _harness(_ResidentScriptVendor(script), tmp_path)
    await harness.create_session()
    outcome, events = await _run(harness)
    assert outcome.success
    assert harness.session_id == "s-1"
    assert any(event["type"] == "token" and event["content"] == "Hi" for event in events)
    await harness.close()


@pytest.mark.asyncio
async def test_resident_start_sweeps_orphaned_servers(tmp_path: Path) -> None:
    # A bridge crash leaves its resident server alive (own process group), and
    # the orphan keeps the conversation's locks — the replacement server's
    # resume is rejected until it dies. Starting a new server sweeps it.
    import subprocess

    orphan = subprocess.Popen(
        [sys.executable, "-c", _SERVER_SCRIPT], stdin=subprocess.PIPE
    )  # stdin held open so the scripted server blocks forever
    assert orphan.poll() is None

    harness = _harness(_ResidentScriptVendor(_SERVER_SCRIPT), tmp_path)
    await harness.create_session()
    outcome, _ = await _run(harness)
    assert outcome.success
    assert orphan.wait(timeout=5) == -signal.SIGKILL
    await harness.close()


@pytest.mark.asyncio
async def test_resident_exit_after_final_answer_settles_successfully(tmp_path: Path) -> None:
    # pi 0.87.1 delivers the final message and exits immediately; a complete
    # turn must not be failed by the server's own death.

    script = (
        _SERVER_HEAD + "    elif kind == 'prompt':\n"
        "        send({'id': message['id'], 'result': {'accepted': True}})\n"
        "        send({'type': 'delta', 'text': 'The full answer.'})\n"
        "        send({'type': 'final_message'})\n"
        "        import sys\n"
        "        sys.exit(0)\n"
    )

    class _ExitVendor(_ResidentScriptVendor):
        settle_after_final_message = 0.5

        def parse_server_message(self, message: dict[str, Any], state: CliTurnState) -> list[Any]:
            if message.get("type") == "final_message":
                state.final_message_seen_at = __import__("time").monotonic()
                return []
            return super().parse_server_message(message, state)

    harness = _harness(_ExitVendor(script), tmp_path)
    await harness.create_session()
    outcome, events = await _run(harness)
    assert outcome.success
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert tokens[-1] == "The full answer."
    await harness.close()


# A pi rpc server whose first assistant response requests a tool and carries
# no text: the settle grace must not start on it, so the tool below may
# outlive the grace and the answer that follows still settles the turn.
_PI_RPC_SERVER = (
    "import json, sys, time\n"
    "def send(obj):\n"
    "    print(json.dumps(obj), flush=True)\n"
    "for line in sys.stdin:\n"
    "    message = json.loads(line)\n"
    "    kind = message.get('type')\n"
    "    if kind == 'get_state':\n"
    "        send({'id': message['id'], 'type': 'response', 'command': 'get_state',"
    " 'success': True, 'data': {'sessionId': 'p-1'}})\n"
    "    elif kind == 'prompt':\n"
    "        send({'id': message['id'], 'type': 'response', 'command': 'prompt', 'success': True})\n"
    "        send({'type': 'message_end', 'message': {'role': 'assistant', 'stopReason': 'toolUse',"
    " 'content': [{'type': 'toolCall', 'id': 'c1', 'name': 'bash',"
    " 'arguments': {'command': 'make'}}]}})\n"
    "        time.sleep(1.0)\n"
    "        send({'type': 'message_end', 'message': {'role': 'assistant', 'stopReason': 'stop',"
    " 'content': [{'type': 'text', 'text': 'The build passed.'}]}})\n"
    "        send({'type': 'agent_settled'})\n"
)


class _ScriptedPiVendor(PiVendor):
    """PiVendor over a scripted server that speaks pi's rpc dialect."""

    settle_after_final_message = 0.3

    def __init__(self, script: str) -> None:
        super().__init__()
        self.binary = sys.executable
        self._script = script

    def server_argv(
        self,
        *,
        session_id: str | None,
        model: str | None,
        reasoning_effort: str | None,
    ) -> list[str]:
        return ["-c", self._script]


@pytest.mark.asyncio
async def test_pi_tool_message_does_not_settle_the_turn_mid_step(tmp_path: Path) -> None:
    harness = _harness(_ScriptedPiVendor(_PI_RPC_SERVER), tmp_path)
    await harness.create_session()
    outcome, events = await _run(harness)
    assert outcome.success, outcome.error
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert tokens == ["The build passed."]
    await harness.close()


# A pi rpc server whose prompt arrives while the previous run is still live:
# pi rejects it without a streamingBehavior, the run's tail then drains, and
# only the resubmitted prompt may be answered.
_PI_BUSY_SERVER = (
    "import json, sys, time\n"
    "def send(obj):\n"
    "    print(json.dumps(obj), flush=True)\n"
    "busy = True\n"
    "for line in sys.stdin:\n"
    "    message = json.loads(line)\n"
    "    kind = message.get('type')\n"
    "    if kind == 'get_state':\n"
    "        send({'id': message['id'], 'type': 'response', 'command': 'get_state',"
    " 'success': True, 'data': {'sessionId': 'p-1'}})\n"
    "    elif kind == 'prompt':\n"
    "        if busy:\n"
    "            busy = False\n"
    "            send({'id': message['id'], 'type': 'response', 'command': 'prompt',"
    " 'success': False, 'error': \"Agent is already processing. Specify"
    " streamingBehavior ('steer' or 'followUp') to queue the message.\"})\n"
    "            send({'type': 'message_end', 'message': {'role': 'assistant',"
    " 'stopReason': 'stop', 'content': [{'type': 'text',"
    " 'text': 'stale previous run answer'}]}})\n"
    "            time.sleep(0.2)\n"
    "            send({'type': 'agent_settled'})\n"
    "        else:\n"
    "            send({'id': message['id'], 'type': 'response', 'command': 'prompt',"
    " 'success': True})\n"
    "            send({'type': 'message_end', 'message': {'role': 'assistant',"
    " 'stopReason': 'stop', 'content': [{'type': 'text', 'text': 'The fresh answer.'}]}})\n"
    "            send({'type': 'agent_settled'})\n"
)


@pytest.mark.asyncio
async def test_pi_busy_prompt_waits_for_the_previous_run_and_resubmits(tmp_path: Path) -> None:
    harness = _harness(_ScriptedPiVendor(_PI_BUSY_SERVER), tmp_path)
    await harness.create_session()
    outcome, events = await _run(harness)
    assert outcome.success, outcome.error
    # The rejection's tail belongs to the turn that already settled and must
    # be dropped; only the resubmitted prompt's answer reaches the client.
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert tokens == ["The fresh answer."]
    await harness.close()


@pytest.mark.asyncio
async def test_turn_start_requests_are_bounded_by_the_remaining_turn_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pi's prompt acknowledgement lands only after its preflight, which can run
    # a full compaction model call, so the turn's own requests ride what is
    # left of the turn budget instead of the fixed protocol timeout. Setup
    # requests (a pure protocol op) keep the default.
    recorded: list[float | None] = []
    original = cli_harness._ResidentServer.request

    async def recording_request(
        server: Any, payload: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any]:
        recorded.append(timeout)
        return await original(server, payload, timeout)

    monkeypatch.setattr(cli_harness._ResidentServer, "request", recording_request)

    harness = _harness(_ScriptedPiVendor(_PI_BUSY_SERVER), tmp_path)
    await harness.create_session()
    events: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        events.append(event)

    outcome = await harness.run_prompt(
        HarnessPrompt(message_id="m1", text="hi", max_duration_seconds=120.0), emit
    )
    assert outcome.success, outcome.error
    # get_state (setup), then the rejected prompt and its resubmission.
    assert recorded[0] is None
    assert recorded[1] is not None and 100.0 < recorded[1] <= 120.0
    # The resubmission recomputes against the deadline, so it may only shrink.
    assert recorded[2] is not None and 100.0 < recorded[2] <= recorded[1]
    await harness.close()


def _vendor_shim(tmp_path: Path, name: str) -> Path:
    """An executable named like a vendor CLI, the shape an npm bin shim has."""
    shim = tmp_path / name
    shim.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(600)\n")
    shim.chmod(0o755)
    return shim


@pytest.mark.asyncio
async def test_orphan_sweep_matches_the_shim_shape_and_spares_other_processes(
    tmp_path: Path,
) -> None:
    # The install execs npm shims, so the process shows the shim path after
    # the interpreter — never the bare binary name a prefix match requires —
    # and the sweep must only claim processes whose argv names this vendor.
    import subprocess

    orphan = subprocess.Popen(
        [str(_vendor_shim(tmp_path, "sweeptool")), "serve", "--forever"],
        stdin=subprocess.PIPE,
    )
    bystander = subprocess.Popen(
        [str(_vendor_shim(tmp_path, "othertool")), "serve", "--forever"],
        stdin=subprocess.PIPE,
    )
    try:
        killed = await cli_harness.kill_orphaned_resident_servers(
            "sweeptool", ["serve", "--forever"]
        )
        assert orphan.pid in killed
        assert bystander.pid not in killed
        assert orphan.wait(timeout=5) == -signal.SIGKILL
        assert bystander.poll() is None
    finally:
        bystander.kill()
        bystander.wait(timeout=5)


@pytest.mark.asyncio
async def test_resident_kill_stops_the_whole_server_process_group(tmp_path: Path) -> None:
    # Codex's npm launcher spawns the native app-server as its child and
    # forwards only SIGINT/SIGTERM/SIGHUP; killing the launcher alone would
    # leave the real server — holding the conversation's writer lock —
    # running.
    child_pid_file = tmp_path / "child.pid"
    script = (
        "import json, subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        f"open({str(child_pid_file)!r}, 'w').write(str(child.pid))\n"
        "def send(obj):\n"
        "    print(json.dumps(obj), flush=True)\n"
        "for line in sys.stdin:\n"
        "    message = json.loads(line)\n"
        "    kind = message.get('type')\n"
        "    if kind == 'ensure':\n"
        "        send({'id': message['id'], 'result': {'session': 's-1'}})\n"
        "    elif kind == 'prompt':\n"
        "        send({'id': message['id'], 'result': {'accepted': True}})\n"
        "        send({'type': 'delta', 'text': 'ok'})\n"
        "        send({'type': 'done'})\n"
    )
    harness = _harness(_ResidentScriptVendor(script), tmp_path)
    await harness.create_session()
    outcome, _ = await _run(harness)
    assert outcome.success
    child_pid = int(child_pid_file.read_text())
    assert not cli_harness._process_reaped(child_pid)
    await harness.close()
    deadline = asyncio.get_running_loop().time() + 5
    while not cli_harness._process_reaped(child_pid):
        assert asyncio.get_running_loop().time() < deadline, "server child survived the kill"
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_terminate_kills_group_members_after_the_child_was_reaped(
    tmp_path: Path,
) -> None:
    # A launcher can exit before the server it spawned, and the reaped child
    # makes os.getpgid unusable — but the survivors still hold the
    # conversation locks, so the group must be signalled anyway.
    child_pid_file = tmp_path / "child.pid"
    launcher = (
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        f"open({str(child_pid_file)!r}, 'w').write(str(child.pid))\n"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", launcher, start_new_session=True
    )
    assert await process.wait() == 0  # the launcher exits; its child outlives it
    child_pid = int(child_pid_file.read_text())
    assert not cli_harness._process_reaped(child_pid)
    await cli_harness._terminate_process(process)
    deadline = asyncio.get_running_loop().time() + 5
    while not cli_harness._process_reaped(child_pid):
        assert asyncio.get_running_loop().time() < deadline, "group member survived the kill"
        await asyncio.sleep(0.05)


class _ScriptedCodexVendor(CodexVendor):
    """CodexVendor over a scripted server that speaks the app-server dialect."""

    def __init__(self, script: str) -> None:
        super().__init__()
        self.binary = sys.executable
        self._script = script

    def server_argv(
        self,
        *,
        session_id: str | None,
        model: str | None,
        reasoning_effort: str | None,
    ) -> list[str]:
        return ["-c", self._script]


def _codex_lock_server(requests_log: Path) -> str:
    return (
        "import json, sys\n"
        "def send(obj):\n"
        "    print(json.dumps(obj), flush=True)\n"
        "resumed = False\n"
        "for line in sys.stdin:\n"
        "    message = json.loads(line)\n"
        "    method = message.get('method')\n"
        f"    open({str(requests_log)!r}, 'a').write(str(method) + '\\n')\n"
        "    if method == 'initialize':\n"
        "        send({'id': message['id'], 'result': {}})\n"
        "    elif method == 'thread/resume':\n"
        "        if not resumed:\n"
        "            resumed = True\n"
        "            send({'id': message['id'], 'error': {'code': -32603, 'message':"
        " 'thread %s already has an active writer' % message['params']['threadId']}})\n"
        "        else:\n"
        "            send({'id': message['id'], 'result':"
        " {'thread': {'id': message['params']['threadId']}}})\n"
        "    elif method == 'turn/start':\n"
        "        send({'id': message['id'], 'result': {}})\n"
        "        send({'method': 'item/completed', 'params': {'item': {'id': 'i-1',"
        " 'type': 'agentMessage', 'text': 'Done.'}}})\n"
        "        send({'method': 'turn/completed', 'params': {'turn': {'id': 't-1'}}})\n"
    )


@pytest.mark.asyncio
async def test_codex_resume_lock_contention_sweeps_and_retries(tmp_path: Path) -> None:
    # "Already has an active writer" means another live app-server owns the
    # thread; the resume is retried after the sweep, which must spare the
    # server that could not take the lock (it is not the holder). A persisted
    # id (resume_session) is what makes the first turn send thread/resume.
    requests_log = tmp_path / "requests.log"
    harness = _harness(_ScriptedCodexVendor(_codex_lock_server(requests_log)), tmp_path)
    await harness.resume_session("chosen-1")
    outcome, _ = await _run(harness)
    assert outcome.success, outcome.error
    assert harness.session_id == "chosen-1"
    assert requests_log.read_text().splitlines().count("thread/resume") == 2
    await harness.close()
