"""The generic CliHarness: process lifecycle, streaming, timeout, cancellation."""

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

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
            state.completed = True
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
            state.completed = True
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
