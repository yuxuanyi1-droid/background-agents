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
