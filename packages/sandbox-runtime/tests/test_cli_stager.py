"""CliStager: staging only, a handoff the bridge can read, registry wiring."""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sandbox_runtime.cli_stager import CliHarnessHandoff, CliStager
from sandbox_runtime.constants import BIN_INSTALL_DIR_ENV_VAR
from sandbox_runtime.entrypoint import build_harness_process, managed_skills_destination
from sandbox_runtime.harness.base import HarnessId, HarnessProcessOwner
from sandbox_runtime.runtime_config import CliStagerConfig, RuntimeConfig


def _config(harness: str) -> RuntimeConfig:
    return RuntimeConfig.from_env(
        {"SANDBOX_ID": "sbx", "SESSION_CONFIG": json.dumps({"harness": harness})}
    )


def test_handoff_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "handoff.json"
    CliHarnessHandoff(workdir=Path("/workspace"), has_repository=True).write(path)
    assert CliHarnessHandoff.read(path) == CliHarnessHandoff(
        workdir=Path("/workspace"), has_repository=True
    )
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_start_stages_bin_scripts_and_handoff(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(BIN_INSTALL_DIR_ENV_VAR, str(tmp_path / "bin"))
    stager = CliStager(
        CliStagerConfig(has_repository=False),
        MagicMock(),
        handoff_path=tmp_path / "handoff.json",
    )
    assert isinstance(stager, HarnessProcessOwner)
    assert stager.exit_code() is None
    await stager.start([], tmp_path / "workspace")
    assert stager.started
    handoff = CliHarnessHandoff.read(tmp_path / "handoff.json")
    assert handoff.workdir == tmp_path / "workspace"
    assert handoff.has_repository is False


@pytest.mark.parametrize("harness", ["codex", "pi", "dsh", "zcode"])
def test_build_harness_process_selects_cli_stager(harness: str) -> None:
    process = build_harness_process(
        _config(harness), asyncio.Event(), MagicMock(), MagicMock(), None
    )
    assert isinstance(process, CliStager)


@pytest.mark.parametrize(
    ("harness", "suffix"),
    [
        (HarnessId.CODEX, ".codex/skills"),
        (HarnessId.PI, ".pi/agent/skills"),
        (HarnessId.DSH, ".dsh/skills"),
        (HarnessId.ZCODE, ".zcode/skills"),
    ],
)
def test_managed_skills_destination(harness: HarnessId, suffix: str) -> None:
    path = managed_skills_destination(harness, None)
    assert path == Path.home() / suffix
