"""Supervisor half of the subprocess CLI harnesses (Codex, Pi, dsh, ZCode).

Like ``ClaudeStager``, there is no resident vendor process: every turn is a
fresh child spawned by the bridge. ``start()`` only prepares what the child
needs before the first prompt — the standalone bin scripts the agent calls
from Bash — and writes a small handoff telling the bridge which workspace the
supervisor chose. Vendor credentials and session stores stay wherever the
vendor keeps them under ``$HOME``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .constants import CLI_HARNESS_FILE_PATH
from .sandbox_bin import install_bin_scripts

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .repo_config import RepoEntry
    from .runtime_config import CliStagerConfig


@dataclass(frozen=True)
class CliHarnessHandoff:
    """What the supervisor decided and the bridge must use. Never a secret."""

    workdir: Path
    has_repository: bool

    def write(self, path: Path = Path(CLI_HARNESS_FILE_PATH)) -> None:
        payload = json.dumps({"workdir": str(self.workdir), "hasRepository": self.has_repository})
        staging = path.with_name(path.name + ".tmp")
        descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(payload)
        staging.replace(path)

    @classmethod
    def read(cls, path: Path = Path(CLI_HARNESS_FILE_PATH)) -> CliHarnessHandoff:
        data = json.loads(path.read_text())
        return cls(
            workdir=Path(str(data["workdir"])),
            has_repository=bool(data.get("hasRepository")),
        )


class CliStager:
    """``HarnessProcessOwner`` for the subprocess CLI harnesses; no child of its own."""

    def __init__(
        self,
        config: CliStagerConfig,
        log: Any,
        *,
        handoff_path: Path = Path(CLI_HARNESS_FILE_PATH),
    ) -> None:
        self.config = config
        self.log = log
        self.handoff_path = handoff_path
        self.started = False

    async def start(self, repositories: Sequence[RepoEntry], workdir: Path) -> None:
        self.log.info("cli.stage", workdir=str(workdir))
        install_bin_scripts(self.log)
        CliHarnessHandoff(
            workdir=workdir,
            has_repository=self.config.has_repository,
        ).write(self.handoff_path)
        self.started = True
        self.log.info("cli.staged", repo_count=len(repositories))

    async def stop(self) -> None:
        return None

    def exit_code(self) -> int | None:
        # There is no resident vendor process; each turn's child belongs to the bridge.
        return None
