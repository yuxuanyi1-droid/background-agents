"""Minimal configuration for the Daytona snapshot bootstrap script."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# The snapshot class Daytona's own "daytona-large" general snapshot ships.
# Session sandboxes inherit the snapshot's resources, so these three numbers
# decide the class of every sandbox this deployment spawns; the org's total
# memory cap must fit at least one of them plus whatever else runs concurrently.
DEFAULT_BASE_SNAPSHOT_CPU_CORES = 4
DEFAULT_BASE_SNAPSHOT_MEMORY_GIB = 8
DEFAULT_BASE_SNAPSHOT_DISK_GIB = 10


@dataclass(frozen=True)
class BaseSnapshotResources:
    """Resources stamped on every built snapshot; sandboxes inherit them."""

    cpu_cores: int
    memory_gib: int
    disk_gib: int


@dataclass(frozen=True)
class DaytonaBootstrapConfig:
    """Configuration needed by bootstrap.py."""

    api_key: str
    api_url: str | None
    target: str | None
    base_snapshot: str
    resources: BaseSnapshotResources
    repo_root: Path


def _positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    try:
        parsed = int(value) if value not in (None, "") else default
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer") from error
    if parsed <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return parsed


def load_config() -> DaytonaBootstrapConfig:
    """Load bootstrap configuration from environment variables."""
    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        raise RuntimeError("DAYTONA_API_KEY is required")

    base_snapshot = os.environ.get("DAYTONA_BASE_SNAPSHOT")
    if not base_snapshot:
        raise RuntimeError("DAYTONA_BASE_SNAPSHOT is required")

    resources = BaseSnapshotResources(
        cpu_cores=_positive_int_env("DAYTONA_BASE_SNAPSHOT_CPU", DEFAULT_BASE_SNAPSHOT_CPU_CORES),
        memory_gib=_positive_int_env(
            "DAYTONA_BASE_SNAPSHOT_MEMORY_GIB", DEFAULT_BASE_SNAPSHOT_MEMORY_GIB
        ),
        disk_gib=_positive_int_env(
            "DAYTONA_BASE_SNAPSHOT_DISK_GIB", DEFAULT_BASE_SNAPSHOT_DISK_GIB
        ),
    )

    repo_root = Path(os.environ.get("OPEN_INSPECT_REPO_ROOT", Path(__file__).resolve().parents[3]))

    return DaytonaBootstrapConfig(
        api_key=api_key,
        api_url=os.environ.get("DAYTONA_API_URL") or None,
        target=os.environ.get("DAYTONA_TARGET") or None,
        base_snapshot=base_snapshot,
        resources=resources,
        repo_root=repo_root,
    )
