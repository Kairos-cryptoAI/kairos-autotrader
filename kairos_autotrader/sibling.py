"""Running code that lives in a sibling Kairos repository.

Each Kairos layer has its own repository and its own locked environment. The live runner
does not copy their code: it starts a small worker script with that repository's own
Python, passing JSON in and out. This module is the one place where such a process is
started and where a repository's revision is read.
"""

from __future__ import annotations

import functools
import json
import os
import shutil

# Processes are started with fixed argument lists and never through a shell.
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

MARKER = "@@KAIROS-WORKER@@"  # every worker script prints this before its JSON answer
_PASSED_ENVIRONMENT = ("SYSTEMROOT", "PATH", "TEMP", "TMP", "HTTPS_PROXY", "HTTP_PROXY")


def workspace_root() -> Path:
    """The directory that holds the Kairos checkouts side by side.

    By default it is the parent of this repository, matching ``workspaceRoot`` in the
    meta-repository manifest. ``KAIROS_WORKSPACE_ROOT`` overrides it for an installed wheel.
    """
    configured = os.environ.get("KAIROS_WORKSPACE_ROOT")
    return Path(configured) if configured else Path(__file__).resolve().parents[2]


def sibling_checkout(name: str) -> Path:
    return workspace_root() / name


def sibling_python(name: str) -> Path:
    """The interpreter of a sibling repository's own ``uv sync`` environment."""
    environment = sibling_checkout(name) / ".venv"
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


class SiblingError(RuntimeError):
    """A worker in a sibling repository could not be run or answered with an error."""


@functools.lru_cache(maxsize=32)
def git_revision(source: Path, tracked: str | None = None) -> str:
    """HEAD of a sibling checkout, marked when ``tracked`` has uncommitted changes."""
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        head = subprocess.run(
            [git, "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip()  # nosec B603
        dirty = ""
        if tracked is not None:
            dirty = subprocess.run(
                [git, "-C", str(source), "status", "--porcelain", "--", tracked],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            ).stdout.strip()  # nosec B603
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return head + ("+local-changes" if dirty else "")


def run_worker(
    python: Path,
    script: Path,
    arguments: list[str],
    request: dict[str, Any],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    """Run ``script`` with a sibling repository's Python and return its JSON answer.

    The worker prints ``MARKER`` before its JSON; library log lines may precede it.
    """
    environment = {key: os.environ[key] for key in _PASSED_ENVIRONMENT if key in os.environ}
    environment["PYTHONIOENCODING"] = "utf-8"
    try:
        done = subprocess.run(
            [str(python), "-s", "-B", str(script), *arguments],
            input=json.dumps(request, ensure_ascii=False).encode("utf-8"),
            capture_output=True,
            timeout=timeout_s,
            env=environment,
        )  # nosec B603
    except (OSError, subprocess.SubprocessError) as exc:
        raise SiblingError(f"worker did not run: {type(exc).__name__}") from exc
    text = done.stdout.decode("utf-8", errors="replace")
    if done.returncode != 0 or MARKER not in text:
        raise SiblingError("worker failed: " + done.stderr.decode("utf-8", "replace").strip()[-300:])
    try:
        result = json.loads(text.rsplit(MARKER, 1)[1])
    except ValueError as exc:
        raise SiblingError("worker returned malformed output") from exc
    if not isinstance(result, dict):
        raise SiblingError("worker returned malformed output")
    if "error" in result:
        raise SiblingError(str(result["error"])[:300])
    return result
