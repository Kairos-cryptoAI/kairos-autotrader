"""A serialized request/reply stream for a preloaded sibling interpreter."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from .sibling import _PASSED_ENVIRONMENT, MARKER, SiblingError
else:
    # Worker scripts import this module before replacing sys.path[0] with their sibling checkout.
    from sibling import _PASSED_ENVIRONMENT, MARKER, SiblingError


def serve_requests(handle: Callable[[dict], dict]) -> int:
    """Read one line per request, followed by one length-prefixed JSON reply."""

    def send(value: dict) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        output = sys.stdout.buffer
        output.write(f"\n{MARKER}{len(body)}\n".encode("ascii"))
        output.write(body)
        output.flush()

    send({"ready": True})
    for line in sys.stdin.buffer:
        try:
            reply = handle(json.loads(line))
        except Exception as exc:  # noqa: BLE001 - report only the type, requests may carry credentials.
            reply = {"error": f"worker request failed ({type(exc).__name__})"}
        send(reply)
    return 0


class WorkerSession:
    """Serialize full requests, dropping a broken process without replaying dispatched work."""

    def __init__(self, python: Path, script: Path, arguments: list[str], *, timeout_s: float) -> None:
        self._command = [str(python), "-s", "-B", str(script), *arguments, "--serve"]
        self._timeout_s = timeout_s
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def _reply(self, process: asyncio.subprocess.Process) -> dict[str, Any]:
        if process.stdout is None:
            raise SiblingError("worker output is unavailable")
        marker = MARKER.encode("ascii")
        while line := await process.stdout.readline():
            if line.startswith(marker):
                length = int(line[len(marker) :].strip())
                if length <= 0:
                    raise SiblingError("worker returned an invalid response length")
                result = json.loads(await process.stdout.readexactly(length))
                if not isinstance(result, dict):
                    break
                if "error" in result:
                    raise SiblingError(str(result["error"])[:300])
                return result
        raise SiblingError("worker exited without a valid response")

    async def _start(self) -> asyncio.subprocess.Process:
        if self._process is None or self._process.returncode is not None:
            env = {key: os.environ[key] for key in _PASSED_ENVIRONMENT if key in os.environ}
            env["PYTHONIOENCODING"] = "utf-8"
            self._process = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=env,
            )
            if (await self._reply(self._process)).get("ready") is not True:
                raise SiblingError("worker failed to initialize")
        return self._process

    async def _discard(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.communicate()

    async def stop(self) -> None:
        async with self._lock:
            await self._discard()

    async def _exchange(self, request: dict[str, Any] | None) -> dict[str, Any]:
        async with self._lock:
            try:
                async with asyncio.timeout(self._timeout_s):
                    process = await self._start()
                    if request is None:
                        return {}
                    if process.stdin is None:
                        raise SiblingError("worker input is unavailable")
                    payload = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
                    process.stdin.write(payload + b"\n")
                    await process.stdin.drain()
                    return await self._reply(process)
            except asyncio.CancelledError:
                await self._discard()
                raise
            except (OSError, ValueError, TimeoutError, EOFError, SiblingError) as exc:
                await self._discard()
                if isinstance(exc, SiblingError):
                    raise
                raise SiblingError(f"worker failed ({type(exc).__name__})") from exc

    async def warm(self) -> None:
        await self._exchange(None)

    async def ask(self, request: dict[str, Any]) -> dict[str, Any]:
        return await self._exchange(request)
