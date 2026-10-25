"""Subprocess execution with logging, progress reporting and cancellation cleanup."""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

CREATE_NO_WINDOW = 0x08000000

_active: set[subprocess.Popen] = set()
_active_lock = threading.Lock()


class ToolError(RuntimeError):
    """A tool failed, or a pipeline invariant was violated."""

    def __init__(self, message: str, *, cmd: Optional[list[str]] = None, tail: Optional[list[str]] = None):
        super().__init__(message)
        self.cmd = cmd
        self.tail = tail or []

    def __str__(self) -> str:
        text = super().__str__()
        if self.tail:
            text += "\n  last output:\n    " + "\n    ".join(self.tail[-15:])
        return text


def kill_all() -> None:
    """Terminate every child process that is still running."""
    with _active_lock:
        procs = list(_active)
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
    for proc in procs:
        try:
            proc.wait(timeout=10)
        except Exception:
            pass


def _register(proc: subprocess.Popen) -> None:
    with _active_lock:
        _active.add(proc)


def _unregister(proc: subprocess.Popen) -> None:
    with _active_lock:
        _active.discard(proc)


def spawn(cmd: list[str], **kwargs) -> subprocess.Popen:
    """Start a child that kill_all() will terminate on cancellation; pair with release()."""
    proc = subprocess.Popen([str(c) for c in cmd], **kwargs)
    _register(proc)
    return proc


def release(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait()
    _unregister(proc)


@dataclass
class CommandRecord:
    label: str
    cmd: list[str]
    cwd: Optional[str]
    returncode: Optional[int] = None
    elapsed_s: Optional[float] = None
    dry_run: bool = False


@dataclass
class RunLog:
    """Collects executed commands for the report and mirrors tool output to a log file."""

    path: Optional[Path] = None
    records: list[CommandRecord] = field(default_factory=list)
    verbose: bool = False

    def write(self, text: str) -> None:
        if self.path is None:
            return
        with open(self.path, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(text)
            if not text.endswith("\n"):
                fh.write("\n")

    def note(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {text}"
        print(line, file=sys.stderr, flush=True)
        self.write(line)


def format_cmd(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd) if os.name == "nt" else " ".join(_posix_quote(c) for c in cmd)


def _posix_quote(arg: str) -> str:
    import shlex

    return shlex.quote(arg)


class Progress:
    """Single-line progress display on stderr."""

    def __init__(self, label: str, total: Optional[float] = None, unit: str = "frames"):
        self.label = label
        self.total = total
        self.unit = unit
        self.start = time.monotonic()
        self._last_draw = 0.0
        self._drawn = False

    def update(self, done: float, rate: Optional[float] = None, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_draw < 1.0:
            return
        self._last_draw = now
        elapsed = now - self.start
        if rate is None and elapsed > 0:
            rate = done / elapsed
        parts = [f"{self.label}: {done:.0f}"]
        if self.total:
            pct = 100.0 * done / self.total
            parts[0] += f"/{self.total:.0f} {self.unit} ({pct:5.1f}%)"
        else:
            parts[0] += f" {self.unit}"
        if rate:
            parts.append(f"{rate:.2f} {self.unit}/s")
            if self.total and rate > 0:
                parts.append(f"ETA {_hms(max(self.total - done, 0) / rate)}")
        parts.append(f"elapsed {_hms(elapsed)}")
        sys.stderr.write("\r" + " | ".join(parts) + "   ")
        sys.stderr.flush()
        self._drawn = True

    def finish(self) -> None:
        if self._drawn:
            sys.stderr.write("\n")
            sys.stderr.flush()


def _hms(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


ProgressParser = Callable[[str, Progress], None]

_V2X_RE = re.compile(r"frame=(\d+)/(\d+).*?fps=([\d.]+)")
_FF_FRAME_RE = re.compile(r"frame=\s*(\d+)")
_FF_TIME_RE = re.compile(r"time=\s*(-?[\d:.]+)")


def video2x_progress(line: str, prog: Progress) -> None:
    m = _V2X_RE.search(line)
    if m:
        done, total, fps = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if total > 0 and not prog.total:
            prog.total = total
        prog.update(done, fps if fps > 0 else None)


def ffmpeg_frame_progress(line: str, prog: Progress) -> None:
    m = _FF_FRAME_RE.search(line)
    if m and line.lstrip().startswith("frame="):
        prog.update(int(m.group(1)))


def ffmpeg_time_progress(line: str, prog: Progress) -> None:
    m = _FF_TIME_RE.search(line)
    if m and "time=" in line and ("size=" in line or "frame=" in line):
        prog.update(_parse_clock(m.group(1)))


def _parse_clock(text: str) -> float:
    try:
        sign = -1 if text.startswith("-") else 1
        parts = [float(p) for p in text.lstrip("-").split(":")]
        value = 0.0
        for p in parts:
            value = value * 60 + p
        return sign * value
    except ValueError:
        return 0.0


@dataclass
class Result:
    returncode: int
    output: str
    elapsed_s: float


def run(
    cmd: list[str],
    *,
    log: RunLog,
    label: str,
    progress: Optional[ProgressParser] = None,
    progress_total: Optional[float] = None,
    progress_unit: str = "frames",
    cwd: Optional[Path] = None,
    no_window: bool = False,
    check: bool = True,
    keep_output: bool = False,
    monitor: Optional[Callable[[], Optional[str]]] = None,
) -> Result:
    """Run a tool with merged stdout/stderr, logging all non-progress output.

    No shell is involved; arguments are passed as a list. ``monitor`` is polled every few seconds;
    a non-None return value kills the process and raises ToolError with that message.
    """
    cmd = [str(c) for c in cmd]
    record = CommandRecord(label=label, cmd=cmd, cwd=str(cwd) if cwd else None)
    log.records.append(record)
    log.write(f"\n$ ({label}) {format_cmd(cmd)}" + (f"\n  cwd: {cwd}" if cwd else ""))
    flags = CREATE_NO_WINDOW if (no_window and os.name == "nt") else 0
    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(cwd) if cwd else None,
        creationflags=flags,
    )
    _register(proc)
    chunks: "queue.Queue[Optional[bytes]]" = queue.Queue()

    def reader() -> None:
        assert proc.stdout is not None
        while True:
            data = proc.stdout.read1(65536) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
            if not data:
                break
            chunks.put(data)
        chunks.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    prog = Progress(label, progress_total, progress_unit) if progress else None
    tail: deque[str] = deque(maxlen=60)
    kept: list[str] = []
    pending = ""
    last_progress_log = 0.0
    last_monitor = time.monotonic()
    try:
        finished = False
        while not finished:
            if monitor and time.monotonic() - last_monitor >= 5:
                last_monitor = time.monotonic()
                problem = monitor()
                if problem:
                    proc.kill()
                    proc.wait()
                    raise ToolError(f"{label} stopped: {problem}", cmd=cmd, tail=list(tail))
            try:
                data = chunks.get(timeout=0.25)
            except queue.Empty:
                continue
            if data is None:
                finished = True
                text = pending
                pending = ""
            else:
                text = pending + data.decode("utf-8", errors="replace")
                parts = re.split(r"[\r\n]", text)
                pending = parts.pop()
                text = "\n".join(parts)
            for line in text.split("\n"):
                if not line.strip():
                    continue
                is_progress = line.lstrip().startswith("frame=") or line.lstrip().startswith("size=")
                if prog and progress:
                    progress(line, prog)
                if is_progress:
                    now = time.monotonic()
                    if now - last_progress_log >= 30:
                        log.write("  " + line.strip())
                        last_progress_log = now
                    continue
                tail.append(line.rstrip())
                log.write("  " + line.rstrip())
                if keep_output:
                    kept.append(line.rstrip())
        proc.wait()
    except BaseException:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        raise
    finally:
        _unregister(proc)
        if prog:
            prog.finish()
    elapsed = time.monotonic() - start
    record.returncode = proc.returncode
    record.elapsed_s = round(elapsed, 3)
    log.write(f"  -> exit {proc.returncode} after {elapsed:.1f}s")
    if check and proc.returncode != 0:
        raise ToolError(f"{label} failed with exit code {proc.returncode}", cmd=cmd, tail=list(tail))
    return Result(proc.returncode, "\n".join(kept if keep_output else tail), elapsed)


def capture(cmd: list[str], *, timeout: Optional[float] = None, cwd: Optional[Path] = None,
            no_window: bool = False) -> subprocess.CompletedProcess:
    """Run a short command and capture stdout/stderr as UTF-8 text."""
    cmd = [str(c) for c in cmd]
    flags = CREATE_NO_WINDOW if (no_window and os.name == "nt") else 0
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd) if cwd else None,
        creationflags=flags,
    )
    _register(proc)
    try:
        out, err = proc.communicate(timeout=timeout)
    except BaseException:
        proc.kill()
        proc.communicate()
        raise
    finally:
        _unregister(proc)
    return subprocess.CompletedProcess(
        cmd, proc.returncode, out.decode("utf-8", errors="replace"), err.decode("utf-8", errors="replace")
    )


def stream_stdout_lines(cmd: list[str], on_line: Callable[[str], None], *, stderr_path: Optional[Path] = None) -> int:
    """Run a command, feeding each stdout line to ``on_line``. stderr goes to a file (or is discarded)."""
    cmd = [str(c) for c in cmd]
    err_fh = open(stderr_path, "ab") if stderr_path else subprocess.DEVNULL
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err_fh)
        _register(proc)
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                on_line(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
            proc.wait()
        except BaseException:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            raise
        finally:
            _unregister(proc)
        return proc.returncode
    finally:
        if stderr_path:
            err_fh.close()
