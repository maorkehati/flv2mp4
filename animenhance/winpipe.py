"""Windows named-pipe relay: lets Video2X stream its output into an FFmpeg process without a temp file.

Video2X's bundled FFmpeg has no pipe: protocol, but its file protocol can open \\\\.\\pipe\\NAME. This module
creates the pipe server end, waits for the writer to connect, and copies everything into a sink (the
consumer process's stdin) on a background thread.
"""

from __future__ import annotations

import ctypes
import os
import threading
import uuid
from ctypes import wintypes
from typing import BinaryIO, Optional

_k32 = ctypes.WinDLL("kernel32", use_last_error=True) if os.name == "nt" else None

PIPE_ACCESS_INBOUND = 0x00000001
PIPE_TYPE_BYTE = 0x00000000
PIPE_WAIT = 0x00000000
INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value
ERROR_PIPE_CONNECTED = 535
ERROR_BROKEN_PIPE = 109
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3

if _k32 is not None:
    _k32.CreateNamedPipeW.restype = wintypes.HANDLE
    _k32.CreateNamedPipeW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID]
    _k32.ConnectNamedPipe.argtypes = [wintypes.HANDLE, wintypes.LPVOID]
    _k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                              wintypes.LPVOID]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.CreateFileW.restype = wintypes.HANDLE
    _k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                 wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]


def available() -> bool:
    return _k32 is not None


class PipeRelay:
    """Server end of a named pipe whose bytes are copied to ``sink`` until the writer closes it."""

    def __init__(self, suffix: str = ".mkv", buffer: int = 4 * 1024 * 1024):
        self.name = rf"\\.\pipe\animenhance-{uuid.uuid4().hex}{suffix}"
        self.buffer = buffer
        self.handle = _k32.CreateNamedPipeW(self.name, PIPE_ACCESS_INBOUND, PIPE_TYPE_BYTE | PIPE_WAIT, 1,
                                            0, buffer, 0, None)
        if self.handle == INVALID_HANDLE_VALUE:
            raise OSError(ctypes.get_last_error(), "CreateNamedPipeW failed")
        self.bytes = 0
        self.connected = False
        self.error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, sink: BinaryIO) -> None:
        def worker() -> None:
            try:
                if not _k32.ConnectNamedPipe(self.handle, None):
                    err = ctypes.get_last_error()
                    if err != ERROR_PIPE_CONNECTED:
                        raise OSError(err, "ConnectNamedPipe failed")
                self.connected = True
                buf = ctypes.create_string_buffer(self.buffer)
                got = wintypes.DWORD(0)
                while True:
                    ok = _k32.ReadFile(self.handle, buf, self.buffer, ctypes.byref(got), None)
                    if not ok:
                        err = ctypes.get_last_error()
                        if err == ERROR_BROKEN_PIPE:
                            break
                        raise OSError(err, "ReadFile on pipe failed")
                    if got.value == 0:
                        continue
                    sink.write(buf.raw[:got.value])
                    self.bytes += got.value
            except BaseException as e:  # surfaced by join()
                self.error = e
            finally:
                try:
                    sink.close()
                except OSError:
                    pass

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def unblock(self) -> None:
        """If the writer never connected (it failed early), connect as a dummy client so the thread can exit."""
        if not self.connected:
            h = _k32.CreateFileW(self.name, GENERIC_WRITE, 0, None, OPEN_EXISTING, 0, None)
            if h != INVALID_HANDLE_VALUE:
                _k32.CloseHandle(h)

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def close(self) -> None:
        if self.handle and self.handle != INVALID_HANDLE_VALUE:
            _k32.CloseHandle(self.handle)
            self.handle = None
