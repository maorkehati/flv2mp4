from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from animenhance import tools  # noqa: E402
import make_fixtures  # noqa: E402

FFPROBE = str(tools.BUNDLED_FFPROBE)
FFMPEG = str(tools.BUNDLED_FFMPEG)
PY = sys.executable
CLI = str(ROOT / "enhance_animation.py")


@pytest.fixture(scope="session")
def sources() -> dict[str, Path]:
    return make_fixtures.build()


def run_cli(*args: str, timeout: float = 1800) -> subprocess.CompletedProcess:
    return subprocess.run([PY, CLI, *args], capture_output=True, timeout=timeout, cwd=ROOT,
                          encoding="utf-8", errors="replace")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ffprobe(path: Path) -> dict:
    cp = subprocess.run([FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                        capture_output=True, check=True, encoding="utf-8")
    return json.loads(cp.stdout)


def frame_times(path: Path, stream: str = "v:0") -> list[float]:
    """Decoded frame presentation times (independent of the pipeline's own probing code)."""
    cp = subprocess.run([FFPROBE, "-v", "error", "-select_streams", stream, "-show_entries",
                         "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path)],
                        capture_output=True, check=True, encoding="utf-8")
    return [float(x.strip().rstrip(",")) for x in cp.stdout.split() if x.strip().rstrip(",") not in ("", "N/A")]


def audio_extent(path: Path, stream: str) -> tuple[float, float]:
    cp = subprocess.run([FFPROBE, "-v", "error", "-select_streams", stream, "-show_entries",
                         "frame=best_effort_timestamp_time,duration_time", "-of", "csv=p=0", str(path)],
                        capture_output=True, check=True, encoding="utf-8")
    rows = [l.split(",") for l in cp.stdout.splitlines() if l.strip()]
    starts = [float(r[0]) for r in rows]
    ends = [float(r[0]) + float(r[1]) for r in rows]
    return min(starts), max(ends)


def global_start(path: Path) -> float:
    return float(ffprobe(path)["format"].get("start_time", 0) or 0)


def decode_errors(path: Path) -> str:
    cp = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v", "-map", "0:a?",
                         "-f", "null", "-"], capture_output=True, encoding="utf-8", errors="replace")
    return (cp.stderr or "").strip() + ("" if cp.returncode == 0 else f" (exit {cp.returncode})")


def gray_frames(path: Path, size: tuple[int, int]) -> list[np.ndarray]:
    out = []
    with av.open(str(path)) as c:
        for f in c.decode(video=0):
            out.append(f.reformat(width=size[0], height=size[1], format="gray", interpolation="AREA")
                       .to_ndarray().astype(np.float32))
    return out


def sar_of(stream: dict) -> Fraction:
    s = stream.get("sample_aspect_ratio")
    if not s or s in ("N/A", "0:1"):
        return Fraction(1)
    a, b = s.split(":")
    return Fraction(int(a), int(b))
