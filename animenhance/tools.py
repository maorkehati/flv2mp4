"""Discovery and capability checks for Video2X, FFmpeg and FFprobe."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .procs import ToolError, capture

ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = ROOT / "third_party"
EXE = ".exe" if os.name == "nt" else ""
MANIFEST = THIRD_PARTY / "VERSIONS.json"

BUNDLED_VIDEO2X = THIRD_PARTY / "video2x-6.4.0" / f"video2x{EXE}"
BUNDLED_FFMPEG = THIRD_PARTY / "ffmpeg-9.0.2-essentials_build" / "bin" / f"ffmpeg{EXE}"
BUNDLED_FFPROBE = THIRD_PARTY / "ffmpeg-9.0.2-essentials_build" / "bin" / f"ffprobe{EXE}"

X264_PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow", "placebo"]

REQUIRED_FFMPEG = {
    "encoders": ["libx264", "aac", "ffv1", "mov_text"],
    "decoders": ["h264", "flv", "vp6f", "aac", "mp3", "ffv1"],
    "filters": ["scale", "unsharp", "hstack", "drawtext", "crop", "pad", "atrim", "asetpts", "idet", "setsar"],
    "muxers": ["mp4", "matroska", "ipod"],
    "bsfs": ["h264_metadata"],
}


@dataclass
class Toolset:
    video2x: Path
    ffmpeg: Path
    ffprobe: Path

    def as_dict(self) -> dict:
        return {"video2x": str(self.video2x), "ffmpeg": str(self.ffmpeg), "ffprobe": str(self.ffprobe)}


def _resolve(explicit: Optional[str], bundled: Path, name: str) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_dir():
            p = p / f"{name}{EXE}"
        if not p.is_file():
            raise ToolError(f"{name}: explicit path does not exist: {p}")
        return p.resolve()
    if bundled.is_file():
        return bundled
    found = shutil.which(name)
    if found:
        return Path(found).resolve()
    raise ToolError(
        f"{name} not found. Expected the project-local copy at {bundled} or {name} on PATH; "
        f"pass --{name} PATH to use another installation (see README 'Installation')."
    )


def resolve_tools(video2x: Optional[str] = None, ffmpeg: Optional[str] = None, ffprobe: Optional[str] = None) -> Toolset:
    return Toolset(
        video2x=_resolve(video2x, BUNDLED_VIDEO2X, "video2x"),
        ffmpeg=_resolve(ffmpeg, BUNDLED_FFMPEG, "ffmpeg"),
        ffprobe=_resolve(ffprobe, BUNDLED_FFPROBE, "ffprobe"),
    )


def manifest() -> dict:
    if MANIFEST.is_file():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


@lru_cache(maxsize=None)
def ffmpeg_version(path: Path) -> str:
    cp = capture([path, "-hide_banner", "-version"], timeout=60)
    first = (cp.stdout or cp.stderr).splitlines()[0] if (cp.stdout or cp.stderr) else ""
    return first.strip()


@lru_cache(maxsize=None)
def video2x_version(path: Path) -> str:
    cp = capture([path, "--version"], timeout=60, cwd=path.parent, no_window=True)
    m = re.search(r"Video2X version\s+(\S+)", cp.stdout + cp.stderr)
    if not m:
        raise ToolError(f"Could not determine Video2X version from {path}: {(cp.stdout + cp.stderr)[:300]}")
    return m.group(1)


@lru_cache(maxsize=None)
def video2x_help(path: Path) -> str:
    cp = capture([path, "--help"], timeout=60, cwd=path.parent, no_window=True)
    return cp.stdout + cp.stderr


def video2x_cli_options(path: Path) -> set[str]:
    return set(re.findall(r"--([a-z0-9-]+)", video2x_help(path)))


def video2x_realesrgan_models(path: Path) -> list[str]:
    """Model names accepted by the installed Video2X CLI for the realesrgan processor."""
    text = re.sub(r"\s+", " ", video2x_help(path))
    m = re.search(r"--realesrgan-model arg \(=([^)]+)\) .*?\(([^)]+)\)", text)
    if not m:
        raise ToolError("Installed Video2X does not expose --realesrgan-model; this wrapper requires Video2X 6.x.")
    return [s.strip() for s in m.group(2).split(",") if s.strip()]


def model_files(video2x: Path, model: str, scale: int) -> tuple[Path, Path]:
    base = video2x.parent / "models" / "realesrgan" / f"{model}-x{scale}"
    return base.with_suffix(".param"), base.with_suffix(".bin")


def available_models(video2x: Path) -> dict[str, list[int]]:
    """Map of model name -> scales whose .param and .bin files are present on disk."""
    result: dict[str, list[int]] = {}
    for model in video2x_realesrgan_models(video2x):
        scales = []
        for scale in (2, 3, 4):
            param, binf = model_files(video2x, model, scale)
            if param.is_file() and binf.is_file() and param.stat().st_size > 0 and binf.stat().st_size > 0:
                scales.append(scale)
        result[model] = scales
    return result


@lru_cache(maxsize=None)
def video2x_devices(path: Path) -> tuple[dict, ...]:
    cp = capture([path, "--list-devices"], timeout=120, cwd=path.parent, no_window=True)
    if cp.returncode != 0:
        raise ToolError(f"video2x --list-devices failed ({cp.returncode}): {(cp.stdout + cp.stderr)[-500:]}")
    devices = []
    current: Optional[dict] = None
    for line in cp.stdout.splitlines():
        m = re.match(r"^(\d+)\.\s+(.*)$", line.strip())
        if m:
            current = {"index": int(m.group(1)), "name": m.group(2).strip()}
            devices.append(current)
            continue
        if current is not None and ":" in line:
            key, value = line.split(":", 1)
            current[key.strip().lower().replace(" ", "_")] = value.strip()
    return tuple(devices)


@lru_cache(maxsize=None)
def ffmpeg_capabilities(ffmpeg: Path) -> dict[str, set[str]]:
    caps: dict[str, set[str]] = {}
    for kind, flag in (("encoders", "-encoders"), ("decoders", "-decoders"), ("filters", "-filters"),
                       ("muxers", "-muxers"), ("bsfs", "-bsfs")):
        cp = capture([ffmpeg, "-hide_banner", flag], timeout=60)
        names: set[str] = set()
        for line in cp.stdout.splitlines():
            parts = line.split()
            if kind == "bsfs":
                if len(parts) == 1 and not line.endswith(":"):
                    names.add(parts[0])
                continue
            if kind == "filters":
                if len(parts) >= 3 and re.match(r"^[TSC.]{2,3}$", parts[0]):
                    names.add(parts[1])
                continue
            if kind == "muxers":
                if len(parts) >= 2 and re.match(r"^[DEd. ]*E", line[:5]):
                    for n in parts[1].split(","):
                        names.add(n)
                continue
            if len(parts) >= 2 and re.match(r"^[VASDFXBI.]{6}$", parts[0]):
                names.add(parts[1])
        caps[kind] = names
    return caps


def missing_ffmpeg_features(ffmpeg: Path) -> dict[str, list[str]]:
    caps = ffmpeg_capabilities(ffmpeg)
    missing = {}
    for kind, names in REQUIRED_FFMPEG.items():
        absent = [n for n in names if n not in caps.get(kind, set())]
        if absent:
            missing[kind] = absent
    return missing


def find_label_font() -> Optional[Path]:
    candidates = []
    windir = os.environ.get("WINDIR")
    if windir:
        candidates += [Path(windir) / "Fonts" / n for n in ("arial.ttf", "segoeui.ttf", "tahoma.ttf", "verdana.ttf")]
    candidates += [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None
