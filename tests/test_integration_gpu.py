"""REAL backend tests: Video2X runs Real-ESRGAN inference on the local Vulkan GPU.

Each test drives the public CLI (enhance_animation.py) and then verifies the result independently
with FFprobe/PyAV against the source file, not just through the pipeline's own validation report.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from conftest import (CLI, PY, ROOT, audio_extent, decode_errors, ffprobe, frame_times, global_start, gray_frames,
                      run_cli, sar_of, sha256)

pytestmark = pytest.mark.gpu

SCALE = 2
PTS_TOL = 0.0011  # one 1 ms FLV tick plus float slack
CASES = ["cfr_audio", "vfr", "offset", "noaudio", "unicode", "multi_audio"]


def _verify(src: Path, out: Path, *, expect_audio: int, expect_subs: int, sharpen: bool = False) -> None:
    si, oi = ffprobe(src), ffprobe(out)
    sv = next(s for s in si["streams"] if s["codec_type"] == "video")
    ov = [s for s in oi["streams"] if s["codec_type"] == "video"]
    assert len(ov) == 1
    ov = ov[0]
    assert ov["codec_name"] == "h264" and ov["pix_fmt"] == "yuv420p"
    assert (ov["width"], ov["height"]) == (sv["width"] * SCALE, sv["height"] * SCALE)
    assert sar_of(ov) == sar_of(sv)
    assert Fraction(ov["width"], ov["height"]) * sar_of(ov) == Fraction(sv["width"], sv["height"]) * sar_of(sv)
    with open(out, "rb") as fh:
        head = fh.read(4096)
    assert head.find(b"moov") != -1 and (head.find(b"mdat") == -1 or head.find(b"moov") < head.find(b"mdat")), \
        "moov atom must precede mdat (faststart)"

    # Frame count and per-frame presentation times relative to the earliest stream.
    g = global_start(src)
    st = [t - g for t in frame_times(src)]
    ot = frame_times(out)
    assert len(ot) == len(st), "frame count changed"
    worst = max(abs(a - b) for a, b in zip(ot, st))
    assert worst <= PTS_TOL, f"presentation time error {worst * 1000:.3f} ms"

    # Audio tracks: same count, start and end preserved relative to the shared timeline.
    sa = [s for s in si["streams"] if s["codec_type"] == "audio"]
    oa = [s for s in oi["streams"] if s["codec_type"] == "audio"]
    assert len(oa) == len(sa) == expect_audio
    for i, (s_a, o_a) in enumerate(zip(sa, oa)):
        s0, s1 = audio_extent(src, f"a:{i}")
        o0, o1 = audio_extent(out, f"a:{i}")
        frame = 1024 / int(o_a["sample_rate"])
        assert abs(o0 - (s0 - g)) <= 0.002, f"audio {i} start {o0} vs {s0 - g}"
        assert abs(o1 - (s1 - g)) <= 0.002 + frame, f"audio {i} end {o1} vs {s1 - g}"
        assert o_a["channels"] == s_a["channels"]
        if s_a.get("tags", {}).get("language"):
            assert o_a.get("tags", {}).get("language") == s_a["tags"]["language"]
        assert o_a["disposition"]["default"] == (s_a["disposition"]["default"] if any(
            x["disposition"]["default"] for x in sa) else int(i == 0))
    assert len([s for s in oi["streams"] if s["codec_type"] == "subtitle"]) == expect_subs

    assert decode_errors(out) == ""

    # Content: every output frame must match its own source frame better than the neighbouring ones,
    # which rules out dropped, duplicated, shifted or interpolated frames anywhere in the clip.
    size = (sv["width"], sv["height"])
    sf, of = gray_frames(src, size), gray_frames(out, size)
    assert len(sf) == len(of)
    for i in range(len(sf)):
        same = np.abs(of[i] - sf[i]).mean()
        for j in (i - 1, i + 1):
            if 0 <= j < len(sf):
                assert same < np.abs(of[i] - sf[j]).mean(), f"frame {i} resembles source frame {j} more"
        if not sharpen:
            assert same < 12, f"frame {i} deviates too much from source ({same:.1f})"


@pytest.mark.parametrize("case", CASES)
def test_enhance_end_to_end(case, sources, tmp_path):
    src = sources[case]
    before = sha256(src)
    out = tmp_path / ("出力 結果.mp4" if case == "unicode" else f"{case} out.mp4")
    cp = run_cli("enhance", str(src), "-o", str(out))
    assert cp.returncode == 0, cp.stderr[-3000:]
    assert out.exists()
    assert sha256(src) == before, "source must be untouched"
    rep = json.loads(out.with_suffix(".report.json").read_text(encoding="utf-8"))
    assert rep["status"] == "ok" and rep["validation"]["ok"]
    for key in ("source_identity", "source", "tools", "settings", "commands", "speed", "elapsed_s", "validation",
                "stream_handling"):
        assert key in rep
    assert rep["tools"]["video2x"]["version"] == "6.4.0"
    assert rep["source_identity"]["sha256"] == before
    assert all(e["action"] for e in rep["stream_handling"])
    expect_audio = {"cfr_audio": 1, "vfr": 1, "offset": 1, "noaudio": 0, "unicode": 1, "multi_audio": 2}[case]
    expect_subs = {"cfr_audio": 1, "unicode": 1, "multi_audio": 1}.get(case, 0)
    _verify(src, out, expect_audio=expect_audio, expect_subs=expect_subs)
    assert not list(tmp_path.glob(".*partial*")) and not (tmp_path / ".animenhance-tmp").exists()


def test_enhance_with_sharpening_uses_single_lossy_encode(sources, tmp_path):
    src = sources["vfr"]
    out = tmp_path / "sharp.mp4"
    cp = run_cli("enhance", str(src), "-o", str(out), "--sharpen", "0.6", "--keep-temp")
    assert cp.returncode == 0, cp.stderr[-3000:]
    rep = json.loads(out.with_suffix(".report.json").read_text(encoding="utf-8"))
    v2x_cmd = next(c["cmd"] for c in rep["commands"] if c["label"] == "Video2X")
    assert v2x_cmd[v2x_cmd.index("-c") + 1] == "ffv1", "AI output must be lossless when FFmpeg filtering follows"
    lossy = [c for c in rep["commands"] if "libx264" in c["cmd"]]
    assert len(lossy) == 1 and "unsharp" in " ".join(lossy[0]["cmd"])
    _verify(src, out, expect_audio=1, expect_subs=0, sharpen=True)


def test_preview_outputs(sources, tmp_path):
    src = sources["cfr_audio"]
    cp = run_cli("preview", str(src), "--start", "0.5", "--duration", "1.2", "--output-dir", str(tmp_path))
    assert cp.returncode == 0, cp.stderr[-3000:]
    rep = json.loads(next(tmp_path.glob("*_preview_report.json")).read_text(encoding="utf-8"))
    assert rep["status"] == "ok"
    n = rep["interval"]["frames"]
    sv = next(s for s in ffprobe(src)["streams"] if s["codec_type"] == "video")
    g = global_start(src)
    expected = [t - g for t in frame_times(src) if 0.5 - 1e-9 <= t - g < 1.7 - 1e-9]
    assert n == len(expected)
    clips = {k: Path(rep["outputs"][k]) for k in ("baseline", "ai", "ai_sharp", "side_by_side")}
    for key, path in clips.items():
        info = ffprobe(path)
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        times = frame_times(path)
        assert len(times) == n, key
        rel = [t - times[0] for t in times]
        assert max(abs(a - (b - expected[0])) for a, b in zip(rel, expected)) <= PTS_TOL, key
        assert decode_errors(path) == "", key
        if key != "side_by_side":
            assert (v["width"], v["height"]) == (sv["width"] * SCALE, sv["height"] * SCALE)
            assert any(s["codec_type"] == "audio" for s in info["streams"]), f"{key} must keep audio"
        else:
            assert v["width"] == 3 * sv["width"] * SCALE
    assert rep["speed"]["inference_fps_reported_by_video2x"] > 0
    assert "ESTIMATE" in rep["speed"]["note"]
    # AI and AI+sharpen come from one inference run.
    assert sum(1 for c in rep["commands"] if c["label"] == "Video2X inference") == 1
    # The baseline has no sharpening; the AI+sharpen clip differs from AI only by the unsharp filter.
    base_cmd = next(c["cmd"] for c in rep["commands"] if c["label"] == "encode baseline")
    assert "unsharp" not in " ".join(base_cmd) and "lanczos" in " ".join(base_cmd)


def _video2x_pids() -> set[int]:
    cp = subprocess.run(["tasklist", "/FI", "IMAGENAME eq video2x.exe", "/FO", "CSV", "/NH"], capture_output=True,
                        encoding="utf-8", errors="replace")
    pids = set()
    for line in cp.stdout.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) > 1 and parts[1].isdigit():
            pids.add(int(parts[1]))
    return pids


@pytest.mark.skipif(os.name != "nt", reason="uses CTRL_BREAK_EVENT / tasklist")
def test_cancellation_kills_backend_and_cleans_up(tmp_path):
    src = tmp_path / "long.flv"
    subprocess.run([str(ROOT / "third_party/ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe"), "-v", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24", "-frames:v", "600", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    out = tmp_path / "cancelled.mp4"
    before = _video2x_pids()
    proc = subprocess.Popen([PY, CLI, "enhance", str(src), "-o", str(out)], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    deadline = time.time() + 120
    started = False
    while time.time() < deadline:
        if _video2x_pids() - before:
            started = True
            break
        time.sleep(0.5)
    assert started, "Video2X never started"
    time.sleep(3)
    proc.send_signal(signal.CTRL_BREAK_EVENT)
    output = proc.communicate(timeout=60)[0].decode("utf-8", "replace")
    assert proc.returncode == 130, output[-2000:]
    time.sleep(1)
    assert not (_video2x_pids() - before), "Video2X child still running after cancellation"
    assert not out.exists()
    assert not list(tmp_path.glob(".*partial*"))
    assert not (tmp_path / ".animenhance-tmp").exists()
    rep = json.loads(out.with_suffix(".failed.report.json").read_text(encoding="utf-8"))
    assert rep["status"] == "cancelled"
    assert not out.with_suffix(".report.json").exists()
