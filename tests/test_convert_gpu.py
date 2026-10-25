"""REAL backend tests for convert.py: Video2X Real-ESRGAN runs on the local Vulkan GPU."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from conftest import (FFPROBE, PY, ROOT, audio_extent, decode_errors, ffprobe, global_start, gray_frames, sar_of,
                      sha256)

pytestmark = pytest.mark.gpu
CONVERT = str(ROOT / "convert.py")


def run_convert(*args, timeout=1800):
    return subprocess.run([PY, CONVERT, *map(str, args)], capture_output=True, cwd=ROOT, timeout=timeout,
                          encoding="utf-8", errors="replace")


def raw_pts(path: Path) -> list[float]:
    """Decoded frames' own PTS in seconds (not best-effort guesses)."""
    cp = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time",
                         "-of", "csv=p=0", str(path)], capture_output=True, check=True, encoding="utf-8")
    return [float(x.strip(",")) for x in cp.stdout.split()]


def check_geometry(src: Path, out: Path, scale: int = 2) -> None:
    sv = next(s for s in ffprobe(src)["streams"] if s["codec_type"] == "video")
    ov = next(s for s in ffprobe(out)["streams"] if s["codec_type"] == "video")
    assert (ov["width"], ov["height"]) == (sv["width"] * scale, sv["height"] * scale)
    assert ov["pix_fmt"] == "yuv420p" and ov["codec_name"] == "h264"
    assert Fraction(ov["width"], ov["height"]) * sar_of(ov) == Fraction(sv["width"], sv["height"]) * sar_of(sv)
    assert decode_errors(out) == ""


def test_terminal_duplicate_repaired_end_to_end(sources, tmp_path):
    src = sources["terminal_dup"]
    before = sha256(src)
    out = tmp_path / "תוצאה עם רווח.mp4"
    cp = run_convert(src, "--output", out)
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
    assert sha256(src) == before
    rep = json.loads(out.with_suffix(".report.json").read_text(encoding="utf-8"))
    assert rep["status"] == "ok" and rep["validation"]["ok"] and rep["validated_against"] == "repaired timeline"
    assert rep["encode"]["streamed"] is True and not any(c["cmd"][-1].endswith("ai_ffv1.mkv") for c in rep["commands"])
    assert "AI inference + concurrent sharpen/libx264 encode (Video2X streamed into FFmpeg)" in rep["stage_times_s"]
    r = rep["timestamp_repair"]
    f = r["frames"]
    assert (r["anchor_index"], f[0]["index"], f[0]["original_pts_s"], f[0]["repaired_pts_s"]) == (48, 49, 2.0, 2.04)
    assert r["inferred_interval_s"] == "1/25" and abs(r["video_duration_change_s"] - 0.04) < 1e-9
    assert rep["timestamp_changes"] == [{"frame_index": 49, "original_pts_s": 2.0, "repaired_pts_s": 2.04,
                                         "changed": True}]
    assert rep["resolved_settings"]["sharpen"] == 0.5 and rep["resolved_settings"]["profile"] == "ai_sharpen0.5"
    check_geometry(src, out)

    g = global_start(src)
    s_pts = [t - g for t in raw_pts(src)]
    o_pts = raw_pts(out)
    assert len(o_pts) == len(s_pts) == 50, "final frame must be preserved"
    assert max(abs(a - b) for a, b in zip(o_pts[:-1], s_pts[:-1])) <= 0.0011, "earlier frames must keep their times"
    assert abs(o_pts[-1] - (o_pts[-2] + 0.040)) <= 0.0011 and abs(s_pts[-1] - s_pts[-2]) < 1e-9

    # Final frame content is the source's final frame, not a copy of the previous one.
    sv = next(s for s in ffprobe(src)["streams"] if s["codec_type"] == "video")
    size = (sv["width"], sv["height"])
    sf, of = gray_frames(src, size), gray_frames(out, size)
    assert np.abs(of[-1] - sf[-1]).mean() < np.abs(of[-1] - sf[-2]).mean()

    s0, s1 = audio_extent(src, "a:0")
    o0, o1 = audio_extent(out, "a:0")
    assert abs(o0 - (s0 - g)) <= 0.002 and abs(o1 - (s1 - g)) <= 0.002
    assert abs((o0 - o_pts[0]) - (s0 - g - s_pts[0])) <= 0.002, "A/V offset preserved"


@pytest.mark.parametrize("key,orig,repaired_rel", [
    ("tail_0_1_3", [2.32, 2.36, 2.44], [0.04, 0.08, 0.12]),   # episode 26-like
    ("tail_m1_0_1", [2.28, 2.32, 2.36], [0.04, 0.08, 0.12]),  # episode 27-like (PTS < DTS in the source)
])
def test_multi_frame_terminal_suffix_repaired(sources, tmp_path, key, orig, repaired_rel):
    src = sources[key]
    out = tmp_path / f"{key} out.mp4"
    cp = run_convert(src, "--output", out)
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
    rep = json.loads(out.with_suffix(".report.json").read_text(encoding="utf-8"))
    r = rep["timestamp_repair"]
    assert [f["original_pts_s"] for f in r["frames"]] == orig and r["suffix_length"] == 3
    check_geometry(src, out)
    g = global_start(src)
    s_pts = [t - g for t in raw_pts(src)]
    o_pts = raw_pts(out)
    assert len(o_pts) == len(s_pts), "every frame preserved"
    assert max(abs(a - b) for a, b in zip(o_pts[:-3], s_pts[:-3])) <= 0.0011, "frames through the anchor unchanged"
    anchor = o_pts[-4]
    assert [round(t - anchor, 4) for t in o_pts[-3:]] == repaired_rel
    sv = next(s for s in ffprobe(src)["streams"] if s["codec_type"] == "video")
    sf, of = gray_frames(src, (sv["width"], sv["height"])), gray_frames(out, (sv["width"], sv["height"]))
    for i in range(len(sf) - 4, len(sf)):  # order kept: each final frame resembles its own source frame most
        same = np.abs(of[i] - sf[i]).mean()
        assert all(same < np.abs(of[i] - sf[j]).mean() for j in (i - 1, i + 1) if 0 <= j < len(sf))
    s0, s1 = audio_extent(src, "a:0")
    o0, o1 = audio_extent(out, "a:0")
    assert abs(o0 - (s0 - g)) <= 0.002 and abs(o1 - (s1 - g)) <= 0.002, "audio timing unchanged"


def _pids(image: str) -> set[int]:
    cp = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {image}", "/FO", "CSV", "/NH"], capture_output=True,
                        encoding="utf-8", errors="replace")
    return {int(p[1]) for p in (l.strip('"').split('","') for l in cp.stdout.splitlines()) if len(p) > 1 and p[1].isdigit()}


@pytest.mark.skipif(sys.platform != "win32", reason="named-pipe streaming and CTRL_BREAK are Windows-only")
def test_streamed_conversion_cancellation_cleans_up(tmp_path):
    import signal
    import time
    src = tmp_path / "long.flv"
    subprocess.run([str(ROOT / "third_party/ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe"), "-v", "error", "-y", "-f",
                    "lavfi", "-i", "testsrc2=size=320x240:rate=25", "-frames:v", "500", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    out = tmp_path / "out" / "long.mp4"
    v_before, f_before = _pids("video2x.exe"), _pids("ffmpeg.exe")
    proc = subprocess.Popen([PY, CONVERT, str(src), "--output", str(out)], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    deadline = time.time() + 180
    while time.time() < deadline and not (_pids("video2x.exe") - v_before and _pids("ffmpeg.exe") - f_before):
        time.sleep(0.5)
    assert _pids("video2x.exe") - v_before, "Video2X never started"
    time.sleep(4)
    proc.send_signal(signal.CTRL_BREAK_EVENT)
    output = proc.communicate(timeout=90)[0].decode("utf-8", "replace")
    assert proc.returncode == 130, output[-2000:]
    time.sleep(1)
    assert not (_pids("video2x.exe") - v_before), "Video2X left running"
    assert not (_pids("ffmpeg.exe") - f_before), "streaming FFmpeg encoder left running"
    assert not out.exists() and not list(out.parent.glob(".*partial*"))
    assert not (out.parent / ".animenhance-tmp").exists()
    assert json.loads(out.with_suffix(".failed.report.json").read_text(encoding="utf-8"))["status"] == "cancelled"


def test_folder_batch_order_failure_skip_and_overwrite(sources, tmp_path):
    folder = tmp_path / "פרקים עם רווח"
    folder.mkdir()
    shutil.copyfile(sources["cfr_audio"], folder / "1.FLV")
    shutil.copyfile(sources["interior_dup"], folder / "2.flv")
    shutil.copyfile(sources["noaudio"], folder / "10.Flv")
    cp = run_convert(folder, "--continue-on-error")
    assert cp.returncode == 1, "a failed file must make the batch exit nonzero"
    text = cp.stdout
    assert text.index("[1/3] 1.FLV") < text.index("[2/3] 2.flv") < text.index("[3/3] 10.Flv")
    out_dir = folder / "enhanced"
    summary = json.loads(sorted(out_dir.glob("batch_summary_*.json"))[-1].read_text(encoding="utf-8"))
    assert summary["counts"] == {"converted": 2, "failed": 1}
    assert [j["status"] for j in summary["jobs"]] == ["converted", "failed", "converted"]
    assert "Unsupported timestamp defect" in summary["jobs"][1]["detail"]
    assert (out_dir / "2.failed.report.json").exists() and not (out_dir / "2.mp4").exists()
    for name in ("1.mp4", "10.mp4"):
        check_geometry(folder / name.replace(".mp4", ".FLV" if name == "1.mp4" else ".Flv"), out_dir / name)
        assert json.loads((out_dir / name).with_suffix(".report.json").read_text(encoding="utf-8"))["status"] == "ok"

    first = (out_dir / "1.mp4").stat().st_mtime_ns
    cp = run_convert(folder / "1.FLV")
    assert cp.returncode == 0 and "SKIPPED" in cp.stdout and (out_dir / "1.mp4").stat().st_mtime_ns == first

    cp = run_convert(folder / "1.FLV", "--overwrite", "--profile", "ai", "--sharpen", "0")
    assert cp.returncode == 0, cp.stdout[-2000:] + cp.stderr[-2000:]
    assert (out_dir / "1.mp4").stat().st_mtime_ns != first
    rep = json.loads((out_dir / "1.report.json").read_text(encoding="utf-8"))
    assert rep["resolved_settings"]["profile"] == "ai" and rep["resolved_settings"]["sharpen"] == 0.0
    assert not list(out_dir.glob(".*partial*")) and not (out_dir / ".animenhance-tmp").exists()


def test_clip_mode_default_profile(sources, tmp_path):
    src = sources["cfr_audio"]
    cp = run_convert(src, "--start", "0.5", "--duration", "1", "--output", tmp_path / "clip.mp4")
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
    out = tmp_path / "clip.mp4"
    rep = json.loads(out.with_suffix(".report.json").read_text(encoding="utf-8"))
    assert rep["status"] == "ok" and rep["kind"] == "enhance-clip"
    check_geometry(src, out)
    g = global_start(src)
    expected = [t - g for t in raw_pts(src) if 0.5 - 1e-9 <= t - g < 1.5 - 1e-9]
    got = raw_pts(out)
    assert len(got) == len(expected) == rep["interval"]["frames"]
    assert max(abs((a - got[0]) - (b - expected[0])) for a, b in zip(got, expected)) <= 0.0011
    o0, _ = audio_extent(out, "a:0")
    assert abs(o0 - got[0]) <= 0.003, "clip audio starts with the first frame"


def test_baseline_profile_full_file(sources, tmp_path):
    src = sources["vfr"]
    out = tmp_path / "base.mp4"
    cp = run_convert(src, "--profile", "baseline", "--output", out)
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
    rep = json.loads(out.with_suffix(".report.json").read_text(encoding="utf-8"))
    assert rep["status"] == "ok" and rep["resolved_settings"]["upscaler"] == "lanczos"
    assert not any(c["label"] == "Video2X" for c in rep["commands"])
    check_geometry(src, out)
    g = global_start(src)
    s_pts = [t - g for t in raw_pts(src)]
    o_pts = raw_pts(out)
    assert len(o_pts) == len(s_pts) and max(abs(a - b) for a, b in zip(o_pts, s_pts)) <= 0.0011
