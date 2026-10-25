"""Tests that do NOT run Video2X.

Pure/mocked tests exercise logic with synthetic data or monkeypatched tool queries. Tests marked
``ffmpeg`` use the real FFmpeg/FFprobe/PyAV binaries but never run AI inference.
"""

from __future__ import annotations

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from animenhance import avtools, pipeline, probe, tools
from animenhance.procs import Result, RunLog, ToolError
from conftest import FFMPEG, FFPROBE, ffprobe, run_cli, sha256

# ---------------------------------------------------------------- pure / mocked


def scan(ts, tb=Fraction(1, 1000)):
    return probe.FrameScan("decoded-frames", tb, list(ts), [None] * len(ts))


def test_timing_exact_cfr():
    t = probe.analyze_timing(scan([i * 40 for i in range(100)]), [Fraction(25), Fraction(25)])
    assert t.classification == "cfr" and t.cfr_fit["rate"] == "25/1"


def test_timing_ms_rounded_ntsc_prefers_24000_1001_over_metadata_rational():
    ts = [round(i * 1001 / 24) for i in range(500)]  # ms-rounded 23.976 fps, like FLV
    t = probe.analyze_timing(scan(ts), [Fraction(24000, 1001), Fraction(983, 41)])
    assert t.classification == "cfr-within-rounding"
    assert t.cfr_fit["rate"] == "24000/1001"
    assert t.cfr_fit["max_deviation_ms"] <= 1.0


def test_timing_vfr_not_normalised_to_average():
    ts, t0 = [], 0
    for i in range(200):
        ts.append(t0)
        t0 += 42 if i % 10 else 83  # periodic long frames: average is "nice" but timing is variable
    t = probe.analyze_timing(scan(ts), [Fraction(1000), Fraction(24)])
    assert t.classification == "vfr" and t.cfr_fit is None


def test_timing_non_monotonic():
    t = probe.analyze_timing(scan([0, 40, 40, 120]), [None, None])
    assert t.classification == "non-monotonic"


def test_color_plan_untagged_sd_is_bt601_and_hd_is_bt709():
    sd = pipeline.color_plan({"height": 480})
    assert (sd["color_space"], sd["color_primaries"], sd["color_range"]) == ("smpte170m", "smpte170m", "tv")
    pal = pipeline.color_plan({"height": 576})
    assert pal["color_space"] == "bt470bg"
    hd = pipeline.color_plan({"height": 720})
    assert hd["color_space"] == "bt709"
    tagged = pipeline.color_plan({"height": 480, "color_space": "bt709", "color_primaries": "bt709",
                                  "color_transfer": "bt709"})
    assert tagged["color_space"] == "bt709" and not tagged["assumed"]


def test_stream_plan_reports_every_stream():
    info = {"streams": [
        {"index": 0, "codec_type": "video", "codec_name": "h264"},
        {"index": 1, "codec_type": "audio", "codec_name": "mp3", "channels": 2},
        {"index": 2, "codec_type": "audio", "codec_name": "nellymoser", "channels": 1, "channel_layout": "mono"},
        {"index": 3, "codec_type": "subtitle", "codec_name": "text"},
        {"index": 4, "codec_type": "subtitle", "codec_name": "dvd_subtitle"},
        {"index": 5, "codec_type": "data", "codec_name": "none"},
    ]}
    plan = {e["index"]: e for e in pipeline.stream_plan(info, 0)}
    assert len(plan) == 6
    assert plan[1]["action"] == "copy" and plan[1]["default"] is True
    assert plan[2]["action"] == "convert" and plan[2]["bitrate"] == "128k" and plan[2]["default"] is False
    assert plan[3]["action"] == "convert" and plan[3]["target"] == "mov_text"
    assert plan[4]["action"] == "exclude" and plan[5]["action"] == "exclude"


def test_video2x_teardown_crash_accepted_only_with_success_message():
    ok = Result(0xC0000409, "...\nVideo processed successfully\n...", 1.0)
    assert "teardown" in pipeline.video2x_exit_status(ok)
    with pytest.raises(ToolError):
        pipeline.video2x_exit_status(Result(0xC0000409, "Error processing frames", 1.0))
    with pytest.raises(ToolError):
        pipeline.video2x_exit_status(Result(1, "Video processed successfully", 1.0))
    assert pipeline.video2x_exit_status(Result(0, "", 1.0)) is None


def test_alignment_check_detects_shift():
    rng = np.random.default_rng(1)
    src = [rng.uniform(0, 255, (16, 16)).astype(np.float32) for _ in range(16)]
    assert avtools.alignment_check(src, [f + 2 for f in src])["result"] == "aligned"
    dropped = src[1:] + [src[-1]]
    assert avtools.alignment_check(src, dropped)["result"] == "misaligned"
    static = [np.zeros((16, 16), np.float32)] * 16
    assert avtools.alignment_check(static, static)["result"] == "inconclusive"


def test_output_guards(tmp_path):
    src = tmp_path / "movie.flv"
    src.write_bytes(b"FLV")
    with pytest.raises(pipeline.UsageError, match="extension"):
        pipeline.check_output(src, tmp_path / "out.mkv", False)
    with pytest.raises(pipeline.UsageError, match="source"):
        pipeline.check_output(src, src, True)
    existing = tmp_path / "out.mp4"
    existing.write_bytes(b"x")
    with pytest.raises(pipeline.UsageError, match="exists"):
        pipeline.check_output(src, existing, False)
    assert pipeline.check_output(src, existing, True) == existing.resolve()


def test_settings_validation_with_mocked_backend(monkeypatch):
    ts = tools.Toolset(Path("video2x"), Path("ffmpeg"), Path("ffprobe"))
    monkeypatch.setattr(tools, "available_models", lambda p: {"realesr-animevideov3": [2, 3, 4], "realesrgan-plus": [4]})
    monkeypatch.setattr(tools, "video2x_devices", lambda p: ({"index": 0, "name": "Fake GPU"},))
    monkeypatch.setattr(tools, "missing_ffmpeg_features", lambda p: {})
    pipeline.check_settings(ts, pipeline.Settings())
    bad = [dict(scale=5), dict(crf=60), dict(preset="turbo"), dict(sharpen=3.0), dict(model="nope"),
           dict(model="realesrgan-plus", scale=2), dict(gpu=1)]
    for kw in bad:
        with pytest.raises(pipeline.UsageError):
            pipeline.check_settings(ts, pipeline.Settings(**kw))


def test_parse_time():
    assert pipeline.parse_time("00:05:00") == 300
    assert pipeline.parse_time("1:02.5") == 62.5
    with pytest.raises(pipeline.UsageError):
        pipeline.parse_time("abc")


# ---------------------------------------------------------------- real FFmpeg / PyAV, no Video2X

@pytest.mark.ffmpeg
def test_retime_restores_irregular_timestamps_with_bframes(tmp_path):
    raw = tmp_path / "cfr.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=25", "-frames:v", "40",
                    "-c:v", "libx264", "-bf", "3", "-pix_fmt", "yuv420p", str(raw)], check=True)
    target = []
    t = 100
    for i in range(40):
        target.append(t)
        t += 40 + (i % 3) * 7 + (50 if i == 20 else 0)
    out = tmp_path / "timed.mp4"
    info = avtools.retime_remux(raw, out, target, Fraction(1, 1000), 40)
    assert info["reorder_delay"] > 0
    cp = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts,dts",
                         "-of", "csv=p=0", str(out)], capture_output=True, encoding="utf-8", check=True)
    pkts = [tuple(int(x) for x in l.split(",")[:2]) for l in cp.stdout.split()]
    tb_den = int(ffprobe(out)["streams"][0]["time_base"].split("/")[1])
    pts = sorted(p for p, _ in pkts)
    assert [round(p * 1000 / tb_den) for p in pts] == target
    dts = [d for _, d in pkts]
    assert all(b > a for a, b in zip(dts, dts[1:]))
    assert all(d <= p for p, d in pkts)


@pytest.mark.ffmpeg
def test_dry_run_launches_no_processing(sources, tmp_path, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("dry run must not execute processing commands")

    monkeypatch.setattr(pipeline, "run", forbidden)
    ts = tools.resolve_tools()
    out = tmp_path / "never.mp4"
    rep = pipeline.enhance(ts, sources["cfr_audio"], out, pipeline.Settings(dry_run=True), False, RunLog())
    assert rep["dry_run"] and rep["commands"] and not out.exists()
    assert not (tmp_path / ".animenhance-tmp").exists()
    assert any("video2x" in c.lower() for c in rep["commands"])


@pytest.mark.ffmpeg
def test_inspect_flags_interlaced_and_enhance_refuses(sources, tmp_path):
    cp = run_cli("inspect", str(sources["interlaced"]), "--full-scan")
    assert cp.returncode == 0 and "likely-interlaced" in cp.stdout
    cp = run_cli("enhance", str(sources["interlaced"]), "-o", str(tmp_path / "x.mp4"))
    assert cp.returncode == 2 and "interlaced" in cp.stderr.lower()
    assert not (tmp_path / "x.mp4").exists()
    cp = run_cli("enhance", str(sources["interlaced"]), "-o", str(tmp_path / "x.mp4"), "--allow-interlaced", "--dry-run")
    assert cp.returncode == 0, cp.stderr


@pytest.mark.ffmpeg
def test_inspect_reports_vfr_and_offsets(sources):
    cp = run_cli("inspect", str(sources["vfr"]), "--json")
    import json
    rep = json.loads(cp.stdout)
    assert rep["timing"]["classification"] == "vfr"
    assert rep["video"]["r_frame_rate"] == "1000/1"
    cp = run_cli("inspect", str(sources["offset"]), "--json")
    rep = json.loads(cp.stdout)
    codes = {i["code"] for i in rep["issues"]}
    assert {"av-start-offset", "nonzero-start", "non-square-pixels", "audio-convert"} <= codes


@pytest.mark.ffmpeg
def test_cli_rejects_bad_arguments_before_processing(sources, tmp_path):
    src = sources["cfr_audio"]
    for extra in (["--scale", "5"], ["--model", "nonexistent"], ["--gpu", "7"], ["--crf", "99"]):
        cp = run_cli("enhance", str(src), "-o", str(tmp_path / "o.mp4"), *extra)
        assert cp.returncode == 2, (extra, cp.stderr)
    mp4src = tmp_path / "movie source.mp4"
    shutil.copyfile(src, mp4src)
    before = sha256(mp4src)
    cp = run_cli("enhance", str(mp4src), "-o", str(mp4src), "--overwrite")
    assert cp.returncode == 2 and "source" in cp.stderr.lower()
    assert sha256(mp4src) == before
    assert [p.name for p in tmp_path.iterdir()] == ["movie source.mp4"]
