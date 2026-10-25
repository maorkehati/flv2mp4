"""Independent checks of the episode-1 convert.py clips (FFprobe/PyAV only, not the pipeline's validator)."""

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FP = str(ROOT / "third_party/ffmpeg-9.0.2-essentials_build/bin/ffprobe.exe")
FF = str(ROOT / "third_party/ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe")
SRC = Path(r"C:\Users\mayoa\Desktop\avatar\1.FLV")
sys.stdout.reconfigure(encoding="utf-8")


def q(args):
    return subprocess.run([FP, "-v", "error", *args], capture_output=True, text=True, check=True).stdout


def frame_pts(path, read_intervals=None):
    a = ["-select_streams", "v:0", "-show_entries", "frame=pts_time", "-of", "csv=p=0"]
    if read_intervals:
        a += ["-read_intervals", read_intervals]
    return [float(x.strip(",")) for x in q(a + [str(path)]).split()]


def audio_extent(path, read_intervals=None):
    a = ["-select_streams", "a:0", "-show_entries", "frame=pts_time,duration_time", "-of", "csv=p=0"]
    if read_intervals:
        a += ["-read_intervals", read_intervals]
    rows = [l.split(",") for l in q(a + [str(path)]).splitlines() if l.strip()]
    return min(float(r[0]) for r in rows), max(float(r[0]) + float(r[1]) for r in rows)


def gray_window(path, start=None, size=(160, 120)):
    out = []
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        if start:
            c.seek(int(start * 1e6), backward=True)
        for f in c.decode(st):
            out.append((float(f.pts * st.time_base), f.reformat(width=size[0], height=size[1], format="gray")
                        .to_ndarray().astype(np.float32)))
    return out


def check(out: Path, start: float, dur: float):
    print(f"== {out.name}")
    rep = json.loads(out.with_suffix(".report.json").read_text(encoding="utf-8"))
    info = json.loads(q(["-print_format", "json", "-show_streams", "-show_format", str(out)]))
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    sar = Fraction(*map(int, v.get("sample_aspect_ratio", "1:1").split(":")))
    print(f"  video {v['codec_name']} {v['pix_fmt']} {v['width']}x{v['height']} SAR {sar} "
          f"DAR {Fraction(v['width'], v['height']) * sar}, faststart moov-first="
          f"{open(out, 'rb').read(64).find(b'moov') != -1}")
    src_all = frame_pts(SRC, f"{max(start - 15, 0)}%{start + dur + 2}")
    win = [t for t in src_all if start - 1e-9 <= t < start + dur - 1e-9]
    got = frame_pts(out)
    rep_r = rep.get("timestamp_repair")
    expected = list(win)
    if rep_r and rep["interval"]["includes_repaired_frame"]:
        expected[-1] = rep_r["replacement_pts_s"]
    rel_err = max(abs((g - got[0]) - (e - expected[0])) for g, e in zip(got, expected))
    print(f"  frames: source window {len(win)}, output {len(got)}; max per-frame timing error {rel_err * 1000:.3f} ms")
    if rep_r:
        print(f"  source last two frame PTS: {win[-2]:.3f}, {win[-1]:.3f}; output last two (relative): "
              f"{got[-2]:.3f}, {got[-1]:.3f} -> interval {1000 * (got[-1] - got[-2]):.3f} ms")
        print(f"  report repair: {rep_r['original_pts_s']} -> {rep_r['replacement_pts_s']} "
              f"(+{rep_r['inferred_interval_s']} s, {rep_r['inferred_rate_fps']} fps, video end "
              f"{rep_r['video_duration_change_s'] * 1000:+.1f} ms)")
    a0, a1 = audio_extent(out)
    sa0, sa1 = audio_extent(SRC, f"{max(start - 15, 0)}%{start + dur + 2}")
    exp_a0 = max(sa0, win[0]) - win[0]
    vend = got[-1] + (got[-1] - got[-2])
    exp_end = min(sa1, (expected[-1] + (expected[-1] - expected[-2]))) - win[0]
    print(f"  audio start {a0:.4f}s (expected {exp_a0:.4f}), audio end {a1:.4f}s (expected {exp_end:.4f}), "
          f"video end {vend:.4f}s; A/V start offset {1000 * (a0 - got[0]):+.2f} ms")
    err = subprocess.run([FF, "-v", "error", "-i", str(out), "-f", "null", "-"], capture_output=True, text=True).stderr
    print(f"  full decode errors: {err.strip() or 'none'}")
    # Content: output frames vs source frames of the same index (and neighbours).
    sf = [f for t, f in gray_window(SRC, start - 5) if start - 1e-9 <= t < start + dur - 1e-9]
    of = [f for _, f in gray_window(out)]
    worst_ok = all(np.abs(of[i] - sf[i]).mean() <= min(np.abs(of[i] - sf[j]).mean() for j in (i - 1, i + 1)
                                                         if 0 <= j < len(sf)) + 0.5 for i in range(len(sf)))
    print(f"  content: {len(of)} output frames vs {len(sf)} source frames; each matches its own source frame "
          f"at least as well as neighbours: {worst_ok}; last frame MAD vs source last {np.abs(of[-1] - sf[-1]).mean():.2f}, "
          f"vs source second-last {np.abs(of[-1] - sf[-2]).mean():.2f}")
    print(f"  report status {rep['status']}, validation ok {rep['validation']['ok']} "
          f"({sum(c['ok'] for c in rep['validation']['checks'])}/{len(rep['validation']['checks'])} checks), "
          f"validated against {rep['validated_against']}, Video2X {rep['speed']['video2x_fps']} fps")


check(ROOT / "output/convert_tests/ep1 clip 0153 4s.mp4", 113.0, 4.0)
check(ROOT / "output/convert_tests/ep1 tail 2230.mp4", 1350.0, 5.0)
