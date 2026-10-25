"""Independent checks of convert.py clips that contain a repaired terminal suffix (FFprobe/PyAV only).

Usage: python scripts/verify_tail_clips.py EPISODE:START [EPISODE:START ...]
Clips are expected at output/convert_tests/tails/ep<N> tail.mp4, sources at C:/Users/mayoa/Desktop/avatar/<N>.FLV.
"""

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
SRC_DIR = Path(r"C:\Users\mayoa\Desktop\avatar")
sys.stdout.reconfigure(encoding="utf-8")


def q(args):
    return subprocess.run([FP, "-v", "error", *args], capture_output=True, text=True, check=True).stdout


def video_pts(path, interval=None):
    a = ["-select_streams", "v:0", "-show_entries", "frame=pts_time", "-of", "csv=p=0"]
    if interval:
        a += ["-read_intervals", interval]
    return [float(x.strip(",")) for x in q(a + [str(path)]).split()]


def audio_extent(path, interval=None):
    a = ["-select_streams", "a:0", "-show_entries", "frame=pts_time,duration_time", "-of", "csv=p=0"]
    if interval:
        a += ["-read_intervals", interval]
    rows = [l.split(",") for l in q(a + [str(path)]).splitlines() if l.strip()]
    return min(float(r[0]) for r in rows), max(float(r[0]) + float(r[1]) for r in rows)


def gray(path, seek=None, size=(160, 120)):
    out = []
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        if seek:
            c.seek(int(seek * 1e6), backward=True)
        for f in c.decode(st):
            out.append((float(f.pts * st.time_base), f.reformat(width=size[0], height=size[1], format="gray")
                        .to_ndarray().astype(np.float32)))
    return out


def verify(ep: int, start: float) -> bool:
    src = SRC_DIR / f"{ep}.FLV"
    out = ROOT / "output" / "convert_tests" / "tails" / f"ep{ep} tail.mp4"
    rep = json.loads(out.with_suffix(".report.json").read_text(encoding="utf-8"))
    r = rep["timestamp_repair"]
    m = r["suffix_length"]
    iv = f"{start - 15:.3f}%"
    dec = video_pts(src, iv)
    first = next(i for i, t in enumerate(dec) if t >= start - 1e-9)
    window = dec[first:]
    expected = window[:-m] + [f["repaired_pts_s"] for f in r["frames"]]
    got = video_pts(out)
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"   [{'ok' if cond else 'FAIL'}] {name} {detail}")

    print(f"== ep{ep}: pattern {r['pattern']}, anchor #{r['anchor_index']} at {r['anchor_pts_s']:.3f}s")
    print(f"   source last {m + 2} PTS : {[round(x, 3) for x in window[-m - 2:]]}")
    print(f"   output last {m + 2} PTS : {[round(x + window[0] - got[0], 3) for x in got[-m - 2:]]}  (re-based to source time)")
    chk("frame count preserved", len(got) == len(window), f"{len(got)} output vs {len(window)} source frames")
    rel_err = max(abs((g - got[0]) - (s - window[0])) for g, s in zip(got[:-m], window[:-m]))
    chk("earlier frames' timing unchanged", rel_err <= 0.0011, f"max error {rel_err * 1000:.3f} ms")
    suf_err = max(abs((g - got[0]) - (e - window[0])) for g, e in zip(got[-m:], expected[-m:]))
    chk("suffix on consecutive slots after anchor", suf_err <= 0.0011, f"max error {suf_err * 1000:.3f} ms")
    deltas = [round((b - a) * 1000, 3) for a, b in zip(got[-m - 2:], got[-m - 1:])]
    chk("strictly increasing at the end", all(d > 0 for d in deltas), f"last intervals {deltas} ms")

    sv = json.loads(q(["-print_format", "json", "-show_streams", str(out)]))["streams"]
    v = next(s for s in sv if s["codec_type"] == "video")
    sar = Fraction(*map(int, v.get("sample_aspect_ratio", "1:1").split(":")))
    chk("1280x960 DAR 4:3", (v["width"], v["height"]) == (1280, 960) and Fraction(1280, 960) * sar == Fraction(4, 3))
    err = subprocess.run([FF, "-v", "error", "-i", str(out), "-f", "null", "-"], capture_output=True, text=True).stderr
    chk("complete decode without errors", not err.strip(), err.strip()[:200])

    a0, a1 = audio_extent(out)
    sa0, sa1 = audio_extent(src, iv)
    exp_a0 = max(sa0, window[0]) - window[0]
    v_end_out = got[-1] - got[0] + (got[-1] - got[-2])
    exp_a1 = min(sa1, r["repaired_video_end_s"]) - window[0]
    chk("audio starts with the first frame", abs(a0 - got[0] - exp_a0) <= 0.002, f"offset {1000 * (a0 - got[0]):+.2f} ms")
    chk("audio end = unchanged source audio end", abs(a1 - exp_a1) <= 0.0235,
        f"{a1:.4f}s vs {exp_a1:.4f}s (source audio {sa1:.3f}s); video ends {v_end_out:.3f}s; "
        f"source video end {r['original_video_end_s']:.3f} -> {r['repaired_video_end_s']:.3f}s")

    sf = [f for t, f in gray(src, start - 6)]
    # locate the window start in the decoded source frames by content of frame 0
    of = [f for _, f in gray(out)]
    best0 = min(range(len(sf)), key=lambda i: np.abs(sf[i] - of[0]).mean())
    sf = sf[best0:best0 + len(of)]
    order_ok = all(np.abs(of[i] - sf[i]).mean() <= min([np.abs(of[i] - sf[j]).mean() for j in (i - 1, i + 1)
                                                        if 0 <= j < len(sf)] + [1e9]) + 0.5 for i in range(len(of)))
    tail_mad = [round(float(np.abs(of[i] - sf[i]).mean()), 2) for i in range(len(of) - m - 1, len(of))]
    chk("every frame matches its own source frame (order kept)", order_ok and len(sf) == len(of),
        f"MAD of last {m + 1} frames vs same-index source: {tail_mad}")
    print(f"   report: status {rep['status']}, validation {sum(c['ok'] for c in rep['validation']['checks'])}/"
          f"{len(rep['validation']['checks'])}, validated against {rep['validated_against']}, "
          f"interval verification {rep['interval_verification']}, Video2X {rep['speed']['video2x_fps']} fps")
    return ok


if __name__ == "__main__":
    results = [verify(int(a.split(":")[0]), float(a.split(":")[1])) for a in sys.argv[1:]]
    print("ALL OK" if all(results) else "SOME CHECKS FAILED")
