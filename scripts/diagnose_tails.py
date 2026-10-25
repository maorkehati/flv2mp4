"""Decode the last seconds of each episode and describe the presentation-timestamp sequence at the end.

Frames are listed in decoder output order (= presentation order). For each frame: its own PTS (not
best-effort), the packet DTS it came from, picture type, duration and packet size. Usage:
  python scripts/diagnose_tails.py FOLDER [--seconds 10] [--json out.json]
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from animenhance import batch, tools  # noqa: E402

FP = str(tools.BUNDLED_FFPROBE)


def probe_tail(path: Path, seconds: float) -> dict:
    fmt = json.loads(subprocess.run([FP, "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                                     str(path)], capture_output=True, text=True, check=True).stdout)
    dur = float(fmt["format"]["duration"])
    v = next(s for s in fmt["streams"] if s["codec_type"] == "video")
    a = next((s for s in fmt["streams"] if s["codec_type"] == "audio"), None)
    out = subprocess.run([FP, "-v", "error", "-select_streams", str(v["index"]), "-read_intervals",
                          f"{max(dur - seconds, 0):.3f}%", "-show_entries",
                          "frame=pts,pkt_dts,duration,pict_type,pkt_size", "-of", "csv=p=0:nk=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    frames = [dict(kv.split("=", 1) for kv in l.split(",") if "=" in kv) for l in out.splitlines() if l.strip()]
    aend = None
    if a is not None:
        aout = subprocess.run([FP, "-v", "error", "-select_streams", str(a["index"]), "-read_intervals",
                               f"{max(dur - seconds, 0):.3f}%", "-show_entries", "frame=pts_time,duration_time",
                               "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
        rows = [l.split(",") for l in aout.splitlines() if l.strip()]
        aend = max(float(r[0]) + float(r[1]) for r in rows) if rows else None
    return {"file": path.name, "time_base": v["time_base"], "r_frame_rate": v["r_frame_rate"], "frames": frames,
            "audio_end_s": aend, "format_duration_s": dur}


def describe(rec: dict) -> dict:
    fr = rec["frames"]
    ts = [int(f["pts"]) for f in fr if f.get("pts") not in (None, "N/A")]
    missing = len(fr) - len(ts)
    bad = [i for i in range(1, len(ts)) if ts[i] <= ts[i - 1]]
    deltas = [b - a for a, b in zip(ts, ts[1:])]
    first_bad = bad[0] if bad else None
    pre = deltas[:first_bad - 1] if first_bad else deltas
    step = max(set(pre), key=pre.count) if pre else None
    irregular_before = [i for i, d in enumerate(pre) if d != step]
    res = {"file": rec["file"], "tail_frames": len(ts), "missing_pts": missing, "defect_indices_from_end":
           [i - len(ts) for i in bad], "normal_interval_ticks": step,
           "irregular_intervals_before_defect": len(irregular_before)}
    if first_bad is not None:
        anchor = first_bad - 1
        res["last_reliable_from_end"] = anchor - len(ts)
        res["last_reliable_pts"] = ts[anchor]
        res["suffix_length"] = len(ts) - first_bad
        res["suffix_offsets_in_steps"] = [(t - ts[anchor]) / step for t in ts[first_bad:]]
    lo = max(0, (first_bad - 4) if first_bad else len(fr) - 5)
    res["sequence"] = [f"{f.get('pts')}({f.get('pict_type')},dts={f.get('pkt_dts')},dur={f.get('duration')},"
                       f"size={f.get('pkt_size')})" for f in fr[lo:]]
    res["video_end_s"] = (ts[-1] + int(fr[-1].get("duration") or step or 0)) / 1000 if ts else None
    res["audio_end_s"] = rec["audio_end_s"]
    return res


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    files = batch.discover(a.folder, recursive=False, exclude=[])
    with ThreadPoolExecutor(4) as pool:
        recs = list(pool.map(lambda p: probe_tail(p, a.seconds), files))
    results = [describe(r) for r in recs]
    for r in results:
        sig = ("OK" if not r["defect_indices_from_end"] else
               f"defects at {r['defect_indices_from_end']}, anchor {r['last_reliable_from_end']}, suffix "
               f"{r['suffix_length']}, offsets {r['suffix_offsets_in_steps']}")
        print(f"{r['file']:>7}: tail {r['tail_frames']} frames, step {r['normal_interval_ticks']} ticks, "
              f"irregular-before {r['irregular_intervals_before_defect']}, missing {r['missing_pts']}: {sig}; "
              f"video end {r['video_end_s']}, audio end {r['audio_end_s']}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
