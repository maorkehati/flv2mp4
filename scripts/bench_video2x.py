"""Benchmark one Video2X run with a per-frame timeline and Windows performance counters.

Usage: python scripts/bench_video2x.py CLIP.mkv OUT.mkv LABEL -- <extra video2x args>
Writes output/diagnostics/bench_<LABEL>.json with startup time, steady-state fps and resource usage.
"""

import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from animenhance import tools  # noqa: E402

COUNTERS = [
    r"\GPU Engine(*)\Utilization Percentage",
    r"\GPU Adapter Memory(*)\Shared Usage",
    r"\GPU Adapter Memory(*)\Dedicated Usage",
    r"\GPU Process Memory(*)\Shared Usage",
    r"\Process(video2x)\% Processor Time",
    r"\Process(video2x)\Working Set - Private",
    r"\Processor(_Total)\% Processor Time",
    r"\Processor Information(_Total)\% Processor Performance",
    r"\PhysicalDisk(_Total)\Disk Bytes/sec",
    r"\Memory\Available MBytes",
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    clip, out, label = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
    extra = sys.argv[sys.argv.index("--") + 1:]
    diag = ROOT / "output" / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    perf_csv = diag / f"bench_{label}_perf.csv"
    perf_csv.unlink(missing_ok=True)
    perf = subprocess.Popen(["typeperf", *COUNTERS, "-si", "2", "-f", "CSV", "-o", str(perf_csv), "-y"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    cmd = [str(tools.BUNDLED_VIDEO2X), "-i", str(clip), "-o", str(out), "--no-copy-streams", *extra]
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=tools.BUNDLED_VIDEO2X.parent, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, creationflags=0x08000000)
    timeline, buf, log = [], b"", []
    while True:
        chunk = proc.stdout.read1(4096)
        if not chunk:
            break
        buf += chunk
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        for p in parts:
            s = p.decode("utf-8", "replace")
            m = re.search(r"frame=(\d+)/(\d+)", s)
            if m:
                done = int(m.group(1))
                if not timeline or timeline[-1][1] != done:
                    timeline.append((time.monotonic() - t0, done))
            elif s.strip():
                log.append(s.strip())
    proc.wait()
    wall = time.monotonic() - t0
    time.sleep(3)
    perf.terminate()
    perf.wait()
    pid = proc.pid
    total = timeline[-1][1] if timeline else 0
    first = next((t for t, d in timeline if d >= 1), None)
    # steady state: frames between 20% and 100% of the run, by linear fit of time vs frame
    pts = [(t, d) for t, d in timeline if d >= max(2, total * 0.2)]
    steady = None
    if len(pts) >= 2:
        (t1, d1), (t2, d2) = pts[0], pts[-1]
        steady = (d2 - d1) / (t2 - t1) if t2 > t1 else None
    res = {"label": label, "cmd": cmd, "exit_code": proc.returncode, "frames": total, "wall_s": wall,
           "time_to_first_frame_s": first, "overall_fps": total / wall if wall else None,
           "steady_state_fps": steady, "steady_state_s_per_frame": 1 / steady if steady else None,
           "startup_overhead_s": (wall - total / steady) if steady else None,
           "video2x_summary": [l for l in log if "Average processing" in l or "Total time" in l or "Vulkan device" in l]}
    # performance counters
    with open(perf_csv, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.reader(fh))
    head, data = rows[0], rows[1:]
    tag = f"pid_{pid}_"

    def series(pred):
        cols = [i for i, h in enumerate(head) if pred(h)]
        vals = []
        for r in data:
            try:
                vals.append(sum(float(r[i]) for i in cols if i < len(r) and r[i].strip() not in ("", " ")))
            except ValueError:
                pass
        return cols, vals

    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        busy = vals[len(vals) // 5: -1] or vals  # drop startup/teardown samples
        return {"mean": round(sum(busy) / len(busy), 1), "max": round(max(vals), 1), "samples": len(vals)}

    engines = sorted({re.search(r"engtype_([^)\\]+)", h).group(1) for h in head if tag in h and "engtype_" in h})
    res["gpu_engine_utilization_pct_by_type"] = {
        e: stats(series(lambda h, e=e: tag in h and h.split("engtype_")[-1].split(")")[0] == e)[1]) for e in engines}
    res["gpu_process_shared_memory_mb"] = stats([v / 2**20 for v in series(lambda h: "GPU Process Memory" in h and tag in h)[1]])
    res["gpu_adapter_shared_usage_mb"] = stats([v / 2**20 for v in series(lambda h: "Adapter Memory" in h and "Shared" in h)[1]])
    res["video2x_cpu_pct_of_one_core"] = stats(series(lambda h: "Process(video2x)" in h and "Processor Time" in h)[1])
    res["video2x_private_working_set_mb"] = stats([v / 2**20 for v in series(lambda h: "Working Set - Private" in h)[1]])
    res["system_cpu_pct"] = stats(series(lambda h: "Processor(_Total)" in h)[1])
    res["cpu_performance_pct_of_nominal"] = stats(series(lambda h: "Processor Performance" in h)[1])
    res["disk_mb_per_s"] = stats([v / 2**20 for v in series(lambda h: "Disk Bytes/sec" in h)[1]])
    res["available_ram_mb"] = stats(series(lambda h: "Available MBytes" in h)[1])
    (diag / f"bench_{label}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "cmd"}, indent=1))


if __name__ == "__main__":
    main()
