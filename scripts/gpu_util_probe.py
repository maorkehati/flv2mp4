"""Sample GPU engine utilisation of a running Video2X process (counters attached after it starts)."""

import collections
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

clip, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
extra = sys.argv[sys.argv.index("--") + 1:]
diag = ROOT / "output" / "diagnostics"
proc = subprocess.Popen([str(tools.BUNDLED_VIDEO2X), "-i", str(clip), "-o", str(out), "--no-copy-streams",
                         "--no-progress", *extra], cwd=tools.BUNDLED_VIDEO2X.parent, stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=0x08000000)
time.sleep(12)  # past model load, into steady inference
csv_path = diag / "gpu_util_probe.csv"
csv_path.unlink(missing_ok=True)
subprocess.run(["typeperf", r"\GPU Engine(*)\Utilization Percentage", r"\GPU Process Memory(*)\Shared Usage",
                r"\GPU Adapter Memory(*)\Shared Usage", r"\Process(video2x)\% Processor Time",
                "-si", "2", "-sc", "20", "-f", "CSV", "-o", str(csv_path), "-y"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
still_running = proc.poll() is None
proc.wait()
rows = list(csv.reader(open(csv_path, encoding="utf-8", errors="replace")))
head, data = rows[0], rows[1:]
tag = f"pid_{proc.pid}_"
util = collections.defaultdict(list)
others = collections.defaultdict(list)
mem = []
for i, h in enumerate(head):
    m = re.search(r"pid_(\d+)_.*engtype_([^)]+)\)", h)
    vals = []
    for r in data:
        try:
            vals.append(float(r[i]))
        except (ValueError, IndexError):
            pass
    if not vals:
        continue
    if m and "GPU Engine" in h:
        (util if tag in h else others)[m.group(2)].append(sum(vals) / len(vals))
    if "GPU Process Memory" in h and tag in h:
        mem.append(max(vals) / 2**20)
res = {"video2x_pid": proc.pid, "sampled_while_running": still_running, "samples": len(data),
       "video2x_gpu_engine_util_pct": {k: round(sum(v), 1) for k, v in util.items()},
       "other_processes_gpu_engine_util_pct": {k: round(sum(v), 1) for k, v in others.items() if sum(v) >= 0.5},
       "video2x_gpu_shared_memory_mb_max": round(sum(mem), 1)}
(diag / "gpu_util_probe.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
