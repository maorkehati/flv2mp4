"""Locate non-increasing video timestamps in a source and show the surrounding frames and packets."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animenhance import tools  # noqa: E402

src = sys.argv[1]
fp = str(tools.BUNDLED_FFPROBE)


def rows(entries):
    out = subprocess.run([fp, "-v", "error", "-select_streams", "v:0", "-show_entries", entries,
                          "-of", "csv=p=0:nk=0", src], capture_output=True, text=True, check=True).stdout
    return [dict(kv.split("=", 1) for kv in l.split(",") if "=" in kv) for l in out.splitlines() if l.strip()]


frames = rows("frame=best_effort_timestamp,pts,pkt_dts,pkt_pos,pict_type,key_frame")
ts = [int(f["best_effort_timestamp"]) for f in frames]
bad = [i for i in range(1, len(ts)) if ts[i] <= ts[i - 1]]
print("decoded frames", len(ts), "| non-increasing at frame indices", bad)
for i in bad:
    for j in range(max(0, i - 3), min(len(frames), i + 4)):
        print("  frame", j, frames[j])
pk = rows("packet=pts,dts,duration,size,pos,flags")
print("packets", len(pk))
seen = {}
for i, p in enumerate(pk):
    seen.setdefault(p["pts"], []).append(i)
dups = {k: v for k, v in seen.items() if len(v) > 1}
print("packets sharing a PTS:", dups)
for k, idxs in dups.items():
    for j in range(max(0, idxs[0] - 3), min(len(pk), idxs[-1] + 4)):
        print("  packet", j, pk[j])
