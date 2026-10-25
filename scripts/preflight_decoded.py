"""Full decoded-frame timestamp preflight (the check a real conversion performs) for every FLV in a folder."""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check(path: str) -> dict:
    from animenhance import pipeline, tools
    from animenhance.procs import RunLog

    ts = tools.resolve_tools()
    import io
    import contextlib
    with contextlib.redirect_stderr(io.StringIO()):
        src = pipeline.analyze_source(ts, Path(path), decode_scan=True, log=RunLog(), run_idet=False,
                                      decode_audio=True, timestamp_policy="repair-terminal", scale=2)
    errors = [i["message"] for i in src.issues if i["severity"] == "error"]
    r = src.timing_repair
    return {"file": Path(path).name, "frames": len(src.scan.ts), "scan": src.original_scan.source,
            "ok": not errors, "errors": errors,
            "repair": None if not r else {k: r[k] for k in ("pattern", "anchor_index", "suffix_length", "frames",
                                                           "original_video_end_s", "repaired_video_end_s",
                                                           "video_duration_change_s")}
            | {k: r.get(k) for k in ("audio_end_s", "video_minus_audio_end_before_ms", "video_minus_audio_end_after_ms")},
            "timing_after_repair": src.timing.classification}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    from animenhance import batch
    files = [str(p) for p in batch.discover(Path(sys.argv[1]), recursive=False, exclude=[])]
    with ProcessPoolExecutor(3) as pool:
        results = list(pool.map(check, files))
    for r in results:
        rep = r["repair"]
        print(f"{r['file']:>7} {'PASS' if r['ok'] else 'FAIL'} {r['frames']} frames, timing {r['timing_after_repair']}"
              + (f", repair {rep['pattern']} end {rep['video_duration_change_s'] * 1000:+.0f} ms" if rep else ", no repair")
              + ("" if r["ok"] else f" :: {r['errors'][0][:200]}"), flush=True)
    out = ROOT / "output" / "diagnostics" / "preflight_decoded.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"PASS {sum(r['ok'] for r in results)} / {len(results)}; saved {out}")
