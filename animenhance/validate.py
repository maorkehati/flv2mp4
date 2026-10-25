"""Validation of produced MP4 files against the expectations derived from the source."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional

from . import probe
from .procs import Progress, RunLog, run


@dataclass
class AudioExpectation:
    source_index: int
    codec: str
    channels: int
    sample_rate: int
    language: Optional[str]
    default: bool
    start_s: Optional[float]
    end_s: Optional[float]
    start_tol_s: float
    end_tol_s: float


@dataclass
class Expectation:
    width: int
    height: int
    sar: Fraction
    frame_count: int
    pts_s: list[float]
    last_frame_duration_s: float
    pts_tol_s: float
    color: dict
    audio: list[AudioExpectation] = field(default_factory=list)
    subtitles: int = 0
    video_start_s: Optional[float] = None
    source_pts_s: Optional[list[float]] = None
    source_tol_s: float = 0.0011


def _check(results: list, name: str, ok: bool, detail) -> bool:
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    return ok


def validate_output(ffmpeg: Path, ffprobe: Path, path: Path, exp: Expectation, *, log: RunLog,
                    full_decode: bool = True, label: str = "validate") -> dict:
    results: list[dict] = []
    info = probe.ffprobe_json(ffprobe, path)
    streams = info["streams"]
    videos = [s for s in streams if s.get("codec_type") == "video"]
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    subs = [s for s in streams if s.get("codec_type") == "subtitle"]
    others = [s for s in streams if s.get("codec_type") not in ("video", "audio", "subtitle")]
    _check(results, "exactly one video stream", len(videos) == 1, len(videos))
    _check(results, "audio stream count", len(audios) == len(exp.audio),
           {"expected": len(exp.audio), "found": len(audios)})
    _check(results, "subtitle stream count", len(subs) == exp.subtitles, {"expected": exp.subtitles, "found": len(subs)})
    _check(results, "no unexpected streams", not others, [s.get("codec_type") for s in others])
    if not videos:
        return {"ok": False, "checks": results}
    v = videos[0]
    _check(results, "video codec h264", v.get("codec_name") == "h264", v.get("codec_name"))
    _check(results, "pixel format yuv420p", v.get("pix_fmt") == "yuv420p", v.get("pix_fmt"))
    _check(results, "dimensions", (v.get("width"), v.get("height")) == (exp.width, exp.height),
           {"expected": [exp.width, exp.height], "found": [v.get("width"), v.get("height")]})
    sar = probe.parse_ratio(v.get("sample_aspect_ratio")) or Fraction(1)
    _check(results, "sample aspect ratio", sar == exp.sar, {"expected": probe.ratio_str(exp.sar), "found": probe.ratio_str(sar)})
    dar_exp = Fraction(exp.width, exp.height) * exp.sar
    dar = Fraction(v["width"], v["height"]) * sar
    _check(results, "display aspect ratio", dar == dar_exp, {"expected": probe.ratio_str(dar_exp), "found": probe.ratio_str(dar)})
    color_found = {k: v.get(k) for k in ("color_range", "color_space", "color_transfer", "color_primaries")}
    color_ok = all(color_found.get(k) == exp.color.get(k) for k in color_found)
    _check(results, "color tags", color_ok, {"expected": exp.color, "found": color_found})

    # Frame timestamps from packets (display order), compared one-to-one with the source timing table.
    tb = probe.parse_ratio(v.get("time_base"))
    scan = probe.scan_frames(ffprobe, path, int(v["index"]), decode=False, time_base=tb, show_progress=False)
    got = [t * float(tb) for t in scan.ts]
    _check(results, "video frame count (packets)", len(got) == exp.frame_count,
           {"expected": exp.frame_count, "found": len(got)})
    timing_detail: dict = {"tolerance_ms": exp.pts_tol_s * 1000}
    if len(got) == exp.frame_count and got:
        errs = [abs(a - b) for a, b in zip(got, exp.pts_s)]
        worst = max(range(len(errs)), key=lambda i: errs[i])
        timing_detail.update({"max_abs_error_ms": errs[worst] * 1000, "worst_frame": worst,
                              "first_pts_s": got[0], "expected_first_pts_s": exp.pts_s[0]})
        _check(results, "per-frame presentation timestamps", errs[worst] <= exp.pts_tol_s, timing_detail)
        if exp.source_pts_s is not None and len(exp.source_pts_s) == len(got):
            serr = [abs(a - b) for a, b in zip(got, exp.source_pts_s)]
            sw = max(range(len(serr)), key=lambda i: serr[i])
            _check(results, "per-frame timestamps vs original source", serr[sw] <= exp.source_tol_s,
                   {"max_abs_error_ms": serr[sw] * 1000, "worst_frame": sw, "tolerance_ms": exp.source_tol_s * 1000})
        dupes = len(got) - len(set(scan.ts))
        _check(results, "no duplicate frame timestamps", dupes == 0, dupes)
        last_dur = (scan.durations[-1] or 0) * float(tb) if scan.durations else 0.0
        v_end = got[-1] + last_dur
        exp_end = exp.pts_s[-1] + exp.last_frame_duration_s
        _check(results, "video end time", abs(v_end - exp_end) <= exp.pts_tol_s + 0.002,
               {"expected_s": exp_end, "found_s": v_end})
    else:
        _check(results, "per-frame presentation timestamps", False, "frame count mismatch; cannot compare")
    video_start = got[0] if got else None

    # Audio: codec, layout, language, default flag, start/end and offset relative to video.
    for i, (a, e) in enumerate(zip(audios, exp.audio)):
        tags = a.get("tags", {}) or {}
        _check(results, f"audio {i}: codec", a.get("codec_name") == e.codec, {"expected": e.codec, "found": a.get("codec_name")})
        _check(results, f"audio {i}: channels", a.get("channels") == e.channels, {"expected": e.channels, "found": a.get("channels")})
        _check(results, f"audio {i}: sample rate", int(a.get("sample_rate", 0)) == e.sample_rate,
               {"expected": e.sample_rate, "found": a.get("sample_rate")})
        if e.language:
            _check(results, f"audio {i}: language", tags.get("language") == e.language,
                   {"expected": e.language, "found": tags.get("language")})
        _check(results, f"audio {i}: default flag", bool(a.get("disposition", {}).get("default")) == e.default,
               {"expected": e.default, "found": a.get("disposition", {}).get("default")})
        ext = probe.stream_extent(ffprobe, path, int(a["index"]), decode=True)
        if e.start_s is not None and ext["start_s"] is not None:
            _check(results, f"audio {i}: start time", abs(ext["start_s"] - e.start_s) <= e.start_tol_s,
                   {"expected_s": e.start_s, "found_s": ext["start_s"], "tolerance_ms": e.start_tol_s * 1000})
            if video_start is not None and exp.video_start_s is not None:
                off_exp = e.start_s - exp.video_start_s
                off_got = ext["start_s"] - video_start
                _check(results, f"audio {i}: offset relative to video", abs(off_got - off_exp) <= e.start_tol_s + exp.pts_tol_s,
                       {"expected_s": off_exp, "found_s": off_got})
        if e.end_s is not None and ext["end_s"] is not None:
            _check(results, f"audio {i}: end time", abs(ext["end_s"] - e.end_s) <= e.end_tol_s,
                   {"expected_s": e.end_s, "found_s": ext["end_s"], "tolerance_ms": e.end_tol_s * 1000})

    if full_decode:
        decoded = {"frames": None}

        def on_progress(line: str, prog: Progress) -> None:
            import re
            m = re.search(r"frame=\s*(\d+)", line)
            if m:
                decoded["frames"] = int(m.group(1))
                prog.update(int(m.group(1)))

        res = run([ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-stats", "-xerror", "-i", str(path),
                   "-map", "0:v:0", "-map", "0:a?", "-f", "null", "-"],
                  log=log, label=f"{label}: full decode", progress=on_progress, progress_total=exp.frame_count,
                  check=False, keep_output=True)
        errors = [l for l in res.output.splitlines() if l.strip()]
        _check(results, "full decode without errors", res.returncode == 0 and not errors,
               {"exit": res.returncode, "messages": errors[:20]})
        _check(results, "decoded video frame count", decoded["frames"] == exp.frame_count,
               {"expected": exp.frame_count, "found": decoded["frames"]})

    fmt = info.get("format", {})
    summary = {
        "container_duration_s": probe._num(fmt.get("duration")),
        "size_bytes": probe._int(fmt.get("size")),
        "bit_rate": probe._int(fmt.get("bit_rate")),
        "streams": probe.summarize_streams(info),
    }
    ok = all(r["ok"] for r in results)
    return {"ok": ok, "checks": results, "output": summary}
