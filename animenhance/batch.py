"""File/folder conversion front end: profiles, discovery, output planning, and the batch loop."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import pipeline, tools
from .procs import RunLog, ToolError, kill_all

DEFAULT_PROFILE = "ai_sharpen0.5"

# Each profile reproduces a preview variant that was rendered and validated on episode 1
# (output/previews/ep1/*/..._preview_report.json): same model, scale, unsharp filter string,
# libx264 CRF/preset/profile, pixel format and colour/SAR tagging.
PROFILES = {
    "ai_sharpen0.5": {"upscaler": "ai", "model": "realesr-animevideov3", "scale": 2, "sharpen": 0.5,
                      "crf": 18, "preset": "slow",
                      "preview_variant": "3_ai_sharpen0.5"},
    "ai": {"upscaler": "ai", "model": "realesr-animevideov3", "scale": 2, "sharpen": 0.0, "crf": 18, "preset": "slow",
           "preview_variant": "2_ai_realesr-animevideov3"},
    "baseline": {"upscaler": "lanczos", "model": "realesr-animevideov3", "scale": 2, "sharpen": 0.0, "crf": 18,
                 "preset": "slow", "preview_variant": "1_baseline_lanczos"},
}

# Where a full conversion deliberately differs from how the tested preview clip was produced.
PREVIEW_DIFFERENCES = {
    "common": [
        "Audio: a full conversion stream-copies MP4-compatible audio (the episodes' AAC) instead of re-encoding it; "
        "the preview re-encoded AAC 192k only because it had to cut an interval. Clip mode (--start/--duration) "
        "re-encodes AAC exactly like the preview.",
        "Input: a full conversion decodes the FLV directly (Video2X reads a stream copy with IDR padding when the "
        "codec has a reorder delay); the preview fed a lossless FFV1 extract of the same decoded frames. H.264 "
        "decoding is bit-exact, so the pixels entering the upscaler are identical.",
    ],
    "ai_sharpen0.5": ["Video chain identical to the preview: Video2X -> lossless FFV1 -> FFmpeg "
                      "'unsharp(...0.500...),setsar,format=yuv420p' -> libx264 crf 18 slow high."],
    "ai": ["The preview 'ai' clip was encoded by FFmpeg's libx264 from a lossless FFV1 intermediate; this profile "
           "keeps the established enhance behaviour of letting Video2X encode libx264 directly (same CRF 18, "
           "preset slow, High profile, yuv420p). No 10 GiB intermediate is needed, but Video2X bundles a different "
           "x264 build, so the bitstream is not byte-identical to the preview."],
    "baseline": ["Same Lanczos filter as the preview; a full conversion scales the source decode directly "
                 "(index timestamps are applied before scaling so a repaired duplicate cannot be dropped)."],
}


def natural_key(text: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text.casefold())]


def is_flv(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".flv"


def discover(folder: Path, *, recursive: bool, exclude: list[Path]) -> list[Path]:
    """FLV files in a folder (case-insensitive extension), naturally sorted by relative path."""
    folder = folder.resolve()
    excl = [e.resolve() for e in exclude]

    def excluded(d: Path) -> bool:
        if d.name == ".animenhance-tmp":
            return True
        return any(d == e or e in d.parents for e in excl)

    found: list[Path] = []
    if not recursive:
        found = [p for p in folder.iterdir() if is_flv(p)]
    else:
        for root, dirs, files in os.walk(folder):
            r = Path(root)
            dirs[:] = [d for d in dirs if not excluded(r / d)]
            found += [r / f for f in files if Path(f).suffix.lower() == ".flv" and (r / f).is_file()]
    return sorted(found, key=lambda p: [natural_key(part) for part in p.relative_to(folder).parts])


@dataclass
class Job:
    index: int
    src: Path
    out: Path
    rel: str
    status: str = "pending"
    detail: str = ""
    elapsed_s: Optional[float] = None
    report: Optional[str] = None
    frames: Optional[int] = None
    fps: Optional[float] = None


def clip_name(stem: str, start_s: float, duration_s: float) -> str:
    return f"{stem}.sample-{start_s:g}s-{duration_s:g}s.mp4"


def plan_jobs(input_path: Path, output: Optional[Path], *, recursive: bool,
              clip: Optional[tuple[float, float]]) -> tuple[str, list[Job], Path]:
    """Return (mode, jobs, output_root). Raises UsageError for invalid combinations."""
    inp = Path(input_path).expanduser()
    if not inp.exists():
        raise pipeline.UsageError(f"Input not found: {inp}")
    inp = inp.resolve()
    if inp.is_file():
        if recursive:
            raise pipeline.UsageError("--recursive applies to folder inputs only")
        if inp.suffix.lower() != ".flv":
            raise pipeline.UsageError(f"Input file is not an .flv file: {inp.name}")
        if output is not None:
            out = Path(output).expanduser().resolve()
            if out.is_dir() or out.suffix.lower() != ".mp4":
                raise pipeline.UsageError(f"For a single input, --output must be an .mp4 file path (got {out})")
        else:
            name = clip_name(inp.stem, *clip) if clip else f"{inp.stem}.mp4"
            out = inp.parent / "enhanced" / name
        return "file", [Job(1, inp, out, inp.name)], out.parent
    if not inp.is_dir():
        raise pipeline.UsageError(f"Input is neither a file nor a folder: {inp}")
    if clip is not None:
        raise pipeline.UsageError("--start/--duration process a short clip of a single file; pass a file, not a folder")
    out_root = Path(output).expanduser().resolve() if output is not None else inp / "enhanced"
    if out_root.suffix.lower() == ".mp4":
        raise pipeline.UsageError(f"For a folder input, --output must be a directory (got {out_root})")
    files = discover(inp, recursive=recursive, exclude=[out_root])
    jobs = []
    for i, f in enumerate(files, 1):
        rel = f.relative_to(inp)
        jobs.append(Job(i, f, out_root / rel.parent / f"{f.stem}.mp4", str(rel)))
    return "folder", jobs, out_root


def preflight(jobs: list[Job], overwrite: bool) -> list[str]:
    """Collision checks over the whole plan (case-insensitive), before anything runs."""
    problems = []
    seen: dict[str, Job] = {}
    sources = {os.path.normcase(str(j.src)) for j in jobs}
    for j in jobs:
        key = os.path.normcase(str(j.out)).casefold()
        if key in seen:
            problems.append(f"{seen[key].rel} and {j.rel} would both write {j.out}")
        seen[key] = j
        if os.path.normcase(str(j.out)) in sources:
            problems.append(f"{j.rel}: output path {j.out} is a source file")
        for extra in (j.out.with_suffix(".report.json"), j.out.with_suffix(".log")):
            if os.path.normcase(str(extra)) in sources:
                problems.append(f"{j.rel}: report/log {extra} would overwrite a source file")
        if j.out.exists() and j.out.is_dir():
            problems.append(f"{j.rel}: output path {j.out} is a directory")
    if not overwrite:
        for j in jobs:
            if j.out.is_file():
                j.status = "skipped-existing"
                j.detail = "output exists; not re-validated (use --overwrite to replace it)"
    return problems


def resolve_settings(profile: str, *, sharpen: Optional[float], scale: Optional[int], crf: Optional[int],
                     preset: Optional[str], gpu: Optional[int], ts: tools.Toolset, timestamp_policy: str,
                     temp_dir: Optional[Path], keep_temp: bool, dry_run: bool, allow_interlaced: bool,
                     file_intermediate: bool = False) -> tuple[pipeline.Settings, dict]:
    if profile not in PROFILES:
        raise pipeline.UsageError(f"Unknown profile {profile!r}; choose one of {', '.join(PROFILES)}")
    p = PROFILES[profile]
    overrides = {}
    values = {"sharpen": p["sharpen"], "scale": p["scale"], "crf": p["crf"], "preset": p["preset"]}
    for key, val in (("sharpen", sharpen), ("scale", scale), ("crf", crf), ("preset", preset)):
        if val is not None and val != values[key]:
            overrides[key] = {"profile": values[key], "used": val}
            values[key] = val
        elif val is not None:
            values[key] = val
    gpu_note = None
    if p["upscaler"] == "ai":
        if gpu is None:
            gpu, name = pipeline.pick_gpu(ts)
            gpu_note = f"auto-selected GPU {gpu}: {name}"
        else:
            gpu_note = f"GPU {gpu} (explicit)"
    s = pipeline.Settings(gpu=gpu if gpu is not None else 0, model=p["model"], scale=values["scale"],
                          sharpen=values["sharpen"], crf=values["crf"], preset=values["preset"],
                          temp_dir=temp_dir, keep_temp=keep_temp, dry_run=dry_run, timing="auto",
                          allow_interlaced=allow_interlaced, upscaler=p["upscaler"],
                          timestamp_policy=timestamp_policy, profile=profile,
                          stream_intermediate=not file_intermediate)
    meta = {"profile": profile, "profile_defaults": {k: p[k] for k in ("upscaler", "model", "scale", "sharpen",
                                                                       "crf", "preset")},
            "tested_preview_variant": p["preview_variant"], "overrides": overrides, "gpu": gpu_note,
            "differences_from_tested_preview": PREVIEW_DIFFERENCES["common"] + PREVIEW_DIFFERENCES[profile]}
    return s, meta


def storage_plan(ts: tools.Toolset, jobs: list[Job], s: pipeline.Settings, out_root: Path) -> dict:
    """Cumulative output size of every pending job plus the largest single-job temporary peak."""
    from . import probe

    pending = [j for j in jobs if j.status == "pending"]
    temp_parent = Path(s.temp_dir).expanduser().resolve() if s.temp_dir else out_root / ".animenhance-tmp"
    per_job = []
    for j in pending:
        info = probe.ffprobe_json(ts.ffprobe, j.src)
        v = probe.first_video_stream(info)
        dur = probe._num(info.get("format", {}).get("duration")) or 0.0
        rate = probe.parse_ratio(v.get("avg_frame_rate")) or probe.parse_ratio(v.get("r_frame_rate")) or 25
        frames = probe._int(v.get("nb_frames")) or int(dur * float(rate))
        w, h = int(v["width"]) * s.scale, int(v["height"]) * s.scale
        audio_bps = sum(probe._int(a.get("bit_rate")) or 0 for a in info["streams"] if a.get("codec_type") == "audio")
        if not audio_bps:
            total_bps = probe._int(info.get("format", {}).get("bit_rate")) or 0
            audio_bps = max(0, total_bps - (probe._int(v.get("bit_rate")) or total_bps))
        out_bytes = w * h * float(rate) * dur * pipeline.OUTPUT_BPP / 8 + audio_bps * dur / 8
        from . import winpipe
        file_intermediate = s.upscaler == "ai" and s.sharpen > 0 and not (s.stream_intermediate and winpipe.available())
        ffv1 = w * h * 1.5 * frames * pipeline.FFV1_RATIO if file_intermediate else 0
        peak = ffv1 + 2 * out_bytes + j.src.stat().st_size
        per_job.append({"file": j.rel, "duration_s": dur, "output_bytes": out_bytes, "temp_peak_bytes": peak})
    total_out = sum(p["output_bytes"] for p in per_job)
    peak_temp = max((p["temp_peak_bytes"] for p in per_job), default=0)
    out_free = pipeline.free_bytes(out_root)
    same_drive = os.path.splitdrive(str(temp_parent))[0].lower() == os.path.splitdrive(str(out_root))[0].lower()
    need_on_output_drive = total_out + (peak_temp if same_drive else 0) + pipeline.MIN_FREE_BYTES
    fits, acc = 0, (peak_temp if same_drive else 0) + pipeline.MIN_FREE_BYTES
    for p in per_job:
        acc += p["output_bytes"]
        if acc > out_free:
            break
        fits += 1
    return {"pending_files": len(per_job), "estimated_total_output_bytes": total_out,
            "estimated_peak_temp_bytes": peak_temp, "temp_dir": str(temp_parent), "temp_on_output_drive": same_drive,
            "temp_free_bytes": pipeline.free_bytes(temp_parent),
            "required_on_output_drive_bytes": need_on_output_drive, "output_free_bytes": out_free,
            "fits": need_on_output_drive <= out_free and (same_drive or peak_temp <= pipeline.free_bytes(temp_parent)),
            "files_that_fit": fits, "assumptions": f"output video {pipeline.OUTPUT_BPP} bits/pixel (measured "
            f"0.12-0.13 in active ai_sharpen0.5 scenes), audio copied at source bitrate, FFV1 intermediate "
            f"{pipeline.FFV1_RATIO} of raw yuv420p (measured ~0.17)", "per_file": per_job}


def _hms(sec: Optional[float]) -> str:
    if sec is None:
        return "--:--:--"
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def run_jobs(ts: tools.Toolset, jobs: list[Job], s: pipeline.Settings, meta: dict, *, overwrite: bool,
             continue_on_error: bool, clip: Optional[tuple[float, float]],
             enhance: Callable = pipeline.enhance, out=sys.stdout) -> dict:
    """Process jobs one at a time. Returns the batch summary; never raises for per-file failures."""
    t0 = time.monotonic()
    total = len(jobs)
    done_times: list[float] = []
    cancelled = False
    for j in jobs:
        if j.status == "skipped-existing":
            print(f"[{j.index}/{total}] {j.rel}: SKIPPED - {j.out} already exists (not re-validated)", file=out, flush=True)
            continue
        remaining = sum(1 for x in jobs if x.status == "pending")
        eta = (sum(done_times) / len(done_times)) * remaining if done_times else None
        print(f"\n=== [{j.index}/{total}] {j.rel} -> {j.out}  (batch elapsed {_hms(time.monotonic() - t0)}, "
              f"batch ETA {_hms(eta) if eta else 'after first file'})", file=out, flush=True)
        start = time.monotonic()
        try:
            rep = enhance(ts, j.src, j.out, s, overwrite, RunLog(), clip=clip, extra_report={"convert": meta})
            j.elapsed_s = time.monotonic() - start
            if rep.get("dry_run"):
                j.status = "planned"
                j.detail = _dry_run_text(rep)
                print(j.detail, file=out, flush=True)
            else:
                j.status = "converted"
                j.report = str(Path(rep["output_path"]).with_suffix(".report.json"))
                j.frames = rep.get("speed", {}).get("frames")
                j.fps = rep.get("speed", {}).get("video2x_fps")
                j.detail = "validated" + (" (final-frame timestamp repaired)" if rep.get("timestamp_repair") else "")
                done_times.append(j.elapsed_s)
                print(f"[{j.index}/{total}] {j.rel}: CONVERTED in {_hms(j.elapsed_s)}"
                      + (f", Video2X {j.fps} fps" if j.fps else "") + f" -> {j.out}", file=out, flush=True)
        except KeyboardInterrupt:
            j.status = "cancelled"
            j.elapsed_s = time.monotonic() - start
            kill_all()
            cancelled = True
            print(f"[{j.index}/{total}] {j.rel}: CANCELLED", file=out, flush=True)
            break
        except (ToolError, OSError) as e:
            j.elapsed_s = time.monotonic() - start
            j.detail = str(e)
            if s.dry_run:
                j.status = "would-fail"
                print(f"[{j.index}/{total}] {j.rel}: WOULD FAIL - {e}", file=out, flush=True)
                continue
            j.status = "failed"
            failed_report = j.out.with_suffix(".failed.report.json")
            if not failed_report.exists():
                try:
                    failed_report.parent.mkdir(parents=True, exist_ok=True)
                    pipeline.write_json(failed_report, {"kind": "enhance", "status": "failed", "source": str(j.src),
                                                        "output": str(j.out), "error": str(e),
                                                        "stage": "before processing (inspection/validation)",
                                                        "settings": asdict(s) | {"temp_dir": str(s.temp_dir) if s.temp_dir else None},
                                                        "convert": meta})
                except OSError:
                    pass
            j.report = str(failed_report) if failed_report.exists() else None
            print(f"[{j.index}/{total}] {j.rel}: FAILED - {e}", file=out, flush=True)
            if not continue_on_error:
                break
    for j in jobs:
        if j.status == "pending":
            j.status = "not-started"
    counts: dict[str, int] = {}
    for j in jobs:
        counts[j.status] = counts.get(j.status, 0) + 1
    return {"started_elapsed_s": time.monotonic() - t0, "counts": counts, "cancelled": cancelled,
            "settings": asdict(s) | {"temp_dir": str(s.temp_dir) if s.temp_dir else None}, "convert": meta,
            "jobs": [asdict(j) | {"src": str(j.src), "out": str(j.out)} for j in jobs]}


def _dry_run_text(rep: dict) -> str:
    sm = rep["summary"]
    lines = [f"  source {sm['source']} -> output {sm['output']}, {sm['frames']} frames, timing {sm['timing']}"]
    if sm.get("timestamp_repair"):
        r = sm["timestamp_repair"]
        moves = ", ".join(f"#{f['index']} {f['original_pts_s']:.3f}->{f['repaired_pts_s']:.3f}s" for f in r["frames"])
        lines.append(f"  timestamp repair: last {r['suffix_length']} frame(s) after #{r['anchor_index']} "
                     f"({r['pattern']}): {moves}; slot {r['inferred_interval_s']} s; video end "
                     f"{r['video_duration_change_s'] * 1000:+.1f} ms"
                     + (f", video-audio end {r['video_minus_audio_end_before_ms']:+.1f} -> "
                        f"{r['video_minus_audio_end_after_ms']:+.1f} ms" if "video_minus_audio_end_after_ms" in r else "")
                     + f" [{r['scan_method']}]")
    for step in rep["plan"]:
        lines.append("  - " + step)
    for i in rep["issues"]:
        if i["severity"] != "info":
            lines.append(f"  [{i['severity']}] {i['message']}")
    return "\n".join(lines)


def write_summary(out_root: Path, summary: dict, *, dry_run: bool) -> Optional[Path]:
    if dry_run:
        return None
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / f"batch_summary_{time.strftime('%Y%m%d-%H%M%S')}.json"
    pipeline.write_json(path, summary)
    return path
