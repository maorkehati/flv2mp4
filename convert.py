#!/usr/bin/env python3
"""Convert one FLV file, or every FLV file in a folder, to enhanced MP4.

Examples:
  python convert.py "C:\\Users\\mayoa\\Desktop\\avatar\\1.FLV"
  python convert.py "C:\\Users\\mayoa\\Desktop\\avatar"
  python convert.py "C:\\Users\\mayoa\\Desktop\\avatar" --dry-run

A file is written to <input-parent>\\enhanced\\<stem>.mp4; a folder to <input-folder>\\enhanced\\.
Existing outputs are skipped unless --overwrite is given. Each output gets <name>.report.json and
<name>.log (or .failed.report.json / .failed.log); folders also get batch_summary_<time>.json.

Profiles (all: Real-ESRGAN realesr-animevideov3 via Video2X unless noted, 2x, H.264 CRF 18 preset slow,
yuv420p, MP4 +faststart, original timing and display aspect, audio copied or converted to AAC):
  ai_sharpen0.5  (default) AI upscale + unsharp 0.5 on luma - the tested preview variant 3
  ai             AI upscale, no additional sharpening      - preview variant 2
  baseline       FFmpeg Lanczos resize, no AI              - preview variant 1

Diagnostics stay in the original tool:
  python enhance_animation.py doctor
  python enhance_animation.py inspect "C:\\Users\\mayoa\\Desktop\\avatar\\1.FLV" --full-scan
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from animenhance import batch, pipeline, tools
from animenhance.procs import ToolError, kill_all


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="convert.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="an .flv file or a folder containing .flv files")
    p.add_argument("--output", type=Path, metavar="PATH",
                   help="output .mp4 path (file input) or output directory (folder input); "
                        "default <input-parent>\\enhanced\\<stem>.mp4 or <folder>\\enhanced")
    p.add_argument("--profile", choices=list(batch.PROFILES), default=batch.DEFAULT_PROFILE,
                   help=f"processing profile (default {batch.DEFAULT_PROFILE})")
    p.add_argument("--sharpen", type=float, metavar="FLOAT",
                   help=f"override the profile's additional unsharp strength (0 disables, max {pipeline.SHARPEN_MAX})")
    p.add_argument("--scale", type=int, metavar="INTEGER", help="scale factor 2, 3 or 4 (default 2)")
    p.add_argument("--crf", type=int, metavar="INTEGER", help="H.264 CRF 0-51 (default 18)")
    p.add_argument("--encoder-preset", metavar="NAME", choices=tools.X264_PRESETS, help="x264 preset (default slow)")
    p.add_argument("--gpu", type=int, metavar="ID",
                   help="Vulkan device index from 'video2x --list-devices' (default: first discrete GPU, else 0)")
    p.add_argument("--recursive", action="store_true",
                   help="include subfolders, mirroring their structure under the output directory")
    p.add_argument("--overwrite", action="store_true",
                   help="replace existing outputs (the old file stays until its replacement is validated)")
    p.add_argument("--continue-on-error", action="store_true",
                   help="keep going after a failed file (the exit code is still nonzero)")
    p.add_argument("--timestamp-policy", choices=["repair-terminal", "strict"], default="repair-terminal",
                   help="repair-terminal (default): when the last 1-3 frames carry duplicate/backwards timestamps "
                        "after a regular section, keep every frame and re-time those frames to the consecutive "
                        "frame slots; strict: reject. Interior or unrecognised timestamp defects always fail.")
    p.add_argument("--start", metavar="TIME", help="single file only: clip start, seconds or HH:MM:SS[.ms]")
    p.add_argument("--duration", type=float, metavar="SECONDS", help="single file only: clip length in seconds")
    p.add_argument("--dry-run", action="store_true",
                   help="inspect sources and show the resolved configuration and planned outputs; process nothing")
    p.add_argument("--keep-temp", action="store_true", help="keep temporary processing files")
    p.add_argument("--file-intermediate", action="store_true",
                   help="sharpening profiles: write Video2X's lossless output to a temporary FFV1 file (~12 GiB per "
                        "episode) and encode afterwards, instead of streaming it into FFmpeg through a named pipe")
    p.add_argument("--ignore-total-space", action="store_true",
                   help="start even if the estimated total output for all files does not fit on the output drive")
    p.add_argument("--temp-dir", type=Path, metavar="PATH",
                   help="directory for intermediates (default <output dir>\\.animenhance-tmp)")
    p.add_argument("--allow-interlaced", action="store_true",
                   help="process sources coded as interlaced as progressive frames (no deinterlacing)")
    g = p.add_argument_group("executables")
    g.add_argument("--video2x", metavar="PATH", help="Video2X CLI executable")
    g.add_argument("--ffmpeg", metavar="PATH", help="FFmpeg executable")
    g.add_argument("--ffprobe", metavar="PATH", help="FFprobe executable")
    return p


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _interrupt)
    a = build_parser().parse_args(argv)
    try:
        return run(a)
    except pipeline.UsageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ToolError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled; child processes terminated.", file=sys.stderr)
        return 130
    finally:
        kill_all()


def run(a: argparse.Namespace) -> int:
    clip = None
    if (a.start is None) != (a.duration is None):
        raise pipeline.UsageError("--start and --duration must be given together")
    if a.start is not None:
        clip = (pipeline.parse_time(a.start), float(a.duration))
        if clip[0] < 0 or clip[1] <= 0:
            raise pipeline.UsageError("--start must be >= 0 and --duration > 0")
    mode, jobs, out_root = batch.plan_jobs(a.input, a.output, recursive=a.recursive, clip=clip)
    if not jobs:
        raise pipeline.UsageError(f"No .flv files found in {a.input}" + ("" if a.recursive else " (not recursive)"))
    problems = batch.preflight(jobs, a.overwrite)
    if problems:
        raise pipeline.UsageError("Output collisions; nothing was processed:\n  - " + "\n  - ".join(problems))
    ts = tools.resolve_tools(a.video2x, a.ffmpeg, a.ffprobe)
    s, meta = batch.resolve_settings(a.profile, sharpen=a.sharpen, scale=a.scale, crf=a.crf, preset=a.encoder_preset,
                                     gpu=a.gpu, ts=ts, timestamp_policy=a.timestamp_policy, temp_dir=a.temp_dir,
                                     keep_temp=a.keep_temp, dry_run=a.dry_run, allow_interlaced=a.allow_interlaced,
                                     file_intermediate=a.file_intermediate)
    pipeline.check_settings(ts, s)

    print(f"Profile {meta['profile']}: {s.upscaler}"
          + (f" {s.model} x{s.scale} on {meta['gpu']}" if s.upscaler == "ai" else f" Lanczos x{s.scale}")
          + f", sharpen {s.sharpen:g}, libx264 crf {s.crf} preset {s.preset}, yuv420p, MP4 +faststart, "
            f"timestamp policy {s.timestamp_policy}")
    for k, v in meta["overrides"].items():
        print(f"  override: {k} {v['profile']} -> {v['used']}")
    print("Differences from the tested preview:")
    for d in meta["differences_from_tested_preview"]:
        print("  - " + d)
    print(f"{'Clip' if clip else mode.capitalize()} input: {len(jobs)} file(s) -> {out_root}"
          + (f" [{clip[0]:g}s + {clip[1]:g}s]" if clip else ""))
    for j in jobs:
        print(f"  {j.index:>3}. {j.rel} -> {j.out}" + ("  [exists: will be skipped]" if j.status == "skipped-existing" else ""))

    space = batch.storage_plan(ts, jobs, s, out_root)
    gb = lambda b: f"{b / 1024 ** 3:.1f} GiB"
    print(f"Storage: {space['pending_files']} file(s) to process need ~{gb(space['estimated_total_output_bytes'])} of "
          f"final output plus a ~{gb(space['estimated_peak_temp_bytes'])} temporary peak "
          f"({'same drive' if space['temp_on_output_drive'] else 'temp on ' + space['temp_dir']}); "
          f"free on the output drive: {gb(space['output_free_bytes'])}.")
    if not space["fits"]:
        msg = (f"Estimated space is insufficient: ~{gb(space['required_on_output_drive_bytes'])} needed on the output "
               f"drive, {gb(space['output_free_bytes'])} free; about {space['files_that_fit']} of "
               f"{space['pending_files']} file(s) would fit. Use --output (and --temp-dir) on a larger drive, or "
               f"--ignore-total-space to start anyway (each file is still checked before it starts).")
        if a.dry_run or a.ignore_total_space:
            print("WARNING: " + msg)
        else:
            raise pipeline.UsageError(msg)

    if not a.dry_run and sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    summary = batch.run_jobs(ts, jobs, s, meta, overwrite=a.overwrite, continue_on_error=a.continue_on_error, clip=clip)
    summary["storage_plan"] = space
    path = batch.write_summary(out_root, summary, dry_run=a.dry_run) if mode == "folder" else None

    print("\nSummary:")
    for j in jobs:
        line = f"  {j.index:>3}. {j.rel}: {j.status}"
        if j.status in ("converted", "failed", "cancelled") and j.elapsed_s is not None:
            line += f" ({batch._hms(j.elapsed_s)})"
        if j.status == "failed":
            line += " - " + j.detail.splitlines()[0][:200]
        print(line)
    print("  counts: " + ", ".join(f"{k}={v}" for k, v in summary["counts"].items()))
    if path:
        print(f"  batch summary: {path}")
    if summary["cancelled"]:
        return 130
    return 1 if summary["counts"].get("failed") or summary["counts"].get("would-fail") else 0


if __name__ == "__main__":
    sys.exit(main())
