#!/usr/bin/env python3
"""Enhance animation FLV files to MP4 with Video2X (Real-ESRGAN) and FFmpeg.

Commands:
  doctor                 check dependencies, versions, GPU access, models and encoders
  inspect INPUT          report source properties and potential processing issues
  preview INPUT          baseline / AI / AI+sharpen clips of one interval plus a comparison
  enhance INPUT          process the full source into an MP4
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from animenhance import pipeline, tools
from animenhance.procs import RunLog, ToolError, kill_all


def _add_tool_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("executables")
    g.add_argument("--video2x", metavar="PATH", help="Video2X CLI executable (default: third_party copy, then PATH)")
    g.add_argument("--ffmpeg", metavar="PATH", help="FFmpeg executable")
    g.add_argument("--ffprobe", metavar="PATH", help="FFprobe executable")


def _add_processing_args(p: argparse.ArgumentParser, *, sharpen: bool = True) -> None:
    g = p.add_argument_group("processing")
    g.add_argument("--gpu", type=int, default=0, help="Vulkan device index from 'video2x --list-devices' (default 0)")
    g.add_argument("--model", default="realesr-animevideov3",
                   help="Real-ESRGAN model; validated against the installed Video2X (default realesr-animevideov3)")
    g.add_argument("--scale", type=int, default=2, help="scale factor 2, 3 or 4 (default 2)")
    if sharpen:
        g.add_argument("--sharpen", type=float, default=0.0,
                       help=f"extra unsharp-mask strength on luma, 0-{pipeline.SHARPEN_MAX} (default 0 = off)")
    g.add_argument("--crf", type=int, default=18, help="H.264 CRF (default 18)")
    g.add_argument("--preset", default="slow", choices=tools.X264_PRESETS, help="x264 preset (default slow)")
    g.add_argument("--timing", default="auto", choices=["auto", "source"],
                   help="auto: snap to an exact constant-rate grid only when every source timestamp is within one "
                        "time-base tick of it; source: always reproduce source ticks (default auto)")
    g.add_argument("--allow-interlaced", action="store_true",
                   help="process sources that look interlaced as progressive frames (no deinterlacing)")
    g.add_argument("--temp-dir", type=Path, help="directory for intermediates (default: <output dir>/.animenhance-tmp)")
    g.add_argument("--keep-temp", action="store_true", help="keep intermediate files")
    g.add_argument("--dry-run", action="store_true", help="show the planned operations without running them")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="enhance_animation.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="check dependencies, GPU access, models and encoders")
    _add_tool_args(p)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model", default="realesr-animevideov3")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--skip-inference-test", action="store_true", help="skip the tiny real GPU inference run")
    p.add_argument("--json", action="store_true", help="print JSON")

    p = sub.add_parser("inspect", help="report source properties and potential processing issues")
    p.add_argument("input", type=Path)
    _add_tool_args(p)
    p.add_argument("--scale", type=int, default=2, help="scale used for dimension checks (default 2)")
    p.add_argument("--full-scan", action="store_true",
                   help="decode every frame for timestamps and interlace flags (slower; default reads packets)")
    p.add_argument("--json", action="store_true", help="print the full JSON report")
    p.add_argument("--report", type=Path, help="also write the JSON report to this file")

    p = sub.add_parser("preview", help="render baseline / AI / AI+sharpen previews of one interval")
    p.add_argument("input", type=Path)
    p.add_argument("--start", default="00:05:00", help="interval start, seconds or HH:MM:SS[.ms] (default 00:05:00)")
    p.add_argument("--duration", type=float, default=20.0, help="interval length in seconds (default 20)")
    p.add_argument("--output-dir", type=Path, default=Path("previews"))
    p.add_argument("--preview-sharpen", type=float, default=0.5,
                   help="unsharp strength for the AI+sharpen preview (default 0.5, mild)")
    _add_tool_args(p)
    _add_processing_args(p, sharpen=False)

    p = sub.add_parser("enhance", help="enhance the full source")
    p.add_argument("input", type=Path)
    p.add_argument("--output", "-o", type=Path, required=True, help="output .mp4 path")
    p.add_argument("--overwrite", action="store_true", help="replace an existing output (never the source)")
    p.add_argument("--skip-decode-check", action="store_true", help="skip the final full-decode validation pass")
    p.add_argument("--timestamp-policy", default="strict", choices=["strict", "repair-terminal"],
                   help="strict (default here): reject timestamp defects; repair-terminal: re-time a short "
                        "(<=3 frames) terminal run of duplicate/backwards timestamps (see README)")
    _add_tool_args(p)
    _add_processing_args(p)
    return parser


def _settings(a: argparse.Namespace) -> pipeline.Settings:
    return pipeline.Settings(
        gpu=a.gpu, model=a.model, scale=a.scale, sharpen=getattr(a, "sharpen", 0.0), crf=a.crf, preset=a.preset,
        temp_dir=a.temp_dir, keep_temp=a.keep_temp, dry_run=a.dry_run, timing=a.timing,
        allow_interlaced=a.allow_interlaced, full_decode_check=not getattr(a, "skip_decode_check", False),
        timestamp_policy=getattr(a, "timestamp_policy", "strict"))


def _tool_names(a: argparse.Namespace) -> dict:
    return {"video2x": a.video2x, "ffmpeg": a.ffmpeg, "ffprobe": a.ffprobe}


def cmd_doctor(a: argparse.Namespace) -> int:
    from animenhance.doctor import doctor
    s = pipeline.Settings(gpu=a.gpu, model=a.model, scale=a.scale)
    result = doctor(_tool_names(a), s, inference_test=not a.skip_inference_test)
    if a.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        for c in result["checks"]:
            detail = c["detail"] if isinstance(c["detail"], str) else json.dumps(c["detail"], default=str)
            print(f"[{c['status'].upper():4}] {c['check']}: {detail}")
        print("\nOK" if result["ok"] else "\nPROBLEMS FOUND")
    return 0 if result["ok"] else 1


def cmd_inspect(a: argparse.Namespace) -> int:
    ts = tools.resolve_tools(**_tool_names(a))
    src_path = pipeline.check_input(a.input)
    log = RunLog()
    src = pipeline.analyze_source(ts, src_path, decode_scan=a.full_scan, log=log, scale=a.scale)
    rep = {"source_identity": pipeline.source_identity(src_path, hash_file=False), **pipeline.source_report(src)}
    if a.report:
        pipeline.write_json(a.report, rep)
    if a.json:
        print(json.dumps(rep, indent=2, default=str, ensure_ascii=False))
        return 0
    v = rep["video"]
    t = rep["timing"]
    print(f"File: {src_path}")
    print(f"Container: {rep['format']['format_long_name']}, duration {rep['format']['duration']} s, "
          f"start {rep['format']['start_time']} s")
    print(f"Video #{v['index']}: {v['codec']} ({v['profile']}), {v['width']}x{v['height']} "
          f"(coded {v['coded_width']}x{v['coded_height']}), SAR {v['sample_aspect_ratio']}, DAR {v['display_aspect_ratio']}, "
          f"{v['pix_fmt']}, field order {v['field_order']}")
    print(f"  colour: {v['color']} -> output tags {rep['output_color_tags']['color_primaries']}/"
          f"{rep['output_color_tags']['color_transfer']}/{rep['output_color_tags']['color_space']}")
    print(f"  rates: nominal {v['r_frame_rate']}, average {v['avg_frame_rate']}, time base {v['time_base']}, "
          f"start {v['start_time_s']} s, decoder reorder delay {v['decoder_reorder_delay']}")
    print(f"  frame timing ({t['scan_method']}): {t['frames']} frames, {t['classification']}, intervals "
          f"{_f(t['delta_min_ms'])}/{_f(t['delta_median_ms'])}/{_f(t['delta_max_ms'])} ms (min/median/max)"
          + (f", fits {t['cfr_fit']['rate']} fps within {t['cfr_fit']['max_deviation_ms']:.3f} ms" if t["cfr_fit"] else ""))
    print(f"  interlacing: {rep['interlacing']['verdict']} ({'; '.join(rep['interlacing']['reasons']) or 'no indicators'})")
    for s in rep["streams"]:
        if s["type"] == "video" and s["index"] == v["index"]:
            continue
        extra = ""
        if s["type"] == "audio":
            ext = rep["audio_extents"].get(str(s["index"]), {})
            extra = (f" {s['sample_rate']} Hz {s['channels']} ch {s['channel_layout']}, start {_f(ext.get('start_s'))} s, "
                     f"end {_f(ext.get('end_s'))} s")
        print(f"Stream #{s['index']}: {s['type']} {s['codec']}{extra}, language {s['language']}, default {s['default']}")
    print("Planned stream handling:")
    for e in rep["stream_plan"]:
        print(f"  #{e['index']} {e['type']}/{e['codec']}: {e['action']} - {e.get('detail', '')}")
    print("Issues:" if rep["issues"] else "Issues: none")
    for i in rep["issues"]:
        print(f"  [{i['severity']}] {i['message']}")
    return 0


def _f(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def cmd_preview(a: argparse.Namespace) -> int:
    ts = tools.resolve_tools(**_tool_names(a))
    log = RunLog()
    rep = pipeline.preview(ts, a.input, a.start, a.duration, a.output_dir, _settings(a), a.preview_sharpen, log)
    if rep.get("dry_run"):
        print("DRY RUN - planned preview:")
        for line in rep["plan"]:
            print("  " + line)
        print("Outputs:")
        for k, v in rep["outputs"].items():
            print(f"  {k}: {v}")
        for i in rep["issues"]:
            print(f"  [{i['severity']}] {i['message']}")
        return 0
    print("\nPreview outputs:")
    for k, v in rep["outputs"].items():
        print(f"  {k}: {v}")
    sp = rep["speed"]
    print(f"Inference speed: {sp['inference_fps_reported_by_video2x']} fps (Video2X), "
          f"{sp['inference_wall_fps_including_model_load']:.2f} fps wall clock incl. model load")
    if sp["estimated_full_run_hours_inference_only"] is not None:
        print(f"ESTIMATED full-movie runtime: ~{_hm(sp['estimated_full_run_hours_inference_only'])} for "
              f"{sp['source_total_frames']} frames (inference only; up to "
              f"{_hm(sp['estimated_full_run_hours_if_encode_is_serial'])} if encoding does not overlap). "
              f"This is an estimate extrapolated from the preview interval.")
    for key, val in rep["validation"].items():
        bad = [c for c in val["checks"] if not c["ok"]]
        print(f"Validation {key}: {'PASS' if not bad else 'FAIL'}" + "".join(f"\n    - {c['check']}: {c['detail']}" for c in bad))
    print(f"Report: {Path(a.output_dir).resolve()}")
    return 0 if rep["status"] == "ok" else 1


def cmd_enhance(a: argparse.Namespace) -> int:
    ts = tools.resolve_tools(**_tool_names(a))
    log = RunLog()
    rep = pipeline.enhance(ts, a.input, a.output, _settings(a), a.overwrite, log)
    if rep.get("dry_run"):
        print("DRY RUN - planned operations:")
        for line in rep["plan"]:
            print("  - " + line)
        print("Commands (paths under <run-id> are created at run time):")
        for c in rep["commands"]:
            print("  " + c)
        print("Stream handling:")
        for e in rep["source"]["stream_plan"]:
            print(f"  #{e['index']} {e['type']}/{e['codec']}: {e['action']} - {e.get('detail', '')}")
        for i in rep["issues"]:
            print(f"  [{i['severity']}] {i['message']}")
        return 0
    print(f"\nDone: {rep['output_path']}")
    print(f"Elapsed {_hm(rep['elapsed_s'] / 3600)}, Video2X {rep['speed']['video2x_fps']} fps; "
          f"report {Path(rep['output_path']).with_suffix('.report.json')}")
    if rep.get("timestamp_repair"):
        print(f"Timestamp repair applied: {rep['timestamp_repair']}")
    return 0


def _hm(hours) -> str:
    if hours is None:
        return "n/a"
    total = int(round(hours * 3600))
    return f"{total // 3600}h{total % 3600 // 60:02d}m{total % 60:02d}s"


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def _keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (released automatically on exit)."""
    if sys.platform == "win32":
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _interrupt)
    a = build_parser().parse_args(argv)
    if a.command in ("preview", "enhance") and not getattr(a, "dry_run", False):
        _keep_awake()
    handler = {"doctor": cmd_doctor, "inspect": cmd_inspect, "preview": cmd_preview, "enhance": cmd_enhance}[a.command]
    try:
        return handler(a)
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


if __name__ == "__main__":
    sys.exit(main())
