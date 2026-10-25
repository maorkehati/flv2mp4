"""Preview and full-movie enhancement pipelines."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional

from . import __version__, avtools, probe, tools, winpipe
from .procs import RunLog, ToolError, format_cmd, ffmpeg_frame_progress, run, video2x_progress
from .validate import AudioExpectation, Expectation, validate_output

PIX_FMT = "yuv420p"
SHARPEN_MAX = 2.0
COPY_AUDIO_CODECS = {"aac", "mp3"}
TEXT_SUB_CODECS = {"text", "subrip", "srt", "ass", "ssa", "mov_text", "webvtt"}
AAC_LAYOUTS = {"mono", "stereo", "2.1", "3.0", "3.0(back)", "4.0", "quad", "quad(side)", "5.0", "5.0(side)",
               "5.1", "5.1(side)", "6.1", "7.1", "7.1(wide)", "7.1(wide-side)"}

PRIMARIES = {"bt709": 1, "bt470m": 4, "bt470bg": 5, "smpte170m": 6, "smpte240m": 7, "film": 8, "bt2020": 9}
TRANSFERS = {"bt709": 1, "gamma22": 4, "gamma28": 5, "smpte170m": 6, "smpte240m": 7, "linear": 8,
             "iec61966-2-1": 13, "bt2020-10": 14, "bt2020-12": 15}
MATRICES = {"bt709": 1, "fcc": 4, "bt470bg": 5, "smpte170m": 6, "smpte240m": 7, "bt2020nc": 9}


class UsageError(ToolError):
    """Invalid arguments or an unsupported source, detected before expensive work starts."""


@dataclass
class Settings:
    gpu: int = 0
    model: str = "realesr-animevideov3"
    scale: int = 2
    sharpen: float = 0.0
    crf: int = 18
    preset: str = "slow"
    temp_dir: Optional[Path] = None
    keep_temp: bool = False
    dry_run: bool = False
    timing: str = "auto"
    allow_interlaced: bool = False
    full_decode_check: bool = True
    upscaler: str = "ai"  # "ai" (Video2X Real-ESRGAN) or "lanczos" (FFmpeg baseline)
    stream_intermediate: bool = True  # sharpen path: pipe Video2X's lossless output into FFmpeg (Windows)
    timestamp_policy: str = "strict"  # or "repair-terminal"
    profile: Optional[str] = None


# --------------------------------------------------------------------------------------------
# Validation of settings and tools
# --------------------------------------------------------------------------------------------

def check_settings(ts: tools.Toolset, s: Settings) -> dict:
    if s.scale not in (2, 3, 4):
        raise UsageError(f"--scale must be 2, 3 or 4 (got {s.scale})")
    if not (0 <= s.crf <= 51):
        raise UsageError(f"--crf must be between 0 and 51 (got {s.crf})")
    if s.preset not in tools.X264_PRESETS:
        raise UsageError(f"--preset must be one of {', '.join(tools.X264_PRESETS)} (got {s.preset})")
    if not (0.0 <= s.sharpen <= SHARPEN_MAX):
        raise UsageError(f"--sharpen must be between 0 and {SHARPEN_MAX} (got {s.sharpen})")
    if s.timing not in ("auto", "source"):
        raise UsageError("--timing must be 'auto' or 'source'")
    if s.timestamp_policy not in ("strict", "repair-terminal"):
        raise UsageError("--timestamp-policy must be 'repair-terminal' or 'strict'")
    if s.upscaler not in ("ai", "lanczos"):
        raise UsageError("upscaler must be 'ai' or 'lanczos'")
    missing = tools.missing_ffmpeg_features(ts.ffmpeg)
    if missing:
        raise UsageError(f"FFmpeg at {ts.ffmpeg} lacks required features: {missing}")
    if s.upscaler == "lanczos":
        return {"devices": [], "models": {}}
    models = tools.available_models(ts.video2x)
    if s.model not in models:
        raise UsageError(f"Model '{s.model}' is not supported by the installed Video2X. "
                         f"Supported: {', '.join(models)}")
    if s.scale not in models[s.model]:
        have = models[s.model]
        raise UsageError(f"Model '{s.model}' has no x{s.scale} weights installed "
                         f"(available scales: {have or 'none'}; expected files "
                         f"{tools.model_files(ts.video2x, s.model, s.scale)[0].name} and .bin)")
    devices = tools.video2x_devices(ts.video2x)
    if not any(d["index"] == s.gpu for d in devices):
        raise UsageError(f"GPU index {s.gpu} not reported by 'video2x --list-devices'. "
                         f"Available: {[(d['index'], d['name']) for d in devices] or 'none'}")
    return {"devices": list(devices), "models": models}


def pick_gpu(ts: tools.Toolset) -> tuple[int, str]:
    """Default Vulkan device: the first discrete GPU reported by Video2X, else device 0."""
    devices = tools.video2x_devices(ts.video2x)
    if not devices:
        raise UsageError("Video2X reports no Vulkan devices; AI profiles need a Vulkan-capable GPU "
                         "(run 'enhance_animation.py doctor').")
    discrete = [d for d in devices if "discrete" in str(d.get("type", "")).lower()]
    d = discrete[0] if discrete else devices[0]
    return int(d["index"]), d["name"]


def check_input(path: Path) -> Path:
    path = Path(path).expanduser()
    if not path.exists():
        raise UsageError(f"Input not found: {path}")
    if not path.is_file():
        raise UsageError(f"Input is not a file: {path}")
    try:
        with open(path, "rb") as fh:
            fh.read(16)
    except OSError as e:
        raise UsageError(f"Input is not readable: {path}: {e}")
    return path.resolve()


def same_file(a: Path, b: Path) -> bool:
    try:
        if a.exists() and b.exists():
            return os.path.samefile(a, b)
    except OSError:
        pass
    return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))


def check_output(src: Path, out: Path, overwrite: bool) -> Path:
    out = Path(out).expanduser().resolve()
    if same_file(src, out):
        raise UsageError("Output path is the source file; refusing to overwrite the original.")
    if out.suffix.lower() != ".mp4":
        raise UsageError(f"Output must have a .mp4 extension: {out}")
    if out.exists() and not overwrite:
        raise UsageError(f"Output already exists: {out} (use --overwrite to replace it)")
    if out.exists() and out.is_dir():
        raise UsageError(f"Output path is a directory: {out}")
    return out


# --------------------------------------------------------------------------------------------
# Source analysis
# --------------------------------------------------------------------------------------------

def color_plan(v: dict) -> dict:
    """Colour tags for the output. Pixel values keep the source's YUV encoding, so tags must too.

    Untagged sources are tagged with what a player assumes for the *source* resolution, because the
    upscaled output may cross the SD/HD boundary where players switch their default to BT.709.
    """
    h = v.get("height") or 0
    family = "bt709" if h >= 720 else ("bt470bg" if h in (288, 576) else "smpte170m")
    assumed = []

    def pick(value, table, fallback):
        if value not in probe.UNKNOWN_COLOR and value in table:
            return value
        assumed.append(fallback)
        return fallback

    prim = pick(v.get("color_primaries"), PRIMARIES, family)
    trc = pick(v.get("color_transfer"), TRANSFERS, "bt709" if family == "bt709" else "smpte170m")
    mat = pick(v.get("color_space"), MATRICES, family)
    notes = []
    if assumed:
        notes.append(f"source colour metadata incomplete; tagged output as {prim}/{trc}/{mat} "
                     f"(the default a player assumes for {h}-line video)")
    pix = v.get("pix_fmt") or ""
    if v.get("color_range") == "pc" and not pix.startswith("yuvj"):
        notes.append("source is flagged full range on a non-J pixel format; Video2X 6.4.0 converts it as limited "
                     "range, so the darkest/brightest code values may be clipped")
    return {"color_primaries": prim, "color_transfer": trc, "color_space": mat, "color_range": "tv",
            "h264_metadata": {"colour_primaries": PRIMARIES[prim], "transfer_characteristics": TRANSFERS[trc],
                              "matrix_coefficients": MATRICES[mat], "video_full_range_flag": 0},
            "assumed": bool(assumed), "notes": notes}


def aac_bitrate(channels: int) -> str:
    if channels <= 1:
        return "128k"
    if channels == 2:
        return "192k"
    return f"{min(64 * channels, 512)}k"


def stream_plan(info: dict, video_index: int) -> list[dict]:
    plan = []
    audio_seen = 0
    any_default = any(s.get("disposition", {}).get("default") for s in info["streams"] if s.get("codec_type") == "audio")
    for s in info["streams"]:
        idx = s["index"]
        typ = s.get("codec_type")
        codec = s.get("codec_name")
        tags = s.get("tags", {}) or {}
        entry = {"index": idx, "type": typ, "codec": codec, "language": tags.get("language")}
        if idx == video_index:
            entry.update(action="enhance", detail="Real-ESRGAN via Video2X, H.264 output")
        elif typ == "video":
            what = "cover art" if s.get("disposition", {}).get("attached_pic") else "additional video stream"
            entry.update(action="exclude", detail=f"{what}; only the first video stream is processed")
        elif typ == "audio":
            channels = s.get("channels") or 2
            layout = s.get("channel_layout")
            default = bool(s.get("disposition", {}).get("default")) if any_default else audio_seen == 0
            entry.update(channels=channels, channel_layout=layout, sample_rate=probe._int(s.get("sample_rate")),
                         default=default)
            if codec in COPY_AUDIO_CODECS:
                entry.update(action="copy", detail=f"{codec} is MP4-compatible; stream copied")
            elif layout and layout not in AAC_LAYOUTS:
                entry.update(action="exclude", detail=f"channel layout {layout} not supported by the AAC encoder")
            else:
                entry.update(action="convert", target="aac", bitrate=aac_bitrate(channels),
                             detail=f"{codec} cannot be stored in MP4 compatibly; converted to AAC "
                                    f"{aac_bitrate(channels)} keeping {channels} channels"
                                    + (f" ({layout})" if layout else ""))
            audio_seen += 1
        elif typ == "subtitle":
            if codec in TEXT_SUB_CODECS:
                entry.update(action="convert", target="mov_text", detail=f"text subtitles ({codec}) converted to MP4 mov_text")
            else:
                entry.update(action="exclude", detail=f"bitmap/unsupported subtitle codec {codec} cannot be stored in MP4")
        elif typ == "data":
            entry.update(action="exclude", detail="data stream (e.g. FLV script/cue data) has no MP4 equivalent")
        elif typ == "attachment":
            entry.update(action="exclude", detail="attachments are not supported in MP4")
        else:
            entry.update(action="exclude", detail=f"stream type {typ} not supported")
        plan.append(entry)
    return plan


@dataclass
class Source:
    path: Path
    info: dict
    video: dict
    video_index: int
    time_base: Fraction
    global_start_s: float
    scan: probe.FrameScan
    timing: probe.TimingAnalysis
    interlace: dict
    streams: list[dict]
    plan: list[dict]
    color: dict
    issues: list[dict]
    audio_extents: dict = field(default_factory=dict)
    original_scan: Optional[probe.FrameScan] = None
    timing_repair: Optional[dict] = None
    timing_defect: Optional[str] = None
    timestamp_policy: str = "strict"

    @property
    def width(self) -> int:
        return int(self.video["width"])

    @property
    def height(self) -> int:
        return int(self.video["height"])

    @property
    def sar(self) -> Fraction:
        return probe.parse_ratio(self.video.get("sample_aspect_ratio")) or Fraction(1)

    @property
    def decoder_delay(self) -> int:
        return int(self.video.get("has_b_frames") or 0)

    @property
    def duration_s(self) -> Optional[float]:
        return probe._num(self.info.get("format", {}).get("duration"))


def analyze_source(ts: tools.Toolset, path: Path, *, decode_scan: bool, log: RunLog,
                   run_idet: bool = True, scale: Optional[int] = None, timestamp_policy: str = "strict",
                   decode_audio: bool = True) -> Source:
    log.note(f"Inspecting {path}")
    info = probe.ffprobe_json(ts.ffprobe, path)
    v = probe.first_video_stream(info)
    vidx = int(v["index"])
    tb = probe.parse_ratio(v.get("time_base"))
    if tb is None:
        raise ToolError("Video stream has no time base")
    duration = probe._num(info.get("format", {}).get("duration"))
    scan = probe.scan_frames(ts.ffprobe, path, vidx, decode=decode_scan, time_base=tb, expected_duration_s=duration,
                             label="decoding frame timestamps" if decode_scan else "reading packet timestamps")
    if not decode_scan:
        scan = probe.splice_decoded_tail(ts.ffprobe, path, vidx, scan, duration)
    original_scan = scan
    repair = defect = None
    try:
        scan, repair = probe.repair_terminal_suffix(scan, timestamp_policy, probe.nominal_rates(v))
    except probe.TimingDefect as e:
        defect = str(e)
    timing = probe.analyze_timing(scan, probe.nominal_rates(v))
    idet = probe.run_idet(ts.ffmpeg, path, vidx, duration) if run_idet else None
    inter = probe.interlace_verdict(v, scan, idet)
    plan = stream_plan(info, vidx)
    color = color_plan(v)
    audio_ext = {}
    for s in info["streams"]:
        if s.get("codec_type") == "audio":
            audio_ext[s["index"]] = probe.stream_extent(ts.ffprobe, path, s["index"], decode=decode_audio)
    src = Source(path=path, info=info, video=v, video_index=vidx, time_base=tb,
                 global_start_s=probe.source_global_start(info), scan=scan, timing=timing, interlace=inter,
                 streams=probe.summarize_streams(info), plan=plan, color=color, issues=[], audio_extents=audio_ext,
                 original_scan=original_scan, timing_repair=repair, timing_defect=defect,
                 timestamp_policy=timestamp_policy)
    src.issues = find_issues(src, scale)
    return src


def find_issues(src: Source, scale: Optional[int]) -> list[dict]:
    issues = []

    def add(severity: str, code: str, message: str) -> None:
        issues.append({"severity": severity, "code": code, "message": message})

    t = src.timing
    if src.timing_defect:
        add("error", "timestamp-defect", src.timing_defect)
    elif t.classification == "non-monotonic":
        add("error", "timestamp-defect",
            "Video frame timestamps are not strictly increasing. MP4 cannot represent this timing faithfully; "
            "remux or repair the source first (e.g. fix the FLV timestamps) and re-run.")
    if src.timing_repair:
        r = src.timing_repair
        changes = ", ".join(f"#{f['index']} {f['original_pts_s']:.3f}->{f['repaired_pts_s']:.3f}s" for f in r["frames"])
        audio_ends = [e.get("end_s") for e in src.audio_extents.values() if e.get("end_s") is not None]
        rel = ""
        if audio_ends:
            a_end = max(audio_ends)
            rel = (f" Audio (unchanged) ends at {a_end:.3f} s: video ended {1000 * (r['original_video_end_s'] - a_end):+.1f} ms "
                   f"relative to audio before, {1000 * (r['repaired_video_end_s'] - a_end):+.1f} ms after.")
            r["audio_end_s"] = a_end
            r["video_minus_audio_end_before_ms"] = 1000 * (r["original_video_end_s"] - a_end)
            r["video_minus_audio_end_after_ms"] = 1000 * (r["repaired_video_end_s"] - a_end)
        add("warning", "timestamp-repair",
            f"The last {r['suffix_length']} frame(s) after frame #{r['anchor_index']} ({r['anchor_pts_s']:.3f} s) had "
            f"duplicate/backwards timestamps ({r['pattern']}); all frames are kept and re-timed to consecutive "
            f"{r['inferred_interval_s']} s slots ({r['inferred_rate_fps']} fps from {r['intervals_used']} stable "
            f"intervals): {changes}. Video end {r['original_video_end_s']:.3f} -> {r['repaired_video_end_s']:.3f} s "
            f"({r['video_duration_change_s'] * 1000:+.1f} ms).{rel} Earlier frames and audio are unchanged.")
    if t.classification == "vfr":
        add("info", "variable-frame-timing",
            f"Frame intervals vary ({t.delta_min_ms:.2f}-{t.delta_max_ms:.2f} ms). Timestamps are restored "
            f"frame-by-frame after enhancement; no frame-rate conversion is applied.")
    elif t.classification == "cfr-within-rounding":
        add("info", "timestamp-rounding",
            f"Frame timestamps match {t.cfr_fit['rate']} fps within {t.cfr_fit['max_deviation_ms']:.3f} ms "
            f"(container rounding).")
    if src.scan.missing_ts:
        add("warning", "missing-timestamps", f"{src.scan.missing_ts} frames had no timestamp and were not counted")
    v = src.video
    nominal = probe.parse_ratio(v.get("r_frame_rate"))
    avg = probe.parse_ratio(v.get("avg_frame_rate"))
    if nominal and avg and nominal != avg:
        add("info", "nominal-vs-average-rate",
            f"Nominal rate {probe.ratio_str(nominal)} differs from average {probe.ratio_str(avg)}; timing is taken "
            f"from the actual frame timestamps, not from either rate.")
    verdict = src.interlace["verdict"]
    if verdict == "likely-interlaced":
        add("error", "interlaced",
            "Source looks interlaced (" + "; ".join(src.interlace["reasons"]) + "). Deinterlacing or inverse "
            "telecine is not supported by this tool, and Real-ESRGAN would upscale the combing. Deinterlace the "
            "source separately, or pass --allow-interlaced to process the frames as-is.")
    elif verdict == "possibly-interlaced-or-telecined":
        add("warning", "possibly-interlaced",
            "Content analysis suggests combing or telecine (" + "; ".join(src.interlace["reasons"]) + "), but the "
            "stream is not coded as interlaced. This heuristic also triggers on sharp, fast-moving progressive "
            "animation. Processing continues as progressive frames; check the preview for comb artifacts and "
            "deinterlace/IVTC the source separately if they are present.")
    if src.sar != 1:
        add("info", "non-square-pixels",
            f"Sample aspect ratio {probe.ratio_str(src.sar)}; it is preserved, so the display aspect ratio stays "
            f"{probe.ratio_str(Fraction(src.width, src.height) * src.sar)}.")
    if src.video.get("coded_width") and (src.video.get("coded_width"), src.video.get("coded_height")) != (src.width, src.height):
        add("info", "coded-size",
            f"Coded size {src.video.get('coded_width')}x{src.video.get('coded_height')} differs from the display "
            f"size {src.width}x{src.height} (decoder cropping); the display size is used.")
    if scale and ((src.width * scale) % 2 or (src.height * scale) % 2):
        add("error", "odd-dimensions",
            f"{src.width}x{src.height} x{scale} gives odd dimensions, which yuv420p H.264 cannot encode. "
            f"Use an even scale factor.")
    if src.decoder_delay:
        add("info", "decoder-delay",
            f"The video decoder has a reorder delay of {src.decoder_delay} frame(s). Video2X 6.4.0 does not drain "
            f"its decoder at end of input, so the wrapper appends {src.decoder_delay} IDR copies to a stream-copied "
            f"input to recover the final frames, and verifies the frame count.")
    for note in src.color["notes"]:
        add("info", "color", note)
    vstart = probe._num(v.get("start_time"))
    for s in src.info["streams"]:
        st = probe._num(s.get("start_time"))
        if s.get("codec_type") == "audio" and st is not None and vstart is not None and abs(st - vstart) > 0.0005:
            add("info", "av-start-offset",
                f"Audio stream {s['index']} starts {(st - vstart) * 1000:+.1f} ms relative to video; the offset is preserved.")
    if src.global_start_s and abs(src.global_start_s) > 0.0005:
        add("info", "nonzero-start",
            f"Source timeline starts at {src.global_start_s:.3f} s; output is shifted so the earliest stream starts "
            f"at 0 while relative offsets are kept.")
    for e in src.plan:
        if e["action"] == "exclude":
            add("warning", "stream-excluded", f"Stream {e['index']} ({e['type']}/{e['codec']}): {e['detail']}")
        elif e["type"] == "audio" and e["action"] == "convert":
            add("info", "audio-convert", f"Stream {e['index']}: {e['detail']}")
    if not any(s.get("codec_type") == "audio" for s in src.info["streams"]):
        add("info", "no-audio", "Source has no audio streams; output will be video-only.")
    vdur = t.end_s - t.first_pts_s if t.end_s is not None and t.first_pts_s is not None else None
    for idx, ext in src.audio_extents.items():
        if ext.get("end_s") is not None and t.end_s is not None and abs(ext["end_s"] - t.end_s) > 0.5:
            add("info", "duration-mismatch",
                f"Audio stream {idx} ends at {ext['end_s']:.3f} s while video ends at {t.end_s:.3f} s; both "
                f"are kept as-is (no -shortest trimming).")
    return issues


def source_identity(path: Path, *, hash_file: bool = True) -> dict:
    st = path.stat()
    ident = {"path": str(path), "size_bytes": st.st_size,
             "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))}
    if hash_file:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
                h.update(chunk)
        ident["sha256"] = h.hexdigest()
    return ident


def source_report(src: Source) -> dict:
    return {
        "format": {k: src.info.get("format", {}).get(k) for k in ("format_name", "format_long_name", "duration",
                                                                  "start_time", "bit_rate", "size")},
        "format_tags": src.info.get("format", {}).get("tags", {}),
        "streams": src.streams,
        "video": {
            "index": src.video_index, "codec": src.video.get("codec_name"), "profile": src.video.get("profile"),
            "width": src.width, "height": src.height,
            "coded_width": src.video.get("coded_width"), "coded_height": src.video.get("coded_height"),
            "sample_aspect_ratio": probe.ratio_str(src.sar),
            "display_aspect_ratio": probe.ratio_str(Fraction(src.width, src.height) * src.sar),
            "pix_fmt": src.video.get("pix_fmt"), "field_order": src.video.get("field_order"),
            "color": {k: src.video.get(k) for k in ("color_range", "color_space", "color_transfer", "color_primaries")},
            "r_frame_rate": src.video.get("r_frame_rate"), "avg_frame_rate": src.video.get("avg_frame_rate"),
            "time_base": probe.ratio_str(src.time_base), "start_time_s": probe._num(src.video.get("start_time")),
            "decoder_reorder_delay": src.decoder_delay,
        },
        "timing": {**asdict(src.timing), "scan_method": src.scan.source},
        "timestamp_policy": src.timestamp_policy,
        "timestamp_repair": src.timing_repair,
        "timestamp_defect": src.timing_defect,
        "interlacing": src.interlace,
        "audio_extents": {str(k): v for k, v in src.audio_extents.items()},
        "global_start_s": src.global_start_s,
        "stream_plan": src.plan,
        "output_color_tags": {k: v for k, v in src.color.items() if k != "h264_metadata"},
        "issues": src.issues,
    }


# --------------------------------------------------------------------------------------------
# Command builders
# --------------------------------------------------------------------------------------------

def video2x_cmd(ts: tools.Toolset, inp: Path, out: Path, s: Settings, codec: str) -> list[str]:
    cmd = [str(ts.video2x), "-i", str(inp), "-o", str(out), "-p", "realesrgan", "--realesrgan-model", s.model,
           "-s", str(s.scale), "-d", str(s.gpu), "--no-copy-streams", "-c", codec, "--pix-fmt", PIX_FMT,
           "--log-level", "info"]
    if codec == "libx264":
        cmd += ["-e", f"crf={s.crf}", "-e", f"preset={s.preset}", "-e", "profile=high"]
    return cmd


def x264_args(s: Settings, color: dict) -> list[str]:
    return ["-c:v", "libx264", "-crf", str(s.crf), "-preset", s.preset, "-profile:v", "high", "-pix_fmt", PIX_FMT,
            "-color_range", "tv", "-colorspace", color["color_space"], "-color_primaries", color["color_primaries"],
            "-color_trc", color["color_transfer"]]


def h264_metadata_bsf(color: dict, sar: Fraction) -> str:
    m = color["h264_metadata"]
    return (f"h264_metadata=sample_aspect_ratio={sar.numerator}/{sar.denominator}"
            f":colour_primaries={m['colour_primaries']}:transfer_characteristics={m['transfer_characteristics']}"
            f":matrix_coefficients={m['matrix_coefficients']}:video_full_range_flag={m['video_full_range_flag']}")


def unsharp_filter(strength: float) -> str:
    return f"unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount={strength:.3f}:chroma_msize_x=5:chroma_msize_y=5:chroma_amount=0"


def encode_filter(s: Settings, sar: Fraction, *, width: int, height: int, sharpen: float) -> str:
    """Filter chain for the single lossy encode, identical to the tested preview variants.

    AI input is already at the target size: '[unsharp,]setsar,format' (preview 'ai' / 'ai_sharp').
    Lanczos input is the source: index timestamps first (the source may carry a repaired duplicate),
    then 'scale=...:flags=lanczos[,unsharp],setsar,format' (preview 'baseline').
    """
    tail = f"setsar={sar.numerator}/{sar.denominator},format={PIX_FMT}"
    sharp = f"{unsharp_filter(sharpen)}," if sharpen > 0 else ""
    if s.upscaler == "lanczos":
        return f"setpts=N/(25*TB),scale={width}:{height}:flags=lanczos,{sharp}{tail}"
    return f"{sharp}{tail}"


def ffbase(ts: tools.Toolset) -> list[str]:
    return [str(ts.ffmpeg), "-hide_banner", "-nostdin", "-y", "-stats_period", "2"]


# --------------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------------

def make_workdir(base: Optional[Path], default_parent: Path, tag: str) -> Path:
    parent = Path(base).expanduser().resolve() if base else default_parent / ".animenhance-tmp"
    parent.mkdir(parents=True, exist_ok=True)
    work = parent / f"{tag}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    work.mkdir()
    return work


def free_bytes(path: Path) -> int:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


MIN_FREE_BYTES = 1024 ** 3


class Stopwatch:
    """Wall time per pipeline stage, in the order the stages ran."""

    def __init__(self):
        self.laps: dict[str, float] = {}
        self._t = time.monotonic()

    def lap(self, name: str) -> None:
        now = time.monotonic()
        self.laps[name] = round(self.laps.get(name, 0.0) + now - self._t, 3)
        self._t = now


def disk_guard(*paths: Path):
    """Monitor for run(): stops a long step before the disk fills completely."""
    def check() -> Optional[str]:
        for p in paths:
            free = free_bytes(p)
            if free < MIN_FREE_BYTES:
                return f"free space on the drive holding {p} fell to {gib(free)}; stopped to avoid filling the disk"
        return None
    return check


def gib(n: float) -> str:
    return f"{n / 1024 ** 3:.2f} GiB"


def tool_versions(ts: tools.Toolset, s: Settings) -> dict:
    param, binf = tools.model_files(ts.video2x, s.model, s.scale)
    import av
    import numpy
    return {
        "animenhance": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "video2x": {"path": str(ts.video2x), "version": tools.video2x_version(ts.video2x)},
        "ffmpeg": {"path": str(ts.ffmpeg), "version": tools.ffmpeg_version(ts.ffmpeg)},
        "ffprobe": {"path": str(ts.ffprobe), "version": tools.ffmpeg_version(ts.ffprobe)},
        "pyav": {"version": av.__version__, "libavformat": ".".join(map(str, av.library_versions["libavformat"]))},
        "numpy": numpy.__version__,
        "model": {"name": s.model, "scale": s.scale, "param": str(param), "bin": str(binf),
                  "param_sha256": _sha256(param), "bin_sha256": _sha256(binf)},
        "gpu": next((d for d in tools.video2x_devices(ts.video2x) if d["index"] == s.gpu), None),
        "third_party_manifest": tools.manifest(),
    }


def _sha256(p: Path) -> Optional[str]:
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def target_timeline(src: Source, frame_ts: list[int], origin_ticks: int, s: Settings) -> tuple[list[int], Fraction, int, str]:
    """Output timestamps for the given source frames, shifted by ``origin_ticks``.

    'source' mode reproduces the source ticks exactly. 'auto' mode snaps to the exact constant-rate grid
    only when every source timestamp is already within one tick of that grid (container rounding).
    """
    tb = src.time_base
    last = src.timing.last_frame_duration_ticks or 1
    if len(frame_ts) > 1 and frame_ts[-1] == src.scan.ts[-1]:
        last_dur = last
    else:
        nxt = [t for t in src.scan.ts if t > frame_ts[-1]]
        last_dur = (nxt[0] - frame_ts[-1]) if nxt else last
    fit = src.timing.cfr_fit
    if (s.timing == "auto" and fit and src.timing.classification in ("cfr", "cfr-within-rounding")
            and tb.numerator == 1 and frame_ts[0] in src.scan.ts):
        rate = probe.parse_ratio(fit["rate"])
        out_den = math.lcm(rate.numerator, tb.denominator)
        out_tb = Fraction(1, out_den)
        frame_ticks = Fraction(out_den) / rate
        if frame_ticks.denominator == 1:
            first_idx = src.scan.ts.index(frame_ts[0])
            base = (src.scan.ts[0] - origin_ticks) * (out_den // tb.denominator)
            pts = [base + (first_idx + i) * int(frame_ticks) for i in range(len(frame_ts))]
            return pts, out_tb, int(frame_ticks), f"constant {fit['rate']} fps grid (source within " \
                                                  f"{fit['max_deviation_ms']:.3f} ms)"
    return [t - origin_ticks for t in frame_ts], tb, last_dur, "exact source timestamps"


# --------------------------------------------------------------------------------------------
# Audio extraction for previews and final mux
# --------------------------------------------------------------------------------------------

def audio_entries(src: Source) -> list[dict]:
    return [e for e in src.plan if e["type"] == "audio" and e["action"] in ("copy", "convert")]


def sub_entries(src: Source) -> list[dict]:
    return [e for e in src.plan if e["type"] == "subtitle" and e["action"] == "convert"]


def final_mux_cmd(ts: tools.Toolset, src: Source, video: Path, out: Path, origin_s: float) -> list[str]:
    cmd = ffbase(ts) + ["-copyts", "-i", str(video)]
    if abs(origin_s) > 0:
        cmd += ["-itsoffset", f"{-origin_s:.6f}"]
    cmd += ["-i", str(src.path), "-map", "0:v:0"]
    for e in audio_entries(src):
        cmd += ["-map", f"1:{e['index']}"]
    for e in sub_entries(src):
        cmd += ["-map", f"1:{e['index']}"]
    cmd += ["-c:v", "copy", "-bsf:v", h264_metadata_bsf(src.color, src.sar)]
    for i, e in enumerate(audio_entries(src)):
        if e["action"] == "copy":
            cmd += [f"-c:a:{i}", "copy"]
        else:
            cmd += [f"-c:a:{i}", "aac", f"-b:a:{i}", e["bitrate"]]
        cmd += [f"-disposition:a:{i}", "default" if e["default"] else "0"]
        if e.get("language"):
            cmd += [f"-metadata:s:a:{i}", f"language={e['language']}"]
    for i, e in enumerate(sub_entries(src)):
        cmd += [f"-c:s:{i}", "mov_text"]
        if e.get("language"):
            cmd += [f"-metadata:s:s:{i}", f"language={e['language']}"]
    cmd += ["-map_metadata", "1", "-map_chapters", "-1", "-movflags", "+faststart", "-f", "mp4", str(out)]
    return cmd


def audio_expectations(src: Source, origin_s: float, window: Optional[tuple[float, float]] = None,
                       force_aac: bool = False) -> list[AudioExpectation]:
    exps = []
    for e in audio_entries(src):
        ext = src.audio_extents.get(e["index"], {})
        start, end = ext.get("start_s"), ext.get("end_s")
        if window is not None and start is not None and end is not None:
            start, end = max(start, window[0]), min(end, window[1])
        convert = force_aac or e["action"] == "convert"
        sr = e.get("sample_rate") or 48000
        frame = 1024 / sr if convert else 0.0
        exps.append(AudioExpectation(
            source_index=e["index"], codec="aac" if convert else e["codec"], channels=e["channels"],
            sample_rate=sr, language=e.get("language"), default=e["default"],
            start_s=None if start is None else start - origin_s, end_s=None if end is None else end - origin_s,
            start_tol_s=0.0015 + (2.0 / sr if convert else 0.0), end_tol_s=0.0015 + frame))
    return exps


# --------------------------------------------------------------------------------------------
# Video2X execution with checks
# --------------------------------------------------------------------------------------------

TEARDOWN_CRASH_CODES = {0xC0000409, -1073740791}


def video2x_exit_status(res) -> Optional[str]:
    """None if the run succeeded; a warning string for the known teardown crash; raises otherwise.

    Video2X 6.4.0 on this Windows/Intel setup exits with 0xC0000409 (fail-fast) during process teardown,
    after the output trailer has been written and success has been logged. That case is accepted only when
    the success message is present; the output is then verified independently by the caller.
    """
    if res.returncode == 0:
        return None
    if res.returncode in TEARDOWN_CRASH_CODES and "Video processed successfully" in res.output:
        return ("Video2X exited with 0xC0000409 during teardown after logging success; output accepted only "
                "after independent frame-count/decode verification")
    raise ToolError(f"Video2X failed with exit code {res.returncode}", tail=res.output.splitlines())


def run_video2x(ts: tools.Toolset, inp: Path, out: Path, s: Settings, codec: str, *, log: RunLog,
                expected_frames: int, label: str, monitor=None, output_is_file: bool = True) -> dict:
    cmd = video2x_cmd(ts, inp, out, s, codec)
    res = run(cmd, log=log, label=label, progress=video2x_progress, progress_total=expected_frames,
              cwd=ts.video2x.parent, no_window=True, check=False,
              monitor=monitor or (disk_guard(out.parent) if output_is_file else None))
    warning = video2x_exit_status(res)
    if warning:
        log.note("warning: " + warning)
    speed = None
    frames_reported = None
    for line in res.output.splitlines():
        if "Average processing speed" in line:
            try:
                speed = float(line.rsplit(":", 1)[1].split()[0])
            except (ValueError, IndexError):
                pass
        if "Total frames processed" in line:
            try:
                frames_reported = int(line.rsplit(":", 1)[1].strip())
            except ValueError:
                pass
    if output_is_file and (not out.is_file() or out.stat().st_size == 0):
        raise ToolError(f"Video2X reported success but produced no output at {out}", tail=res.output.splitlines())
    return {"elapsed_s": res.elapsed_s, "reported_fps": speed, "frames_reported": frames_reported,
            "wall_fps": expected_frames / res.elapsed_s if res.elapsed_s else None,
            "exit_code": res.returncode, "warning": warning}


def run_video2x_streamed(ts: tools.Toolset, inp: Path, encoded: Path, s: Settings, vf: str, color: dict, *,
                         log: RunLog, expected_frames: int, monitor=None) -> tuple[dict, dict]:
    """Video2X writes lossless FFV1/Matroska into a named pipe; FFmpeg reads it from stdin and applies the
    profile's filter and libx264 encode concurrently. Same frames and the same single lossy encode as the
    file route, without the multi-GiB intermediate and without a serial encode after inference."""
    import threading
    from collections import deque

    from .procs import CommandRecord, Progress, release, spawn

    relay = winpipe.PipeRelay(".mkv")
    ff_cmd = ffbase(ts) + ["-f", "matroska", "-i", "pipe:0", "-map", "0:v:0", "-vf", vf, "-fps_mode", "passthrough"] \
        + x264_args(s, color) + ["-an", str(encoded)]
    rec = CommandRecord(label="sharpen + encode (streamed from Video2X)", cmd=[str(c) for c in ff_cmd], cwd=None)
    log.records.append(rec)
    log.write(f"\n$ ({rec.label}) {format_cmd(rec.cmd)}")
    t0 = time.monotonic()
    ff = spawn(ff_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    ff_tail: deque = deque(maxlen=40)
    ff_frames = {"n": 0}

    def read_ff() -> None:
        assert ff.stderr is not None
        pending = b""
        while True:
            chunk = ff.stderr.read1(65536) if hasattr(ff.stderr, "read1") else ff.stderr.read(4096)
            if not chunk:
                break
            parts = re.split(rb"[\r\n]", pending + chunk)
            pending = parts.pop()
            for p in parts:
                line = p.decode("utf-8", "replace").strip()
                if not line:
                    continue
                m = re.search(r"frame=\s*(\d+)", line)
                if m and line.startswith("frame="):
                    ff_frames["n"] = int(m.group(1))
                    continue
                ff_tail.append(line)
                log.write("  [ffmpeg] " + line)

    reader = threading.Thread(target=read_ff, daemon=True)
    reader.start()
    relay.start(ff.stdin)
    try:
        v2x = run_video2x(ts, inp, Path(relay.name), s, "ffv1", log=log, expected_frames=expected_frames,
                          label="Video2X", monitor=monitor, output_is_file=False)
        relay.unblock()
        relay.join()
        ff.wait()
        reader.join(10)
    except BaseException:
        release(ff)
        relay.unblock()
        relay.join(5)
        raise
    finally:
        relay.close()
        release(ff)
    rec.returncode = ff.returncode
    rec.elapsed_s = round(time.monotonic() - t0, 3)
    if relay.error:
        raise ToolError(f"Pipe relay failed: {relay.error}")
    if ff.returncode != 0:
        raise ToolError(f"Streamed FFmpeg encode failed with exit code {ff.returncode}", cmd=rec.cmd, tail=list(ff_tail))
    enc = {"streamed": True, "pipe_bytes": relay.bytes, "elapsed_s": rec.elapsed_s,
           "encoder_finished_after_video2x_s": round(rec.elapsed_s - v2x["elapsed_s"], 3)}
    return v2x, enc


def count_frames(ts: tools.Toolset, path: Path) -> int:
    info = probe.ffprobe_json(ts.ffprobe, path)
    v = probe.first_video_stream(info)
    tb = probe.parse_ratio(v.get("time_base"))
    return len(probe.scan_frames(ts.ffprobe, path, int(v["index"]), decode=False, time_base=tb,
                                 show_progress=False).ts)


# --------------------------------------------------------------------------------------------
# Preview
# --------------------------------------------------------------------------------------------

def parse_time(text: str) -> float:
    text = str(text).strip()
    try:
        if ":" in text:
            value = 0.0
            for part in text.split(":"):
                value = value * 60 + float(part)
            return value
        return float(text)
    except ValueError:
        raise UsageError(f"Invalid time value: {text!r} (use seconds or HH:MM:SS[.ms])")


def even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


def extract_interval(ts: tools.Toolset, src_path: Path, video_index: int, frame_ts: list[int], tb: Fraction,
                     seek_s: float, out: Path, log: RunLog) -> tuple[list[str], dict]:
    """Decode exactly the frames with the given original timestamps (presentation order) into lossless FFV1.

    Frames are taken from the first timestamp onward by count, so defective stamps inside the interval
    cannot drop or reorder frames; the clip carries index timestamps. The result is verified against an
    independent decode: same PTS sequence, pixel-identical frames. Returns (seek args used, verification).
    """
    n = len(frame_ts)
    lo = (min(frame_ts) - 0.5) * float(tb)
    seek_args: list[str] = ["-ss", f"{seek_s:.3f}"] if seek_s > 0 else []
    last_error = None
    for attempt in ("seek", "from-start") if seek_args else ("from-start",):
        if attempt == "from-start":
            seek_args = []
        run(ffbase(ts) + ["-copyts"] + seek_args + ["-i", str(src_path), "-map", f"0:{video_index}",
                          "-vf", f"trim=start={lo:.6f},setpts=N/(25*TB)", "-frames:v", str(n),
                          "-fps_mode", "passthrough", "-c:v", "ffv1", "-level", "3", "-an", "-sn", "-dn", str(out)],
            log=log, label=f"extract lossless interval ({attempt})", progress=ffmpeg_frame_progress,
            progress_total=n)
        got = count_frames(ts, out)
        if got != n:
            last_error = f"{got} frames instead of {n}"
        else:
            try:
                check = avtools.verify_interval_exact(src_path, video_index, seek_s if seek_args else 0.0,
                                                      frame_ts, out)
                return seek_args, check
            except ToolError as e:
                last_error = str(e)
        log.note(f"Interval extraction ({attempt}) rejected: {last_error}")
    raise ToolError(f"Could not extract the interval losslessly: {last_error}")


def cut_audio(ts: tools.Toolset, src: Source, seek_args: list[str], first_s: float, end_s: float, origin_s: float,
              out: Path, log: RunLog, label: str) -> Optional[Path]:
    """AAC-encode every kept audio track over [first_s, end_s), re-based so origin_s becomes zero."""
    audio = audio_entries(src)
    if not audio:
        return None
    fc = [f"[0:{e['index']}]atrim=start={first_s:.6f}:end={end_s:.6f},asetpts=PTS-{origin_s:.6f}/TB[a{i}]"
          for i, e in enumerate(audio)]
    cmd = ffbase(ts) + ["-copyts"] + seek_args + ["-i", str(src.path), "-filter_complex", ";".join(fc)]
    for i, e in enumerate(audio):
        cmd += ["-map", f"[a{i}]", f"-c:a:{i}", "aac", f"-b:a:{i}", aac_bitrate(e["channels"])]
    cmd += ["-vn", str(out)]
    run(cmd, log=log, label=label)
    return out


def mux_clip(ts: tools.Toolset, src: Source, video: Path, audio_file: Optional[Path], out: Path, log: RunLog,
             label: str) -> None:
    cmd = ffbase(ts) + ["-copyts", "-i", str(video)]
    if audio_file:
        cmd += ["-i", str(audio_file)]
    cmd += ["-map", "0:v:0"]
    if audio_file:
        for i, e in enumerate(audio_entries(src)):
            cmd += ["-map", f"1:a:{i}", f"-disposition:a:{i}", "default" if e["default"] else "0"]
            if e.get("language"):
                cmd += [f"-metadata:s:a:{i}", f"language={e['language']}"]
    cmd += ["-c", "copy", "-bsf:v", h264_metadata_bsf(src.color, src.sar),
            "-movflags", "+faststart", "-f", "mp4", str(out)]
    run(cmd, log=log, label=label)


def preview(ts: tools.Toolset, input_path: Path, start: str, duration: float, out_dir: Path, s: Settings,
            preview_sharpen: float, log: RunLog) -> dict:
    t_start = time.monotonic()
    src_path = check_input(input_path)
    start_s = parse_time(start)
    if duration <= 0 or start_s < 0:
        raise UsageError("--start must be >= 0 and --duration > 0")
    if not (0 < preview_sharpen <= SHARPEN_MAX):
        raise UsageError(f"--preview-sharpen must be in (0, {SHARPEN_MAX}]")
    env = check_settings(ts, s)
    out_dir = Path(out_dir).expanduser().resolve()
    src = analyze_source(ts, src_path, decode_scan=False, log=log, scale=s.scale)
    # Whole-file timestamp defects only matter for a preview if they fall inside the interval; the
    # interval's own decoded timestamps are checked for strict monotonicity below.
    errors = [i for i in src.issues if i["severity"] == "error"
              and not (i["code"] == "interlaced" and s.allow_interlaced)
              and i["code"] != "timestamp-defect"]
    if errors:
        raise UsageError("Source cannot be processed:\n  - " + "\n  - ".join(e["message"] for e in errors))
    tb = src.time_base
    S = src.global_start_s + start_s
    E = S + duration
    if src.timing.last_pts_s is None or S >= src.timing.last_pts_s:
        raise UsageError(f"--start {start} is beyond the last video frame ({(src.timing.last_pts_s or 0) - src.global_start_s:.3f} s)")
    W, H = src.width * s.scale, src.height * s.scale
    stem = src_path.stem
    tag = f"{stem}_{start_s:09.3f}s_{duration:g}s_x{s.scale}".replace(":", "-")
    names = {
        "baseline": out_dir / f"{tag}_1_baseline_lanczos.mp4",
        "ai": out_dir / f"{tag}_2_ai_{s.model}.mp4",
        "ai_sharp": out_dir / f"{tag}_3_ai_sharpen{preview_sharpen:g}.mp4",
        "side_by_side": out_dir / f"{tag}_compare_side_by_side.mp4",
        "crop": out_dir / f"{tag}_compare_crop_1to1.mp4",
        "report": out_dir / f"{tag}_preview_report.json",
        "log": out_dir / f"{tag}_preview.log",
    }
    for key in ("baseline", "ai", "ai_sharp", "side_by_side", "crop"):
        if same_file(src_path, names[key]):
            raise UsageError("Preview output would overwrite the source")

    # Display-size comparison layout: square pixels at the source display aspect, three panels wide.
    disp_w = even(W * float(src.sar))
    disp_h = H
    factor = min(1.0, 3840 / (3 * disp_w))
    pw, ph = even(disp_w * factor), even(disp_h * factor)
    crop_w, crop_h = min(W, 1280) // 2 * 2, min(H, 720) // 2 * 2
    seek_s = max(0.0, start_s - 10.0)

    plan_text = [
        f"source: {src_path}",
        f"interval: {start_s:.3f}s + {duration:g}s (absolute source time {S:.3f}-{E:.3f}s)",
        f"target size: {W}x{H}, SAR {probe.ratio_str(src.sar)}; comparison panels {pw}x{ph}"
        + (" (downscaled to fit 3840 px)" if factor < 1 else ""),
        "1. decode exact interval to lossless FFV1 (frame list from decoded timestamps)",
        f"2. Video2X {s.model} x{s.scale} on GPU {s.gpu} -> lossless FFV1",
        "3. baseline: Lanczos resize of the same lossless clip, no sharpening",
        f"4. encode baseline / AI / AI+unsharp({preview_sharpen:g}) with libx264 crf {s.crf} preset {s.preset}",
        "5. restore source frame timestamps, add audio cut to the same interval, side-by-side comparison",
    ]
    if s.dry_run:
        return {"dry_run": True, "plan": plan_text, "outputs": {k: str(v) for k, v in names.items()},
                "issues": src.issues, "source": source_report(src)}

    out_dir.mkdir(parents=True, exist_ok=True)
    work = make_workdir(s.temp_dir, out_dir, f"preview-{stem[:40]}")
    log.path = work / "run.log"
    log.note(f"Working directory: {work}")
    report: dict = {"kind": "preview", "status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "settings": {**asdict(s), "temp_dir": str(work), "preview_sharpen": preview_sharpen,
                                 "start_s": start_s, "duration_s": duration}}
    try:
        report["tools"] = tool_versions(ts, s)
        report["source_identity"] = source_identity(src_path, hash_file=False)
        report["source"] = source_report(src)
        need = W * H * 1.5 * max(duration, 1) * 60 * 0.6 * 1.5
        if free_bytes(work) < need:
            raise ToolError(f"Not enough free space in {work}: need about {gib(need)}")

        # 1. Exact frame list for the interval from decoded timestamps.
        win = probe.scan_frames(ts.ffprobe, src_path, src.video_index, decode=True, time_base=tb,
                                read_intervals=f"{max(S - 15, 0):.3f}%{E + 2:.3f}", show_progress=False)
        in_window = [t for t in win.ts if S - 1e-9 <= t * float(tb) < E - 1e-9]
        frame_ts = sorted(set(in_window))
        if not frame_ts:
            raise ToolError("No video frames fall inside the requested interval")
        if len(frame_ts) != len(in_window) or in_window != frame_ts:
            raise UsageError("Frame timestamps inside the requested interval are not strictly increasing; "
                             "choose another interval")
        report["source_timestamp_defects"] = [i for i in src.issues if i["code"] == "timestamp-defect"]
        n = len(frame_ts)
        first_s = frame_ts[0] * float(tb)
        all_after = [t for t in win.ts if t > frame_ts[-1]]
        last_dur_ticks = (min(all_after) - frame_ts[-1]) if all_after else (src.timing.last_frame_duration_ticks or 1)
        end_s = (frame_ts[-1] + last_dur_ticks) * float(tb)
        report["interval"] = {"frames": n, "first_frame_pts_s": first_s, "end_s": end_s,
                              "first_frame_relative_s": first_s - src.global_start_s}

        clip = work / "clip_source_ffv1.mkv"
        seek_args, report["interval_verification"] = extract_interval(ts, src_path, src.video_index, frame_ts, tb,
                                                                      seek_s, clip, log)

        # 2. AI inference (once), lossless output.
        ai_raw = work / "ai_ffv1.mkv"
        log.note(f"Running Video2X on {n} frames (GPU {s.gpu})")
        v2x = run_video2x(ts, clip, ai_raw, s, "ffv1", log=log, expected_frames=n, label="Video2X inference")
        ai_frames = count_frames(ts, ai_raw)
        if ai_frames != n:
            raise ToolError(f"Video2X produced {ai_frames} frames from {n}")
        report["inference"] = {**v2x, "frames": n, "input_size": [src.width, src.height], "output_size": [W, H]}

        # 3-4. Encode the three variants identically. The clip timeline starts at its first frame; when the
        # constant-rate grid puts that frame up to one tick early, the origin moves one tick back so no
        # frame gets a negative time (MP4 edit lists treat those as pre-roll and players drop them).
        origin_ticks = frame_ts[0]
        target_pts, out_tb, last_dur, timing_mode = target_timeline(src, frame_ts, origin_ticks, s)
        if target_pts[0] < 0:
            origin_ticks -= 1
            target_pts, out_tb, last_dur, timing_mode = target_timeline(src, frame_ts, origin_ticks, s)
        origin_s = origin_ticks * float(tb)
        variants = {
            "baseline": (clip, f"scale={W}:{H}:flags=lanczos,setsar={src.sar.numerator}/{src.sar.denominator},format={PIX_FMT}"),
            "ai": (ai_raw, f"setsar={src.sar.numerator}/{src.sar.denominator},format={PIX_FMT}"),
            "ai_sharp": (ai_raw, f"{unsharp_filter(preview_sharpen)},setsar={src.sar.numerator}/{src.sar.denominator},format={PIX_FMT}"),
        }
        encoded = {}
        enc_stats = {}
        for key, (inp, vf) in variants.items():
            raw = work / f"{key}_raw.mp4"
            r = run(ffbase(ts) + ["-i", str(inp), "-map", "0:v:0", "-vf", vf, "-fps_mode", "passthrough"]
                    + x264_args(s, src.color) + ["-an", str(raw)],
                    log=log, label=f"encode {key}", progress=ffmpeg_frame_progress, progress_total=n)
            enc_stats[key] = {"elapsed_s": r.elapsed_s, "fps": n / r.elapsed_s if r.elapsed_s else None}
            timed = work / f"{key}_timed.mp4"
            avtools.retime_remux(raw, timed, target_pts, out_tb, last_dur)
            encoded[key] = timed

        # Audio cut to exactly the video interval, timestamps relative to the first frame.
        audio = audio_entries(src)
        audio_file = cut_audio(ts, src, seek_args, first_s, end_s, origin_s, work / "audio.mka", log,
                               label="cut preview audio")

        def mux(video: Path, out: Path) -> None:
            tmp = out.with_name(out.stem + ".partial.mp4")
            mux_clip(ts, src, video, audio_file, tmp, log, label=f"mux {out.name}")
            os.replace(tmp, out)

        for key in ("baseline", "ai", "ai_sharp"):
            mux(encoded[key], names[key])

        # 5. Comparisons built from the lossless sources (one encode for the comparison itself).
        font = tools.find_label_font()
        if font:
            shutil.copyfile(font, work / "label.ttf")
        labels = ["Baseline - Lanczos (no sharpening)", f"Real-ESRGAN {s.model} x{s.scale}",
                  f"Real-ESRGAN + unsharp {preview_sharpen:g}"]

        def label_filter(text: str, height: int) -> str:
            if not font:
                return ""
            size = max(14, height // 26)
            return (f",drawtext=fontfile=label.ttf:text='{text}':x=12:y=12:fontsize={size}:fontcolor=white"
                    f":box=1:boxcolor=black@0.6:boxborderw=6")

        def comparison(out: Path, mode: str) -> None:
            sar = f"{src.sar.numerator}/{src.sar.denominator}"
            # Both inputs get identical index-based timestamps so hstack pairs frame i with frame i
            # (the source clip carries absolute times, Video2X output starts at zero). Real timing is
            # restored by retime_remux afterwards.
            sync = "setpts=N/(24*TB)"
            if mode == "display":
                size = f"scale={pw}:{ph}:flags=lanczos,setsar=1"
                fc = (f"[0:v]{sync},scale={pw}:{ph}:flags=lanczos,setsar=1{label_filter(labels[0], ph)}[p0];"
                      f"[1:v]{sync},split[x1][x2];"
                      f"[x1]{size}{label_filter(labels[1], ph)}[p1];"
                      f"[x2]{unsharp_filter(preview_sharpen)},{size}{label_filter(labels[2], ph)}[p2];")
            else:
                crop = f"crop={crop_w}:{crop_h}:(iw-{crop_w})/2:(ih-{crop_h})/2"
                fc = (f"[0:v]{sync},scale={W}:{H}:flags=lanczos,{crop},setsar={sar}{label_filter(labels[0], crop_h)}[p0];"
                      f"[1:v]{sync},split[x1][x2];"
                      f"[x1]{crop},setsar={sar}{label_filter(labels[1], crop_h)}[p1];"
                      f"[x2]{unsharp_filter(preview_sharpen)},{crop},setsar={sar}{label_filter(labels[2], crop_h)}[p2];")
            fc += f"[p0][p1][p2]hstack=inputs=3,format={PIX_FMT}[v]"
            raw = work / f"cmp_{mode}_raw.mp4"
            run(ffbase(ts) + ["-i", str(clip), "-i", str(ai_raw), "-filter_complex", fc, "-map", "[v]",
                              "-fps_mode", "passthrough"] + x264_args(s, src.color) + ["-an", str(raw)],
                log=log, label=f"comparison ({mode})", progress=ffmpeg_frame_progress, progress_total=n, cwd=work)
            timed = work / f"cmp_{mode}_timed.mp4"
            avtools.retime_remux(raw, timed, target_pts, out_tb, last_dur)
            tmp = out.with_name(out.stem + ".partial.mp4")
            sar_out = Fraction(1) if mode == "display" else src.sar
            cmd = ffbase(ts) + ["-copyts", "-i", str(timed)]
            if audio_file:
                cmd += ["-i", str(audio_file), "-map", "0:v:0", "-map", "1:a?"]
            else:
                cmd += ["-map", "0:v:0"]
            cmd += ["-c", "copy", "-bsf:v", h264_metadata_bsf(src.color, sar_out), "-movflags", "+faststart",
                    "-f", "mp4", str(tmp)]
            run(cmd, log=log, label=f"mux {out.name}")
            os.replace(tmp, out)

        comparison(names["side_by_side"], "display")
        made_crop = factor < 1 or src.sar != 1
        if made_crop:
            comparison(names["crop"], "crop")

        # Validate the individual clips.
        window = (first_s, end_s)
        exp = Expectation(
            width=W, height=H, sar=src.sar, frame_count=n,
            pts_s=[p * float(out_tb) for p in target_pts], last_frame_duration_s=last_dur * float(out_tb),
            pts_tol_s=0.0011, color={k: src.color[k] for k in ("color_range", "color_space", "color_transfer", "color_primaries")},
            audio=audio_expectations(src, origin_s, window, force_aac=True), subtitles=0,
            video_start_s=target_pts[0] * float(out_tb),
            source_pts_s=[(t - origin_ticks) * float(tb) for t in frame_ts], source_tol_s=float(tb) + 0.0001)
        report["validation"] = {}
        for key in ("baseline", "ai", "ai_sharp"):
            report["validation"][key] = validate_output(ts.ffmpeg, ts.ffprobe, names[key], exp, log=log,
                                                        full_decode=True, label=f"validate {key}")
        all_ok = all(v["ok"] for v in report["validation"].values())

        total_frames = src.timing.frames
        fps = v2x["reported_fps"] or v2x["wall_fps"]
        est_ai = total_frames / fps if fps else None
        x264_fps = enc_stats["ai"]["fps"]
        report["speed"] = {
            "inference_fps_reported_by_video2x": v2x["reported_fps"],
            "inference_wall_fps_including_model_load": v2x["wall_fps"],
            "x264_encode_fps_at_target_size": x264_fps,
            "source_total_frames": total_frames,
            "estimated_full_run_hours_inference_only": None if est_ai is None else est_ai / 3600,
            "estimated_full_run_hours_if_encode_is_serial": None if not (est_ai and x264_fps) else (est_ai + total_frames / x264_fps) / 3600,
            "note": "ESTIMATE only, extrapolated from this preview interval; scene complexity, thermal throttling "
                    "and other GPU load change the real runtime.",
        }
        report["timing_mode"] = timing_mode
        report["outputs"] = {k: str(v) for k, v in names.items() if k not in ("report", "log")
                             and (k != "crop" or made_crop)}
        report["comparison"] = {"panel_size": [pw, ph], "downscaled_for_display": factor < 1,
                                "crop_size": [crop_w, crop_h] if made_crop else None,
                                "labels": labels, "audio": "AAC cut to the same interval" if audio else "none"}
        report["status"] = "ok" if all_ok else "validation-failed"
        report["elapsed_s"] = time.monotonic() - t_start
        return report
    except BaseException as e:
        report["status"] = "cancelled" if isinstance(e, KeyboardInterrupt) else "failed"
        report["error"] = str(e)
        raise
    finally:
        report["commands"] = [asdict(r) for r in log.records]
        write_json(names["report"], report)
        if log.path and log.path.is_file():
            shutil.copyfile(log.path, names["log"])
        if not s.keep_temp:
            shutil.rmtree(work, ignore_errors=True)
            _rmdir_if_empty(work.parent)


def _rmdir_if_empty(p: Path) -> None:
    try:
        p.rmdir()
    except OSError:
        pass


# --------------------------------------------------------------------------------------------
# Full enhancement
# --------------------------------------------------------------------------------------------

FFV1_RATIO = 0.3  # measured ~0.17 of raw yuv420p for upscaled episode-1 content; doubled for safety
OUTPUT_BPP = 0.15  # H.264 CRF 18 output bits per pixel; measured 0.12-0.13 in active ai_sharpen0.5 scenes


def report_paths(out: Path) -> dict:
    return {"ok_report": out.with_suffix(".report.json"), "ok_log": out.with_suffix(".log"),
            "failed_report": out.with_suffix(".failed.report.json"), "failed_log": out.with_suffix(".failed.log")}


def resolved_settings(s: Settings, src: Optional[Source], ts: tools.Toolset) -> dict:
    d = {"profile": s.profile, "upscaler": s.upscaler, "model": s.model if s.upscaler == "ai" else None,
         "scale": s.scale, "sharpen": s.sharpen,
         "sharpen_filter": unsharp_filter(s.sharpen) if s.sharpen > 0 else None,
         "crf": s.crf, "encoder_preset": s.preset, "x264_profile": "high", "pix_fmt": PIX_FMT,
         "gpu": s.gpu if s.upscaler == "ai" else None, "timestamp_policy": s.timestamp_policy,
         "timing": s.timing, "frame_interpolation": False, "denoise": False, "color_adjustment": False,
         "container": "mp4 +faststart"}
    if s.upscaler == "ai":
        dev = next((x for x in tools.video2x_devices(ts.video2x) if x["index"] == s.gpu), None)
        d["gpu_name"] = dev["name"] if dev else None
    if src is not None:
        d["encode_filter"] = encode_filter(s, src.sar, width=src.width * s.scale, height=src.height * s.scale,
                                           sharpen=s.sharpen)
        d["color_tags"] = {k: src.color[k] for k in ("color_range", "color_space", "color_transfer", "color_primaries")}
    return d


def enhance(ts: tools.Toolset, input_path: Path, output: Path, s: Settings, overwrite: bool, log: RunLog,
            clip: Optional[tuple[float, float]] = None, extra_report: Optional[dict] = None) -> dict:
    """Enhance a whole source, or the interval clip=(start_s, duration_s) of it, into one validated MP4."""
    t_start = time.monotonic()
    src_path = check_input(input_path)
    out = check_output(src_path, output, overwrite)
    check_settings(ts, s)
    out_dir = out.parent
    paths = report_paths(out)
    for p in paths.values():
        if same_file(src_path, p):
            raise UsageError(f"Report/log path {p} would overwrite the source")
    if clip is not None and (clip[0] < 0 or clip[1] <= 0):
        raise UsageError("--start must be >= 0 and --duration > 0")

    sw = Stopwatch()
    src = analyze_source(ts, src_path, decode_scan=not s.dry_run, log=log, scale=s.scale,
                         timestamp_policy=s.timestamp_policy, run_idet=not s.dry_run, decode_audio=not s.dry_run)
    sw.lap("inspection: metadata, decoded timestamp scan, idet, audio extents")
    errors = [i for i in src.issues if i["severity"] == "error"
              and not (i["code"] == "interlaced" and s.allow_interlaced)]
    if errors:
        raise UsageError("Source cannot be processed:\n  - " + "\n  - ".join(e["message"] for e in errors))

    tb = src.time_base
    W, H = src.width * s.scale, src.height * s.scale
    all_ts = src.scan.ts
    orig_ts = src.original_scan.ts if src.original_scan else all_ts
    if clip is not None:
        S = src.global_start_s + clip[0]
        E = S + clip[1]
        idx = [i for i, t in enumerate(all_ts) if S - 1e-9 <= t * float(tb) < E - 1e-9]
        if not idx:
            raise UsageError(f"No video frames between {clip[0]:.3f}s and {clip[0] + clip[1]:.3f}s "
                             f"(video ends at {(src.timing.end_s or 0) - src.global_start_s:.3f}s)")
        frame_ts = all_ts[idx[0]:idx[-1] + 1]
        orig_frame_ts = orig_ts[idx[0]:idx[-1] + 1]
    else:
        idx = list(range(len(all_ts)))
        frame_ts, orig_frame_ts = all_ts, orig_ts
    n = len(frame_ts)
    repaired_in_range = bool(src.timing_repair and idx[-1] >= src.timing_repair["suffix_start_index"])
    span = (frame_ts[-1] - frame_ts[0]) * float(tb) + (src.timing.last_frame_duration_ticks or 0) * float(tb)
    fps_nominal = n / span if span > 0 else 25.0
    est_h264 = W * H * fps_nominal * span * OUTPUT_BPP / 8 + 1e6
    est_ffv1 = W * H * 1.5 * n * FFV1_RATIO
    video_bytes = (probe._int(src.video.get("bit_rate")) or 0) * span / 8 or src_path.stat().st_size
    pad_needed = clip is None and s.upscaler == "ai" and src.decoder_delay > 0
    ffv1_needed = s.upscaler == "ai" and s.sharpen > 0
    streamed = ffv1_needed and s.stream_intermediate and winpipe.available()
    temp_need = 2 * est_h264 + (video_bytes if pad_needed else 0) + (est_ffv1 if ffv1_needed and not streamed else 0) \
        + (src.width * src.height * 1.5 * n * 0.6 if clip is not None else 0)
    out_need = est_h264 * 1.1 + (src_path.stat().st_size * 0.2 if clip is None else 1e6)
    work_parent = Path(s.temp_dir).expanduser().resolve() if s.temp_dir else out_dir / ".animenhance-tmp"

    route = []
    if clip is not None:
        route.append(f"decode frames {idx[0]}-{idx[-1]} ({n} frames, {clip[0]:.3f}s + {clip[1]:g}s) to a lossless FFV1 clip")
    elif pad_needed:
        route.append(f"stream-copy video to Matroska + {src.decoder_delay} IDR copies (Video2X end-of-stream workaround)")
    if s.upscaler == "ai":
        if streamed:
            route.append(f"Video2X {s.model} x{s.scale} on GPU {s.gpu} -> lossless FFV1 streamed through a named pipe "
                         f"into FFmpeg {encode_filter(s, src.sar, width=W, height=H, sharpen=s.sharpen)} -> libx264 "
                         f"crf {s.crf} preset {s.preset} (the only lossy encode; runs concurrently, no intermediate file)")
        elif ffv1_needed:
            route.append(f"Video2X {s.model} x{s.scale} on GPU {s.gpu} -> lossless FFV1 (~{gib(est_ffv1)})")
            route.append(f"FFmpeg {encode_filter(s, src.sar, width=W, height=H, sharpen=s.sharpen)} -> libx264 "
                         f"crf {s.crf} preset {s.preset} (the only lossy encode)")
        else:
            route.append(f"Video2X {s.model} x{s.scale} on GPU {s.gpu} -> libx264 crf {s.crf} preset {s.preset} "
                         f"(the only lossy encode, streaming)")
    else:
        route.append(f"FFmpeg {encode_filter(s, src.sar, width=W, height=H, sharpen=s.sharpen)} -> libx264 "
                     f"crf {s.crf} preset {s.preset} (the only lossy encode)")
    route.append("restore source frame timestamps on the encoded video (packet copy)"
                 + (" using the repaired terminal timestamps" if repaired_in_range else ""))
    route.append("mux: video stream copy + " + ("audio cut to the interval as AAC" if clip is not None
                                                 else "source audio copied/converted, text subtitles") + ", +faststart")
    route.append("validate (dimensions, aspect, frame count, per-frame timing, audio offsets/ends, full decode), "
                 "then publish under the final name")
    route.append(f"disk: temp ~{gib(temp_need)} in {work_parent} (free {gib(free_bytes(work_parent))}); "
                 f"output ~{gib(out_need)} in {out_dir} (free {gib(free_bytes(out_dir))})")
    summary = {
        "frames": n, "source_frames": len(all_ts),
        "source": f"{src.width}x{src.height} SAR {probe.ratio_str(src.sar)} "
                  f"DAR {probe.ratio_str(Fraction(src.width, src.height) * src.sar)}",
        "output": f"{W}x{H} SAR {probe.ratio_str(src.sar)} DAR {probe.ratio_str(Fraction(W, H) * src.sar)}",
        "timing": src.timing.classification, "timestamp_repair": src.timing_repair,
    }
    placeholder = work_parent / "<run-id>"
    up_in = (placeholder / "clip_source_ffv1.mkv" if clip is not None
             else placeholder / "video_padded.mkv" if pad_needed else src_path)
    commands = []
    vf = encode_filter(s, src.sar, width=W, height=H, sharpen=s.sharpen)
    if s.upscaler == "ai" and streamed:
        pipe_name = Path(r"\\.\pipe\animenhance-<id>.mkv")
        commands.append(format_cmd(video2x_cmd(ts, up_in, pipe_name, s, "ffv1")))
        commands.append(format_cmd(ffbase(ts) + ["-f", "matroska", "-i", "pipe:0", "-map", "0:v:0", "-vf", vf,
                                                 "-fps_mode", "passthrough"] + x264_args(s, src.color)
                                   + ["-an", str(placeholder / "encoded_h264.mp4")]))
    elif s.upscaler == "ai":
        ai_out = placeholder / ("ai_ffv1.mkv" if ffv1_needed else "ai_h264.mp4")
        commands.append(format_cmd(video2x_cmd(ts, up_in, ai_out, s, "ffv1" if ffv1_needed else "libx264")))
        if ffv1_needed:
            commands.append(format_cmd(ffbase(ts) + ["-i", str(ai_out), "-map", "0:v:0", "-vf", vf, "-fps_mode",
                                                     "passthrough"] + x264_args(s, src.color)
                                       + ["-an", str(placeholder / "encoded_h264.mp4")]))
    else:
        commands.append(format_cmd(ffbase(ts) + ["-i", str(up_in), "-vf", vf, "-fps_mode", "passthrough"]
                                   + x264_args(s, src.color) + ["-an", str(placeholder / "encoded_h264.mp4")]))
    if clip is None:
        commands.append(format_cmd(final_mux_cmd(ts, src, placeholder / "video_timed.mp4",
                                                 out.with_name(f".{out.stem}.partial-<run-id>.mp4"),
                                                 round(src.global_start_s / float(tb)) * float(tb))))
    if s.dry_run:
        return {"dry_run": True, "plan": route, "commands": commands, "summary": summary, "issues": src.issues,
                "settings": resolved_settings(s, src, ts), "source": source_report(src), "output": str(out),
                "scan_note": "dry run reads packet timestamps (sorted into presentation order); the real run "
                             "decodes every frame"}

    if free_bytes(work_parent) < temp_need * 1.1 + 512 * 1024 ** 2:
        raise UsageError(f"Not enough free space for intermediates in {work_parent}: need ~{gib(temp_need)}, "
                         f"free {gib(free_bytes(work_parent))}. Use --temp-dir on a larger drive.")
    if free_bytes(out_dir) < out_need + 256 * 1024 ** 2:
        raise UsageError(f"Not enough free space for the output in {out_dir}: need ~{gib(out_need)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    work = make_workdir(s.temp_dir, out_dir, f"enhance-{src_path.stem[:40]}")
    log.path = work / "run.log"
    log.note(f"Working directory: {work}")
    partial = out.with_name(f".{out.stem}.partial-{work.name[-6:]}.mp4")
    report: dict = {"kind": "enhance-clip" if clip else "enhance", "status": "running",
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "settings": {**asdict(s), "temp_dir": str(work), "output": str(out),
                                 "clip": None if clip is None else {"start_s": clip[0], "duration_s": clip[1]}},
                    "resolved_settings": resolved_settings(s, src, ts), "plan": route, **(extra_report or {})}
    published = False
    guard = disk_guard(work, out_dir)
    try:
        report["stage_times_s"] = sw.laps
        report["tools"] = tool_versions(ts, s)
        log.note("Hashing source for the report")
        report["source_identity"] = source_identity(src_path)
        report["source"] = source_report(src)
        report["timestamp_repair"] = src.timing_repair
        sw.lap("planning, tool versions, source hash")
        if src.timing_repair:
            report["timestamp_changes"] = [{"frame_index": f["index"], "original_pts_s": f["original_pts_s"],
                                            "repaired_pts_s": f["repaired_pts_s"], "changed": f["changed"]}
                                           for f in src.timing_repair["frames"]]

        # Input to the upscaler.
        seek_args: list[str] = []
        if clip is not None:
            clip_src = work / "clip_source_ffv1.mkv"
            seek_s = max(0.0, clip[0] - 10.0)
            seek_args, report["interval_verification"] = extract_interval(
                ts, src_path, src.video_index, orig_frame_ts, tb, seek_s, clip_src, log)
            up_input, up_stream = clip_src, None
        elif pad_needed:
            padded = work / "video_padded.mkv"
            log.note(f"Stream-copying video with {src.decoder_delay} IDR copies appended")
            report["padding"] = avtools.pad_remux(src_path, padded, src.video_index, src.decoder_delay)
            up_input, up_stream = padded, None
        else:
            up_input, up_stream = src_path, src.video_index
        align_ref, align_stream = (up_input, None) if clip is not None else (src_path, src.video_index)
        sw.lap("preparation: lossless clip extraction + verification, or padded stream copy")

        v2x = None
        encoded = work / "encoded_h264.mp4"
        if s.upscaler == "ai":
            codec = "ffv1" if ffv1_needed else "libx264"
            ai_out = work / ("ai_ffv1.mkv" if ffv1_needed else "ai_h264.mp4")
            log.note(f"Running Video2X on {n} frames: {s.model} x{s.scale} on GPU {s.gpu}")
            if streamed:
                vf = encode_filter(s, src.sar, width=W, height=H, sharpen=s.sharpen)
                v2x, enc = run_video2x_streamed(ts, up_input, encoded, s, vf, src.color, log=log, expected_frames=n,
                                                monitor=guard)
                sw.lap("AI inference + concurrent sharpen/libx264 encode (Video2X streamed into FFmpeg)")
                got = count_frames(ts, encoded)
                report["inference"] = {**v2x, "frames_in_output": got, "frames_expected": n}
                report["encode"] = {**enc, "filter": vf}
                if got != n:
                    raise ToolError(f"Streamed encode has {got} frames but {n} were expected. Refusing to publish a "
                                    f"result with dropped or duplicated frames.")
                if pad_needed and not s.keep_temp:
                    up_input.unlink(missing_ok=True)
                check_alignment(report, align_ref, align_stream, encoded, src)
                sw.lap("frame count + head/tail alignment check")
                ai_out = None
            else:
                v2x = run_video2x(ts, up_input, ai_out, s, codec, log=log, expected_frames=n, label="Video2X")
                sw.lap("AI inference: Video2X process (decode, Real-ESRGAN, " + ("FFV1" if ffv1_needed else "x264") + " write)")
        if s.upscaler == "ai" and not streamed:
            got = count_frames(ts, ai_out)
            report["inference"] = {**v2x, "frames_in_output": got, "frames_expected": n}
            if got != n:
                raise ToolError(f"Video2X output has {got} frames but {n} were expected. Refusing to publish a "
                                f"result with dropped or duplicated frames.")
            if pad_needed and not s.keep_temp:
                up_input.unlink(missing_ok=True)
            check_alignment(report, align_ref, align_stream, ai_out, src)
            sw.lap("frame count + head/tail alignment check")
            if ffv1_needed:
                vf = encode_filter(s, src.sar, width=W, height=H, sharpen=s.sharpen)
                r = run(ffbase(ts) + ["-i", str(ai_out), "-map", "0:v:0", "-vf", vf, "-fps_mode", "passthrough"]
                        + x264_args(s, src.color) + ["-an", str(encoded)],
                        log=log, label="sharpen + encode", progress=ffmpeg_frame_progress, progress_total=n,
                        monitor=guard)
                report["encode"] = {"elapsed_s": r.elapsed_s, "filter": vf, "fps": n / r.elapsed_s if r.elapsed_s else None}
                if not s.keep_temp:
                    ai_out.unlink(missing_ok=True)
                sw.lap("sharpen + libx264 encode (FFmpeg)")
            else:
                encoded = ai_out
        elif s.upscaler != "ai":
            vf = encode_filter(s, src.sar, width=W, height=H, sharpen=s.sharpen)
            m = ["-map", f"0:{up_stream}"] if up_stream is not None else ["-map", "0:v:0"]
            r = run(ffbase(ts) + ["-i", str(up_input)] + m + ["-vf", vf, "-fps_mode", "passthrough"]
                    + x264_args(s, src.color) + ["-an", "-sn", "-dn", str(encoded)],
                    log=log, label="lanczos + encode", progress=ffmpeg_frame_progress, progress_total=n, monitor=guard)
            report["encode"] = {"elapsed_s": r.elapsed_s, "filter": vf, "fps": n / r.elapsed_s if r.elapsed_s else None}
            got = count_frames(ts, encoded)
            if got != n:
                raise ToolError(f"Lanczos encode has {got} frames but {n} were expected")
            check_alignment(report, align_ref, align_stream, encoded, src)
            sw.lap("lanczos + libx264 encode (FFmpeg) + alignment check")

        # Timeline: whole file relative to the earliest stream, or the clip relative to its first frame.
        if clip is None:
            origin_ticks = round(src.global_start_s / float(tb))
        else:
            origin_ticks = frame_ts[0]
        target_pts, out_tb, last_dur, timing_mode = target_timeline(src, frame_ts, origin_ticks, s)
        if target_pts[0] < 0:
            origin_ticks -= 1
            target_pts, out_tb, last_dur, timing_mode = target_timeline(src, frame_ts, origin_ticks, s)
        origin_s = origin_ticks * float(tb)
        timed = work / "video_timed.mp4"
        log.note(f"Restoring frame timestamps ({timing_mode})")
        report["retime"] = {**avtools.retime_remux(encoded, timed, target_pts, out_tb, last_dur), "mode": timing_mode}
        if not s.keep_temp and encoded.exists():
            encoded.unlink()
        sw.lap("timestamp restoration (packet copy)")

        if clip is None:
            run(final_mux_cmd(ts, src, timed, partial, origin_s), log=log, label="final mux",
                progress=ffmpeg_frame_progress, progress_total=n)
            audio_exp = audio_expectations(src, origin_s)
            subs = len(sub_entries(src))
            handling = src.plan
        else:
            first_s = frame_ts[0] * float(tb)
            end_s = (frame_ts[-1] + last_dur * out_tb / tb) * float(tb)
            audio_file = cut_audio(ts, src, seek_args, first_s, float(end_s), origin_s, work / "audio.mka", log,
                                   label="cut clip audio")
            mux_clip(ts, src, timed, audio_file, partial, log, label="mux clip")
            audio_exp = audio_expectations(src, origin_s, (first_s, float(end_s)), force_aac=True)
            subs = 0
            handling = [({**e, "action": "convert", "target": "aac", "detail": "clip: cut to the interval and "
                                                                           "encoded as AAC (sample-accurate cut)"}
                         if e["type"] == "audio" and e["action"] != "exclude" else
                         ({**e, "action": "exclude", "detail": "clip mode: subtitles are not cut"}
                          if e["type"] == "subtitle" else e)) for e in src.plan]
            report["interval"] = {"first_frame_index": idx[0], "last_frame_index": idx[-1], "frames": n,
                                  "first_frame_pts_s": first_s, "end_s": float(end_s),
                                  "includes_repaired_frames": repaired_in_range}

        exp = Expectation(
            width=W, height=H, sar=src.sar, frame_count=n, pts_s=[p * float(out_tb) for p in target_pts],
            last_frame_duration_s=last_dur * float(out_tb), pts_tol_s=float(tb) + 0.0001,
            color={k: src.color[k] for k in ("color_range", "color_space", "color_transfer", "color_primaries")},
            audio=audio_exp, subtitles=subs, video_start_s=target_pts[0] * float(out_tb),
            source_pts_s=[(t - origin_ticks) * float(tb) for t in frame_ts], source_tol_s=float(tb) + 0.0001)
        sw.lap("mux (plus audio cut in clip mode)")
        log.note("Validating output" + (" against the repaired timeline" if src.timing_repair else ""))
        val = validate_output(ts.ffmpeg, ts.ffprobe, partial, exp, log=log, full_decode=s.full_decode_check)
        sw.lap("validation: probes, per-frame timing, audio extents, full decode")
        report["validation"] = val
        report["validated_against"] = "repaired timeline" if src.timing_repair else "source timeline"
        report["stream_handling"] = handling
        report["timing_mode"] = timing_mode
        if not val["ok"]:
            failed = [c for c in val["checks"] if not c["ok"]]
            kept = out.with_name(f"{out.stem}.FAILED-VALIDATION.mp4")
            os.replace(partial, kept)
            report["rejected_output"] = str(kept)
            raise ToolError("Output failed validation; it was kept as " + str(kept) + " for inspection:\n  - "
                            + "\n  - ".join(f"{c['check']}: {c['detail']}" for c in failed))
        os.replace(partial, out)
        published = True
        report["output_path"] = str(out)
        report["status"] = "ok"
        elapsed = time.monotonic() - t_start
        report["elapsed_s"] = elapsed
        report["speed"] = {"frames": n, "overall_fps": n / elapsed if elapsed else None,
                           "video2x_fps": v2x["reported_fps"] if v2x else None,
                           "video2x_elapsed_s": v2x["elapsed_s"] if v2x else None}
        return report
    except BaseException as e:
        report["status"] = "cancelled" if isinstance(e, KeyboardInterrupt) else "failed"
        report["error"] = str(e) or type(e).__name__
        raise
    finally:
        if not published and partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass
        report["commands"] = [asdict(r) for r in log.records]
        try:
            if published:
                write_json(paths["ok_report"], report)
                if log.path and log.path.is_file():
                    shutil.copyfile(log.path, paths["ok_log"])
                for stale in (paths["failed_report"], paths["failed_log"]):
                    stale.unlink(missing_ok=True)
            else:
                write_json(paths["failed_report"], report)
                if log.path and log.path.is_file():
                    shutil.copyfile(log.path, paths["failed_log"])
        except OSError:
            pass
        if not s.keep_temp:
            shutil.rmtree(work, ignore_errors=True)
            _rmdir_if_empty(work.parent)


def check_alignment(report: dict, ref: Path, ref_stream: Optional[int], produced: Path, src: Source) -> None:
    """Head/tail image comparison between the frames fed to the upscaler and the frames it produced."""
    size = (max(32, src.width // 4), max(32, src.height // 4))
    align = {}
    for which in ("head", "tail"):
        a = avtools.gray_frames(ref, which=which, count=16, size=size, stream_index=ref_stream)
        b = avtools.gray_frames(produced, which=which, count=16, size=size)
        align[which] = avtools.alignment_check(a, b)
    report["frame_alignment"] = align
    bad = [k for k, v in align.items() if v["result"] == "misaligned"]
    if bad:
        raise ToolError(f"Processed frames are misaligned with the source at the {', '.join(bad)} "
                        f"(details: {align}). Not publishing.")
