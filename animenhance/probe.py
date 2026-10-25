"""Source inspection with FFprobe: streams, frame timing, interlacing and stream extents."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional

from .procs import Progress, ToolError, capture, stream_stdout_lines

STANDARD_RATES = [
    Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30),
    Fraction(48), Fraction(50), Fraction(60000, 1001), Fraction(60), Fraction(12), Fraction(15),
    Fraction(12000, 1001), Fraction(15000, 1001), Fraction(20), Fraction(10), Fraction(8), Fraction(6),
]

UNKNOWN_COLOR = {None, "", "unknown", "reserved", "unspecified"}


def parse_ratio(text: Optional[str]) -> Optional[Fraction]:
    if not text or text in ("N/A", "0/0", "0:1", "0/1"):
        return None
    m = re.match(r"^\s*(-?\d+)\s*[:/]\s*(\d+)\s*$", str(text))
    if m:
        den = int(m.group(2))
        return Fraction(int(m.group(1)), den) if den else None
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def ratio_str(fr: Optional[Fraction]) -> Optional[str]:
    return None if fr is None else f"{fr.numerator}/{fr.denominator}"


def ffprobe_json(ffprobe: Path, path: Path) -> dict:
    cp = capture([ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                  "-show_chapters", "-show_programs", str(path)], timeout=300)
    if cp.returncode != 0:
        raise ToolError(f"ffprobe could not read {path}: {cp.stderr.strip()[-800:]}")
    data = json.loads(cp.stdout or "{}")
    if not data.get("streams"):
        raise ToolError(f"ffprobe found no streams in {path}")
    return data


def _num(v) -> Optional[float]:
    try:
        if v in (None, "N/A", ""):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v) -> Optional[int]:
    try:
        if v in (None, "N/A", ""):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def first_video_stream(info: dict) -> dict:
    vids = [s for s in info["streams"] if s.get("codec_type") == "video"
            and not s.get("disposition", {}).get("attached_pic")]
    if not vids:
        raise ToolError("Source has no video stream.")
    return vids[0]


# --------------------------------------------------------------------------------------------
# Frame timestamp scanning
# --------------------------------------------------------------------------------------------

@dataclass
class FrameScan:
    """Presentation timestamps of the video frames, in stream time-base ticks, display order."""

    source: str  # "decoded-frames" or "packets"
    time_base: Fraction
    ts: list[int]
    durations: list[Optional[int]]
    interlaced_flags: int = 0
    tff_flags: int = 0
    missing_ts: int = 0


def scan_frames(ffprobe: Path, path: Path, stream_index: int, *, decode: bool, time_base: Fraction,
                expected_duration_s: Optional[float] = None, read_intervals: Optional[str] = None,
                label: str = "scanning frame timestamps", stderr_path: Optional[Path] = None,
                show_progress: bool = True) -> FrameScan:
    """Collect per-frame timestamps.

    decode=True decodes every frame (authoritative: these are exactly the frames a decoder emits).
    decode=False reads packets only (fast) and sorts their PTS into display order.
    """
    if decode:
        # The frame's own PTS, not best_effort_timestamp: the latter silently substitutes DTS-derived
        # guesses when PTS values are faulty, which would hide exactly the defects checked for here.
        entries = "frame=pts,duration,interlaced_frame,top_field_first"
        kind = "frame"
    else:
        entries = "packet=pts,dts,duration,flags"
        kind = "packet"
    cmd = [ffprobe, "-v", "error", "-select_streams", str(stream_index), "-show_entries", entries,
           "-of", "csv=p=0:nk=0"]
    if read_intervals:
        cmd += ["-read_intervals", read_intervals]
    cmd.append(str(path))
    ts: list[int] = []
    durs: list[Optional[int]] = []
    inter = tff = missing = 0
    prog = Progress(label, expected_duration_s, "s of media") if show_progress else None
    tb = float(time_base)

    def on_line(line: str) -> None:
        nonlocal inter, tff, missing
        if not line:
            return
        fields = dict(kv.split("=", 1) for kv in line.split(",") if "=" in kv)
        if kind == "frame":
            t = _int(fields.get("pts"))
            if fields.get("interlaced_frame") == "1":
                inter += 1
                if fields.get("top_field_first") == "1":
                    tff += 1
        else:
            t = _int(fields.get("pts"))
            if t is None:
                t = _int(fields.get("dts"))
        if t is None:
            missing += 1
            return
        ts.append(t)
        durs.append(_int(fields.get("duration")))
        if prog and len(ts) % 200 == 0:
            prog.update(t * tb)

    try:
        rc = stream_stdout_lines(cmd, on_line, stderr_path=stderr_path)
    finally:
        if prog:
            prog.finish()
    if rc != 0:
        raise ToolError(f"ffprobe frame scan failed for {path} (exit {rc})")
    if kind == "packet":
        order = sorted(range(len(ts)), key=lambda i: ts[i])
        ts = [ts[i] for i in order]
        durs = [durs[i] for i in order]
    return FrameScan("decoded-frames" if decode else "packets", time_base, ts, durs, inter, tff, missing)


# --------------------------------------------------------------------------------------------
# Timestamp defects and the terminal-duplicate repair
# --------------------------------------------------------------------------------------------

class TimingDefect(ToolError):
    """Frame timestamps cannot be represented faithfully and are not covered by the narrow repair."""


TERMINAL_WINDOW = 24  # valid intervals before the defect used to infer the nominal frame interval


def _fmt_s(ticks: int, tb: Fraction) -> str:
    return f"{float(ticks * tb):.6f} s"


def describe_defects(scan: FrameScan, limit: int = 8) -> str:
    ts, tb = scan.ts, scan.time_base
    bad = [i for i in range(1, len(ts)) if ts[i] <= ts[i - 1]]
    lines = []
    if scan.missing_ts:
        lines.append(f"{scan.missing_ts} decoded frames have no timestamp")
    for i in bad[:limit]:
        kind = "duplicate" if ts[i] == ts[i - 1] else "backwards"
        lo, hi = max(0, i - 2), min(len(ts), i + 3)
        around = ", ".join(f"#{j}={_fmt_s(ts[j], tb)}" for j in range(lo, hi))
        lines.append(f"frame #{i} of {len(ts)}: {kind} timestamp ({around})")
    if len(bad) > limit:
        lines.append(f"... {len(bad) - limit} more")
    return "; ".join(lines)


# Limits of the terminal repair, set from the defects observed in the Avatar episodes (61 files):
# suffixes of 1-3 frames, each frame stamped 0-2 slots too early, all stamps on the regular grid.
MAX_TERMINAL_SUFFIX = 3
MAX_SHIFT_SLOTS = 2

# The only suffix signatures accepted: slot offsets of the suffix stamps relative to the anchor, exactly as
# observed in decoded presentation order across the 61 episodes (scripts/diagnose_tails.py). Anything else,
# even if it would fit the grid, is rejected as unrecognised.
RECOGNISED_SUFFIXES = {
    (0,): "final frame duplicates the anchor",
    (0, 1, 3): "duplicate of the anchor, then +1, then the final frame already on slot +3",
    (0, 2): "duplicate of the anchor, then the final frame already on slot +2",
    (-1,): "final frame one slot before the anchor",
    (-1, 0, 1): "three frames stamped one slot early (-1, 0, +1)",
    (-1, 0): "two frames stamped one slot early (-1, 0)",
}


def _fit_rate(window: list[int], tb: Fraction, nominal: list[Optional[Fraction]]) -> Optional[tuple[Fraction, Fraction]]:
    deltas = [b - a for a, b in zip(window, window[1:])]
    median = sorted(deltas)[len(deltas) // 2]
    candidates: list[Fraction] = []
    for r in list(nominal) + STANDARD_RATES + [Fraction(1) / (Fraction(median) * tb)]:
        if r and r > 0 and r not in candidates:
            candidates.append(r)
    for rate in candidates:
        step = 1 / (rate * tb)  # exact ticks per frame
        if max(abs(t - window[0] - i * step) for i, t in enumerate(window)) <= 1:
            return rate, step
    return None


def repair_terminal_suffix(scan: FrameScan, policy: str, nominal: list[Optional[Fraction]]
                           ) -> tuple[FrameScan, Optional[dict]]:
    """Validate presentation-order timestamps; optionally re-time a short defective terminal suffix.

    Recognition predicate (all must hold, otherwise TimingDefect is raised with diagnostics):
      1. every frame has a PTS;
      2. let k be the first frame whose PTS is not greater than its predecessor's; frames k..n-1 form
         the suffix and frame k-1 is the anchor; the suffix has at most MAX_TERMINAL_SUFFIX frames
         (so the defect is terminal, never interior);
      3. the TERMINAL_WINDOW intervals ending at the anchor follow one constant rate within one tick
         (the anchor is on that grid, i.e. reliable);
      4. every original suffix PTS lies on the same grid (within one tick) between one slot before the
         anchor and the last reconstructed slot;
      5. re-timing never moves a frame earlier and moves it at most MAX_SHIFT_SLOTS slots later;
      6. the suffix's slot-offset signature is one of RECOGNISED_SUFFIXES.
    Repair: frame k+j gets grid slot anchor+1+j. Frames, their order and pixels are preserved; all
    timestamps up to and including the anchor are unchanged. Minimality: the m suffix frames need m
    distinct increasing slots after the anchor, so frame k+j needs slot >= j+1; a frame is changed only if
    its original slot is below that bound (it cannot keep its stamp in any order-preserving assignment),
    and a frame already on slot j+1 (e.g. the final frame in (0, 1, 3) and (0, 2)) is left unchanged.
    """
    ts, tb = scan.ts, scan.time_base
    n = len(ts)
    bad = [i for i in range(1, n) if ts[i] <= ts[i - 1]]
    if not bad and not scan.missing_ts:
        return scan, None
    how = ("Remux or repair the source timestamps first (for example with a tool that rewrites FLV "
           "timestamps), or choose another source.")
    if scan.missing_ts:
        raise TimingDefect(f"Unsupported timestamp defect: {describe_defects(scan)}. {how}")
    k = bad[0]
    m = n - k
    if m > MAX_TERMINAL_SUFFIX:
        raise TimingDefect(
            f"Unsupported timestamp defect: the first non-increasing timestamp is frame #{k}, {m} frames before the "
            f"end; only a terminal suffix of at most {MAX_TERMINAL_SUFFIX} frames can be re-timed. "
            f"{describe_defects(scan)}. {how}")
    anchor = k - 1
    suffix_desc = ", ".join(f"#{j}={_fmt_s(ts[j], tb)}" for j in range(max(0, anchor - 1), n))
    if policy == "strict":
        raise TimingDefect(
            f"The last {m} frame(s) have duplicate or backwards timestamps after frame #{anchor} ({suffix_desc}). "
            f"--timestamp-policy strict rejects this; use --timestamp-policy repair-terminal to re-time them to the "
            f"consecutive frame slots after frame #{anchor}.")
    if anchor < TERMINAL_WINDOW:
        raise TimingDefect(f"Terminal timestamp defect after frame #{anchor}, but fewer than {TERMINAL_WINDOW} "
                           f"preceding intervals are available to infer the frame rate; cannot repair.")
    window = ts[anchor - TERMINAL_WINDOW:anchor + 1]
    fit = _fit_rate(window, tb, nominal)
    if fit is None:
        deltas = [b - a for a, b in zip(window, window[1:])]
        raise TimingDefect(
            f"Terminal timestamp defect after frame #{anchor}, but the {TERMINAL_WINDOW} intervals before it are not "
            f"a constant rate (intervals in ticks: {deltas}); the anchor cannot be trusted. {how}")
    rate, step = fit
    residuals = [t - window[0] - i * step for i, t in enumerate(window)]
    offset = (max(residuals) + min(residuals)) / 2

    def slot(i: int) -> Fraction:  # exact grid position of frame index i
        return window[0] + offset + (i - (anchor - TERMINAL_WINDOW)) * step

    def rnd(x: Fraction) -> int:
        return int(math.floor(x + Fraction(1, 2)))

    frames = []
    problems = []
    for j in range(k, n):
        orig = ts[j]
        rep = rnd(slot(j))
        on_grid = abs((orig - slot(anchor)) / step - round((orig - slot(anchor)) / step)) * step <= 1
        lo, hi = slot(anchor) - step - 1, slot(n - 1) + 1
        shift = rep - orig
        if not on_grid:
            problems.append(f"#{j} PTS {_fmt_s(orig, tb)} is not on the {ratio_str(rate)} fps grid")
        if not (lo <= orig <= hi):
            problems.append(f"#{j} PTS {_fmt_s(orig, tb)} is outside [{float(lo * tb):.6f}, {float(hi * tb):.6f}] s")
        if shift < -1 or shift > MAX_SHIFT_SLOTS * step + 1:
            problems.append(f"#{j} would move by {float(shift * tb) * 1000:+.3f} ms (allowed 0 to "
                            f"+{MAX_SHIFT_SLOTS} slots)")
        slot_off = (orig - ts[anchor]) / step
        if rep == orig:
            reason = "already on its slot; unchanged"
        elif orig <= ts[j - 1]:
            reason = "not later than the previous frame (duplicate/backwards stamp)"
        else:
            reason = (f"its slot {float(slot_off):+g} is taken by the {j - k} earlier suffix frame(s), which need "
                      f"slots +1..+{j - k}; order-preserving re-timing must move it")
        frames.append({"index": j, "original_pts_ticks": orig, "original_pts_s": float(orig * tb),
                       "repaired_pts_ticks": rep, "repaired_pts_s": float(rep * tb), "changed": rep != orig,
                       "shift_ms": float(shift * tb * 1000), "reason": reason,
                       "original_slot_offset": float(slot_off),
                       "repaired_slot_offset": float((rep - ts[anchor]) / step)})
    signature = tuple(round(f["original_slot_offset"]) for f in frames)
    if not problems and signature not in RECOGNISED_SUFFIXES:
        problems.append(f"slot-offset signature {list(signature)} is not one of the recognised terminal patterns "
                        f"{[list(p) for p in RECOGNISED_SUFFIXES]}")
    if problems:
        raise TimingDefect(f"Unrecognised terminal timestamp pattern after frame #{anchor} ({suffix_desc}): "
                           + "; ".join(problems) + f". {how}")
    new_ts = ts[:k] + [f["repaired_pts_ticks"] for f in frames]
    if any(b <= a for a, b in zip(new_ts[anchor:], new_ts[anchor + 1:])):
        raise TimingDefect("Re-timed suffix would not be strictly increasing; cannot repair.")
    step_ticks = rnd(step)
    new_durs = list(scan.durations[:k]) + [step_ticks] * m
    repaired = FrameScan(scan.source, tb, new_ts, new_durs, scan.interlaced_flags, scan.tff_flags, 0)
    last_dur = scan.durations[-1] if scan.durations and scan.durations[-1] else step_ticks
    old_end = max(ts) + last_dur
    new_end = new_ts[-1] + step_ticks
    orig_offsets = [round(f["original_slot_offset"], 3) for f in frames]
    record = {
        "kind": "terminal-suffix-retime",
        "pattern": f"slot offsets after anchor {orig_offsets} -> {[j + 1 for j in range(m)]}",
        "pattern_description": RECOGNISED_SUFFIXES[signature],
        "scan_method": scan.source,
        "time_base": ratio_str(tb),
        "frames_total": n,
        "anchor_index": anchor,
        "anchor_pts_ticks": ts[anchor],
        "anchor_pts_s": float(ts[anchor] * tb),
        "suffix_start_index": k,
        "suffix_length": m,
        "frames": frames,
        "frames_changed": sum(f["changed"] for f in frames),
        "inferred_rate_fps": ratio_str(rate),
        "inferred_interval_s": ratio_str(step * tb),
        "inferred_interval_ms": float(step * tb * 1000),
        "intervals_used": TERMINAL_WINDOW,
        "original_video_end_s": float(old_end * tb),
        "repaired_video_end_s": float(new_end * tb),
        "video_duration_change_s": float((new_end - old_end) * tb),
        "limits": {"max_suffix_frames": MAX_TERMINAL_SUFFIX, "max_shift_slots": MAX_SHIFT_SLOTS,
                   "stable_intervals_required": TERMINAL_WINDOW},
        "note": "every frame kept in order with its pixels; only the suffix timestamps changed; earlier frames "
                "and all audio timestamps are untouched",
    }
    return repaired, record


repair_terminal_duplicate = repair_terminal_suffix  # name used before the suffix generalisation


def splice_decoded_tail(ffprobe: Path, path: Path, stream_index: int, packets: FrameScan,
                        duration_s: Optional[float], seconds: float = 12.0) -> FrameScan:
    """Packet timestamps for most of the file, decoded presentation order for its last seconds.

    Sorting packet PTS cannot reveal duplicate/backwards stamps at the end (sorting reorders them), so
    the fast scan replaces its final part with decoder output. Falls back to the packet scan if the
    two do not line up frame-for-frame.
    """
    if not duration_s or not packets.ts:
        return packets
    tail = scan_frames(ffprobe, path, stream_index, decode=True, time_base=packets.time_base,
                       read_intervals=f"{max(duration_s - seconds, 0):.3f}%", show_progress=False)
    if len(tail.ts) < 4:
        return packets
    cutoff = tail.ts[len(tail.ts) // 2]
    cut = tail.ts.index(cutoff)
    head = [(t, d) for t, d in zip(packets.ts, packets.durations) if t < cutoff]
    if len(head) + len(tail.ts) - cut != len(packets.ts):
        return packets
    return FrameScan("packets+decoded-tail", packets.time_base, [t for t, _ in head] + tail.ts[cut:],
                     [d for _, d in head] + tail.durations[cut:], tail.interlaced_flags, tail.tff_flags,
                     packets.missing_ts + tail.missing_ts)


# --------------------------------------------------------------------------------------------
# Timing analysis
# --------------------------------------------------------------------------------------------

@dataclass
class TimingAnalysis:
    frames: int
    time_base: str
    first_pts_s: Optional[float]
    last_pts_s: Optional[float]
    end_s: Optional[float]
    last_frame_duration_ticks: Optional[int]
    strictly_increasing: bool
    non_increasing_count: int
    delta_min_ms: Optional[float]
    delta_median_ms: Optional[float]
    delta_max_ms: Optional[float]
    distinct_deltas_ticks: dict
    irregular_delta_count: int
    cfr_fit: Optional[dict]
    classification: str
    notes: list[str] = field(default_factory=list)


def analyze_timing(scan: FrameScan, nominal_rates: list[Optional[Fraction]]) -> TimingAnalysis:
    ts = scan.ts
    tb = scan.time_base
    n = len(ts)
    notes: list[str] = []
    if n == 0:
        return TimingAnalysis(0, ratio_str(tb), None, None, None, None, False, 0, None, None, None, {}, 0, None,
                              "no-frames", ["no video frames found"])
    deltas = [b - a for a, b in zip(ts, ts[1:])]
    non_inc = sum(1 for d in deltas if d <= 0)
    to_ms = float(tb) * 1000.0
    hist: dict[int, int] = {}
    for d in deltas:
        hist[d] = hist.get(d, 0) + 1
    top = dict(sorted(hist.items(), key=lambda kv: -kv[1])[:8])
    median = statistics.median(deltas) if deltas else None
    irregular = sum(1 for d in deltas if median is not None and abs(d - median) > 1)

    last_dur = None
    if scan.durations and scan.durations[-1] and scan.durations[-1] > 0:
        last_dur = scan.durations[-1]
    elif median and median > 0:
        last_dur = int(round(median))

    fit = None
    if n >= 2 and non_inc == 0:
        # Priority order: the stream's nominal rate, then standard rates, then the average rate and a
        # median-interval guess. The first rate that fits every timestamp within one tick wins, so a
        # metadata-derived rational such as 983/41 does not displace 24000/1001.
        nominal, average = (nominal_rates + [None, None])[:2]
        ordered = [nominal] + STANDARD_RATES + [average]
        if median and median > 0:
            ordered.append((Fraction(1) / (Fraction(int(round(median))) * tb)).limit_denominator(1001))
        candidates: list[Fraction] = []
        for r in ordered:
            if r and 0 < r <= 240 and r not in candidates:
                candidates.append(r)
        best = None
        t0 = ts[0]
        for rate in candidates:
            frame_ticks = 1 / (rate * tb)  # ticks per frame, as p/q
            p, q = frame_ticks.numerator, frame_ticks.denominator
            worst = 0
            for i, t in enumerate(ts):
                dev = abs(q * (t - t0) - i * p)
                if dev > worst:
                    worst = dev
                    if worst > q:  # more than one tick off: this rate cannot fit
                        break
            dev_ticks = Fraction(worst, q)
            if dev_ticks <= 1:
                best = (rate, dev_ticks)
                break
            if best is None or dev_ticks < best[1]:
                best = (rate, dev_ticks)
        if best is not None and best[1] <= 1:
            fit = {"rate": ratio_str(best[0]), "rate_float": float(best[0]),
                   "max_deviation_ticks": float(best[1]), "max_deviation_ms": float(best[1]) * to_ms}

    if n < 2:
        classification = "single-frame"
    elif non_inc:
        classification = "non-monotonic"
        notes.append(f"{non_inc} frame timestamps are not strictly increasing")
    elif fit and fit["max_deviation_ticks"] == 0:
        classification = "cfr"
    elif fit:
        classification = "cfr-within-rounding"
        notes.append(
            f"timestamps match {fit['rate']} fps within {fit['max_deviation_ms']:.3f} ms "
            f"(container time-base rounding)")
    else:
        classification = "vfr"
        notes.append(f"{irregular} of {len(deltas)} frame intervals deviate from the median interval by more than "
                     f"one time-base tick; timing is variable and will be preserved frame-by-frame")
    end = ts[-1] + (last_dur or 0)
    return TimingAnalysis(
        frames=n, time_base=ratio_str(tb), first_pts_s=ts[0] * float(tb), last_pts_s=ts[-1] * float(tb),
        end_s=end * float(tb), last_frame_duration_ticks=last_dur, strictly_increasing=non_inc == 0,
        non_increasing_count=non_inc,
        delta_min_ms=min(deltas) * to_ms if deltas else None,
        delta_median_ms=float(median) * to_ms if median is not None else None,
        delta_max_ms=max(deltas) * to_ms if deltas else None,
        distinct_deltas_ticks={str(k): v for k, v in top.items()}, irregular_delta_count=irregular,
        cfr_fit=fit, classification=classification, notes=notes,
    )


# --------------------------------------------------------------------------------------------
# Interlacing
# --------------------------------------------------------------------------------------------

_IDET_RE = re.compile(r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)")


def run_idet(ffmpeg: Path, path: Path, stream_index: int, duration_s: Optional[float], frames: int = 240) -> dict:
    """Run FFmpeg's idet filter on a few windows of the source."""
    positions = [0.0]
    if duration_s and duration_s > 30:
        positions = [duration_s * f for f in (0.15, 0.5, 0.8)]
    totals = {"tff": 0, "bff": 0, "progressive": 0, "undetermined": 0}
    for pos in positions:
        seek = ["-ss", f"{pos:.3f}"] if pos > 0 else []
        cmd = [ffmpeg, "-hide_banner", "-nostdin", "-nostats", *seek, "-i", str(path),
               "-map", f"0:{stream_index}", "-frames:v", str(frames), "-vf", "idet", "-an", "-f", "null", "-"]
        cp = capture(cmd, timeout=600)
        matches = _IDET_RE.findall(cp.stderr)
        if matches:
            t, b, p, u = (int(x) for x in matches[-1])
            totals["tff"] += t
            totals["bff"] += b
            totals["progressive"] += p
            totals["undetermined"] += u
    totals["windows_start_s"] = [round(p, 3) for p in positions]
    return totals


def interlace_verdict(vstream: dict, scan: Optional[FrameScan], idet: Optional[dict]) -> dict:
    reasons = []
    field_order = vstream.get("field_order")
    coded_interlaced = field_order in ("tt", "bb", "tb", "bt")
    if coded_interlaced:
        reasons.append(f"stream field_order={field_order}")
    flag_ratio = None
    if scan and scan.source == "decoded-frames" and scan.ts:
        flag_ratio = scan.interlaced_flags / len(scan.ts)
        if flag_ratio > 0:
            reasons.append(f"{scan.interlaced_flags}/{len(scan.ts)} decoded frames flagged interlaced")
    idet_ratio = None
    if idet:
        inter = idet["tff"] + idet["bff"]
        decided = inter + idet["progressive"]
        if decided:
            idet_ratio = inter / decided
            reasons.append(f"idet: {inter} interlaced-looking vs {idet['progressive']} progressive frames "
                           f"({idet['undetermined']} undetermined)")
    # Coded field order or decoder interlace flags are definitive. idet is a content heuristic that also
    # fires on sharp, fast-moving progressive material (line art, synthetic patterns), so on its own it
    # only produces a warning.
    if coded_interlaced or (flag_ratio is not None and flag_ratio > 0.5):
        verdict = "likely-interlaced"
    elif (flag_ratio is not None and flag_ratio > 0.05) or (idet_ratio is not None and idet_ratio > 0.15
                                                            and idet["tff"] + idet["bff"] >= 20):
        verdict = "possibly-interlaced-or-telecined"
    else:
        verdict = "progressive"
    return {"verdict": verdict, "field_order": field_order, "flagged_frame_ratio": flag_ratio,
            "idet_interlaced_ratio": idet_ratio, "idet": idet, "reasons": reasons}


# --------------------------------------------------------------------------------------------
# Stream extents (start/end) for sync checks
# --------------------------------------------------------------------------------------------

def stream_extent(ffprobe: Path, path: Path, stream_index: int, *, decode: bool) -> dict:
    """First presentation time and end time (last pts + duration) of one stream, in seconds.

    decode=True uses decoded frames, which honours encoder priming/skip-samples for audio.
    """
    kind = "frame" if decode else "packet"
    entries = f"{kind}=pts_time,duration_time" + (",best_effort_timestamp_time" if decode else "")
    cmd = [ffprobe, "-v", "error", "-select_streams", str(stream_index), "-show_entries", entries,
           "-of", "csv=p=0:nk=0", str(path)]
    first = None
    end = None
    count = 0

    def on_line(line: str) -> None:
        nonlocal first, end, count
        fields = dict(kv.split("=", 1) for kv in line.split(",") if "=" in kv)
        t = _num(fields.get("best_effort_timestamp_time")) if decode else None
        if t is None:
            t = _num(fields.get("pts_time"))
        if t is None:
            return
        d = _num(fields.get("duration_time")) or 0.0
        count += 1
        if first is None or t < first:
            first = t
        if end is None or t + d > end:
            end = t + d

    rc = stream_stdout_lines(cmd, on_line)
    if rc != 0:
        raise ToolError(f"ffprobe extent scan failed for stream {stream_index} of {path}")
    return {"start_s": first, "end_s": end, "units": count, "method": "decoded-frames" if decode else "packets"}


# --------------------------------------------------------------------------------------------
# High-level inspection
# --------------------------------------------------------------------------------------------

def nominal_rates(v: dict) -> list[Optional[Fraction]]:
    return [parse_ratio(v.get("r_frame_rate")), parse_ratio(v.get("avg_frame_rate"))]


def source_global_start(info: dict) -> float:
    s = _num(info.get("format", {}).get("start_time"))
    if s is not None:
        return s
    starts = [_num(st.get("start_time")) for st in info["streams"]]
    starts = [x for x in starts if x is not None]
    return min(starts) if starts else 0.0


def summarize_streams(info: dict) -> list[dict]:
    out = []
    for s in info["streams"]:
        tags = s.get("tags", {}) or {}
        d = {
            "index": s.get("index"), "type": s.get("codec_type"), "codec": s.get("codec_name"),
            "codec_long": s.get("codec_long_name"), "profile": s.get("profile"),
            "time_base": s.get("time_base"), "start_time_s": _num(s.get("start_time")),
            "duration_s": _num(s.get("duration")), "bit_rate": _int(s.get("bit_rate")),
            "language": tags.get("language"), "title": tags.get("title"),
            "default": bool(s.get("disposition", {}).get("default")),
        }
        if s.get("codec_type") == "video":
            sar = parse_ratio(s.get("sample_aspect_ratio")) or Fraction(1)
            d.update({
                "width": s.get("width"), "height": s.get("height"),
                "coded_width": s.get("coded_width"), "coded_height": s.get("coded_height"),
                "sample_aspect_ratio": ratio_str(sar),
                "display_aspect_ratio": ratio_str(Fraction(s["width"], s["height"]) * sar) if s.get("width") else None,
                "pix_fmt": s.get("pix_fmt"), "field_order": s.get("field_order"),
                "color_range": s.get("color_range"), "color_space": s.get("color_space"),
                "color_transfer": s.get("color_transfer"), "color_primaries": s.get("color_primaries"),
                "chroma_location": s.get("chroma_location"),
                "r_frame_rate": s.get("r_frame_rate"), "avg_frame_rate": s.get("avg_frame_rate"),
                "has_b_frames": s.get("has_b_frames"), "nb_frames": _int(s.get("nb_frames")),
                "attached_pic": bool(s.get("disposition", {}).get("attached_pic")),
            })
        elif s.get("codec_type") == "audio":
            d.update({"sample_rate": _int(s.get("sample_rate")), "channels": s.get("channels"),
                      "channel_layout": s.get("channel_layout"), "sample_fmt": s.get("sample_fmt")})
        out.append(d)
    return out


ProgressCb = Callable[[str], None]
