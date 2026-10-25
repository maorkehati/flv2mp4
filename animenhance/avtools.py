"""Packet-level operations with PyAV (libav* bindings). No re-encoding happens here."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Optional

import av
import numpy as np

from .procs import ToolError


def _nal_types_avcc(data: bytes, length_size: int, hevc: bool) -> list[int]:
    types = []
    pos = 0
    while pos + length_size <= len(data):
        size = int.from_bytes(data[pos:pos + length_size], "big")
        pos += length_size
        if size <= 0 or pos + size > len(data):
            break
        b = data[pos]
        types.append((b >> 1) & 0x3F if hevc else b & 0x1F)
        pos += size
    return types


def _avcc_length_size(extradata: Optional[bytes], hevc: bool) -> Optional[int]:
    if not extradata:
        return None
    if not hevc and len(extradata) > 4 and extradata[0] == 1:
        return (extradata[4] & 0x03) + 1
    if hevc and len(extradata) > 21 and extradata[0] == 1:
        return (extradata[21] & 0x03) + 1
    return None


def pad_remux(src: Path, dst: Path, stream_index: int, pad_count: int) -> dict:
    """Stream-copy one video stream into Matroska and append ``pad_count`` copies of an IDR packet.

    Video2X 6.4.0 does not drain its decoder at end of input, so a codec with a reorder delay of
    k frames loses its last k frames. The appended IDR copies push the real frames out of the
    decoder; the copies themselves stay buffered and are never emitted.
    """
    with av.open(str(src)) as inp:
        ist = inp.streams[stream_index]
        codec = ist.codec_context.name
        if codec not in ("h264", "hevc"):
            raise ToolError(
                f"Source codec {codec} has a decoder reorder delay ({pad_count} frames), but the end-of-stream "
                f"workaround only supports H.264/HEVC. Video2X 6.4.0 would drop the last frames.")
        hevc = codec == "hevc"
        extradata = ist.codec_context.extradata
        length_size = _avcc_length_size(extradata, hevc)
        if length_size is None:
            raise ToolError("Could not parse the codec configuration record to locate an IDR packet.")
        idr_types = {19, 20} if hevc else {5}
        tb = ist.time_base
        idr: Optional[bytes] = None
        count = 0
        last_ts = None
        typical = None
        prev = None
        pts_clamped = 0
        with av.open(str(dst), "w", format="matroska") as out:
            ost = out.add_stream_from_template(ist)
            for pkt in inp.demux(ist):
                if pkt.size == 0:
                    continue
                # libavformat refuses PTS < DTS (present in some defective sources). Video2X discards input
                # timestamps and numbers frames by decode output index, so this copy only needs valid ones;
                # decode order and the decoded frames are unaffected.
                if pkt.pts is not None and pkt.dts is not None and pkt.pts < pkt.dts:
                    pkt.pts = pkt.dts
                    pts_clamped += 1
                if idr is None and pkt.is_keyframe:
                    payload = bytes(pkt)
                    if idr_types & set(_nal_types_avcc(payload, length_size, hevc)):
                        idr = payload
                ts = pkt.pts if pkt.pts is not None else pkt.dts
                if ts is not None:
                    if prev is not None and ts > prev and typical is None:
                        typical = ts - prev
                    prev = ts
                    last_ts = ts if last_ts is None else max(last_ts, ts, pkt.dts or ts)
                pkt.stream = ost
                out.mux(pkt)
                count += 1
            if idr is None:
                raise ToolError("No IDR keyframe found in the video stream; cannot apply the end-of-stream "
                                "workaround needed by Video2X 6.4.0.")
            step = typical or 1
            for i in range(pad_count):
                p = av.Packet(idr)
                p.pts = p.dts = (last_ts or 0) + step * (i + 1)
                p.time_base = tb
                p.is_keyframe = True
                p.stream = ost
                out.mux(p)
    return {"packets_copied": count, "idr_copies_appended": pad_count, "codec": codec,
            "packets_with_pts_below_dts_clamped": pts_clamped}


def read_packet_pts(path: Path) -> tuple[list[int], Fraction]:
    with av.open(str(path)) as inp:
        ist = inp.streams.video[0]
        pts = []
        for pkt in inp.demux(ist):
            if pkt.size == 0:
                continue
            if pkt.pts is None:
                raise ToolError(f"{path}: video packet without PTS; cannot retime")
            pts.append(pkt.pts)
        return pts, ist.time_base


def retime_remux(src: Path, dst: Path, target_pts: list[int], target_tb: Fraction,
                 last_duration: int) -> dict:
    """Copy the video packets of ``src`` to MP4, replacing timestamps with ``target_pts``.

    ``target_pts`` lists the desired presentation time of every frame in display order. Each packet's
    display index is the rank of its original PTS. Decode timestamps are rebuilt from the same table,
    delayed by the stream's reorder depth so that DTS <= PTS and DTS is strictly increasing.
    """
    src_pts, _ = read_packet_pts(src)
    n = len(src_pts)
    if n != len(target_pts):
        raise ToolError(f"Retiming needs one timestamp per frame: video has {n} frames, "
                        f"timing table has {len(target_pts)}")
    if len(set(src_pts)) != n:
        raise ToolError("Enhanced video has duplicate packet timestamps; cannot map frames to source timing")
    if any(b <= a for a, b in zip(target_pts, target_pts[1:])):
        raise ToolError("Target timestamps are not strictly increasing")
    if target_pts and target_pts[0] < 0:
        raise ToolError("Target timestamps start before zero; MP4 players would drop those frames as pre-roll")
    order = sorted(range(n), key=lambda i: src_pts[i])
    rank = [0] * n
    for display_idx, decode_idx in enumerate(order):
        rank[decode_idx] = display_idx
    delay = max(i - rank[i] for i in range(n)) if n else 0
    first_step = target_pts[1] - target_pts[0] if n > 1 else max(last_duration, 1)

    def dts_for(j: int) -> int:
        return target_pts[j] if j >= 0 else target_pts[0] + j * first_step

    with av.open(str(src)) as inp, av.open(str(dst), "w", format="mp4") as out:
        ist = inp.streams.video[0]
        ost = out.add_stream_from_template(ist)
        ost.time_base = target_tb
        i = 0
        for pkt in inp.demux(ist):
            if pkt.size == 0:
                continue
            r = rank[i]
            pkt.pts = target_pts[r]
            pkt.dts = dts_for(i - delay)
            pkt.duration = (target_pts[r + 1] - target_pts[r]) if r + 1 < n else last_duration
            pkt.time_base = target_tb
            pkt.stream = ost
            out.mux(pkt)
            i += 1
    return {"frames": n, "reorder_delay": delay, "time_base": f"{target_tb.numerator}/{target_tb.denominator}"}


def verify_interval_exact(src: Path, stream_index: int, seek_s: float, expected_pts: list[int], clip: Path) -> dict:
    """Prove a lossless clip holds exactly the intended source frames.

    The source is decoded independently from ``seek_s``; the window is located by its first frame's PTS and
    must then reproduce ``expected_pts`` frame by frame (decoder output order, defective stamps included).
    Each of those frames must be pixel-identical to the clip's frame at the same index.
    """
    n = len(expected_pts)
    with av.open(str(src)) as s, av.open(str(clip)) as c:
        sst = s.streams[stream_index]
        sst.thread_type = "AUTO"
        if seek_s > 0:
            s.seek(int(seek_s / float(sst.time_base)), stream=sst, backward=True, any_frame=False)
        src_frames = (f for f in s.decode(sst) if f.pts is not None)
        clip_frames = c.decode(video=0)
        started = False
        i = 0
        for f in src_frames:
            if not started:
                if f.pts != expected_pts[0]:
                    continue
                started = True
            if f.pts != expected_pts[i]:
                raise ToolError(f"Source frame {i} of the interval has PTS {f.pts}, expected {expected_pts[i]}")
            g = next(clip_frames, None)
            if g is None:
                raise ToolError(f"Lossless clip ended after {i} of {n} frames")
            if f.format.name != g.format.name or not np.array_equal(f.to_ndarray(), g.to_ndarray()):
                raise ToolError(f"Lossless clip frame {i} differs from source frame with PTS {f.pts}")
            i += 1
            if i == n:
                break
        if i != n:
            raise ToolError(f"Only {i} of {n} interval frames found in the source decode")
        if next(clip_frames, None) is not None:
            raise ToolError("Lossless clip has more frames than the interval")
    return {"frames": n, "pts_sequence_matches": True, "pixels_identical": True}


def gray_frames(path: Path, *, which: str, count: int, size: tuple[int, int],
                stream_index: Optional[int] = None, tail_margin_s: float = 15.0) -> list[np.ndarray]:
    """Decode ``count`` frames from the head or tail of a video, as small grayscale arrays.

    For the tail, the container is sought to ``tail_margin_s`` before its end and decoded to EOF.
    """
    w, h = size
    frames: list[np.ndarray] = []
    with av.open(str(path)) as c:
        st = c.streams[stream_index] if stream_index is not None else c.streams.video[0]
        st.thread_type = "AUTO"
        if which == "tail" and c.duration and c.duration / 1e6 > tail_margin_s:
            start = c.start_time or 0
            target = start + c.duration - int(tail_margin_s * 1e6)
            c.seek(target, backward=True, any_frame=False)
        for f in c.decode(st):
            arr = f.reformat(width=w, height=h, format="gray", interpolation="AREA").to_ndarray().astype(np.float32)
            frames.append(arr)
            if which == "head" and len(frames) >= count:
                break
            if which == "tail" and len(frames) > count:
                frames.pop(0)
    return frames


def alignment_check(src_frames: list[np.ndarray], out_frames: list[np.ndarray], max_shift: int = 2) -> dict:
    """Compare frame-index alignment hypotheses between a source window and the matching output window.

    For each shift s, the score is the mean absolute difference between output frame i and source
    frame i+s. Correct alignment means shift 0 scores best. A frame dropped or duplicated inside or
    before the window shows up as a better score at a non-zero shift. When content barely changes
    (held or static frames) the hypotheses cannot be separated and the result is "inconclusive".
    """
    n = min(len(src_frames), len(out_frames))
    if n < 2 * max_shift + 3:
        return {"result": "inconclusive", "reason": f"only {n} frames sampled", "scores": {}}
    idx = range(max_shift, n - max_shift)
    scores = {}
    for s in range(-max_shift, max_shift + 1):
        scores[s] = float(np.mean([np.abs(out_frames[i] - src_frames[i + s]).mean() for i in idx]))
    best_shift = min(scores, key=lambda s: scores[s])
    spread = max(scores.values()) - min(scores.values())
    if spread < 0.5:
        result = "inconclusive"
    elif scores[0] <= scores[best_shift] + 0.05 * spread:
        result = "aligned"
    else:
        result = "misaligned"
    return {"result": result, "best_shift": best_shift, "frames": n,
            "scores": {str(k): round(v, 3) for k, v in scores.items()}}
