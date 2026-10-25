"""Generate small synthetic sources for the integration tests (run directly or via pytest fixtures)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from animenhance import tools  # noqa: E402

DATA = ROOT / "tests" / "_data"
SPACED_DIR = DATA / "dir with spaces"
UNICODE_NAME = "アニメ テスト ü.flv"


def ff(*args: str) -> None:
    ffmpeg = str(tools.BUNDLED_FFMPEG)
    subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y", *args], check=True)


def build(force: bool = False) -> dict[str, Path]:
    DATA.mkdir(parents=True, exist_ok=True)
    out = {
        "cfr_audio": DATA / "cfr_audio_subs.flv",
        "vfr": DATA / "vfr_irregular.flv",
        "offset": DATA / "offset_anamorphic_nelly.flv",
        "noaudio": DATA / "noaudio_flv1.flv",
        "interlaced": DATA / "interlaced.flv",
        "multi_audio": DATA / "multi_audio.mkv",
        "unicode": SPACED_DIR / UNICODE_NAME,
    }
    tmp = DATA / "_tmp"
    tmp.mkdir(exist_ok=True)

    # 60 frames at 24000/1001, H.264 with B-frames (decoder reorder delay), AAC stereo, text subtitles.
    if force or not out["cfr_audio"].exists():
        srt = tmp / "subs.srt"
        srt.write_text("1\n00:00:00,500 --> 00:00:01,500\nHello\n\n2\n00:00:01,600 --> 00:00:02,200\nWorld\n",
                       encoding="utf-8")
        ff("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=24000/1001",
           "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100,aformat=channel_layouts=stereo",
           "-i", str(srt), "-map", "0:v", "-map", "1:a", "-map", "2:s",
           "-frames:v", "60", "-t", "2.5", "-c:v", "libx264", "-bf", "3", "-g", "24", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-c:s", "text", str(out["cfr_audio"]))

    # Irregular timestamps: base 42 ms with jitter and two gaps; MP3 audio (stream-copy path).
    if force or not out["vfr"].exists():
        ff("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25",
           "-f", "lavfi", "-i", "sine=frequency=660:sample_rate=44100",
           "-frames:v", "50", "-t", "2.4",
           "-vf", "settb=1/1000,setpts='(N*42+17*gte(N\\,20)+60*gte(N\\,35)+mod(N*N\\,7))/1000/TB'",
           "-fps_mode", "passthrough", "-enc_time_base", "1/1000",
           "-c:v", "libx264", "-bf", "2", "-g", "30", "-pix_fmt", "yuv420p", "-c:a", "libmp3lame", "-b:a", "96k",
           str(out["vfr"]))

    # Video starts 0.5 s after audio, whole timeline starts at 2.0 s; SAR 4:3; Nellymoser mono audio.
    if force or not out["offset"].exists():
        v = tmp / "offset_v.mkv"
        a = tmp / "offset_a.wav"
        ff("-f", "lavfi", "-i", "testsrc2=size=128x96:rate=24", "-frames:v", "48", "-vf", "setsar=4/3",
           "-c:v", "libx264", "-bf", "2", "-pix_fmt", "yuv420p", str(v))
        ff("-f", "lavfi", "-i", "sine=frequency=300:sample_rate=22050", "-t", "2.6", "-ac", "1", str(a))
        ff("-itsoffset", "0.5", "-i", str(v), "-i", str(a), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
           "-c:a", "nellymoser", "-output_ts_offset", "2", str(out["offset"]))

    # Sorenson Spark (flv1), no audio, no B-frames.
    if force or not out["noaudio"].exists():
        ff("-f", "lavfi", "-i", "testsrc2=size=176x144:rate=25", "-frames:v", "40", "-c:v", "flv1", "-q:v", "3",
           str(out["noaudio"]))

    # Interlaced-coded H.264 (field_order tt) with real field motion.
    if force or not out["interlaced"].exists():
        ff("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=50", "-frames:v", "100",
           "-vf", "tinterlace=interleave_top,fieldorder=tff", "-c:v", "libx264", "-flags", "+ilme+ildct",
           "-x264-params", "tff=1", "-pix_fmt", "yuv420p", str(out["interlaced"]))

    # Two audio tracks with language/default flags plus a subtitle, in Matroska.
    if force or not out["multi_audio"].exists():
        srt = tmp / "subs2.srt"
        srt.write_text("1\n00:00:00,200 --> 00:00:01,000\nこんにちは\n", encoding="utf-8")
        ff("-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24", "-f", "lavfi", "-i", "sine=frequency=500:sample_rate=48000",
           "-f", "lavfi", "-i", "sine=frequency=800:sample_rate=48000", "-i", str(srt),
           "-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:s", "-frames:v", "36", "-t", "1.5",
           "-c:v", "libx264", "-bf", "2", "-pix_fmt", "yuv420p",
           "-c:a:0", "aac", "-c:a:1", "flac", "-ac:a:1", "2", "-c:s", "srt",
           "-metadata:s:a:0", "language=jpn", "-metadata:s:a:1", "language=eng", "-metadata:s:s:0", "language=jpn",
           "-disposition:a:0", "0", "-disposition:a:1", "default", str(out["multi_audio"]))

    if force or not out["unicode"].exists():
        SPACED_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out["cfr_audio"], out["unicode"])

    # Episode-1-like defect: the frame shown last gets the previous frame's PTS (B-frame stream, AAC).
    out["terminal_dup"] = DATA / "terminal_dup.flv"
    out["interior_dup"] = DATA / "interior_dup.flv"
    if force or not out["terminal_dup"].exists() or not out["interior_dup"].exists():
        base = tmp / "dup_base.flv"
        ff("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25", "-f", "lavfi", "-i", "sine=frequency=520:sample_rate=44100",
           "-frames:v", "50", "-t", "2.0", "-c:v", "libx264", "-bf", "2", "-g", "25", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "96k", str(base))
        _retime_one_frame(base, out["terminal_dup"], which="last")
        _retime_one_frame(base, out["interior_dup"], which="middle")

    # Episode 26-61 patterns: the last frames carry stamps at these slot offsets after the anchor.
    for key, offsets in (("tail_0_1_3", [0, 1, 3]), ("tail_m1_0_1", [-1, 0, 1])):
        out[key] = DATA / f"{key}.flv"
        if force or not out[key].exists():
            base = tmp / "tail_base.flv"
            if not base.exists():
                ff("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25", "-f", "lavfi",
                   "-i", "sine=frequency=610:sample_rate=44100", "-frames:v", "60", "-t", "2.4", "-c:v", "libx264",
                   "-bf", "2", "-g", "30", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", str(base))
            _set_tail_pts(base, out[key], offsets)
    return out


def _set_tail_pts(src: Path, dst: Path, offsets: list[int]) -> None:
    """Give the last len(offsets) frames (presentation order) the stamps anchor + offset * frame interval.

    Patches the signed 24-bit AVC CompositionTime of the FLV video tags (pts = tag timestamp + CTS), which can
    express pts < dts exactly like the real episodes; FFmpeg's muxer refuses to write such packets.
    """
    data = bytearray(src.read_bytes())
    pos = 9 + 4
    tags = []
    while pos + 11 <= len(data):
        typ = data[pos]
        size = int.from_bytes(data[pos + 1:pos + 4], "big")
        dts = int.from_bytes(data[pos + 4:pos + 7], "big") | (data[pos + 7] << 24)
        body = pos + 11
        if typ == 9 and size >= 5 and (data[body] & 0x0F) == 7 and data[body + 1] == 1:
            cts = int.from_bytes(data[body + 2:body + 5], "big", signed=True)
            tags.append((body + 2, dts, dts + cts))
        pos = body + size + 4
    pts_sorted = sorted(p for _, _, p in tags)
    m = len(offsets)
    step = pts_sorted[1] - pts_sorted[0]
    anchor = pts_sorted[-m - 1]
    new = {pts_sorted[-m + j]: anchor + off * step for j, off in enumerate(offsets)}
    for off, dts, pts in tags:
        if pts in new:
            data[off:off + 3] = (new[pts] - dts).to_bytes(3, "big", signed=True)
    dst.write_bytes(bytes(data))


def _retime_one_frame(src: Path, dst: Path, which: str) -> None:
    """Copy src, giving one video frame the presentation timestamp of the frame displayed before it."""
    import av

    with av.open(str(src)) as inp:
        v = inp.streams.video[0]
        pts = sorted(p.pts for p in inp.demux(v) if p.size)
    target = pts[-1] if which == "last" else pts[len(pts) // 2]
    previous = pts[pts.index(target) - 1]
    with av.open(str(src)) as inp, av.open(str(dst), "w", format="flv") as out:
        smap = {s.index: out.add_stream_from_template(s) for s in inp.streams if s.type in ("video", "audio")}
        for pkt in inp.demux():
            if not pkt.size or pkt.stream.index not in smap:
                continue
            if pkt.stream.type == "video" and pkt.pts == target:
                pkt.pts = previous
                if pkt.dts is not None and pkt.dts > previous:
                    pkt.dts = previous
            pkt.stream = smap[pkt.stream.index]
            out.mux(pkt)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for k, p in build(force="--force" in sys.argv).items():
        print(f"{k}: {p}")
