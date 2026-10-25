"""Environment checks: tools, versions, GPU access through Video2X, models and encoder support."""

from __future__ import annotations

import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

from . import __version__, probe, tools
from .pipeline import Settings, video2x_cmd, video2x_exit_status
from .procs import RunLog, ToolError, capture, run

REQUIRED_V2X_OPTIONS = {"input", "output", "processor", "realesrgan-model", "scaling-factor", "device",
                        "list-devices", "no-copy-streams", "pix-fmt", "codec", "extra-encoder-option"}


def doctor(names: dict, s: Settings, *, inference_test: bool = True) -> dict:
    checks: list[dict] = []

    def add(name: str, status: str, detail) -> None:
        checks.append({"check": name, "status": status, "detail": detail})

    add("platform", "info", {"os": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0],
                             "executable": sys.executable, "venv": sys.prefix != sys.base_prefix,
                             "animenhance": __version__})
    try:
        import av
        import numpy
        add("python packages", "pass", {"av": av.__version__, "numpy": numpy.__version__,
                                        "pyav_ffmpeg_libavformat": ".".join(map(str, av.library_versions["libavformat"]))})
    except ImportError as e:
        add("python packages", "fail", f"{e}; run: pip install -r requirements.txt")

    try:
        ts = tools.resolve_tools(**names)
    except ToolError as e:
        add("tool paths", "fail", str(e))
        return {"ok": False, "checks": checks}
    add("tool paths", "pass", ts.as_dict())

    pinned = tools.manifest()
    try:
        v = tools.video2x_version(ts.video2x)
        want = pinned.get("video2x", {}).get("release")
        status = "pass" if not want or want == v else "warn"
        add("video2x version", status, {"installed": v, "pinned_release": want,
                                        "source_revision": pinned.get("video2x", {}).get("source_commit")})
    except ToolError as e:
        add("video2x version", "fail", str(e))
    opts = tools.video2x_cli_options(ts.video2x)
    missing_opts = sorted(REQUIRED_V2X_OPTIONS - opts)
    add("video2x CLI options", "fail" if missing_opts else "pass",
        {"missing": missing_opts} if missing_opts else "all options used by the wrapper are present")

    for name, path in (("ffmpeg", ts.ffmpeg), ("ffprobe", ts.ffprobe)):
        try:
            add(f"{name} version", "pass", tools.ffmpeg_version(path))
        except Exception as e:  # noqa: BLE001
            add(f"{name} version", "fail", str(e))
    missing = tools.missing_ffmpeg_features(ts.ffmpeg)
    add("ffmpeg encoders/filters/muxers", "fail" if missing else "pass",
        {"missing": missing} if missing else {k: v for k, v in tools.REQUIRED_FFMPEG.items()})

    try:
        devices = list(tools.video2x_devices(ts.video2x))
        add("GPU list (video2x --list-devices)", "pass" if devices else "fail", devices or "no Vulkan devices")
        if not any(d["index"] == s.gpu for d in devices):
            add("selected GPU", "fail", f"GPU index {s.gpu} not in the Video2X device list")
        else:
            add("selected GPU", "pass", next(d for d in devices if d["index"] == s.gpu))
    except ToolError as e:
        add("GPU list (video2x --list-devices)", "fail", str(e))

    try:
        models = tools.available_models(ts.video2x)
        add("Real-ESRGAN models (installed backend)", "pass", models)
        param, binf = tools.model_files(ts.video2x, s.model, s.scale)
        ok = s.model in models and s.scale in models[s.model]
        add(f"model files {s.model} x{s.scale}", "pass" if ok else "fail",
            {"param": str(param), "param_bytes": param.stat().st_size if param.exists() else None,
             "bin": str(binf), "bin_bytes": binf.stat().st_size if binf.exists() else None})
        want_hash = pinned.get("models", {}).get(f"{s.model}-x{s.scale}")
        if want_hash:
            import hashlib
            got = {"param": hashlib.sha256(param.read_bytes()).hexdigest(), "bin": hashlib.sha256(binf.read_bytes()).hexdigest()}
            add("model checksum vs manifest", "pass" if got == want_hash else "warn", {"expected": want_hash, "found": got})
    except ToolError as e:
        add("Real-ESRGAN models (installed backend)", "fail", str(e))

    font = tools.find_label_font()
    add("label font for comparisons", "pass" if font else "warn", str(font) if font else "none found; labels omitted")

    if inference_test:
        add_results = _inference_test(ts, s)
        for item in add_results:
            checks.append(item)

    ok = not any(c["status"] == "fail" for c in checks)
    return {"ok": ok, "checks": checks}


def _inference_test(ts: tools.Toolset, s: Settings) -> list[dict]:
    """Real Vulkan inference on a tiny generated clip, through both output paths the pipeline uses."""
    results = []
    tmp = Path(tempfile.mkdtemp(prefix="animenhance-doctor-"))
    log = RunLog(path=tmp / "doctor.log")
    try:
        clip = tmp / "tiny source.mkv"
        run([ts.ffmpeg, "-hide_banner", "-nostdin", "-y", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=24",
             "-frames:v", "6", "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(clip)], log=log, label="make test clip")
        for codec, name in (("libx264", "out h264.mp4"), ("ffv1", "out ffv1.mkv")):
            out = tmp / name
            t0 = time.monotonic()
            try:
                res = run(video2x_cmd(ts, clip, out, s, codec), log=log, label=f"video2x {codec}",
                          cwd=ts.video2x.parent, no_window=True, check=False)
                warning = video2x_exit_status(res)
                info = probe.ffprobe_json(ts.ffprobe, out)
                v = probe.first_video_stream(info)
                frames = len(probe.scan_frames(ts.ffprobe, out, int(v["index"]), decode=True,
                                               time_base=probe.parse_ratio(v["time_base"]), show_progress=False).ts)
                exp = (96 * s.scale, 64 * s.scale)
                ok = (v["width"], v["height"]) == exp and frames == 6
                results.append({"check": f"real GPU inference ({codec} output)", "status": "pass" if ok else "fail",
                                "detail": {"gpu": s.gpu, "model": s.model, "scale": s.scale,
                                           "output_size": [v["width"], v["height"]], "expected_size": list(exp),
                                           "frames": frames, "expected_frames": 6,
                                           "seconds": round(time.monotonic() - t0, 2),
                                           "exit_code": res.returncode, "exit_warning": warning}})
            except ToolError as e:
                results.append({"check": f"real GPU inference ({codec} output)", "status": "fail", "detail": str(e)})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return results
