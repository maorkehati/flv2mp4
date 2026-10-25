"""convert.py front end and the terminal-timestamp repair.

Unit tests use synthetic data and a mocked enhance(); tests marked ``ffmpeg`` run real FFprobe on
generated sources but never run Video2X. Real GPU tests for convert.py are in test_convert_gpu.py.
"""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from animenhance import batch, pipeline, probe, tools
from animenhance.procs import ToolError
from conftest import CLI, PY, ROOT

CONVERT = str(ROOT / "convert.py")


def run_convert(*args, timeout=900):
    return subprocess.run([PY, CONVERT, *map(str, args)], capture_output=True, cwd=ROOT, timeout=timeout,
                          encoding="utf-8", errors="replace")


def touch(p: Path, data: bytes = b"FLV") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


# ---------------------------------------------------------------- discovery and planning

def test_natural_order_and_case_insensitive_discovery(tmp_path):
    for name in ("10.FLV", "2.flv", "1.Flv", "notes.txt", "3.mp4", "11.FlV"):
        touch(tmp_path / name)
    (tmp_path / "sub").mkdir()
    touch(tmp_path / "sub" / "4.flv")
    found = [p.name for p in batch.discover(tmp_path, recursive=False, exclude=[])]
    assert found == ["1.Flv", "2.flv", "10.FLV", "11.FlV"]


def test_recursive_discovery_excludes_output_and_temp(tmp_path):
    touch(tmp_path / "1.flv")
    touch(tmp_path / "season 2" / "10.flv")
    touch(tmp_path / "season 2" / "9.flv")
    touch(tmp_path / "enhanced" / "old.flv")
    touch(tmp_path / "x" / ".animenhance-tmp" / "t.flv")
    rels = [str(p.relative_to(tmp_path)) for p in batch.discover(tmp_path, recursive=True, exclude=[tmp_path / "enhanced"])]
    assert rels == ["1.flv", str(Path("season 2") / "9.flv"), str(Path("season 2") / "10.flv")]


def test_plan_file_defaults_and_output_override(tmp_path):
    f = touch(tmp_path / "פרק 1 עם רווח.FLV")
    mode, jobs, root = batch.plan_jobs(f, None, recursive=False, clip=None)
    assert mode == "file" and jobs[0].out == tmp_path / "enhanced" / "פרק 1 עם רווח.mp4"
    _, jobs, _ = batch.plan_jobs(f, tmp_path / "out dir" / "x.mp4", recursive=False, clip=None)
    assert jobs[0].out == (tmp_path / "out dir" / "x.mp4").resolve()
    _, jobs, _ = batch.plan_jobs(f, None, recursive=False, clip=(1353.0, 5.0))
    assert jobs[0].out.name == "פרק 1 עם רווח.sample-1353s-5s.mp4"
    with pytest.raises(pipeline.UsageError):
        batch.plan_jobs(f, tmp_path / "somedir", recursive=False, clip=None)
    with pytest.raises(pipeline.UsageError):
        batch.plan_jobs(f, None, recursive=True, clip=None)


def test_plan_folder_defaults_structure_and_errors(tmp_path):
    touch(tmp_path / "2.flv")
    touch(tmp_path / "10.flv")
    touch(tmp_path / "s2" / "1.flv")
    mode, jobs, root = batch.plan_jobs(tmp_path, None, recursive=True, clip=None)
    assert mode == "folder" and root == tmp_path.resolve() / "enhanced"
    assert [j.out.relative_to(root).as_posix() for j in jobs] == ["2.mp4", "10.mp4", "s2/1.mp4"]
    _, jobs, root = batch.plan_jobs(tmp_path, tmp_path / "elsewhere", recursive=False, clip=None)
    assert root == (tmp_path / "elsewhere").resolve() and [j.src.name for j in jobs] == ["2.flv", "10.flv"]
    with pytest.raises(pipeline.UsageError):
        batch.plan_jobs(tmp_path, tmp_path / "x.mp4", recursive=False, clip=None)
    with pytest.raises(pipeline.UsageError):
        batch.plan_jobs(tmp_path, None, recursive=False, clip=(0, 5))


def test_preflight_collisions_case_insensitive_and_existing_outputs(tmp_path):
    a = batch.Job(1, tmp_path / "a" / "1.flv", tmp_path / "out" / "Ep.mp4", "a/1.flv")
    b = batch.Job(2, tmp_path / "b" / "1.flv", tmp_path / "OUT" / "ep.MP4", "b/1.flv")
    assert any("would both write" in p for p in batch.preflight([a, b], overwrite=False))
    existing = touch(tmp_path / "out2" / "1.mp4", b"old")
    c = batch.Job(1, tmp_path / "1.flv", existing, "1.flv")
    assert batch.preflight([c], overwrite=False) == [] and c.status == "skipped-existing"
    d = batch.Job(1, tmp_path / "1.flv", existing, "1.flv")
    assert batch.preflight([d], overwrite=True) == [] and d.status == "pending"


# ---------------------------------------------------------------- batch loop (mocked enhance)

def _mock_settings():
    return pipeline.Settings(profile="ai_sharpen0.5", sharpen=0.5)


def _jobs(tmp_path, n):
    return [batch.Job(i, tmp_path / f"{i}.flv", tmp_path / "enhanced" / f"{i}.mp4", f"{i}.flv") for i in range(1, n + 1)]


def _fake_enhance(fail_on: set[int]):
    calls = []

    def enhance(ts, src, out, s, overwrite, log, clip=None, extra_report=None):
        idx = int(Path(src).stem)
        calls.append(idx)
        if idx in fail_on:
            raise ToolError(f"synthetic failure for {idx}")
        return {"output_path": str(out), "speed": {"frames": 10, "video2x_fps": 1.0}}
    return enhance, calls


def test_batch_stops_on_first_failure(tmp_path):
    jobs = _jobs(tmp_path, 4)
    enhance, calls = _fake_enhance({2})
    import os
    summary = batch.run_jobs(None, jobs, _mock_settings(), {}, overwrite=False, continue_on_error=False, clip=None,
                             enhance=enhance, out=open(os.devnull, "w"))
    assert calls == [1, 2]
    assert [j.status for j in jobs] == ["converted", "failed", "not-started", "not-started"]
    assert summary["counts"] == {"converted": 1, "failed": 1, "not-started": 2}
    assert json.loads((tmp_path / "enhanced" / "2.failed.report.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_batch_continue_on_error_still_reports_failure(tmp_path):
    jobs = _jobs(tmp_path, 4)
    jobs[3].status = "skipped-existing"
    enhance, calls = _fake_enhance({2})
    import os
    summary = batch.run_jobs(None, jobs, _mock_settings(), {}, overwrite=False, continue_on_error=True, clip=None,
                             enhance=enhance, out=open(os.devnull, "w"))
    assert calls == [1, 2, 3]
    assert summary["counts"] == {"converted": 2, "failed": 1, "skipped-existing": 1}


def test_profiles_and_overrides(monkeypatch):
    ts = tools.Toolset(Path("v"), Path("f"), Path("p"))
    monkeypatch.setattr(pipeline, "pick_gpu", lambda ts: (0, "Fake GPU"))
    s, meta = batch.resolve_settings("ai_sharpen0.5", sharpen=None, scale=None, crf=None, preset=None, gpu=None, ts=ts,
                                     timestamp_policy="repair-terminal", temp_dir=None, keep_temp=False, dry_run=True,
                                     allow_interlaced=False)
    assert (s.upscaler, s.model, s.scale, s.sharpen, s.crf, s.preset) == ("ai", "realesr-animevideov3", 2, 0.5, 18, "slow")
    assert meta["overrides"] == {} and "auto-selected" in meta["gpu"]
    # The default chain must be byte-for-byte the tested preview's ai_sharp filter.
    assert pipeline.encode_filter(s, Fraction(1), width=1280, height=960, sharpen=s.sharpen) == (
        "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.500:chroma_msize_x=5:chroma_msize_y=5:chroma_amount=0,"
        "setsar=1/1,format=yuv420p")
    s, meta = batch.resolve_settings("ai_sharpen0.5", sharpen=0.0, scale=None, crf=20, preset="medium", gpu=3, ts=ts,
                                     timestamp_policy="strict", temp_dir=None, keep_temp=False, dry_run=True,
                                     allow_interlaced=False)
    assert (s.sharpen, s.crf, s.preset, s.gpu) == (0.0, 20, "medium", 3)
    assert set(meta["overrides"]) == {"sharpen", "crf", "preset"}
    s, _ = batch.resolve_settings("baseline", sharpen=None, scale=None, crf=None, preset=None, gpu=None, ts=ts,
                                  timestamp_policy="strict", temp_dir=None, keep_temp=False, dry_run=True,
                                  allow_interlaced=False)
    assert s.upscaler == "lanczos" and pipeline.encode_filter(s, Fraction(1), width=4, height=4, sharpen=0).startswith(
        "setpts=N/(25*TB),scale=4:4:flags=lanczos,")


# ---------------------------------------------------------------- timestamp repair (pure)

def scan(ts, tb=Fraction(1, 1000), durs=None):
    return probe.FrameScan("decoded-frames", tb, list(ts), durs or [40] * len(ts))


BASE = [i * 40 for i in range(100)]  # regular 25 fps history; anchor = BASE[-1] = 3960
R25 = [Fraction(25), Fraction(25)]


def repair(ts, policy="repair-terminal", tb=Fraction(1, 1000), nominal=R25):
    return probe.repair_terminal_suffix(scan(ts, tb), policy, nominal)


# Every pattern observed in the 61 episodes, as slot offsets after the anchor (see README).
@pytest.mark.parametrize("name,suffix_offsets,episodes,end_change_ms", [
    ("final duplicate", [0], "1 4 6 9 12 15 22 23 24", 40),
    ("dup, +1, +3", [0, 1, 3], "26 28 37 40 44 47 55 57 58 59 61", 0),
    ("dup, +2", [0, 2], "29 30 32 33 38 39 41 42 43 45 48 50 52 53 60", 0),
    ("final one slot back", [-1], "31 34 36 46 49 51 56", 40),
    ("back one, then 0, +1", [-1, 0, 1], "27 35", 80),
    ("back one, then 0", [-1, 0], "54", 80),
])
def test_observed_terminal_patterns_repaired(name, suffix_offsets, episodes, end_change_ms):
    ts = BASE + [BASE[-1] + 40 * o for o in suffix_offsets]
    fixed, rec = repair(ts)
    m = len(suffix_offsets)
    assert len(fixed.ts) == len(ts), "every frame kept"
    assert fixed.ts[:-m] == ts[:-m], "timestamps through the anchor unchanged"
    assert fixed.ts[-m:] == [BASE[-1] + 40 * (j + 1) for j in range(m)], "consecutive slots after the anchor"
    assert rec["anchor_index"] == len(BASE) - 1 and rec["suffix_length"] == m
    assert [f["original_pts_ticks"] for f in rec["frames"]] == ts[-m:]
    assert [f["repaired_pts_ticks"] for f in rec["frames"]] == fixed.ts[-m:]
    assert rec["inferred_interval_s"] == "1/25" and rec["inferred_rate_fps"] == "25/1"
    assert round(rec["video_duration_change_s"] * 1000, 6) == end_change_ms
    # Only timestamps that cannot keep their value change; a final frame already on its slot is kept.
    for j, f in enumerate(rec["frames"]):
        assert f["changed"] == (suffix_offsets[j] != j + 1), f
        assert f["changed"] == (suffix_offsets[j] < j + 1), "every change is forced by order preservation"
        assert f["reason"]


def test_valid_terminal_sequences_unchanged():
    for ts in (BASE, BASE + [4000], BASE + [4400]):  # regular end, and a genuinely long last interval
        fixed, rec = repair(ts)
        assert rec is None and fixed.ts == ts


def test_interval_inferred_not_hardcoded():
    ntsc = [round(i * 1001 / 24) for i in range(60)]  # 23.976 fps in 1 ms ticks
    fixed, rec = repair(ntsc + [ntsc[-1]], nominal=[None, None])
    assert rec["inferred_rate_fps"] == "24000/1001"
    assert abs(fixed.ts[-1] - 60 * 1001 / 24) <= 1 and fixed.ts[-1] > ntsc[-1]
    t90 = [i * 3600 for i in range(30)]  # 25 fps in a 90 kHz time base, pattern [0, 2]
    fixed, rec = repair(t90 + [t90[-1], t90[-1] + 7200], tb=Fraction(1, 90000), nominal=[None, None])
    assert fixed.ts[-2:] == [30 * 3600, 31 * 3600] and rec["inferred_interval_s"] == "1/25"
    t30 = [i * 1001 for i in range(40)]  # 30000/1001 fps in a 1/30000 time base, pattern [-1, 0, 1]
    fixed, rec = repair(t30 + [t30[-1] - 1001, t30[-1], t30[-1] + 1001], tb=Fraction(1, 30000), nominal=[None, None])
    assert rec["inferred_rate_fps"] == "30000/1001" and fixed.ts[-3:] == [40 * 1001, 41 * 1001, 42 * 1001]


def test_strict_rejects_terminal_defects():
    for suffix in ([0], [0, 1, 3], [-1]):
        with pytest.raises(probe.TimingDefect, match="strict rejects"):
            repair(BASE + [BASE[-1] + 40 * o for o in suffix], policy="strict")


@pytest.mark.parametrize("ts,match", [
    (BASE[:50] + [BASE[49]] + BASE[50:], "Unsupported"),                       # interior duplicate
    (BASE[:50] + [BASE[48]] + BASE[51:], "Unsupported"),                       # interior backwards
    (BASE + [3960, 4000, 4040, 4080], "Unsupported"),                          # suffix of 4 > maximum 3
    (BASE + [3950], "not on the"),                                             # off-grid stamp
    (BASE + [3880], "outside"),                                                # two slots back: not observed
    (BASE + [3960, 4200], "outside"),                                          # points far ahead
    # Nearby shapes that fit the grid and shift bounds but were never observed: rejected by signature.
    (BASE + [3960, 4000, 4040], "not one of the recognised"),                  # [0, 1, 2]
    (BASE + [3960, 3960, 4080], "not one of the recognised"),                  # [0, 0, 3]
    (BASE + [3960, 3960], "not one of the recognised"),                        # [0, 0]
    (BASE + [3920, 4000], "not one of the recognised"),                        # [-1, 1]
    (BASE + [3920, 3960, 4040], "not one of the recognised"),                  # [-1, 0, 2]
    (BASE + [3960, 4000, 4120], "outside"),                                    # [0, 1, 4]: final beyond slot +3
    (BASE[:90] + [3600, 3655, 3680, 3720, 3790, 3800, 3840, 3880, 3920, 3960, 3960], "not a constant rate"),
    ([0, 40, 80, 120, 160, 160], "fewer than"),                                # too little history
])
def test_unrecognised_or_interior_defects_rejected(ts, match):
    with pytest.raises(probe.TimingDefect, match=match):
        repair(ts)


def test_missing_timestamps_rejected():
    s = scan(BASE)
    s.missing_ts = 2
    with pytest.raises(probe.TimingDefect, match="no timestamp"):
        probe.repair_terminal_suffix(s, "repair-terminal", R25)


def test_legacy_name_still_available():
    assert probe.repair_terminal_duplicate is probe.repair_terminal_suffix



# ---------------------------------------------------------------- real FFprobe on generated sources, no Video2X

@pytest.mark.ffmpeg
def test_decoded_scan_sees_real_pts_not_best_effort(sources):
    tb = Fraction(1, 1000)
    sc = probe.scan_frames(tools.BUNDLED_FFPROBE, sources["interior_dup"], 0, decode=True, time_base=tb,
                           show_progress=False)
    with pytest.raises(probe.TimingDefect, match="frame #25"):
        probe.repair_terminal_duplicate(sc, "repair-terminal", [Fraction(25), None])
    sc = probe.scan_frames(tools.BUNDLED_FFPROBE, sources["terminal_dup"], 0, decode=True, time_base=tb,
                           show_progress=False)
    fixed, rec = probe.repair_terminal_suffix(sc, "repair-terminal", [Fraction(25), None])
    f = rec["frames"]
    assert (rec["anchor_index"], f[0]["index"], f[0]["original_pts_ticks"], f[0]["repaired_pts_ticks"]) == (48, 49, 2000, 2040)
    for key, orig, want in (("tail_0_1_3", [2320, 2360, 2440], [2360, 2400, 2440]),
                            ("tail_m1_0_1", [2280, 2320, 2360], [2360, 2400, 2440])):
        sc = probe.scan_frames(tools.BUNDLED_FFPROBE, sources[key], 0, decode=True, time_base=tb, show_progress=False)
        fixed, rec = probe.repair_terminal_suffix(sc, "repair-terminal", [Fraction(25), None])
        assert [x["original_pts_ticks"] for x in rec["frames"]] == orig, key
        assert fixed.ts[-3:] == want and fixed.ts[:-3] == sc.ts[:-3], key


@pytest.mark.ffmpeg
def test_dry_run_preflight_uses_decoded_tail_order(sources):
    # Sorting packet PTS would turn [-1, 0, 1] into a duplicate pattern; the preflight must see decoder order.
    from animenhance.procs import RunLog
    src = pipeline.analyze_source(tools.resolve_tools(), sources["tail_m1_0_1"], decode_scan=False, log=RunLog(),
                                  run_idet=False, decode_audio=False, timestamp_policy="repair-terminal")
    assert src.original_scan.source == "packets+decoded-tail"
    assert [f["original_pts_ticks"] for f in src.timing_repair["frames"]] == [2280, 2320, 2360]


@pytest.mark.ffmpeg
def test_convert_rejects_interior_defect_and_strict_terminal(sources, tmp_path):
    cp = run_convert(sources["interior_dup"], "--output", tmp_path / "i.mp4")
    assert cp.returncode != 0 and "Unsupported timestamp defect" in cp.stdout + cp.stderr
    assert not (tmp_path / "i.mp4").exists()
    cp = run_convert(sources["terminal_dup"], "--output", tmp_path / "t.mp4", "--timestamp-policy", "strict")
    assert cp.returncode != 0 and "strict rejects" in cp.stdout + cp.stderr
    assert not (tmp_path / "t.mp4").exists()


@pytest.mark.ffmpeg
def test_convert_dry_run_folder_with_hebrew_and_spaces(sources, tmp_path):
    folder = tmp_path / "תיקייה עם רווחים"
    folder.mkdir()
    import shutil
    shutil.copyfile(sources["cfr_audio"], folder / "פרק 10.FLV")
    shutil.copyfile(sources["terminal_dup"], folder / "פרק 2.flv")
    (folder / "readme.txt").write_text("x", encoding="utf-8")
    cp = run_convert(folder, "--dry-run")
    assert cp.returncode == 0, cp.stdout + cp.stderr
    text = cp.stdout
    assert text.index("פרק 2.flv") < text.index("פרק 10.FLV")
    assert "timestamp repair: last 1 frame(s) after #48" in text and "planned=2" in text
    assert not (folder / "enhanced").exists(), "dry run must not create outputs"


@pytest.mark.ffmpeg
def test_convert_skips_existing_without_overwrite(sources, tmp_path):
    import shutil
    folder = tmp_path / "eps"
    folder.mkdir()
    shutil.copyfile(sources["cfr_audio"], folder / "1.flv")
    existing = folder / "enhanced" / "1.mp4"
    existing.parent.mkdir()
    existing.write_bytes(b"previous output")
    cp = run_convert(folder)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "SKIPPED" in cp.stdout and "skipped-existing=1" in cp.stdout
    assert existing.read_bytes() == b"previous output"
    summary = json.loads(next((folder / "enhanced").glob("batch_summary_*.json")).read_text(encoding="utf-8"))
    assert summary["jobs"][0]["status"] == "skipped-existing"


@pytest.mark.ffmpeg
def test_storage_preflight_refuses_when_total_output_does_not_fit(sources, tmp_path, monkeypatch):
    import shutil
    import convert
    folder = tmp_path / "eps"
    folder.mkdir()
    for name in ("1.flv", "2.flv"):
        shutil.copyfile(sources["cfr_audio"], folder / name)
    ts = tools.resolve_tools()
    _, jobs, root = batch.plan_jobs(folder, None, recursive=False, clip=None)
    s, _ = batch.resolve_settings("ai_sharpen0.5", sharpen=None, scale=None, crf=None, preset=None, gpu=0, ts=ts,
                                  timestamp_policy="repair-terminal", temp_dir=None, keep_temp=False, dry_run=False,
                                  allow_interlaced=False)
    plan = batch.storage_plan(ts, jobs, s, root)
    assert plan["pending_files"] == 2 and plan["estimated_total_output_bytes"] > 0 and plan["fits"]
    s_file, _ = batch.resolve_settings("ai_sharpen0.5", sharpen=None, scale=None, crf=None, preset=None, gpu=0, ts=ts,
                                       timestamp_policy="repair-terminal", temp_dir=None, keep_temp=False,
                                       dry_run=False, allow_interlaced=False, file_intermediate=True)
    plan_file = batch.storage_plan(ts, jobs, s_file, root)
    assert plan_file["estimated_peak_temp_bytes"] > plan["estimated_peak_temp_bytes"], "streaming avoids the FFV1 file"
    monkeypatch.setattr(pipeline, "free_bytes", lambda p: 1024 ** 2)
    with pytest.raises(pipeline.UsageError, match="insufficient"):
        convert.run(convert.build_parser().parse_args([str(folder)]))
    assert not (folder / "enhanced").exists(), "nothing may be created when the total does not fit"


def test_convert_cli_argument_errors(sources, tmp_path):
    for args in ([sources["cfr_audio"], "--start", "5"],
                 [sources["cfr_audio"], "--recursive"],
                 [sources["cfr_audio"].parent, "--start", "1", "--duration", "2"],
                 [sources["cfr_audio"], "--scale", "5", "--dry-run"],
                 [tmp_path / "missing.flv"]):
        cp = run_convert(*args)
        assert cp.returncode == 2, (args, cp.stdout, cp.stderr)
    cp = run_convert("--help")
    assert cp.returncode == 0 and "--timestamp-policy" in cp.stdout and "ai_sharpen0.5" in cp.stdout
