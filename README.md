# FLV animation → enhanced MP4

`enhance_animation.py` is a command-line pipeline that upscales animation with Real-ESRGAN
(`realesr-animevideov3`) through the Video2X native Vulkan/ncnn backend. FFmpeg and FFprobe
handle inspection, muxing and validation. PyTorch and BasicSR are not used.

## Pinned versions (used on the development PC)

| Component | Version / revision | Location |
|---|---|---|
| Video2X CLI (official prebuilt release) | 6.4.0, asset SHA-256 `0337B6DC…F793` | `third_party/video2x-6.4.0/` |
| Video2X source (read for behaviour) | tag 6.4.0, commit `a96bda9b4d79616cc6b71b94e6945146b5b4d509` | `third_party/video2x-src/` |
| FFmpeg / FFprobe | 9.0.2 gyan.dev essentials build | `third_party/ffmpeg-9.0.2-essentials_build/bin/` |
| Model | `realesr-animevideov3` x2/x3/x4 (bundled with Video2X 6.4.0) | `third_party/video2x-6.4.0/models/realesrgan/` |
| Python packages | av 19.0.2 (FFmpeg 9.0.2 libraries), numpy 2.5.4, pytest 9.1.1 | `.venv/` |

`third_party/VERSIONS.json` records the download URLs, SHA-256 hashes and model hashes. The model
weights are byte-identical to the official Real-ESRGAN v0.2.5.0 ncnn release
(`realesrgan-ncnn-vulkan-20220424-windows.zip`). The `.param` files differ only in line endings.

Hardware tested: Windows 10 22H2 (19045), Intel Core i5-7200U, **Intel HD Graphics 620**
(Vulkan 1.3.215, driver 31.0.101.2135), 16 GB RAM. This PC has no NVIDIA GPU, so there is no CUDA.
Video2X lists the Intel GPU as Vulkan device 0, and `doctor` runs real inference on it.

## Installation (Windows, PowerShell)

The repository contains only the code and the version manifest `third_party/VERSIONS.json`. Executables, model weights, downloads and the Video2X source checkout are not committed. To fetch and verify the pinned versions and create the virtual environment:

```powershell
git clone git@github.com:maorkehati/flv2mp4.git
cd flv2mp4
powershell -ExecutionPolicy Bypass -File scripts\setup_tools.ps1      # add -SkipSource to skip the Video2X source clone
.\.venv\Scripts\python.exe enhance_animation.py doctor                 # tools, versions, GPU, models, tiny real inference
```

`setup_tools.ps1` downloads the Video2X 6.4.0 release zip and the FFmpeg 9.0.2 essentials zip into `third_party\downloads`. It checks both against the SHA-256 values in `VERSIONS.json` and extracts them. It also checks the bundled `realesr-animevideov3` x2/x3/x4 model files, optionally clones the Video2X source at tag 6.4.0 (reference only) and checks its commit, then creates `.venv` from `requirements.txt`. The script was run in verification mode against existing downloads on the development PC; the fresh-download and fresh-clone branches have not been exercised on a clean machine.

The manual equivalent:

```powershell
Invoke-WebRequest https://github.com/k4yt3x/video2x/releases/download/6.4.0/video2x-windows-amd64.zip -OutFile v2x.zip
Expand-Archive v2x.zip third_party\video2x-6.4.0
Invoke-WebRequest https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.zip -OutFile ff.zip
Expand-Archive ff.zip third_party
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Sample episode.** `samples/1.FLV` is the original Avatar episode 1 FLV: 141,239,386 bytes, SHA-256 `DDCF750FDC242B7AD23188811D76EDB0A62D0F7D172839BDEF12C3E876FC446D`. It exceeds GitHub's 100 MiB file limit, so it is stored with Git LFS. Install Git LFS (`git lfs install`) before cloning, or run `git lfs pull` afterwards, to get the video instead of a pointer file. Example: `.\.venv\Scripts\python.exe convert.py samples\1.FLV --start 00:01:53 --duration 20`.

No third-party code is modified. The wrapper works around Video2X 6.4.0's behaviour from the outside (see "Backend behaviour" below), so there are no patches to apply.

Other installations can be used with `--video2x`, `--ffmpeg` and `--ffprobe`. Without those flags,
the tool uses the `third_party` copies first and then `PATH`. Video2X needs a Vulkan-capable GPU and driver.

## Converting episodes with `convert.py`

`convert.py` takes one FLV file or a folder of FLV files. It converts immediately with the default
profile and never asks questions. Run these from `C:\Users\mayoa\flv2mp4` in PowerShell:

```powershell
cd C:\Users\mayoa\flv2mp4
$py = ".\.venv\Scripts\python.exe"

# 1. Convert episode 1  ->  C:\Users\mayoa\Desktop\avatar\enhanced\1.mp4
& $py convert.py "C:\Users\mayoa\Desktop\avatar\1.FLV"

# 2. Convert the whole folder (1, 2, ..., 10, ... in numeric order)  ->  C:\Users\mayoa\Desktop\avatar\enhanced\
& $py convert.py "C:\Users\mayoa\Desktop\avatar"

# 3. Choose another output folder (folder input) or output file (single file)
& $py convert.py "C:\Users\mayoa\Desktop\avatar" --output "D:\Avatar enhanced"
& $py convert.py "C:\Users\mayoa\Desktop\avatar\1.FLV" --output "C:\Users\mayoa\Videos\Avatar 1.mp4"

# 4. Disable the additional sharpening (AI upscale only)
& $py convert.py "C:\Users\mayoa\Desktop\avatar\1.FLV" --sharpen 0

# 5. A 20-second sample  ->  ...\enhanced\1.sample-113s-20s.mp4
& $py convert.py "C:\Users\mayoa\Desktop\avatar\1.FLV" --start 00:01:53 --duration 20

# 6. Dry run: inspect and show the resolved settings and planned outputs; nothing is processed or created
& $py convert.py "C:\Users\mayoa\Desktop\avatar" --dry-run
```

Other options:

- `--profile ai_sharpen0.5|ai|baseline`
- `--scale 2|3|4`, `--crf N`, `--encoder-preset NAME`, `--gpu ID`
- `--recursive`, `--overwrite`, `--continue-on-error`
- `--timestamp-policy repair-terminal|strict`
- `--keep-temp`, `--temp-dir PATH`
- `--file-intermediate`, `--ignore-total-space`
- `--video2x/--ffmpeg/--ffprobe PATH`

Options you give explicitly override the profile's values. Use `python convert.py --help` for the full list.

**Profiles.** Each profile reproduces a preview variant that was rendered and validated on episode 1:

- `ai_sharpen0.5` (default): Real-ESRGAN `realesr-animevideov3` 2× through Video2X, whose lossless FFV1 output is streamed through a named pipe (or written to a temporary file with `--file-intermediate`). FFmpeg applies `unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.500:chroma_msize_x=5:chroma_msize_y=5:chroma_amount=0,setsar,format=yuv420p` during the only lossy encode (libx264 CRF 18, preset slow, High, yuv420p). This is the same filter string as the preview.
- `ai`: the same model without additional sharpening. Video2X encodes libx264 directly with the same CRF, preset and profile; this is the established `enhance` behaviour.
- `baseline`: FFmpeg Lanczos 2×, no AI.

All profiles keep the original frame timing (no interpolation) and the display aspect ratio. There is no denoising and no colour adjustment; colour tags are set so players don't shift colours. Output is MP4 with `+faststart`. AAC/MP3 audio is copied, other audio becomes AAC with the same channel layout, and text subtitles become `mov_text`.

**Deliberate differences from the tested preview** (also printed at start-up and saved in every report):

- Full conversions copy the episodes' AAC audio. The preview re-encoded AAC 192k only because it cut an interval; clip mode re-encodes exactly like the preview.
- Full conversions decode the FLV directly instead of a lossless FFV1 extract. H.264 decoding is bit-exact, so the upscaler receives identical pixels.

**Streaming and disk space.** With sharpening, Video2X's lossless FFV1 output does not go to a file. It is streamed through a Windows named pipe (`\\.\pipe\animenhance-….mkv`) into the FFmpeg `unsharp` + libx264 encode, which runs while inference continues. This was verified on real frames: the result is frame-for-frame identical to the file route (framemd5), and the encoder finishes within 0.2 s of Video2X. It saves the ~12 GiB intermediate per episode (364 KB/frame measured) and about an hour of serial encoding at x264's 9–12 fps. `--file-intermediate` restores the old file route.

The peak temporary space per episode is now roughly 2 GB: the stream copy, the encoded H.264 and its re-timed copy. Before it starts, `convert.py` adds up the estimated output of every pending file plus that peak. If the total doesn't fit on the output drive, it refuses to start and says how many files would fit. `--ignore-total-space` overrides this; each file is still checked before it starts. A watchdog stops a step if free space drops below 1 GiB.

**Batch behaviour:**

- One episode is processed at a time. The console shows `[i/N] name`, step progress with fps/ETA, the elapsed time per episode, and a batch ETA.
- Output-name collisions are checked case-insensitively before anything runs.
- Existing outputs are reported as *skipped* and not re-validated. `--overwrite` keeps the old file until its validated replacement is moved into place.
- Every file gets `<name>.report.json` and `<name>.log`, or `<name>.failed.report.json` and `<name>.failed.log` if it failed. Folders also get `batch_summary_<time>.json`.
- The exit code is 0 only if nothing failed; this holds with `--continue-on-error` too. A cancelled run exits 130.

**Timestamp policy.** In 45 of the 61 Avatar episodes, the last 1–3 frames (in decoder output order) carry duplicate or backwards presentation timestamps. Everything before them is a regular 25 fps grid. `scripts\diagnose_tails.py` and `scripts\preflight_decoded.py` found six patterns. Each is listed as the suffix stamps' slot offsets relative to the last reliable (anchor) frame:

| Pattern (offsets) | Episodes | Example, ms (anchor, then suffix) | After repair | Video end |
|---|---|---|---|---|
| `[0]` | 1 4 6 9 12 15 22 23 24 | ep1: 1352600, **1352600** | 1352640 | +40 ms |
| `[0, 1, 3]` | 26 28 37 40 44 47 55 57 58 59 61 | ep26: 1409120, **1409120, 1409160, 1409240** | 1409160, 1409200, 1409240 | 0 |
| `[0, 2]` | 29 30 32 33 38 39 41 42 43 45 48 50 52 53 60 | ep29: 1407280, **1407280, 1407360** | 1407320, 1407360 | 0 |
| `[-1]` | 31 34 36 46 49 51 56 | ep31: 1409200, **1409160** | 1409240 | +40 ms |
| `[-1, 0, 1]` | 27 35 | ep27: 1407080, **1407040, 1407080, 1407120** | 1407120, 1407160, 1407200 | +80 ms |
| `[-1, 0]` | 54 | ep54: 1405360, **1405320, 1405360** | 1405400, 1405440 | +80 ms |

With `--timestamp-policy repair-terminal` (the default for `convert.py`), the tool keeps every frame, in order and with its pixels. It re-times only that suffix, to the consecutive slots after the anchor. All earlier timestamps and all audio timestamps are unchanged. A source is recognised only if all of the following hold, and anything else fails with frame-level diagnostics:

1. Every decoded frame has a PTS. The tool reads each frame's own PTS, not FFmpeg's best-effort guess, which hides defects.
2. The first non-increasing PTS starts a suffix of at most **3** frames (the maximum observed), so interior defects are rejected.
3. The 24 intervals ending at the anchor follow one constant rate within one time-base tick. The rate is inferred with exact rational time-base arithmetic, not hard-coded.
4. Every original suffix stamp lies on that grid, between one slot before the anchor and the last reconstructed slot.
5. No frame moves earlier, and none moves more than **2** slots later (the maximum observed).
6. The suffix's slot-offset signature is one of the six observed in the table above. Nearby shapes such as `[0,1,2]`, `[0,0,3]`, `[0,0]`, `[-1,1]` or `[-1,0,2]` are rejected, even though they would fit the grid.

Every change is forced by order preservation. The j-th suffix frame (counting from 1) needs a slot of at least +j, so only frames stamped below that bound change. A final frame already on its slot is kept, as in `[0,1,3]` and `[0,2]`. The report gives each changed frame's reason.

The report lists every suffix frame's original and repaired PTS (`timestamp_repair.frames` and `timestamp_changes`). It also records the video end before and after the repair and the video-minus-audio end difference. Audio is never stretched, and `-shortest` is never used. Output validation uses the repaired timeline.

`--timestamp-policy strict` rejects all of these (it is the default for `enhance_animation.py`). A dry run reads packets for most of the file but decodes the last 12 seconds, so it sees real decoder order. A real run decodes every frame again and repeats the check.

**Diagnostics** are still in `enhance_animation.py`:

```powershell
& $py enhance_animation.py doctor
& $py enhance_animation.py inspect "C:\Users\mayoa\Desktop\avatar\1.FLV" --full-scan
& $py enhance_animation.py preview "C:\Users\mayoa\Desktop\avatar\1.FLV" --start 00:01:53 --duration 15 --output-dir output\previews\ep1
```

## Lower-level usage (`enhance_animation.py`)

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py enhance_animation.py doctor                       # tools, versions, GPU list, models, encoders, tiny real inference
& $py enhance_animation.py inspect input\movie.flv      # add --full-scan to decode every frame, --json for everything
& $py enhance_animation.py preview input\movie.flv --start 00:05:00 --duration 20 --scale 2 --output-dir previews
& $py enhance_animation.py enhance input\movie.flv --output output\movie_x2.mp4 --scale 2
& $py enhance_animation.py enhance input\movie.flv --output output\movie_x2.mp4 --dry-run
```

Options for `preview` and `enhance`:

- `--gpu` (default 0)
- `--model` (validated against the installed Video2X model list and its model files)
- `--scale` 2/3/4 (default 2)
- `--crf` (default 18)
- `--preset` (default slow)
- `--sharpen` 0–2 for `enhance` (default 0); `--preview-sharpen` for the preview variant (default 0.5)
- `--timing auto|source`
- `--allow-interlaced`
- `--temp-dir`, `--keep-temp`, `--dry-run`
- `--video2x`, `--ffmpeg`, `--ffprobe`
- `enhance` only: `--overwrite` (never applies to the source) and `--skip-decode-check`

Exit codes: 0 means success, 1 means a processing or validation failure, 2 means invalid arguments or an unsupported source, and 130 means the run was cancelled.

### What `enhance` does

1. Probes the source with FFprobe and decodes every frame timestamp. It checks interlacing (field order, frame flags and the `idet` filter), timestamp regularity, aspect ratio, colour tags and every stream. It validates arguments, the model, the GPU and free disk space before any expensive work starts.
2. Runs Video2X on the original FLV. Video2X streams decode → Real-ESRGAN → libx264 (CRF 18, slow, High profile, yuv420p) in one pass, with no image files and one lossy encode. With `--sharpen > 0`, Video2X writes lossless FFV1 instead. On Windows it streams through a named pipe, otherwise it goes to a temporary file. FFmpeg applies `unsharp` (luma only) during the single libx264 encode.
3. Restores the source frame timestamps on the encoded H.264 packets with PyAV. This is a packet copy, not a re-encode.
4. Muxes with FFmpeg. Video is stream-copied. The `h264_metadata` bitstream filter sets SAR and colour tags. AAC/MP3 audio is copied, and other audio is converted to AAC with the same channel layout. Text subtitles become `mov_text`. Stream language and default flags are kept. The output uses `+faststart`. `-shortest` is never used.
5. Validates a temporary file: dimensions, SAR/DAR, colour tags, frame count, per-frame timestamps against the source, audio start/end/offset per track, and a full error-checked decode. Only then is the file renamed to the requested name. A JSON report (`<output>.report.json`) and a log (`<output>.log`) are written next to the output.

`preview` decodes exactly the frames whose timestamps fall in `[start, start+duration)` into a
lossless FFV1 clip. It runs inference once and encodes three clips with identical encoder settings:
Lanczos baseline with no sharpening, AI, and AI+unsharp. Each clip gets audio cut to the same
interval. It also produces a labelled side-by-side at matching display size; when panels are
downscaled or pixels are non-square, it adds a 1:1-pixel centre crop comparison. It reports
inference fps and an estimated full-movie runtime.

## Backend behaviour found while testing (Video2X 6.4.0) and how the wrapper handles it

These come from reading the 6.4.0 source and from runs on this PC.

- **Timestamps are rebuilt as `frame_index / guessed_rate`** (`libvideo2x.cpp`, `encoder.cpp`). Variable frame timing and video start offsets would be lost. The wrapper never relies on Video2X timing. It rewrites every packet's PTS/DTS from the source's decoded frame timestamps. In `--timing auto` (the default), if every source timestamp is within one time-base tick of an exact rate such as 24000/1001, the output uses that exact grid. Otherwise the source ticks are reproduced exactly. The average frame rate is never used.
- **The decoder is not flushed at end of input.** H.264 with B-frame reordering lost its last 2 frames (72 → 70). The wrapper stream-copies the video into Matroska and appends `has_b_frames` copies of an IDR packet, which pushes the real frames out. It then verifies the frame count and checks frame alignment at the head and tail by image comparison.
- **The process exits with 0xC0000409 during teardown** on this PC, after it has written the trailer and logged success. The wrapper accepts that exit code only together with the success message, and the output must still pass the frame-count and decode checks. The report records a warning.
- **Colour conversion uses swscale's default BT.601 limited range in both directions**, and Video2X copies the source's (often missing) tags. Without tags, an SD source upscaled to HD would be shown with BT.709 colours. The wrapper tags the output with the source's colour metadata, or with the default a player would assume for the source's resolution.
- **x2 models are internally x4 followed by a bicubic 0.5 resize** (see the `.param` graph), so x2 inference costs the same as x4.
- Video2X looks up models relative to the current directory first, so the wrapper runs it with its own folder as the working directory and passes absolute paths. It runs without a console window so the CLI's pause/abort keys cannot be triggered by accident.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_unit.py tests\test_convert.py -v                 # no Video2X: pure/mocked + real FFmpeg/PyAV
.\.venv\Scripts\python.exe -m pytest tests\test_integration_gpu.py tests\test_convert_gpu.py -v  # REAL Video2X inference on the local GPU
```

`test_convert.py` covers:

- file and folder argument handling, natural ordering, and case-insensitive discovery
- recursive exclusion of output and temp folders
- collision preflight, skip without `--overwrite`, and batch failure reporting (mocked `enhance`)
- the cumulative storage preflight, which refuses before anything is created
- profile overrides, including the exact preview filter string
- the terminal-suffix repair for all six observed patterns: every frame kept, earlier stamps unchanged, consecutive slots, and only forced changes
- inferred intervals at 25 fps in a 1 ms base, 23.976 fps, 25 fps in a 90 kHz base and 29.97 fps
- strict rejection of the defect
- unchanged valid endings
- rejection of interior duplicates and backwards stamps, suffixes longer than 3 frames, off-grid or far-off stamps, nearby unobserved signatures, unstable preceding timing, too little history, and missing timestamps
- the dry run's decoded-tail preflight

`test_convert_gpu.py` runs real inference:

- the single-frame terminal-duplicate fixture and two multi-frame fixtures (`[0,1,3]`, and `[-1,0,1]` with PTS < DTS) as whole-file conversions: frames kept in order, earlier timing unchanged, suffix repaired, audio unchanged
- a Hebrew/space-named folder batch with one bad file under `--continue-on-error`, then skip and `--overwrite`
- clip mode
- the Lanczos baseline profile
- cancellation of a streamed conversion: Video2X and the FFmpeg encoder are both terminated, and no partial or temp files remain

All 72 tests passed on the development PC, an Intel HD 620 running Video2X 6.4.0, on 2026-10-24.

`tests/make_fixtures.py` generates these synthetic sources:

- CFR H.264 FLV with B-frames, AAC and a text subtitle
- FLV with irregular timestamps (nominal rate 1000/1, intervals 39–101 ms) and MP3 audio
- FLV with video starting 0.5 s after audio, the timeline starting at 2.0 s, SAR 4:3 and Nellymoser audio
- Sorenson FLV1 without audio
- interlaced H.264
- MKV with two audio tracks that have language and default flags, plus a subtitle
- a copy of the first source in a folder with spaces and a Japanese/accented file name
- FLVs with a final-frame duplicate timestamp and with an interior duplicate (rewritten with PyAV)
- FLVs whose last three frames carry `[0,1,3]` and `[-1,0,1]` stamps (FLV composition times patched directly)

Integration tests check every output independently of the tool's own report: dimensions, SAR/DAR,
moov before mdat, decoded frame count, per-frame presentation times against the source (≤ 1.1 ms),
audio start (≤ 2 ms) and end (≤ 2 ms + one AAC frame) per track, language/default flags, subtitle
count, a full decode with no errors, and frame-by-frame content matching. Each output frame must
resemble its own source frame more than either neighbour, which rules out dropped, duplicated or
interpolated frames. The tests also check that the source hash is unchanged, that the sharpen path
uses one lossy encode, the preview outputs, and cancellation (Ctrl+Break kills the Video2X child,
removes temp/partial files and writes `status: cancelled`).

## Status: what has and has not been verified

These have been run and verified:

- `doctor` with real GPU inference, `inspect`, and the episode-1 previews (two 15 s intervals, three variants each)
- a timestamp preflight that decodes every frame of all 61 Avatar episodes: 61 pass, 45 with a terminal repair and 16 unchanged
- short real `convert.py` clips, including one tail clip per observed timestamp pattern (episodes 1, 26, 29, 31, 27, 54), each verified independently
- the 72 automated tests

These have **not** been done yet:

- no full episode and no full folder has been converted
- the full-folder runtime (about two months on this GPU) and storage (about 33–45 GiB, more than C: had free) are estimates from measurements, not observed totals

## Helper scripts

These are in `scripts/`:

- `setup_tools.ps1`: pinned downloads, verification and venv setup
- `diagnose_tails.py`: decoded-order tail timestamps per episode
- `preflight_decoded.py`: the full decoded timestamp preflight for a folder
- `ts_anomaly.py`: locate timestamp defects in one file
- `bench_video2x.py` and `gpu_util_probe.py`: throughput, GPU and CPU measurements
- `verify_ep1_clips.py` and `verify_tail_clips.py`: independent checks of the test clips; they assume the Avatar paths on the development PC
- `rename_episodes.py`: one-off episode renaming with an undo log

## Limitations

- **Speed is limited by the GPU.** Measured with `scripts\bench_video2x.py` and `scripts\gpu_util_probe.py` on a 150-frame clip of episode 1:
  - Video2X uses Vulkan device 0, the Intel HD Graphics 620, with fp16 enabled.
  - Its 3D/compute engine is 89% busy, about 95% including other applications, while the Video2X process uses 4.6% of one CPU core and disk traffic is 1.5 MB/s.
  - Steady state is 0.42 fps (2.39 s per frame), and startup costs about 2 s.
  - `realesr-animevideov3` costs about 0.38 TFLOP per 640×480 frame (the x2 model runs the x4 network and then downsizes), which is about 0.19 TFLOP/s, roughly half of this GPU's fp32 peak.
  - Measured runs vary between 0.27 and 0.42 fps with background GPU load (browser, editor) and thermal state.
  - A full episode of 33,500–35,300 frames therefore takes about 23–26 h at 0.38–0.42 fps, plus about 5 minutes of inspection and validation. That is roughly 60–65 days for all 61 episodes.
  - Anime4K through libplacebo runs at 8.6 fps on the same clip (about 1.1 h per episode) but removes much less compression noise. It is not offered as a profile.
- There is no resume inside an episode: a crash or cancellation restarts that episode. Finished episodes are skipped when the folder is run again. The tools ask Windows not to sleep while they run.
- Deinterlacing and inverse telecine are not implemented. Coded-interlaced sources are refused unless you pass `--allow-interlaced`. Heuristic (`idet`-only) detections are warnings, because sharp synthetic patterns and line art trigger them.
- Only the first video stream is enhanced. Bitmap subtitles, data streams (FLV cue/script data) and attachments are excluded and listed in the report.
- Sources flagged full-range on a non-J pixel format may be clipped by Video2X's limited-range conversion; this is reported as an issue.
