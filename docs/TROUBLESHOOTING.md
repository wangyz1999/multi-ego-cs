# Troubleshooting

Failures we actually hit, and what they mean.

---

## Stage 01 · discover

**`No FACEIT API key`** — export `MECS_FACEIT_API_KEY`. Create one free at
<https://developers.faceit.com> with the Data API scope.

**Only a handful of matches accepted.** Read the rejection histogram the stage
prints on exit:

```
Rejection reasons: {'map': 412, 'no-demo-url': 88, 'rounds': 31, 'status': 12}
```

- `map` dominating → your `discover.maps` allowlist is the bottleneck.
- `no-demo-url` → those matches never had a demo published; unavoidable.
- `rounds` → `min_rounds` is filtering forfeits, which is usually correct.

Raise `num_players` before anything else; it is the widest lever.

**Constant 429s.** Lower `discover.requests_per_minute`. The limiter is a
sliding window shared across threads, but FACEIT's own limit varies by key.

---

## Stage 02 · download

**Everything fails with DNS errors.**

```
24 matches failed with DNS errors. The CDN host does not resolve from this
network - this is a connectivity problem, NOT demo expiry.
```

This is the single most important distinction in the pipeline. The stage
checks DNS *before* the first request precisely so this cannot be misread as
mass expiry. Institutional networks and HPC clusters frequently cannot resolve
FACEIT's CDN host. Retry from a network with unrestricted outbound DNS. Do not
conclude the demos are gone.

**HTTP 403/404/410.** These genuinely are gone — FACEIT's retention window has
passed. Nothing recovers them. The affected matches can still contribute video;
they will show up in `summary.json` under `matches_without_demo` and their
clips carry `has_actions = 0`.

**`Cannot decompress .dem.zst`.** Install `zstandard`, or make the `zstd`
binary available (`module load zstd` on HPC).

---

## Stage 03 · metadata / Stage 05 · actions

**Worker processes killed, job OOMs.** awpy holds a full tick table per demo —
4–8 GB for a long match. Lower `--cpus-per-task` rather than raising `--mem`;
workers scale memory linearly.

**`expected 10 players, found 9`.** Recorded in `validation.problems`, the
match is flagged `usable: false`, and stage 07 excludes it unless
`--include-unusable`. This is a report, not a crash — one bad demo must not
kill a 12-hour job.

**Deaths appear to be missing.** Do not "fix" the death join to use Steam IDs.
The kill feed stores them as float64 and Steam64 ids exceed float64's exact
integer range (2^53), so some round to neighbouring values and never join.
Matching on player name is deliberate and correct within a match.

**A re-run reprocesses everything.** `_round_is_complete()` compares *columns*,
not just file existence. If you changed the schema — added a `k_*` column,
toggled `keep_raw_buttons` — every existing file is correctly invalidated.

---

## Stage 04 · record (Windows)

**`Stage 04 ... only runs on Windows`.** Correct. Use `--dry-run` on Linux to
preview the capture plan, and run the real thing on the gaming machine.

**`record.capture_output_dir is unset`.** Point it at your capture tool's
output folder. Nothing else can find the finished clips.

**`No finished clip appeared`.** The capture hotkey did not fire, or the tool
writes somewhere else. Check that `record.capture_hotkey` matches the tool's
binding and that CS2 has focus — `pyautogui` sends keystrokes to the focused
window, so anything stealing focus breaks capture.

**Clips start mid-action.** The `demo_gototick → demo_pause → demo_gototick →
demo_resume` sequence exists because seeking a *playing* demo overshoots.
If you shortened it, restore it.

**Every frame has a wall-hack outline.** `spec_show_xray 0` did not take.
It is issued once per demo after `playdemo`; if the demo was reloaded after a
crash, make sure the recovery path re-issues it.

**A session died mid-match.** Just re-run. `record_progress.tsv` is fsynced
after every clip; the session resumes at the next unrecorded one.

---

## Stage 06 · align

**A whole match fails, every round, `0/10 clips aligned`.**

Almost certainly a resolution mismatch. Check the actual capture:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate -of default=noprint_wrappers=1 clip.mp4
```

CS2 scales its HUD with screen **height** and anchors the top bar to the
horizontal **centre** — it does not stretch with width. `scale_boxes()` models
exactly that, so 1024×720, 1920×1080 and 2560×1440 all work from the same
1280×720 reference boxes.

We hit this: two matches in a 101-match collection were captured at 1024×720
instead of 1280×720. With naive linear-width scaling the timer read at
correlation **0.249** (garbage); with the centre-anchored model, **0.989**.
If you are on an older checkout that scales x by width, that is the bug.

**Scattered single-round failures.** Raise `align.max_scan_seconds`. The stage
needs to see one whole-second boundary, and a clip that starts just after a
tick has to wait almost a full second for the next.

**Diagnose a specific clip** with per-frame timer readings and scores:

```bash
mecs align -v --force -m <match_id>
```

**Large `round_offset_spread_sec`.** All ten clips were started by the same
automation, so they should agree to ~50 ms. A large spread means at least one
misread; check the `outlier` flags in that round's `offsets.json`.

**Offsets are negative.** Expected. The round goes live slightly before the
capture tool's first frame, so `game_sec = 0` sits before the clip starts.
Typical values are −0.7 to −3 s.

---

## Stage 07 · package

**Matches are missing from the release.** Two causes, both reported:
`excluded_unusable` (failed stage-03 validation — pass `--include-unusable`),
and matches with neither metadata nor video.

**The release tree copied hundreds of GB.** Hard links only work within one
filesystem. If the release lives on scratch and the sources on project storage,
`link_mode: auto` falls back to a **symlink**, which costs nothing. Use
`link_mode: copy` only when the release must be self-contained — tarred,
moved, or outliving its sources.

**Video-only matches have null tick columns.** Correct. Those matches never had
a demo; the structure comes from the video tree and the ticks do not exist.

---

## Stage 08 · publish

**`Refusing to publish without confirmation`.** Add `--yes`. Use `--dry-run`
first to see the file list and byte count.

**`No Hugging Face token`.** Either export `HF_TOKEN`, or run `hf auth login`
once — the stage falls back to the stored credential.

**Token rejected.** It needs **write** scope. Check with `hf auth whoami`.

**The upload died halfway.** Re-run the same command. The stage diffs against
`list_repo_files` and skips what is already there, so a resubmit resumes.
Lower `files_per_commit` if commits themselves are timing out.

**`HF_TOKEN is not set in the job environment`.** SLURM does not inherit your
login shell. Export it at submit time:

```bash
HF_TOKEN=... sbatch --export=ALL slurm/08_publish.sbatch --yes
```

---

## Environment

**`blas_thread_init: pthread_create failed` / `Resource temporarily unavailable`.**
numpy/OpenBLAS and OpenCV each spawn one thread per visible core. On a 64-core
node with 16 workers that is 1000+ threads, and on a shared login node it trips
`RLIMIT_NPROC` outright. Cap them:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 OPENCV_FFMPEG_THREADS=1
```

`slurm/_common.sh` does this for you. This also manifests as
`swscaler: Failed initializing scaling graph` — video decoding silently
returning unusable frames — which looks like an alignment bug and is not one.

**`ImportError: libGL.so.1`.** Install `opencv-python-headless`, not
`opencv-python`. Headless nodes have no OpenGL.

**`uv` panics with `failed to initialize global rayon pool`.** Same process
limit. Install from a compute node, or cap uv's concurrency.

**Jobs sit in `ReqNodeNotAvail, Reserved for maintenance`.** A cluster
reservation is active and your job would overrun its start. Check
`scontrol show reservation`, then either shorten `--time` to fit before the
window or wait it out — queued jobs start automatically when it lifts.
