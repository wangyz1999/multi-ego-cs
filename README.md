# multi-ego-cs

**An end-to-end pipeline for collecting synchronized multi-egocentric Counter-Strike 2 datasets.**

Ten players. One match. Ten simultaneous first-person recordings, each paired
with a 64 Hz stream of that player's exact keyboard, mouse and view-angle
inputs, all on a common clock.

This is the collection machinery behind [**X-Ego-CS**](https://huggingface.co/datasets/wangyz1999/X-EGO-CS)
([paper](https://arxiv.org/abs/2510.19150)). It is published as a pipeline, not
a one-off script, so that anyone can point it at FACEIT and build a comparable
corpus of their own — 100 matches or 1000.

## See it

**Ten synchronized points of view, one pistol round** — the grid is for display;
the dataset ships each POV separately.

[![Ten synchronized POVs from one CS2 round](https://huggingface.co/datasets/wangyz1999/X-EGO-CS/resolve/main/assets/multi-ego-sync-demo-pistol-poster.jpg)](https://huggingface.co/datasets/wangyz1999/X-EGO-CS/resolve/main/assets/multi-ego-sync-demo-pistol-h264.mp4)

**One player's video with their actual inputs overlaid** — what stage 05 and
stage 06 produce together: a 64 Hz action trace on the exact frame that
produced it.

https://github.com/user-attachments/assets/7f378edb-62e2-46b7-85be-c8e275b16654

```
 01 discover  ──▶ 02 download ──▶ 03 metadata ──▶ 04 record  ──┐
   FACEIT API       .dem files      rounds +        CS2 +      │
                                    alive windows   capture    │
                                          │                    │
                                          ▼                    ▼
                                    05 actions           (per-player
                                    64 Hz parquet         round clips)
                                          │                    │
                                          └────────┬───────────┘
                                                   ▼
                                             06 align
                                        HUD-timer matching
                                                   │
                                                   ▼
                                       07 package ──▶ 08 publish
                                       manifests        HF Hub
```

---

## Why this is not just "record some gameplay"

Three problems have to be solved together, and each one quietly ruins the
dataset if you get it wrong.

**1. The camera must belong to one player.** A dead player's spectator camera
follows their killer. Record whole rounds and a third of your "egocentric"
footage is silently someone else's viewpoint. Stage 03 computes each player's
exact alive window from the demo's kill feed, and stage 04 records precisely
that window — one clip is one player's life in one round.

**2. Video and actions do not share a clock.** Screen capture starts some
unpredictable fraction of a second after the demo resumes. Naively pairing
frame *i* with tick `start + i·(64/fps)` drifts by a few hundred milliseconds
per clip — enough to put a flick on the wrong frame and make any
action-prediction target subtly wrong. Stage 06 measures the true offset per
clip by reading the CS2 HUD round timer with template matching, which gives an
exact whole-second landmark visible in both timelines.

**3. Demos expire, recordings don't.** FACEIT retains CS2 demos for weeks, not
forever. Once a demo is gone, that match can never have action data — but its
video is still fine. Treating the two modalities as interchangeable produces a
dataset whose coverage nobody can state. Every manifest row here carries
`has_video` / `has_actions` / `has_align`, and `summary.json` reports the
totals.

---

## Install

```bash
git clone https://github.com/wangyz1999/multi-ego-cs
cd multi-ego-cs

# Linux / macOS / HPC — everything except recording
uv venv .venv && uv pip install --python .venv/bin/python -e '.[fast-upload]'

# On the Windows capture machine (stage 04 only)
pip install -e '.[record]'
```

Secrets come from the environment, never a config file:

```bash
export MECS_FACEIT_API_KEY=...   # stage 01 — free, https://developers.faceit.com
export HF_TOKEN=...              # stage 08 — needs *write* scope
```

## Run it

```bash
cp configs/default.yaml my.yaml      # edit data_root, target_matches, …

mecs --config my.yaml status         # what exists, what's next
mecs --config my.yaml discover       # 01  pick matches
mecs --config my.yaml download       # 02  fetch demos   ← do this early
mecs --config my.yaml metadata       # 03  rounds + alive windows
mecs --config my.yaml record         # 04  capture video   (Windows)
mecs --config my.yaml actions        # 05  64 Hz parquet
mecs --config my.yaml align          # 06  video ↔ tick offsets
mecs --config my.yaml package        # 07  build release/
mecs --config my.yaml publish --yes  # 08  upload
```

Or chain them: `mecs run --from metadata --through package`.

Every stage is **idempotent and resumable** — re-running skips finished work,
so a SLURM time limit or a mid-session reboot costs you the items in flight and
nothing else. `--force` recomputes, `--dry-run` shows the plan, `--match <id>`
scopes to one match.

## Collecting 1000 matches

The pipeline scales, but not every stage scales the same way, and one of them
is bound by wall-clock reality rather than compute:

| Stage | Cost per 1000 matches | Bound by |
|---|---|---|
| 01 discover | ~2–4 h | FACEIT rate limit |
| 02 download | ~3–6 h, ~350 GB | Network, **demo expiry** |
| 03 metadata | ~4 h on 16 cores | Memory (one tick table per worker) |
| **04 record** | **~700–1000 h** | **Real time — capture is 1×** |
| 05 actions | ~8 h on 16 cores | Memory |
| 06 align | ~1 h on 32 cores | CPU (decodes ~10 s per clip) |
| 07 package | minutes, or hours with transcodes | I/O |
| 08 publish | ~10–30 h | Upload bandwidth |

Stage 04 is the whole budget. Recording is real-time: a 24-round match with 10
players is ~45–60 minutes of capture, and no amount of hardware changes that on
a single machine. **1000 matches is roughly 6 machine-months, so shard it** —
`mecs record --match-file shard_03.txt` on each of N machines, pointed at a
shared `data_root`, is the intended pattern. Everything else here is a
background job.

Read [`docs/COLLECTING_AT_SCALE.md`](docs/COLLECTING_AT_SCALE.md) before you
start a large run. It covers sharding, the demo-expiry race, disk budgeting,
and what to do when a capture machine dies at match 340.

## What comes out

```
release/
  manifest/clips.csv        one row per (match, round, player) + coverage flags
  manifest/rounds.csv       one row per (match, round)
  manifest/matches.csv      one row per match
  manifest/summary.json     counts, per-map totals, coverage percentages
  metadata/<match>.json     rounds, kills, alive windows, validation
  video/match=…/round=…/<steamid>.mp4
  state_action/match=…/round=…/<steamid>.parquet
  align/match=…/round=…/offsets.json
```

Trajectory columns (64 Hz, one row per tick):

| Column | Meaning |
|---|---|
| `tick`, `tick_norm`, `game_sec` | raw tick, tick zeroed at round start, seconds |
| `x`, `y`, `z`, `health` | world state |
| `pitch`, `yaw` | absolute view angles |
| `delta_pitch`, `delta_yaw` | per-tick view change, ±180° wrap corrected |
| `usercmd_mouse_dx/dy` | raw mouse counts from the usercmd |
| `k_w`, `k_a`, `k_mouse_left`, … | 25 binary input columns |
| `buttons` | raw engine input names, for bits this schema doesn't name |

To line video up with ticks, read the offset from `align/…/offsets.json`:

```python
video_time = game_sec + offset_sec      # the only convention used anywhere here
frame_index = round(video_time * video_fps)
```

## Alignment, measured

Stage 06's timer reader was validated against real 1280×720 / 30 fps captures:

```
player          start  trans  frame    offset   score
8064353169       1:55   1:54      7    -0.768   0.984
8089583895       1:55   1:54      8    -0.735   0.989
8131414911       1:55   1:54      7    -0.768   0.987
8145303401       1:55   1:54      6    -0.801   0.985
8183003671       1:55   1:54      8    -0.735   0.980
…
10/10 aligned   median -0.768 s   stdev 0.048 s   0 frames rejected
```

Digit-match confidence runs 0.98–0.99, and the ten independently-measured
offsets within a round agree to ~50 ms — which is the cross-check that matters,
since all ten clips were started by the same automation and *should* land
together. Throughput is ~0.7 s per round.

Digit crop boxes are configured in reference 1280×720 geometry and scaled to
whatever the capture actually is, so other resolutions work without new
constants. If a round fails to align, `mecs align -v` prints per-frame timer
readings and scores.

## Layout

```
src/multi_ego_cs/
  cli.py            the `mecs` entry point
  config.py         typed YAML config + env overrides
  paths.py          the on-disk layout, defined once
  stages/
    faceit_api.py   rate-limited Data API client + payload readers
    discover.py     01
    download.py     02
    metadata.py     03
    record.py       04  (Windows)
    buttons.py          CS2 input bitfield → named columns
    actions.py      05
    hud_timer.py        template-matching timer reader
    align.py        06
    package.py      07
    publish.py      08
  util/             logging, atomic IO, match-id handling
slurm/              ready-to-submit SLURM jobs (USC CARC defaults)
tools/              ingest_legacy.py — import a pre-pipeline collection
docs/               pipeline internals, scale guide, data format
```

## Docs

- [`docs/PIPELINE.md`](docs/PIPELINE.md) — what each stage does, and why it does it that way
- [`docs/COLLECTING_AT_SCALE.md`](docs/COLLECTING_AT_SCALE.md) — planning a 1000-match run
- [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) — full schemas
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — failures and their fixes
- [`docs/HPC.md`](docs/HPC.md) — running on a SLURM cluster

## Citation

```bibtex
@article{wang2025xego,
  title  = {X-Ego-CS: A Dataset for Cross-Egocentric Multi-Agent Video Understanding},
  author = {Wang, Yunzhe and others},
  journal= {arXiv preprint arXiv:2510.19150},
  year   = {2025}
}
```

## License & terms

Code: MIT (see [LICENSE](LICENSE)).

The pipeline reads publicly available FACEIT match demos through FACEIT's own
API and CDN. You are responsible for complying with FACEIT's terms of service
and with Valve's terms for Counter-Strike 2, and for the privacy expectations
of the players whose gameplay you record. Player Steam IDs are pseudonymous but
not anonymous; think before you republish them.
