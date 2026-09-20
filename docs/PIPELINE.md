# Pipeline internals

One section per stage: what it reads, what it writes, and the non-obvious
decisions baked into it. Read this before modifying a stage — most of the
awkward-looking code is load-bearing.

---

## 01 · discover

**Reads** the FACEIT Data API. **Writes** `matches.jsonl`.

FACEIT has no "recent competitive matches" endpoint, so discovery fans out over
the regional leaderboard: walk top players, pull each player's match history,
deduplicate, then fetch each candidate's full payload.

### Reading the map correctly

A match payload exposes the map in several places, and only one of them is
right:

| Field | What it actually is |
|---|---|
| `payload.maps[]` | the server's map pool |
| `payload.matchCustom.tree.map.values` | the configured pool for that lobby |
| `voting.map.entities[]` | candidates **surviving the ban phase** — often still several |
| **`voting.map.pick`** | **the map that was played** |

Reading `entities` is the tempting mistake. On a 101-match sample it yields a
single decided map for only 40 matches; the other 61 still list 2–7 candidates,
so you either get the wrong map or no map. `voting.map.pick` resolved all 101,
and agreed with ground truth on all 77 matches where the map was independently
known from the demo header.

`read_picked_map()` tries `pick` first and only falls back to a
single-element `entities` list.

### Rate limiting

FACEIT returns 429 aggressively. `_RateLimiter` is a sliding 60-second window
shared across threads, and 429/5xx are retried with exponential backoff
honouring `Retry-After`. Discovery is deliberately sequential — the limit, not
concurrency, is the bound.

---

## 02 · download

**Reads** `matches.jsonl`. **Writes** `demo/<match_id>.dem`.

Plain parallel HTTP from the CDN URL in the payload, then zstd decompression
(`zstandard` if installed, else the `zstd` binary). No browser needed on the
happy path.

### Run this early

FACEIT retains CS2 demos for a limited window — weeks, not months. **A match
whose demo expires can never have action data**, though its recordings remain
perfectly good. This stage therefore gates stages 03 and 05, and the right
order of operations is: discover, download *immediately*, then record at
leisure.

### Expiry vs unreachability

These look identical from a failed download and have opposite remedies, so the
stage separates them:

- `EXPIRED` — the server answered 403/404/410. The object is gone. Not retryable.
- `UNREACHABLE` — DNS did not resolve, or the connection failed. The demo may
  well still exist; you just can't reach it from this network.

`_host_resolves()` checks DNS before the first request, so a locked-down
network fails fast and loudly instead of being misreported as mass expiry.

---

## 03 · metadata

**Reads** `demo/*.dem`. **Writes** `metadata/<match_id>.json`.

Parses each demo once into everything downstream needs: players, rounds, kills,
per-player alive windows, statistics.

### Alive windows are the contract with stage 04

```
alive_start = round.freeze_end_tick        # when the round goes live
alive_end   = first death tick, else round.end_tick
```

Two details matter:

- **`freeze_end`, not `start`.** The interval between them is the buy phase,
  where nobody can move. Recording it wastes capture time and pollutes the
  clip with a static frame.
- **A dead player's camera follows their killer.** Ending the clip at the death
  tick is what makes the dataset egocentric rather than "mostly egocentric".

### Deaths are matched on player name, not Steam ID

The kill feed stores Steam IDs as float64. Steam64 IDs are 17 digits — past
float64's exact-integer range (2^53) — so a fraction of them round to a
neighbouring value and never join back to the player list. Names are stable
within a single match and are used instead. This is a real bug that silently
drops deaths if you "fix" it to use IDs.

### Validation reports, it does not abort

At 1000 matches you will meet demos with 9 players, a missing side assignment,
or a truncated tick stream. One bad demo must not kill a 12-hour job, so
problems are collected into `validation.problems`, the match is flagged
`usable: false`, and stage 07 excludes it unless `--include-unusable`.

---

## 04 · record  (Windows only)

**Reads** `metadata/`, `demo/`. **Writes** `video/match=…/round=…/<steamid>.mp4`.

CS2 has no scriptable capture API, so the recorder drives the game like a
human: focus the window, type console commands, toggle a capture tool's hotkey.

### The seek dance

```
demo_gototick <start>     # lands a few ticks late — the demo is playing
demo_pause
demo_gototick <start>     # now lands exactly
demo_resume
<capture on>  … alive_duration_ticks / 64 seconds …  <capture off>
```

The doubled seek is not redundant. Seeking a *playing* demo overshoots by a
variable few ticks; pausing first makes the second seek exact. Skip it and
clips start mid-action, which shows up downstream as a systematic bias in the
stage-06 offsets.

`spec_show_xray 0` and `r_show_build_info 0` are issued once per demo — without
them every frame carries a wall-hack outline and debug text.

### Resumability

`record_progress.tsv` is appended after every clip, with an `fsync`. A crashed
or rebooted session resumes at the next unrecorded clip. Given stage 04 is
~45–60 minutes per match, this is the difference between losing a clip and
losing an afternoon.

---

## 05 · actions

**Reads** `demo/`, `metadata/`. **Writes**
`state_action/match=…/round=…/<steamid>.parquet`.

One parquet per clip, covering exactly the same alive window, at 64 Hz.

### The input bitfield

`buttons` is a 64-bit field of held inputs. `buttons.py` maps engine `IN_*`
constants to published `k_*` columns. Two subtleties:

- **`IN_WALK` and `IN_SPEED` are both "shift"** depending on the player's
  config, so their masks are OR-ed into one `k_shift` column rather than
  emitted as two columns that are never both meaningful.
- **Bits 25–31 are set in practice but unnamed.** They stay in `KEY_MAPPING`
  so the raw `buttons` list can report them, and are deliberately absent from
  `KEY_TO_INPUT` — inventing a column name for a bit whose meaning is unknown
  would bake a guess into the published schema.

### Yaw wrap correction

Yaw lives in (−180, 180]. A player panning through due south produces a raw
diff near ±360, which would read as a physically impossible flick. Diffs
outside ±180 are folded back:

```python
when(d >  180).then(d - 360)
when(d < -180).then(d + 360)
otherwise(d)
```

### One schema, 200k files

Every file is written with identical column order and dtypes. Without that,
parquet readers infer per-file schemas and a union across the tree fails on the
first disagreement. `_round_is_complete()` checks *columns*, not just
existence, so a schema change invalidates stale files instead of leaving a
mixed-schema tree.

### Only the needed properties

`dem.parse(player_props=[...])` requests six properties. This is the single
biggest lever on parse time and peak memory — the default parses everything.

---

## 06 · align

**Reads** `video/`, `align` config. **Writes**
`align/match=…/round=…/offsets.json`.

The problem: capture starts an unpredictable fraction of a second after
`demo_resume`, and the recordings carry no tick information.

The solution: the CS2 HUD round timer is the one clock visible in both
timelines. Find the frame where it ticks over to a new second — that is, by
construction, a whole-second boundary in game time — and you have one exact
correspondence.

```
offset = video_time_of_transition − game_time_of_transition
video_time = game_sec + offset
```

### Why template matching and not OCR

The timer is a fixed font at a fixed position. Three 8×13 crops — minutes, tens
of seconds, units — are compared against ten reference glyphs with normalised
cross-correlation (`TM_CCOEFF_NORMED`). Normalised correlation is brightness-
and contrast-invariant, which is exactly what survives h264 artefacts and a
wildly varying game background behind a transparent HUD. A general OCR engine
is both slower and worse here: it hallucinates on crops this small.

Measured confidence on real captures: **0.98–0.99**, zero rejected frames.

### Three independent rejections

1. **Score floor** (`min_match_score`, default 0.35) — discards a crop the
   templates don't actually fit.
2. **Arithmetic plausibility** — `tens > 5` is impossible in base 60, and a
   clock above the round time is a misread.
3. **Exactly-one-second transitions** — the readings before and after must
   both be confident and differ by exactly 1. A single-frame glitch (flashbang,
   kill-feed overlay) therefore cannot fabricate a landmark.

### Cross-player agreement is the real check

All ten clips in a round were started by the same automation within seconds, so
their offsets *should* agree. Per-player offsets are compared against the round
median and flagged beyond `max_offset_deviation_seconds`. Observed spread
within a round: **~50 ms stdev**. A genuine outlier means a misread, not real
latency — so `round_offset_sec` is the mean over non-outliers.

### Resolution independence

`digit_boxes` are declared in reference 1280×720 geometry and scaled by
`scale_boxes()` to the actual frame size, with templates resized to match. A
1080p or 1440p capture needs no new constants.

---

## 07 · package

**Reads** everything. **Writes** `release/`.

### Splits are per match, always

Rounds inside one match share players, economy state, and map callouts.
Splitting them across train and test leaks. `split_by: match` is the only
supported granularity, and `assign_split()` hashes `sha256(seed:match_id)` into
`[0,1)` rather than shuffling a list — so **a match keeps its split no matter
how many matches are added later**. That matters when a dataset is published in
waves and papers cite "the test split".

### Coverage is recorded, not assumed

Each clip row carries `has_video`, `has_actions`, `has_align`. `summary.json`
aggregates them into percentages and lists `matches_without_demo`. A consumer
filters on a column instead of discovering the gap as a missing-file crash.

### Hard links, not copies

With `use_hardlinks: true` and no transcode variants, building the release tree
is a metadata operation. Materialising a few hundred GB of video twice on the
same filesystem is pure waste; `link_or_copy()` falls back to a copy only
across devices.

---

## 08 · publish

**Reads** `release/`. **Writes** to the Hugging Face Hub.

- **Bounded commits** (`files_per_commit`, default 400) — an interruption costs
  one chunk, not a six-hour run.
- **Skips files already on the Hub** by diffing against `list_repo_files`, so a
  resubmit is a cheap resume.
- **Small-to-large ordering** — manifests, metadata, align, parquet, then
  video. The dataset is browsable and partially usable long before the bulk
  transfer finishes.
- **Refuses without `--yes`.** Publishing is public, attributable and
  effectively irreversible.

`mecs card <path> --yes` updates only `README.md`, so the dataset card can be
reviewed and shipped independently of a data upload.
