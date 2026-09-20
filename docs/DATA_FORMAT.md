# Data format

Every path is keyed on `match_id` — the FACEIT match id with the demo's
`-<match>-<map>` suffix, e.g.
`1-0076bc6b-4ce9-45fa-8e0b-35fd140ddd60-1-1`. Players are keyed on Steam64 id.

```
release/
  match_round_partitioned.csv          legacy 4-column index (root)
  manifest/
    clips.csv                          one row per (match, round, player)
    rounds.csv                         one row per (match, round)
    matches.csv                        one row per match
    match_round_partitioned.csv        same legacy index
    summary.json                       counts, coverage, split policy
  metadata/<match_id>.json             rounds, kills, alive windows, validation
  video/match=<id>/round=<n>/<steamid>.mp4
  state_action/match=<id>/round=<n>/<steamid>.parquet
  align/match=<id>/round=<n>/offsets.json
  demo/<match_id>.dem                  (when --include-demos)
```

Hive-style `match=`/`round=` partitioning means polars, pandas, pyarrow and the
HF `datasets` loader all discover the tree without a custom reader, and
`match_id`/`round` come back as columns:

```python
import polars as pl
df = pl.scan_parquet("state_action/**/*.parquet", hive_partitioning=True)
```

---

## `state_action/…/<steamid>.parquet`

One row per tick (64 Hz), covering exactly the window in which that player was
alive in that round. Identical column order and dtypes in every file.

| Column | Type | Meaning |
|---|---|---|
| `tick` | Int32 | raw demo tick |
| `tick_norm` | Int32 | `tick − alive_start_tick`; 0 at the first alive tick |
| `game_sec` | Float32 | `tick_norm / 64` |
| `x`, `y`, `z` | Float32 | world position |
| `health` | Float32 | 0–100 |
| `pitch`, `yaw` | Float32 | absolute view angles, degrees |
| `delta_pitch` | Float32 | per-tick pitch change |
| `delta_yaw` | Float32 | per-tick yaw change, **±180° wrap corrected** |
| `usercmd_mouse_dx` | Float32 | raw horizontal mouse counts |
| `usercmd_mouse_dy` | Float32 | raw vertical mouse counts |
| `k_*` (25 columns) | Int8 | 1 while the input is held |
| `buttons` | List[Utf8] | raw engine `IN_*` names set this tick |

### The 25 input columns

```
k_w  k_a  k_s  k_d                     movement
k_space  k_ctrl  k_shift               jump, crouch, walk
k_mouse_left  k_mouse_right  k_mouse_middle
k_r  k_e  k_f  k_tab                   reload, use, inspect, scoreboard
k_zoom                                 scope
k_grenade1  k_grenade2
k_weapon1  k_weapon2
k_turn_left  k_turn_right              keyboard turning (rare)
k_alt1  k_alt2  k_bullrush  k_cancel
```

`k_shift` merges `IN_WALK` and `IN_SPEED` — both are bound to shift depending
on the player's config, so they are OR-ed into one column rather than emitted
as two that are never both meaningful.

`buttons` carries the raw engine names, including bits 25–31 which are set in
practice but have no documented meaning. They are deliberately *not* given
`k_*` columns: naming them would bake a guess into the published schema.

### Angles

`yaw` is in (−180, 180] and `pitch` in [−89, 89]. Raw `yaw.diff()` produces
values near ±360 whenever a player pans through due south; `delta_yaw` folds
those back into ±180, so it is always the true angular change.

`usercmd_mouse_dx/dy` are raw device counts, **not** degrees. Converting
requires the player's sensitivity, which the demo does not record;
`scripts/estimate_sensitivity.py` in the upstream `CS2_Action` repo estimates
it by regressing `delta_yaw` on `usercmd_mouse_dx`.

---

## `align/…/offsets.json`

```json
{
  "schema_version": 1,
  "match_id": "1-0076bc6b-…-1-1",
  "round": 5,
  "sign_convention": "video_time = game_sec + offset_sec",
  "round_clock_seconds": 115,
  "players": {
    "76561198064353169": {
      "offset_sec": -0.768,
      "video_fps": 30.19,
      "frame_size": [1280, 720],
      "start_timer": "1:55",
      "transition_timer": "1:54",
      "transition_frame": 7,
      "transition_video_sec": 0.232,
      "transition_game_sec": 1.0,
      "match_score": 0.984,
      "rejected_frames": 0,
      "deviation_from_median_sec": 0.0,
      "outlier": false
    }
  },
  "round_offset_sec": -0.771,
  "round_offset_median_sec": -0.768,
  "round_offset_spread_sec": 0.166,
  "aligned_players": 10,
  "total_players": 10,
  "outliers": [],
  "usable": true
}
```

**The only sign convention used anywhere in this project:**

```python
video_time  = game_sec + offset_sec
game_sec    = video_time - offset_sec
frame_index = round(video_time * video_fps)
```

`offset_sec` is typically **negative** (−0.7 to −3 s): the round goes live
before the capture tool's first frame lands, so game time 0 sits slightly
before the clip starts.

Use the per-player `offset_sec` for single-player work. Use
`round_offset_sec` — the mean over non-outlier players — when compositing
several players' views into one synchronised grid.

Check `usable` before trusting a round, and skip players flagged `outlier`.

---

## `metadata/<match_id>.json`

```
schema_version, match_id, demo_file, processed_at, map_name, tickrate
header{}                     demo header as parsed
players[]                    steamid, name, team_number (0/1)
rounds[]                     round_number, start_tick, freeze_end_tick,
                             end_tick, official_end_tick, winner, reason,
                             bomb_plant_tick, bomb_site,
                             t_team_number, ct_team_number
kills[]                      tick, round_number, victim/attacker
                             (steamid, name, side), weapon, headshot,
                             victim_position{x,y,z}, attacker_position{x,y,z}
player_alive_times{}         round -> [ {steamid, player_name, team_number,
                                         alive_start_tick, alive_end_tick,
                                         alive_duration_ticks, died_in_round} ]
statistics{}                 aggregates
validation{problems[], usable}
usable                       bool
```

`player_alive_times` is the join key between video and trajectories:
`alive_start_tick` is `tick_norm == 0`, and the clip covers exactly
`alive_duration_ticks / 64` seconds.

`team_number` is remapped from the engine's 2/3 to 0/1. `t_team_number` and
`ct_team_number` per round tell you which of those was T and which CT — they
swap at halftime.

---

## `manifest/clips.csv`

One row per (match, round, player) — the table to filter on.

| Column | Meaning |
|---|---|
| `match_id`, `round_number`, `steamid` | identity |
| `player_name`, `team_number`, `side` | `side` is `T`/`CT` for that round |
| `map_name`, `split` | `train` / `val` / `test` |
| `alive_start_tick`, `alive_end_tick`, `alive_duration_ticks` | tick window |
| `alive_duration_sec` | seconds |
| `died_in_round` | 1 if the clip ends at a death |
| **`has_video`**, **`has_actions`**, **`has_align`** | **coverage flags** |
| `video_offset_sec`, `video_fps` | alignment, denormalised for convenience |
| `video_bytes` | clip size |

**Filter on the coverage flags.** Not every clip has every modality — a match
whose demo expired before collection has video but can never have actions.
Nulls in the tick columns mean exactly that.

```python
import polars as pl
clips = pl.read_csv("manifest/clips.csv")
full = clips.filter(
    (pl.col("has_video") == 1)
    & (pl.col("has_actions") == 1)
    & (pl.col("has_align") == 1)
)
```

## `manifest/rounds.csv`

`match_id`, `round_number`, `map_name`, `split`, `n_players`, `n_video`,
`n_actions`, `winner`, `reason`, `bomb_planted`, `bomb_site`,
`round_offset_sec`, `round_offset_spread_sec`.

`round_offset_spread_sec` is a quality signal: all ten clips were started by
the same automation, so a large spread means at least one misread.

## `manifest/matches.csv`

`match_id`, `map_name`, `split`, `n_rounds`, `n_clips`, `n_video`,
`n_actions`, `has_demo`, `usable`, `total_video_bytes`, `total_video_sec`.

## `manifest/summary.json`

Counts, `maps`, `maps_by_split`, `matches_by_split`, `rounds_by_split`,
`coverage` (with percentages), `matches_without_demo`, `video_only_matches`,
`excluded_unusable`, `total_video_hours`, and `split_policy`.

Read this first. It is the only place the dataset states its own coverage.

---

## Splits

Assigned **per match**, never per round — rounds within a match share players,
economy and callouts, so splitting them across train and test leaks.

```python
position = int.from_bytes(sha256(f"{seed}:{match_id}").digest()[:7], "big") / 2**56
```

Because it hashes the id rather than shuffling a list, **a match keeps its
split no matter how many matches are added later**. A dataset published in
waves keeps "the test split" meaning the same thing across versions.

---

## Loading examples

```python
import json, polars as pl

clips = pl.read_csv("manifest/clips.csv")
row = clips.filter(
    (pl.col("has_actions") == 1) & (pl.col("has_align") == 1)
).row(0, named=True)

m, r, s = row["match_id"], row["round_number"], row["steamid"]

traj = pl.read_parquet(f"state_action/match={m}/round={r}/{s}.parquet")
off  = json.load(open(f"align/match={m}/round={r}/offsets.json"))["players"][s]

# frame index for every tick
traj = traj.with_columns(
    ((pl.col("game_sec") + off["offset_sec"]) * off["video_fps"])
    .round().cast(pl.Int32).alias("frame_index")
)
```

Whole-split scan, coverage-filtered:

```python
good = set(
    clips.filter(
        (pl.col("split") == "train") & (pl.col("has_actions") == 1)
    ).select(["match_id", "round_number", "steamid"]).iter_rows()
)
lf = pl.scan_parquet("state_action/**/*.parquet", hive_partitioning=True)
```
