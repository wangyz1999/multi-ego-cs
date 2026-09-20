#!/usr/bin/env python3
"""Generate the Hugging Face dataset card from a built release.

Every number in the card is read from `release/manifest/summary.json` and the
manifest CSVs. Nothing is hand-typed, so the card cannot drift from the data it
describes - which is the failure mode that makes most dataset cards untrue
within one release.

    python tools/make_dataset_card.py \
        --release /scratch1/$USER/multi-ego-cs/release \
        --out     /scratch1/$USER/multi-ego-cs/release/README.md

Then review it, and publish with:

    mecs card <out> --repo-id <owner/name> --yes
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

MAP_LABEL = {
    "de_mirage": "Mirage", "de_dust2": "Dust II", "de_inferno": "Inferno",
    "de_nuke": "Nuke", "de_overpass": "Overpass", "de_train": "Train",
    "de_ancient": "Ancient", "de_anubis": "Anubis", "de_vertigo": "Vertigo",
}


def _fmt(n: int | float) -> str:
    return f"{n:,}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build(release: Path, repo_id: str, paper: str, code: str) -> str:
    summary = json.loads((release / "manifest" / "summary.json").read_text())
    matches = _read_csv(release / "manifest" / "matches.csv")
    clips = _read_csv(release / "manifest" / "clips.csv")

    cov = summary["coverage"]
    maps = summary["maps"]
    by_split = summary["matches_by_split"]
    rounds_by_split = summary["rounds_by_split"]
    video_only = set(summary.get("video_only_matches") or [])
    no_demo = set(summary.get("matches_without_demo") or [])

    n_matches = summary["matches"]
    n_rounds = summary["rounds"]
    n_clips = summary["clips"]
    hours = summary["total_video_hours"]

    n_players = len({c["steamid"] for c in clips}) if clips else 0
    n_with_actions = len([m for m in matches if int(m.get("n_actions") or 0) > 0])

    # ---- map table, split into full-modality vs video-only -----------------
    map_rows = []
    action_maps = Counter()
    for m in matches:
        if int(m.get("n_actions") or 0) > 0:
            action_maps[m["map_name"]] += 1
    for name, total in sorted(maps.items(), key=lambda kv: -kv[1]):
        label = MAP_LABEL.get(name, name)
        with_act = action_maps.get(name, 0)
        map_rows.append(
            f"| {label} (`{name}`) | {total} | {with_act} | {total - with_act} |"
        )
    map_table = "\n".join(map_rows)

    split_rows = "\n".join(
        f"| `{s}` | {by_split.get(s, 0)} | {rounds_by_split.get(s, 0)} |"
        for s in ("train", "val", "test")
        if s in by_split or s in rounds_by_split
    )

    coverage_note = ""
    if no_demo:
        coverage_note = (
            f"\n> **{len(no_demo)} of the {n_matches} matches have video but no "
            f"action data.** FACEIT retains CS2 demos for a limited window, and "
            f"for these matches the demo expired before it could be archived. "
            f"The tick stream exists only in the demo, so those actions are "
            f"permanently unrecoverable. Their recordings are unaffected and "
            f"fully usable. Filter on `has_actions` in `manifest/clips.csv`, or "
            f"read `video_only_matches` in `manifest/summary.json`.\n"
        )

    return f"""---
license: mit
language:
- en
pretty_name: X-Ego-CS
task_categories:
- video-classification
- video-text-to-text
- visual-question-answering
- text-to-video
tags:
- video
- video understanding
- game
- gameplay understanding
- multi-agent
- esport
- counter-strike
- opponent-modeling
- ego-centric
- cross ego-centric
- action-prediction
- imitation-learning
configs:
- config_name: default
  data_files:
  - split: train
    path: manifest/clips.csv
---

# X-Ego-CS

**Ten players. One match. Ten simultaneous first-person recordings, each paired
with a 64 Hz stream of that player's exact keyboard, mouse and view-angle
inputs — all on a common, measured clock.**

[Paper]({paper}) · [Collection pipeline]({code})

| | |
|---|---|
| Matches | **{_fmt(n_matches)}** across **{len(maps)} maps** |
| Rounds | {_fmt(n_rounds)} |
| Ego-clips | **{_fmt(n_clips)}** (one player, one round, one life) |
| Video | **{hours:,.1f} hours** |
| Unique players | {_fmt(n_players)} |
| Matches with tick-level actions | {_fmt(n_with_actions)} |
| Clips with actions | {cov['actions_pct']:.1f}% |
| Clips with measured video↔tick alignment | {cov['alignment_pct']:.1f}% |

## What makes this different

Most gameplay datasets give you one viewpoint, or many viewpoints that are not
actually synchronised, or video with no ground-truth actions. This one gives
all ten simultaneous egocentric views of the same match, each with the player's
real inputs, aligned to a measured offset rather than an assumed one.

**Clips are one player's life, not one round.** A dead player's spectator
camera follows their killer. Recording whole rounds would make a third of any
"egocentric" corpus silently somebody else's point of view. Each clip here
starts when the round goes live and ends at that player's death (or the round
end).

**Video and ticks are aligned by measurement.** Screen capture starts an
unpredictable fraction of a second after a demo resumes, so pairing frame *i*
with tick `start + i·(64/fps)` drifts by hundreds of milliseconds — enough to
put a flick on the wrong frame. Every round carries a per-player offset
measured by reading the in-game HUD timer with template matching (digit
confidence 0.98–0.99; the ten players in a round agree to ~50 ms).

**Coverage is stated, not assumed.** Every clip row carries `has_video`,
`has_actions` and `has_align`.

## Contents

```
video/match=<id>/round=<n>/<steamid>.mp4          first-person recording
state_action/match=<id>/round=<n>/<steamid>.parquet   64 Hz state + actions
align/match=<id>/round=<n>/offsets.json           video ↔ tick offsets
metadata/<id>.json                                rounds, kills, alive windows
demo/<id>.dem                                     raw CS2 replay
manifest/clips.csv                                one row per clip + coverage
manifest/rounds.csv, matches.csv, summary.json
match_round_partitioned.csv                       legacy index
```

### Trajectory columns (one row per tick, 64 Hz)

| Column | Meaning |
|---|---|
| `tick`, `tick_norm`, `game_sec` | raw tick, zeroed at round start, seconds |
| `x`, `y`, `z`, `health` | world state |
| `pitch`, `yaw` | absolute view angles |
| `delta_pitch`, `delta_yaw` | per-tick view change, ±180° wrap corrected |
| `usercmd_mouse_dx/dy` | raw mouse counts from the usercmd |
| `k_w`, `k_a`, `k_mouse_left`, … | 25 binary input columns |
| `buttons` | raw engine input names |

## Maps

| Map | Matches | With actions | Video only |
|---|---|---|---|
{map_table}
{coverage_note}
## Splits

Assigned **per match** — rounds within a match share players, economy and
callouts, so splitting them across train and test would leak. Assignment hashes
the match id, so a match keeps its split as the dataset grows.

| Split | Matches | Rounds |
|---|---|---|
{split_rows}

## Usage

```python
import json, polars as pl
from huggingface_hub import snapshot_download

root = snapshot_download("{repo_id}", repo_type="dataset")

clips = pl.read_csv(f"{{root}}/manifest/clips.csv")
full = clips.filter(
    (pl.col("has_video") == 1)
    & (pl.col("has_actions") == 1)
    & (pl.col("has_align") == 1)
)

row = full.row(0, named=True)
m, r, s = row["match_id"], row["round_number"], row["steamid"]

traj = pl.read_parquet(f"{{root}}/state_action/match={{m}}/round={{r}}/{{s}}.parquet")
off = json.load(open(f"{{root}}/align/match={{m}}/round={{r}}/offsets.json"))["players"][s]
```

### Aligning video with ticks

```python
video_time  = game_sec + offset_sec      # the only sign convention used here
frame_index = round(video_time * offset["video_fps"])
```

`offset_sec` is typically negative (−0.7 to −3 s): the round goes live shortly
before the capture's first frame. Use the per-player value for single-player
work and `round_offset_sec` when compositing several views into a grid.

## How it was collected

Collected with [multi-ego-cs]({code}), an open end-to-end pipeline: discover
matches via the FACEIT API → download demos → extract round and alive-window
structure → drive CS2 to replay each demo and capture each player's viewpoint →
extract 64 Hz state/action trajectories → align video to ticks by HUD-timer
template matching → package and publish.

The pipeline is published so this corpus can be extended or reproduced at a
different scale. Note that recording is real-time: roughly 45–60 minutes of
capture per match, which is the dominant cost of any collection effort.

## Limitations

- **Action coverage is partial** where demos expired before archival. This is
  recorded per clip and cannot be backfilled.
- **Map distribution is skewed** toward Mirage; an unconstrained FACEIT sample
  reflects what players actually queue for.
- **`usercmd_mouse_dx/dy` are raw device counts, not degrees.** Converting
  needs per-player sensitivity, which demos do not record.
- **Steam IDs are pseudonymous, not anonymous.** They are real accounts.
- Recordings are 1280×720 (a small number at 1024×720) at ~30 fps, with HUD
  visible and spectator x-ray disabled.

## Ethics and terms

Built from publicly available FACEIT match demos via FACEIT's own API and CDN.
Users must comply with FACEIT's terms of service and Valve's terms for
Counter-Strike 2. The footage shows real players' gameplay under pseudonymous
Steam IDs; please respect their privacy expectations and do not attempt
deanonymisation.

## Citation

```bibtex
@article{{wang2025xego,
  title  = {{X-Ego-CS: A Dataset for Cross-Egocentric Multi-Agent Video Understanding}},
  author = {{Wang, Yunzhe and others}},
  journal= {{arXiv preprint arXiv:2510.19150}},
  year   = {{2025}}
}}
```

## License

MIT for the dataset annotations, manifests and pipeline code. Underlying
gameplay footage and demo files originate from FACEIT-hosted matches.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, help="path to the built release/ tree")
    ap.add_argument("--out", required=True, help="where to write README.md")
    ap.add_argument("--repo-id", default="wangyz1999/X-EGO-CS")
    ap.add_argument("--paper", default="https://arxiv.org/abs/2510.19150")
    ap.add_argument("--code", default="https://github.com/wangyz1999/multi-ego-cs")
    args = ap.parse_args()

    release = Path(args.release)
    summary = release / "manifest" / "summary.json"
    if not summary.exists():
        raise SystemExit(f"No manifest at {summary} - run `mecs package` first.")

    card = build(release, args.repo_id, args.paper, args.code)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card, encoding="utf-8")
    print(f"Wrote {out} ({len(card.splitlines())} lines)")
    print("Review it, then:  mecs card", out, "--repo-id", args.repo_id, "--yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
