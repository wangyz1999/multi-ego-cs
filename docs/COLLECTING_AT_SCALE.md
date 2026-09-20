# Collecting at scale

Say you want 1000 matches. This is what that actually costs and how to get
there without losing a month to a preventable mistake.

---

## The one number that matters

**Stage 04 (recording) is real-time.** A 24-round match with 10 players is
about 45–60 minutes of screen capture, and no hardware makes that faster —
you are watching a replay.

```
1000 matches × ~50 min  ≈  830 hours  ≈  35 machine-days of continuous capture
```

Everything else in the pipeline is a background job measured in hours:

| Stage | 1000 matches | Bound by | Parallel? |
|---|---|---|---|
| 01 discover | 2–4 h | FACEIT rate limit | no — the limit is the bound |
| 02 download | 3–6 h, ~350 GB | network, **demo expiry** | yes, 4–8 ways |
| 03 metadata | ~4 h / 16 cores | memory | yes, per match |
| **04 record** | **830 h** | **real time** | **yes — across machines** |
| 05 actions | ~8 h / 16 cores | memory | yes, per match |
| 06 align | ~1 h / 32 cores | CPU | yes, per round |
| 07 package | minutes (hard links) | I/O | yes |
| 08 publish | 10–30 h | upload bandwidth | chunked, resumable |

So: **plan around stage 04, and shard it.**

---

## Shard the recording

Split the match list and run one shard per capture machine, all writing to a
shared `data_root` (NAS, SMB mount, synced folder — anything they can all see).

```bash
# split 1000 matches across 6 machines
mecs status --json | jq -r '.??'          # or just use your matches.jsonl
awk 'NR%6==0' ids.txt > shard_0.txt
awk 'NR%6==1' ids.txt > shard_1.txt
# …

# on each capture machine
mecs record --config my.yaml --match-file shard_2.txt
```

Six machines turns 35 days into ~6 days. Shards never touch the same
`(match, round, player)`, so there is no coordination beyond the shared root.

Stage 04 keeps a `record_progress.tsv` next to the data and appends to it after
every clip with an `fsync`. A machine that reboots, crashes, or has CS2 fall
over resumes at the next unrecorded clip. **Check this file, not the clock, to
know where a shard is.**

---

## The demo-expiry race — read this before anything else

FACEIT keeps CS2 demos for a limited window: weeks, not months. This creates
a hard ordering constraint that is easy to miss and impossible to undo.

```
discover  →  download   (do this NOW)  →  record (whenever)  →  actions
                  │                                              │
                  └────────── if you skip ahead, this ───────────┘
                             can never run for that match
```

A match whose demo expires keeps its video forever and loses its action data
forever. There is no recovery: the tick stream only exists in the demo.

**Rules:**

1. Run `mecs download` the same day as `mecs discover`. Always.
2. Run `mecs metadata` and `mecs actions` early too — both need the demo. You
   can record months later, but you cannot parse months later.
3. Keep the `.dem` files. They are ~28 GB per 100 matches, which is cheap
   insurance against ever needing to re-extract anything.
4. Check `summary.json` → `matches_without_demo` before publishing, and state
   the coverage in your dataset card.

If downloads fail, read the stage's classification carefully — it distinguishes
two situations that look identical but are not:

- **`EXPIRED`** (HTTP 403/404/410) — genuinely gone. Nothing to do.
- **`UNREACHABLE`** (DNS/connection failure) — a network problem on *your*
  side. The demos may be perfectly fine. Retry from a network with
  unrestricted outbound DNS before concluding anything. Institutional networks
  and HPC clusters commonly cannot resolve the FACEIT CDN host.

---

## Disk budget

Measured on a real 101-match, 8-map collection:

| Artefact | Per 100 matches | Per 1000 matches |
|---|---|---|
| Demos (`.dem`) | 28 GB | 280 GB |
| Native video (1280×720, 30 fps) | 380 GB | **3.8 TB** |
| Action parquet (64 Hz, zstd) | ~8 GB | 80 GB |
| Metadata + align JSON | <1 GB | ~5 GB |

Native video dominates by an order of magnitude. Two things follow:

**Put the data root on scratch-class storage.** 3.8 TB will not fit in a home
directory, and the write pattern (21,680 files per 100 matches) is hard on any
filesystem tuned for small files.

**Publish a downscaled variant.** Most training does not need 720p30. Add a
variant in `package.variants` and the release carries both:

```yaml
package:
  variants:
    native: {}
    small:
      width: 306
      height: 306
      fps: 4
      vcodec: libx264
      crf: 28
      preset: veryfast
```

306×306 at 4 fps runs about 16× smaller — ~240 GB for 1000 matches instead of
3.8 TB — and is what most video models actually ingest.

---

## Inode pressure

100 matches is ~21,700 video files plus ~21,700 parquet files. 1000 matches is
~430,000 files. On a shared cluster filesystem that metadata load is a real
cost borne by every other user, and many quotas cap inodes long before bytes.

- Check your inode quota before starting (`myquota` on USC CARC).
- Hard-link rather than copy when building the release
  (`package.use_hardlinks: true`, the default).
- Do not untar archives or run `pip install` inside the data root.

---

## Tuning discovery for 1000 matches

Default discovery walks 200 players × 20 matches = 4000 candidates, which is
comfortable for 100 accepted matches but thin for 1000 once filters bite.

```yaml
discover:
  target_matches: 1000
  num_players: 1200          # the main lever
  matches_per_player: 30
  min_elo: 2000
  maps: []                   # [] accepts every map — see below
  min_rounds: 16
```

Every run logs a rejection histogram:

```
Rejection reasons: {'map': 412, 'no-demo-url': 88, 'rounds': 31, 'status': 12}
```

Read it. If `map` dominates, your allowlist is the bottleneck.

**On map balance.** An unconstrained FACEIT sample is heavily skewed — in one
101-match collection: mirage 46, dust2 19, inferno 13, anubis 6, overpass 5,
nuke 5, train 4, ancient 3. If you want balance you must either set
`maps: [...]` and run separate passes per map, or over-collect and subsample.
Deciding this *after* recording wastes capture hours you cannot get back.

---

## Running it on a cluster

`slurm/` has ready-to-submit jobs. The dependency chain that matters:

```bash
A=$(sbatch --parsable slurm/03_metadata.sbatch)
B=$(sbatch --parsable slurm/06_align.sbatch)                     # needs only video
C=$(sbatch --parsable --dependency=afterok:$A slurm/05_actions.sbatch)
D=$(sbatch --parsable --dependency=afterok:$A:$B:$C slurm/07_package.sbatch)
```

Align depends only on video, so it runs in parallel with metadata rather than
waiting behind it.

Stages 03 and 05 are **memory**-bound (one full tick table per worker, 4–8 GB
for a long match); stage 06 is **CPU**-bound and cheap on memory. Size them
differently — see [HPC.md](HPC.md).

---

## When a capture machine dies at match 340

Nothing special. That is what the design is for.

```bash
mecs status --config my.yaml          # what exists across all shards
mecs record --config my.yaml --match-file shard_2.txt   # resumes mid-match
```

If the machine is gone for good, hand its shard file to another machine — the
progress log lives with the data, not with the machine, so the replacement
picks up exactly where the dead one stopped.

---

## Pre-publication checklist

```bash
mecs package --config my.yaml
cat <data_root>/release/manifest/summary.json
```

Confirm before you upload:

- [ ] `coverage.video_pct`, `actions_pct`, `alignment_pct` — and that you
      **state these in the dataset card**
- [ ] `matches_without_demo` — these have no action data, permanently
- [ ] `excluded_unusable` — matches dropped by stage-03 validation
- [ ] `maps` — is the distribution what you claimed?
- [ ] `matches_by_split` — split is per *match*; rounds never straddle splits
- [ ] `total_video_hours` — the headline number, computed rather than estimated

Then:

```bash
mecs publish --config my.yaml --dry-run   # inspect the plan
mecs publish --config my.yaml --yes       # go
```
