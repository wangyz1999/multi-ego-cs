# Running on a SLURM cluster

`slurm/` holds ready-to-submit jobs, defaulted for USC CARC (Discovery /
Endeavour). They are ordinary sbatch scripts — adapt the account, partition
and paths for another site.

## Setup

```bash
module purge && module load usc ffmpeg zstd
cd ~/projects/multi-ego-cs
uv venv .venv
uv pip install --python .venv/bin/python -e '.[fast-upload]'
```

Install from a **compute node** (`srun --partition=debug --pty bash`): `uv`
builds with a thread-per-core rayon pool and trips the login node's process
limit.

Point `data_root` at scratch-class storage — `configs/carc-hpc.yaml` uses
`/scratch1/$USER/multi-ego-cs` (env vars in the path are expanded). Native
video runs ~3.8 GB per match; it will not fit in a home directory.

### Scratch is not durable — design for that

On most clusters scratch is unbacked and periodically purged, and a maintenance
window can wipe it outright. We had exactly that happen mid-project: a two-day
maintenance reservation emptied `/scratch1/$USER` completely.

Nothing was lost, because of how the layout divides:

| Lives on scratch | Cost to rebuild |
|---|---|
| `metadata/`, `state_action/`, `align/`, `release/` | minutes of compute |
| `video/`, `demo/` (as symlinks into durable storage) | one `ingest` run |

**The rule: scratch holds only what a job can regenerate.** Keep the two things
that cannot be regenerated — the **recordings** and the **demo files** — on
backed-up project storage, and point the pipeline at them with symlinks
(`tools/ingest_legacy.py`, or `link_mode: symlink`). Demos especially: once
FACEIT's retention window passes, a lost demo is gone for good, and a scratch
purge would destroy action data you can never re-derive.

Re-running after a purge is just the normal chain again — every stage is
resumable and finds nothing to skip:

```bash
python tools/ingest_legacy.py --root "$DATA_ROOT" --video ... --demo-glob ...
sbatch slurm/03_metadata.sbatch     # ~1 min for 77 demos
sbatch slurm/06_align.sbatch        # ~2 min for 2168 rounds
```

## Job shapes

Stages differ in what binds them, so size them differently:

| Job | CPUs | Mem | Bound by |
|---|---|---|---|
| `03_metadata` | 16 | 120 G | **memory** — one tick table per worker |
| `05_actions` | 16 | 120 G | **memory** — same |
| `06_align` | 32 | 64 G | **CPU** — decodes ~10 s per clip |
| `07_package` | 32 | 64 G | I/O (or CPU, with transcode variants) |
| `08_publish` | 8 | 32 G | network; long wall clock |

If stages 03/05 get OOM-killed, lower `--cpus-per-task`. Worker memory scales
linearly with worker count, so fewer workers is the fix, not more memory.

## Dependency chain

Align needs only video, so it runs beside metadata rather than behind it:

```bash
A=$(sbatch --parsable slurm/03_metadata.sbatch)
B=$(sbatch --parsable slurm/06_align.sbatch)
C=$(sbatch --parsable --dependency=afterok:$A slurm/05_actions.sbatch)
D=$(sbatch --parsable --dependency=afterok:$A:$B:$C slurm/07_package.sbatch --include-demos)
```

`08_publish` is deliberately not chained. Publishing is public and
irreversible; submit it yourself.

## Measured timings

101 matches / 21,680 clips / 27 GB of demos, on CARC `main`:

| Stage | Wall clock | Shape |
|---|---|---|
| 03 metadata | **1 min 21 s** | 77 demos, 16 workers |
| 06 align | **1 min 36 s** | 2168 rounds, 32 workers |
| 05 actions | **2 min 12 s** | 77 demos → 16,570 parquet files |
| 07 package | 29 min (copying) | drops to minutes with symlinks |

The compute stages are far cheaper than they look. Capture (stage 04) and
upload (stage 08) are the only slow parts, and neither is CPU-bound.

## Thread caps are mandatory

`slurm/_common.sh` exports `OMP_NUM_THREADS=1` and friends. Without them,
numpy/OpenBLAS and OpenCV spawn a thread per visible core inside every worker —
1000+ threads fighting over 16 allocated cores. On a login node it fails
outright with `blas_thread_init: pthread_create failed`; inside a job it
silently corrupts video decoding (`swscaler: Failed initializing scaling
graph`), which looks like an alignment bug.

`_common.sh` also sets `TMPDIR=/scratch1/$USER/tmp`. On compute nodes `/tmp` is
RAM-backed and billed against the job's memory allocation.

## Sourcing `_common.sh`

SLURM copies the batch script to a spool directory, so `$(dirname "$0")` is
*not* the repo. The scripts use:

```bash
source "${MECS_REPO_ROOT:-$HOME/projects/multi-ego-cs}/slurm/_common.sh"
```

Set `MECS_REPO_ROOT`, `MECS_PYTHON` or `MECS_CONFIG` to override.

## Maintenance windows

A job whose `--time` would overrun a cluster reservation will not start; it
sits in `ReqNodeNotAvail, Reserved for maintenance` until the window lifts.

```bash
scontrol show reservation
```

Either shorten `--time` to fit before the window, or submit and let it wait —
queued jobs start automatically afterwards. Every stage is resumable, so a
job truncated by a time limit loses only the items in flight.

## Filesystem etiquette

100 matches is ~21,700 video files plus ~16,600 parquet files. At 1000 matches
that is ~400,000 inodes, and the metadata load is borne by every other user of
a shared filesystem.

- Check quotas first (`myquota` on CARC) — inode limits bite before byte limits.
- Keep `link_mode: auto` so the release tree costs inodes, not terabytes.
- Never `pip install` or untar into the data root.
