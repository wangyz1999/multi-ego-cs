#!/bin/bash
# Shared setup for every multi-ego-cs SLURM job. Sourced, not executed.
#
# Two things here are not optional on a shared cluster:
#
#  1. Thread caps. numpy/OpenBLAS and OpenCV each default to one thread per
#     visible core. On a 64-core node with 16 worker processes that is 1000+
#     threads competing for 16 allocated cores, and on a login node it trips
#     RLIMIT_NPROC outright ("blas_thread_init: pthread_create failed").
#     One thread per process, with parallelism from processes, is correct here.
#
#  2. TMPDIR off /tmp. On compute nodes /tmp is RAM-backed and billed against
#     the job's memory allocation, so a video decode that spills there can OOM
#     the job.

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENCV_FFMPEG_THREADS=1
export POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK:-4}

export TMPDIR="/scratch1/${USER}/tmp"
mkdir -p "$TMPDIR"

module purge
module load usc
module load ffmpeg
module load zstd

REPO_ROOT="${MECS_REPO_ROOT:-$HOME/projects/multi-ego-cs}"
PYTHON="${MECS_PYTHON:-$REPO_ROOT/.venv/bin/python}"
CONFIG="${MECS_CONFIG:-$REPO_ROOT/configs/carc-hpc.yaml}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: interpreter not found at $PYTHON" >&2
  echo "Create it with:  cd $REPO_ROOT && uv venv .venv && uv pip install --python .venv/bin/python -e '.[fast-upload]'" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mecs() { "$PYTHON" -m multi_ego_cs.cli --config "$CONFIG" "$@"; }

echo "=============================================================="
echo " job          : ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-none})"
echo " node         : $(hostname)"
echo " cpus         : ${SLURM_CPUS_PER_TASK:-?}"
echo " mem          : ${SLURM_MEM_PER_NODE:-?} MB"
echo " repo         : $REPO_ROOT"
echo " python       : $PYTHON"
echo " config       : $CONFIG"
echo " started      : $(date -Is)"
echo "=============================================================="
