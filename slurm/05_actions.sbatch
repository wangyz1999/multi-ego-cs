#!/bin/bash
#SBATCH --job-name=mecs-actions
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mecs-actions-%j.out
#SBATCH --error=logs/mecs-actions-%j.out

# Stage 05: tick-level state/action parquet, one file per (match, round, player).
#
# Same memory profile as stage 03 - one demo's tick table per worker. The stage
# is resumable at match granularity, so a time-limit kill loses at most the
# matches in flight; just resubmit.

source "${MECS_REPO_ROOT:-$HOME/projects/multi-ego-cs}/slurm/_common.sh"
mecs actions "$@"
echo "finished: $(date -Is)"
