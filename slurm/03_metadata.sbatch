#!/bin/bash
#SBATCH --job-name=mecs-metadata
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mecs-metadata-%j.out
#SBATCH --error=logs/mecs-metadata-%j.out

# Stage 03: parse every demo into rounds + per-player alive windows.
#
# Memory is the binding constraint, not CPU: awpy holds a full tick table per
# demo, which runs 4-8 GB for a long match. 120G / 16 workers leaves headroom;
# if you see OOM kills, lower --cpus-per-task rather than raising --mem.

source "${MECS_REPO_ROOT:-$HOME/projects/multi-ego-cs}/slurm/_common.sh"
mecs metadata "$@"
echo "finished: $(date -Is)"
