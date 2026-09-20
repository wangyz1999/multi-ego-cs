#!/bin/bash
#SBATCH --job-name=mecs-align
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mecs-align-%j.out
#SBATCH --error=logs/mecs-align-%j.out

# Stage 06: HUD-timer alignment. Decodes only the first few seconds of each
# clip, so this is CPU-bound and cheap on memory - the opposite profile to
# stages 03/05. Scale --cpus-per-task up freely.
#
# Measured: ~0.7 s per round (10 clips) per worker.

source "${MECS_REPO_ROOT:-$HOME/projects/multi-ego-cs}/slurm/_common.sh"
mecs align "$@"
echo "finished: $(date -Is)"
