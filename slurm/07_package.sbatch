#!/bin/bash
#SBATCH --job-name=mecs-package
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mecs-package-%j.out
#SBATCH --error=logs/mecs-package-%j.out

# Stage 07: build the release tree and manifests.
#
# With package.use_hardlinks and no transcode variants this is metadata-only
# and finishes in minutes. Adding a downscaled variant makes it an ffmpeg
# batch - then the 32 CPUs and the 12 h limit both matter.

source "${MECS_REPO_ROOT:-$HOME/projects/multi-ego-cs}/slurm/_common.sh"
mecs package "$@"
echo "finished: $(date -Is)"
