#!/bin/bash
#SBATCH --job-name=mecs-publish
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/mecs-publish-%j.out
#SBATCH --error=logs/mecs-publish-%j.out

# Stage 08: upload to the Hugging Face Hub.
#
# Network-bound, so few CPUs but a long wall clock - a few hundred GB over the
# cluster's uplink runs for hours. The 2-day limit is the `main` partition
# maximum; for a larger release use --partition=oneweek.
#
# Requires a write token:   export HF_TOKEN=...
# The stage skips files already on the Hub, so a resubmit resumes rather than
# re-uploading. It also refuses to run without --yes.

source "${MECS_REPO_ROOT:-$HOME/projects/multi-ego-cs}/slurm/_common.sh"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set in the job environment." >&2
  echo "SLURM does not inherit your login shell, so export it at submit time:" >&2
  echo "  HF_TOKEN=\$(cat ~/.cache/huggingface/token) sbatch --export=ALL slurm/08_publish.sbatch --yes" >&2
  exit 1
fi

mecs publish "$@"
echo "finished: $(date -Is)"
