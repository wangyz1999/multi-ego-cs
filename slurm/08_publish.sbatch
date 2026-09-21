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

# A token can come from the environment OR from `hf auth login`'s credential
# store, which lives in $HOME and so IS visible inside a job. Check whether one
# resolves at all rather than insisting on the env var - demanding HF_TOKEN
# would reject a perfectly good stored credential and push people toward
# pasting secrets into job scripts.
if ! "$PYTHON" - <<'PYCHECK'
import sys
try:
    from huggingface_hub import get_token
    sys.exit(0 if get_token() else 1)
except Exception:
    sys.exit(1)
PYCHECK
then
  echo "ERROR: no Hugging Face token available in this job." >&2
  echo "Fix either way:" >&2
  echo "  hf auth login                       # stores it under \$HOME (recommended)" >&2
  echo "  HF_TOKEN=... sbatch --export=ALL slurm/08_publish.sbatch --yes" >&2
  exit 1
fi

mecs publish "$@"
echo "finished: $(date -Is)"
