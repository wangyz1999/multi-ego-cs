"""multi-ego-cs — collect synchronized multi-egocentric Counter-Strike 2 datasets.

The package is organised as a linear pipeline. Each stage reads from the shared
data root, writes into its own subtree, and is independently resumable:

    01 discover  FACEIT API      -> matches.jsonl        (which matches to collect)
    02 download  FACEIT CDN      -> demo/                (.dem replay files)
    03 metadata  demo            -> metadata/            (rounds, alive windows)
    04 record    CS2 + capture   -> video/               (per-player round .mp4)
    05 actions   demo            -> state_action/        (tick-level parquet)
    06 align     video + parquet -> align/               (video<->tick offsets)
    07 package   all of the above-> release/             (manifest + partitions)
    08 publish   release/        -> Hugging Face Hub

Stage 04 is the only Windows-only stage; everything else runs on Linux/HPC.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
