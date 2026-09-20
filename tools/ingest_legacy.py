#!/usr/bin/env python3
"""Import a pre-multi-ego-cs collection into the pipeline layout.

The X-Ego-CS recordings were collected before this pipeline existed, with a
different directory convention:

    <legacy>/video/<match_id>/<steamid>/round_<n>.mp4      # player-major
    <legacy>/faceit_metadata/<match_id>/<match_id>.json    # FACEIT payload
    <other>/<map>/demo/<match_id>.dem                      # per-map demo dirs

The pipeline expects round-major partitions:

    <root>/video/match=<match_id>/round=<n>/<steamid>.mp4
    <root>/demo/<match_id>.dem
    <root>/matches.jsonl

This script bridges the two. Video and demos are **symlinked** by default -
the source trees are hundreds of gigabytes and there is no reason to duplicate
them; stage 07 dereferences links when it materialises the release. Use
``--hardlink`` if the pipeline root is on the same filesystem and you want the
release to survive the source being moved.

Nothing is written outside ``--root``, and the source trees are only ever read.

Usage
-----
    python tools/ingest_legacy.py \
        --root /scratch1/$USER/multi-ego-cs \
        --video /project2/<group>/CTFM/data/full/recording/video \
        --faceit /project2/<group>/CTFM/data/full/recording/faceit_metadata \
        --demo-glob '/project2/<group>/x-ego/data/*/demo' \
        --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_ego_cs.paths import Layout  # noqa: E402
from multi_ego_cs.stages.faceit_api import (  # noqa: E402
    read_demo_urls,
    read_finished_at,
    read_picked_map,
    read_players,
)
from multi_ego_cs.util.io_utils import write_jsonl_atomic  # noqa: E402

ROUND_RE = re.compile(r"round_(\d+)\.mp4$", re.IGNORECASE)
STEAMID_RE = re.compile(r"^7656\d{13}$")


def link(src: Path, dst: Path, mode: str, dry_run: bool) -> str:
    if dst.exists() or dst.is_symlink():
        return "exists"
    if dry_run:
        return "would-" + mode
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass  # cross-device; fall through to a symlink
    dst.symlink_to(src)
    return "symlink"


def ingest_video(video_root: Path, layout: Layout, mode: str, dry_run: bool,
                 only: set[str] | None) -> dict[str, int]:
    stats = {"matches": 0, "clips": 0, "linked": 0, "exists": 0, "skipped": 0}
    if not video_root.is_dir():
        print(f"  video root missing: {video_root}")
        return stats

    for match_dir in sorted(video_root.iterdir()):
        if not match_dir.is_dir():
            continue
        match_id = match_dir.name
        if only and match_id not in only:
            continue
        stats["matches"] += 1

        for player_dir in sorted(match_dir.iterdir()):
            if not player_dir.is_dir():
                continue
            steamid = player_dir.name
            if not STEAMID_RE.match(steamid):
                stats["skipped"] += 1
                continue
            for clip in sorted(player_dir.glob("round_*.mp4")):
                m = ROUND_RE.search(clip.name)
                if not m:
                    stats["skipped"] += 1
                    continue
                round_num = int(m.group(1))
                dst = layout.video_file(match_id, round_num, steamid)
                result = link(clip, dst, mode, dry_run)
                stats["clips"] += 1
                stats["exists" if result == "exists" else "linked"] += 1
    return stats


def ingest_demos(patterns: list[str], layout: Layout, mode: str, dry_run: bool,
                 only: set[str] | None) -> dict[str, int]:
    stats = {"found": 0, "linked": 0, "exists": 0}
    for pattern in patterns:
        for directory in sorted(glob.glob(pattern)):
            for dem in sorted(Path(directory).glob("*.dem")):
                match_id = dem.stem
                if only and match_id not in only:
                    continue
                stats["found"] += 1
                result = link(dem, layout.demo_file(match_id), mode, dry_run)
                stats["exists" if result == "exists" else "linked"] += 1
    return stats


def build_index(faceit_root: Path, layout: Layout, dry_run: bool,
                only: set[str] | None) -> dict[str, int]:
    """Reconstruct matches.jsonl from archived FACEIT payloads."""
    stats = {"payloads": 0, "records": 0, "no_map": 0}
    records = []
    if not faceit_root.is_dir():
        print(f"  faceit metadata root missing: {faceit_root}")
        return stats

    for entry in sorted(faceit_root.iterdir()):
        path = entry
        if entry.is_dir():
            candidates = sorted(entry.glob("*.json"))
            if not candidates:
                continue
            path = candidates[0]
        elif entry.suffix != ".json":
            continue

        stats["payloads"] += 1
        bare = path.stem
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  unreadable payload {path.name}: {exc}")
            continue

        picked = read_picked_map(payload)
        if not picked:
            stats["no_map"] += 1

        # The canonical id carries the demo suffix; recover it from whichever
        # artefact actually exists on disk.
        canonical = f"{bare}-1-1"
        for demo in layout.demo.glob(f"{bare}*.dem"):
            canonical = demo.stem
            break
        else:
            for vd in layout.video.glob(f"match={bare}*"):
                canonical = vd.name.removeprefix("match=")
                break

        if only and canonical not in only:
            continue

        records.append(
            {
                "match_id": canonical,
                "bare_id": bare,
                "map": picked,
                "demo_urls": read_demo_urls(payload),
                "finished_at": read_finished_at(payload),
                "region": None,
                "competition": None,
                "players": read_players(payload),
                "discovered_via": "ingest_legacy",
            }
        )
        stats["records"] += 1

    if records and not dry_run:
        write_jsonl_atomic(layout.matches_index, sorted(records, key=lambda r: r["match_id"]))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="pipeline data_root to populate")
    ap.add_argument("--video", help="legacy video root (<match>/<steamid>/round_N.mp4)")
    ap.add_argument("--faceit", help="legacy FACEIT payload root")
    ap.add_argument("--demo-glob", action="append", default=[],
                    help="glob matching directories that contain .dem files (repeatable)")
    ap.add_argument("--match-file", help="restrict to match ids listed in this file")
    ap.add_argument("--hardlink", action="store_true",
                    help="hard-link instead of symlinking (same filesystem only)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    layout = Layout(args.root)
    if not args.dry_run:
        layout.ensure()
    mode = "hardlink" if args.hardlink else "symlink"

    only: set[str] | None = None
    if args.match_file:
        from multi_ego_cs.util.ids import extract_ids

        text = Path(args.match_file).read_text(encoding="utf-8")
        ids = extract_ids(text)
        only = {i if i.count("-") > 5 else f"{i}-1-1" for i in ids}
        only |= set(ids)
        print(f"Restricting to {len(ids)} match ids from {args.match_file}")

    print(f"pipeline root : {layout.root}")
    print(f"link mode     : {mode}{'  (dry run)' if args.dry_run else ''}\n")

    if args.demo_glob:
        print("demos:")
        print(f"  {ingest_demos(args.demo_glob, layout, mode, args.dry_run, only)}")
    if args.video:
        print("video:")
        print(f"  {ingest_video(Path(args.video), layout, mode, args.dry_run, only)}")
    if args.faceit:
        print("match index:")
        print(f"  {build_index(Path(args.faceit), layout, args.dry_run, only)}")

    print("\nNext: mecs --config <cfg> status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
