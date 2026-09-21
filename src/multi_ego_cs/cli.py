"""``mecs`` - one entry point for every pipeline stage.

    mecs status                       what exists, what is missing
    mecs discover                     01  choose matches
    mecs download                     02  fetch demos
    mecs metadata                     03  rounds + alive windows
    mecs record                       04  capture video   (Windows)
    mecs actions                      05  tick-level parquet
    mecs align                        06  video <-> tick offsets
    mecs package                      07  build release tree
    mecs publish                      08  upload to the Hub
    mecs run --through align          run consecutive stages

Every stage accepts ``--config``, ``--data-root``, ``--match`` (repeatable) and
``--dry-run`` where meaningful, and every stage is idempotent: re-running skips
completed work unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from .config import Config, load_config
from .util.logging_setup import get_logger, setup_logging

log = get_logger("cli")

STAGE_ORDER = [
    "discover",
    "download",
    "metadata",
    "record",
    "actions",
    "align",
    "package",
    "publish",
]


def _common_parser() -> argparse.ArgumentParser:
    """Flags shared by the top-level parser and every subparser.

    SUPPRESS defaults matter: they are attached in both places so that
    `mecs --config x run` and `mecs run --config x` both work, and without
    SUPPRESS the subparser's default would overwrite whatever the top-level
    parser had already set.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", default=argparse.SUPPRESS,
                        help="path to a YAML config file")
    common.add_argument("--data-root", default=argparse.SUPPRESS,
                        help="override data_root from the config")
    common.add_argument(
        "-m", "--match", action="append", dest="matches", default=argparse.SUPPRESS,
        metavar="MATCH_ID", help="restrict to this match id (repeatable)",
    )
    common.add_argument("--match-file", default=argparse.SUPPRESS,
                        help="file listing match ids, one per line")
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
                        help="debug logging")
    common.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
                        help="file logging only")
    common.add_argument("--no-log-file", action="store_true", default=argparse.SUPPRESS,
                        help="do not write a log file")
    return common


def _build_parser() -> argparse.ArgumentParser:
    common = _common_parser()
    parser = argparse.ArgumentParser(
        prog="mecs",
        description="Collect synchronized multi-egocentric Counter-Strike 2 datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        parents=[common],
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")


    sub = parser.add_subparsers(dest="command", metavar="<stage>")

    p = sub.add_parser("status", parents=[common], help="report what exists on disk")
    p.add_argument("--json", action="store_true", help="machine-readable output")

    p = sub.add_parser("discover", parents=[common], help="stage 01: choose matches")
    p.add_argument("--target", type=int, help="override discover.target_matches")
    p.add_argument("--region", help="override discover.region")
    p.add_argument("--maps", help="comma-separated map allowlist, e.g. de_mirage,de_dust2")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("download", parents=[common], help="stage 02: fetch demos")
    p.add_argument("--concurrency", type=int)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("metadata", parents=[common], help="stage 03: rounds + alive windows")
    p.add_argument("--force", action="store_true", help="re-parse demos that already have metadata")

    p = sub.add_parser("record", parents=[common], help="stage 04: capture video (Windows)")
    p.add_argument("--dry-run", action="store_true", help="print the capture plan and exit")

    p = sub.add_parser("actions", parents=[common], help="stage 05: tick-level parquet")
    p.add_argument("--workers", type=int)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("align", parents=[common], help="stage 06: video/tick offsets")
    p.add_argument("--workers", type=int)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("package", parents=[common], help="stage 07: build release tree")
    p.add_argument("--include-unusable", action="store_true",
                   help="keep matches that failed stage-03 validation")
    p.add_argument("--include-demos", action="store_true",
                   help="copy .dem files into the release (large)")
    p.add_argument("--skip-video", action="store_true",
                   help="manifest + actions only; do not materialise video")
    p.add_argument("--fresh", action="store_true",
                   help="delete the existing release tree before rebuilding")

    p = sub.add_parser("publish", parents=[common], help="stage 08: upload to the Hub")
    p.add_argument("--repo-id", help="override publish.repo_id")
    p.add_argument("--include", help="comma-separated subset of release/ to upload")
    p.add_argument("--revision", help="target branch (default: main)")
    p.add_argument("--yes", action="store_true", help="confirm the upload")
    p.add_argument("--reupload", help="comma-separated repo paths to re-upload even if unchanged")
    p.add_argument("--delete", help="comma-separated repo paths to DELETE after uploading "
                                    "(folders need a trailing slash); irreversible")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("card", parents=[common], help="upload only the dataset card")
    p.add_argument("path", help="path to the README.md to publish")
    p.add_argument("--repo-id")
    p.add_argument("--yes", action="store_true")

    p = sub.add_parser("run", parents=[common], help="run consecutive stages")
    p.add_argument("--from", dest="from_stage", default="discover", choices=STAGE_ORDER)
    p.add_argument("--through", dest="through_stage", default="package", choices=STAGE_ORDER)
    p.add_argument("--force", action="store_true")

    return parser


def _load(args: argparse.Namespace) -> Config:
    cfg = load_config(getattr(args, "config", None))
    if getattr(args, "data_root", None):
        cfg.data_root = args.data_root
    return cfg


def _match_ids(args: argparse.Namespace) -> list[str] | None:
    ids: list[str] = list(getattr(args, "matches", None) or [])
    path = getattr(args, "match_file", None)
    if path:
        from pathlib import Path

        from .util.ids import extract_ids

        ids.extend(extract_ids(Path(path).read_text(encoding="utf-8")))
    return ids or None


def _cmd_status(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    import json as _json
    from pathlib import Path

    from .util.io_utils import du_bytes, human_bytes, read_jsonl

    layout = cfg.layout
    discovered = sum(1 for _ in read_jsonl(layout.matches_index))
    demos = len(layout.match_ids_with_demos())
    metadata = len(list(layout.metadata.glob("*.json"))) if layout.metadata.exists() else 0
    video_matches = len(layout.match_ids_with_video())

    def _count(root: Path, pattern: str) -> int:
        return sum(1 for _ in root.rglob(pattern)) if root.exists() else 0

    report = {
        "data_root": str(layout.root),
        "discovered_matches": discovered,
        "demos": demos,
        "metadata": metadata,
        "video_matches": video_matches,
        "video_clips": _count(layout.video, "*.mp4"),
        "action_files": _count(layout.state_action, "*.parquet"),
        "aligned_rounds": _count(layout.align, "offsets.json"),
        "release_exists": layout.release.exists(),
        "sizes": {
            "demo": human_bytes(du_bytes(layout.demo)),
            "video": human_bytes(du_bytes(layout.video)),
            "state_action": human_bytes(du_bytes(layout.state_action)),
            "release": human_bytes(du_bytes(layout.release)),
        },
    }

    if args.json:
        print(_json.dumps(report, indent=2))
    else:
        print(f"\ndata root: {report['data_root']}\n")
        rows = [
            ("01 discover", f"{discovered} matches in matches.jsonl"),
            ("02 download", f"{demos} demos ({report['sizes']['demo']})"),
            ("03 metadata", f"{metadata} match documents"),
            ("04 record", f"{video_matches} matches, {report['video_clips']} clips "
                          f"({report['sizes']['video']})"),
            ("05 actions", f"{report['action_files']} parquet files "
                           f"({report['sizes']['state_action']})"),
            ("06 align", f"{report['aligned_rounds']} rounds with offsets"),
            ("07 package", f"release/ {'present' if report['release_exists'] else 'not built'} "
                           f"({report['sizes']['release']})"),
        ]
        width = max(len(a) for a, _ in rows)
        for stage, detail in rows:
            print(f"  {stage:<{width}}  {detail}")
        print()
        if demos and metadata < demos:
            print(f"  next: mecs metadata      ({demos - metadata} demos unparsed)")
        elif report["video_clips"] and not report["action_files"] and demos:
            print("  next: mecs actions")
        elif report["action_files"] and not report["aligned_rounds"]:
            print("  next: mecs align")
        print()
    return report


def _dispatch(command: str, cfg: Config, args: argparse.Namespace) -> Any:
    matches = _match_ids(args)

    if command == "status":
        return _cmd_status(cfg, args)

    if command == "discover":
        from .stages import discover

        if args.target:
            cfg.discover.target_matches = args.target
        if args.region:
            cfg.discover.region = args.region
        if args.maps:
            cfg.discover.maps = [m.strip() for m in args.maps.split(",") if m.strip()]
        return discover.run(cfg, dry_run=args.dry_run)

    if command == "download":
        from .stages import download

        if args.concurrency:
            cfg.download.concurrency = args.concurrency
        return download.run(cfg, match_ids=matches, dry_run=args.dry_run)

    if command == "metadata":
        from .stages import metadata

        return metadata.run(cfg, match_ids=matches, force=args.force)

    if command == "record":
        from .stages import record

        return record.run(cfg, match_ids=matches, dry_run=args.dry_run)

    if command == "actions":
        from .stages import actions

        if args.workers:
            cfg.actions.workers = args.workers
        return actions.run(cfg, match_ids=matches, force=args.force)

    if command == "align":
        from .stages import align

        if args.workers:
            cfg.align.workers = args.workers
        return align.run(cfg, match_ids=matches, force=args.force)

    if command == "package":
        from .stages import package

        return package.run(
            cfg,
            match_ids=matches,
            include_unusable=args.include_unusable,
            include_demos=args.include_demos,
            skip_video=args.skip_video,
            fresh=getattr(args, "fresh", False),
        )

    if command == "publish":
        from .stages import publish

        if args.repo_id:
            cfg.publish.repo_id = args.repo_id
        include = [i.strip() for i in args.include.split(",")] if args.include else None
        reupload = (
            [x.strip() for x in args.reupload.split(",")] if getattr(args, "reupload", None) else None
        )
        deletions = (
            [x.strip() for x in args.delete.split(",")] if getattr(args, "delete", None) else None
        )
        return publish.run(
            cfg, yes=args.yes, dry_run=args.dry_run, include=include,
            revision=args.revision, reupload=reupload, delete=deletions,
        )

    if command == "card":
        from .stages import publish

        if args.repo_id:
            cfg.publish.repo_id = args.repo_id
        return publish.upload_card(cfg, args.path, yes=args.yes)

    if command == "run":
        start = STAGE_ORDER.index(args.from_stage)
        end = STAGE_ORDER.index(args.through_stage)
        if end < start:
            raise SystemExit(f"--through {args.through_stage} precedes --from {args.from_stage}")
        stages = STAGE_ORDER[start : end + 1]
        if "publish" in stages:
            # Publishing is deliberately excluded from chained runs: it is
            # public and irreversible, so it always needs its own invocation.
            stages.remove("publish")
            log.warning("`run` never publishes; invoke `mecs publish --yes` separately.")
        results: dict[str, Any] = {}
        for stage in stages:
            log.info("========== %s ==========", stage)
            stage_args = argparse.Namespace(**vars(args))
            for flag, default in (
                ("dry_run", False), ("force", getattr(args, "force", False)),
                ("workers", None), ("target", None), ("region", None), ("maps", None),
                ("concurrency", None), ("include_unusable", False),
                ("include_demos", False), ("skip_video", False), ("json", False),
                ("fresh", False),
            ):
                if not hasattr(stage_args, flag):
                    setattr(stage_args, flag, default)
            results[stage] = _dispatch(stage, cfg, stage_args)
        return results

    raise SystemExit(f"Unknown command: {command}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__

        print(f"multi-ego-cs {__version__}")
        return 0

    if not args.command:
        parser.print_help()
        return 1

    cfg = _load(args)
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    log_dir = (None if getattr(args, "no_log_file", False) or args.command == "status"
               else cfg.layout.logs)
    log_path = setup_logging(args.command, log_dir=log_dir, level=level,
                             quiet=getattr(args, "quiet", False))
    if log_path:
        log.info("Logging to %s", log_path)

    try:
        _dispatch(args.command, cfg, args)
    except KeyboardInterrupt:
        log.warning("Interrupted. Re-run the same command to resume.")
        return 130
    except Exception as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        if getattr(args, "verbose", False):
            raise
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
