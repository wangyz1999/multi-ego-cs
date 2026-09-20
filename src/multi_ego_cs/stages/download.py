"""Stage 02 - fetch demo files.

FACEIT serves demos as zstd-compressed archives from a CDN URL embedded in the
match payload, so the happy path is a plain parallel HTTP download followed by
decompression. No browser required.

Two things make this stage less trivial than it looks:

*   **Demos expire.** FACEIT keeps CS2 demos for a limited window (on the order
    of weeks). A URL that was valid at discovery time can 403/404 later, and
    once it is gone the match's tick data is unrecoverable - the recordings
    survive but stage 05 can never run for it. Collect demos *early*, and treat
    ``download`` as the stage that gates everything else. Run it on the same day
    as discovery if you can.
*   **The CDN host may be unreachable** from a locked-down network even when
    the demo still exists. The stage distinguishes DNS failure (host-level, all
    matches fail) from per-URL 404 (expiry) and reports them separately, because
    the remedies are completely different.
"""

from __future__ import annotations

import concurrent.futures as cf
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from ..config import Config
from ..util.io_utils import du_bytes, human_bytes
from ..util.logging_setup import get_logger
from .discover import load_matches

log = get_logger("download")

_CHUNK = 1 << 20  # 1 MiB


class DownloadOutcome:
    OK = "ok"
    SKIPPED = "skipped"
    EXPIRED = "expired"          # server says the object is gone
    UNREACHABLE = "unreachable"  # DNS/connection failure - network, not expiry
    FAILED = "failed"


def _decompress_zst(archive: Path, target: Path) -> None:
    """Decompress `archive` to `target`, atomically.

    Prefers the `zstandard` Python package; falls back to the `zstd` CLI, which
    is what HPC module trees usually provide.
    """
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        import zstandard  # type: ignore

        dctx = zstandard.ZstdDecompressor()
        with archive.open("rb") as src, tmp.open("wb") as dst:
            dctx.copy_stream(src, dst, read_size=_CHUNK, write_size=_CHUNK)
    except ImportError:
        zstd_bin = shutil.which("zstd")
        if not zstd_bin:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                "Cannot decompress .dem.zst: install the `zstandard` package "
                "or make the `zstd` binary available (on HPC: `module load zstd`)."
            ) from None
        proc = subprocess.run(
            [zstd_bin, "-d", "-f", "-o", str(tmp), str(archive)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"zstd failed: {proc.stderr.strip()[:300]}")
    tmp.replace(target)


def _host_resolves(url: str) -> bool:
    host = urlparse(url).hostname
    if not host:
        return False
    try:
        socket.getaddrinfo(host, 443)
        return True
    except socket.gaierror:
        return False


def _download_one(
    record: dict[str, Any],
    cfg: Config,
    session_factory,
) -> tuple[str, str, str]:
    """Returns (match_id, outcome, detail)."""
    match_id = record["match_id"]
    layout = cfg.layout
    dem = layout.demo_file(match_id)
    archive = layout.demo_archive(match_id)

    if dem.exists() and dem.stat().st_size > 0:
        return match_id, DownloadOutcome.SKIPPED, "already present"

    urls = record.get("demo_urls") or []
    if not urls:
        return match_id, DownloadOutcome.FAILED, "no demo_urls in index"

    session = session_factory()
    last_detail = ""

    for url in urls:
        if not _host_resolves(url):
            return match_id, DownloadOutcome.UNREACHABLE, f"DNS failure for {urlparse(url).hostname}"

        for attempt in range(1, cfg.download.retries + 1):
            part = archive.with_suffix(archive.suffix + ".part")
            try:
                archive.parent.mkdir(parents=True, exist_ok=True)
                with session.get(url, stream=True, timeout=120) as resp:
                    if resp.status_code in (403, 404, 410):
                        last_detail = f"HTTP {resp.status_code} (demo retention window passed)"
                        break  # expiry: retrying will not help
                    resp.raise_for_status()
                    with part.open("wb") as fh:
                        for chunk in resp.iter_content(_CHUNK):
                            if chunk:
                                fh.write(chunk)
                part.replace(archive)

                if cfg.download.decompress:
                    _decompress_zst(archive, dem)
                    if not cfg.download.keep_archive:
                        archive.unlink(missing_ok=True)
                    size = dem.stat().st_size
                else:
                    size = archive.stat().st_size
                return match_id, DownloadOutcome.OK, human_bytes(size)

            except requests.RequestException as exc:
                part.unlink(missing_ok=True)
                last_detail = f"{type(exc).__name__}: {exc}"
                if attempt < cfg.download.retries:
                    time.sleep(cfg.download.retry_backoff_seconds * attempt)
            except Exception as exc:  # decompression, disk, …
                part.unlink(missing_ok=True)
                last_detail = f"{type(exc).__name__}: {exc}"
                break

        if last_detail.startswith("HTTP "):
            return match_id, DownloadOutcome.EXPIRED, last_detail

    return match_id, DownloadOutcome.FAILED, last_detail or "unknown"


def run(
    cfg: Config,
    match_ids: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    layout = cfg.layout.ensure()
    records = load_matches(cfg)
    if not records:
        log.error("No matches in %s - run `mecs discover` first.", layout.matches_index)
        return {"ok": 0, "total": 0}

    if match_ids:
        wanted = set(match_ids)
        records = [r for r in records if r["match_id"] in wanted]

    pending = [r for r in records if not layout.demo_file(r["match_id"]).exists()]
    log.info(
        "%d matches in index, %d already downloaded, %d pending",
        len(records), len(records) - len(pending), len(pending),
    )
    if dry_run:
        for r in pending[:20]:
            log.info("  [dry-run] would fetch %s (%s)", r["match_id"], r.get("map"))
        return {"pending": len(pending), "dry_run": True}
    if not pending:
        return {"ok": 0, "skipped": len(records), "total": len(records)}

    budget = cfg.download.max_total_gb * (1 << 30)
    used = du_bytes(layout.demo)

    def session_factory() -> requests.Session:
        s = requests.Session()
        s.headers.update({"User-Agent": "multi-ego-cs/0.1 (+https://github.com/wangyz1999/multi-ego-cs)"})
        return s

    counts: dict[str, int] = {}
    details: dict[str, list[str]] = {}

    with cf.ThreadPoolExecutor(max_workers=max(1, cfg.download.concurrency)) as pool:
        futures = {
            pool.submit(_download_one, rec, cfg, session_factory): rec["match_id"]
            for rec in pending
        }
        done = 0
        for fut in cf.as_completed(futures):
            match_id, outcome, detail = fut.result()
            counts[outcome] = counts.get(outcome, 0) + 1
            details.setdefault(outcome, []).append(f"{match_id}: {detail}")
            done += 1
            level = log.info if outcome in (DownloadOutcome.OK, DownloadOutcome.SKIPPED) else log.warning
            level("[%d/%d] %-12s %s  %s", done, len(pending), outcome, match_id[:24], detail)

            used = du_bytes(layout.demo)
            if used > budget:
                log.error(
                    "Demo directory reached %s, over the %.1f GB budget - stopping.",
                    human_bytes(used), cfg.download.max_total_gb,
                )
                for f in futures:
                    f.cancel()
                break

    log.info("Download summary: %s", counts)
    log.info("Demo directory now %s", human_bytes(du_bytes(layout.demo)))

    if counts.get(DownloadOutcome.UNREACHABLE):
        log.error(
            "%d matches failed with DNS errors. The CDN host does not resolve from "
            "this network - this is a connectivity problem, NOT demo expiry. Retry "
            "from a network with unrestricted outbound DNS before concluding the "
            "demos are gone.",
            counts[DownloadOutcome.UNREACHABLE],
        )
    if counts.get(DownloadOutcome.EXPIRED):
        log.error(
            "%d demos are past FACEIT's retention window and cannot be recovered. "
            "Those matches can still contribute video, but stage 05 (actions) will "
            "have no tick data for them.",
            counts[DownloadOutcome.EXPIRED],
        )

    return {"counts": counts, "details": details, "total": len(records)}
